"""VP viewpoint-robustness eval for BEVDepth (3D-detection NDS) -- the BEVDepth
analogue of eval_vp_robustness_det.py.

A sedan-trained BEVDepth model is evaluated on carla_VR viewpoint perturbations:
    Normal (oracle / RRS denom) | ER (extrinsic) | VR (image, primary) | CR (both)
  axes yaw/pitch/roll x signed mags {4,8,12,16,20} x protocols (6 per-cam + all-cam)
Full grid = 1 + 3*3*10*7 = 631 cells.

Mirrors the BEVFormer driver's optimisations so 631 cells stay tractable:
  * model + NuScenes GT DB loaded ONCE, cells iterated in-process (no per-cell
    weight/DB reload); per cell only ds.infos is swapped (make_vp_infos_bevdepth).
  * deterministic fp32 (cudnn.benchmark/TF32 off + seed) -> reproducible NDS.
  * large eval batch (BatchNorm is batch-invariant in eval -> NDS unchanged).
  * optional tmpfs RAM-staging (VP_STAGE_ROOT) so decode is GPU-bound not Lustre.
  * --shard i/n cell-sharding across GPUs + --merge.

RRS(cell)=NDS_cell/NDS_Normal ; mRRS_c=mean per-cam ; RRSALL_c=all-cam ;
mVRS_c=0.5*(mRRS_c+RRSALL_c). VR is primary.

Launch via run_vp_bevdepth.sh.
"""
import argparse
import copy
import csv
import json
import os
import os.path as osp
import sys
import tempfile
import time
from functools import partial

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
BEVF_ROOT = osp.dirname(HERE)
BEVDEPTH = osp.join(BEVF_ROOT, 'BEVDepth')
sys.path.insert(0, HERE)
sys.path.insert(0, BEVDEPTH)
import build_condition_pkls_bevdepth as B  # noqa: E402

CAM_NAMES = B.CAM_NAMES
P = 'pts_bbox_NuScenes/'   # detail-dict key prefix (det_evaluators result_name='pts_bbox')
COMP_COLS = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'nds_10class', 'map_10class']


def nds_components(metrics):
    g = (lambda k: metrics.get(P + k)) if metrics else (lambda k: None)
    return {'mATE': g('mATE_6class'), 'mASE': g('mASE_6class'),
            'mAOE': g('mAOE_6class'), 'mAVE': g('mAVE_6class'),
            'mAAE': g('mAAE_6class'), 'nds_10class': g('NDS_10class'),
            'map_10class': g('mAP_10class')}


def set_deterministic(seed=0):
    """Reproducible NDS: kill cudnn algo-selection + TF32 variation, fix seed.
    (Mirrors the BEVFormer driver. BEVDepth runs fp32 already -- carla exps set
    precision=32. The voxel_pooling forward uses atomicAdd, so a sub-1e-3 residual
    may remain; cudnn/TF32 are the dominant sources and are removed here.)"""
    import pytorch_lightning as pl
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    pl.seed_everything(seed, workers=True)


def patch_nuscenes_cache():
    """Cache NuScenes by (version, dataroot) so the GT DB loads only ONCE across
    all 631 cells. det_evaluators._evaluate_single does `from nuscenes import
    NuScenes; NuScenes(version=self.version, ...)` every call; for VP the version
    (sedan) is identical, so one cached instance is correct."""
    import nuscenes
    real = nuscenes.NuScenes
    cache = {}

    def cached(version=None, dataroot=None, verbose=False, **kw):
        key = (version, dataroot)
        if key not in cache:
            cache[key] = real(version=version, dataroot=dataroot,
                              verbose=verbose, **kw)
        return cache[key]

    nuscenes.NuScenes = cached


def build_subset_infos(frames_per_scene):
    """Frozen N-frames-per-scene subset of the BEVDepth sedan val pkl (even
    stride). Test-mode temporal is off (num_sweeps=1, key_idxes=[]) so per-frame
    subsetting is unbiased. Returns a bare list of sample dicts."""
    infos = B.load_pkl(B.SEDAN_VAL)            # BEVDepth val = bare list
    by_scene = {}
    for s in infos:
        by_scene.setdefault(B.info_key(s)[0], []).append(s)
    sub = []
    for scene in sorted(by_scene):
        ss = sorted(by_scene[scene], key=lambda i: int(B.info_key(i)[1]))
        if frames_per_scene >= len(ss):
            pick = ss
        else:
            idx = np.linspace(0, len(ss) - 1, frames_per_scene)
            pick = [ss[j] for j in sorted(set(idx.round().astype(int)))]
        sub.extend(pick)
    return sub


# --------------------------------------------------------------------------- #
# RAM-staging (tmpfs) -- decode from RAM, not Lustre, so the loop is GPU-bound.
# Every image any cell will read (baseline subset + the VR variants in the grid)
# is copied ONCE to stage_root preserving its real absolute path; per-cell we
# rewrite each cam filename to the staged path (and run with data_root='').
# --------------------------------------------------------------------------- #
def _real_abs(filename, data_root_real):
    """Real absolute path of a cam filename: VR variants are already absolute;
    baseline filenames are relative to the (symlink-resolved) carla_geobev root."""
    return filename if osp.isabs(filename) else osp.join(data_root_real, filename)


def staged_path(real_abs, stage_root):
    return osp.join(stage_root, real_abs.lstrip('/'))


def collect_real_paths(base, conditions, axes, mags, data_root_real):
    """Every real image path any cell reads: baseline (subset x 6 cams) + VR
    variants (if VR/CR present) for all (axis, mag) x all 6 cams x subset."""
    paths = set()
    for s in base:
        for cam in CAM_NAMES:
            paths.add(_real_abs(s['cam_infos'][cam]['filename'], data_root_real))
    if any(c in ('VR', 'CR') for c in conditions):
        variants = {B.variant_key(ax, mg) for ax in axes for mg in mags}
        for s in base:
            sc, fr = B.info_key(s)
            for cam in CAM_NAMES:
                for v in variants:
                    paths.add(B.vr_image_path(sc, fr, cam, v))
    return paths


def stage_images(real_paths, stage_root, workers=16):
    """Copy each real image to tmpfs (skip-if-exists), in parallel. -> (n_copied, GB)."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    todo = [(rp, staged_path(rp, stage_root)) for rp in real_paths
            if not osp.exists(staged_path(rp, stage_root))]

    def cp(args):
        rp, dst = args
        os.makedirs(osp.dirname(dst), exist_ok=True)
        # atomic: copy to a unique tmp then rename, so two concurrent shards
        # staging the same image can't write a half-copied file (content is
        # identical, so last-rename-wins is fine).
        tmp = f'{dst}.tmp.{os.getpid()}.{abs(hash(rp)) & 0xffffff}'
        shutil.copy2(rp, tmp)
        os.replace(tmp, dst)
        return osp.getsize(dst)

    nbytes = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for sz in ex.map(cp, todo):
                nbytes += sz
    return len(todo), nbytes / 1e9


def stage_rewrite_infos(infos, data_root_real, stage_root):
    """In-place: point every cam filename at its staged (tmpfs) absolute path."""
    for s in infos:
        for cam in CAM_NAMES:
            ci = s['cam_infos'][cam]
            ci['filename'] = staged_path(
                _real_abs(ci['filename'], data_root_real), stage_root)


def build_model_and_dataset(ckpt, stage_root=None):
    """Load the sedan BEVDepth LightningModule once (saved hyper_parameters) and
    build the val CarlaDetDataset once. We mutate ds.infos per cell. If stage_root
    is set, point data_root at the tmpfs staged image tree (RAM-staging)."""
    from bevdepth.exps.nuscenes.carla.carla_sedan import BEVDepthLightningModel
    from bevdepth.datasets.carla_det_dataset import CarlaDetDataset
    # NOT load_from_checkpoint: CarlaBEVDepthBase.__init__ injects class_names +
    # head_conf itself, so passing the saved hyper_parameters back (which also
    # carry them) raises "multiple values for keyword argument". Re-init with the
    # saved hp minus those two, then load the state_dict.
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    hp = {k: v for k, v in ck['hyper_parameters'].items()
          if k not in ('class_names', 'head_conf')}
    model = BEVDepthLightningModel(**hp)
    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    if missing or unexpected:
        print(f'[VP] load_state_dict: {len(missing)} missing / '
              f'{len(unexpected)} unexpected keys '
              f'(missing[:2]={missing[:2]} unexpected[:2]={unexpected[:2]})',
              flush=True)
    # depth_net's DeformConv2dPack ships im2col_step=128; mmcv asserts
    # input.size(0) % min(im2col_step, input.size(0)) == 0. The deform input is
    # batch*num_cams (always a multiple of 6), so e.g. batch 32 -> 192 % 128 != 0
    # raises. Set im2col_step=6 so EVERY batch (incl. a partial last batch) is
    # divisible -- it's a memory-tiling param only, so NDS is unchanged and any
    # eval batch is allowed.
    for mod in model.modules():
        if hasattr(mod, 'im2col_step'):
            mod.im2col_step = 6
    model = model.cuda()
    model.eval()
    # With staging, every cam filename is rewritten to a staged ABSOLUTE path, so
    # data_root='' (os.path.join('', abs)==abs). Without staging, the exp's
    # relative data_root='data/carla' is used as-is.
    data_root = '' if stage_root else model.data_root
    ds = CarlaDetDataset(
        ida_aug_conf=model.ida_aug_conf, bda_aug_conf=model.bda_aug_conf,
        classes=model.class_names, data_root=data_root,
        info_paths=model.val_info_paths, is_train=False, img_conf=model.img_conf,
        num_sweeps=model.num_sweeps, sweep_idxes=model.sweep_idxes,
        key_idxes=model.key_idxes, return_depth=False, use_fusion=False,
        gt_visibility_min=model.gt_visibility_min)
    return model, ds


def infer(model, ds, infos, batch, workers):
    """In-process inference over `infos` (mutated into ds.infos). Returns
    (all_pred_results, all_img_metas) as det_evaluators.evaluate expects."""
    from torch.utils.data import DataLoader
    from bevdepth.datasets.nusc_det_dataset import collate_fn
    ds.infos = infos
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=partial(collate_fn, is_return_depth=False),
                        drop_last=False, pin_memory=False)
    all_pred, all_meta = [], []
    with torch.no_grad():
        for bd in loader:
            sweep_imgs, mats, _, img_metas, _, _ = bd
            sweep_imgs = sweep_imgs.cuda()
            mats = {k: v.cuda() for k, v in mats.items()}
            preds = model.model(sweep_imgs, mats)
            results = model.model.get_bboxes(preds, img_metas)
            for i in range(len(results)):
                all_pred.append([results[i][0].detach().cpu().numpy(),
                                 results[i][1].detach().cpu().numpy(),
                                 results[i][2].detach().cpu().numpy()])
                all_meta.append(img_metas[i])
    return all_pred, all_meta


def eval_nds(model, all_pred, all_meta, tmpdir):
    """6-class NDS via the (patched, visibility>=2) BEVDepth evaluator."""
    detail = model.evaluator.evaluate(all_pred, all_meta,
                                      jsonfile_prefix=osp.join(tmpdir, 'vp'))
    nds = next((v for k, v in detail.items() if k.endswith('/NDS')), None)
    m6 = next((v for k, v in detail.items() if k.endswith('/mAP')), None)
    if nds is None:
        raise RuntimeError(f'no 6-class NDS in detail; keys={list(detail)[:8]}')
    return float(nds), float(m6), dict(detail)


def run_cell(model, ds, base, cond, axis, mag, proto, batch, workers, tmpdir,
             stage=None):
    """One VP cell: swap infos, infer, eval -> (nds, m6, detail).
    stage=(data_root_real, stage_root) rewrites filenames to tmpfs paths."""
    if cond == 'Normal':
        infos = copy.deepcopy(base)
    else:
        infos = B.make_vp_infos_bevdepth(base, cond, axis, mag, proto)
    if stage is not None:
        stage_rewrite_infos(infos, stage[0], stage[1])
    pred, meta = infer(model, ds, infos, batch, workers)
    return eval_nds(model, pred, meta, tmpdir)


def all_cells(conditions, axes, mags, protocol):
    """Ordered (cond, axis, mag, proto) cells. protocol: percam (6) / allcam (1)
    / both (7). Normal is handled separately (the RRS denominator)."""
    protos = []
    if protocol in ('both', 'percam'):
        protos += list(CAM_NAMES)
    if protocol in ('both', 'allcam'):
        protos += ['all']
    cells = []
    for cond in conditions:
        for axis in axes:
            for mag in mags:
                for proto in protos:
                    cells.append((cond, axis, mag, proto))
    return cells


def shard_slice(cells, shard):
    i, n = (int(x) for x in shard.split('/'))
    return [c for k, c in enumerate(cells) if k % n == i]


# --------------------------------------------------------------------------- #
# outputs / aggregation
# --------------------------------------------------------------------------- #
def write_outputs(outdir, args, nds_norm, m6_norm, rows):
    os.makedirs(outdir, exist_ok=True)
    # per-cell csv
    csv_path = osp.join(outdir, 'eval_vp_per_config.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'axis', 'mag', 'protocol', 'nds', 'map6', 'rrs'])
        for r in rows:
            w.writerow([r['cond'], r['axis'], r['mag'], r['proto'],
                        f"{r['nds']:.4f}", f"{r['map6']:.4f}", f"{r['rrs']:.4f}"])
    # aggregate per condition: mRRS (per-cam), RRSALL (all-cam), mVRS
    agg = {}
    for cond in sorted({r['cond'] for r in rows}):
        percam = [r['rrs'] for r in rows
                  if r['cond'] == cond and r['proto'] != 'all']
        allcam = [r['rrs'] for r in rows
                  if r['cond'] == cond and r['proto'] == 'all']
        mrrs = float(np.mean(percam)) if percam else float('nan')
        rrsall = float(np.mean(allcam)) if allcam else float('nan')
        agg[cond] = {'mRRS_percam': mrrs, 'RRSALL_allcam': rrsall,
                     'mVRS': 0.5 * (mrrs + rrsall)}
    js = {'tag': args.tag, 'frames_per_scene': args.frames_per_scene,
          'nds_normal': nds_norm, 'map6_normal': m6_norm,
          'aggregate': agg, 'rows': rows}
    with open(osp.join(outdir, 'eval_vp.json'), 'w') as f:
        json.dump(js, f, indent=2)
    lines = [f'VP viewpoint-robustness (BEVDepth NDS)   tag={args.tag}',
             f'  frames-per-scene={args.frames_per_scene}  '
             f'NDS_Normal={nds_norm:.4f}  mAP6_Normal={m6_norm:.4f}',
             '  RRS = NDS_cell / NDS_Normal   [VR primary]', '',
             '  condition   mRRS(per-cam)   RRSALL(all-cam)        mVRS']
    for cond in ['ER', 'VR', 'CR']:
        if cond not in agg:
            continue
        a = agg[cond]
        star = '  <- primary' if cond == 'VR' else ''
        lines.append(f'  {cond:<10}{a["mRRS_percam"]:>13.4f}'
                     f'{a["RRSALL_allcam"]:>16.4f}{a["mVRS"]:>12.4f}{star}')
    txt = '\n'.join(lines)
    with open(osp.join(outdir, 'eval_vp_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\nwrote {csv_path}\n      {osp.join(outdir, "eval_vp.json")}\n'
          f'      {osp.join(outdir, "eval_vp_summary.txt")}')


def merge_shards(outdir, args):
    """Combine per-shard json (rows + the shared Normal) into the final outputs."""
    rows, nds_norm, m6_norm = [], None, None
    for fn in sorted(os.listdir(outdir)):
        if not fn.startswith('shard_') or not fn.endswith('.json'):
            continue
        d = json.load(open(osp.join(outdir, fn)))
        rows.extend(d['rows'])
        nds_norm = d['nds_normal']
        m6_norm = d['map6_normal']
    if nds_norm is None:
        raise RuntimeError(f'no shard_*.json in {outdir}')
    write_outputs(outdir, args, nds_norm, m6_norm, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='pth/BEVDeth/bevdepth_sedan.ckpt')
    ap.add_argument('--frames-per-scene', type=int, default=4)
    ap.add_argument('--conditions', nargs='+', default=['ER', 'VR', 'CR'],
                    choices=['ER', 'VR', 'CR'])
    ap.add_argument('--axes', nargs='+', default=B.VP_AXES)
    ap.add_argument('--mags', nargs='+', type=int, default=B.VP_MAGNITUDES)
    ap.add_argument('--protocol', default='both',
                    choices=['both', 'allcam', 'percam'])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shard', default='0/1', help='i/n cell-shard')
    ap.add_argument('--tag', default='bevdepth_sedan')
    ap.add_argument('--outdir', default=osp.join(HERE, 'out'))
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()

    outdir = osp.join(args.outdir, f'vp_{args.tag}')
    if args.merge:
        merge_shards(outdir, args)
        return

    ckpt = args.ckpt if osp.isabs(args.ckpt) else osp.join(BEVF_ROOT, args.ckpt)
    stage_root = os.environ.get('VP_STAGE_ROOT')   # tmpfs image tree (VP-c)

    set_deterministic(0)
    patch_nuscenes_cache()
    model, ds = build_model_and_dataset(ckpt, stage_root=stage_root)
    base = build_subset_infos(args.frames_per_scene)

    # RAM-staging: copy every image this run reads to tmpfs ONCE, then per-cell
    # rewrite filenames to the staged paths (decode from RAM -> GPU-bound).
    stage = None
    if stage_root:
        data_root_real = osp.realpath(osp.join(BEVDEPTH, 'data', 'carla'))
        rp = collect_real_paths(base, args.conditions, args.axes, args.mags,
                                data_root_real)
        ts = time.perf_counter()
        n_copied, gb = stage_images(rp, stage_root, workers=max(16, args.workers))
        stage = (data_root_real, stage_root)
        print(f'[VP] staged {n_copied}/{len(rp)} images ({gb:.1f} GB) to '
              f'{stage_root} in {time.perf_counter()-ts:.0f}s', flush=True)

    print(f'[VP] model+GT loaded once | subset={len(base)} frames '
          f'(frames-per-scene={args.frames_per_scene}) | batch={args.batch} '
          f'workers={args.workers} | shard={args.shard} | '
          f'stage={"on" if stage_root else "off"}', flush=True)

    tmpdir = tempfile.mkdtemp(prefix='vp_bevdepth_')
    # The evaluator's format_results writes results_nusc.json to self.output_dir
    # (=./outputs/carla_sedan) when set, IGNORING our jsonfile_prefix. Two shards
    # would then race on the same file (and it clobbers the baseline preds). Point
    # it at this process's private tmpdir so every cell/shard is isolated.
    model.evaluator.output_dir = tmpdir
    t0 = time.perf_counter()

    # Normal (oracle) -- every shard runs it (RRS denominator).
    nds_norm, m6_norm, _ = run_cell(model, ds, base, 'Normal', None, None,
                                    None, args.batch, args.workers, tmpdir,
                                    stage=stage)
    print(f'[VP] Normal NDS={nds_norm:.4f} mAP6={m6_norm:.4f} '
          f'({time.perf_counter()-t0:.0f}s)', flush=True)

    cells = shard_slice(all_cells(args.conditions, args.axes, args.mags,
                                  args.protocol), args.shard)
    rows = []
    for k, (cond, axis, mag, proto) in enumerate(cells):
        tc = time.perf_counter()
        nds, m6, _ = run_cell(model, ds, base, cond, axis, mag, proto,
                              args.batch, args.workers, tmpdir, stage=stage)
        rrs = nds / nds_norm if nds_norm else float('nan')
        rows.append({'cond': cond, 'axis': axis, 'mag': mag, 'proto': proto,
                     'nds': nds, 'map6': m6, 'rrs': rrs})
        print(f'[VP {k+1}/{len(cells)}] {cond} {axis}{mag:+d} {proto:14s} '
              f'NDS={nds:.4f} RRS={rrs:.4f} ({time.perf_counter()-tc:.0f}s)',
              flush=True)

    i, n = (int(x) for x in args.shard.split('/'))
    os.makedirs(outdir, exist_ok=True)
    if n == 1:
        write_outputs(outdir, args, nds_norm, m6_norm, rows)
    else:                                            # write this shard; merge later
        sp = osp.join(outdir, f'shard_{i}of{n}.json')
        json.dump({'nds_normal': nds_norm, 'map6_normal': m6_norm, 'rows': rows},
                  open(sp, 'w'), indent=2)
        print(f'[VP] shard {args.shard} wrote {sp} '
              f'({len(rows)} cells, {time.perf_counter()-t0:.0f}s total)',
              flush=True)


if __name__ == '__main__':
    main()
