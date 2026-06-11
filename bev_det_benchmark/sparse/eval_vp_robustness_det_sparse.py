"""VP viewpoint-robustness eval for sparse (CAPE/PETRv2/DETR3D) (3D-detection NDS) -- the sparse (CAPE/PETRv2/DETR3D)
analogue of eval_vp_robustness_det.py (BEVFormer) / _bevdepth.py.

A sedan-trained sparse (CAPE/PETRv2/DETR3D) model is evaluated on carla_VR viewpoint perturbations:
    Normal (oracle / RRS denom) | ER (extrinsic) | VR (image, primary) | CR (both)
  axes yaw/pitch/roll x signed mags {4,8,12,16,20} x protocols (6 per-cam + all-cam)
Full grid = 1 + 3*3*10*7 = 631 cells.

Mirrors the BEVFormer driver's optimisations so 631 cells stay tractable:
  * model + NuScenes GT DB loaded ONCE, cells iterated in-process (no per-cell
    weight/DB reload); per cell only the val infos are swapped (make_vp_infos).
  * inference + eval go through sparse (CAPE/PETRv2/DETR3D)'s OWN dataloader (PrepareImageInputs builds
    img_inputs) + CarlaNuScenesDataset.evaluate -- i.e. the exact tools/test.py
    path, so it is correct-by-construction (no hand-assembled img_inputs).
  * deterministic fp32 (cudnn.benchmark/TF32 off + seed) -> reproducible NDS.
  * optional tmpfs RAM-staging (VP_STAGE_ROOT) so decode is GPU-bound not Lustre.
  * --shard i/n cell-sharding across GPUs + --merge.

Eval GT = visibility>=2 is enforced inside CarlaNuScenesDataset._evaluate_single
(num_pts override); here we additionally restrict GT to the frozen frames-per-scene
subset tokens (patch_eval_subset) so the devkit's pred==gt assertion holds.

RRS(cell)=NDS_cell/NDS_Normal ; mRRS_c=mean per-cam ; RRSALL_c=all-cam ;
mVRS_c=0.5*(mRRS_c+RRSALL_c). VR is primary. Launch via run_vp_bevdet.sh.
"""
import argparse
import copy
import csv
import json
import os
import os.path as osp
import pickle as _pk
import sys
import tempfile
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

HERE = osp.dirname(osp.abspath(__file__))
BEVF_ROOT = osp.dirname(osp.dirname(HERE))
BEVDET = BEVF_ROOT                                  # repo root for relative image-path resolution (data/nuscenes -> carla_geobev)
DATA_ROOT = osp.join(BEVF_ROOT, 'data', 'nuscenes')  # -> carla_geobev (v1.0-carla DBs + sweeps); CAPE/PETR/DETR3D symlink the same
sys.path.insert(0, HERE)
import build_condition_pkls as B  # standard nuScenes pkls (CAPE/PETRv2/DETR3D), same as bevformer  # noqa: E402

CAM_NAMES = B.CAM_NAMES
P = 'pts_bbox_NuScenes/'   # detail-dict key prefix
COMP_COLS = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'nds_10class', 'map_10class']


def nds_components(metrics):
    g = (lambda k: metrics.get(P + k)) if metrics else (lambda k: None)
    return {'mATE': g('mATE_6class'), 'mASE': g('mASE_6class'),
            'mAOE': g('mAOE_6class'), 'mAVE': g('mAVE_6class'),
            'mAAE': g('mAAE_6class'), 'nds_10class': g('NDS_10class'),
            'map_10class': g('mAP_10class')}


def set_deterministic(seed=0):
    """Reproducible NDS: kill cudnn algo-selection + TF32 variation, fix seed.
    bev_pool_v2 uses a deterministic interval-reduction (not atomicAdd), so this
    should make the eval bit-stable; verify by running a cell twice."""
    from mmdet.apis import set_random_seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    set_random_seed(seed, deterministic=True)


def single_gpu_infer(model, data_loader):
    """In-memory single-GPU inference (the tools/test.py forward path). Returns the
    ordered list[dict] CarlaNuScenesDataset.evaluate expects."""
    model.eval()
    results = []
    prog = mmcv.ProgressBar(len(data_loader.dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        if isinstance(result, dict):
            result = result['bbox_results']
        results.extend(result)
        for _ in range(len(result)):
            prog.update()
    return results


def patch_nuscenes_cache():
    """Cache NuScenes by (version, dataroot) so the GT DB loads only ONCE across
    all 631 cells (_evaluate_single calls NuScenes(...) every call; the VP version
    + subset are identical, so one cached instance is correct)."""
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


def patch_eval_subset(allowed_tokens):
    """Restrict eval GT to exactly our subset tokens.

    CarlaNuScenesDataset._evaluate_single restricts GT to the *scenes* in
    data_infos, then load_gt pulls EVERY frame of those scenes. With a per-frame
    subset the GT sample set is a superset of the predictions and the devkit's
    pred==gt assertion fires. We wrap the stock loaders.load_gt (the binding the
    carla scene-patch wraps in turn) to drop non-subset tokens. The subset is
    identical across all VP cells, so RRS stays a fair ratio."""
    from nuscenes.eval.common import loaders as nl
    real = nl.load_gt

    def load_gt_subset(nusc, eval_split, box_cls, verbose=False):
        gt = real(nusc, eval_split, box_cls, verbose=verbose)
        for st in list(gt.boxes.keys()):
            if st not in allowed_tokens:
                del gt.boxes[st]
        return gt

    nl.load_gt = load_gt_subset


def build_subset_infos(frames_per_scene):
    """Frozen N-frames-per-scene subset of the BEVDet sedan val pkl (even stride).
    Single-frame eval (no temporal) -> per-frame subsetting is unbiased.
    Returns (metadata, infos)."""
    d = B.load_pkl(B.SEDAN_VAL)                 # {'infos': [...], 'metadata': {...}}
    by_scene = {}
    for info in d['infos']:
        by_scene.setdefault(B.info_key(info)[0], []).append(info)
    sub = []
    for scene in sorted(by_scene):
        infos = sorted(by_scene[scene], key=lambda i: int(B.info_key(i)[1]))
        if frames_per_scene >= len(infos):
            pick = infos
        else:
            idx = np.linspace(0, len(infos) - 1, frames_per_scene)
            pick = [infos[j] for j in sorted(set(idx.round().astype(int)))]
        sub.extend(pick)
    return d['metadata'], sub


# --------------------------------------------------------------------------- #
# RAM-staging (tmpfs) -- decode from RAM, not Lustre, so the loop is GPU-bound.
# --------------------------------------------------------------------------- #
def _real_abs(data_path, repo_root):
    """Real (symlink-resolved) absolute path of a cam data_path. VR variants are
    already absolute; baseline paths are relative to the BEVDet repo root (where
    data/nuscenes -> carla_geobev)."""
    p = data_path if osp.isabs(data_path) else osp.join(repo_root, data_path)
    return osp.realpath(p)


def staged_path(real_abs, stage_root):
    return osp.join(stage_root, real_abs.lstrip('/'))


def collect_real_paths(base, conditions, axes, mags):
    """Every real image any cell reads: baseline (subset x 6 cams) + VR variants
    (if VR/CR present) for all (axis, mag) x 6 cams x subset."""
    paths = set()
    for info in base:
        for cam in CAM_NAMES:
            paths.add(_real_abs(info['cams'][cam]['data_path'], BEVDET))
    if any(c in ('VR', 'CR') for c in conditions):
        variants = {B.variant_key(ax, mg) for ax in axes for mg in mags}
        for info in base:
            sc, fr = B.info_key(info)
            for cam in CAM_NAMES:
                for v in variants:
                    paths.add(_real_abs(B.vr_image_path(sc, fr, cam, v), BEVDET))
    return paths


def stage_images(real_paths, stage_root, workers=16):
    """Copy each real image to tmpfs (skip-if-exists), in parallel. -> (n, GB)."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    todo = [(rp, staged_path(rp, stage_root)) for rp in real_paths
            if not osp.exists(staged_path(rp, stage_root))]

    def cp(args):
        rp, dst = args
        os.makedirs(osp.dirname(dst), exist_ok=True)
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


def stage_rewrite_infos(infos, stage_root):
    """In-place: point every cam data_path at its staged (tmpfs) absolute path."""
    for info in infos:
        for cam in CAM_NAMES:
            ci = info['cams'][cam]
            ci['data_path'] = staged_path(_real_abs(ci['data_path'], BEVDET),
                                          stage_root)


def build_model_once(cfg, ckpt):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location='cpu')
    # depth_net's DeformConv2dPack ships a large im2col_step; mmcv asserts
    # input.size(0) % min(im2col_step, size0) == 0. The deform input is
    # batch*num_cams (multiple of 6), so im2col_step=6 keeps EVERY batch divisible
    # -- a memory-tiling param only, so NDS is unchanged and any batch is allowed.
    for m in model.modules():
        if hasattr(m, 'im2col_step'):
            m.im2col_step = 6
    model = MMDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    model.eval()
    return model


def run_condition(model, cfg, metadata, infos, ann_path, tmpdir, workers, batch):
    """Write the swapped subset, build dataset/loader, infer -> (NDS, mAP6, detail)."""
    B.dump_pkl({'metadata': metadata, 'infos': infos}, ann_path)
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = ann_path
    test_cfg['data_root'] = DATA_ROOT
    test_cfg['test_mode'] = True
    dataset = build_dataset(test_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=batch,
                              workers_per_gpu=workers, dist=False, shuffle=False)
    outputs = single_gpu_infer(model, loader)
    res = dataset.evaluate(outputs, metric='bbox',
                           jsonfile_prefix=osp.join(tmpdir, 'eval'))
    nds = res.get(P + 'NDS')
    m6 = res.get(P + 'mAP')
    if nds is None:
        raise RuntimeError(f'no NDS in evaluate() result; keys={list(res)[:8]}')
    return float(nds), float(m6), dict(res)


def run_cell(model, cfg, metadata, base, cond, axis, mag, proto, ann_path, tmpdir,
             workers, batch, stage_root=None):
    """One VP cell: swap infos, (optionally) stage-rewrite, infer+eval."""
    if cond == 'Normal':
        infos = copy.deepcopy(base)
    else:
        infos = B.make_vp_infos(base, cond, axis, mag, proto)
    if stage_root:
        stage_rewrite_infos(infos, stage_root)
    return run_condition(model, cfg, metadata, infos, ann_path, tmpdir, workers,
                         batch)


def all_cells(conditions, axes, mags, protocol):
    protos = []
    if protocol in ('both', 'percam'):
        protos += list(CAM_NAMES)
    if protocol in ('both', 'allcam'):
        protos += ['all']
    return [(c, a, m, p) for c in conditions for a in axes for m in mags
            for p in protos]


def shard_slice(cells, shard):
    i, n = (int(x) for x in shard.split('/'))
    return [c for k, c in enumerate(cells) if k % n == i]


def _cell_path(cell_dir, cond, axis, mag, proto):
    """Per-cell result-cache path (resume: skip already-done cells on restart)."""
    name = f"{cond}_{axis}_{mag:+d}_{proto}".replace('/', '-')
    return osp.join(cell_dir, name + '.pkl')


def _save_cell(path, obj):
    """Crash-safe cell-cache write: dump to a .tmp, fsync, then atomically replace.
    A SIGKILL mid-dump leaves only the .tmp truncated; `path` stays valid (or absent)."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        _pk.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)                # atomic on POSIX


def _load_cell(path):
    """Safe cell-cache read: return the cached obj, or None (dropping a corrupt/
    truncated cache so the cell recomputes) if it is missing or unpicklable."""
    try:
        with open(path, 'rb') as f:
            return _pk.load(f)
    except Exception:
        try:
            os.remove(path)             # drop the corrupt/truncated cache
        except OSError:
            pass
        return None


# --------------------------------------------------------------------------- #
# outputs / aggregation
# --------------------------------------------------------------------------- #
def write_outputs(outdir, args, nds_norm, m6_norm, rows, normal_metrics=None):
    os.makedirs(outdir, exist_ok=True)
    csv_path = osp.join(outdir, 'eval_vp_per_config.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'axis', 'mag', 'protocol', 'nds', 'map6', 'rrs'])
        for r in rows:
            w.writerow([r['cond'], r['axis'], r['mag'], r['proto'],
                        f"{r['nds']:.4f}", f"{r['map6']:.4f}", f"{r['rrs']:.4f}"])
    agg = {}
    for cond in sorted({r['cond'] for r in rows}):
        percam = [r['rrs'] for r in rows if r['cond'] == cond and r['proto'] != 'all']
        allcam = [r['rrs'] for r in rows if r['cond'] == cond and r['proto'] == 'all']
        mrrs = float(np.mean(percam)) if percam else float('nan')
        rrsall = float(np.mean(allcam)) if allcam else float('nan')
        agg[cond] = {'mRRS_percam': mrrs, 'RRSALL_allcam': rrsall,
                     'mVRS': 0.5 * (mrrs + rrsall)}
    # rows[].metrics + normal_metrics carry the FULL per-class detail dict
    # (per-class AP at 4 dist thresholds + per-class TP errors), so any subset
    # (e.g. 5-class) mAP/NDS can be recomputed offline from the saved values.
    js = {'tag': args.tag, 'frames_per_scene': args.frames_per_scene,
          'nds_normal': nds_norm, 'map6_normal': m6_norm,
          'normal_metrics': normal_metrics, 'aggregate': agg, 'rows': rows}
    with open(osp.join(outdir, 'eval_vp.json'), 'w') as f:
        json.dump(js, f, indent=2)
    lines = [f'VP viewpoint-robustness (BEVDet NDS)   tag={args.tag}',
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
    rows, nds_norm, m6_norm, norm_m = [], None, None, None
    for fn in sorted(os.listdir(outdir)):
        if not fn.startswith('shard_') or not fn.endswith('.json'):
            continue
        d = json.load(open(osp.join(outdir, fn)))
        rows.extend(d['rows'])
        nds_norm = d['nds_normal']
        m6_norm = d['map6_normal']
        norm_m = d.get('normal_metrics')
    if nds_norm is None:
        raise RuntimeError(f'no shard_*.json in {outdir}')
    write_outputs(outdir, args, nds_norm, m6_norm, rows, norm_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',
                    default='BEVDet/configs/bevdet/carla/bevdet-r50-carla.py')
    ap.add_argument('--ckpt',
                    default='BEVDet/work_dirs/bevdet-r50-carla_sedan/epoch_24.pth')
    ap.add_argument('--frames-per-scene', type=int, default=4)
    ap.add_argument('--conditions', nargs='+', default=['ER', 'VR', 'CR'],
                    choices=['ER', 'VR', 'CR'])
    ap.add_argument('--axes', nargs='+', default=B.VP_AXES)
    ap.add_argument('--mags', nargs='+', type=int, default=B.VP_MAGNITUDES)
    ap.add_argument('--protocol', default='both',
                    choices=['both', 'allcam', 'percam'])
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shard', default='0/1', help='i/n cell-shard')
    ap.add_argument('--tag', default='bevdet_sedan')
    ap.add_argument('--outdir', default=osp.join(HERE, 'out'))
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()

    outdir = osp.join(args.outdir, f'vp_{args.tag}')
    if args.merge:
        merge_shards(outdir, args)
        return

    # relative config/ckpt resolve against CWD (= the model repo; run_vp_<fw>.sh cd's there)
    config = args.config if osp.isabs(args.config) else osp.abspath(args.config)
    ckpt = args.ckpt if osp.isabs(args.ckpt) else osp.abspath(args.ckpt)
    stage_root = os.environ.get('VP_STAGE_ROOT')

    set_deterministic(0)
    patch_nuscenes_cache()
    cfg = Config.fromfile(config)
    # Register the model repo's plugin (CAPE/Petr3D/Detr3D etc.) so build_model
    # finds the custom detector. Run from the repo root (run_vp_<fw>.sh cd's there
    # + puts it on PYTHONPATH) so `projects.mmdet3d_plugin` imports.
    if cfg.get('plugin', False):
        import importlib
        plugin_mod = cfg.get('plugin_dir', 'projects/mmdet3d_plugin/').rstrip('/').replace('/', '.')
        importlib.import_module(plugin_mod)
    metadata, base = build_subset_infos(args.frames_per_scene)
    patch_eval_subset({info['token'] for info in base})
    model = build_model_once(cfg, ckpt)

    if stage_root:
        rp = collect_real_paths(base, args.conditions, args.axes, args.mags)
        ts = time.perf_counter()
        n_copied, gb = stage_images(rp, stage_root, workers=max(16, args.workers))
        print(f'[VP] staged {n_copied}/{len(rp)} images ({gb:.1f} GB) to '
              f'{stage_root} in {time.perf_counter()-ts:.0f}s', flush=True)

    print(f'[VP] model+GT loaded once | subset={len(base)} frames '
          f'(frames-per-scene={args.frames_per_scene}) | batch={args.batch} '
          f'workers={args.workers} | shard={args.shard} | '
          f'stage={"on" if stage_root else "off"}', flush=True)

    tmpdir = tempfile.mkdtemp(prefix='vp_bevdet_')
    # PER-CELL result cache (resume): each finished cell's row is dumped here the
    # moment it completes; on restart, done cells load from here and are SKIPPED.
    # Persistent (in outdir), unlike the tempfile.mkdtemp scratch above. Per-shard
    # to avoid clashes between concurrent shard jobs. Single process per cell here
    # (MMDataParallel, no torch.distributed), so no rank gating is needed.
    si0 = int(args.shard.split('/')[0])
    cell_dir = osp.join(outdir, f'cells_shard{si0}')
    os.makedirs(cell_dir, exist_ok=True)
    t0 = time.perf_counter()

    # Normal (oracle) -- every shard runs it (the RRS denominator).
    # resume: cache the oracle cell so a restart does not recompute it.
    _normal_path = osp.join(cell_dir, 'normal.pkl')
    _cached = _load_cell(_normal_path)
    if _cached is not None:
        _d = _cached
        nds_norm, m6_norm, norm_detail = _d['nds_norm'], _d['m6_norm'], _d['norm_detail']
        print(f'[VP] Normal (cached) NDS={nds_norm:.4f} mAP6={m6_norm:.4f}',
              flush=True)
    else:
        nds_norm, m6_norm, norm_detail = run_cell(
            model, cfg, metadata, base, 'Normal', None, None, None,
            osp.join(tmpdir, 'cell.pkl'), tmpdir, args.workers, args.batch,
            stage_root)
        _save_cell(_normal_path, {'nds_norm': nds_norm, 'm6_norm': m6_norm,
                                  'norm_detail': norm_detail})
        print(f'[VP] Normal NDS={nds_norm:.4f} mAP6={m6_norm:.4f} '
              f'({time.perf_counter()-t0:.0f}s)', flush=True)

    cells = shard_slice(all_cells(args.conditions, args.axes, args.mags,
                                  args.protocol), args.shard)
    rows = []
    for k, (cond, axis, mag, proto) in enumerate(cells):
        cp = _cell_path(cell_dir, cond, axis, mag, proto)
        _cached = _load_cell(cp)                  # resume: skip already-done cell
        if _cached is not None:
            row = _cached
            rows.append(row)
            print(f'[VP {k+1}/{len(cells)}] {cond} {axis}{mag:+d} {proto:14s} '
                  f'(cached) NDS={row["nds"]:.4f} RRS={row["rrs"]:.4f}', flush=True)
            continue
        tc = time.perf_counter()
        nds, m6, detail = run_cell(model, cfg, metadata, base, cond, axis, mag,
                                   proto, osp.join(tmpdir, 'cell.pkl'), tmpdir,
                                   args.workers, args.batch, stage_root)
        rrs = nds / nds_norm if nds_norm else float('nan')
        row = {'cond': cond, 'axis': axis, 'mag': mag, 'proto': proto,
               'nds': nds, 'map6': m6, 'rrs': rrs, 'metrics': detail}
        _save_cell(cp, row)                       # resume: per-cell save (atomic)
        rows.append(row)
        print(f'[VP {k+1}/{len(cells)}] {cond} {axis}{mag:+d} {proto:14s} '
              f'NDS={nds:.4f} RRS={rrs:.4f} ({time.perf_counter()-tc:.0f}s)',
              flush=True)

    i, n = (int(x) for x in args.shard.split('/'))
    os.makedirs(outdir, exist_ok=True)
    if n == 1:
        write_outputs(outdir, args, nds_norm, m6_norm, rows, norm_detail)
    else:
        sp = osp.join(outdir, f'shard_{i}of{n}.json')
        json.dump({'nds_normal': nds_norm, 'map6_normal': m6_norm,
                   'normal_metrics': norm_detail, 'rows': rows},
                  open(sp, 'w'), indent=2)
        print(f'[VP] shard {args.shard} wrote {sp} ({len(rows)} cells, '
              f'{time.perf_counter()-t0:.0f}s total)', flush=True)


if __name__ == '__main__':
    main()
