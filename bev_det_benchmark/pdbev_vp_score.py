"""VP viewpoint-robustness SCORING for PD-BEV -- phase 2 (bevdet-b200 env).

Scores the per-cell detection pkls written by pdbev_vp_infer.py with the VERIFIED
BEVDet CarlaNuScenesDataset 6-class NDS (visibility>=2), on the SAME frozen 768
subset. Mirrors the BEVDet VP driver's scoring half:
  * NuScenes GT DB cached so it loads ONCE across all cells.
  * eval GT restricted to the subset tokens (patch_eval_subset) so the devkit's
    pred==gt assertion holds.
  * RRS = NDS_cell / NDS_Normal ; mRRS_c=mean per-cam ; RRSALL_c=all-cam ;
    mVRS_c=0.5*(mRRS_c+RRSALL_c). VR primary.
Outputs out/vp_<tag>/{eval_vp_per_config.csv,eval_vp.json,eval_vp_summary.txt}.

  conda activate bevdet-b200
  python pdbev_vp_score.py --tag pdbev_sedan384 --vehicle sedan
"""
import argparse
import csv
import json
import os
import os.path as osp
import sys
import tempfile

import numpy as np
import torch
from mmcv import Config

HERE = osp.dirname(osp.abspath(__file__))
BEVF_ROOT = osp.dirname(HERE)
BEVDET = osp.join(BEVF_ROOT, 'BEVDet')
sys.path.insert(0, HERE)
import build_condition_pkls_bevdet as B  # noqa: E402

P = 'pts_bbox_NuScenes/'
CAM_NAMES = B.CAM_NAMES
CFG = {'sedan': 'configs/bevdet/carla/bevdet-r50-carla.py',
       'suv':   'configs/bevdet/carla/bevdet-r50-carla_suv.py',
       'bus':   'configs/bevdet/carla/bevdet-r50-carla_bus.py'}


def patch_nuscenes_cache():
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
    from nuscenes.eval.common import loaders as nl
    real = nl.load_gt

    def load_gt_subset(nusc, eval_split, box_cls, verbose=False):
        gt = real(nusc, eval_split, box_cls, verbose=verbose)
        for st in list(gt.boxes.keys()):
            if st not in allowed_tokens:
                del gt.boxes[st]
        return gt
    nl.load_gt = load_gt_subset


def all_cells(conditions, axes, mags, protocol):
    protos = []
    if protocol in ('both', 'percam'):
        protos += list(CAM_NAMES)
    if protocol in ('both', 'allcam'):
        protos += ['all']
    return [(c, a, m, p) for c in conditions for a in axes for m in mags
            for p in protos]


def cell_id(cond, axis, mag, proto):
    if cond == 'Normal':
        return 'Normal'
    return f'{cond}_{axis}_{mag:+d}_{proto}'


def score_one(ds, dets, tmpdir):
    """6-class NDS for one cell's dets on the (subset-filtered) dataset."""
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    outputs = []
    for info in ds.data_infos:
        d = dets[info['token']]
        bx = np.asarray(d['boxes'], np.float32)
        outputs.append({'pts_bbox': {
            'boxes_3d': LiDARInstance3DBoxes(torch.tensor(bx), box_dim=bx.shape[1]),
            'scores_3d': torch.tensor(np.asarray(d['scores'], np.float32)),
            'labels_3d': torch.tensor(np.asarray(d['labels'], np.int64))}})
    res = ds.evaluate(outputs, metric='bbox',
                      jsonfile_prefix=osp.join(tmpdir, 'eval'))
    return float(res.get(P + 'NDS')), float(res.get(P + 'mAP')), dict(res)


def write_outputs(outdir, tag, fps, nds_norm, m6_norm, rows, norm_metrics):
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
        # 1/7 mVRS = mean over all 7 protocols (6 per-cam + 1 all-cam) = the
        # headline metric used in the paper Table 2: (6*mRRS + RRSALL)/7.
        mvrs_1of7 = (6 * mrrs + rrsall) / 7
        agg[cond] = {'mRRS_percam': mrrs, 'RRSALL_allcam': rrsall,
                     'mVRS': 0.5 * (mrrs + rrsall), 'mVRS_1of7': mvrs_1of7}
    js = {'tag': tag, 'frames_per_scene': fps, 'nds_normal': nds_norm,
          'map6_normal': m6_norm, 'normal_metrics': norm_metrics,
          'aggregate': agg, 'rows': rows}
    json.dump(js, open(osp.join(outdir, 'eval_vp.json'), 'w'), indent=2)
    lines = [f'VP viewpoint-robustness (PD-BEV NDS)   tag={tag}',
             f'  frames-per-scene={fps}  NDS_Normal={nds_norm:.4f}  '
             f'mAP6_Normal={m6_norm:.4f}',
             '  RRS = NDS_cell / NDS_Normal   [VR primary]',
             '  mVRS(1/7) = (6*mRRS + RRSALL)/7  = paper Table-2 headline', '',
             '  condition   mRRS(per-cam)   RRSALL(all-cam)    mVRS(1/2)   mVRS(1/7)']
    for cond in ['ER', 'VR', 'CR']:
        if cond not in agg:
            continue
        a = agg[cond]
        star = '  <- primary' if cond == 'VR' else ''
        lines.append(f'  {cond:<10}{a["mRRS_percam"]:>13.4f}'
                     f'{a["RRSALL_allcam"]:>16.4f}{a["mVRS"]:>12.4f}'
                     f'{a["mVRS_1of7"]:>12.4f}{star}')
    lines.append('')
    lines.append('  Table-2 row (1/7 mVRS %, ER=Ext VR=Img CR=Cal): '
                 + ' '.join(f'{c}={agg[c]["mVRS_1of7"]*100:.1f}'
                            for c in ['ER', 'VR', 'CR'] if c in agg))
    txt = '\n'.join(lines)
    open(osp.join(outdir, 'eval_vp_summary.txt'), 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'\nwrote {csv_path}\n      {osp.join(outdir, "eval_vp.json")}\n'
          f'      {osp.join(outdir, "eval_vp_summary.txt")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='pdbev_sedan384')
    ap.add_argument('--vehicle', default='sedan', choices=list(CFG))
    ap.add_argument('--conditions', nargs='+', default=['ER', 'VR', 'CR'])
    ap.add_argument('--axes', nargs='+', default=B.VP_AXES)
    ap.add_argument('--mags', nargs='+', type=int, default=B.VP_MAGNITUDES)
    ap.add_argument('--protocol', default='both',
                    choices=['both', 'allcam', 'percam'])
    ap.add_argument('--outdir', default=osp.join(HERE, 'out'))
    ap.add_argument('--shard', default='0/1', help='i/n cell-shard for parallel CPU scoring')
    ap.add_argument('--merge', action='store_true', help='combine score_shard_*.json -> eval_vp')
    args = ap.parse_args()

    outdir = osp.join(args.outdir, f'vp_{args.tag}')
    detsdir = osp.join(outdir, 'dets')
    meta = json.load(open(osp.join(outdir, 'subset_meta.json')))
    sub_tokens = set(meta['tokens'])
    fps = meta['frames_per_scene']

    if args.merge:
        rows, nds_norm, m6_norm, norm_metrics = [], None, None, None
        for fn in sorted(os.listdir(outdir)):
            if not fn.startswith('score_shard_') or not fn.endswith('.json'):
                continue
            d = json.load(open(osp.join(outdir, fn)))
            rows.extend(d['rows']); nds_norm = d['nds_normal']
            m6_norm = d['map6_normal']; norm_metrics = d.get('normal_metrics')
        if nds_norm is None:
            raise SystemExit(f'no score_shard_*.json in {outdir}')
        print(f'[VP-score merge] {len(rows)} cells from shards', flush=True)
        write_outputs(outdir, args.tag, fps, nds_norm, m6_norm, rows, norm_metrics)
        return

    os.chdir(BEVDET)                       # configs use relative ann_file
    patch_nuscenes_cache()
    patch_eval_subset(sub_tokens)
    cfg = Config.fromfile(osp.join(BEVDET, CFG[args.vehicle]))
    cfg.data.test.test_mode = True
    from mmdet3d.datasets import build_dataset
    ds = build_dataset(cfg.data.test)
    ds.data_infos = [d for d in ds.data_infos if d['token'] in sub_tokens]
    print(f'[VP-score] {args.vehicle} subset={len(ds.data_infos)} frames '
          f'(fps={fps}); expected {meta["n_frames"]}', flush=True)
    assert len(ds.data_infos) == meta['n_frames'], 'subset token mismatch'
    tmpdir = tempfile.mkdtemp(prefix='vp_pdbev_score_')

    import pickle
    # Normal (RRS denominator)
    nd = pickle.load(open(osp.join(detsdir, 'Normal.pkl'), 'rb'))
    nds_norm, m6_norm, norm_metrics = score_one(ds, nd, tmpdir)
    print(f'[VP-score] Normal NDS={nds_norm:.4f} mAP6={m6_norm:.4f}', flush=True)

    rows = []
    cells = all_cells(args.conditions, args.axes, args.mags, args.protocol)
    si, sn = (int(x) for x in args.shard.split('/'))
    my_cells = [(k, c) for k, c in enumerate(cells) if k % sn == si]
    miss = []
    for j, (k, (cond, axis, mag, proto)) in enumerate(my_cells):
        cid = cell_id(cond, axis, mag, proto)
        pk = osp.join(detsdir, f'{cid}.pkl')
        if not osp.exists(pk):
            miss.append(cid)
            continue
        dets = pickle.load(open(pk, 'rb'))
        nds, m6, _ = score_one(ds, dets, tmpdir)
        rrs = nds / nds_norm if nds_norm else float('nan')
        rows.append({'cond': cond, 'axis': axis, 'mag': mag, 'proto': proto,
                     'nds': nds, 'map6': m6, 'rrs': rrs})
        if (j + 1) % 25 == 0 or j + 1 == len(my_cells):
            print(f'[VP-score {args.shard} {j+1}/{len(my_cells)}] {cid:32s} '
                  f'NDS={nds:.4f} RRS={rrs:.4f}', flush=True)
    if miss:
        print(f'[VP-score] WARNING: {len(miss)} cells missing dets '
              f'(e.g. {miss[:3]}) -- infer incomplete?', flush=True)
    if sn == 1:
        write_outputs(outdir, args.tag, fps, nds_norm, m6_norm, rows, norm_metrics)
    else:
        sp = osp.join(outdir, f'score_shard_{si}of{sn}.json')
        json.dump({'nds_normal': nds_norm, 'map6_normal': m6_norm,
                   'normal_metrics': norm_metrics, 'rows': rows}, open(sp, 'w'))
        print(f'[VP-score] shard {args.shard} wrote {len(rows)} cells -> {sp}',
              flush=True)


if __name__ == '__main__':
    main()
