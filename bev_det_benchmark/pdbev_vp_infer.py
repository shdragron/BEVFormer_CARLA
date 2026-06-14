"""VP viewpoint-robustness INFERENCE for PD-BEV (Generalizable-BEV / BEVDepth_DG).

PD-BEV's NDS eval lives in a different env (CarlaNuScenesDataset is a BEVDet-repo
class, scored in bevdet-b200), so the BEVDet/BEVFormer single-process "infer+score
in one loop" pattern can't be used verbatim. Instead this is PHASE 1 of a two-phase
split:

  phase 1 (this file, pdbev-b200, run from the Generalizable-BEV repo):
      load BEVDepth_DG ONCE, loop the VP cells, and for each cell write the raw
      detections keyed by sample token to out/vp_<tag>/dets/<cellid>.pkl.
  phase 2 (pdbev_vp_score.py, bevdet-b200):
      load the verified CarlaNuScenesDataset (sedan, visibility>=2) + NuScenes GT
      ONCE, score every cell's dets on the SAME frozen 768-frame subset, compute
      RRS = NDS_cell / NDS_Normal and the mRRS/RRSALL/mVRS aggregates.

Cells mirror the other detectors EXACTLY (so the cross-model table stays matched):
  Normal | ER (extrinsic) | VR (image, primary) | CR (both)
  axes yaw/pitch/roll x signed mags {4,8,12,16,20} x protocols (6 per-cam + all-cam)
  Full grid = 1 + 3*3*10*7 = 631 cells. The committed 768-subset uses
  --frames-per-scene 16 (630 cells + Normal).

The condition swaps (make_vp_infos) and the carla_VR 2N frame-doubling fix are
reused verbatim from build_condition_pkls_bevdet -- PD-BEV reads the IDENTICAL
sedan val pkl (same inode) and the same cams[CAM]['data_path' / 'sensor2ego_*']
fields. PrepareImageInputs_UDA opens data_path directly (no data_root join), so
relative baseline paths resolve from the repo-root CWD and absolute carla_VR
variant paths load as-is.

Crash-resumable: a cell whose dets pkl already exists is skipped. --shard i/n
splits cells across GPUs.

  cd Generalizable-BEV
  python ../bev_det_benchmark/pdbev_vp_infer.py \
      --config configs/bevdet_our/bevdepth-r50-cbgs-pc-carla-sedan384.py \
      --ckpt   work_dirs/pdbev-r50-cbgs-CARLA-dg-sedan384/epoch_24.pth \
      --frames-per-scene 16 --protocol both --tag pdbev_sedan384 --shard 0/2
"""
import argparse
import copy
import json
import os
import os.path as osp
import pickle
import sys
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

HERE = osp.dirname(osp.abspath(__file__))
BEVF_ROOT = osp.dirname(HERE)
sys.path.insert(0, HERE)
import build_condition_pkls_bevdet as B  # noqa: E402

CAM_NAMES = B.CAM_NAMES


def set_deterministic(seed=0):
    from mmdet.apis import set_random_seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    set_random_seed(seed, deterministic=True)


def build_subset_infos(frames_per_scene):
    """Frozen N-frames-per-scene subset of the sedan val pkl (even stride),
    IDENTICAL logic to the BEVDet/BEVDepth drivers so the 768 subset matches.
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


def build_model_once(cfg, ckpt):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location='cpu')
    for m in model.modules():            # DeformConv im2col tiling (NDS-neutral)
        if hasattr(m, 'im2col_step'):
            m.im2col_step = 6
    model = MMDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    model.eval()
    return model


def infer_dets(model, cfg, metadata, infos, ann_path, workers, batch):
    """Write the swapped subset pkl, build the PD-BEV val dataset/loader, run the
    tools/test.py forward path, and return {token: {boxes,scores,labels}} raw numpy
    (keyed by ds.data_infos token order -> robust to any internal re-sort)."""
    B.dump_pkl({'metadata': metadata, 'infos': infos}, ann_path)
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = ann_path
    test_cfg['test_mode'] = True
    ds = build_dataset(test_cfg)
    loader = build_dataloader(ds, samples_per_gpu=batch, workers_per_gpu=workers,
                              dist=False, shuffle=False)
    tokens = [info['token'] for info in ds.data_infos]
    dets, k = {}, 0
    with torch.no_grad():
        for data in loader:
            out = model(return_loss=False, rescale=True, **data)
            for o in out:
                pbx = o['pts_bbox']
                dets[tokens[k]] = {
                    'boxes': pbx['boxes_3d'].tensor.cpu().numpy().astype(np.float32),
                    'scores': pbx['scores_3d'].cpu().numpy().astype(np.float32),
                    'labels': pbx['labels_3d'].cpu().numpy().astype(np.int64)}
                k += 1
    assert k == len(tokens), f'{k} preds vs {len(tokens)} tokens'
    return dets


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


def shard_slice(cells, shard):
    i, n = (int(x) for x in shard.split('/'))
    return [c for k, c in enumerate(cells) if k % n == i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--frames-per-scene', type=int, default=16)
    ap.add_argument('--conditions', nargs='+', default=['ER', 'VR', 'CR'],
                    choices=['ER', 'VR', 'CR'])
    ap.add_argument('--axes', nargs='+', default=B.VP_AXES)
    ap.add_argument('--mags', nargs='+', type=int, default=B.VP_MAGNITUDES)
    ap.add_argument('--protocol', default='both',
                    choices=['both', 'allcam', 'percam'])
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shard', default='0/1', help='i/n cell-shard')
    ap.add_argument('--tag', default='pdbev_sedan384')
    ap.add_argument('--outdir', default=osp.join(HERE, 'out'))
    args = ap.parse_args()

    config = args.config if osp.isabs(args.config) else osp.abspath(args.config)
    ckpt = args.ckpt if osp.isabs(args.ckpt) else osp.abspath(args.ckpt)
    outdir = osp.join(args.outdir, f'vp_{args.tag}')
    detsdir = osp.join(outdir, 'dets')
    os.makedirs(detsdir, exist_ok=True)

    torch.multiprocessing.set_sharing_strategy('file_system')
    set_deterministic(0)
    cfg = Config.fromfile(config)
    metadata, base = build_subset_infos(args.frames_per_scene)

    # Persist the subset token list + cell index so phase-2 scoring uses the
    # IDENTICAL frozen subset and can map cellid -> (cond,axis,mag,proto).
    sub_tokens = [info['token'] for info in base]
    meta_path = osp.join(outdir, 'subset_meta.json')
    if not osp.exists(meta_path):
        json.dump({'frames_per_scene': args.frames_per_scene,
                   'n_frames': len(base), 'tokens': sub_tokens,
                   'tag': args.tag}, open(meta_path, 'w'))

    model = build_model_once(cfg, ckpt)
    print(f'[VP-infer] model loaded | subset={len(base)} frames '
          f'(fps={args.frames_per_scene}) | batch={args.batch} '
          f'workers={args.workers} | shard={args.shard}', flush=True)

    # Normal first (every shard writes it; phase-2 reads it as the RRS denom).
    cells = [('Normal', None, None, None)] + all_cells(
        args.conditions, args.axes, args.mags, args.protocol)
    cells = shard_slice(cells, args.shard) if args.shard != '0/1' else cells
    # Normal must exist for scoring; ensure shard 0 owns it (or run unsharded).
    if ('Normal', None, None, None) not in cells:
        cells = [('Normal', None, None, None)] + cells

    ann_tmp = osp.join(outdir, f'_cell_{args.shard.replace("/", "of")}.pkl')
    t0 = time.perf_counter()
    n_new = n_skip = 0
    for k, (cond, axis, mag, proto) in enumerate(cells):
        cid = cell_id(cond, axis, mag, proto)
        out_pkl = osp.join(detsdir, f'{cid}.pkl')
        if osp.exists(out_pkl):
            n_skip += 1
            continue
        tc = time.perf_counter()
        if cond == 'Normal':
            infos = copy.deepcopy(base)
        else:
            infos = B.make_vp_infos(base, cond, axis, mag, proto)
        dets = infer_dets(model, cfg, metadata, infos, ann_tmp,
                          args.workers, args.batch)
        tmp = out_pkl + f'.tmp.{os.getpid()}'
        with open(tmp, 'wb') as f:
            pickle.dump(dets, f)
        os.replace(tmp, out_pkl)
        n_new += 1
        print(f'[VP-infer {k+1}/{len(cells)}] {cid:32s} '
              f'{len(dets)} frames ({time.perf_counter()-tc:.0f}s)', flush=True)
    print(f'[VP-infer] shard {args.shard}: {n_new} new + {n_skip} skipped '
          f'-> {detsdir} ({time.perf_counter()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
