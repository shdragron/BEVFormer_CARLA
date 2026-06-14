"""Dump PD-BEV (Generalizable-BEV) val detections to raw arrays keyed by sample token,
so the VERIFIED BEVDet CarlaNuScenesDataset.evaluate (bevdet-b200 env) can score them
without re-porting the eval. Run in the pdbev-b200 env from the Generalizable-BEV repo.

  python pdbev_dump_val.py <config> <ckpt> <out.pkl> [--gpu 0]

Out: {token: {'boxes': (N,box_dim) f32, 'scores': (N,), 'labels': (N,)}} -- raw numpy
(no LiDARInstance3DBoxes pickling across envs).
"""
import argparse, os, pickle
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
import mmcv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--ann-file', default=None,
                    help='override cfg.data.test.ann_file (e.g. a CTS condition pkl)')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    if args.ann_file is not None:
        cfg.data.test.ann_file = os.path.abspath(args.ann_file)
    ds = build_dataset(cfg.data.test)
    loader = build_dataloader(ds, samples_per_gpu=args.batch, workers_per_gpu=args.workers,
                              dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    tokens = [info['token'] for info in ds.data_infos]
    dets, k = {}, 0
    pb = mmcv.ProgressBar(len(ds))
    for data in loader:
        with torch.no_grad():
            out = model(return_loss=False, rescale=True, **data)
        for o in out:
            pbx = o['pts_bbox']
            dets[tokens[k]] = {
                'boxes': pbx['boxes_3d'].tensor.cpu().numpy().astype(np.float32),
                'scores': pbx['scores_3d'].cpu().numpy().astype(np.float32),
                'labels': pbx['labels_3d'].cpu().numpy().astype(np.int64),
            }
            k += 1
        pb.update(len(out))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump(dets, open(args.out, 'wb'))
    print(f"\ndumped {len(dets)} frames (box_dim="
          f"{next(iter(dets.values()))['boxes'].shape[1] if dets else '?'}) -> {args.out}")


if __name__ == '__main__':
    main()
