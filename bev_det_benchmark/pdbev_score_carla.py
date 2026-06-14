"""Score PD-BEV's dumped val detections with the VERIFIED BEVDet CARLA 6-class NDS
(CarlaNuScenesDataset.evaluate: v1.0-carla_<veh>_eval, visibility>=2). Run in the
bevdet-b200 env. Token-aligns the dumped dets to CarlaNuScenesDataset.data_infos order
(the devkit requires pred tokens == gt tokens; the dataset reorders by token).

  python pdbev_score_carla.py <dets.pkl> [--vehicle sedan]

Prints CARLA 6-class NDS/mAP -- apples-to-apples with the other detectors.
"""
import argparse, os, pickle, tempfile
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

BEVDET = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDet'
CFG = {'sedan': 'configs/bevdet/carla/bevdet-r50-carla.py',
       'suv':   'configs/bevdet/carla/bevdet-r50-carla_suv.py',
       'bus':   'configs/bevdet/carla/bevdet-r50-carla_bus.py'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dets')
    ap.add_argument('--vehicle', default='sedan', choices=list(CFG))
    args = ap.parse_args()

    dets_path = os.path.abspath(args.dets)
    os.chdir(BEVDET)   # configs use relative ann_file (data/nuscenes/...): cwd must be BEVDet
    cfg = Config.fromfile(os.path.join(BEVDET, CFG[args.vehicle]))
    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)            # CarlaNuScenesDataset <veh> val + v1.0-carla_<veh>_eval
    dets = pickle.load(open(dets_path, 'rb'))
    miss = [info['token'] for info in ds.data_infos if info['token'] not in dets]
    if miss:
        raise SystemExit(f'{len(miss)} val tokens missing from dets (need full-val dump); e.g. {miss[:2]}')

    outputs = []
    for info in ds.data_infos:                   # devkit needs pred order == gt (data_infos) order
        d = dets[info['token']]
        bx = np.asarray(d['boxes'], np.float32)
        outputs.append({'pts_bbox': {
            'boxes_3d': LiDARInstance3DBoxes(torch.tensor(bx), box_dim=bx.shape[1]),
            'scores_3d': torch.tensor(np.asarray(d['scores'], np.float32)),
            'labels_3d': torch.tensor(np.asarray(d['labels'], np.int64)),
        }})
    tmp = tempfile.mkdtemp(prefix='pdbev_score_')
    res = ds.evaluate(outputs, metric='bbox', jsonfile_prefix=os.path.join(tmp, 'eval'))
    P = 'pts_bbox_NuScenes/'
    nds, mAP = res.get(P + 'NDS'), res.get(P + 'mAP')
    print(f'\n[PDBEV-CARLA-{args.vehicle}] 6-class NDS={nds:.4f} mAP={mAP:.4f}', flush=True)
    print('NDS_RESULT', f'{nds:.4f}', f'{mAP:.4f}', flush=True)


if __name__ == '__main__':
    main()
