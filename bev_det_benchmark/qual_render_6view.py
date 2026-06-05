"""Qualitative BEVDet CARLA viz -- mirrors BEVDepth/_viz_qual.py for a fair, matched
comparison. Projects GT (green) + native-model predicted (red) 3D boxes onto the 6
original camera images per platform (sedan/suv/bus) for the highest IN-RANGE-object
val samples. Output: a 2x3 six-view grid per (platform, sample), one set per score
threshold (thr0.3, thr0.5).

Coordinate frame (verified): GT (ann_infos) and preds are both in the global-axes
EGO frame (the frame sensor2ego maps to); pixel = K @ inv(sensor2ego) @ corners on
the ORIGINAL image (intrinsic principal point == image centre, no IDA). Only in-pc-
range (|x|,|y|<51.2) boxes whose centre is in front + inside the camera FOV (and all
8 corners ahead of the near plane) are drawn -- same culling as the BEVDepth script.

Inputs from /tmp/qual/: {veh}_qual10.pkl (10 label-aligned infos), preds_{veh}.pkl
(native-model dets), labels.json. Run in the bevdet-b200 env (CPU is fine):
    CUDA_VISIBLE_DEVICES="" python qual_render_6view.py
"""
import os, json, pickle
import numpy as np
import cv2
import mmcv
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

ROOT = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDet'
OUTDIR = '/home/hanyan_arch/viewpoint/BEVFormer/results/BEVDet/qual'
CAM_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
VEHICLES = ['sedan', 'suv', 'bus']
THRESHOLDS = [0.30, 0.50]
PC = 51.2                       # detection / point-cloud range (xy half-extent)
GT_COLOR = (0, 230, 0)          # green (BGR)
PRED_COLOR = (0, 0, 255)        # red   (BGR)
PLATFORM_TAG = {'sedan': 'subcompact', 'suv': 'suv', 'bus': 'bus'}


def ego2cam_and_K(cam):
    """inv(sensor2ego) (ego->cam 4x4) and the 3x3 intrinsic."""
    R = Quaternion(cam['sensor2ego_rotation']).rotation_matrix      # cam->ego
    t = np.asarray(cam['sensor2ego_translation'], np.float64)
    e2c = np.eye(4)
    e2c[:3, :3] = R.T
    e2c[:3, 3] = -R.T @ t
    return e2c, np.asarray(cam['cam_intrinsic'], np.float64)


def in_range_mask(boxes):
    """gravity-centre within the +/-PC pc-range square (matches eval range)."""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    c = boxes.gravity_center.numpy()
    return (np.abs(c[:, 0]) < PC) & (np.abs(c[:, 1]) < PC)


def visible_mask(boxes, e2c, K, W=1600, H=900):
    """Same cull as BEVDepth/_viz_qual.draw_boxes: centre depth in [1.5,70], centre
    projects within image+/-100px, and every corner ahead of the near plane (z>0.3)."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, bool)
    R, t = e2c[:3, :3], e2c[:3, 3]
    ctr = (boxes.gravity_center.numpy() @ R.T) + t                  # (n,3) cam
    z = ctr[:, 2]
    cu = ctr @ K.T
    cx = cu[:, 0] / np.clip(cu[:, 2], 1e-3, None)
    cy = cu[:, 1] / np.clip(cu[:, 2], 1e-3, None)
    centre_ok = (z > 1.5) & (z < 70) & (cx > -100) & (cx < W + 100) \
        & (cy > -100) & (cy < H + 100)
    corn = (boxes.corners.numpy().reshape(-1, 3) @ R.T) + t
    corn_ok = (corn[:, 2].reshape(n, 8) > 0.3).all(1)
    return centre_ok & corn_ok


def main():
    labels = json.load(open('/tmp/qual/labels.json'))
    for veh in VEHICLES:
        infos = pickle.load(open(f'/tmp/qual/{veh}_qual10.pkl', 'rb'))['infos']
        preds = mmcv.load(f'/tmp/qual/preds_{veh}.pkl')
        for i, (info, pr) in enumerate(zip(infos, preds)):
            gtb = np.asarray(info['ann_infos'][0], np.float32).reshape(-1, 9)
            GT_all = LiDARInstance3DBoxes(gtb, box_dim=9, origin=(0.5, 0.5, 0.5))
            GT = GT_all[in_range_mask(GT_all)]
            pb = pr['pts_bbox']
            score = pb['scores_3d'].numpy()
            base = {cam: mmcv.imread(os.path.join(ROOT, info['cams'][cam]['data_path']))
                    for cam in CAM_NAMES}
            e2c = {cam: ego2cam_and_K(info['cams'][cam]) for cam in CAM_NAMES}
            for thr in THRESHOLDS:
                pm = score >= thr
                PRED_all = pb['boxes_3d'][pm]
                PRED = PRED_all[in_range_mask(PRED_all)]
                panels = {}
                for cam in CAM_NAMES:
                    img = base[cam].copy()
                    M, K = e2c[cam]
                    l2i = np.eye(4); l2i[:3, :3] = K; l2i = l2i @ M
                    mg, mp = visible_mask(GT, M, K), visible_mask(PRED, M, K)
                    if mg.any():
                        img = draw_lidar_bbox3d_on_img(GT[mg], img, l2i, {},
                                                       color=GT_COLOR, thickness=3)
                    if mp.any():
                        img = draw_lidar_bbox3d_on_img(PRED[mp], img, l2i, {},
                                                       color=PRED_COLOR, thickness=2)
                    cv2.putText(img, cam, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (255, 255, 255), 3, cv2.LINE_AA)
                    panels[cam] = img
                h, w = panels['CAM_FRONT'].shape[:2]
                grid = np.vstack([np.hstack([panels[c] for c in row]) for row in GRID])
                cv2.putText(grid, f'{veh} ({PLATFORM_TAG[veh]})  {labels[i]}  native model, normal  '
                            f'GT(green,in-range)={len(GT)}  pred>{thr:.1f}(red)={len(PRED)}',
                            (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 255, 255), 3, cv2.LINE_AA)
                out = os.path.join(OUTDIR, f'thr{thr:.1f}', f'{veh}_{labels[i]}.jpg')
                os.makedirs(os.path.dirname(out), exist_ok=True)
                cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f'[{veh}] {labels[i]} thr{thr:.1f} GT={len(GT)} pred={len(PRED)} -> {out}',
                      flush=True)


if __name__ == '__main__':
    main()
