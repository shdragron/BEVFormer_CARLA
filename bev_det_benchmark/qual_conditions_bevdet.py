"""Condition-aware qual for BEVDet: ONE scene, VP {Normal/ER/VR/CR}@(axis,mag) and
CTS-bus {NORMAL/EXT/IMG/CAL}, GT (green) + native-sedan-model pred (red) 3D boxes.

BEVDet-specific coordinate recipe (cf. BEVDepth's qual_conditions.py which uses
sensor2lidar): `build_condition_pkls_bevdet.make_vp_infos` perturbs **sensor2ego**
(BEVDet's PrepareImageInputs lifts the frustum with sensor2ego); sensor2lidar is left
STALE. So GT (ego-frame `ann_infos`) AND pred (ego-frame model output) are projected via
`l2i = K @ inv(sensor2ego)` using EACH CONDITION'S OWN sensor2ego (perturbed for ER/CR/EXT,
sedan for VR/IMG-img-only & NORMAL). NORMAL/CAL boxes wrap objects; IMG/EXT boxes float
(image tilted/swapped but extrinsic mismatched) — that float IS the domain gap the model sees.

Inputs (built by the caller into /tmp/qual_cond/): {VP_*,CTS-bus_*}.pkl (1-frame condition
infos) + *_pred.pkl (sedan-model dets on each). Out: results/BEVDet/qual_conditions/<tag>/.
CPU only. Edges drawn directly (mmdet3d draw needs np.int).
"""
import os, pickle
import numpy as np
import cv2, mmcv
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes

QC = '/tmp/qual_cond'
OUT = '/home/hanyan_arch/viewpoint/BEVFormer/results/BEVDet/qual_conditions'
ROOT = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDet'
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
PC = 51.2
SCORE_THR = 0.30
GT_COLOR = (0, 230, 0)      # green
PRED_COLOR = (0, 0, 255)    # red
_EDGES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
          (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
VP_LABEL = {'Normal': 'NORMAL', 'ER': 'EXT', 'VR': 'IMG', 'CR': 'CAL'}   # VP internal -> perturbation-type
TAG = os.environ.get('TAG', 'scene-0269-frame-0150_yaw+12')


def ego2img(cam):
    R = Quaternion(cam['sensor2ego_rotation']).rotation_matrix      # cam->ego
    t = np.asarray(cam['sensor2ego_translation'], float)
    e2c = np.eye(4); e2c[:3, :3] = R.T; e2c[:3, 3] = -R.T @ t       # ego->cam
    K = np.eye(4); K[:3, :3] = np.asarray(cam['cam_intrinsic'], float)
    return K @ e2c, e2c


def vis_mask(boxes, e2c):
    n = len(boxes)
    if n == 0:
        return np.zeros(0, bool)
    R, t = e2c[:3, :3], e2c[:3, 3]
    c = boxes.gravity_center.numpy()
    inr = (np.abs(c[:, 0]) < PC) & (np.abs(c[:, 1]) < PC)
    ctr = (c @ R.T) + t
    corn = (boxes.corners.numpy().reshape(-1, 3) @ R.T) + t
    corn_ok = (corn[:, 2].reshape(n, 8) > 0.3).all(1)
    return inr & (ctr[:, 2] > 1.0) & (ctr[:, 2] < 80) & corn_ok


def draw(img, boxes, l2i, color, th=3):
    if len(boxes) == 0:
        return img
    cor = boxes.corners.numpy()
    ph = np.concatenate([cor.reshape(-1, 3), np.ones((len(cor) * 8, 1))], 1)
    uvw = ph @ np.asarray(l2i).T
    uv = (uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-4, None)).reshape(-1, 8, 2)
    for i in range(len(cor)):
        p = uv[i].astype(np.int32)
        for a, b in _EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), color, th, cv2.LINE_AA)
    return img


def gt_boxes(info):
    gtb = np.asarray(info['ann_infos'][0], np.float32).reshape(-1, 9)
    return LiDARInstance3DBoxes(gtb, box_dim=9, origin=(0.5, 0.5, 0.5))


def pred_boxes(pred):
    pb = pred['pts_bbox']
    keep = pb['scores_3d'].numpy() >= SCORE_THR
    return pb['boxes_3d'][keep]


def render_cam(cam, GT, PR):
    img = mmcv.imread(cam['data_path'] if os.path.isabs(cam['data_path'])
                      else os.path.join(ROOT, cam['data_path']))
    if img is None:
        img = np.zeros((900, 1600, 3), np.uint8)
        cv2.putText(img, 'MISSING IMG', (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        return img
    l2i, e2c = ego2img(cam)
    mg, mp = vis_mask(GT, e2c), vis_mask(PR, e2c)
    img = draw(img, GT[mg], l2i, GT_COLOR, 3)
    img = draw(img, PR[mp], l2i, PRED_COLOR, 2)
    return img


def six_view(info, GT, PR, banner):
    panels = {}
    for cam in CAMS:
        im = render_cam(info['cams'][cam], GT, PR)
        cv2.putText(im, cam, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        panels[cam] = im
    grid = np.vstack([np.hstack([panels[c] for c in row]) for row in GRID])
    cv2.putText(grid, banner, (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 255, 255), 3, cv2.LINE_AA)
    return grid, panels['CAM_FRONT']


def main():
    odir = f'{OUT}/{TAG}'
    os.makedirs(odir, exist_ok=True)
    fronts = {}   # (study, perturb-type) -> CAM_FRONT panel
    studies = [('VP', ['Normal', 'ER', 'VR', 'CR']),
               ('CTS-bus', ['NORMAL', 'EXT', 'IMG', 'CAL'])]
    for study, conds in studies:
        for cond in conds:
            info = pickle.load(open(f'{QC}/{study}_{cond}.pkl', 'rb'))['infos'][0]
            pred = mmcv.load(f'{QC}/{study}_{cond}_pred.pkl')[0]
            GT, PR = gt_boxes(info), pred_boxes(pred)
            ptype = VP_LABEL[cond] if study == 'VP' else cond
            banner = (f'{study} {cond}' + (f' (~{ptype})' if study == 'VP' else '')
                      + f'  {TAG}  GT(green,vis>=2)={len(GT)}  pred>{SCORE_THR}(red)={len(PR)}')
            grid, front = six_view(info, GT, PR, banner)
            cv2.imwrite(f'{odir}/{study}_{cond}_6view.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
            fronts[(study, ptype)] = (front, f'{study} {cond}')
            print(f'[{study} {cond}] GT={len(GT)} pred={len(PR)} -> {study}_{cond}_6view.jpg', flush=True)

    # CAM_FRONT montage: rows = perturbation type, cols = [VP, CTS-bus]
    rows = []
    for ptype in ['NORMAL', 'EXT', 'IMG', 'CAL']:
        cells = []
        for study in ['VP', 'CTS-bus']:
            im, lab = fronts[(study, ptype)]
            im = im.copy()
            cv2.putText(im, lab, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 4, cv2.LINE_AA)
            cells.append(im)
        rows.append(np.hstack(cells))
    montage = np.vstack(rows)
    montage = cv2.resize(montage, (montage.shape[1] // 2, montage.shape[0] // 2))
    cv2.imwrite(f'{odir}/compare_CAM_FRONT.jpg', montage, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'[montage] {montage.shape[1]}x{montage.shape[0]} -> compare_CAM_FRONT.jpg\nDONE -> {odir}')


if __name__ == '__main__':
    main()
