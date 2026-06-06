"""Condition-aware BEVDet qual, TRUE-POSE variant (per user spec): show GT+pred on the
ACTUAL image projected with THAT image's true camera pose, so GT lands on the visible
objects and the prediction reveals the degradation as a shift (no float).

Layout per study cell (2x2 CAM_FRONT montage), columns/rows = perturbation type:
  NORMAL  | EXT        rows1: the NORMAL/un-tilted image (baseline sedan)
  IMG     | CAL        rows2: the tilted/swapped image (variant / target)
For VP: NORMAL/EXT use the baseline image+pose; IMG/CAL use the variant image+pose.
For CTS: NORMAL/EXT use the sedan image+pose; IMG/CAL use the target image+pose.
On each image the CONSISTENT condition's pred (NORMAL/CAL) wraps objects; the MISMATCHED
condition's pred (EXT/IMG) shifts -- that shift is the model's image<->extrinsic gap.

GT (green) from the bg info's ego ann_infos; pred (red, score>=thr) from each condition's
sedan-model inference; both projected via K @ inv(sensor2ego) using the bg image's TRUE
pose (BEVDet's sensor2ego). Inputs in /tmp/qual_cond2/. Out: results/BEVDet/qual_conditions/.
"""
import os, pickle
import numpy as np
import cv2, mmcv
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes

QC = '/tmp/qual_cond2'
OUT = '/home/hanyan_arch/viewpoint/BEVFormer/results/BEVDet/qual_conditions/scene-0269-frame-0150_truepose'
ROOT = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDet'
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
PC = 51.2
SCORE_THR = 0.30
GT_COLOR = (0, 230, 0)
PRED_COLOR = (0, 0, 255)
_EDGES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
          (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))


def ego2img(cam):
    R = Quaternion(cam['sensor2ego_rotation']).rotation_matrix
    t = np.asarray(cam['sensor2ego_translation'], float)
    e2c = np.eye(4); e2c[:3, :3] = R.T; e2c[:3, 3] = -R.T @ t
    K = np.eye(4); K[:3, :3] = np.asarray(cam['cam_intrinsic'], float)
    return K @ e2c, e2c


def boxes(arr):
    return LiDARInstance3DBoxes(np.asarray(arr, np.float32).reshape(-1, 9), box_dim=9,
                                origin=(0.5, 0.5, 0.5))


def vis(b, e2c):
    n = len(b)
    if n == 0:
        return np.zeros(0, bool)
    R, t = e2c[:3, :3], e2c[:3, 3]
    c = b.gravity_center.numpy()
    inr = (np.abs(c[:, 0]) < PC) & (np.abs(c[:, 1]) < PC)
    ctr = (c @ R.T) + t
    corn = (b.corners.numpy().reshape(-1, 3) @ R.T) + t
    return inr & (ctr[:, 2] > 1) & (ctr[:, 2] < 80) & (corn[:, 2].reshape(n, 8) > 0.3).all(1)


def draw(img, b, l2i, color, th):
    if len(b) == 0:
        return img
    cor = b.corners.numpy()
    ph = np.concatenate([cor.reshape(-1, 3), np.ones((len(cor) * 8, 1))], 1)
    uv = (ph @ np.asarray(l2i).T)
    uv = (uv[:, :2] / np.clip(uv[:, 2:3], 1e-4, None)).reshape(-1, 8, 2)
    for i in range(len(cor)):
        p = uv[i].astype(np.int32)
        for a, b2 in _EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b2]), color, th, cv2.LINE_AA)
    return img


def render(bg_info, GT, PR, cam_name):
    cam = bg_info['cams'][cam_name]
    img = mmcv.imread(cam['data_path'] if os.path.isabs(cam['data_path'])
                      else os.path.join(ROOT, cam['data_path']))
    if img is None:
        return np.zeros((900, 1600, 3), np.uint8)
    l2i, e2c = ego2img(cam)
    img = draw(img, GT[vis(GT, e2c)], l2i, GT_COLOR, 3)
    img = draw(img, PR[vis(PR, e2c)], l2i, PRED_COLOR, 2)
    return img


def six_view(bg_info, GT, PR, banner):
    pn = {}
    for cam in CAMS:
        im = render(bg_info, GT, PR, cam)
        cv2.putText(im, cam, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        pn[cam] = im
    grid = np.vstack([np.hstack([pn[c] for c in row]) for row in GRID])
    cv2.putText(grid, banner, (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 255, 255), 3, cv2.LINE_AA)
    return grid


def front(bg_info, GT, PR, label):
    im = render(bg_info, GT, PR, 'CAM_FRONT')
    cv2.putText(im, label, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(im, f'GT={int(vis(GT, ego2img(bg_info["cams"]["CAM_FRONT"])[1]).sum())} '
                    f'pred>={SCORE_THR}={int(vis(PR, ego2img(bg_info["cams"]["CAM_FRONT"])[1]).sum())}',
                (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return im


def load_bg(name):
    return pickle.load(open(f'{QC}/{name}.pkl', 'rb'))['infos'][0]


def load_pred(name):
    pb = mmcv.load(f'{QC}/{name}_pred.pkl')[0]['pts_bbox']
    return pb['boxes_3d'][pb['scores_3d'].numpy() >= SCORE_THR]


def study_cell(tag, gt_src, cells):
    """cells: dict ptype-> (bg_name, pred_name). Renders 4 six-views + a 2x2 CAM_FRONT."""
    os.makedirs(OUT, exist_ok=True)
    GTcache = {}
    fronts = {}
    for ptype in ['NORMAL', 'EXT', 'IMG', 'CAL']:
        bg_name, pred_name = cells[ptype]
        bg = load_bg(bg_name)
        GT = GTcache.setdefault(bg_name, boxes(bg['ann_infos'][0]))
        PR = load_pred(pred_name)
        banner = f'{tag} {ptype}  scene-0269  GT(green) pred>{SCORE_THR}(red)  [{pred_name}]'
        grid = six_view(bg, GT, PR, banner)
        cv2.imwrite(f'{OUT}/{tag}_{ptype}_6view.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        fronts[ptype] = front(bg, GT, PR, f'{tag} {ptype}')
        print(f'[{tag} {ptype}] GT={len(GT)} pred={len(PR)} bg={bg_name} pred={pred_name}', flush=True)
    montage = np.vstack([np.hstack([fronts['NORMAL'], fronts['EXT']]),
                         np.hstack([fronts['IMG'], fronts['CAL']])])
    montage = cv2.resize(montage, (montage.shape[1] // 2, montage.shape[0] // 2))
    cv2.imwrite(f'{OUT}/{tag}_compare.jpg', montage, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'[{tag}] montage -> {tag}_compare.jpg')


def main():
    for ax in ['yaw', 'pitch', 'roll']:
        study_cell(f'VP-{ax}+20', 'sedan', {
            'NORMAL': ('VP_Normal', 'VP_Normal'),
            'EXT':    ('VP_Normal', f'VP_ER_{ax}'),
            'IMG':    (f'VP_CR_{ax}', f'VP_VR_{ax}'),
            'CAL':    (f'VP_CR_{ax}', f'VP_CR_{ax}'),
        })
    for tgt in ['bus', 'suv']:
        study_cell(f'CTS-{tgt}', tgt, {
            'NORMAL': (f'CTS-{tgt}_NORMAL', f'CTS-{tgt}_NORMAL'),
            'EXT':    (f'CTS-{tgt}_NORMAL', f'CTS-{tgt}_EXT'),
            'IMG':    (f'CTS-{tgt}_CAL', f'CTS-{tgt}_IMG'),
            'CAL':    (f'CTS-{tgt}_CAL', f'CTS-{tgt}_CAL'),
        })
    print('DONE ->', OUT)


if __name__ == '__main__':
    main()
