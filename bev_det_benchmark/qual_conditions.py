"""Condition-aware qualitative viz: ONE scene, VP {NORMAL/EXT/IMG/CAL} and
CTS-{suv,bus} {NORMAL/EXT/IMG/CAL}, GT boxes projected the COORDINATE-EXACT way.

Coordinate recipe (verified, verify_qual_coords.py):
  Raw pkl gt_boxes are in the LIDAR frame (ground at z=-1.8). The model consumes
  lidar2img = K @ inv(sensor2lidar) (verify_coords.py). So we draw lidar-frame
  gt_boxes via inv(sensor2lidar) using EACH CONDITION'S OWN sensor2lidar (the field
  the condition pkls actually perturb). This matches the model's projection to
  machine precision and needs no sensor2ego (whose VP field is left stale).

VP conditions (make_vp_infos): NORMAL(F,F) EXT/ER(F,T) IMG/VR(T,F) CAL/CR(T,T),
axis+signed-mag, protocol 'all'. Image = sedan baseline (NORMAL/EXT) or carla_VR
variant (IMG/CAL); extrinsic = sedan (NORMAL/IMG) or perturbed variant (EXT/CAL).
CTS conditions (make_cts logic): base = TARGET pkl; swap image/extrinsic toward
sedan per CTS_CONDITIONS. GT = target gt_boxes + target valid_flag.

Output: results/_qual_conditions/<scene>/...  (local; GT green; per-(study,cond)
6-view grid + a CAM_FRONT comparison montage).  CPU only (no model).
"""
import os, sys, copy, pickle
import numpy as np
import cv2, mmcv
from mmcv import Config
from mmdet3d.core.bbox import LiDARInstance3DBoxes

# 12 edges of a mmdet3d corner box (same connectivity as plot_rect3d_on_img)
_EDGES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
          (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))


def draw_box3d_lidar(img, boxes, l2i, color, thickness=3):
    """Project lidar-frame box corners via l2i (=K@inv(s2l)) and draw 12 edges.
    Self-contained (no mmdet3d viz / np.int dependency)."""
    if len(boxes) == 0:
        return img
    corners = boxes.corners.numpy()                       # (N,8,3) lidar
    N = corners.shape[0]
    ph = np.concatenate([corners.reshape(-1, 3), np.ones((N * 8, 1))], 1)
    uvw = ph @ np.asarray(l2i).T                           # (N*8,4)
    w = np.clip(uvw[:, 2:3], 1e-4, None)
    uv = (uvw[:, :2] / w).reshape(N, 8, 2)
    for i in range(N):
        p = uv[i].astype(np.int32)
        for a, b in _EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), color, thickness, cv2.LINE_AA)
    return img

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_condition_pkls as B
# NOTE: build_condition_pkls.vr_image_path already applies the carla_VR frame-2x fix
# (geobev frame N == carla_VR frame 2N); do NOT patch here or it double-applies (->4x).

BEVF = '/home/hanyan_arch/viewpoint/BEVFormer'
DATA = f'{BEVF}/data/nuscenes'
OUT = f'{BEVF}/results/_qual_conditions'
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
PC = 51.2
GT_COLOR = (0, 230, 0)
VEH_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
VP_LABEL = {'Normal': 'NORMAL', 'ER': 'EXT', 'VR': 'IMG', 'CR': 'CAL'}   # internal->display
CTS_ORDER = ['NORMAL', 'EXT', 'IMG', 'CAL']


def load(p):
    return pickle.load(open(p, 'rb'))['infos']


def s2l_inv(cam):
    """lidar->cam 4x4 = inv(sensor2lidar). Rotation stored as 3x3 in the pkl."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(cam['sensor2lidar_rotation'], float)
    T[:3, 3] = np.asarray(cam['sensor2lidar_translation'], float)
    return np.linalg.inv(T)


def resolve_img(data_path):
    return data_path if os.path.isabs(data_path) else os.path.join(BEVF, data_path)


def boxes_from(gt_boxes, valid_flag):
    gtb = np.asarray(gt_boxes, np.float32).reshape(-1, 7)
    vf = np.asarray(valid_flag, bool)
    gtb = gtb[vf]                                          # vis>=2 (matches eval GT)
    return LiDARInstance3DBoxes(gtb, box_dim=7, origin=(0.5, 0.5, 0.5))


def in_range_mask(boxes):
    if len(boxes) == 0:
        return np.zeros(0, bool)
    c = boxes.gravity_center.numpy()
    return (np.abs(c[:, 0]) < PC) & (np.abs(c[:, 1]) < PC)


def visible_mask(boxes, l2c, W=1600, H=900):
    n = len(boxes)
    if n == 0:
        return np.zeros(0, bool)
    R, t = l2c[:3, :3], l2c[:3, 3]
    K = None  # filled by caller via project; here use center depth + corner front
    ctr = (boxes.gravity_center.numpy() @ R.T) + t
    z = ctr[:, 2]
    corn = (boxes.corners.numpy().reshape(-1, 3) @ R.T) + t
    corn_ok = (corn[:, 2].reshape(n, 8) > 0.3).all(1)
    return (z > 1.0) & (z < 80) & corn_ok


def render_cam(img, boxes, cam):
    K = np.asarray(cam['cam_intrinsic'], float)
    l2c = s2l_inv(cam)
    l2i = np.eye(4); l2i[:3, :3] = K; l2i = l2i @ l2c
    m = in_range_mask(boxes) & visible_mask(boxes, l2c)
    if m.any():
        img = draw_box3d_lidar(img, boxes[m], l2i, GT_COLOR, thickness=3)
    return img, int(m.sum())


def six_view(info, boxes, banner):
    panels = {}
    for cam in CAMS:
        c = info['cams'][cam]
        img = mmcv.imread(resolve_img(c['data_path']))
        if img is None:
            img = np.zeros((900, 1600, 3), np.uint8)
            cv2.putText(img, 'MISSING IMG', (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        img, nv = render_cam(img, boxes, c)
        cv2.putText(img, f'{cam} (vis GT={nv})', (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        panels[cam] = img
    grid = np.vstack([np.hstack([panels[c] for c in row]) for row in GRID])
    cv2.putText(grid, banner, (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    return grid, panels


def front_panel(info, boxes, label):
    c = info['cams']['CAM_FRONT']
    img = mmcv.imread(resolve_img(c['data_path']))
    if img is None:
        img = np.zeros((900, 1600, 3), np.uint8)
    img, nv = render_cam(img, boxes, c)
    cv2.putText(img, label, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, f'in-range vis GT drawn={nv}', (16, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def pick_scene(sidx, uidx, bidx, meta):
    """object-rich in-range scene present in sedan+suv+bus+VRmeta; prefer vehicles."""
    best = None
    for k, info in sidx.items():
        sc, fr = k
        if k not in uidx or k not in bidx or f'scene_{sc}' not in meta['scenes']:
            continue
        bx = boxes_from(info['gt_boxes'], info['valid_flag'])
        nm = np.asarray(info['gt_names'])[np.asarray(info['valid_flag'], bool)]
        m = in_range_mask(bx)
        nveh = int(sum(m[i] and nm[i] in VEH_CLASSES for i in range(len(nm))))
        ntot = int(m.sum())
        score = nveh * 10 + ntot                          # weight vehicles for visual clarity
        if best is None or score > best[0]:
            best = (score, sc, fr, nveh, ntot)
    return best


def make_cts_info(condition, tinfo, sinfo):
    img_src, ext_src = B.CTS_CONDITIONS[condition]
    new = copy.deepcopy(tinfo)
    for cam_name, cam in new['cams'].items():
        scam = sinfo['cams'][cam_name]
        if img_src == 's':
            cam['data_path'] = scam['data_path']
        if ext_src == 's':
            B._copy_extrinsic(cam, scam)
    return new


def main():
    axis = os.environ.get('AXIS', 'pitch')
    mag = int(os.environ.get('MAG', '20'))
    sedan = load(f'{DATA}/sedan_infos_val.pkl')
    suv = load(f'{DATA}/suv_infos_val.pkl')
    bus = load(f'{DATA}/bus_infos_val.pkl')
    sidx = {B.info_key(i): i for i in sedan}
    uidx = {B.info_key(i): i for i in suv}
    bidx = {B.info_key(i): i for i in bus}
    meta = B.load_vr_metadata()

    forced = os.environ.get('SCENE')
    if forced:
        sc, fr = forced.split('-')
        key = (sc, fr)
        sinfo = sidx[key]
        bx = boxes_from(sinfo['gt_boxes'], sinfo['valid_flag'])
        print(f"forced scene-{sc}-frame-{fr} in-range vis GT={int(in_range_mask(bx).sum())}")
    else:
        score, sc, fr, nveh, ntot = pick_scene(sidx, uidx, bidx, meta)
        key = (sc, fr)
        print(f"picked scene-{sc}-frame-{fr}: in-range vehicles={nveh} total in-range vis GT={ntot}")

    tag = f'scene-{sc}-frame-{fr}'
    odir = f'{OUT}/{tag}_{axis}{mag:+d}'
    os.makedirs(odir, exist_ok=True)
    sinfo = sidx[key]
    sgt = boxes_from(sinfo['gt_boxes'], sinfo['valid_flag'])

    # ---- VP study (sedan GT) ----
    # Render GT on the DISPLAYED IMAGE's true geometry so GT always wraps objects:
    # variant geom if the shown image is the variant (VR/CR), baseline otherwise (Normal/ER).
    vp_fronts = {}
    for cond in ['Normal', 'ER', 'VR', 'CR']:
        swap_img = cond in ('VR', 'CR')
        ci = B.make_vp_infos([sinfo], 'CR' if swap_img else 'Normal', axis, mag, 'all')[0]
        disp = VP_LABEL[cond]
        imgtag = 'img=tilted' if swap_img else 'img=normal'
        banner = f'VP {disp}  {tag}  {axis}{mag:+d} all-cam ({imgtag})   GT(green, vis>=2, in-range, fixed)'
        grid, _ = six_view(ci, sgt, banner)
        cv2.imwrite(f'{odir}/VP_{disp}_6view.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vp_fronts[disp] = front_panel(ci, sgt, f'VP {disp}  ({axis}{mag:+d}, {imgtag})')
        print(f'  [VP {disp}] -> VP_{disp}_6view.jpg')

    # ---- CTS study (target GT) ----
    cts_fronts = {'suv': {}, 'bus': {}}
    for tgt, tidx in (('suv', uidx), ('bus', bidx)):
        tinfo = tidx[key]
        tgt_gt = boxes_from(tinfo['gt_boxes'], tinfo['valid_flag'])
        for cond in CTS_ORDER:
            img_src = B.CTS_CONDITIONS[cond][0]                 # 's'=sedan img, 't'=target img
            ci = sinfo if img_src == 's' else tinfo            # render on the displayed image's true geom
            imgtag = 'img=sedan' if img_src == 's' else f'img={tgt}'
            banner = f'CTS-{tgt} {cond} ({imgtag})  {tag}   GT(green, target vis>=2, in-range, fixed)'
            grid, _ = six_view(ci, tgt_gt, banner)
            cv2.imwrite(f'{odir}/CTS-{tgt}_{cond}_6view.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cts_fronts[tgt][cond] = front_panel(ci, tgt_gt, f'CTS-{tgt} {cond}')
            print(f'  [CTS-{tgt} {cond}] -> CTS-{tgt}_{cond}_6view.jpg')

    # ---- headline CAM_FRONT montage: rows=conditions, cols=[VP, CTS-suv, CTS-bus] ----
    rows = []
    for disp in ['NORMAL', 'EXT', 'IMG', 'CAL']:
        row = np.hstack([vp_fronts[disp], cts_fronts['suv'][disp], cts_fronts['bus'][disp]])
        rows.append(row)
    montage = np.vstack(rows)
    montage = cv2.resize(montage, (montage.shape[1] // 2, montage.shape[0] // 2))
    cv2.imwrite(f'{odir}/compare_CAM_FRONT.jpg', montage, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'  [montage] -> compare_CAM_FRONT.jpg  ({montage.shape[1]}x{montage.shape[0]})')
    print(f'DONE -> {odir}')


if __name__ == '__main__':
    main()
