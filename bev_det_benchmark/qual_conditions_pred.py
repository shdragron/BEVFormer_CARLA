"""Condition-aware qual WITH BEVFormer predictions (GT green + pred red).

Extends qual_conditions.py: same coordinate-exact projection (lidar-frame boxes via
each condition's own sensor2lidar = the model's lidar2img), now overlaying the
BEVFormer-tiny SEDAN model's predictions per condition. Confirmed by the existing
qual_viz_bevformer.py: BEVFormer pred boxes_3d are LIDAR frame, projected with the
SAME lidar2img as gt_boxes -> no z-shift, GT/pred share one projection.

Studies:
  VP  : sedan model on sedan data, conditions NORMAL/EXT/IMG/CAL over axes roll/pitch/yaw
        (all-cam, |mag|=MAG). NORMAL is axis-independent (rendered once).
  CTS : sedan model on suv/bus data, conditions NORMAL/EXT/IMG/CAL (no axes).

Pred = BEVFormer-tiny sedan model, single-frame inference (prev_bev=None, like
qual_viz_bevformer), score>THR, in-pc-range, FOV-visible. Run from BEVFormer root in
the bevformer-b200 env on a GPU (fits in spare VRAM next to DETR3D training):
    CUDA_VISIBLE_DEVICES=0 AXES=roll,pitch,yaw MAG=20 THR=0.3 \
        python bev_det_benchmark/qual_conditions_pred.py
"""
import os, sys, copy, pickle
os.chdir('/home/hanyan_arch/viewpoint/BEVFormer')
sys.path.insert(0, '.')
sys.path.insert(0, 'bev_det_benchmark')
import numpy as np, cv2, torch, importlib
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import build_condition_pkls as B
import qual_conditions as Q
importlib.import_module('projects.mmdet3d_plugin')

# NOTE: build_condition_pkls.vr_image_path already applies the carla_VR frame-2x fix
# (geobev frame N == carla_VR frame 2N); do NOT patch here or it double-applies (->4x).

BEVF = '/home/hanyan_arch/viewpoint/BEVFormer'
DATA = f'{BEVF}/data/nuscenes'
OUT = f'{BEVF}/results/_qual_conditions'
TMP = '/tmp/qual_cond'
CFG = 'projects/configs/bevformer/bevformer_tiny_carla.py'
CKPT = 'work_dirs/bevformer_tiny_carla_sedan/latest.pth'
CAMS, GRID, PC = Q.CAMS, Q.GRID, Q.PC
GT_COLOR = (0, 230, 0)
PRED_COLOR = (0, 0, 255)
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
os.makedirs(TMP, exist_ok=True)


def load_full(p):
    d = pickle.load(open(p, 'rb'))
    return d['infos'], d.get('metadata', {'version': 'v1.0-carla_sedan_eval'})


def in_pc_range_pts(c):
    return (PC_RANGE[0] <= c[0] <= PC_RANGE[3] and PC_RANGE[1] <= c[1] <= PC_RANGE[4]
            and PC_RANGE[2] <= c[2] <= PC_RANGE[5])


class Infer:
    """Load the BEVFormer-tiny sedan model once; run single-frame inference on any
    condition info by writing a 1-frame pkl and building the test dataset."""
    def __init__(self):
        self.cfg = Config.fromfile(CFG)
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        load_checkpoint(self.model, CKPT, map_location='cpu')
        self.model.cuda().eval()

    def __call__(self, info, metadata):
        path = f'{TMP}/one.pkl'
        pickle.dump({'infos': [copy.deepcopy(info)], 'metadata': metadata}, open(path, 'wb'))
        cfg = copy.deepcopy(self.cfg)
        cfg.data.test.ann_file = path
        cfg.data.test.test_mode = True
        ds = build_dataset(cfg.data.test)
        data = scatter(collate([ds[0]], samples_per_gpu=1), [0])[0]
        with torch.no_grad():
            res = self.model(return_loss=False, rescale=True, **data)
        pb = res[0]['pts_bbox']
        return pb['boxes_3d'], pb['scores_3d'].numpy()


def draw(img, boxes, l2c, K, color, thick):
    if len(boxes) == 0:
        return img, 0
    l2i = np.eye(4); l2i[:3, :3] = K; l2i = l2i @ l2c
    m = Q.in_range_mask(boxes) & Q.visible_mask(boxes, l2c)
    if m.any():
        img = Q.draw_box3d_lidar(img, boxes[m], l2i, color, thick)
    return img, int(m.sum())


def render_cam(info, gt, pred, pmask, cam):
    c = info['cams'][cam]
    img = cv2.imread(Q.resolve_img(c['data_path']))
    if img is None:
        img = np.zeros((900, 1600, 3), np.uint8)
        cv2.putText(img, 'MISSING IMG', (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    l2c = Q.s2l_inv(c); K = np.asarray(c['cam_intrinsic'], float)
    img, ng = draw(img, gt, l2c, K, GT_COLOR, 3)
    img, npd = draw(img, pred[pmask] if len(pred) else pred, l2c, K, PRED_COLOR, 2)
    cv2.putText(img, f'{cam}  GT={ng} pred={npd}', (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def six_view(info, gt, pred, pmask, banner, outpath):
    panels = {cam: render_cam(info, gt, pred, pmask, cam) for cam in CAMS}
    grid = np.vstack([np.hstack([panels[c] for c in row]) for row in GRID])
    cv2.putText(grid, banner, (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(outpath, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return panels['CAM_FRONT']


def front(info, gt, pred, pmask, label):
    img = render_cam(info, gt, pred, pmask, 'CAM_FRONT')
    cv2.putText(img, label, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, 'GT', (16, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, GT_COLOR, 4, cv2.LINE_AA)
    cv2.putText(img, 'pred', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, PRED_COLOR, 4, cv2.LINE_AA)
    return img


def pred_mask(pred, scores, thr):
    if len(pred) == 0:
        return np.zeros(0, bool)
    c = pred.gravity_center.numpy()
    inr = np.array([in_pc_range_pts(x) for x in c], bool)
    return (scores > thr) & inr


def main():
    axes = os.environ.get('AXES', 'roll,pitch,yaw').split(',')
    mag = int(os.environ.get('MAG', '20'))
    thr = float(os.environ.get('THR', '0.3'))
    scene = os.environ.get('SCENE', '0267-0016')
    sc, fr = scene.split('-')
    key = (sc, fr)

    sedan, smeta = load_full(f'{DATA}/sedan_infos_val.pkl')
    suv, umeta = load_full(f'{DATA}/suv_infos_val.pkl')
    bus, bmeta = load_full(f'{DATA}/bus_infos_val.pkl')
    sidx = {B.info_key(i): i for i in sedan}
    uidx = {B.info_key(i): i for i in suv}
    bidx = {B.info_key(i): i for i in bus}
    sinfo = sidx[key]
    sgt = Q.boxes_from(sinfo['gt_boxes'], sinfo['valid_flag'])

    inf = Infer()
    tag = f'scene-{sc}-frame-{fr}'
    odir = f'{OUT}/{tag}_pred_thr{thr}'
    os.makedirs(odir, exist_ok=True)
    print(f'scene {tag}  axes={axes} mag=+{mag} thr={thr}', flush=True)

    # ---- VP: NORMAL once, then EXT/IMG/CAL per axis ----
    # CRITICAL: GT/pred are projected via the DISPLAYED IMAGE's TRUE geometry (so GT
    # always wraps the real objects); the model still runs on the CONDITION geometry.
    #   render geom = variant s2l if the shown image is the variant (IMG/CAL) else baseline.
    vp_fronts = {}     # (axis or 'NORMAL', cond) -> front panel
    ni = B.make_vp_infos([sinfo], 'Normal', axes[0], mag, 'all')[0]   # cond == render (baseline)
    pbox, psc = inf(ni, smeta)
    pm = pred_mask(pbox, psc, thr)
    six_view(ni, sgt, pbox, pm, f'VP NORMAL  {tag}   GT=green(fixed)  pred=red>{thr}', f'{odir}/VP_NORMAL_6view.jpg')
    vp_fronts[('NORMAL', 'NORMAL')] = front(ni, sgt, pbox, pm, 'VP NORMAL')
    print('  [VP NORMAL] done', flush=True)
    for axis in axes:
        for cond, disp, swap_img in [('ER', 'EXT', False), ('VR', 'IMG', True), ('CR', 'CAL', True)]:
            cond_info = B.make_vp_infos([sinfo], cond, axis, mag, 'all')[0]                       # MODEL input
            render_info = B.make_vp_infos([sinfo], 'CR' if swap_img else 'Normal', axis, mag, 'all')[0]  # DISPLAY (img+true geom)
            pbox, psc = inf(cond_info, smeta)
            pm = pred_mask(pbox, psc, thr)
            imgtag = 'img=tilted' if swap_img else 'img=normal'
            six_view(render_info, sgt, pbox, pm,
                     f'VP {disp} {axis}+{mag} all-cam ({imgtag})  {tag}   GT=green(fixed) pred=red>{thr}',
                     f'{odir}/VP_{disp}_{axis}+{mag}_6view.jpg')
            vp_fronts[(axis, disp)] = front(render_info, sgt, pbox, pm, f'VP {disp} {axis}+{mag} ({imgtag})')
            print(f'  [VP {disp} {axis}] done', flush=True)

    # ---- CTS: suv/bus NORMAL/EXT/IMG/CAL ----
    # render geom = sedan cams (img_src=s: NORMAL/EXT) or target cams (img_src=t: IMG/CAL).
    cts_fronts = {}
    for tgt, tidx, tmeta in (('suv', uidx, umeta), ('bus', bidx, bmeta)):
        tinfo = tidx[key]
        tgt_gt = Q.boxes_from(tinfo['gt_boxes'], tinfo['valid_flag'])
        for cond in ['NORMAL', 'EXT', 'IMG', 'CAL']:
            cond_info = Q.make_cts_info(cond, tinfo, sinfo)                # MODEL input
            img_src = B.CTS_CONDITIONS[cond][0]                            # 's'=sedan img, 't'=target img
            render_info = sinfo if img_src == 's' else tinfo              # DISPLAY (img+true geom)
            pbox, psc = inf(cond_info, tmeta)
            pm = pred_mask(pbox, psc, thr)
            imgtag = 'img=sedan' if img_src == 's' else f'img={tgt}'
            six_view(render_info, tgt_gt, pbox, pm,
                     f'CTS-{tgt} {cond} ({imgtag})  {tag}   GT=green(target,fixed) pred=red>{thr}',
                     f'{odir}/CTS-{tgt}_{cond}_6view.jpg')
            cts_fronts[(tgt, cond)] = front(render_info, tgt_gt, pbox, pm, f'CTS-{tgt} {cond} ({imgtag})')
            print(f'  [CTS-{tgt} {cond}] done', flush=True)

    # ---- montages (CAM_FRONT) ----
    # VP: rows = NORMAL/EXT/IMG/CAL, cols = axes (NORMAL spans, shown in col0)
    def stack(rows):
        m = np.vstack([np.hstack(r) for r in rows])
        return cv2.resize(m, (m.shape[1] // 2, m.shape[0] // 2))
    blank = np.zeros_like(vp_fronts[('NORMAL', 'NORMAL')])
    vp_rows = [[vp_fronts[('NORMAL', 'NORMAL')]] + [blank] * (len(axes) - 1)]
    for disp in ['EXT', 'IMG', 'CAL']:
        vp_rows.append([vp_fronts[(axis, disp)] for axis in axes])
    cv2.imwrite(f'{odir}/VP_compare_axes.jpg', stack(vp_rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cts_rows = [[cts_fronts[('suv', c)], cts_fronts[('bus', c)]] for c in ['NORMAL', 'EXT', 'IMG', 'CAL']]
    cv2.imwrite(f'{odir}/CTS_compare.jpg', stack(cts_rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'DONE -> {odir}', flush=True)


if __name__ == '__main__':
    main()
