"""Unified 6-CAMERA qual-grid panel renderer for BEVFormer-schema detectors
(DFA3D, CAPE). Renders a 2x3 six-camera grid per row condition on a fixed scene,
GT(green)+pred(red), for the RoboGeo vertical 2x2-per-condition figure.

Rows (env ROW selects one, or 'all'):
  pitch12_img : VP IMG, per-camera FRONT, pitch +12  (only FRONT img tilted; ext stale)
  yaw8_img    : VP IMG, all-camera,        yaw  +8
  suv_cal     : CTS SUV CAL  (suv img + suv ext);  GT = suv target vis>=2
  bus_cal     : CTS BUS CAL  (bus img + bus ext);  GT = bus target vis>=2
Each camera drawn with ITS OWN display geometry (variant for perturbed cams,
baseline elsewhere; target for CTS). Output: clean 4800x1800 six-cam PNG.
"""
import os, sys, copy, pickle
import numpy as np, cv2, torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.insert(0, os.getcwd())
import importlib; importlib.import_module('projects.mmdet3d_plugin')

BENCH = '/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark'; sys.path.insert(0, BENCH)
import build_condition_pkls as B
import qual_conditions as Q

MODEL = os.environ['MODEL']
CFG = os.environ['CFG']; CKPT = os.environ['CKPT']
ROW = os.environ.get('ROW', 'all')
THR = float(os.environ.get('THR', '0.3'))
SUF = os.environ.get('SUF', '')                         # output filename suffix
# VP display geometry: 'aligned' = variant img + variant ext (boxes registered);
# 'input' = variant img + STALE baseline ext = exactly what the model ingests
# (boxes projected with the stale extrinsic -> visible image-geometry mismatch).
VP_DISPLAY = os.environ.get('VP_DISPLAY', 'aligned')
DATA = '/home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes'
OUT = os.environ.get('OUT', '/home/hanyan_arch/viewpoint/BEVFormer/results/qual_grid')
os.makedirs(OUT, exist_ok=True)
# per-row scene: (a)/(b) distinct, (c)/(d) share one CTS scene
ROW_SCENE = {'pitch12_img': os.environ.get('SCENE_A', '0230-0032'),
             'yaw8_img':    os.environ.get('SCENE_B', '0256-0120'),
             'suv_cal':     os.environ.get('SCENE_C', '0267-0016'),
             'bus_cal':     os.environ.get('SCENE_C', '0267-0016')}
GT_GREEN = (0, 230, 0); PRED_RED = (0, 0, 255); SVER = 'v1.0-carla_sedan_eval'


def set_cams(info, cams, scene, frame, variant, swap_img, swap_ext):
    for cam in cams:
        c = info['cams'][cam]
        if swap_img:
            c['data_path'] = B.vr_image_path(scene, frame, cam, variant)
        if swap_ext:
            rot, trans = B.variant_extrinsic(scene, cam, variant)
            c['sensor2lidar_rotation'] = rot; c['sensor2lidar_translation'] = trans
    return info


def vr_base(sinfo, cams, scene, frame):
    return set_cams(copy.deepcopy(sinfo), cams, scene, frame, 'yaw0pitch0roll0', True, True)


class Infer:
    def __init__(self):
        self.cfg = Config.fromfile(CFG)
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        load_checkpoint(self.model, CKPT, map_location='cpu')
        self.model.cuda().eval()

    def __call__(self, info, version):
        pickle.dump({'infos': [copy.deepcopy(info)], 'metadata': {'version': version}},
                    open('/tmp/qgrid_one.pkl', 'wb'))
        cfg = copy.deepcopy(self.cfg)
        cfg.data.test.ann_file = '/tmp/qgrid_one.pkl'; cfg.data.test.test_mode = True
        ds = build_dataset(cfg.data.test)
        data = scatter(collate([ds[0]], samples_per_gpu=1), [0])[0]
        with torch.no_grad():
            res = self.model(return_loss=False, rescale=True, **data)
        pb = res[0]['pts_bbox']
        return pb['boxes_3d'], pb['scores_3d'].numpy()


def pred_keep(pred, scores):
    return (scores > THR) & Q.in_range_mask(pred) if len(pred) else np.zeros(0, bool)


def six_cam(disp_info, gt, pred, pk):
    """2x3 six-camera grid; GT green + pred red via EACH cam's own geometry."""
    panels = {}
    ng = npd = 0
    for cam in B.CAM_NAMES:
        c = disp_info['cams'][cam]
        l2c = Q.s2l_inv(c); K = np.asarray(c['cam_intrinsic'], float)
        l2i = np.eye(4); l2i[:3, :3] = K; l2i = l2i @ l2c
        img = cv2.imread(Q.resolve_img(c['data_path']))
        if img is None:
            img = np.zeros((900, 1600, 3), np.uint8)
        if gt is not None:
            mg = Q.in_range_mask(gt) & Q.visible_mask(gt, l2c)
            img = Q.draw_box3d_lidar(img, gt[mg], l2i, GT_GREEN, 5); ng += int(mg.sum())
        if pred is not None and len(pred):
            pm = pk & Q.visible_mask(pred, l2c)
            img = Q.draw_box3d_lidar(img, pred[pm], l2i, PRED_RED, 4); npd += int(pm.sum())
        panels[cam] = img
    grid = np.vstack([np.hstack([panels[c] for c in row]) for row in Q.GRID])
    return grid, ng, npd


def build_row(row, key, sidx, tdata):
    sinfo = sidx[key]; sc, fr = B.info_key(sinfo)
    if row in ('pitch12_img', 'yaw8_img'):
        sgt = Q.boxes_from(sinfo['gt_boxes'], sinfo['valid_flag'])
        axis, mag = ('pitch', 12) if row == 'pitch12_img' else ('yaw', 8)
        cams = ['CAM_FRONT'] if row == 'pitch12_img' else B.CAM_NAMES
        v = B.variant_key(axis, mag)
        mi = set_cams(vr_base(sinfo, B.CAM_NAMES, sc, fr), cams, sc, fr, v, True, False)
        di = set_cams(vr_base(sinfo, B.CAM_NAMES, sc, fr), cams, sc, fr, v, True, True)
        return mi, di, sgt, SVER
    tgt = 'suv' if row == 'suv_cal' else 'bus'
    tinfo = tdata[tgt]['idx'][key]
    tgt_gt = Q.boxes_from(tinfo['gt_boxes'], tinfo['valid_flag'])
    ci = Q.make_cts_info('CAL', tinfo, sinfo)
    return ci, ci, tgt_gt, tdata[tgt]['ver']


def main():
    sidx = {B.info_key(i): i for i in pickle.load(open(f'{DATA}/sedan_infos_val.pkl', 'rb'))['infos']}
    tdata = {}
    for tgt, ver in (('suv', 'v1.0-carla_suv_eval'), ('bus', 'v1.0-carla_bus_eval')):
        d = pickle.load(open(f'{DATA}/{tgt}_infos_val.pkl', 'rb'))['infos']
        tdata[tgt] = {'idx': {B.info_key(i): i for i in d}, 'ver': ver}

    rows = ['pitch12_img', 'yaw8_img', 'suv_cal', 'bus_cal'] if ROW == 'all' else [ROW]
    inf = Infer()
    for row in rows:
        key = tuple(ROW_SCENE[row].split('-'))
        assert key in sidx, f'{ROW_SCENE[row]} not in sedan val'
        mi, di, gt, ver = build_row(row, key, sidx, tdata)
        pb, ps = inf(mi, ver); pk = pred_keep(pb, ps)
        # VP rows: 'input' draws on mi (variant img + stale ext); else di (registered).
        vp = row in ('pitch12_img', 'yaw8_img')
        disp = mi if (VP_DISPLAY == 'input' and vp) else di
        grid, ng, npd = six_cam(disp, gt, pb, pk)
        cv2.imwrite(f'{OUT}/{MODEL}_{row}_6cam{SUF}.png', grid)
        gg, _, _ = six_cam(disp, gt, None, None)
        cv2.imwrite(f'{OUT}/GT_{row}_6cam{SUF}.png', gg)
        print(f'[{MODEL}] {row:12s} GT={ng} pred={npd} -> {MODEL}_{row}_6cam{SUF}.png', flush=True)


if __name__ == '__main__':
    main()
