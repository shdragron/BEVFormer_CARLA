"""Qualitative viz for BEVFormer-tiny CARLA: top-10 (distinct-scene) most-object
samples, 6 camera views, GT (green) vs prediction (red, score>0.3), per platform
(sedan/suv/bus), NORMAL condition (each platform's own model on its own data).
Saves results/BEVFormer/qualitative/<platform>/<scene-frame>.jpg.
"""
import os, sys, re
os.chdir('/home/hanyan_arch/viewpoint/BEVFormer'); sys.path.insert(0, '.')
import numpy as np, torch, cv2, importlib
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
importlib.import_module('projects.mmdet3d_plugin')

# Top-10 distinct scenes by IN-PC-RANGE vis>=2 object count (detectable objects;
# visibility-only ranking over-weighted far/out-of-grid GT that BEVFormer cannot detect).
TARGET_SF = ['scene-0245-frame-0108', 'scene-0247-frame-0142', 'scene-0241-frame-0066',
             'scene-0244-frame-0050', 'scene-0222-frame-0126', 'scene-0246-frame-0142',
             'scene-0267-frame-0026', 'scene-0266-frame-0040', 'scene-0230-frame-0000',
             'scene-0269-frame-0150']
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # BEV grid: only in-range objects are detectable


def in_pc_range(center):
    return (PC_RANGE[0] <= center[0] <= PC_RANGE[3] and
            PC_RANGE[1] <= center[1] <= PC_RANGE[4] and
            PC_RANGE[2] <= center[2] <= PC_RANGE[5])
PLATFORMS = [
    ('sedan', 'projects/configs/bevformer/bevformer_tiny_carla.py',     'work_dirs/bevformer_tiny_carla_sedan/latest.pth'),
    ('suv',   'projects/configs/bevformer/bevformer_tiny_carla_suv.py', 'work_dirs/bevformer_tiny_carla_suv/latest.pth'),
    ('bus',   'projects/configs/bevformer/bevformer_tiny_carla_bus.py', 'work_dirs/bevformer_tiny_carla_bus/latest.pth'),
]
DISPLAY = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
EDGES = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]
SCORE_THRS = [0.3, 0.5]   # render one figure per threshold (same inference)
GT_COLOR = (0, 255, 0)      # green (BGR)
PRED_COLOR = (0, 0, 255)    # red (BGR)


def draw_box(img, corners_3d, l2i, color, thick=2):
    pts = np.concatenate([corners_3d, np.ones((8, 1))], 1)         # (8,4)
    uvw = (l2i @ pts.T).T                                          # (8,4)
    depth = uvw[:, 2]
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-3, None)
    if (depth > 0.1).sum() < 4:
        return 0
    drew = 0
    for a, b in EDGES:
        if depth[a] > 0.1 and depth[b] > 0.1:
            pa = (int(round(uv[a, 0])), int(round(uv[a, 1])))
            pb = (int(round(uv[b, 0])), int(round(uv[b, 1])))
            cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)
            drew += 1
    return drew


def run_platform(platform, cfgpath, ckpt):
    print(f'=== {platform} ===', flush=True)
    cfg = Config.fromfile(cfgpath)
    ds = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt, map_location='cpu')
    model.cuda().eval()
    sf2idx = {}
    for k, info in enumerate(ds.data_infos):
        m = re.search(r'scene-\d+-frame-\d+', info['cams']['CAM_FRONT']['data_path'])
        if m:
            sf2idx[m.group(0)] = k
    outdir = f'results/BEVFormer/qualitative/{platform}'
    os.makedirs(outdir, exist_ok=True)
    for sf in TARGET_SF:
        idx = sf2idx.get(sf)
        if idx is None:
            print(f'  {sf}: NOT FOUND in {platform}', flush=True); continue
        data = ds[idx]
        data = scatter(collate([data], samples_per_gpu=1), [0])[0]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        pb = result[0]['pts_bbox']
        pbox = pb['boxes_3d']
        pscore = pb['scores_3d'].numpy()
        pcent = pbox.gravity_center.numpy()
        inr = np.array([in_pc_range(c) for c in pcent], dtype=bool) if len(pcent) else np.zeros(0, bool)
        info = ds.data_infos[idx]
        valid = np.asarray(info['valid_flag']).astype(bool)
        gtb = np.asarray(info['gt_boxes'])[valid]
        if len(gtb):                                   # keep only in-pc-range (detectable) GT
            gtb = gtb[np.array([in_pc_range(b[:3]) for b in gtb], dtype=bool)]
        gcorners = (LiDARInstance3DBoxes(torch.tensor(gtb, dtype=torch.float32),
                    box_dim=gtb.shape[-1], origin=(0.5, 0.5, 0.5)).corners.numpy()
                    if len(gtb) else np.zeros((0, 8, 3)))
        di = ds.get_data_info(idx)
        l2i = np.asarray(di['lidar2img'])
        imgpaths = di['img_filename']
        cams = list(info['cams'].keys())
        # base full-res tiles with GT (green) drawn once, reused for every threshold
        gt_imgs = {}
        for ci, cam in enumerate(cams):
            img = cv2.imread(imgpaths[ci])
            if img is None:
                img = np.zeros((900, 1600, 3), np.uint8)
            for gc in gcorners:
                draw_box(img, gc, l2i[ci], GT_COLOR, 2)
            gt_imgs[cam] = img
        for thr in SCORE_THRS:
            keep = (pscore > thr) & inr if len(inr) else np.zeros(0, bool)
            pcorners = pbox[keep].corners.numpy() if keep.sum() else np.zeros((0, 8, 3))
            tiles = {}
            for cam in DISPLAY:
                ci = cams.index(cam)
                img = gt_imgs[cam].copy()
                for pc in pcorners:
                    draw_box(img, pc, l2i[ci], PRED_COLOR, 2)
                tiles[cam] = cv2.resize(img, (800, 450))
            grid = np.vstack([np.hstack([tiles[c] for c in DISPLAY[:3]]),
                              np.hstack([tiles[c] for c in DISPLAY[3:]])])
            cv2.putText(grid, f'{platform}  {sf}  [pc_range +-51.2m]  GT(green)={len(gcorners)}  Pred(red,>{thr})={len(pcorners)}',
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            out = f'{outdir}/{sf}_thr{thr}.jpg'
            cv2.imwrite(out, grid)
            print(f'  {sf} thr{thr}: GT={len(gcorners)} pred={len(pcorners)} -> {out}', flush=True)
    del model
    torch.cuda.empty_cache()


for p in PLATFORMS:
    try:
        run_platform(*p)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'PLATFORM {p[0]} FAILED: {e!r}', flush=True)
print('DONE', flush=True)
