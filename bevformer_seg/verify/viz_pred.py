"""Diagnose low IoU: pred vs GT BEV overlay on a few val frames + prob histogram."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
import importlib; importlib.import_module('projects.mmdet3d_plugin')

cfg = Config.fromfile('projects/configs/bevformer/bevformer_seg_r50_carla.py')
ds = build_dataset(cfg.data.val)
m = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(m, 'work_dirs/bevformer_seg_r50_carla_sedan/epoch_24.pth', map_location='cpu')
m = m.cuda().eval()

# pick frames with vehicles
idxs = [i for i in range(0, 3792, 197)][:5]
fig, axes = plt.subplots(3, len(idxs), figsize=(4*len(idxs), 11))
allp = []
for c, i in enumerate(idxs):
    b = scatter(collate([ds[i]], samples_per_gpu=1), [0])[0]
    with torch.no_grad():
        prob = m.forward_test(img_metas=b['img_metas'], img=b['img'])[0]  # (200,200)
    scene, fi, fr = ds.frames[i]
    gt = ds._veh_mask(scene, fr['bev'])
    allp.append(prob[gt > 0])
    axes[0, c].imshow(gt, cmap='gray'); axes[0, c].set_title(f'GT {scene} f{fi} ({gt.sum()}px)', fontsize=8)
    axes[1, c].imshow(prob, cmap='turbo', vmin=0, vmax=1); axes[1, c].set_title(f'pred prob max={prob.max():.2f}', fontsize=8)
    axes[2, c].imshow((prob >= 0.5), cmap='gray'); axes[2, c].set_title(f'pred>=0.5 ({(prob>=0.5).sum()}px)', fontsize=8)
    for r in range(3): axes[r, c].axis('off')
fig.tight_layout(); fig.savefig('bevformer_seg/out/viz_pred.png', dpi=90)
ph = np.concatenate([p for p in allp if len(p)]) if allp else np.array([0])
print(f'pred prob AT GT-vehicle pixels: mean={ph.mean():.3f} p90={np.percentile(ph,90):.3f} max={ph.max():.3f} frac>0.5={np.mean(ph>0.5):.3f}')
print('wrote bevformer_seg/out/viz_pred.png')
