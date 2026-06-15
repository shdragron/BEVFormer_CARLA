"""Smoke-test the camera-dim fix: confirm img reaches extract_feat as
(bs, 6, 3, 224, 480) and a fresh forward_train produces a loss without
collapsing to a single camera."""
import warnings; warnings.filterwarnings('ignore')
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
import importlib; importlib.import_module('projects.mmdet3d_plugin')

cfg = Config.fromfile('projects/configs/bevformer/bevformer_seg_r50_carla.py')
ds = build_dataset(cfg.data.train)
m = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda().train()

# wrap extract_feat to print the img shape it actually receives
orig = m.extract_feat
def spy(img=None, img_metas=None, **kw):
    print('  extract_feat sees img:', tuple(img.shape), 'dim', img.dim())
    return orig(img=img, img_metas=img_metas, **kw)
m.extract_feat = spy

b = scatter(collate([ds[0], ds[1]], samples_per_gpu=2), [0])[0]
print('dataloader img:', tuple(b['img'].data.shape) if hasattr(b['img'], 'data') else tuple(b['img'].shape))
img = b['img'].data if hasattr(b['img'], 'data') else b['img']
gt = b['gt_seg'].data if hasattr(b['gt_seg'], 'data') else b['gt_seg']
imetas = b['img_metas'].data if hasattr(b['img_metas'], 'data') else b['img_metas']
out = m.forward_train(img_metas=imetas, img=img, gt_seg=gt)
print('losses:', {k: float(v) for k, v in out.items()})
print('OK')
