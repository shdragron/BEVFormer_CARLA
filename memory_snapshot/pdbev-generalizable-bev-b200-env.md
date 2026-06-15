---
name: pdbev-generalizable-bev-b200-env
description: "Generalizable-BEV (PD-BEV, AAAI'25) env on B200 = clone bevdet-b200 + copied setup.py + mmseg0.30. Use `pdbev-b200`. User chose this OVER CoIn3D."
metadata: 
  node_type: memory
  type: project
  originSessionId: 09f0506b-7473-403c-8e74-a3ace5c5a83b
---

**Generalizable-BEV** (`/home/hanyan_arch/viewpoint/BEVFormer/Generalizable-BEV`, upstream
`EnVision-Research/Generalizable-BEV`) = **PD-BEV** "Towards Generalizable Multi-Camera 3D
Object Detection via Perspective Debiasing" (AAAI 2025). A **plug-and-play domain-generalization
+ UDA training scheme on BEVDepth** — "does not change the model infrastructure, only used for
BEVDepth evaluation". Source→target DG across nuScenes / Lyft / DeepAccident (DeepAccident is
CARLA-simulated). The user picked this OVER [[coin3d-b200-env]] (CoIn3D was blocked: missing
datasets layer + needs a gaussian/NVS preprocess; PD-BEV has neither — directly on-theme for the
CARLA viewpoint/cross-platform robustness benchmark).

**Run it with conda env `pdbev-b200`.** README says "prepare env refer to BEVDet" → same stack as
the other b200 envs, so built by **cloning `bevdet-b200`** (torch 2.11.0+cu128, sm_100, mmcv-full
1.7.1, mmdet 2.28.2, mmdet3d 1.0.0rc4). Recipe that worked:
- `conda create --clone bevdet-b200 -n pdbev-b200`.
- **The repo ships WITHOUT setup.py** (genuinely absent, not gitignored; only a stale py3.8 `build/`
  exists). Its `mmdet3d/ops/` is IDENTICAL to CoIn3D/BEVDet's, so just
  `cp CoIn3D/BEVDet/setup.py Generalizable-BEV/setup.py` and it builds.
- `pip install mmsegmentation==0.30.0` (clone had 0.14.1; its mmdet3d asserts mmseg ≥0.20.0 — same
  as CoIn3D; leaves mmcv-full 1.7.1 untouched).
- patch `mmdet3d/__init__.py`: `mmcv_maximum_version '1.7.0'→'1.7.1'` (same B200 patch as CoIn3D).
- `export CUDA_HOME=$CONDA_PREFIX CC=gcc-13 CXX=g++-13 TORCH_CUDA_ARCH_LIST=10.0` then
  `pip install -e . --no-deps --no-build-isolation` (**--no-deps required**: requirements pin
  `numba==0.53.0`, no py3.10 wheel; clone already has working deps. Builds bev_pool_v2 py3.10/sm_100).
- All other runtime deps (lyft_dataset_sdk, plyfile, trimesh, scikit-image, nuscenes-devkit, numba)
  already present in the clone.
Verified: B200 (10,0), `mmdet3d`/`mmdet3d.datasets` (incl. `CarlaDataset`, `DeepAccidentDataset`),
`bev_pool_v2_ext`, B200 matmul all OK.

**DEPTH MECHANISM + the suv/bus from-scratch collapse (hard-won, 2026-06-12).** PD-BEV's
depth loss GT is NOT lidar/DPT — it is `kwargs['ann_maps_2d']` ch2 (pcbev.py:142, upstream):
**virtual depth = real × 450/fx' at GT box-center pixels only (~5% of depthnet pixels)**;
the dense `gt_depth` from PointToMultiViewDepth_UDA/CarlaDPT is vestigial for the loss (so
swapping lidar→DPT changes nothing). The lift converts virtual→real per camera (k=450/fx',
get_lidar_coor). CONSEQUENCE: 95% of depth pixels are anchored only by detection gradients
through the LSS lift → on TALL platforms (suv cams 2.35m; bus 2.87-4.08m +20° pitch) a
from-scratch run falls into a **bin-1 delta attractor** (probe: softmax entropy 0.64 vs 3.4
healthy, argmax bin 1/112, logits 8914 vs 39) → all features splat into an ego-ring → boxes
pile at ego/grid-edges → real eval NDS 0.0000 even though task losses "converge" (low-conf
floor ~5.5; regression is GT-masked). Failed 3×: clip5/warmup200, clip35/warmup1000,
DPT-swap. Data/coords/eval all verified innocent (sedan-final model on suv data is healthy,
entropy 3.63; suv lidar/GT projections clean; all 3 vehicles share IDENTICAL trajectories so
the shared per-scene LIDAR npz is valid for every vehicle). **FIX (verified): restore the
AUTHORS-NATIVE depth regime — input 384×704 (not 256, which crops top 35%) + depth grid
[1.0,100.0,1.0] (not BEVDepth's [2,58,0.5], which truncated 35-40% of virtual-depth targets
since virtual ≈ 1.07×real reaches ~125)**: suv then trains from scratch (ep2 NDS 0.248,
sedan-like losses dep~13/hm~1.9). Warm-start from sedan ep24 also escapes the attractor
(instantly healthy) if ever needed. Sedan itself trained fine even with [2,58]/256 (NDS
0.5170/mAP 0.5308) — its lower camera (1.60m) keeps early lift gradients inside the margin.
Eval configs must match the trained arch (D=100bins/384×704) or the ckpt won't load.

**CARLA-fit caveat (not done yet):** the built-in `CarlaDataset` (mmdet3d/datasets/CARLA_dataset.py)
is **mislabeled — it's actually for the SHIFT dataset**: hardcoded author path
`/mnt/cfs/.../SHIFT/val_oder.pkl` and rotation handled as a **scalar pitch in degrees**
(`pitch2matrix_shift_box`), NOT the user's geobev quaternion convention. Its CLASSES =
(car,truck,bus,motorcycle,bicycle,pedestrian) **= the user's CARLA 6-class** ✓. So to run on the
geobev data: either adapt this dataset class to the geobev pkl schema (quaternion sensor2ego, real
paths) like the existing `CarlaNuScenesDataset` in [[carla-qual-viz-ego-frame]], or convert geobev
pkls to the repo's "uniform format" (see tools/Deepaccident_converter.py) and use
DeepAccidentDataset. PD-BEV being plug-and-play on BEVDepth means it can ride the user's existing
BEVDepth-CARLA setup ([[bevdepth-carla-sedan-result]]). Training estimate anchor: existing BEVDet
sedan = ~7-9h (24ep, 2×B200, batch128); PD-BEV-on-BEVDepth similar order.
