---
name: coin3d-b200-env
description: "CoIn3D conda env on B200 = clone bevdet-b200 + layer CoIn3D pieces (NOT the README's cu116/torch1.12). Use `coin3d`."
metadata: 
  node_type: memory
  type: project
  originSessionId: 09f0506b-7473-403c-8e74-a3ace5c5a83b
---

CoIn3D (`/home/hanyan_arch/viewpoint/BEVFormer/CoIn3D`, CVPR'26 config-invariant
multi-cam 3D det) = a BEVDet fork (mmdet3d 1.0.0rc4, bev_pool_v2) + **3D Gaussian
Splatting renderer** (`third_party/diff-gaussian-rasterization`, INRIA) + custom
nuscenes-devkit (Lyft/Waymo→nuScenes) + image-inpainting preprocess (ZITS++/SPNet).

**Run it with conda env `coin3d`.** Its README install recipe (python3.8.5,
torch1.12.1+**cu116**, mmcv1.6) does NOT run on B200 (sm_100 needs cu12.x). So the env
was built by **cloning `bevdet-b200`** (torch 2.11.0+cu128, arch_list has sm_100/sm_120,
mmcv-full 1.7.1, mmdet 2.28.2, mmdet3d 1.0.0rc4 — same family; nvcc 12.8 lives INSIDE
the env at `$CONDA_PREFIX/bin/nvcc`) and layering the CoIn3D-specific pieces. Pattern
mirrors the other `*-b200` envs (clone, don't rebuild the painfully-built B200 mmcv).
[[bevdet-dev30-b200-env]] is the sibling.

Build recipe that worked (`conda create --clone bevdet-b200 -n coin3d`, then):
- ext builds need `export CUDA_HOME=$CONDA_PREFIX CC=gcc-13 CXX=g++-13
  TORCH_CUDA_ARCH_LIST=10.0` (B200 = capability (10,0) = sm_100; system /usr/local/cuda
  is 13.1 and would mismatch torch's cu128 — must use the in-env nvcc 12.8).
- `pip install mmsegmentation==0.30.0` (clone had 0.14.1; CoIn3D mmdet3d asserts mmseg
  ≥0.20.0; 0.30 is mmcv-1.7-compatible, leaves mmcv-full 1.7.1 untouched).
- **2 source patches** (env-specific adaptations, committed-tree edits):
  1. `BEVDet/mmdet3d/__init__.py`: `mmcv_maximum_version '1.7.0'→'1.7.1'` (else import
     asserts; the B200 mmcv is 1.7.1, API-compat).
  2. `third_party/nuscenes-devkit/setup/setup.py`: its recursive package discovery
     globs the build-generated `*.egg-info` dir into `packages` → `error: package
     directory python-sdk/nuscenes_devkit/egg-info does not exist`. Fix: add
     `and 'egg-info' not in p` to the `__pycache__` filter (and `rm -rf python-sdk/*.egg-info`).
- `pip install -e BEVDet --no-deps --no-build-isolation` (the **--no-deps is required** —
  its requirements pin `numba==0.53.0` which has no py3.10 wheel; the clone already has a
  working numba. This re-points editable mmdet3d to CoIn3D/BEVDet and rebuilds
  `bev_pool_v2_ext` for sm_100).
- `pip install -e third_party/nuscenes-devkit/setup --no-deps` (custom devkit 1.1.11).
- `pip install jaxtyping spconv-cu120` (einops/omegaconf already in clone; spconv-cu120
  works on cu12.x; emits harmless `torch.cuda.amp` FutureWarnings).
- `pip install -e third_party/diff-gaussian-rasterization --no-deps --no-build-isolation`
  (3DGS CUDA ext; glm submodule was already populated; built clean for sm_100 — `_C`
  exposes `rasterize_gaussians`/`mark_visible`).

Verified: torch sees `NVIDIA B200`, B200 matmul launches, and `mmdet3d`,
`bev_pool_v2_ext`, `diff_gaussian_rasterization._C`, custom `nuscenes`, `spconv` all
import. Data prep + ckpts (mix/nuscenes/waymo/lyft .pth from modelscope) not done yet.
