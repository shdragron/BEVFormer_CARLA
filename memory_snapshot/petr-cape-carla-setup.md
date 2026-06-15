---
name: petr-cape-carla-setup
description: "PETRv2 + CAPE CARLA port state — env, configs, dataset, R50-caffe ckpt, what's verified vs pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

Adding PETRv2 + CAPE to the CARLA VP/CTS benchmark (see [[bev-fair-comparison-matrix]]).
GPU-free prep DONE + verified (2026-06-03); training pending (waits for the VP run to free GPUs).

**Env:** `legacy-mmdet140-b200` — verified: python 3.10.20, torch 2.11.0+cu128, CUDA 12.8,
mmcv-full 1.4.0, mmdet **2.14.0**, mmdet3d 0.17.1, mmseg **0.14.1**. NOTE the stock
install.md lists mmdet 2.24.1 / mmseg 0.20.2 — the *verified* env uses 2.14.0 / 0.14.1.
mmcv-full 1.4.0 + mmdet3d 0.17.1 are **gwangik's editable source builds** at
`/NHNHOME/.../gwangik/project/jepa-driving/external/legacy_openmmlab/{mmcv-v1.4.0-official,
mmdetection3d-v0.17.1-official}` (both git HEAD == exact tag; the env's working set = their
uncommitted working-tree diffs). The actual CARLA working repos are `viewpoint/BEVFormer/{PETR,CAPE}` (have the carla
configs/ckpt/`Add CARLA geobev support` commits; `BEVFormer/PETR/mmdetection3d` symlink →
`BEVFormer/CAPE/mmdetection3d`). The bare `viewpoint/{PETR,CAPE}` are clean upstream clones
with NO CARLA work — do NOT use them. Both working repos' `origin` is still upstream
(megvii-research/PETR, kaixinbear/CAPE), so sharing needs your own fork.
`PETR/mmdetection3d` is a symlink → `CAPE/mmdetection3d` (so PETR config `_base_` resolves).
`{PETR,CAPE}/data/nuscenes` → carla_geobev. NOTE: run scripts from inside the repo, NOT
`/tmp` (a `/tmp/mmcv` checkout shadows the mmcv install → ImportError).

**Configs created** (sedan/suv/bus each; suv/bus are full copies differing only in ann_file):
`PETR/projects/configs/petrv2/petrv2_carla_{sedan,suv,bus}.py` (base: denoise/petrv2_..._800x320_dn.py),
`CAPE/projects/configs/CAPE/cape_carla_{sedan,suv,bus}.py` (base: cape_r50_1408x512_wocbgs_imagenet).
Both: `CarlaNuScenesDataset` ported into each repo's `projects/mmdet3d_plugin/datasets/`
(subclasses the repo's CustomNuScenesDataset for get_data_info; visibility>=2 GT filter +
6-class NDS, same as BEVFormer). Fairness per [[bev-fair-comparison-matrix]]: R50+DCN, DN KEPT,
single-frame (with_time=False), no aug/EMA/CBGS/fp16. Geometry: pc_range=[-51.2,-51.2,-5,51.2,51.2,3],
position_range/bound=[-61.2,..,61.2,61.2,10], num_classes=6.

**GOTCHA (found+fixed):** msra R50 weights are **caffe-style** → backbone needs
`style='caffe'` + `img_norm_cfg std=[1.0,1.0,1.0], to_rgb=False` (NOT pytorch std). The
impl agent wrote PETRv2 with style='pytorch'+pytorch-std (wrong, would hurt convergence);
fixed in all 3 PETRv2 configs. CAPE was already correct.

**Ckpt:** `resnet50_msra-5891d200.pth` (downloaded from open-mmlab, 94MB) at `PETR/ckpts/`,
symlinked into `CAPE/ckpts/`. Loads cleanly (missing=18 = DCN offset layers fresh-init, expected).

**Verified:** build_model OK, build_dataset(3792) OK, get_data_info lidar2img OK, Test B
GT-projection = 1.0000 (geometry sound — see geometry verify: lidar2img byte-identical to
BEVFormer's), ckpt loads. **PENDING (needs GPU):** smoke-train → P_NORMAL gate (per
[[bev-fair-comparison-matrix]]: if P_NORMAL craters it's convergence/pretrain not coord bug,
since Test B passed) → full train (sedan VP + suv/bus CTS oracles) → VP/CTS slow-path eval.
