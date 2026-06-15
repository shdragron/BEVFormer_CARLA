---
name: dfa3d-carla-setup
description: "DFA3D (7th detector) port plan — fills the empty gates-sampling×depth quadrant; BEVFormer vs BEVFormer_DFA3D = single-variable (depth) controlled comparison. Repo, mechanism (verified), risks, plan."
metadata:
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

**Why DFA3D (decided 2026-06-08):** the VP cross-model 2×2 (mechanism × depth, see
[[vp-cross-model-mechanism-finding]]) has one empty cell — **gates-sampling × uses-depth**.
DFA3D fills it. Repo `viewpoint/BEVFormer/3D-deformable-attention` provides **DFA3D-enabled
BEVFormer**, so **BEVFormer (gates, no-depth) vs BEVFormer_DFA3D (gates, +depth) is a clean
single-variable controlled comparison** (same BEV base, only depth-aware 3D lifting added) →
strongest evidence for VP=mechanism / CTS=depth axis independence.

**Mechanism VERIFIED (code, 2026-06-08) = gates-sampling + explicit depth:** `DepthHead_MLVGDpt`
(dense_heads/depth_head.py) predicts a depth distribution (supervised by GT depth via
`get_downsampled_gt_depth`), which **expands each view's 2D feature to 3D**; then
`MSDeformableAttention3D_DFA3D` / `SpatialCrossAttention_DFA3D` projects BEV queries into images
and **samples from the depth-expanded 3D features** (deformable = gates-sampling). So depth IS in
the feature/value path (unlike pure 2D-attn BEVFormer/DETR3D). Abstract confirms: LSS=splat
(extract-then-place), 2D-attn=no-depth (gates), DFA3D=depth-expand-then-3D-attn (gates+depth).

**Repo layout:** `3D-deformable-attention/` = `DFA3D/` (custom CUDA op: dfa3D/ops/csrc,
multi_scale_3D_deform_attn.py — UNBUILT) + `BEVFormer_DFA3D/` (full BEVFormer fork with
projects/mmdet3d_plugin: BEVFormer_DFA3D detector, DepthHead, SpatialCrossAttention_DFA3D,
MSDeformableAttention3D_DFA3D, BEVFormerHead_DFA3D, PerceptionTransformer_DFA3D). Base config
`bevformer_base_DFA3D.py` = R101+DCN, queue_length 4 (temporal), 10-class, DepthHead at FPN
indice_layer=2, pipeline has `LoadMultiViewDepthFromFiles`.

**TWO RISKS (check first):** (1) **CUDA op compile on B200** — env bevformer-b200 = torch 2.11.0+cu128,
sm_100; the ICCV'23 op (torch ~1.x) likely needs patching (cf BEVDet bev_pool_v2 gcc-13 pain
[[bevdet-dev30-b200-env]]). (2) **CARLA depth GT** — DepthHead needs dense depth maps for
supervision; reuse the CARLA depth the forward models (BEVDet/BEVDepth, "DPT Image") use — locate
path+format, write/point LoadMultiViewDepthFromFiles at it.

**PLAN (fair-protocol, match our bevformer_tiny_carla):** R50 ImageNet + DCN (no FCOS3D), BEV 50×50,
single-frame (queue_length=2, unique scene_token→prev_bev=None), 6-class CARLA, aug/EMA/CBGS off,
fp32. Port CarlaNuScenesDataset + depth loading; add DepthHead with CARLA depth supervision. Steps:
(a) compile op → (b) config+dataset+depth port → (c) verify coords/projection + depth load + build_model
→ (d) train sedan + suv/bus oracles → (e) VP (frame-fixed builder) + CTS.

**RISKS BOTH RESOLVED (2026-06-08):** (1) **op COMPILES + unittest PASSES on B200** — the ONLY patch
needed was `sed -i 's/c++14/c++17/g' DFA3D/setup.py` (torch 2.11 needs C++17); build with
`TORCH_CUDA_ARCH_LIST=10.0 CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 pip install -e . --no-build-isolation`
(gcc-13 needed: env gcc-14 > CUDA-12.8 max). No kernel API drift. ops:
MultiScale3DDeformableAttnFunction / WeightedMultiScaleDeformableAttnFunction /
MultiScaleDepthScoreSampleFunction. unittest_DFA3D.py prints "Error" only on NaN → silent+exit0 = pass.
(2) **depth GT found:** carla_geobev/sweeps/**DPT-{cam}** (sedan) + **DPT-{suv,bus}-{cam}** dense depth
images; BEVDet's `CarlaDPTMultiViewDepth` pipeline (configs/bevdet/carla/bevdet-r50-carla.py:151,
grid_config+downsample) already loads them → adapt for DFA3D DepthHead.

**PORT COMPLETE + COORDS VERIFIED (2026-06-08):** config `bevformer_DFA3D_carla.py` runs end-to-end.
Files in `3D-deformable-attention/BEVFormer_DFA3D/projects/mmdet3d_plugin/`:
- `datasets/carla_nuscenes_dataset.py` = byte-copy of our main-repo CarlaNuScenesDataset (imports
  resolve to DFA3D's **dpt-aware** `CustomNuScenesDataset` @ nuscenes_dataset.py:255, whose union2one
  stacks `dpt` across the queue + sets prev_bev_exists; nuscnes_eval.py is byte-IDENTICAL to ours).
  Registered in datasets/__init__.py.
- `datasets/pipelines/loading.py::CarlaDPTMultiViewDepthDFA3D` — RGB→DPT path (`.replace('RGB','DPT')
  .replace('.jpg','.png')`, auto-selects per-vehicle DPT-{suv,bus,cam} folder), decode
  `(R+G*256+B*256²)/(256³−1)*1000` planar-Z meters → `results['dpt']` = list of (H,W) dense maps.
  Registered in pipelines/__init__.py.
- **Pipeline gotcha (FIXED):** base `RandomScaleImageMultiViewImage` does NOT scale dpt; only
  `RandomScaleImageMultiViewImageDpt` (transform_3d.py:357, nearest) does → config train_pipeline uses
  the Dpt variant. PadMultiViewImage already pads dpt. CustomDefaultFormatBundle3D stacks dpt→(N,1,H,W).
  dpt contract: detector wants `(B,queue,N_cam,1,H,W)`; DepthHead_MLVGDpt down-samples /32, quantizes
  into dbound=[2,58,0.5] (sky 1000m auto-masked, 0=invalid). Test pipeline has NO dpt (BEVFormer_DFA3D
  .simple_test predicts depth from features) — fine.
- **VERIFICATION (bevformer-b200, GPU0):** build_dataset train=17380/val=3792, single-frame trick intact
  (token==scene_token all-unique→prev_bev=None); sample img (2,6,3,480,800) + dpt (2,6,1,480,800) /32-ok;
  build_model 40.1M w/ DepthHead_MLVGDpt; forward_train all det losses + **loss_dpt=2.18**, grad-norm 45.7
  finite, backward OK; simple_test 300 boxes. **COORDS: projrate 1.000** (= our BEVFormer baseline),
  **depth-align median(DPT_surface/box_center_Z)=0.988, |Δ|=0.19m** (DPT in same metric units +
  pixel-aligned to scaled+padded image as lidar2img). op = `dfa3D` (lowercase, editable-installed,
  compiled _ext.so). STATUS: ready to train sedan + suv/bus oracles.

**SEDAN TRAINING LAUNCHED + STABLE (2026-06-08 23:11, GPU1, sharing box w/ 3 full-val VP jobs).**
Loss stepping: Epoch[1][50/1087] loss 20.67, **loss_dpt 1.28**, grad_norm 13.9 (<clip 35), mem 47.9GB.
**FAIR CONFIG (공정/제대로):** matches BEVFormer-tiny baseline = global batch 16 @ lr4e-4 (single-GPU
samples_per_gpu=16; frozen BN ⇒ ≡ baseline 2GPU×bs8), R50/BEV50×50/single-frame/6-class/no-aug/fp32,
24 epochs, eval interval=2. wandb: **Robust_Ex/BEVFormer-CARLA** run `bevformer_DFA3D_carla_sedan`.
Launch: `CUDA_VISIBLE_DEVICES=1 bash tools/train_DFA3D_carla.sh <cfg> 28533` (tools/train_DFA3D_carla.sh,
single-GPU torch.distributed.launch nproc=1). work_dir `work_dirs/bevformer_DFA3D_carla`.
**4 LAUNCH FIXES (all needed):** (1) script must use env python `/NHNHOME/.../bevformer-b200/bin/python`
not system 3.12; (2) tools/train.py: `--local-rank` alias added (torch2.x launch passes dash form);
(3) **SIGKILL/-9 OOM at iter1** w/ 16 workers+dense-depth → added `torch.multiprocessing.set_sharing_strategy
('file_system')` to train.py + dropped **workers_per_gpu 16→4** (cgroup-capped docker box);
(4) im2col_step constraint: DFA3D CUDA op needs (bs*num_cams)%im2col_step==0, im2col_step_=min(batch,64);
bs16→batch96 fails default 64 → set **im2col_step=48** in MSDeformableAttention3D_DFA3D (numerically inert).
eval@epoch2 = first untested full _evaluate_single on this fork (byte-identical to main repo's, low risk).
NEXT after sedan: suv+bus oracles (need per-veh cfgs or cfg-options vehicle+ann_file+wandb name) → VP
(frame-fixed builder) + CTS.

**RUNNING 2-GPU (updated 2026-06-08 ~23:55, user moved off single-GPU to balance VRAM):** restarted as
2-GPU DDP `CUDA_VISIBLE_DEVICES=0,1 bash tools/train_DFA3D_carla.sh <cfg> 28533 2` (script now takes a 3rd
GPUS arg). **samples_per_gpu 16→8** so global batch stays 16 (= baseline), lr 4e-4 unchanged (frozen BN ⇒
1×bs16 ≡ 2×bs8; im2col_step=48 still divides 8×6=48). GPUs balanced 64GB each, 100% util; once VP jobs'
NFS contention eased, data_time dropped 5.7s→0.15s, **ETA ~1d3h** (~1h/epoch ×24). **wandb project FIXED
to `DFA3D_CARLA`** (was wrongly `BEVFormer-CARLA`) run `DFA3D_carla_sedan`; **val NDS/mAP auto-logs** via
standard CustomDistEvalHook→log_buffer→WandbLoggerHook (same as detr3d), no extra code. **PUSHED**:
`git@github.com:shdragron/DFA3D_CARLA.git` main (commit by shdragron, no Claude trailer; upstream README→
README_DFA3D.md, IDEA License 1.0 attribution kept). git diff = 6 files, ZERO model/op code (only CARLA
dataset+depth loader+config+infra fixes).

**PARADIGM & CONFIG CLASSIFICATION (user wants this remembered, 2026-06-08):**
- **DFA3D paradigm = Backward, NOT Sparse.** `BEVFormer_DFA3D` *subclasses* BEVFormer: BEV grid 50×50 +
  BEV-query→image deformable sampling = Backward (same row as BEVFormer in the benchmark table). It is
  NOT Sparse (Sparse = object-query, no BEV grid: DETR3D/PETRv2/CAPE). On the *mechanism* axis it IS
  gates-sampling (w/ DETR3D, BEVFormer) — the ONLY depth-using one there ([[vp-cross-model-mechanism-finding]]).
- **Did NOT use the base config.** `bevformer_DFA3D_carla.py` = `bevformer_tiny_carla.py` skeleton (our
  BEVFormer baseline) + DFA3D depth grafts from `bevformer_small_DFA3D.py` (single-level). base
  `bevformer_base_DFA3D.py` (R101+DCN+FCOS3D, 1600×900, BEV 200×200, queue_length=4 temporal, 10-class,
  num_levels=4) is reference-only, NOT used.
- **Backbone = R50 ImageNet** (`torchvision://resnet50`), out_indices=(3,) single FPN level, no DCN,
  frozen_stages=1 — vs base's R101+DCNv2+FCOS3D. **= BEVFormer-tiny.**
- **Image resolution = 800×450** (RandomScaleImageMultiViewImageDpt 0.5 from 1600×900) — vs base's full
  1600×900. **= BEVFormer-tiny.**
- **Temporal OFF (single-frame):** queue_length=2 (forward_train min) + CARLA pkl scene_token==token
  (all unique, sedan 17380/17380) ⇒ union2one sets prev_bev_exists=False ⇒ prev_bev=None always;
  video_test_mode=False, rotate_prev_bev=False, use_can_bus=False. **= BEVFormer-tiny method.**
- So benchmark-table DFA3D row = **Backward | BEV50×50, aug/EMA/CBGS/fp16 OFF, single-frame, Depth⇒DPT
  Image (depth-aware 3D deform attn) | Temporal(Original) O | 800×450 | ResNet50** — i.e. "BEVFormer row +
  depth". The ONLY variable vs BEVFormer is DFA3D's depth-aware lifting (clean single-variable study).
- Other models' paradigms verified correct: BEVDet/BEVDepth=Forward(LSS), BEVFormer=Backward,
  DETR3D/PETRv2/CAPE=Sparse. (The 3 Sparse split on mechanism: DETR3D=gates-sampling, PETRv2/CAPE=extract-then-place.)

**TRAINING DONE + EVAL PORT PUSHED + EVAL MOVES TO ANOTHER SERVER (2026-06-11):**
- All 3 oracles trained to epoch 24: sedan NDS6 0.4892 (06-09), suv 0.5138 (06-10), bus 0.5560 (06-11)
  — bus>suv>sedan, same ordering as DETR3D. ckpts mirrored to main repo `results/DFA3D/ckpts/`
  (gitignored; md5 sedan bcfc0523/suv 8930b9e6/bus b642f5b9) + `results/DFA3D/{README,indist/}` committed.
- **VP+CTS eval code ported INTO the fork** `BEVFormer_DFA3D/bev_det_benchmark/` (commits de4d04d,
  bf0f081 on shdragron/DFA3D_CARLA main): copies of main-repo drivers (plugins can't coexist in one
  process — same `projects` module name); only diffs = BEVF_ROOT/defaults/runner. KEY: DFA3D predicts
  depth at TEST time (loss_dpt train-only) → no DPT files needed for perturbed images; --fast path
  pipeline-identical to tiny. Smoke PASSED both paths (VP Normal NDS 0.4841@48f, RRS 0.362; CTS
  suv-oracle 1-scene NDS 0.4743). 2 blockers fixed: fork tools/test.py --local-rank alias; per-tag
  /tmp/vpeval_<tag>_shard<i> in BOTH repos (old shared path could cross-score concurrent runs).
- **USER RUNS DFA3D EVAL ON A DIFFERENT SERVER** — remote setup guide in the fork's
  bev_det_benchmark/README.md ("Running on another server": pip install -e DFA3D for _ext, copy 3
  ckpts, data/nuscenes + carla_VR mounts, edit BEVF_ROOT/VR_ROOT/DFA3D_ROOT, smoke first). Merged
  outputs to be copied back to main-repo results/DFA3D/{vp,cts}/.
