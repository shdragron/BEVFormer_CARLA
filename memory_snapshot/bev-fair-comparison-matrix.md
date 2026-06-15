---
name: bev-fair-comparison-matrix
description: Fair-comparison settings per model (BEVDepth/BEVDet/BEVFormer/PETRv2/CAPE) for the CARLA geobev VP+CTS NDS benchmark
metadata: 
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

The CARLA geobev VP (viewpoint robustness) + CTS (cross-platform transfer) NDS benchmark
compares camera-only 3D detectors under **standardized "fair" settings** that strip
model-specific tricks. Per-model knobs (user-confirmed 2026-06-03):

| model | settings |
|---|---|
| **BEVDepth** | BEV grid **128×128**, img **256×704**, **EMA OFF**, **CBGS OFF**, depth GT = **CARLA dense DPT render depth image** (not LiDAR) |
| **BEVDet** | same: 128×128 / 256×704 / EMA OFF / CBGS OFF / **DPT-image depth** (`LSSViewTransformerBEVDepth` + `CarlaDPTMultiViewDepth`, `loss_depth_weight=3.0`) |
| **BEVFormer-tiny** | BEV grid **50×50** (`bev_h_=bev_w_=50`); see [[carla-bevformer-conda-env]] |
| **PETRv2** | run **single-frame (temporal OFF)**, **no data-aug**, EMA/CBGS/fp16 OFF — sparse query/3D-PE, no BEV grid. **Query Denoising (DN) KEPT** → use `PETRv2DNHead` (not the DN-free `PETRv2Head`) |
| **CAPE** | **single-frame** (not CAPE-T), **no data-aug**, EMA/CBGS/fp16 OFF — camera-view PE, no BEV grid. **DN KEPT** → leave `CAPETransformer.prepare_for_dn` on, NO code edit |
| **DETR3D** | **single-frame** (native), **no data-aug** (grid_mask + PhotoMetricDistortion off), EMA/CBGS/fp16 OFF, R50-ImageNet (drop FCOS3D pretrain, `load_from=None`), native FPN + full-res 1600×900 KEPT. **NO DN** (predates query denoising → no-DN IS native; nothing to keep). See [[detr3d-carla-setup]] |
| **DFA3D** | **Paradigm = Backward** (NOT Sparse — subclasses BEVFormer, BEV grid). = "BEVFormer row + depth": BEV **50×50**, **R50 ImageNet** (no DCN/FCOS3D), **800×450**, single-frame, aug/EMA/CBGS/fp16 OFF, fp32, **depth = CARLA DPT image** (the ONLY added variable, depth-aware 3D deform attn). Config = tiny_carla skeleton + small_DFA3D grafts (NOT base_DFA3D). Global batch 16 @ lr4e-4 (= BEVFormer-tiny). See [[dfa3d-carla-setup]] |

**Fairness boundary (decided 2026-06-03):** equalize only model-AGNOSTIC boosters and
the data/temporal axes — OFF: data augmentation (GridMask, IDA resize/crop/flip,
BDA/GlobalRotScaleTrans), temporal fusion, EMA, CBGS, fp16. KEEP each architecture's
NATIVE training paradigm — **Query Denoising (DN) is KEPT** for PETRv2/CAPE because it's
the design intent of sparse-query detectors (stabilizes the bipartite matching that IS
the method), not a bolt-on trick; BEVFormer is dense BEV-grid so DN doesn't apply (no
queries to denoise) — forcing DN off would handicap the query models with no matching
gain to BEVFormer, i.e. LESS fair. FocalLoss is common to all three → kept. **DCN KEPT** (user 2026-06-03: "DCN은 괜찮고" —
architecture choice, like DN). STILL OPEN: PETRv2's VoVNet + FCOS3D/DD3D depth-pretrained
backbone (a pretraining/extra-supervision advantage vs BEVFormer/CAPE's ImageNet R50) —
decide keep-native vs ImageNet-parity.

The visibility>=2 eval fix is now CANONICAL in `CarlaNuScenesDataset._evaluate_single`
(`carla_nuscenes_dataset.py` `_patch_nuscenes_eval_for_carla` → `patched_load_gt`, keeps
vis>=`CARLA_MIN_VISIBILITY=2` + sets num_pts=1). This covers VP AND CTS AND training-val
(all route through `_evaluate_single`), so CTS needs only a RE-RUN, no code change. The
`patch_eval_visibility` monkeypatch in `eval_vp_robustness_det.py` is now redundant.

**GT filtering invariant (all models, user-flagged 2026-06-03):** filter GT by
**visibility ≥ 2** (`VISIBLE_TOKENS={'2','3','4'}`, ≥40% visible), NOT by the standard
nuScenes `valid_flag = num_lidar_pts>0` (CARLA has no LiDAR → would be empty/wrong).
BEVFormer achieves this because `create_carla_data.py` sets `valid_flag = visibility_token∈{2,3,4}`
and `bevformer_tiny_carla.py` sets `use_valid_flag=True`. So PETRv2/CAPE must **consume the
same CARLA pkls** (whose valid_flag is already visibility-based) and set `use_valid_flag=True`
— do NOT recompute valid_flag in any PETRv2/CAPE converter. This keeps the eval GT set
identical across models so NDS is comparable.

**Box-conversion invariant (verified 2026-06-05):** the prediction→nuScenes-box
conversion (`_format_bbox`+`output_to_nusc_box`) is **byte-identical mmdet3d 0.17.1
canonical** across ALL FOUR (BEVFormer/CAPE/PETRv2/DETR3D) — none of their
`CustomNuScenesDataset` subclasses override it (they only override `get_data_info`
and `_evaluate_single`), so it resolves to the installed mmdet3d file, NOT each repo's
`multi_nuscenes_dataset.py` custom version (that's TEMPORAL-only, unused by our
single-frame CarlaNuScenesDataset — a red herring). **Yaw:** `box_yaw = -box_yaw - π/2`,
`Quaternion(axis=[0,0,1])`, then lidar2ego→ego2global rotation. **Translation:** built at
`gravity_center` (= bottom-center + h/2, geometric; NOT bottom-center → z correct), then
lidar→ego→global `translate`, final `box.center`. All trained against
`gt_bboxes_3d.gravity_center + tensor[:,3:]` (same mmdet3d yaw), so pred↔GT yaw align.
BEVFormer round-trips correctly on this GT (P_NORMAL 0.5002, sane mAOE/mATE) ⇒ the others'
conventions are validated. NO code fix needed; eval is apples-to-apples.

**P_NORMAL gate (HARD requirement, user 2026-06-03):** mVRS (VP) and CTS are RATIO
metrics normalized by the baseline (mVRS=NDS_cond/P_NORMAL, CTS=NDS_cond/P_TARGET). If a
baseline is at the floor, the ratio is meaningless noise. PETR-family notoriously fails
to train without 3D-aware init (FCOS3D/DD3D) — which we REMOVE for fairness (ImageNet
R50). So before trusting/reporting ANY PETRv2/CAPE ratio: smoke-train, then VERIFY
P_NORMAL (the Normal-cell NDS) converges to a REASONABLE value on CARLA (sanity ref:
BEVFormer-tiny P_NORMAL = 0.5002 @ vis>=2). If P_NORMAL floors → ratios are invalid;
escalate (longer schedule / fallback to 3D-init with documented caveat / mark
"non-converged under fair init"). ALWAYS expose the absolute P_NORMAL (and CTS's P_TARGET
oracle) as columns in Table 2 — never hide the denominator behind the ratio. The VP
harness already saves absolute per-cell NDS + Normal NDS in eval_vp_per_config.csv.

**Why:** fairness — remove EMA, CBGS resampling, data augmentation, and temporal
fusion so robustness (VP) / transfer (CTS) differences reflect the architecture, not
training tricks; standardize the BEV grid + use DPT monocular depth as the common
depth source instead of LiDAR; and keep the GT set identical (visibility≥2).

**How to apply:** BEVDepth/BEVDet CARLA work lives in SEPARATE forks/sessions, NOT this
repo: `shdragron/BEVDepth_CARLA` (main, commit 67e556b) and `shdragron/BEVDet_CARLA`
(dev3.0, commit a3691a0); their Claude session is rooted at `viewpoint/BEVFormer/BEVDepth`
(`~/.claude/projects/-home-hanyan-arch-viewpoint-BEVFormer-BEVDepth`). BEVFormer + the
shared VP/CTS NDS harness live here ([[carla-geobev-dataset-layout]], `bev_det_benchmark/`).
PETRv2 + CAPE are being added now (fresh GitHub clones under this repo); configure them
no-aug + temporal-off to match this matrix.
