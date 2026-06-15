---
name: vp-carla-vr-frame-2x-misalignment
description: CRITICAL VP benchmark bug — carla_VR frame N != geobev frame N; correct match is VR frame 2N. All VR/CR image-swap results invalid until refixed.
metadata: 
  node_type: memory
  type: project
  originSessionId: d6e20ce2-7fd8-462d-9469-4a1b57b88da9
---

**carla_geobev is a 1/2-rate RELABEL of the carla_VR capture: geobev frame N is the
same world-moment as carla_VR frame 2N.** Verified by image match: geobev sedan
scene-0269-frame-0150 == VR baseline frame-0300 (exact 2.000x on every clean-motion
scene; mean 6-cam pixel-diff 16.1 @2N vs 26.3 @N; low-motion scenes tie within noise).
Both `data/nuscenes` (BEVFormer) and `BEVDepth/data/carla` symlink the same
`/NHNHOME/.../jeongtae/carla_geobev`; carla_VR is `/NHNHOME/.../jeongtae/carla_VR`
(50 scenes 0220–0269, 31 variants, frames 0–~317 step 1; geobev frames 0–156 step 2).

**The bug:** every `vr_image_path(scene, frame, cam, variant)` in the VP builders uses
`int(frame):04d` (geobev frame N) → loads carla_VR frame N, a DIFFERENT scene moment.
So every image-swapped condition (VR=image-only primary, CR=both) fed the model
mismatched images. Fix = use `int(frame)*2`. Affected builders (each has its own copy):
`bev_det_benchmark/build_condition_pkls.py` (BEVFormer), `build_condition_pkls_bevdet.py`,
`sparse/build_condition_pkls.py` (DETR3D), `build_condition_pkls_bevdepth.py` (BEVDepth),
and the seg `bev_seg_benchmark/eval_vp_robustness_cvt.py`. Extrinsic swap (ER) and Normal
are UNAFFECTED (no image swap); CTS is UNAFFECTED (uses geobev images only).

**Impact:** the committed VP numbers in `results/BEVDepth/vp/` (VR RRSALL=0.151,
CR=0.240, "image perturbation collapses NDS") are INFLATED by frame misalignment.
Corrected qual (frame 2N) shows the model is far more robust: scene-0269-frame-0150
yaw+20 all-cam pred 5→14 (VR), 5→19 (CR); pitch+20 stays worst (pred~4 = genuine
collapse). True story: pitch worst, yaw largely tolerated. Re-run VP with the fix for
correct numbers (large compute — user decides).

**Smoke-confirmed (yaw+20 all-cam) — the fix changes the STORY, esp. for CR:**
| | BUGGY (committed) | FIXED (2N) |
|---|---|---|
| VR (model uninformed) | RRS 0.001 | RRS 0.122 — still collapses (genuine) |
| CR (model told new extrinsic) | RRS 0.490 | **RRS 0.963 — recovers** |
So an *uncalibrated* viewpoint shift (VR) breaks the model; a *calibrated* one (CR) is
largely robust — opposite of the buggy "CR also collapses." VR collapse is real (image–
extrinsic mismatch), not a frame artifact.

**Remediation status (2026-06-06):** all 5 det builders fixed — bevdet committed by user
(3551b46), BEVFormer/BEVDepth/sparse by me (3e987d1). BEVDepth condition qual_cond committed
(7e13e62, frame-2N). Seg/CVT VR path is built in external CVT dataset code (`eval_viewpoint_variant`
flag) — NOT yet fixed, flagged. BEVDepth VP VR+CR re-run (frames-per-scene 16, 2-GPU sharded,
tag bevdepth_sedan_vrcr) running; merge_vp_vrcr_fix.py keeps committed ER+Normal, swaps in
frame-fixed VR/CR, rewrites results/BEVDepth/vp. Other models' VP (BEVFormer/BEVDet/DETR3D)
still need re-run. `BEVDepth/_viz_cond_qual.py` now uses the fixed builder directly (monkeypatch
removed). Discovered 2026-06-06 while building condition-qual.
Related: [[carla-geobev-dataset-layout]], [[bevdepth-carla-sedan-result]], [[carla-qual-conditions-coords]].
