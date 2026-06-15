---
name: vp-cross-model-mechanism-finding
description: "VP cross-model headline: robustness splits by 'does the extrinsic gate feature sampling?' (NOT depth, NOT Dense/Sparse). Code-verified. CAPE is the key disproof of depth-vs-sampling."
metadata:
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

The corrected VP cross-model finding (`results/VP_CROSS_MODEL_ANALYSIS.md`, all-cam, frame-fixed):
the discriminating axis is **whether the camera extrinsic gates feature sampling** — verified by a
5-agent code audit (all HIGH confidence, 2026-06-08), NOT depth and NOT Dense-vs-Sparse.

- **extrinsic-gates-sampling** (BEVFormer `encoder.point_sampling` lidar2img→deformable; DETR3D
  `feature_sampling` lidar2img→grid_sample): backbone extracts 2D feats *independent of extrinsic*,
  extrinsic decides WHERE to sample. Signature: **EXT≈IMG** (BEVFormer 0.428≈0.426, DETR3D 0.438≈0.422),
  **CAL recovers incl pitch** (BEVFormer CAL-pitch 0.661, DETR3D 0.825), **EXT worst-axis=pitch**.
- **extract-then-place** (CAPE camera-view PE via `position_embeding` I_inv/R/t, NO grid_sample;
  BEVDet/BEVDepth LSS backbone+depth then `get_lidar_coor`/voxel splat): feats extracted *independent
  of extrinsic*, geometry applied AFTER as PE or splat. Signature: **EXT≫IMG** (CAPE 0.811≫0.407,
  BEVDet 0.610≫0.364), **CAL-pitch STAYS collapsed** (CAPE 0.182, BEVDet 0.254 — tilted image already
  baked into extraction, consistent extrinsic can't undo), **EXT worst-axis=yaw**.

**CAPE is the proof it's NOT depth:** CAPE has no explicit depth (camera-view PE) yet patterns with
the LSS depth models on every axis. Its camera-view PE (designed to decouple from extrinsic) makes it
the MOST extrinsic-robust model (EXT all-cam 0.811, roll/pitch ~0.96-0.98). I got this WRONG twice
first (depth-vs-sampling, then BEVDet "softened double-edge") before CAPE arrived and forced the
mechanism reframing — verify mechanism claims against code, don't infer from "depth".

CAL-yaw recovers for ALL models (~0.92-0.98: yaw preserves ground-plane); only the horizon-tilting
axes (roll/pitch) defeat extract-then-place under consistency. **ALL 5 MODELS NOW COMPLETE**
(BEVDepth was already frame-fixed in results/BEVDepth/vp/ — I wrongly called it pending; it CONFIRMS
extract-then-place: EXT 0.652≫IMG 0.328, CAL-pitch 0.132). Within extract-then-place there's a
**depth-supervision gradient on CAL-pitch**: more depth supervision → more pitch-locked (BEVDepth
explicit-depth 0.132 < BEVDet implicit 0.254; CAPE no-depth 0.182) — 2nd-order, within-class.
Frame-2x bug ([[vp-carla-vr-frame-2x-misalignment]]) only touched VR/CR. results/{model}/vp all frame-fixed.
