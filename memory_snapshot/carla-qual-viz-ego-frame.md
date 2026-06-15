---
name: carla-qual-viz-ego-frame
description: CARLA geobev GT/pred 3D boxes are EGO-frame; project to images with sensor2ego (not sensor2lidar). Qual viz + results-folder layout.
metadata: 
  node_type: memory
  type: project
  originSessionId: 09f0506b-7473-403c-8e74-a3ace5c5a83b
---

For the CARLA geobev viewpoint benchmark, GT boxes (`ann_infos`) AND model predictions are
in the **EGO frame** (ground at z=0; box-center z ≈ object half-height), NOT the lidar frame.

**How to apply:** to project 3D boxes onto camera images, use `inv(sensor2ego)` (cam←ego),
i.e. `pixel = K @ inv(sensor2ego) @ corners` on the ORIGINAL 900×1600 image (intrinsic
principal point = image centre, no IDA). Using `sensor2lidar` floats every box ~1.8 m too
high, because `lidar2ego=[0,0,1.8]` (the lidar sits 1.8 m above ego). This bit me twice on
the BEVDet qual viz before I checked empirically (GT-on-truck aligned only with sensor2ego).
Verify on a clean cars-only scene (e.g. sedan scene-0269): red pred should wrap the parked
cars and overlap green GT.

BEV top-down plots are unaffected (x/y identical in ego vs lidar — only z differs), so the
per-distance recall plot ([[bevdepth-carla-sedan-result]] sibling work) was already correct.

**Gotcha (cost real debugging):** when pred boxes don't sit on objects but GT does, suspect a
**sample↔pred frame misalignment**, NOT the model. `CarlaNuScenesDataset.load_annotations`
REORDERS infos (by token), so `test.py`/`single_gpu_test` outputs are in `dataset.data_infos`
order, not the raw pkl order. The qual renderer must map each pred to its frame by **sample
token** (`{data_infos[j]['token']: preds[j]}`), else it draws scene B's boxes on scene A's
image — which looks exactly like the model spewing pedestrian FPs / missing cars (it isn't).
Same root cause as the BEV-recall index bug. Always drive the render from data_infos order or
token-match; never `zip(raw_pkl_infos, preds)`.

**Condition qual (VP/CTS EXT/IMG/CAL), BEVDet** — `qual_conditions_bevdet2.py`. Two BEVDet
specifics vs the BEVDepth recipe in [[carla-qual-conditions-coords]]: (1) BEVDet
`make_vp_infos` perturbs **sensor2ego** (sensor2lidar stale) — BEVDepth is the opposite — so
project via `inv(sensor2ego)`. (2) Per user's spec, render GT+pred with the **image's TRUE
camera pose** (variant pose for tilted IMG/CAL, baseline for EXT, sedan/target for CTS), NOT
the model-fed (mismatched) ext: GT then lands on the visible objects and the PRED reveals the
degradation as a shift (EXT pred rotates ~axis° off GT; IMG pred collapses on the tilted view;
NORMAL/CAL pred on objects). Variant pose verified = clean ±20° rotation of baseline (transl
unchanged). **carla_VR is a SEPARATE render** from the sedan dataset: same camera pose
(VR yaw0pitch0roll0 pose == sedan baseline, 0.0° diff) but a different render pass
(mean|sedan_img − VR_roll0_img| ≈ 40), and the model is sensitive to it (preds ~22 on the
sedan render → ~5 on the carla_VR roll0 render, same scene). So for frame-consistent VP qual
use the **carla_VR roll0 image for NORMAL/EXT** (not the sedan dataset image), else NORMAL
looks like a different scene than the rotated variants. Tradeoff: carla_VR-consistent frames =
sparser preds (render-domain gap); user chose frame-consistency. All preds need token alignment + per-condition inference (sedan ckpt; CTS = sedan
model on target). Caveat: at ±20° some GT objects rotate out of the clear FOV and "float" over
background — real (rotated-out), not a coord bug. Out: `results/BEVDet/qual_conditions/`
(gitignored `results/*/qual_conditions/`).

**CRITICAL carla_VR frame-index bug (geobev N == carla_VR 2N).** carla_VR is rendered at
2× the geobev frame rate: geobev scene frames are 0,2,…,156 (step 2, 79 frames); carla_VR
are 0,1,…,317 (step 1, 318). So geobev `scene-XXXX-frame-N` == carla_VR `frame-2N`
(verified: geobev 0269-150 vs VR-300 pixel-diff 17.7 = same scene+render-gap, vs VR-150 = 40
= different scene). But every `vr_image_path(scene, frame, …)` uses `int(frame)` (N), so the
VP **image-swap** conditions (VR/CR, and CTS-VR if any) fed the model a DIFFERENT scene's
image than the GT → GT floated. Fix = `int(frame)*2` (pose is per scene+cam+variant, frame-
independent, so ONLY the image path is wrong; sensor2ego was always fine). Impact: VP
Normal/ER (no image swap) + CTS (geobev imgs only) UNAFFECTED; **VP VR/CR INVALID** in the
committed `results/*/vp/` — the VR RRSALL≈0.15-0.19 / CR≈0.24 "image-collapse" headline is
INFLATED by the frame mismatch. After fix the model is far more robust (yaw+20 all-cam IMG
pred 6→22); real conclusion: **pitch worst, yaw robust** (not "image≫extrinsic"). Bug is
copy-pasted in EVERY model's builder (BEVFormer/BEVDet/BEVDepth/DETR3D/CVT-seg). Re-running
the VP NDS benchmark with the fix is the user's call.

**Results-folder convention** (matched across models, see [[bev-fair-comparison-matrix]]):
`results/<Model>/` = `cts/` (committed) + `vp/` (committed, RRS table) + `ckpts/` (gitignored
`*.pth`/`*.ckpt`) + `qual/{thr0.3,thr0.5}/{veh}_scene-XXXX-frame-YYYY.jpg` (gitignored,
local-only, large). BEVDet qual renderer: `bev_det_benchmark/qual_render_6view.py`, mirrors
`BEVDepth/_viz_qual.py` (in-range ±51.2 filter, FOV-cull: centre depth∈[1.5,70] + centre in
image±100px + all 8 corners z>0.3). `.gitignore` rule `results/*/qual/` keeps the jpgs local.
Qual samples = the 10 highest IN-pc-range vis≥2 distinct-scene val frames.
