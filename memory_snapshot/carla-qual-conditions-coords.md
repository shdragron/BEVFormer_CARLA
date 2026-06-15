---
name: carla-qual-conditions-coords
description: "Condition-aware qual (VP/CTS EXT/IMG/CAL) — project raw LIDAR-frame gt_boxes via inv(sensor2lidar) using each condition's OWN sensor2lidar; sidesteps the stale-sensor2ego VP trap"
metadata:
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

For per-condition qual rendering (VP {NORMAL/EXT/IMG/CAL}, CTS-{suv,bus}
{NORMAL/EXT/IMG/CAL} on one scene) the **coordinate-exact recipe differs from the
NORMAL-only renderer** in [[carla-qual-viz-ego-frame]]. Two facts settle it:

1. The **raw val-pkl `gt_boxes` are LIDAR frame** (7-dim `[x,y,z,dx,dy,dz,yaw]`,
   ground at z=-1.8; truck box-bottom z-h/2 == -1.800 exactly; `lidar2ego=[0,0,1.8]`).
   The "ego-frame" claim in [[carla-qual-viz-ego-frame]] was about the DERIVED
   `ann_infos` field (gt_boxes shifted +1.8 to ego), which the raw val pkl does NOT
   contain — only `gt_boxes`/`gt_names`/`valid_flag`/`gt_velocity`.
2. **VP condition pkls perturb ONLY `sensor2lidar_*`; `sensor2ego_*` is left STALE**
   (`make_vp_infos` in build_condition_pkls.py). CTS `_copy_extrinsic` updates both.

So: **draw lidar-frame `gt_boxes` via `l2i = K @ inv(sensor2lidar)`** — BUT use the
**DISPLAYED IMAGE's TRUE geometry**, NOT the condition's model-input ext. GT must stay
fixed on the real objects; only PRED should move to reveal model error (user caught
this 2026-06-06: "GT가 같이 움직이면 안되지"). Image's true s2l = **variant s2l if the
shown image is the variant (IMG/CAL), baseline s2l otherwise (NORMAL/EXT)** — it tracks
`swap_img`, NOT `swap_ext`. Equivalently render via the *consistent* info: VP EXT renders
on the NORMAL info, VP IMG renders on the CAL info (img+matching geom). The lidar frame
is invariant to camera perturbation, so GT and PRED (model output, both lidar-frame) are
projected with this SAME image-true geom and are directly comparable: GT wraps objects;
pred deviates (EXT=shifted by wrong extrinsic, IMG=collapsed/off, CAL=recovered). The
model INFERENCE still runs on the condition's own (perturbed) sensor2lidar = the benchmark
input. (My earlier "use each condition's OWN sensor2lidar to draw" was WRONG for EXT/IMG —
it floats GT off objects; correct only for NORMAL/CAL where img-geom==model-geom.)
GT filter = `valid_flag` (vis>=2) +
in-range (|x|,|y|<51.2) + FOV-cull (centre depth in [1,80], all 8 corners z>0.3).
mmdet3d's `draw_lidar_bbox3d_on_img` is unusable here (np.int removed in this numpy)
— draw the 12 edges directly. Equivalence to ego path proven: `sensor2ego_cond =
lidar2ego @ sensor2lidar_cond` gives the same pixels to 1e-15 (verify_qual_coords.py).

**Verified visually** (scene-0267-frame-0016, pitch+20 all-cam, GT-only): NORMAL +
CAL boxes wrap the cars (CAL = consistent tilt, objects shoved to image bottom, boxes
follow); IMG boxes float into the sky OFF the cars (image tilted but extrinsic baseline
= the model's image↔extrinsic mismatch); CTS-bus EXT boxes float up (sedan image + bus
mount). Code: `bev_det_benchmark/qual_conditions.py` (CPU, GT only) +
`qual_conditions_pred.py` (GT green + BEVFormer-tiny sedan pred red). gt_boxes are
bit-identical across sedan/suv/bus (only `valid_flag` differs by mount height), so CTS
GT == sedan GT geometry. **Pred DONE:** BEVFormer pred boxes_3d are LIDAR frame (the
existing qual_viz_bevformer.py projects GT+pred with the SAME dataset lidar2img), so
pred projects via the condition's sensor2lidar exactly like GT — NO z-shift. Pred via
single-frame inference (prev_bev=None; build 1-frame condition pkl -> build_dataset ->
ds[0] -> model). Ran 3 axes (roll/pitch/yaw, +20, thr0.3) on scene-0267-frame-0016,
GPU0 alongside DETR3D training (spare VRAM, no OOM). out
`results/_qual_conditions/<scene>_pred_thr<t>/` = 18 6-view grids + VP_compare_axes.jpg
+ CTS_compare.jpg. Caveat: single-frame (no temporal context).

**carla_VR frame-2x fix (see [[vp-carla-vr-frame-2x-misalignment]]):** geobev frame N ==
carla_VR frame 2N. The builders' `vr_image_path` are NOW fixed in-repo (`int(frame)*2`:
build_condition_pkls.py + _bevdet + _bevdepth + sparse/). So qual_conditions*.py must NOT
also patch (double-applies -> 4x; I hit this — removed the monkeypatch). The earlier qual
strips (rendered pre-fix) showed the WRONG VR scene for IMG/CAL; re-rendered correct. The
pre-fix "VP IMG pitch -> pred=0 / EXT pitch -> pred=35" numbers were inflated by the wrong
scene; after the fix the model is much more robust (yaw pred ~3x higher), pitch still worst.
The committed results/{model}/vp CSVs + VP_CROSS_MODEL_ANALYSIS.md VR/CR findings are
INVALID pending a VP VR/CR eval re-run; ER/Normal/CTS unaffected.
