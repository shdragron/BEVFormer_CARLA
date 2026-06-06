# BEVDepth CARLA — results & checkpoints

Organized artifacts for the BEVDepth CARLA geobev benchmark.

## cts/  (committed)

`eval_cts.{json,csv,summary.txt}` — cross-platform transfer (CTS): the **sedan**-trained
model deployed on **suv/bus**, vis≥2 GT, 6-class NDS. Conditions NORMAL / EXT / IMG / CAL
(IMG = primary); denominator = target-native oracle (`CTS = NDS_cond / P_TARGET`).

| target | ORACLE (P_TARGET) | NORMAL | EXT | **IMG (primary)** | CAL |
|---|---|---|---|---|---|
| suv | NDS 0.5474 | 0.5047 | 0.3803 | **0.0564 (CTS 0.103)** | 0.0312 |
| bus | NDS 0.4182 | 0.4666 | 0.1252 | **0.0006 (CTS 0.0014)** | 0.0694 |

Image domain shift dominates the degradation (EXT >> IMG): a +viewpoint **image** swap
collapses transfer (suv 0.10, bus ~0). bus's extreme rig (cameras ±5.7 m fwd/back, 4.08 m
high) makes sedan→bus transfer near-zero. Oracles match training (suv 0.5474 / bus 0.4182)
→ deterministic, validated.

## ckpts/  (gitignored — `*.ckpt`, local only)

`bevdepth_carla_{sedan,suv,bus}_epoch23.ckpt` — 24 epochs (ep23 saved), BEV grid 128×128,
single-frame, vis≥2, CBGS off, fp32 (TF32 off), eff-batch 64 (`-b 16 --gpus 2 --accumulate_grad_batches 2`),
lr 2e-4, 867M each.
- **sedan** = the VP/CTS *transferred* model (numerator). NDS 0.5354 (best 0.5407).
- **suv / bus** = the CTS *oracle* denominators (target-native). NDS 0.5474 / 0.4182 (best 0.5564 / 0.4279).
- **suv** used `--gradient_clip_val 35` (the others did not — needed to stop a gradient-spike
  divergence on 2-GPU; noise-level effect on the converged result).

Not committed (size). Stable backup: `BEVDepth/archive_carla_results/` (ckpts + cts + provenance README).

> Old undertrained baseline was NDS 0.3278; these retrains are +0.21 NDS. wandb: `Robust_Ex/BEVDepth-CARLA`.

## vp/  (committed) — viewpoint robustness, sedan model

631-cell yaw/pitch/roll perturbation grid (signed ±{4..20}°), `RRS = NDS_cell / NDS_Normal`.
`NDS_Normal = 0.5324` (frames-per-scene=16 subset, 768 samples). Conditions: ER (extrinsic
only) / **VR (image only, primary)** / CR (both). `mRRS` = mean over per-camera cells (one
cam perturbed); `RRSALL` = all-6-cams perturbed.

> **carla_VR frame-2N fix applied** (commits `3551b46` / `3e987d1`). carla_geobev is a
> ½-rate relabel of carla_VR (geobev frame N == carla_VR frame **2N**, verified: geobev
> 0269-150 == VR-300, pixel-diff 17.7 vs 40 at VR-150). The old `vr_image_path` joined
> frame N, feeding the image-swap **VR/CR** conditions a *different scene's* image than the
> GT — inflating their collapse. **ER + Normal are unaffected** (no image swap). VR/CR were
> re-run with the correct frame; the pre-fix numbers are kept in
> `eval_vp_BUGGY_framemismatch.json` for provenance.

| condition | mRRS (per-cam) | **RRSALL (all-cam)** | mVRS | _(pre-fix RRSALL)_ |
|---|---|---|---|---|
| ER | 0.947 | 0.653 | 0.800 | _0.653 (unchanged)_ |
| **VR (primary)** | 0.915 | **0.328** | 0.621 | _0.151_ |
| CR | 0.942 | **0.492** | 0.717 | _0.240_ |

Perturbing ONE camera's viewpoint keeps ~92% NDS (the other 5 cams compensate); perturbing
ALL 6 is where the conditions separate, and the failure is **axis-specific** (RRSALL all-cam):

| axis | ER | VR (uninformed) | CR (calibrated) |
|---|---|---|---|
| yaw | 0.43 | 0.40 | **0.96** |
| **pitch** | 0.69 | **0.14** | **0.13** |
| roll | 0.84 | 0.44 | 0.38 |

**Yaw is a *consistency* problem**: when the image and the extrinsic agree (CR) the model is
essentially robust (0.96); a yaw mismatch in either alone (ER/VR ≈ 0.4) breaks it. **Pitch
and roll are *content* problems calibration can't fix** (CR ≈ VR): tilting/rolling the cameras
changes what the image shows (objects to the frame edge, altered ground plane) and breaks the
LSS depth lift even with a correct camera pose — **pitch is the worst, ~0.13 regardless**. So
the corrected headline is not "image perturbation collapses everything," but "yaw degradation
is a calibration problem, while pitch (and to a lesser extent roll) is the model's blind spot."

## qual/  (gitignored — `*.jpg`, local only)

10 highest in-pc-range (vis≥2) distinct-scene val samples × sedan/suv/bus × 6 views, GT (green)
vs pred>0.3 (red) 3D boxes projected on the original images (in-range ±51.2 m, FOV-culled).
Shows bus's much lower recall vs sedan/suv on dense scenes. 77 MB local.

## qual_cond/  (committed) — condition qualitative, one scene across all conditions

Same scene (`0269-0150`, 23 cars) for the sedan model: GT (green) + pred (red) projected on
the **display image** using that image's TRUE camera pose (variant pose for the tilted VR/CR
views, baseline for ER, sedan/target for CTS) — so GT lands on the objects and pred reveals
the degradation as a shift. The model RUNS on each condition's actual input (perturbed/swapped
img+ext); pred is then drawn in the display frame. Uses the carla_VR frame-2N fix.
`thr{0.3,0.5}/`: **vp/** roll·pitch·yaw @ +20° all-cam × ER/VR/CR; **cts_{suv,bus}/**
NORMAL/EXT/IMG/CAL. 34 grids. Script: `BEVDepth/_viz_cond_qual.py` (local).

Visual story (matches the vp/ table): VR (model uninformed) drifts pred to the horizon;
CR on **yaw** recovers tight boxes (pred 19/24); **pitch** collapses regardless (pred ~4/24).
