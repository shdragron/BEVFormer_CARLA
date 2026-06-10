# BEVDet-R50 CARLA — results & checkpoints

Organized artifacts for the BEVDet-R50 (single-frame, depth-supervised) CARLA geobev benchmark.

## cts/  (committed)

`eval_cts.{json,csv,summary.txt}` — cross-platform transfer (CTS): the **sedan**-trained
model deployed on **suv/bus**, vis≥2 GT, 6-class NDS. Conditions NORMAL / EXT / IMG / CAL
(IMG = primary); denominator = target-native oracle (`CTS = NDS_cond / P_TARGET`).

| target | ORACLE (P_TARGET) | NORMAL | EXT | **IMG (primary)** | CAL |
|---|---|---|---|---|---|
| suv | NDS 0.5348 | 0.4884 | 0.3290 | **0.0907 (CTS 0.170)** | 0.0860 |
| bus | NDS 0.3729 | 0.4565 | 0.2011 | **0.0008 (CTS 0.002)** | 0.0849 |

Notes: bus NORMAL (0.4565) > bus ORACLE (0.3729) — the sedan model on the un-perturbed bus
platform beats the bus-native oracle (bus is the hardest viewpoint to *train*, so its oracle
is weak); IMG/CAL transfer collapses (bus IMG ≈ 0 — image-domain shift is catastrophic for the
camera-only LSS pipeline). Full per-class detail in `eval_cts.json` (`rows[].metrics`); any
subset metric is recomputable via `bev_det_benchmark/recompute_subset_metric.py`.

## vp/  (committed) — viewpoint robustness, sedan model

631-cell yaw/pitch/roll perturbation grid (signed ±{4..20}°), `RRS = NDS_cell / NDS_Normal`.
`NDS_Normal = 0.5166` (mAP6 0.4717; **full 3792-frame val** — `eval_vp.*` are now the full run).
The fps-16 **768-subset** (NDS_Normal 0.5185) is preserved as `eval_vp.subset768.*`; the
cross-model table stays matched at 768 for every model, and full≈subset (|ΔRRS| mean 0.005,
≤0.01). Conditions: ER (extrinsic only) / **VR (image only, primary)** / CR (both). `mRRS` = mean
over per-camera cells (one cam perturbed); `RRSALL` = all-6-cams perturbed.

| condition | mRRS (per-cam) | **RRSALL (all-cam)** | mVRS |
|---|---|---|---|
| ER (extrinsic) | 0.939 | 0.610 | 0.774 |
| **VR (image, primary)** | 0.908 | **0.360** | 0.634 |
| CR (both) | 0.935 | 0.512 | 0.724 |

Per-axis (all-cam RRS, full 3792): VR yaw 0.417 / **pitch 0.212 (worst)** / roll 0.450; CR yaw
0.944 / pitch 0.243 / roll 0.349 — so **pitch is the worst axis, yaw the most robust**. Perturbing
ONE camera keeps ~91–94% NDS (the other 5 compensate); all-6 drops to VR 0.36 / CR 0.51 / ER 0.61.
Image (VR) still hurts somewhat more than extrinsic (ER) all-cam, but far less catastrophically
than first measured.

> **Frame-index fix (2026-06-07, commit 3551b46).** carla_VR is rendered at 2× the geobev frame
> rate (geobev frame N ≡ carla_VR frame 2N; verified pixel-diff 17.7 @2N vs 40 @N), but
> `vr_image_path` used frame N — so the VR/CR (image-swap) cells were fed a DIFFERENT scene's
> image than the GT, inflating the apparent collapse (old VR RRSALL **0.190**, CR **0.240**).
> Fixed (`int(frame)*2`) and re-run (`bevdet_sedan_fix`): VR 0.190→**0.364**, CR 0.240→**0.521**.
> ER and NDS_Normal are unchanged (Normal/ER use geobev images → unaffected), which confirms the
> fix touched only the image-swap path.

## ckpts/  (gitignored — `*.pth`, local only)

`bevdet-r50-carla_{sedan,suv,bus}_epoch24.pth` — 24 epochs, BEV grid 128×128, single-frame
**BEVDetDepth** (LSS + explicit DPT dense-depth supervision), vis≥2 GT, no EMA / no CBGS,
batch 128 (64×2) @ lr 8e-4, fp32, ~596 MB each.
- **sedan** = the VP/CTS *transferred* model (numerator).
- **suv / bus** = the CTS *oracle* denominators (target-native).

Not committed (size). Stable backup:
`/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev/_proj_check/bevdet_carla_backup/ckpts/`
(md5-verified).

## qual/  (gitignored — `*.jpg`, local only)

10 highest in-pc-range (vis≥2) distinct-scene val samples × sedan/suv/bus × 6 views, GT (green)
vs native-model pred (red) 3D boxes projected on the original images (EGO frame via
`inv(sensor2ego)`, in-range ±51.2 m, FOV-culled). One set per score threshold (`thr0.3/`,
`thr0.5/`), `{veh}_scene-XXXX-frame-YYYY.jpg`. Built by `bev_det_benchmark/qual_render_6view.py`,
mirroring `BEVDepth/_viz_qual.py`. ~122 MB local.

Coordinate frames are verified (scene-0269: pred red wraps the parked cars, overlapping GT
green). NOTE: `CarlaNuScenesDataset` reorders infos on load, so `test.py` predictions are in
`dataset.data_infos` order, NOT the raw pkl order — the renderer maps each pred to its frame
by **sample token** so GT/image/pred line up. (Skipping this draws another scene's boxes on
the image, which looks like spurious detections but is purely a frame-alignment bug.)
