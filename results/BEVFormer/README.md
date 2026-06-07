# BEVFormer-tiny CARLA — results & checkpoints

Organized artifacts for the BEVFormer-tiny CARLA geobev benchmark.

## vp/  (committed)

`eval_vp.{json,per_config.csv,summary.txt}` — viewpoint-robustness (VP): the **sedan**
model under carla_VR camera-geometry perturbations, vis≥2 GT, 6-class NDS, **16/scene
(768-sample) subset**, protocol *both* (per-cam + all-cam), 631 cells. Robustness =
`RRS/VRS = NDS_cond / P_NORMAL` (P_NORMAL = clean Normal NDS = **0.5051**, mAP6 0.4449).

| condition | mRRS (per-cam) | RRSALL (all-cam) | **mVRS** |
|---|---|---|---|
| ER / EXT (extrinsic) | 0.898 | 0.428 | 0.663 |
| **VR / IMG (image, primary)** | 0.907 | **0.427** | 0.667 |
| CR / CAL (consistent) | 0.961 | **0.777** | 0.869 |

**[2026-06-07 frame-fix]** VR/CR re-run after correcting the **carla_VR frame-2× bug**
(geobev frame N == carla_VR frame 2N; the old builders loaded a *different scene* for the
VR/CR image swaps → the collapse was inflated). ER is valid (no image swap). Corrected story:
- **EXT ≈ IMG** (all-cam 0.428 ≈ 0.427, axis-by-axis too) — extrinsic and image perturbation
  hurt **equally**; the old "image ≫ extrinsic" was purely the frame artifact.
- **CAL ≫ EXT/IMG** (0.777 vs 0.43) — damage is image–extrinsic **inconsistency**, not which
  side; consistency restores it (yaw nearly fully, 0.939).
- **Pitch is the worst axis** everywhere (EXT/IMG pitch ~0.17–0.19, magnitude-gated:
  VR pitch 0.46 @±4 → 0.06 @±20); roll/yaw survive (~0.53–0.56). Even CAL pitch (0.661) <
  CAL yaw (0.939).

Per-cell breakdown in `eval_vp_per_config.csv` (buggy frame-N backed up as `*.buggy_frameN`).

## cts/  (committed)

`eval_cts.{json,csv,summary.txt}` — cross-platform transfer (CTS): the **sedan**-trained
model deployed on **suv/bus**, vis≥2 GT, 6-class NDS. Conditions NORMAL / EXT / IMG / CAL
(IMG = primary); denominator = target-native oracle (`CTS = NDS_cond / P_TARGET`).

| target | ORACLE (P_TARGET) | NORMAL | EXT | **IMG (primary)** | CAL |
|---|---|---|---|---|---|
| suv | NDS 0.5029 | 0.4766 | 0.2266 | **0.1872 (CTS 0.372)** | 0.3615 |
| bus | NDS 0.5471 | 0.4368 | 0.1371 | **0.0982 (CTS 0.180)** | 0.2192 |

## ckpts/  (gitignored — `*.pth`, local only)

`bevformer_tiny_carla_{sedan,suv,bus}_epoch24.pth` — 24 epochs, BEV grid 50×50,
single-frame, vis≥2, 383M each.
- **sedan** = the VP/CTS *transferred* model (numerator).
- **suv / bus** = the CTS *oracle* denominators (target-native).

Not committed (size). Stable backup: `/NHNHOME/WORKSPACE/0526040099_A/jeongtae/preserved_bevformer_carla/`
(sha256-verified).
