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

## ckpts/  (gitignored — `*.pth`, local only)

`bevdet-r50-carla_{sedan,suv,bus}_epoch24.pth` — 24 epochs, BEV grid 128×128, single-frame
**BEVDetDepth** (LSS + explicit DPT dense-depth supervision), vis≥2 GT, no EMA / no CBGS,
batch 128 (64×2) @ lr 8e-4, fp32, ~596 MB each.
- **sedan** = the VP/CTS *transferred* model (numerator).
- **suv / bus** = the CTS *oracle* denominators (target-native).

Not committed (size). Stable backup:
`/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev/_proj_check/bevdet_carla_backup/ckpts/`
(md5-verified).
