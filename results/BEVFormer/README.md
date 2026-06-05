# BEVFormer-tiny CARLA — results & checkpoints

Organized artifacts for the BEVFormer-tiny CARLA geobev benchmark.

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
