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
> VP (631-cell viewpoint robustness) results will be added to `vp/` when the run completes.
