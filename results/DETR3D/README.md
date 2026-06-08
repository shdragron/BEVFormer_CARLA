# DETR3D-R50 CARLA — results & checkpoints

The 6th/last detector (3rd sparse, after CAPE/PETRv2). Query-based **Sparse** view
transform (Detr3DCrossAtten samples multi-view features at projected 3D ref points),
R50 ImageNet (no FCOS3D), full-res **1600×900**, no-DN (native), FPN kept.

## Oracle val NDS (each model on its own platform's val, full 3792, 6-class)

| platform | NDS | mAP6 |
|---|---|---|
| sedan | 0.5336 | — |
| suv | 0.5602 | 0.5150 |
| **bus** | **0.6019** | 0.5849 |

**bus > suv > sedan** — the *reverse* of BEVDepth (suv 0.556 > sedan 0.541 > bus 0.428).
DETR3D's full-res 1600×900 handles the higher bus mount well rather than degrading.

## cts/  (committed)

`eval_cts.{csv,json,summary.txt}` — cross-platform transfer (CTS): the **sedan**-trained
model deployed on **suv/bus**, **full 3792-frame val**, vis≥2 GT, 6-class NDS. Conditions
NORMAL / EXT / IMG / CAL (IMG = primary); `CTS = NDS_cond / P_TARGET` (target-native oracle).

| target | ORACLE (P_TARGET) | NORMAL | EXT | **IMG (primary)** | CAL |
|---|---|---|---|---|---|
| suv | 0.5602 | 0.5041 (0.900) | 0.2401 (0.429) | **0.1955 (0.349)** | 0.4308 (0.769) |
| bus | 0.6019 | 0.4660 (0.774) | 0.1848 (0.307) | **0.1593 (0.265)** | 0.2261 (0.376) |

(parenthetical = CTS ratio.) DETR3D transfers to **bus** better than BEVFormer (IMG CTS
bus 0.265 vs BEVFormer 0.180) — full-res robustness to the mount gap. CTS uses **geobev
images only → the carla_VR frame-2× bug does NOT apply**. Ran via the self-contained
`bev_det_benchmark/sparse/` bundle (`eval_cts_det.py --framework detr3d`) — **first
end-to-end validation of that bundle**: the oracle passes reproduce the direct
`tools/test.py` NDS exactly (suv 0.5602, bus 0.6019).

## vp/  (committed)

`eval_vp.{json,per_config.csv,summary.txt}` — viewpoint robustness (VP): the **sedan** model
under carla_VR perturbations, 768-sample subset (16/scene), 631-cell grid, **frame-2×-fixed
builder**. `RRS = NDS_cond / P_NORMAL` (Normal NDS 0.5368). Ran via the `sparse/` bundle
(`run_vp_detr3d.sh`, 2-GPU shard + merge, ~8.8 h). all-cam RRS:

| condition | roll | pitch | yaw | **ALL** | per-cam ALL |
|---|---|---|---|---|---|
| EXT (extrinsic) | 0.549 | 0.253 | 0.513 | 0.438 | 0.909 |
| **IMG (image, primary)** | 0.512 | 0.260 | 0.493 | **0.422** | 0.908 |
| CAL (consistent) | 0.789 | 0.825 | 0.921 | **0.845** | 0.975 |

**EXT ≈ IMG** + **strong CAL recovery (0.845, highest of any model)** — DETR3D (Sparse/query)
patterns with BEVFormer (Backward/Dense), *not* with the Depth/LSS models: no explicit depth →
image and extrinsic perturbation are symmetric and consistency restores all axes (incl. pitch
0.825). See `../VP_CROSS_MODEL_ANALYSIS.md` Finding D. (summary header mislabels "BEVDet NDS" —
a cosmetic string in the shared sparse driver; the data is DETR3D.)

## ckpts/  (gitignored — `*.pth`, local only)

24-epoch oracles live in `detr3d/work_dirs/detr3d_carla_{sedan,suv,bus}/epoch_24.pth`
(403 MB each). Not copied here (size); preserve separately if needed.
