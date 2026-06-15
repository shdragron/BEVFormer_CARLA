# PD-BEV CARLA — results

PD-BEV (**Generalizable-BEV**, AAAI'25, "Towards Generalizable Multi-Camera 3D Object
Detection via Perspective Debiasing") on the CARLA geobev benchmark. PD-BEV is a
domain-generalization training scheme on **BEVDepth** (train-only aux heads:
`img_aux` / `bev_img_aux` / `BEV_aux`); inference is plain `BEVDepth_DG` loading the DG
checkpoint (aux heads dropped, `strict=False`). The three platform models are the
`+PD-BEV` detection variant for the robustness-baselines table (vs the BEVDepth base).

## Recipe — authors-native 384 (NOT BEVDepth-matched 256)

All three trained at **input 384×704 + depth grid [1.0, 100.0, 1.0]** (AdamW 2e-4,
24 ep, step[19,23], eff-batch 64 = 16×2gpu×accum2, fp32/TF32-off, CBGS/EMA off,
PD-BEV native aug). The BEVDepth-matched 256×704 + [2,58,0.5] regime **collapses** PD-BEV
from scratch on the tall platforms (suv/bus): the depth loss supervises only GT-box-center
*virtual* depths (~5% of pixels), so a truncated grid + top-crop leaves 95% of depth pixels
anchored solely by detection gradients → softmax collapses to bin 1 → ego-ring BEV features
→ NDS 0. Native 384 keeps ~88% of virtual-depth targets in-bin and trains cleanly.
sedan tolerates 256 too (low 1.60 m camera), but all three use 384 for a unified recipe.

## Oracles (6-class NDS, vis≥2, own platform)

| platform | recipe | NDS | mAP6 |
|---|---|---|---|
| **sedan384** | native-384 | **0.5601** | 0.5683 |
| **suv** | native-384 | **0.5464** | 0.5555 |
| **bus** | native-384 | **0.6064** | 0.6291 |

sedan384 is +0.043 NDS over the old 256/[2,58] sedan (0.5170). Bus is the strongest
(elevated +20°-pitch cameras see more ground). wandb: `Robust_Ex/PDBEV-CARLA`.

## cts/  (committed) — cross-platform transfer

The **sedan384** model deployed on suv/bus, target GT (vis≥2), `CTS = NDS_cond / P_TARGET`
(target-native oracle). NORMAL/EXT/IMG/CAL.

| target | ORACLE | NORMAL | EXT | IMG (primary) | CAL |
|---|---|---|---|---|---|
| suv | 0.5464 | 0.5281 (.966) | 0.4004 (.733) | 0.3263 (**.597**) | 0.3053 (.559) |
| bus | 0.6064 | 0.4787 (.789) | 0.2288 (.377) | 0.0054 (**.009**) | 0.2537 (.418) |

PD-BEV's perspective-debiasing **massively improves cross-platform transfer** over the
BEVDepth base (suv CTS-IMG 0.103→0.597, CTS-CAL 0.057→0.559; bus CTS-CAL 0.166→0.418).
Transfer still degrades with platform distance — suv (2.35 m, 0° pitch) ≫ bus (2.87–4.08 m,
+20°); bus-IMG collapses (sedan extrinsic on a bus-viewpoint image is geometrically
unrecoverable), but the correct bus extrinsic at CAL recovers it (0.009→0.418).

## vp/  (committed) — viewpoint robustness, sedan384 model

631-cell yaw/pitch/roll grid (signed ±{4..20}°), `RRS = NDS_cell / NDS_Normal`.
Conditions ER (extrinsic) / **VR (image, primary)** / CR (both). Headline = **1/7 mVRS**
= (6·mRRS_percam + RRSALL_allcam)/7 (paper Table-2 metric). carla_VR frame-2N fix is in
the shared `build_condition_pkls_bevdet` builder.

Two runs are present: **`eval_vp.*` = full-3792** (fps-79, NDS_Normal **0.5605**, the
Table-2 source) and **`eval_vp.subset768.*` = fps-16 768-matched** (NDS_Normal 0.5598, the
cross-model subset). They agree to the decimal (full confirmed the interim):

| | 1/7 mVRS Ext | Img | Cal |
|---|---|---|---|
| BEVDepth (base) | 90.2 | 82.8 | 87.4 |
| **PD-BEV (full)** | **91.4** | **88.9** | **95.1** |
| PD-BEV (subset768) | 91.4 | 88.9 | 95.1 |

PD-BEV improves all three; notably **Cal flips above Ext** (95.1 > 91.4) where BEVDepth has
Cal < Ext (87.4 < 90.2) — perspective-debiasing lets the model use the correct calibration
instead of staying collapsed by the baked-in tilt. (Full per-condition: ER mRRS/RRSALL
0.949/0.702, VR 0.938/0.593, CR 0.975/0.803.)

## Eval pipeline (two-env, since CarlaNuScenesDataset NDS lives in bevdet-b200)

- `bev_det_benchmark/pdbev_vp_infer.py` (pdbev-b200): load `BEVDepth_DG` once, loop the 631
  VP cells (reusing `build_condition_pkls_bevdet.make_vp_infos`), dump per-cell dets by token.
- `bev_det_benchmark/pdbev_vp_score.py` (bevdet-b200): verified `CarlaNuScenesDataset` 6-class
  NDS on the same frozen subset, GT cached once, cell-sharded for parallel CPU scoring;
  emits the 1/7 mVRS Table-2 row.
- `bev_det_benchmark/pdbev_cts_run.sh` + `pdbev_cts_format.py`: CTS condition pkls
  (`make_cts_pkl`) → dump (sedan384) → target-GT NDS → CTS table.

ckpts (`work_dirs/pdbev-r50-cbgs-CARLA-dg-{sedan384,suv,bus}/epoch_24.pth`) are gitignored
(local). Configs live in the PD-BEV_CARLA repo (`Generalizable-BEV`).
