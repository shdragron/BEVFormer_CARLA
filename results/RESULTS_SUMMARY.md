# CARLA geobev — RESULTS 총정리 (master summary)

Single entry point for all benchmark results. Two robustness studies on the CARLA geobev
per-vehicle data (sedan/suv/bus): **VP** = viewpoint robustness (sedan model, perturb
yaw/pitch/roll); **CTS** = cross-platform transfer (sedan model → suv/bus). Detection metric =
6-class NDS (vis≥2); segmentation = vehicle-occupancy IoU. Updated 2026-06-16.

- **mVRS = 1/7** = (6·mRRS_percam + RRSALL_allcam)/7 (per-cam ×6 + all-cam ×1), the headline.
- **all-cam** (perturb all 6 cams) = the *discriminating* scope; per-cam is ~0.9 for everyone.
- **RRS/CTS** are within-model ratios (vs the model's own P_NORMAL / target oracle) → resolution &
  temporal differences don't confound.

---

## 1. Detection benchmark — 7 detectors

P_NORMAL = 768-subset sedan NDS. VP cross-model table stays 768-matched (full≈subset, |Δ|≤~0.01).

| model | mechanism | P_NORM | VP 1/7 E/I/C | VP all-cam E/I/C | CTS-IMG suv/bus | oracle sed/suv/bus |
|---|---|---|---|---|---|---|
| CAPE | extract-then-place | 0.5508 | .943/.843/.890 | .811/**.407**/.560 | .338/.360 | .554/.596/.449 |
| BEVDepth | extract-then-place | 0.5324 | .905/.831/.878 | .652/**.328**/.492 | .103/.001 | .532/.547/.418 |
| BEVDet | extract-then-place | 0.5185 | .891/.833/.878 | .610/**.364**/.521 | .170/.002 | .519/.535/.373 |
| DETR3D | gates-sampling | 0.5368 | .842/.839/.956 | .438/**.422**/.845 | .349/.265 | .534/.560/.602 |
| BEVFormer | gates-sampling | 0.5051 | .831/.838/.935 | .428/**.426**/.777 | .372/.179 | .505/.503/.547 |
| DFA3D | gates-sampling ×depth | ~0.49 | .904/.861/.932 | .659/**.496**/.751 | .388/.288 | .489/.514/.556 |
| **PD-BEV** (DG remedy) | extract-then-place +PD | 0.5598 | .914/.889/.951 | .702/**.593**/.803 | .597/.009 | .560/.546/.606 |

- **VP all-cam IMG ranking** (the real signal): DFA3D .496 > BEVFormer .426 ≈ DETR3D .422 > CAPE .407 >
  BEVDet .364 > BEVDepth .328. **PD-BEV .593 tops all** (it's a debiasing *remedy*, trained native-384).
- **CTS-IMG (primary) mean**: CAPE .349 > DETR3D .307 > BEVFormer .276 > DFA3D .338/.288 > BEVDet .086 >
  BEVDepth .052. **PD-BEV suv .597** (10× its BEVDepth base) but **bus .009** (collapses — bus too far).
- PD-BEV/DFA3D use their native recipes (PD-BEV 384×704/[1,100,1]; DFA3D BEVFormer-tiny+DPT) — listed
  for completeness; the **fair-comparison matrix is the 5 (CAPE/BEVDepth/BEVDet/DETR3D/BEVFormer)** in
  `BENCHMARK_SUMMARY.md`. PETRv2 = pending.

## 2. Table 2 — "Do existing techniques close the gap?" (remedy comparison)

Per-task baseline + 3 remedies: train-time Extrinsic Aug. (both tasks), EAFormer (seg DG), PD-BEV
(det DG). mVRS % (1/7), CTS-Cal %; `\rc{%}{rawNDS}` where shown. **Detection (verified by me):**

| method | P_Normal | Ext | Img | Cal | CTS-Cal SUV | Bus |
|---|---|---|---|---|---|---|
| BEVDepth | 0.5354 | 90.2 (.483) | 82.8 (.443) | 87.4 (.468) | 5.7 (.031) | 16.6 (.069) |
| +Extrinsic Aug. | 0.5152 | 90.1 | 83.9 | 88.0 | 10.6 | 13.3 |
| **+PD-BEV** | **0.5605** | **91.4 (.512)** | **88.9 (.498)** | **95.1 (.533)** | **55.9 (.305)** | **41.8 (.254)** |

**Segmentation** (IoU; from `seg_vp_cts.tsv` / SEG team):

| method | P_Normal | Ext | Img | Cal | CTS-Cal SUV | Bus |
|---|---|---|---|---|---|---|
| CVT | 0.420 | 91.2 | 83.8 | 90.9 | 27.1 | 2.4 |
| +Extrinsic Aug. | 0.388 | 100.0 | 87.1 | 87.2 | 42.5 | 8.9 |
| +EAFormer | 0.447 | 92.5 | 85.8 | 91.7 | 49.2 | 11.6 |

Headline: **PD-BEV closes the detection gap** — every column up vs BEVDepth, **Cal flips above Ext**
(95.1>91.4 vs BEVDepth 87.4<90.2: perspective-debiasing lets the model *use* the correct calibration),
and **CTS-Cal jumps ~10× (SUV) / ~2.5× (Bus)** (its DG design goal). +Extrinsic Aug. barely moves
mVRS and even hurts P_Normal. (LaTeX rows + `\rc` raw NDS in `PDBEV/HANDOFF.md` / this file.)

## 3. Segmentation benchmark (6 models, full VP/CTS)

`seg_vp_cts.tsv` (normal IoU; mVRS EXT/IMG/CAL; all-cam; CTS suv/bus EXT/IMG/CAL). Models: CVT,
GaussianLSS, LaRa, LSS, PointBEV, SimpleBEV. See `SEGMENTATION_RESULTS.md` + `SECTION5_ANALYSIS_KR.md`.

## 4. Key findings (don't re-derive)

1. **VP mechanism dichotomy.** Robustness splits by whether the extrinsic *gates feature sampling*
   (BEVFormer/DETR3D/DFA3D: EXT≈IMG, CAL recovers strongly) vs *extract-then-place* (CAPE/BEVDet/
   BEVDepth: EXT≫IMG, CAL stays collapsed). NOT Forward/Backward/Sparse; code-verified.
   `vp-cross-model-mechanism-finding` / `VP_CROSS_MODEL_ANALYSIS.md`.
2. **EXT under-reports viewpoint damage** — extrinsic-only perturbation ≠ re-rendered image (IMG);
   the two groups separate only under all-cam IMG.
3. **Cross-platform transfer ≠ NORMAL ≠ EXT.** Predict deployability from re-rendered all-cam IMG / CTS,
   not clean accuracy or extrinsic-only scores.
4. **Remedies (Table 2):** PD-BEV (det DG) substantially closes the gap, esp. CTS-Cal; Extrinsic Aug.
   is weak for both tasks; EAFormer (seg DG) helps seg moderately.

## 5. Where everything lives

| | |
|---|---|
| 5-detector deep-dive (matrix, per-cam×axis) | `BENCHMARK_SUMMARY.md` |
| VP cross-model mechanism analysis | `VP_CROSS_MODEL_ANALYSIS.md`, `VP_FULLVAL_SUMMARY.md` |
| Paper §5 analysis (KR, defines 1/7 mVRS) | `SECTION5_ANALYSIS_KR.md` |
| Segmentation | `SEGMENTATION_RESULTS.md`, `seg_vp_cts.tsv` |
| Cross-model VP ground-truth (machine-readable) | `_vp_xmodel_ground_truth.json` |
| Per-model (vp/ cts/ README, ckpts gitignored) | `BEVFormer/ BEVDepth/ BEVDet/ CAPE/ DETR3D/ DFA3D/ PDBEV/` |
| PD-BEV detail + machine-move handoff | `PDBEV/README.md`, `PDBEV/HANDOFF.md` |
| Method paper (LatentCalib) | `METHOD_PAPER_SPEC_KR.md`, `ROBUST_MODEL_PROPOSAL_KR.md` |

> Numbers here are 768-subset for the VP cross-model table (matched); PD-BEV/DFA3D/per-model CTS are
> full-3792. PD-BEV is the only model with both subset & full VP committed (identical to the decimal).
