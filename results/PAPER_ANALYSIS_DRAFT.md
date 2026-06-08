# Analysis (paper draft) — camera-geometry robustness of camera-only 3D detectors

> Draft of the experiments/analysis section for the CARLA-geobev robustness benchmark.
> All numbers are frame-2×-fixed; the mechanism classes are code-verified (5-agent audit, HIGH
> confidence). PETRv2 is trained/eval-pending and shown as a prediction. Figures F1/F2 are
> describable from the tables below (generation offered separately).

## Overview — two orthogonal robustness axes

We evaluate six camera-only 3D detectors on the CARLA-geobev benchmark under a single fair-comparison
protocol (augmentation off, EMA/CBGS/fp16 off, ResNet-50 ImageNet backbone without FCOS3D
pre-training, visibility ≥ 2 GT, 6-class NDS). Two studies probe **two complementary axes of
camera-geometry robustness**:

- **VP (viewpoint robustness, within-platform):** the sedan model is evaluated under controlled
  perturbations of its own camera geometry. This isolates **how the view-transform couples to the
  camera extrinsic.**
- **CTS (cross-platform transfer):** the sedan model is deployed on the suv and bus platforms, whose
  cameras sit at different mount heights. This isolates **how a model's depth/scale prior couples to
  the deployment viewpoint.**

The two axes are *not* redundant: as we show in §C, the architectural property that governs VP
(whether the extrinsic gates feature sampling) is independent of the one that governs CTS (whether
the model relies on a learned depth prior), and no single detector is robust on both.

**Table 1. Detectors and the code-verified robustness mechanism.**

| # | model | view-transform | input | robustness mechanism (verified) |
|---|---|---|---|---|
| 1 | BEVDet | Forward / LSS | 256×704 | extract-then-place |
| 2 | BEVDepth | Forward / LSS (explicit depth) | 256×704 | extract-then-place |
| 3 | BEVFormer | Backward / dense BEV | 800×450 | extrinsic-gates-sampling |
| 4 | CAPE | Sparse / query (camera-view PE) | 512×1408 | extract-then-place |
| 5 | PETRv2 | Sparse / query (3D PE) | 512×1408 | *(predicted) extract-then-place* — pending |
| 6 | DETR3D | Sparse / query (projective) | 1600×900 | extrinsic-gates-sampling |

We distinguish two mechanisms by **how the camera extrinsic enters the forward pass** (verified by
auditing each model's view-transform code):
- **extrinsic-gates-sampling** — the extrinsic (`lidar2img`) projects 3D query/reference points into
  the images and *samples* features at those pixel locations (BEVFormer's deformable
  cross-attention, DETR3D's `grid_sample`). Image features are extracted independently of the
  extrinsic; the extrinsic decides *where to look*.
- **extract-then-place** — 2D image features are extracted by the backbone *independently of the
  extrinsic*, and the geometry is applied only afterward, as a position embedding (CAPE's camera-view
  PE) or as an LSS depth-splat into BEV (BEVDet/BEVDepth). The extrinsic *places* already-extracted
  features.

Crucially this split is **orthogonal to the Forward/Backward/Sparse taxonomy and to the presence of
explicit depth**: it cuts *through* the Sparse class (DETR3D = gates-sampling; CAPE/PETRv2 =
extract-then-place), and CAPE — which has no depth network — patterns with the LSS depth models.

---

## A. VP — viewpoint robustness

**Protocol.** We perturb the camera geometry along three rotation axes (roll, pitch, yaw) at signed
magnitudes ±{4,8,12,16,20}°, under three conditions: **EXT** (extrinsic perturbed, image clean),
**IMG** (image rendered from the perturbed viewpoint, extrinsic kept = primary), and **CAL** (both
perturbed consistently). We report **RRS = NDS_cond / NDS_clean** on a frozen 16-frame/scene subset,
in two scopes: **all-cam** (all six cameras perturbed coherently — the discriminating protocol) and
**per-cam** (one camera perturbed). All-cam ALL is the mean over the three axes.

**Table 2. VP all-camera RRS (ALL = mean over roll/pitch/yaw), grouped by mechanism.**

| model | mechanism | EXT | IMG (primary) | CAL | per-cam (ALL) |
|---|---|---|---|---|---|
| CAPE | extract-then-place | 0.811 | 0.407 | 0.560 | 0.92 |
| BEVDepth | extract-then-place | 0.652 | 0.328 | 0.492 | 0.92 |
| BEVDet | extract-then-place | 0.610 | 0.364 | 0.521 | 0.92 |
| DETR3D | extrinsic-gates-sampling | 0.438 | 0.422 | 0.845 | 0.91 |
| BEVFormer | extrinsic-gates-sampling | 0.428 | 0.426 | 0.777 | 0.91 |

**A.1 The mechanism splits EXT vs IMG.** For **gates-sampling** models, perturbing the *extrinsic*
or perturbing the *image* does equal damage (EXT≈IMG: BEVFormer 0.428≈0.426, DETR3D 0.438≈0.422),
because the extrinsic gates the very sampling that fetches features — corrupting either corrupts the
sampled features. For **extract-then-place** models, EXT ≫ IMG (CAPE 0.811≫0.407, BEVDepth
0.652≫0.328, BEVDet 0.610≫0.364): under EXT the backbone extracts features from the *clean* image
and only the placement is wrong, whereas under IMG the *tilted image corrupts extraction itself*.
The classic "depth double-edge" is thus really an **extract-then-place edge** — CAPE, which has no
depth, exhibits the largest gap (it is in fact the most extrinsic-robust model overall, a direct
consequence of its camera-view position embedding being designed to decouple from the extrinsic).

**A.2 Consistency (CAL) recovery follows the mechanism; CAL-pitch is the cleanest single
discriminator.** Under a *consistent* perturbation the gates-sampling models re-align their sampling
end-to-end and recover, including the hardest axis (CAL-pitch: BEVFormer 0.661, DETR3D 0.825),
whereas the extract-then-place models cannot — a consistent extrinsic cannot undo a tilt already
baked into feature extraction (CAL-pitch: BEVDepth 0.132, CAPE 0.182, BEVDet 0.254). [**Figure F1**:
a per-model heatmap of RRS over (EXT/IMG/CAL × roll/pitch/yaw) makes the CAL-pitch collapse-vs-recover
the most visually salient feature.] All models recover CAL-*yaw* (≈0.92–0.98): yaw rotates about the
vertical and preserves the ground-plane appearance, so only the horizon-tilting axes (roll, pitch)
defeat extract-then-place under consistency. *Within* the extract-then-place family, a second-order
gradient appears — stronger depth supervision → more pitch-locked (CAL-pitch: BEVDepth-explicit 0.132
< BEVDet-implicit 0.254; CAPE-no-depth 0.182).

**A.3 At the metric level the VP split is a recall story.** Decomposing NDS into mAP (recall) and the
true-positive errors (mATE/mASE/mAOE) clarifies the failure mode (Table 3): a coherently-tilted
**image collapses recall for every architecture** (IMG mAP6-retention 0.14–0.19), while under **EXT
the mechanisms diverge at the recall level** — extract-then-place *keeps* recall (mAP6 0.40–0.48; the
clean image is still detected, only mis-placed), gates-sampling *loses* it (mAP6 0.15; mis-sampling
prevents the boxes from being proposed). This is the mechanistic root of EXT≫IMG: extrinsic error
preserves detection under extract-then-place but not under gates-sampling. CAL recovers recall for
gates-sampling (mAP6 0.67–0.75) but only partially for extract-then-place (0.37–0.41).

**Table 3. VP all-cam, NDS-RRS / mAP6-retention.**

| model | mechanism | EXT | IMG | CAL |
|---|---|---|---|---|
| BEVDepth | extract-then-place | 0.65 / 0.48 | 0.33 / 0.14 | 0.49 / 0.37 |
| BEVDet | extract-then-place | 0.61 / 0.40 | 0.36 / 0.16 | 0.52 / 0.38 |
| CAPE | extract-then-place | — | 0.41 / 0.16 | 0.56 / 0.41 |
| DETR3D | gates-sampling | 0.44 / 0.15 | 0.42 / 0.15 | 0.85 / 0.75 |
| BEVFormer | gates-sampling | 0.43 / 0.19 | 0.43 / 0.19 | 0.78 / 0.67 |

**A.4 Per-camera robustness is architecture-invariant (appendix).** Under single-camera perturbation
all models lose ≤ 10% RRS, and the *relative* camera importance is identical across architectures
(CAM_BACK most load-bearing, the right-side cameras least) — a property of the scene's object
distribution and 6-view redundancy, not of the view-transform. Consequently a 6-view-averaged score
(or a per-cam-heavy aggregate such as the "1/7" score) compresses the architectural differences; we
report the discriminating **all-camera** numbers in the main text and per-camera in the appendix.

---

## B. CTS — cross-platform transfer

**Protocol.** The sedan-trained model is deployed on suv/bus (full 3792-frame val), with conditions
NORMAL/EXT/IMG/CAL combining the sedan vs target image and extrinsic. We report **CTS = NDS_cond /
P_TARGET**, where P_TARGET is the target-native oracle (a model trained on, and evaluated on, the
target platform). IMG (target image, sedan extrinsic) is the primary condition.

**Table 4. CTS (NDS / P_TARGET); IMG is primary.**

| model | mechanism | suv EXT/IMG/CAL | bus EXT/IMG/CAL | IMG mean |
|---|---|---|---|---|
| CAPE | extract-then-place | 0.823 / 0.338 / 0.343 | 0.575 / 0.360 / 0.329 | **0.349** |
| DETR3D | gates-sampling | 0.429 / 0.349 / 0.769 | 0.307 / 0.265 / 0.376 | 0.307 |
| BEVFormer | gates-sampling | 0.451 / 0.372 / 0.719 | 0.251 / 0.179 / 0.401 | 0.276 |
| BEVDet | extract-then-place | 0.615 / 0.170 / 0.161 | 0.539 / 0.002 / 0.228 | 0.086 |
| BEVDepth | extract-then-place | 0.695 / 0.103 / 0.057 | 0.299 / 0.001 / 0.166 | 0.052 |

**B.1 Cross-platform transfer is governed by depth, not by the VP mechanism.** The explicit/categorical
**depth models collapse on cross-platform image** (BEVDet/BEVDepth bus-IMG ≈ 0.001–0.002): a monocular
depth network trained on the sedan mount predicts the wrong depth for the elevated bus viewpoint, so
features are lifted to the wrong BEV ranges. The depth-free models transfer far better — **CAPE is the
strongest transferer (IMG 0.349) and is platform-robust** (bus 0.360 ≈ suv 0.338), with DETR3D and
BEVFormer in between. The transfer failure carries a depth-specific TP fingerprint: among the few
surviving detections, the depth models are additionally mis-*sized* (CTS-IMG mASE 0.81–0.87 vs
0.31–0.34 for the depth-free models).

---

## C. Synthesis — the two axes are independent

The two studies separate the detectors along **two architectural properties that are independent of
each other**:

- **VP** separates by *mechanism* — gates-sampling (EXT≈IMG, recovers under CAL) vs extract-then-place
  (EXT≫IMG, pitch-locked under CAL).
- **CTS** separates by *depth reliance* — depth models (collapse on cross-platform IMG) vs depth-free
  (transfer).

These are not the same partition. [**Figure F2**: a 2×2 of (VP mechanism) × (uses explicit depth),
with detectors placed.] Three of the four cells are populated:

| | depth-free | uses depth |
|---|---|---|
| **gates-sampling** | BEVFormer, DETR3D | — |
| **extract-then-place** | **CAPE** | BEVDet, BEVDepth |

**CAPE is the decisive evidence**: it is *extract-then-place* (so it shares the VP signature of the
LSS depth models — EXT≫IMG, pitch-locked CAL) yet *depth-free* (so it is the best cross-platform
transferer). A model's VP behaviour therefore does not predict its CTS behaviour, and vice versa.
Practically: the consistency-recovering gates-sampling models (BEVFormer, DETR3D) are the safest under
*self-calibration drift* (CAL), the extrinsic-robust extract-then-place models (esp. CAPE) are safest
under *pure extrinsic error*, and depth-free models are required for *cross-platform deployment* —
no current detector wins all three.

---

*Numbers: `results/{model}/{vp,cts}/`; per-cam/per-axis in `results/vp_percam_peraxis.tsv`; mechanism
detail in `results/VP_CROSS_MODEL_ANALYSIS.md`; consolidated tables in `results/BENCHMARK_SUMMARY.md`.
VP = 768-frame subset (frames-per-scene 16); CTS = full 3792-frame val. PETRv2 = train/eval pending.*
