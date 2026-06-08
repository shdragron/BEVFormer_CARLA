# CARLA geobev camera-only 3D detection — benchmark summary (VP + CTS)

6 detectors under a fair-comparison matrix (aug OFF, EMA/CBGS/fp16 OFF, R50 ImageNet no-FCOS3D,
vis≥2 GT, 6-class NDS). **5 evaluated; PETRv2 = trained/eval pending.**

## Architecture matrix (+ code-verified robustness mechanism)

| # | model | view-transform | key settings | input | **robustness mechanism** (verified) |
|---|---|---|---|---|---|
| 1 | BEVDet | Forward (LSS) | BEV 128² / DPT-depth / depth=categorical | 256×704 | **extract-then-place** |
| 2 | BEVDepth | Forward (LSS) | BEV 128² / DPT-depth / depth=explicit LiDAR | 256×704 | **extract-then-place** |
| 3 | BEVFormer | Backward (Dense) | BEV 50² | 800×450 | **extrinsic-gates-sampling** |
| 4 | CAPE | Sparse (query) | DN kept / camera-view PE | 512×1408 | **extract-then-place** |
| 5 | PETRv2 | Sparse (query) | DN kept / 3D PE | 512×1408 | *(predict extract-then-place)* — **pending** |
| 6 | DETR3D | Sparse (query) | no-DN native / projective sampling | 1600×900 | **extrinsic-gates-sampling** |

**The robustness mechanism does NOT follow Forward/Backward/Sparse.** It is whether the camera
extrinsic *gates feature sampling* (BEVFormer, DETR3D — project query points into images & sample)
or features are *extracted extrinsic-independently then placed* (CAPE PE, BEVDet/BEVDepth LSS splat).
**It splits the Sparse class**: DETR3D = gates-sampling, CAPE/PETRv2 = extract-then-place. Verified by
a 5-agent code audit (all HIGH confidence). Resolution/temporal are kept as native (not equalized);
RRS/CTS are within-model ratios so they don't confound the robustness comparison.

## VP — viewpoint robustness (sedan model, 768-subset, RRS = NDS/P_NORMAL)

Two scopes: **all-cam** (perturb all 6 — the discriminating one) and **1/7** (per-cam as 6 cams +
all-cam as a 7th, the headline single score). Conditions EXT/IMG/CAL, ALL = mean over roll/pitch/yaw.

| model | mech | P_NORM | EXT all/IMG all/CAL all | EXT 1/7 / IMG 1/7 / CAL 1/7 | CAL-pitch (all) |
|---|---|---|---|---|---|
| CAPE | extract-then-place | 0.5508 | 0.811 / 0.407 / 0.560 | 0.943 / 0.843 / 0.890 | 0.182 |
| BEVDepth | extract-then-place | 0.5324 | 0.652 / 0.328 / 0.492 | 0.905 / 0.831 / 0.878 | 0.132 |
| BEVDet | extract-then-place | 0.5185 | 0.610 / 0.364 / 0.521 | 0.891 / 0.833 / 0.878 | 0.254 |
| DETR3D | gates-sampling | 0.5368 | 0.438 / 0.422 / 0.845 | 0.842 / 0.839 / 0.956 | 0.825 |
| BEVFormer | gates-sampling | 0.5051 | 0.428 / 0.426 / 0.777 | 0.831 / 0.838 / 0.935 | 0.661 |

### VP ranking — **by 1/7 (IMG primary), as requested**
1. **CAPE 0.843**  2. DETR3D 0.839  3. BEVFormer 0.838  4. BEVDet 0.833  5. BEVDepth 0.831
*(overall 1/7 = mean EXT/IMG/CAL: CAPE 0.892 > DETR3D 0.879 > BEVDepth 0.871 > BEVFormer 0.868 > BEVDet 0.867)*

> ⚠️ **The 1/7 ranking is nearly flat (0.831–0.843)** — per-cam (6/7 weight) is ~0.9 for everyone, so
> 1/7 compresses the architecture differences. The real signal is **all-cam IMG**:
> **BEVFormer 0.426 ≈ DETR3D 0.422 > CAPE 0.407 > BEVDet 0.364 > BEVDepth 0.328.**

## CTS — cross-platform transfer (sedan→suv/bus, full 3792 val, CTS = NDS/P_TARGET-oracle)

| model | suv: P / EXT / IMG★ / CAL | bus: P / EXT / IMG★ / CAL | IMG mean |
|---|---|---|---|
| CAPE | 0.596 / 0.823 / **0.338** / 0.343 | 0.449 / 0.575 / **0.360** / 0.329 | **0.349** |
| DETR3D | 0.560 / 0.429 / **0.349** / 0.769 | 0.602 / 0.307 / **0.265** / 0.376 | **0.307** |
| BEVFormer | 0.503 / 0.451 / **0.372** / 0.719 | 0.547 / 0.251 / **0.179** / 0.401 | **0.276** |
| BEVDet | 0.535 / 0.615 / **0.170** / 0.161 | 0.373 / 0.539 / **0.002** / 0.228 | **0.086** |
| BEVDepth | 0.547 / 0.695 / **0.103** / 0.057 | 0.418 / 0.299 / **0.001** / 0.166 | **0.052** |

### CTS ranking — by IMG (primary) mean
1. **CAPE 0.349**  2. DETR3D 0.307  3. BEVFormer 0.276  4. BEVDet 0.086  5. BEVDepth 0.052

## Trends (경향성)

1. **VP mechanism dichotomy (the headline).** *extract-then-place* (CAPE, BEVDepth, BEVDet) → **EXT≫IMG**,
   CAL-pitch collapses (0.13–0.25), EXT-worst-axis = yaw. *gates-sampling* (BEVFormer, DETR3D) → **EXT≈IMG**,
   CAL recovers incl. pitch (0.66–0.83), EXT-worst-axis = pitch. CAL-pitch is the cleanest discriminator
   (collapse vs recover). **Not predicted by Forward/Backward/Sparse, nor by depth** — CAPE (no depth)
   patterns with the LSS models.
2. **CTS exposes a *depth-generalization* weakness, distinct from the VP axis.** On cross-platform IMG
   (target-viewpoint image), **the explicit/categorical depth models collapse hardest** — BEVDet/BEVDepth
   bus-IMG ≈ **0.001–0.002** (the monocular depth, trained on sedan mount, is wrong for the bus viewpoint).
   **CAPE (extract-then-place but no depth) transfers best (IMG 0.349)** and DETR3D/BEVFormer (no depth)
   are mid. So for CTS the liability is **depth**, not extract-then-place per se.
3. **Per-cam robustness is architecture-invariant** (~0.90–0.99 all models) — one perturbed camera is
   outvoted by 6-view fusion. All architecture signal is in **all-cam**; the 1/7 score (per-cam heavy)
   masks it.
4. **CAPE's camera-view PE shows up directly**: most EXT-robust model (EXT all-cam 0.811, roll/pitch
   ~0.96–0.98) — its design goal (decouple from extrinsic) is measurable, and it tops both 1/7-VP and
   CTS-IMG rankings.
5. **Within extract-then-place, more depth supervision → more pitch-locked** (CAL-pitch: BEVDepth-explicit
   0.132 < BEVDet-implicit 0.254; CAPE-no-depth 0.182) and worse CTS (BEVDepth IMG 0.052 < BEVDet 0.086).

## Clean / in-distribution NDS (oracle, where available, 6-class)

| model | sedan | suv | bus |
|---|---|---|---|
| CAPE | 0.554 | 0.596 | 0.449 |
| DETR3D | 0.534 | 0.560 | 0.602 |
| BEVDepth | 0.532* | 0.547 | 0.418 |
| BEVDet | 0.519* | 0.535 | 0.373 |
| BEVFormer | 0.505* | 0.503 | 0.547 |

*(sedan = VP-subset P_NORMAL; suv/bus = full-val CTS oracle. DETR3D/bus and BEVFormer/bus run high —
full-res / dense models handle the elevated bus mount well; depth models drop on bus.)*

## Status
- **5/6 evaluated** (VP all-cam + 1/7 + CTS). **PETRv2 = train + eval pending** (the only gap).
- VP all frame-2×-fixed; CTS uses geobev images (frame-bug-unaffected). Per-model files in
  `results/{model}/{vp,cts}/`; mechanism detail in `VP_CROSS_MODEL_ANALYSIS.md`.

---

# Per-camera / per-axis deep-dive (VP, architectural)

VP perturbs the **sedan** model only (no platform axis — "per platform" applies to CTS: suv/bus
below). Full data: `results/vp_percam_peraxis.tsv` (5 models × EXT/IMG/CAL × 6 cams × roll/pitch/yaw).

## Per-camera × per-axis — IMG (primary), RRS

| camera | BEVFormer (r/p/y) | DETR3D (r/p/y) | CAPE (r/p/y) | BEVDet (r/p/y) | BEVDepth (r/p/y) |
|---|---|---|---|---|---|
| CAM_FRONT | .891/.861/.862 | .874/.852/.826 | .874/.864/.838 | .899/.886/.840 | .917/.902/.852 |
| CAM_FRONT_LEFT | .923/.898/.920 | .928/.923/.911 | .943/.934/.931 | .946/.934/.922 | .947/.934/.922 |
| CAM_FRONT_RIGHT | .973/.952/.957 | .969/.958/.964 | .978/.968/.964 | .976/.960/.958 | .980/.965/.963 |
| **CAM_BACK** | **.845/.823/.818** | **.857/.838/.804** | **.833/.825/.810** | **.823/.816/.781** | **.826/.814/.791** |
| CAM_BACK_LEFT | .930/.876/.918 | .940/.910/.928 | .946/.909/.928 | .946/.917/.929 | .933/.900/.910 |
| CAM_BACK_RIGHT | .969/.940/.965 | .961/.951/.956 | .987/.969/.978 | .964/.941/.953 | .977/.964/.966 |

## Three architectural findings from the per-camera data

**(1) Camera-importance asymmetry is DATASET-driven, architecture-INVARIANT.** Every model (incl.
CAPE) ranks the cameras identically: **CAM_BACK is the most load-bearing** (lowest RRS, ~0.78–0.86 —
perturbing it hurts most) and **the RIGHT cameras least** (FRONT/BACK_RIGHT ~0.95–0.99 — perturbing
them barely matters). FRONT3 0.91–0.95 vs BACK3 0.90–0.94; LEFT 0.91–0.95 vs RIGHT 0.95–0.99. This
pattern is **identical across gates-sampling and extract-then-place**, so it reflects the CARLA scene
(object density behind/left of ego, sparse right) + 6-view fusion, **not architecture**. Confirms:
single-camera robustness is dominated by redundancy, not view-transform.

**(2) Per-camera SPREAD per condition IS a (secondary) mechanism signal — it crosses over.**
Spread = max−min over the 6 cameras (axis-mean):

| model | mech | EXT spread | IMG spread | CAL spread |
|---|---|---|---|---|
| BEVDet | extract-then-place | **0.085** | 0.158 | 0.098 |
| BEVDepth | extract-then-place | **0.097** | 0.159 | 0.111 |
| CAPE | extract-then-place | (n/a) | 0.156 | 0.109 |
| BEVFormer | gates-sampling | 0.140 | 0.132 | **0.061** |
| DETR3D | gates-sampling | 0.130 | 0.130 | **0.037** |

Under **EXT**, extract-then-place is **tight** (all cameras ≈ equally robust — clean image features
intact regardless of which camera's extrinsic is off); gates-sampling is **wide**. Under **CAL**,
it flips: gates-sampling is **tight** (DETR3D 0.037 — the consistent projection re-aligns *every*
camera uniformly); extract-then-place is **wide** (BACK still drops, can't undo its tilted-image
extraction). The same mechanism that sets the all-cam EXT≈IMG-vs-EXT≫IMG split shows up at the
single-camera level as a spread crossover.

**(3) Worst-axis FLIPS between per-cam and all-cam: per-cam = yaw, all-cam = pitch.** Within a single
perturbed camera, **yaw is the worst axis** (CAM_BACK yaw 0.78–0.82 < pitch < roll for all models) —
rotating *one* camera horizontally maximally mis-registers it against the 5 clean cameras. But under
all-camera perturbation **pitch is decisively worst** (all-cam IMG pitch 0.14–0.26 ≪ yaw 0.40–0.53):
a *coherent* pitch tilts the shared ground-plane with no camera left to outvote it, whereas a coherent
yaw just rotates the whole BEV (partially recoverable). The redundancy that makes single-camera yaw
the local worst case is exactly what makes coherent pitch the global worst case.

## CTS per-platform (suv vs bus) — depth-generalization, not the VP axis

CTS has no per-cam/per-axis (all-camera condition swaps on full 3792 val). The platform split:
**bus transfer collapses the depth models** (BEVDet bus-IMG 0.002, BEVDepth 0.001) while suv is
milder — the monocular depth, trained on the sedan mount, is wrong for the elevated bus viewpoint.
**CAPE (no depth) is platform-robust** (bus-IMG 0.360 ≈ suv 0.338); DETR3D/BEVFormer mid. So the CTS
liability is **depth-viewpoint-generalization**, an axis orthogonal to the VP gates-vs-extract split
(CAPE is extract-then-place on VP but the *best* transferer on CTS).

---

# NDS TP decomposition — detection (mAP6) vs placement (TP errors)

NDS = ½·mAP + ½·(1−TP errors). Decomposing the NDS drop into **mAP6 (recall/detection)** vs **TP
quality (mATE/mASE/mAOE/…)** reveals the *failure mode*. mAP6 is available for all 5 VP models; full
TP components for BEVFormer & CAPE (VP) and all 5 (CTS).

## VP all-cam: NDS-RRS vs mAP6-retention (cond/Normal)

| model | mech | EXT NDS/mAP6 | IMG NDS/mAP6 | CAL NDS/mAP6 |
|---|---|---|---|---|
| CAPE | extract | n/a | 0.41 / **0.16** | 0.56 / 0.41 |
| BEVDepth | extract | 0.65 / **0.48** | 0.33 / **0.14** | 0.49 / 0.37 |
| BEVDet | extract | 0.61 / **0.40** | 0.36 / **0.16** | 0.52 / 0.38 |
| DETR3D | gates | 0.44 / **0.15** | 0.42 / **0.15** | 0.85 / 0.75 |
| BEVFormer | gates | 0.43 / **0.19** | 0.43 / **0.19** | 0.78 / 0.67 |

**The decomposition sharpens the whole story into a single statement of failure mode:**

- **IMG (all-cam) is a DETECTION collapse for *every* model** — mAP6 falls to **0.14–0.19** (recall
  gone); NDS survives only on the TP floor of the few remaining boxes. A coherently-tilted image
  destroys recall regardless of architecture.
- **EXT (all-cam) splits exactly on the mechanism, *at the mAP level*:**
  - **extract-then-place → detection SURVIVES** (mAP6 **0.40–0.48**): the clean image is still
    detected by the backbone; the wrong extrinsic only **mis-places** the boxes (a TP-error / placement
    failure, not a recall failure). *This is the mechanistic root of EXT≫IMG* — EXT keeps recall, IMG kills it.
  - **gates-sampling → detection COLLAPSES** (mAP6 **0.15**): a wrong extrinsic mis-samples the image
    features, so the boxes are never proposed — same recall collapse as IMG (hence EXT≈IMG).
- **CAL (all-cam): detection recovery follows the mechanism** — gates-sampling recovers recall
  (mAP6 0.67–0.75), extract-then-place only partially (0.37–0.41, the tilted image still suppresses recall).

## Which TP error dominates (where full components exist)

- **BEVFormer VP** (surviving boxes, inflation vs Normal): EXT/IMG inflate **all** spatial errors ~2×
  (mATE 1.9–2.0×, mASE 2.2–2.3×, mAOE 1.9–2.0×, mAAE ~2.5×); CAL recovers them (1.1–1.5×). mAVE is
  degenerate (single-frame velocity ≈ flat) → exclude from VP attribution.
- **CTS-IMG TP** (the few survivors): the **depth models carry extra scale/orientation error** —
  mASE 0.81–0.87 (BEVDet/BEVDepth) vs **0.31–0.34** (CAPE/DETR3D, no depth), mAOE ~1.0 vs ~0.72. A
  wrong cross-platform depth mis-*sizes* the survivors, on top of the recall collapse — a depth-specific
  fingerprint, consistent with the CTS depth-generalization trend. (Small-sample for BEVDet/BEVDepth
  since their CTS-IMG mAP ≈ 0.)

**Bottom line:** the VP mechanism split is, at the metric level, a **recall** story —
*extract-then-place keeps recall under extrinsic error (only mis-places), everyone loses recall under
image tilt, and gates-sampling loses recall under both but regains it under consistency.*
