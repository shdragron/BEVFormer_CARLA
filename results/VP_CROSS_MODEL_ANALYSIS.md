# VP cross-model analysis — view-transform mechanism vs camera-geometry robustness

> **Status (2026-06-08, after the carla_VR frame-2× fix).** The committed VR/CR numbers were
> inflated by a frame bug (**geobev frame N == carla_VR frame 2N**; the builders loaded a
> *different scene* for the VR/CR image swaps — verified: geobev 0269/0150 == VR baseline 0300,
> exact 2.000×). Builders are now fixed (`int(frame)*2`).
>
> | model | mechanism | Normal | ER (EXT) | VR (IMG) / CR (CAL) |
> |---|---|---|---|---|
> | **CAPE** | extract-then-place (PE) | ✅ | ✅ | ✅ frame-fixed |
> | **BEVDet** | extract-then-place (LSS) | ✅ | ✅ | ✅ re-run, corrected |
> | **BEVFormer** | extrinsic-gates-sampling | ✅ | ✅ | ✅ re-run, corrected |
> | **DETR3D** | extrinsic-gates-sampling | ✅ | ✅ | ✅ fresh run, fixed builder |
> | **BEVDepth** | extract-then-place (LSS) | ✅ | ✅ | ✅ re-run, corrected (frame-fixed) |

VP = the **sedan** model under `carla_VR` camera-geometry perturbations, vis≥2 GT, 6-class NDS,
same frozen 16/scene (768-sample) subset and axis×magnitude grid for every model/condition.
`RRS = NDS_cond / P_NORMAL` (within-model ratio → normalizes absolute-NDS/resolution).
Conditions: **ER=EXT** (extrinsic perturbed, image clean), **VR=IMG** (perturbed image, extrinsic
kept = primary), **CR=CAL** (both, consistent). Scopes: per-cam (perturb 1 of 6), all-cam (all 6).
*(All findings below are all-cam unless noted — per-cam is uniformly mild (0.90–0.99) and
non-discriminating: 6-view fusion outvotes one bad camera.)*

## The discriminating axis: **does the extrinsic gate feature sampling?** (verified from code)

A 5-agent code audit (HIGH confidence, all consistent with measured signatures) splits the
detectors by **how the camera extrinsic enters the forward pass** — *not* by depth, and *not* by
Dense-vs-Sparse:

- **extrinsic-gates-sampling** — **BEVFormer** (`encoder.point_sampling`: lidar2img projects BEV
  queries into images → deformable sampling) and **DETR3D** (`detr3d_transformer.feature_sampling`:
  lidar2img projects 3D ref points → `grid_sample`). The backbone extracts 2D features
  **independent of the extrinsic**; the extrinsic decides **where in the image to sample**.
- **extract-then-place** — **CAPE** (`cape_transformer.position_embeding`: features extracted by
  the backbone, geometry applied only as a **camera-view position embedding** via `I_inv`/`R`/`t`,
  **no `grid_sample`**), **BEVDet**/**BEVDepth** (LSS: backbone + depth predicted in image space,
  then `get_lidar_coor`/`voxel_pooling` **splats** the already-extracted features into BEV using
  the extrinsic). Features are extracted **independent of the extrinsic**, which is applied
  **afterward** as a PE or a splat.

**CAPE is the proof this isn't about depth:** it has *no* explicit depth (camera-view PE), yet it
sits firmly with the LSS depth models — because it, too, extracts image features before applying
geometry. Mechanism, not depth, is the axis.

## All-cam RRS, grouped by mechanism

| model (clean NDS) | mechanism | EXT roll/pitch/yaw/**ALL** | IMG roll/pitch/yaw/**ALL** | CAL roll/pitch/yaw/**ALL** |
|---|---|---|---|---|
| CAPE (0.5508) | extract-then-place | 0.978/0.955/0.499/**0.811** | 0.523/0.184/0.513/**0.407** | 0.515/0.182/0.983/**0.560** |
| BEVDepth (0.5324) | extract-then-place | 0.843/0.686/0.428/**0.652** | 0.443/0.137/0.404/**0.328** | 0.379/0.132/0.965/**0.492** |
| BEVDet (0.5185) | extract-then-place | 0.737/0.657/0.434/**0.610** | 0.452/0.221/0.418/**0.364** | 0.358/0.254/0.951/**0.521** |
| DETR3D (0.5368) | extrinsic-gates-sampling | 0.549/0.253/0.513/**0.438** | 0.512/0.260/0.493/**0.422** | 0.789/0.825/0.921/**0.845** |
| BEVFormer (0.5051) | extrinsic-gates-sampling | 0.572/0.173/0.537/**0.428** | 0.560/0.189/0.531/**0.426** | 0.730/0.661/0.939/**0.777** |

## Finding 1 — EXT≈IMG (gates-sampling) vs EXT≫IMG (extract-then-place)

- **extrinsic-gates-sampling → EXT ≈ IMG** (BEVFormer 0.428≈0.426, DETR3D 0.438≈0.422). The
  extrinsic gates the sampling, so corrupting it (EXT) or corrupting the image it samples from
  (IMG) does **equal** damage. *(This is why the pre-fix "image ≫ extrinsic" headline was an
  artifact — corrected, they're equal for these models.)*
- **extract-then-place → EXT ≫ IMG** (CAPE 0.811≫0.407, BEVDepth 0.652≫0.328, BEVDet 0.610≫0.364).
  The backbone extracts features from the **clean** image under EXT → robust; under IMG the **tilted
  image corrupts extraction itself** → fragile. The gap *is* the classic "depth double-edge", but it
  is really an **extract-then-place** edge (CAPE has no depth).
- **CAPE is the extreme of EXT-robustness (0.811, highest of all)** — its camera-view position
  embedding is *designed* to decouple from the extrinsic, and the benchmark shows it directly
  (EXT roll/pitch ≈ 0.96–0.98, near-zero degradation).

## Finding 2 — CAL-pitch is the cleanest single discriminator: recovery vs collapse

Under CAL (image + extrinsic perturbed **consistently**):

| mechanism | CAL-pitch | meaning |
|---|---|---|
| extrinsic-gates-sampling | BEVFormer **0.661**, DETR3D **0.825** | **recovers** — the consistent projection re-aligns sampling end-to-end |
| extract-then-place | BEVDepth **0.132**, CAPE **0.182**, BEVDet **0.254** | **stays collapsed** — a consistent extrinsic cannot undo a tilted image already baked into feature extraction |

Within extract-then-place there is a **depth-supervision gradient**: more depth supervision → *more*
pitch-locked under consistency (BEVDepth, explicit LiDAR depth, **0.132** < BEVDet, implicit/categorical
depth, **0.254**; CAPE, no depth, 0.182). The depth head most aggressively commits the tilted-image
features to fixed BEV cells, which a consistent extrinsic can least undo — but this is a *second-order*
gradient *within* the extract-then-place class, not the primary split.

This is the mechanistic crux: the gates-sampling models can re-fetch correct features when the
projection is made self-consistent; the extract-then-place models cannot, because the corruption
happened in the (extrinsic-blind) backbone before any geometry was applied. Note both families
recover CAL-**yaw** (all ≈ 0.92–0.98): yaw rotates about vertical and preserves the ground-plane
appearance, so even extract-then-place extraction survives it — only the **horizon-tilting** axes
(roll, pitch) defeat extract-then-place under consistency.

## Finding 3 — opposite EXT worst-axes (extrinsic fingerprint)

- **extract-then-place: worst EXT axis = yaw** (CAPE 0.499, BEVDet 0.434, BEVDepth 0.428; roll/pitch
  0.66–0.98). A wrong extrinsic only mis-*places* clean features; **yaw** displaces objects most in
  BEV bearing, while roll/pitch barely move the footprint.
- **extrinsic-gates-sampling: worst EXT axis = pitch** (BEVFormer 0.173, DETR3D 0.253; roll/yaw
  0.51–0.57). A wrong extrinsic mis-*samples*; **pitch** shifts the sampled image rows onto sky/ground,
  the most informative band.

## Superseded framing

An earlier version of this doc framed the split as **depth-LSS vs sampling**. **CAPE disproves
that**: CAPE is depth-free (camera-view PE) yet patterns with the LSS depth models on every axis
(EXT≫IMG, CAL-pitch collapse, EXT-worst-yaw). The correct, code-verified axis is **whether the
extrinsic gates feature sampling**. Dense-vs-Sparse is also a red herring (BEVFormer=Dense and
DETR3D=Sparse are in the *same* group; CAPE=Sparse and BEVDet=Dense-BEV are in the *other*).

## Methodology notes

- **All-camera perturbation is the discriminating protocol** — per-cam is near-architecture-invariant
  (0.90–0.99 for all five); the signal lives in all-cam.
- **BEVDepth confirmed** the extract-then-place prediction once re-run (frame-fixed): EXT 0.652 ≫
  IMG 0.328, CAL-pitch 0.132 (the most collapsed of all). All 5 models now corrected/complete.
- **The frame bug only touched VR/CR** (image-swap conditions); ER/Normal and all of CTS (geobev
  images) are unaffected. Lesson: image-swap robustness studies must verify frame correspondence
  between the perturbation source and the GT/pkl.

*Numbers from `results/{model}/vp/` (CAPE: `vp_cape_sedan_axis_table.txt`; BEVFormer/BEVDet =
frame-fixed; DETR3D = fresh run, fixed builder; BEVDepth = frame-fixed VR/CR, buggy backed up as
`eval_vp_BUGGY_framemismatch.json`). All 5 models corrected/complete. VP = 768-sample subset (fps 16).
Mechanism classes verified by a 5-agent code audit (2026-06-08, all HIGH confidence). CTS results
(full 3792 val, frame-bug-unaffected) are in `results/{model}/cts/`.*
