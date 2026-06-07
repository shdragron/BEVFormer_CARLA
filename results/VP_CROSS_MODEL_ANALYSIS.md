# VP cross-model analysis — view-transform architecture vs camera-geometry robustness

> **Status (2026-06-07, after the carla_VR frame-2× fix).** The committed VR/CR numbers
> were inflated by a frame bug (**geobev frame N == carla_VR frame 2N**; the builders loaded
> a *different scene* for the VR/CR image swaps — verified: geobev 0269/0150 == VR baseline
> 0300, exact 2.000×). Builders are now fixed (`int(frame)*2`). This doc is re-scoped to:
>
> | model | Normal | ER (EXT, extrinsic) | VR (IMG) / CR (CAL) |
> |---|---|---|---|
> | **BEVFormer** | ✅ valid | ✅ valid | ✅ **re-run, corrected** |
> | **BEVDet** | ✅ valid | ✅ valid | ✅ **re-run, corrected** (Finding C) |
> | **BEVDepth** | ✅ valid | ✅ valid | ⏳ **pending re-run** (buggy — do not cite) |
>
> ER/Normal never use a VR image, so they are unaffected. BEVFormer + **BEVDet** VR/CR are now
> re-run (corrected); BEVDepth VR/CR stay pending.

VP = the **sedan** model under `carla_VR` camera-geometry perturbations, vis≥2 GT, 6-class NDS,
same frozen 16/scene (768-sample) subset and axis×magnitude grid for every model/condition.
Robustness `RRS = NDS_cond / P_NORMAL` (within-model ratio → normalizes absolute-NDS/resolution).
Conditions: **ER=EXT** (extrinsic perturbed, image clean), **VR=IMG** (perturbed image, extrinsic
kept = primary), **CR=CAL** (both, consistent). Scopes: per-cam (perturb 1 of 6) and all-cam (all 6).
Families: **Forward/Depth-LSS** (BEVDet, BEVDepth) vs **Backward/Dense** (BEVFormer).

---

## Finding A (VALID, cross-model) — extrinsic (ER) robustness: Depth > Backward, with opposite worst-axes

ER all-cam RRS (mean over 5 magnitudes):

| model (clean NDS) | roll | pitch | yaw | **ALL** |
|---|---|---|---|---|
| BEVDepth (0.5324) | 0.843 | 0.686 | **0.428** | **0.652** |
| BEVDet (0.5185) | 0.737 | 0.657 | **0.434** | **0.610** |
| BEVFormer (0.5051) | 0.572 | **0.173** | 0.537 | **0.428** |

- **Forward/Depth models are markedly more robust to all-camera extrinsic error** (0.61–0.65 vs
  BEVFormer 0.43). Mechanism: LSS computes features+depth in image space first and only the BEV
  *splat placement* uses the (wrong) extrinsic → recoverable; BEVFormer bakes the extrinsic into
  the sampling projection, so every BEV query reads the wrong image location.
- **Opposite worst-axis signatures** (architecture fingerprint): the Depth models' weakest ER axis
  is **yaw** (~0.43, vs roll/pitch 0.66–0.84); BEVFormer's weakest is **pitch** (0.173, vs roll/yaw
  0.54–0.57). per-cam ER is uniformly mild and non-discriminating (0.90–0.95).

## Finding B (CORRECTED, BEVFormer only) — image vs extrinsic are *equally* damaging; the damage is *inconsistency*

BEVFormer all-cam RRS (frame-fixed):

| condition | roll | pitch | yaw | **ALL** | per-cam ALL |
|---|---|---|---|---|---|
| EXT (extrinsic) | 0.572 | 0.173 | 0.537 | 0.428 | 0.898 |
| **IMG (image, primary)** | 0.560 | 0.189 | 0.531 | **0.426** | 0.907 |
| **CAL (consistent)** | 0.730 | 0.661 | 0.939 | **0.777** | 0.961 |

- **EXT ≈ IMG** (all-cam 0.428 ≈ 0.426, and axis-by-axis nearly identical). For a backward/dense
  model, perturbing the *extrinsic* or perturbing the *image* produces the **same** degradation —
  because both create the same image↔projection mismatch the dense sampler depends on.
  *(The old "image ≫ extrinsic" headline was entirely the frame artifact.)*
- **CAL ≫ EXT/IMG** (0.777 vs 0.43): when image and extrinsic are perturbed **consistently**, the
  projection is self-consistent and the model degrades gracefully (yaw nearly fully recovers, 0.939).
  → the damage is image–extrinsic **inconsistency**, not the perturbation magnitude per se.
- **Pitch is the worst axis** in every condition (EXT/IMG pitch 0.17–0.19; even CAL pitch 0.661 <
  CAL yaw 0.939), and it is **magnitude-gated** — IMG pitch: 0.46 @±4 → 0.25 @±8 → 0.09 @±16 →
  0.06 @±20 (a cliff). roll/yaw stay moderate across magnitudes.

## Finding C (CORRECTED, BEVDet) — depth model keeps a residual, pitch-dominated image-tilt fragility

BEVDet all-cam RRS (frame-fixed, 768-sample VP subset):

| condition | roll | pitch | yaw | **ALL** | per-cam ALL |
|---|---|---|---|---|---|
| EXT (extrinsic) | 0.737 | 0.657 | **0.434** | 0.610 | 0.938 |
| **IMG (image, primary)** | 0.452 | **0.221** | 0.418 | **0.364** | 0.911 |
| CAL (consistent) | 0.358 | **0.254** | 0.951 | 0.521 | 0.937 |

- **IMG (0.364) stays well below EXT (0.610)** — it does NOT close to parity the way BEVFormer's did
  (EXT≈IMG 0.428≈0.426). So for the depth/LSS model, perturbing the *image* is genuinely more
  damaging than perturbing the *extrinsic* all-cam → a residual depth-specific image-tilt fragility
  survives the frame fix (the monocular depth/ground prior is broken by the tilted image, beyond the
  recoverable splat-placement error). The "depth double-edge" is **softened but real**, not dissolved.
- **Worst axis = pitch** for IMG (0.221) and CAL (0.254), vs EXT's worst axis **yaw** (0.434): the
  depth model's *extrinsic* weakness (yaw) and *image* weakness (pitch) lie on different axes — pitch
  tilt specifically breaks the ground-plane/depth prior. (IMG pitch-worst matches BEVFormer.)
- CAL yaw recovers to 0.951 (consistent image+extrinsic yaw cancels, as in BEVFormer); but CAL ALL
  0.521 < BEVFormer 0.777 — the depth model degrades more even when consistent, dragged by pitch.

## Open question (BEVDet resolved above; BEVDepth pending)

The original (buggy) analysis claimed a **"depth double-edge"** — Depth models robust to extrinsic
but *uniquely fragile to image tilt* (committed BEVDepth IMG 0.151 < BEVDet 0.190 < BEVFormer 0.304,
a "flip"). **That ranking was built on the inflated VR numbers and must not be cited.** Corrected so far:
- **BEVFormer:** IMG→EXT parity (Finding B) — image fragility was the artifact.
- **BEVDet:** IMG stays below EXT (0.364 < 0.610), worst-axis pitch (Finding C) — a genuine residual,
  softened depth double-edge. So the corrected picture is **between** the two hypotheses: not full
  parity (H1), but a much smaller, pitch-specific gap rather than the old catastrophic "flip" (H2).
- **BEVDepth:** VR/CR re-run still pending (bevdet-b200 env, builder fixed) to confirm the Depth family.

## Methodology notes (valid)

- **All-camera perturbation is the discriminating protocol.** per-cam is near-architecture-invariant
  (ER per-cam spread 0.90–0.95; BEVFormer IMG/EXT per-cam ~0.90–0.91) — the 6-view fusion outvotes a
  single bad camera; the architecture signal lives in all-cam.
- **The frame bug only touched VR/CR** (image-swap conditions). The lesson: image-swap robustness
  studies must verify frame correspondence between the perturbation source and the GT/pkl — here a
  silent 2× relabel turned "viewpoint robustness" into "wrong-scene generalization" for VR/CR.

*Generated from `results/{model}/vp/eval_vp_per_config.csv` (BEVFormer + BEVDet = frame-fixed;
BEVDepth ER/Normal valid, VR/CR buggy-pending). VP = 768-sample subset (frames-per-scene 16);
CTS = full 3792-frame val. The carla_VR frame-2× bug touched only VP VR/CR (carla_VR image swaps);
CTS uses geobev images only → unaffected.*
