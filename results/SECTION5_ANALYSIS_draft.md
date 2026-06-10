# §5 Analysis — draft v3 (restructured to the authors' outline; both tasks)

> Outline mapping (authors' plan → this draft):
> 5.1(1) paradigm-specific IMG vs CAL ranking + recovery → ¶1–2
> 5.1(2) EXT–IMG correlation figure → ¶3 (Fig. A)
> 5.1(3) axis + camera analysis → ¶4–5
> 5.2(1) paradigm-specific IMG/CAL recovery + figure → ¶1–2 (Fig. B) + caveat
> 5.2(2) platform analysis (SUV vs Bus) → ¶3
> 5.3(1) VP–CTS correlation → Fig. C (see the note: data says *partially* aligned, not "same ranking")
> All numbers all-camera RRS unless marked Table-2 (%); verified 2026-06-10 (3-agent pass).

---

## 5. Analysis

### 5.1. Viewpoint Robustness

**Paradigm-specific analysis: the IMG ranking compresses, the CAL ranking re-orders.**
Under the IMG condition — re-rendered viewpoints with the original extrinsics — all paradigms
degrade together: the all-camera scores span only 0.33–0.43 for detection (BEVFormer 0.43,
DETR3D 0.42, CAPE 0.41, BEVDet 0.36, BEVDepth 0.33) and 0.24–0.45 for segmentation, and the
ordering does not follow the forward/backward/sparse labels. Supplying the *correct* changed
extrinsics (CAL) dramatically re-ranks the models: BEVFormer 0.43→0.78, DETR3D 0.44→0.85 and
PointBeV 0.36→0.76 recover to the top, while CAPE (0.81→0.56), BEVDet (0.61→0.51), BEVDepth
(0.65→0.49), CVT (0.69→0.65), LaRa (0.70→0.63) and LSS (0.62→0.51) do not.

**This recovery is not a paradigm property.** The models that recover span backward
(BEVFormer) and sparse (DETR3D, PointBeV); the models that do not span forward (all LSS-family),
sparse (CAPE) and backward (CVT, LaRa). What the recovering models share is *how they consume
extrinsics*: they project 3D queries or BEV points into the images and **sample features at the
projected locations**, so correct extrinsics re-align the projection end-to-end. The
non-recovering models extract features extrinsic-independently and apply geometry afterward —
as a depth-splat (LSS family) or a camera-geometry positional encoding (CVT, LaRa, CAPE) — so
the image tilt is already baked into the features and correct extrinsics cannot undo it. We
refer to the two groups as *sampling-gated* and *extract-then-place*; the EXT–CAL ordering
(CAL > EXT iff sampling-gated) holds for 10 of 12 models (GaussianLSS is borderline,
0.698/0.699; SimpleBEV degrades anomalously under all consistent perturbations and is flagged
for inspection).

**EXT–IMG correlation (Fig. A).** Plotting the all-camera EXT score against IMG makes the same
split visible from the metadata side. Sampling-gated models lie on the diagonal (EXT ≈ IMG:
corrupting the extrinsics they sample with is as harmful as corrupting the image), while every
extract-then-place model falls far below it (EXT ≫ IMG, up to 2×: CAPE 0.81 vs 0.41 — a clean
image survives an extrinsic error, but a tilted image corrupts extraction itself). Across the
eleven models the EXT–IMG rank correlation is only ρ = 0.25. **Extrinsic-only perturbation
therefore under-reports true viewpoint degradation — specifically for extract-then-place
models — while reflecting it faithfully for sampling-gated ones**, refining the paper's first
claim at the mechanism level. Note the headline mVRS in Table 2 hides this gap (e.g., BEVFormer
83.1/83.8): the per-camera component (≈0.9 for all models) dominates the 1/7 average, so the
analysis above uses the all-camera component.

**Axis analysis.** Under IMG, the most damaging axis differs by task: **pitch** for detection
(all-camera 0.14–0.26 across all detectors) and **yaw** for segmentation (0.25–0.31, tightly
clustered across all six models) — a coherent all-camera yaw breaks the cross-camera
correspondence that BEV lifting relies on, while pitch tilts the shared ground plane that
detection's depth and size estimation depend on. Under EXT, the worst axis is itself a
mechanism fingerprint: extract-then-place detectors fail most on **yaw** (0.43–0.50; the
extrinsic error mis-places clean features, and yaw displaces the BEV bearing most), whereas
sampling-gated detectors fail most on **pitch** (0.17–0.25; the wrong extrinsic mis-samples
image rows onto sky or ground). Under CAL, models in both tasks recover yaw almost fully
(detection 0.89–0.98, segmentation 0.78–0.95, the anomalous SimpleBEV excepted) — a consistent
yaw is a global rotation about the vertical axis that preserves ground-plane appearance —
whereas pitch is recovered only by the sampling-gated detectors (0.66–0.83 vs 0.13–0.25),
making CAL-pitch the single cleanest mechanism discriminator in the detection benchmark.

**Camera analysis.** Single-camera perturbations are comparatively benign and architecture-
invariant: per-camera IMG scores stay at 0.81–0.98 for every detector, and the importance
ordering is identical across all five — the **back camera is the most critical** (0.81–0.83),
the front camera second (0.85–0.89), and the right-side cameras least (0.95–0.98), reflecting
the rig's field-of-view overlap and the scene's object distribution rather than any modeling
choice. Six-view fusion simply out-votes one corrupted view. Two consequences: per-camera
robustness measures the *rig's redundancy*, not the architecture, and the all-camera protocol
is the discriminating one — which is why the analysis above reads the all-camera component
even though the headline mVRS weights per-camera 6:1.

### 5.2. Cross-Platform Transfer

**Paradigm-specific analysis: depth dependence, then the CAL response (Fig. B).** Under the
target-viewpoint IMG condition, the forward (depth-based) detectors collapse on the bus
(CTS 0.1%): monocular depth trained on the sedan mount predicts wrong distances from the
elevated bus viewpoint, lifting features to wrong BEV ranges; the failure also fingerprints
the survivors (suv-IMG mASE 0.63–0.73 vs 0.31–0.36 for depth-free; on the bus, depth models'
mASE saturates at 1.0). Depth-free detectors transfer far better — CAPE, the most
extract-then-place model under viewpoint perturbation, is the strongest transferer (bus-IMG
36.0). Supplying the correct target extrinsics (CAL) then separates the mechanisms exactly as
in §5.1 (Fig. B): sampling-gated models convert correct extrinsics into large gains
(suv IMG→CAL: BEVFormer 37.2→71.9, DETR3D 34.9→76.9, PointBeV 20.2→68.4), depth-bound
detectors stay collapsed (bus-CAL 8.5/16.6 — the depth prior, not the projection, is what
fails), and for two embedding-based segmentation models the correct extrinsics actively *hurt*
(CVT bus 17.7→1.8, SimpleBEV 12.2→3.4): their learned geometry embeddings are fit to the
source rig, so the true (far-from-training) target geometry pushes the encoding further out of
distribution than the stale sedan extrinsics. Calibration-aware projection alone is therefore
not sufficient for cross-platform deployment, and *how* a model consumes extrinsics determines
whether better calibration even helps.

*A measurement caveat.* CTS normalizes by each model's own platform-matched oracle, and these
oracles differ by up to 1.6× across detectors; the ratio can favor a model with a weaker
oracle (CAPE and DETR3D have equal absolute bus-IMG transfer, SDS 0.162 vs 0.159, yet CTS
36.0 vs 26.5 because CAPE's bus oracle is lower, 0.449 vs 0.602). The depth-vs-depth-free
group separation (absolute 0.001 vs 0.10–0.16) is robust to this; fine rankings within a
group should be read from absolute transfer, and Table 2 reports the oracle next to each CTS.

**Platform analysis: the bus is qualitatively harder than the SUV.** Every model loses more
transferring to the bus than to the SUV, in both tasks and all conditions (detection IMG:
BEVFormer 37.2→18.0, DETR3D 34.9→26.5, and the depth detectors 9–10→0.1; segmentation IMG:
11.4–27.7 on the bus vs 11.8–44.3 on the SUV, with bus-CAL collapsing to 1.8–3.4 for
CVT/SimpleBEV). The SUV is a mild remount — similar height, small layout change — while the
bus changes the camera height regime entirely, which (i) breaks sedan-trained depth priors
hardest (forward detectors: bus-IMG 0.1%), (ii) moves geometry embeddings furthest from the
training distribution (the CVT/SimpleBEV CAL inversion appears only on the bus), and (iii)
changes what is visible at all — per-class visibility shifts by +9.5 to +19.4 points from
sedan to bus (Table 1), which the per-platform oracle absorbs but the transferred model must
cope with. Cross-platform robustness should therefore be read per platform: SUV measures
tolerance to a layout tweak, the bus measures tolerance to a viewpoint-regime change.

### 5.3. VP–CTS Correlation

Fig. C tracks each model's rank across NORMAL → VP-IMG → VP-CAL → CTS-IMG. The rankings are
**partially aligned exactly where the two protocols share a bottleneck, and de-coupled where
their governing properties differ.** Under IMG, viewpoint and cross-platform rankings
correlate strongly in segmentation (ρ = 0.83) and moderately in detection (ρ = 0.50) — both
protocols are dominated there by the same image-appearance shift. But the alignment breaks
exactly along the two axes identified above: CAPE is among the worst at absorbing a coherent
viewpoint shift (CAL 0.56) yet the best transferer (bus-IMG 36.0, depth-free), while the
sampling-gated BEVFormer/DETR3D dominate VP-CAL (0.78/0.85) but sit mid-pack on CTS-IMG;
PointBeV repeats the pattern in segmentation (best VP-CAL 0.76, near-worst bus-IMG 11.4).
And neither axis is predicted by the standard leaderboard: NORMAL rank correlates at
ρ = −0.20 (det, VP-IMG) and ρ = −0.71 (seg, CTS-IMG) — in segmentation the *best* NORMAL
model (SimpleBEV) is last on every robustness axis and the *worst* (CVT) is the best
transferer. **Camera-geometry robustness is therefore not a single quantity**: viewpoint
robustness is set by whether extrinsics gate sampling (§5.1), transfer by whether the
representation depends on source-rig priors (§5.2), and in-distribution accuracy by neither —
which is why RoboGeo reports mVRS and CTS separately rather than collapsing them into one
score.

---

## Notes for the authors

1. **Outline item 5.3 said "all vr and cross platform same ranking" — the data does not
   support "same".** The defensible version (written above): IMG-based rankings partially
   align (seg ρ=0.83, det ρ=0.50, shared image bottleneck), but CAL/mechanism/depth de-couple
   them (CAPE, PointBeV), and NORMAL predicts neither (det −0.20; seg CTS −0.71). This also
   matches the Introduction's own claim ("viewpoint robustness and cross-platform transfer
   reveal different patterns of model sensitivity") — "same ranking" would contradict the
   intro.
2. **Table 2's 1/7 mVRS hides the EXT–IMG gap** (per-cam ≈ 0.9 everywhere) — §5.1 cites the
   all-camera component; add an all-cam breakdown table/figure (Fig. A covers the figure).
3. **Mechanism framing** (sampling-gated vs extract-then-place) is introduced *inside* the
   paradigm-specific paragraphs, as the property that explains why recovery cuts across
   paradigms — matching the authors' intended flow (paradigm first, then the refinement).
4. **SimpleBEV is an outlier** (CAL 0.18 < IMG 0.24; EXT-yaw 0.953) — flagged, not explained;
   verify before citing.
5. CVT/SimpleBEV **bus-CAL < bus-IMG inversion** — VERIFIED against Table 2 + raw IoU
   (CVT 0.083→0.009 / oracle 0.470 ⇒ 17.7→1.8%). Safe to feature.
6. All numbers verified by a 3-agent pass (2026-06-10); per-camera ordering verified from
   vp_percam_peraxis.tsv (BACK most critical for all five detectors, 0.81–0.83).
7. Pending: DFA3D (bus oracle training), DSPE, PETRv2 (absent from Table 2 — drop or pend?),
   Table 3 baselines (Ext.Aug / EAFormer / PD-BEV).
8. **Figures** (results/figures/, make_section5_figs.py): Fig. A
   `fig_vp_ext_img_correlation.png` (EXT vs IMG scatter, ρ=0.25); Fig. B
   `fig_cts_img_to_cal.png` (CTS IMG→CAL dumbbell, SUV/Bus); Fig. C `fig_ranking_bump.png`
   (rank bump + Spearman; CTS-IMG = mean suv/bus, subject to the §5.2 oracle caveat).
