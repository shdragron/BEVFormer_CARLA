# PROJECT HANDOFF — read this first (machine move)

A new assistant taking over: **read this file, then the two tables of pointers
below.** This snapshots the whole project so you can continue without the prior
chat. Written 2026-06-15.

> ⚠️ **The persistent memory (`~/.claude/.../memory/`) is machine-local and does
> NOT move with the computer.** It has been copied into
> **`results/memory_snapshot/`** (21 files) so it travels with the repo. Treat
> `results/memory_snapshot/MEMORY.md` as the project index; the other files are
> one-fact-each detail notes. They reflect what was true when written — verify a
> file/flag still exists before relying on it.

---

## ⏱ SESSION UPDATE (2026-06-15, late) — full-VP unification + Extrinsic-Aug

This session ran AFTER §2's seg task was set up. Three threads advanced — read this
before re-deriving anything:

**(A) carla_VR frame-2N bug — FIXED in all builders.** `int(frame)*2` applied to
`bev_det_benchmark/build_condition_pkls{,_bevdet,_bevdepth}.py` + `sparse/build_condition_pkls.py`.
(geobev frame N == carla_VR frame 2N; verified 0269/0150==VR-0300.) VR/CR re-run on the fix.

**(B) "5 models unified to full 3792" (user directive). Status:**
| model | full VP | location | NDS_Normal |
|---|---|---|---|
| BEVDepth | ✅ 630 both-proto | `results/BEVDepth/vp/eval_vp.*` (768 → `*.subset768.*`) | 0.5354 |
| BEVDet | ✅ 630 both-proto | `results/BEVDet/vp/eval_vp.*` (768 backed up) | 0.5166 |
| BEVFormer | ✅ all-cam | `results/BEVFormer/vp/eval_vp_full3792_allcam.*` (`eval_vp.*`=768, has per-cam) | 0.5037 |
| DETR3D | ✅ all-cam, **NOT yet copied to results/** | `bev_det_benchmark/sparse/out_vp_detr3d/vp_detr3d_sedan_full/eval_vp.json` | — |
| CAPE | ✅ fullframe | `results/CAPE/vp/vp_cape_sedan_fullframe*.{json,txt,csv}` | 0.5547 |
- full≈subset (|ΔRRS|≤0.01), so the **cross-model table stays 768-matched**:
  `results/_vp_xmodel_aggregate.py` (768 P_NORMAL, prefers `*.subset768.csv`) →
  `results/_vp_xmodel_ground_truth.json`. **mVRS in the paper = 1/7-weighted**
  = `(6*mRRS_percam + RRSALL_allcam)/7`. CTS is full 3792 already (all models).

**(C) Paper Table "Do existing techniques close the gap?" — verified the user's numbers.**
VP 1/7 (Ext/Img/Cal × roll/pitch/yaw/ALL) for BEVDet/BEVDepth/BEVFormer/CAPE all CORRECT.
CTS correct for BEVDepth/BEVFormer/CAPE. **BEVDet CTS row is ABSOLUTE NDS, not ratio** — fix to
ratios: suv Ext/Img/Cal **0.615/0.170/0.161**, bus **0.539/0.0022/0.228**.

**(D) 🔴 RUNNING — BEVDepth + Extrinsic Aug** (the det "+Extrinsic Aug" row). Implementation
(3 files, flag-gated, baseline byte-identical when off):
- `BEVDepth/bevdepth/datasets/nusc_det_dataset.py` — `extrin_noise_conf`, `_sample_extrin_delta()`,
  perturb `sweepsensor2keyego` in `get_image` (per-sample Bernoulli(p), δ per-camera, E'=δ@E,
  rot Euler ±20°, trans 0). Only the LSS-lift sensor2ego is perturbed; **image/GT/intrinsics/depth-GT
  stay clean** (Option A).
- `BEVDepth/bevdepth/exps/nuscenes/base_exp.py` — passes it to the **train** loader only.
- `BEVDepth/bevdepth/exps/nuscenes/carla/carla_sedan_extrinaug.py` — `dict(p=0.5, rot_deg=20.0)`,
  else == `carla_sedan`. Launch: `BEVDepth/_run_sedan_extrinaug_train.sh`; log
  `BEVDepth/_train_sedan_extrinaug.log`; wandb `BEVDepth-CARLA/carla_sedan_extrinaug`
  (run w40ntlzt). Same recipe as baseline sedan (no grad-clip). ckpt →
  `BEVDepth/outputs/carla_sedan_extrinaug/`.
- **At handoff: epoch ~19/24, val/NDS ~0.508** (climbed 0.275→0.50+; LR steps ep19,23 → final
  bump). Slow (~2h/epoch, GPU-contended).
- **WHEN DONE**: run full VP + CTS on that ckpt → compute 1/7 mVRS(Ext/Img/Cal) + CTS-Cal(suv,bus)
  → fill the "+Extrinsic Aug" row. (VP via `run_vp_full_bevdepth.sh` with `--ckpt` pointed at the new
  ckpt, auto-resume wrapper; CTS via the bevdepth cts driver with the new ckpt.)

**Next-session TODO:** (1) finish+eval extrinaug → row; (2) DETR3D full → `results/DETR3D/vp`
(mirror like BEVFormer's `_full3792_allcam.*`); (3) apply the BEVDet-CTS ratio fix in the table;
(4) +PD-BEV row from `pdbev-vp-cts-results.md` (full-VP may still be running); (5) seg track.

**Gotchas reconfirmed this session:** NuScenes eval prints benign `nan` for CARLA-absent classes
(traffic_cone/barrier) — NOT divergence. `/tmp` tmpfs counts against the 400G cgroup; a stale 276G
VP stage caused OOMs — keep `/tmp` clean. Stage ONLY your files when committing (lots of user WIP
untracked). BEVDet VP driver dedups duplicate progress lines on load (harmless).

---

## 0. Who / constraints (verbatim, must persist)

- User: shangmoon@korea.ac.kr. Works in Korean; reply in Korean.
- For the **shdragron/** repos (DFA3D, LatentCalib, etc.): commit as
  `shdragron <shdragron@hanyang.ac.kr>`, **NO "Claude" anywhere**, no co-author
  trailer, push via SSH. (This BEVFormer detector repo is a separate local repo.)
- Don't commit/push unless asked.

## 1. Two papers in flight

1. **RoboGeo** (WACV 2027 Datasets Track) — "Benchmarking Camera-Geometry
   Robustness in Multi-Camera BEV Perception". A CARLA benchmark measuring VP
   (viewpoint perturbation) and CTS (cross-sensor transfer) robustness across
   detectors + seg models. Draft/figures/§5 live in `results/`.
2. **LatentCalib** (method paper, repo `/home/hanyan_arch/viewpoint/LatentCalib`,
   GitHub `git@github.com:shdragron/LatentCalib.git`) — calibration as a latent
   variable via projection-agreement `A(Δ)`. Validation ladder passed; full-model
   training + nuScenes leg are GPU-blocked/deferred. See
   `results/memory_snapshot/latentcalib-method-paper.md` and
   `results/METHOD_PAPER_SPEC_KR.md` §8.

## 2. ACTIVE TASK (what the last session was doing)

**BEVFormer vehicle-occupancy BEV segmentation** — to fill the empty "BEVFormer
(Backward-projection, seg)" row in Table 2. Full detail in
**`bevformer_seg/HANDOFF.md`** (read it). One-line state:

- Found & fixed 4 bugs (camera-dim strip → IoU 0.05; visibility ignore→removal;
  reproject z-height; eval-hook in-place-squeeze bs). All verified.
- Training **resumed from epoch_2, running to epoch 24** (sedan, 2× B200).
  Full-val IoU: epoch4=0.4055, epoch6=0.4123 (vs buggy 0.0506). Target ~0.42–0.45.
  wandb project `BEVFormer_Seg`.
- Next: finish sedan → record IoU; repeat for **suv/bus** (swap `labels_dir`);
  then the **seg VP/CTS harness** (USER is bringing it).

## 3. Benchmark models & where each stands (detail in memory_snapshot)

| model | paradigm | env | status (memory file) |
|-------|----------|-----|----------------------|
| BEVFormer (det) | backward | `bevformer-b200` | done, per-cam VP/CTS full |
| BEVFormer (seg) | backward | `bevformer-b200` | **training now** (this task) |
| BEVDepth | forward | `bevdepth-b200` | done; `bevdepth-carla-sedan-result.md` |
| BEVDet | forward | `bevdet-b200` | `bevdet-carla-retrain.md` |
| PETR / CAPE | sparse | `legacy-mmdet140-b200` | `petr-cape-carla-setup.md` |
| DETR3D | sparse | — | `detr3d-carla-setup.md`, per-cam full |
| DFA3D | backward+depth | — | `dfa3d-carla-setup.md`, VP/CTS full |
| PD-BEV | forward+DG | `pdbev-b200` | `pdbev-vp-cts-results.md`, VP infer running |

Fair-comparison settings per model: `results/memory_snapshot/bev-fair-comparison-matrix.md`.

## 4. Key analysis findings (don't re-derive)

- **VP mechanism** (`vp-cross-model-mechanism-finding.md`): robustness splits by
  "does extrinsic gate feature sampling?" gates-sampling (BEVFormer/DETR3D):
  EXT≈IMG, CAL recovers; extract-then-place (CAPE/BEVDet/BEVDepth): EXT≫IMG,
  CAL-pitch collapses. NOT depth (CAPE disproves).
- **CRITICAL DATA BUG** (`vp-carla-vr-frame-2x-misalignment.md`): geobev frame N
  == carla_VR frame 2N. VR/CR image-swap builders must use frame `2N`. Committed
  VP VR/CR numbers are invalid (inflated collapse). ER/Normal/CTS unaffected.
- §5 analysis: `results/SECTION5_ANALYSIS_KR.md` (v7.1 final).

## 5. Live background jobs at handoff (will NOT survive a machine move)

- seg training (this task), 2 procs, `bevformer-b200`, ~epoch 8/24.
- BEVDepth `carla_sedan_extrinaug` (Table-3 extrinsic-aug baseline), **epoch ~19/24,
  val/NDS ~0.508**, ~14 h in (see SESSION UPDATE §D for impl + eval-when-done).
- `bev_det_benchmark/pdbev_vp_infer.py` (PD-BEV VP inference), ~13 h in.

On the new machine these must be **relaunched** (their work_dirs/checkpoints only
help if copied over).

## 6. Repos & data

- Repos under `/home/hanyan_arch/viewpoint/`: `BEVFormer/` (primary working dir;
  also hosts `BEVDepth/`, `bev_det_benchmark/`), `LatentCalib/`, `CAPE/`, `PETR/`,
  `VIZ/`, `docs/`.
- Shared data on **`/NHNHOME/WORKSPACE/0526040099_A/jeongtae/`**: `carla_geobev`
  (images), `carla_geobev_labels/gaussianlss/{sedan,suv,bus}(+_eval)` (seg GT),
  per-vehicle nuScenes-style DBs for detection. If `/NHNHOME` isn't mounted on the
  new box, fix path constants in configs/datasets.
- Conda envs live in `/NHNHOME/.../giyong/miniconda3/envs/` — each model's env
  setup is in its memory_snapshot file.

## 7. Pointers (read in this order for the active task)

1. `bevformer_seg/HANDOFF.md` — full seg task (bugs, geometry, run commands).
2. `bevformer_seg/NOTES.md` — locked low-level facts.
3. `results/memory_snapshot/MEMORY.md` — whole-project index → individual notes.
4. `results/SEGMENTATION_RESULTS.md`, `results/BENCHMARK_SUMMARY.md` — paper tables.

> After the move, re-establish persistent memory: copy
> `results/memory_snapshot/*.md` back into the new machine's
> `~/.claude/projects/<project>/memory/` so recall works again.
