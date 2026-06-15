# PD-BEV CARLA — HANDOFF (read this first)

Self-contained state for continuing the PD-BEV benchmark work on a **new machine**.
Written 2026-06-15. Authoritative numbers are in this dir (`results/PDBEV/`); this file
explains what is done, what is in flight, what transfers vs is local-only, and how to resume.

---

## 0. One-paragraph context

Project = camera-geometry robustness benchmark of BEV detectors/segmenters on the CARLA
"geobev" per-vehicle dataset (sedan / suv / bus), for a RoboGeo-style paper. Two studies:
**VP** (viewpoint robustness on the sedan rig, perturb yaw/pitch/roll) and **CTS**
(cross-platform transfer: sedan-trained model deployed on suv/bus). This thread added
**PD-BEV** (Generalizable-BEV, AAAI'25 — perspective-debiasing domain generalization on
BEVDepth) as the **`+PD-BEV` row of the robustness-baselines table (Table 2)**, the
detection counterpart of the `+EAFormer` segmentation remedy. Sibling rows: `BEVDepth`
(base, done) and `+Extrinsic Aug.` (a CONCURRENT session is training `carla_sedan_extrinaug.py`).

---

## 1. THE DELIVERABLE — Table 2 `+PD-BEV` row

Metrics (verified against BEVDepth's committed row): **mVRS = 1/7 mVRS = (6·mRRS_percam +
RRSALL_allcam)/7** (NOT ½(mRRS+RRSALL)); **CTS-Cal = CTS_CAL × 100**; **P_Normal = full-val
sedan NDS**. (1/7 reproduces BEVDepth 90.2/82.8/87.4 exactly.)

```latex
% INTERIM (768-subset; full-val pending, expect ≤~0.3pt drift)
\,+\,PD-BEV~\cite{pdbev}  & 0.5601 & 91.4 & 88.9 & 95.1 & 55.9 & 41.8 \\
```

| Method | P_Normal | Ext | Img | Cal | CTS-Cal SUV | Bus |
|---|---|---|---|---|---|---|
| BEVDepth (base) | 0.5354 | 90.2 | 82.8 | 87.4 | 5.7 | 16.6 |
| **+PD-BEV** | **0.5601** | **91.4** | **88.9** | **95.1** | **55.9** | **41.8** |

Story: PD-BEV improves every column; **Cal flips above Ext** (95.1>91.4, BEVDepth has Cal<Ext
— DG lets the model use correct calibration) and **CTS-Cal jumps ~10× (SUV) / ~2.5× (Bus)**
(DG closes the cross-platform gap, its design goal).

CAVEAT to state in the paper: PD-BEV uses the **native-384 recipe** (the BEVDepth-matched
256/[2,58] recipe COLLAPSES PD-BEV from scratch on suv/bus — see §5). So `+PD-BEV` bundles
the DG method with that recipe change; P_Normal 0.5601 > BEVDepth 0.5354 partly reflects it.
CTS is self-normalized (fraction of each method's own oracle) so the transfer-gain is valid.

---

## 2. DONE & COMMITTED (transfers via git clone)

Repo **shdragron/BEVFormer_CARLA** @ `3cdff80` (branch master). Repo
**shdragron/PD-BEV_CARLA** (= the Generalizable-BEV fork) @ `cd6957a` (branch main).

- **Oracles (6-class NDS, vis≥2, native-384):** sedan384 **0.5601**/mAP6 0.5683,
  suv **0.5464**/0.5555, bus **0.6064**/0.6291. (sedan384 = +0.043 over old 256 sedan 0.5170.)
- **CTS (final):** `results/PDBEV/cts/eval_cts.{csv,json,summary.txt}`.
  suv CTS NORMAL/EXT/IMG/CAL = .966/.733/.597/.559; bus = .789/.377/.009/.418.
- **VP subset768 (interim, final for the 768-matched cross-model table):**
  `results/PDBEV/vp/eval_vp.subset768.{json,csv,summary.txt}`. NDS_Normal 0.5598;
  1/7 mVRS Ext/Img/Cal = 91.4/88.9/95.1 (mRRS/RRSALL: ER .949/.702, VR .938/.596, CR .975/.808).
- **Eval drivers** (in `bev_det_benchmark/`, committed): `pdbev_vp_infer.py`,
  `pdbev_vp_score.py`, `pdbev_vp_score_sharded.sh`, `pdbev_cts_run.sh`, `pdbev_cts_format.py`,
  `run_vp_full_pdbev.sh`, and edits to `pdbev_dump_val.py` (`--ann-file`) +
  `pdbev_score_carla.py` (`os.chdir(BEVDET)`).
- **PD-BEV configs** (in Generalizable-BEV, committed `cd6957a`): train
  `configs/PDBEV/pdbev-r50-cbgs-CARLA-dg-{sedan384,suv,bus}.py`; eval
  `configs/bevdet_our/bevdepth-r50-cbgs-pc-carla-{sedan384,suv,bus}.py` (arch 384×704/[1,100,1]).

## 3. LOCAL-ONLY (does NOT transfer via git — rsync or regenerate)

- **Checkpoints** `Generalizable-BEV/work_dirs/pdbev-r50-cbgs-CARLA-dg-{sedan384,suv,bus}/epoch_24.pth`
  (~1.1 GB each, gitignored). NEEDED for any eval. rsync these or retrain (24 ep, ~25h each).
- **VP dets** `bev_det_benchmark/out/vp_pdbev_sedan384{,_full}/dets/*.pkl` (gitignored).
- **Data**: `Generalizable-BEV/data/` + `BEVDet/data/` are symlinks to the geobev/carla_VR
  capture under `/NHNHOME/WORKSPACE/0526040099_A/...`. The eval pkls
  `BEVDet/data/bevdet_infos/{sedan,suv,bus}_infos_{train,val}.pkl` are the schema source.
- **Conda envs** `pdbev-b200`, `bevdet-b200` (under giyong miniconda) — rebuild on new machine.
- **Memory** `~/.claude/.../memory/*.md` — local; key facts are embedded here + in pdbev READMEs.

## 4. IN FLIGHT (will DIE on machine move) — full-3792 VP

`tag=pdbev_sedan384_full` (fps=79 = all 3792 frames, 631 cells) was running at ~278/631 when
this was written, to replace the 3 interim mVRS values with exact full numbers. Orchestration
(all under `/tmp`, machine-local): `run_vp_full_pdbev.sh` (auto-restart 2-shard infer) +
`pdbev_vpfull_score_chain.sh` (waits → 8-shard score). On the new machine these are GONE.

**To finish full VP on the new machine** (needs the 3 ckpts + data present):
```bash
# infer (pdbev-b200, from Generalizable-BEV repo root). resumable: skips existing dets.
bash bev_det_benchmark/run_vp_full_pdbev.sh         # writes out/vp_pdbev_sedan384_full/dets
# score (bevdet-b200): 8-shard parallel CPU, then merge -> 1/7 mVRS Table-2 row
TAG=pdbev_sedan384_full N=8 bash bev_det_benchmark/pdbev_vp_score_sharded.sh
cat bev_det_benchmark/out/vp_pdbev_sedan384_full/eval_vp_summary.txt   # -> Ext/Img/Cal
```
Then copy the full eval_vp.* into `results/PDBEV/vp/` (as `eval_vp.{json,csv,summary.txt}`,
keeping subset768 alongside) and replace the 3 interim mVRS in the Table-2 row + commit.
**If full VP can't be re-run, the interim 768-subset values (91.4/88.9/95.1) are publishable**
(full≈subset ≤~0.3pt, as held for BEVDepth).

## 5. KEY GOTCHAS / DECISIONS (do not relearn the hard way)

- **native-384 is mandatory.** PD-BEV depth loss supervises ONLY GT-box-center *virtual*
  depths (~5% of pixels). BEVDepth-matched 256×704+[2,58,0.5] → from-scratch collapse on
  tall rigs (suv/bus): softmax→bin1, ego-ring BEV, NDS 0. Fix = input 384×704 + depth
  grid [1.0,100.0,1.0]. (Full mechanism in `results/PDBEV/README.md` §Recipe.)
- **2-env eval.** PD-BEV inference runs in `pdbev-b200`; the verified `CarlaNuScenesDataset`
  6-class NDS only exists in `bevdet-b200`. Hence infer→dump dets by token (pdbev-b200),
  then score (bevdet-b200). `pdbev_score_carla.py` does `os.chdir(BEVDET)` so relative
  `ann_file` resolves from any cwd.
- **Condition swaps reused verbatim.** PD-BEV reads the IDENTICAL sedan val pkl (same inode
  as BEVDet's) + same `cams[CAM]['data_path'/'sensor2ego_*']`; `PrepareImageInputs_UDA`
  opens `data_path` directly (no data_root join). So `build_condition_pkls_bevdet.make_vp_infos`
  / `make_cts_pkl` apply unchanged — incl. the **carla_VR frame-2N fix** (geobev frame N ==
  carla_VR frame 2N).
- **1/7 mVRS** is the headline (6 per-cam + 1 all-cam, equal-weighted), NOT the driver's
  default ½(mRRS+RRSALL). The scorer now prints both + a ready "Table-2 row" line.
- **Concurrent session contention.** `carla_sedan_extrinaug.py` (the +ExtAug row) shares the
  GPUs and once SIGTERM'd (143) both VP infer shards at ~cell 275 simultaneously. Drivers are
  resume-safe; `run_vp_full_pdbev.sh` auto-restarts. Don't run the 6-8 CPU scorers AND the
  GPU infer dataloaders at once — they oversubscribe BLAS threads even on 72 cores.

## 6. ENV REBUILD (new machine)

`pdbev-b200` = clone `bevdet-b200` + `cp CoIn3D/BEVDet/setup.py Generalizable-BEV/` +
`pip install mmsegmentation==0.30.0` + patch `mmdet3d/__init__.py` mmcv max 1.7.0→1.7.1 +
`CUDA_HOME=$CONDA_PREFIX CC=gcc-13 CXX=g++-13 TORCH_CUDA_ARCH_LIST=10.0 pip install -e . --no-deps --no-build-isolation`.
`bevdet-b200` = the base (torch 2.11+cu128 sm_100, mmcv-full 1.7.1, mmdet 2.28.2, mmdet3d
1.0.0rc4, nuscenes-devkit). Always run with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
NVIDIA_TF32_OVERRIDE=0`. (Full recipe was in memory `pdbev-generalizable-bev-b200-env`.)

## 7. BROADER PROJECT (for context)

Other detectors already benchmarked (VP+CTS, in `results/<Model>/`): BEVFormer, BEVDepth,
BEVDet, CAPE, DETR3D, DFA3D. Cross-model analysis: `results/VP_CROSS_MODEL_ANALYSIS.md`,
`results/SECTION5_ANALYSIS_KR.md` (defines 1/7 mVRS; has the 90.2/82.8/87.4 etc.). The Table-2
also has segmentation rows (CVT / +ExtAug / +EAFormer) and `+Extrinsic Aug.` detection — those
are separate work, not this thread.

## 8. GIT / IDENTITY RULES (strict)

Push ONLY to **shdragron** remotes (`shdragron/BEVFormer_CARLA`, `shdragron/PD-BEV_CARLA`),
never upstream (fundamentalvision / EnVision-Research). Commit as
`shdragron <shdragron@hanyang.ac.kr>`. **No Claude co-author line.** Never commit large
binaries (.pth/.ckpt/dets pkls/qual jpgs/work_dirs/data symlinks) — already gitignored.
