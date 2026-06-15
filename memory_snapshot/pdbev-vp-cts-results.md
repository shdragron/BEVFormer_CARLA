---
name: pdbev-vp-cts-results
description: "PD-BEV CARLA VP/CTS benchmark results + 2-env eval pipeline (pdbev_vp_infer/score, pdbev_cts_run). Oracles, 1/7 mVRS, CTS table. For the robustness-baselines Table 2 +PD-BEV row."
metadata: 
  node_type: memory
  type: project
  originSessionId: 09f0506b-7473-403c-8e74-a3ace5c5a83b
---

PD-BEV (Generalizable-BEV / BEVDepth_DG) is the **`+PD-BEV` detection row** of the
robustness-baselines table (vs the BEVDepth base row). See [[pdbev-generalizable-bev-b200-env]]
for the env + the native-384 collapse/fix. All 3 trained native-384 (384×704, depth[1,100,1]).

**Oracles (6-class NDS, vis≥2):** sedan384 **0.5601**, suv **0.5464**, bus **0.6064**
(sedan384 = +0.043 over old 256/[2,58] sedan 0.5170).

**VP 1/7 mVRS (768-subset, fps16, NDS_Normal 0.5598)** — headline metric is
**1/7 mVRS = (6·mRRS_percam + RRSALL_allcam)/7** (NOT ½(mRRS+RRSALL); verified it
reproduces BEVDepth's committed 90.2/82.8/87.4). ER=Ext VR=Img CR=Cal:

| | Ext | Img | Cal |
|--|--|--|--|
| BEVDepth base | 90.2 | 82.8 | 87.4 |
| **PD-BEV** | **91.4** | **88.9** | **95.1** |

PD-BEV improves all three; **Cal flips ABOVE Ext** (95.1>91.4) where BEVDepth has Cal<Ext —
perspective-debiasing lets it use the correct calibration instead of staying collapsed.
Full-3792 VP DONE (tag=pdbev_sedan384_full): NDS_Normal 0.5605, 1/7 mVRS Ext/Img/Cal 91.4/88.9/95.1 == subset768 (confirmed identical). results/PDBEV/vp/eval_vp.* (full) + *.subset768.*. P_Normal for Table-2 = 0.5605.

**CTS (sedan384 → target, CTS=NDS_cond/P_TARGET, IMG primary):**
suv NORMAL/EXT/IMG/CAL = .966/.733/**.597**/.559 ; bus = .789/.377/**.009**/.418.
PD-BEV **massively beats the BEVDepth base** (suv CTS-IMG 0.103→0.597, CTS-CAL 0.057→0.559;
bus CTS-CAL 0.166→0.418) — the DG method closes the cross-platform gap. bus-IMG collapses
(sedan extrinsic on bus-viewpoint image unrecoverable) but correct bus ext at CAL recovers it.
**Table-2 CTS-Cal column = CTS_CAL×100: PD-BEV suv 55.9, bus 41.8.**

**Eval pipeline (2-env — CarlaNuScenesDataset NDS only exists in bevdet-b200):**
`bev_det_benchmark/`: `pdbev_vp_infer.py` (pdbev-b200: BEVDepth_DG loaded once, loop 631 cells
reusing `build_condition_pkls_bevdet.make_vp_infos`, dump per-cell dets by token) →
`pdbev_vp_score.py` (bevdet-b200: CarlaNuScenesDataset 6-class NDS, subset-GT filtered,
`--shard i/n` cell-sharded parallel CPU scoring + `--merge`, prints 1/7 mVRS). CTS:
`pdbev_cts_run.sh` + `pdbev_cts_format.py`. `pdbev_dump_val.py` got `--ann-file`;
`pdbev_score_carla.py` got `os.chdir(BEVDET)` (relative ann_file resolves any cwd).
PD-BEV reads the **identical sedan val pkl** (same inode) + same `cams[CAM][data_path/
sensor2ego_*]` as BEVDet, and `PrepareImageInputs_UDA` opens `data_path` directly (no
data_root join) → make_vp_infos/make_cts_pkl swaps apply verbatim. Committed
shdragron/BEVFormer_CARLA `3cdff80`; results in `results/PDBEV/{vp,cts}/`.

**Gotcha (2026-06-15):** a concurrent session runs `carla_sedan_extrinaug.py` (the BEVDepth
**+Extrinsic-Aug** row) on the shared GPUs; it SIGTERM'd (143) both my VP infer shards at
~cell 275 once. Drivers are resume-safe (skip existing dets); `run_vp_full_pdbev.sh` wraps
infer in an auto-restart loop. 6 parallel CPU scorers + 12 infer dataloaders oversubscribe
even on 72 cores (numpy BLAS threads) → pause one when running both.
