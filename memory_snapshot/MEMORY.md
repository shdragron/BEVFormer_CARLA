# Memory Index

> 🟢 **NEW SESSION: START HERE → [HANDOFF.md](HANDOFF.md).** This folder
> (`memory_snapshot/`) is the **self-contained project memory** — `HANDOFF.md` has the live
> state (what's running, 5-model full-VP status, the Extrinsic-Aug experiment, the paper table
> + the BEVDet-CTS fix, next TODOs); the files below are one-fact-each detail notes. After a
> machine move, copy these `*.md` into `~/.claude/projects/<project>/memory/` to restore recall.
> Repo root = wherever `BEVFormer/` lives (was `/home/hanyan_arch/viewpoint/BEVFormer`).

- [📋 HANDOFF — current state / read first](HANDOFF.md) — frame-2N fix, 5-model full-VP, Extrinsic-Aug BEVDepth (running), paper-table verification + BEVDet-CTS fix, TODOs
- [📋 PDBEV_HANDOFF — PD-BEV detection detail](PDBEV_HANDOFF.md) — Table-2 +PD-BEV row (0.5601/91.4/88.9/95.1/55.9/41.8), native-384 + 2-env eval, full-VP resume steps, env rebuild
- [📋 SEG_HANDOFF — BEVFormer seg detail](SEG_HANDOFF.md) — vehicle-occupancy BEV segmentation (CVT / +ExtAug / +EAFormer rows)
- [LatentCalib method paper](latentcalib-method-paper.md) — 2호 논문 확정안: A(Δ) 일치도 1함수=자기보정/게이트/뷰신뢰도, 검증 사다리, /viewpoint/LatentCalib, env 셋업 승인 대기
- [CARLA geobev dataset layout](carla-geobev-dataset-layout.md) — per-vehicle DBs, split.txt, real image path, visibility>=2 filter decision
- [BEVFormer conda env](carla-bevformer-conda-env.md) — use `bevformer-b200` to run the repo
- [BEV fair-comparison matrix](bev-fair-comparison-matrix.md) — per-model VP/CTS settings: BEVDepth/BEVDet 128×128+EMA/CBGS off+DPT depth (separate forks), BEVFormer 50×50, PETRv2/CAPE single-frame no-aug
- [BEVDet dev3.0 on B200 env](bevdet-dev30-b200-env.md) — bevdet-b200 env clone + mmdet2.28/asserts/bev_pool_v2_ext(gcc-13)/weights_only/mmengine/local-rank/numpy fixes
- [BEVDet CARLA retrain](bevdet-carla-retrain.md) — pth/BEVDet provenance (force-pushed mmengine 39aa1b7), retraining dev3.0 BEVDetDepth+DPT 3 vehicles, batch128/lr8e-4, data/bevdet_infos pkls, wandb
- [PETR/CAPE CARLA setup](petr-cape-carla-setup.md) — legacy-mmdet140-b200 env, 6 configs+dataset ported+verified (Test B 1.0), R50-caffe msra ckpt gotcha, training pending
- [DETR3D CARLA setup](detr3d-carla-setup.md) — 6th/last detector (3rd sparse), R50-ImageNet drop-FCOS3D, NO-DN native, FPN+1600×900 kept, verified (projrate 1.0, l2i byte-identical), training pending
- [DFA3D CARLA setup](dfa3d-carla-setup.md) — 7th detector (gates-sampling×depth quadrant). Paradigm=Backward (NOT Sparse; subclasses BEVFormer) = "BEVFormer-tiny + DPT depth" (R50 ImageNet/800×450/BEV50/single-frame, NOT base_DFA3D config). 2-GPU training running, pushed shdragron/DFA3D_CARLA. coords verified (projrate 1.000, depth-align 0.988)
- [BEVDepth CARLA results](bevdepth-carla-sedan-result.md) — all 3 vehicles done: best NDS suv 0.5564 > sedan 0.5407 > bus 0.4279; gotchas: /dev/shm rc=137→file_system+TMPDIR, TF32/cuDNN9.7 depth divergence→fp32, 1-GPU BN16 fails→2-GPU, 2-GPU diverges→grad_clip 35
- [BEVDepth B200 env & eval](bevdepth-b200-env-and-eval.md) — BEVDepth fork env build + eval entry/specifics on B200
- [VP eval performance care](vp-eval-performance-care.md) — VP eval determinism / performance gotchas (fp32, cudnn/TF32 off, batch-invariance, staging)
- [CARLA qual viz ego-frame](carla-qual-viz-ego-frame.md) — GT/pred boxes are EGO-frame (project via inv(sensor2ego), NOT sensor2lidar→1.8m float); results/<Model>/ = cts/+vp/ committed, ckpts/+qual/ gitignored-local; qual mirrors BEVDepth/_viz_qual.py
- [CARLA qual conditions coords](carla-qual-conditions-coords.md) — per-condition (VP/CTS EXT/IMG/CAL) qual: raw gt_boxes are LIDAR-frame, project via inv(sensor2lidar) using each condition's OWN sensor2lidar (VP leaves sensor2ego stale); qual_conditions.py, verified scene-0267, GT-only (pred TODO)
- [VP cross-model mechanism finding](vp-cross-model-mechanism-finding.md) — robustness splits by "does extrinsic gate feature sampling?" (gates-sampling=BEVFormer/DETR3D: EXT≈IMG, CAL recovers; extract-then-place=CAPE/BEVDet/BEVDepth: EXT≫IMG, CAL-pitch collapses). Code-verified. NOT depth — CAPE disproves it
- [VP carla_VR frame 2x misalignment](vp-carla-vr-frame-2x-misalignment.md) — CRITICAL BUG: geobev frame N == carla_VR frame 2N; all VR/CR image-swap builders use frame N → wrong world. Fix=*2. Committed VP VR/CR numbers invalid (inflated collapse). ER/Normal/CTS unaffected.
- [Generalizable-BEV / PD-BEV B200 env](pdbev-generalizable-bev-b200-env.md) — use `pdbev-b200` (clone bevdet-b200 + copied setup.py from CoIn3D/BEVDet + mmseg0.30 + mmcv-patch + --no-deps). AAAI25 domain-generalization on BEVDepth, plug-and-play. User chose this OVER CoIn3D. Built-in CarlaDataset is actually SHIFT-flavored (adapt to geobev quaternion or use DeepAccident converter)
- [PD-BEV VP/CTS results](pdbev-vp-cts-results.md) — Table-2 +PD-BEV row: oracles sedan384 0.5601/suv 0.5464/bus 0.6064; VP 1/7 mVRS 91.4/88.9/95.1 (Cal flips>Ext); CTS-Cal suv 55.9/bus 41.8 (DG closes gap vs BEVDepth). 2-env eval pipeline (pdbev_vp_infer/score, cell-sharded). full-VP in progress
- [CoIn3D B200 env](coin3d-b200-env.md) — use `coin3d` (clone of bevdet-b200, NOT README cu116); +mmseg0.30 +CoIn3D/BEVDet(--no-deps) +nuscenes-devkit1.1.11 +diff-gaussian(3DGS) +spconv-cu120; 2 source patches (mmcv max 1.7.1, devkit egg-info filter); CUDA_HOME=$CONDA_PREFIX gcc-13 arch=10.0
- [VP full-run checkpoint/resume](vp-full-run-checkpoint-resume.md) — full-val(3792) no-stage VP crashes ~6h in (dataloader worker OOM); driver now per-cell fsync checkpoints + resume-skip (same --tag/--shard) + file_system strategy; recover pre-checkpoint crash from shard log ([CARLA-METRICS-JSON]+[VP] pairs).
