---
name: vp-eval-performance-care
description: VP robustness eval (631 cells) is very time-expensive — optimize the BEVDepth VP driver as heavily as the BEVFormer one
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d6e20ce2-7fd8-462d-9469-4a1b57b88da9
---

The user heavily optimized the BEVFormer VP robustness benchmark because per-cell eval is slow and there are 631 cells. When building the BEVDepth VP driver ([[bevdepth-b200-env-and-eval]]), mirror that performance care — do NOT naively launch 631 full evals.

**Why:** VP = `1 + 3 conditions × 3 axes × 10 signed-mags × 7 protocols = 631` cells; a full 3792-frame eval per cell is enormous. The BEVFormer driver (`bev_det_benchmark/eval_vp_robustness_det.py` + `run_vp_full.sh`) optimizes with: model + NuScenes GT DB loaded ONCE in-process (avoid ~100s weight reload ×631); `--fast` ProcessPool JPEG decode overlapping GPU; eval pipelined on a CPU thread vs the next cell's GPU infer (per-cell wall ≈ max(infer, eval)); 2-GPU cell-sharding (`--shard i/n` + merge); NUMA pinning (shard0→node0, shard1→node1); tmpfs RAM-staging of images (`VP_STAGE_ROOT`, decode from RAM not Lustre); `--frames-per-scene N` frozen subset to bound wall-clock.

**How to apply:** for BEVDepth VP — load model once (`load_from_checkpoint`) + GT DB once; large eval batch (64, BatchNorm is batch-invariant in eval → NDS unchanged); measure per-cell throughput and optimize before scaling to the full grid; start with a frames-per-scene subset smoke (2-3 cells); add 2-GPU sharding for the full run. BEVDepth is easier than BEVFormer here: preds are plain numpy (no torch-tensor FD-reducer issue), no deformable-attn determinism hack needed, and batch can be large. Lustre is the I/O bottleneck — few dataloader workers (4) or RAM-stage; more workers thrash the FS.
