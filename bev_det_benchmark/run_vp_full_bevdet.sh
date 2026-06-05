#!/usr/bin/env bash
# Full-data BEVDet VP robustness on 2 GPUs (cell-sharded) with tmpfs image staging
# + NUMA pinning, then merge. Mirrors run_vp_full(_bevdepth).sh.
#   - frames-per-scene 79 -> all val samples ; protocol both -> 631 cells
#   - VP_STAGE_ROOT (tmpfs): every image each shard reads is copied to RAM once,
#     then cam data_paths are rewritten to the staged paths -> decode is GPU-bound.
#   - shard 0/2 on GPU0 (NUMA node 0), 1/2 on GPU1 (node 1): independent jobs.
#
# Usage: run_vp_full_bevdet.sh [FRAMES] [TAG] [BATCH] [WORKERS]
set -e
BEVF=/home/hanyan_arch/viewpoint/BEVFormer
BENCH=$BEVF/bev_det_benchmark

FRAMES=${1:-79}
TAG=${2:-bevdet_sedan}
BATCH=${3:-16}
WORKERS=${4:-8}
CFG=BEVDet/configs/bevdet/carla/bevdet-r50-carla.py
CKPT=BEVDet/work_dirs/bevdet-r50-carla_sedan/epoch_24.pth
A="--config $CFG --ckpt $CKPT --frames-per-scene $FRAMES --protocol both --batch $BATCH --workers $WORKERS --tag $TAG"
OUT=$BENCH/out
mkdir -p "$OUT"

# RAM-staged images (tmpfs): decode reads from RAM, not Lustre -> GPU-bound.
export VP_STAGE_ROOT=${VP_STAGE_ROOT:-/tmp/vpstage_bevdet}
# OMP/MKL modest: 2 shards x decode workers; the eval (devkit) is not BLAS-bound.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# NUMA: both B200s sit on node 0 by default, so pin each shard's CPU work to a
# distinct node (graceful fallback if numactl is absent).
NUMA0=""; NUMA1=""
if command -v numactl >/dev/null 2>&1; then
    NUMA0="numactl --cpunodebind=0"
    NUMA1="numactl --cpunodebind=1"
fi

echo "================ BEVDet VP FULL START $(date)  (2-GPU, staged, NUMA) ================"
echo "VP_STAGE_ROOT=$VP_STAGE_ROOT  FRAMES=$FRAMES  BATCH=$BATCH  WORKERS=$WORKERS"
CUDA_VISIBLE_DEVICES=0 $NUMA0 bash "$BENCH/run_vp_bevdet.sh" $A --shard 0/2 \
    > "$OUT/vp_${TAG}_shard0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $NUMA1 bash "$BENCH/run_vp_bevdet.sh" $A --shard 1/2 \
    > "$OUT/vp_${TAG}_shard1.log" 2>&1 &
P1=$!
echo "shard0 pid=$P0 (GPU0)  shard1 pid=$P1 (GPU1)"
wait $P0; E0=$?
wait $P1; E1=$?
echo "shard0 exit=$E0  shard1 exit=$E1"

echo "================ MERGE $(date) ================"
bash "$BENCH/run_vp_bevdet.sh" $A --merge
echo "================ BEVDet VP FULL DONE $(date) ================"
cat "$OUT/vp_${TAG}/eval_vp_summary.txt" 2>/dev/null
