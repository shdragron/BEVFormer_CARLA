#!/usr/bin/env bash
# Full-data BEVDepth VP robustness on 2 GPUs (cell-sharded), RAM-staged, merged.
# BEVDepth analogue of run_vp_full.sh.
#   frames-per-scene 79 -> all 3792 val samples ; protocol both -> 631 cells
#   - model + GT DB loaded ONCE per shard (in-process cell loop)
#   - VP_STAGE_ROOT (tmpfs) -> decode from RAM, GPU-bound (atomic stage = both
#     shards may stage concurrently safely); large eval batch
#   - shard 0/2 on GPU0, 1/2 on GPU1 (independent, no cross-rank collectives);
#     each shard reruns Normal (the shared RRS denominator)
#   - NUMA pin per shard if numactl is present (both GPUs sit on node0 by default
#     -> contention; pin shard1 to node1 so each owns a full core set)
#
# Usage: run_vp_full_bevdepth.sh [FRAMES_PER_SCENE] [TAG] [BATCH] [WORKERS]
set -e

BEVF=/home/hanyan_arch/viewpoint/BEVFormer
BENCH="$BEVF/bev_det_benchmark"
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200

FRAMES=${1:-79}
TAG=${2:-bevdepth_sedan}
BATCH=${3:-32}
WORKERS=${4:-12}
# Retrained sedan ckpt (NDS 0.5354), NOT the driver's default old undertrained
# baseline. Override with CKPT=... ; path is relative to the BEVF root.
CKPT=${CKPT:-BEVDepth/outputs/cts_ckpt_sedan.ckpt}

export VP_STAGE_ROOT=${VP_STAGE_ROOT:-/tmp/vpstage_bd}
# OMP modest: 2 shards x WORKERS decode + the in-process NuScenes eval shouldn't
# oversubscribe the cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

A="--frames-per-scene $FRAMES --protocol both --batch $BATCH --workers $WORKERS --tag $TAG --ckpt $CKPT"
OUT="$BENCH/out"
mkdir -p "$OUT"

NUMA0=""; NUMA1=""
if command -v numactl >/dev/null 2>&1; then
    NUMA0="numactl --cpunodebind=0"; NUMA1="numactl --cpunodebind=1"
fi

echo "================ VP FULL (BEVDepth) START $(date)  frames=$FRAMES batch=$BATCH stage=$VP_STAGE_ROOT ================"
CUDA_VISIBLE_DEVICES=0 $NUMA0 bash "$BENCH/run_vp_bevdepth.sh" $A --shard 0/2 \
    > "$OUT/vp_${TAG}_shard0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $NUMA1 bash "$BENCH/run_vp_bevdepth.sh" $A --shard 1/2 \
    > "$OUT/vp_${TAG}_shard1.log" 2>&1 &
P1=$!
echo "shard0 pid=$P0 (GPU0)  shard1 pid=$P1 (GPU1)  logs: $OUT/vp_${TAG}_shard{0,1}.log"
wait $P0; E0=$?
wait $P1; E1=$?
echo "shard0 exit=$E0  shard1 exit=$E1"

echo "================ MERGE $(date) ================"
bash "$BENCH/run_vp_bevdepth.sh" $A --merge
echo "================ VP FULL DONE $(date) ================"
cat "$OUT/vp_${TAG}/eval_vp_summary.txt" 2>/dev/null
