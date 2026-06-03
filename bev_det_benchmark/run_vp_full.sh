#!/usr/bin/env bash
# Full-data VP robustness on 2 GPUs (cell-sharded), NDS-exact FAST path, then merge.
#   - frames-per-scene 79  -> all 3792 val samples ; protocol both -> 631 cells
#   - --fast: ProcessPool JPEG decode (overlaps GPU) + manual img_metas, no
#             dataloader/DDP; --batch 1 keeps NDS bit-identical to the standard path
#   - eval pipelined on a CPU thread vs the next cell's GPU infer (per-cell wall
#     ~= max(infer, eval), not their sum)
#   - workers = decode-pool size (16 good; >16 thrashes Lustre)
#   - shard 0/2 on GPU0, 1/2 on GPU1 (independent jobs, no cross-rank collectives)
#
# Usage: run_vp_full.sh [WORKERS] [TAG]
set -e
cd /home/hanyan_arch/viewpoint/BEVFormer
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevformer-b200

# workers=8: shm decode is fast, so 8 keeps the GPU fed (38>24 samples/s) while
# leaving cores for the concurrent eval process (16 would oversubscribe + starve it).
WORKERS=${1:-8}
TAG=${2:-tiny_sedan}
CFG=projects/configs/bevformer/bevformer_tiny_carla.py
CKPT=work_dirs/bevformer_tiny_carla_sedan/latest.pth
A="--config $CFG --ckpt $CKPT --frames-per-scene 79 --protocol both --fast --batch 1 --workers $WORKERS --tag $TAG"
OUT=bev_det_benchmark/out

# RAM-staged images (tmpfs): decode reads from RAM, not Lustre -> GPU-bound, NDS-exact.
export VP_STAGE_ROOT=${VP_STAGE_ROOT:-/tmp/vpstage}
# OMP=1: with 2 shards x 8 decode workers, OMP=8 means 16x8=128 BLAS threads on 72
# cores (2x oversubscription -> per-cell 240s). The decode numpy ops (imnormalize)
# and the NuScenes eval are NOT BLAS-bound, so 1 thread each is plenty and removes
# the oversubscription -> infer+eval stay near standalone speed. cv2 also pinned to 1.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "VP_STAGE_ROOT=$VP_STAGE_ROOT  workers=$WORKERS  OMP=$OMP_NUM_THREADS (staged+shm, eval in process)"

# NUMA pinning: BOTH GPUs sit on NUMA node 0, so both shards' CPU work (decode +
# eval) defaults onto node-0's 36 cores -> contention (per-cell 210-255s) while
# node-1's 36 cores idle. Pin shard0->node0, shard1->node1 so each shard owns a full
# 36-core node -> back to single-shard speed (~187s). shard1's GPU1 H2D is then
# cross-node (27MB/sample, negligible vs the CPU win).
echo "================ VP FULL START $(date)  (2-GPU, FAST+pipeline, workers $WORKERS, NUMA-pinned) ================"
CUDA_VISIBLE_DEVICES=0 PORT=29700 numactl --cpunodebind=0 \
    bash bev_det_benchmark/run_vp_bevformer.sh $A --shard 0/2 \
    > $OUT/vp_${TAG}_shard0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 PORT=29701 numactl --cpunodebind=1 \
    bash bev_det_benchmark/run_vp_bevformer.sh $A --shard 1/2 \
    > $OUT/vp_${TAG}_shard1.log 2>&1 &
P1=$!
echo "shard0 pid=$P0 (GPU0)  shard1 pid=$P1 (GPU1)"
wait $P0; E0=$?
wait $P1; E1=$?
echo "shard0 exit=$E0  shard1 exit=$E1"

echo "================ MERGE $(date) ================"
python bev_det_benchmark/eval_vp_robustness_det.py --merge --tag $TAG --protocol both
echo "================ VP FULL DONE $(date) ================"
cat $OUT/vp_${TAG}/eval_vp_summary.txt 2>/dev/null
