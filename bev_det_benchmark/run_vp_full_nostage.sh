#!/usr/bin/env bash
# Full-data VP robustness on 2 GPUs (cell-sharded), NDS-exact FAST path, then merge.
# SAME as run_vp_full.sh but WITHOUT the tmpfs RAM-staging step: fast_decode reads
# images straight from data/nuscenes/sweeps (Lustre) via the original relative
# data_path (VP_STAGE_ROOT unset -> _resolve returns the path unchanged). Use this
# when /tmp/vpstage has not been pre-populated. Bytes/NDS identical to the staged
# run; only the read source (network vs RAM) differs -> slightly slower per cell.
#
# Usage: run_vp_full_nostage.sh [WORKERS] [TAG]
set -e
cd /home/hanyan_arch/viewpoint/BEVFormer
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevformer-b200

WORKERS=${1:-6}                       # 6/shard = 12 Lustre readers (<16 thrash bound)
TAG=${2:-tiny_sedan}
FPS=${3:-79}                          # frames-per-scene (79=all; 16=1/5 even-stride)
CFG=projects/configs/bevformer/bevformer_tiny_carla.py
CKPT=work_dirs/bevformer_tiny_carla_sedan/latest.pth
A="--config $CFG --ckpt $CKPT --frames-per-scene $FPS --protocol both --fast --batch 1 --workers $WORKERS --tag $TAG"
OUT=bev_det_benchmark/out

# NO VP_STAGE_ROOT -> read from Lustre. Keep OMP=1 + NUMA pinning (still helps the
# 2 shards' decode/eval CPU contention; both GPUs are on NUMA node 0).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset VP_STAGE_ROOT || true
echo "NO-STAGE (Lustre reads)  workers=$WORKERS  frames-per-scene=$FPS  OMP=$OMP_NUM_THREADS"

echo "================ VP FULL (no-stage) START $(date)  (2-GPU, FAST+pipeline, workers $WORKERS, NUMA-pinned) ================"
CUDA_VISIBLE_DEVICES=0 PORT=29710 numactl --cpunodebind=0 \
    bash bev_det_benchmark/run_vp_bevformer.sh $A --shard 0/2 \
    > $OUT/vp_${TAG}_shard0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 PORT=29711 numactl --cpunodebind=1 \
    bash bev_det_benchmark/run_vp_bevformer.sh $A --shard 1/2 \
    > $OUT/vp_${TAG}_shard1.log 2>&1 &
P1=$!
echo "shard0 pid=$P0 (GPU0)  shard1 pid=$P1 (GPU1)"
wait $P0; E0=$?
wait $P1; E1=$?
echo "shard0 exit=$E0  shard1 exit=$E1"

echo "================ MERGE $(date) ================"
python bev_det_benchmark/eval_vp_robustness_det.py --merge --tag $TAG --protocol both
echo "================ VP FULL (no-stage) DONE $(date) ================"
cat $OUT/vp_${TAG}/eval_vp_summary.txt 2>/dev/null
