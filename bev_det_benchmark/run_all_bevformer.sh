#!/usr/bin/env bash
# Run the full BEVFormer detection-robustness benchmark sequentially:
#   1) CTS  (7 runs)         2) VP  (631 cells, frames-per-scene=2)
# Sequential so we never stack two evals on top of the training job (2-way GPU
# contention with training, not 3-way). Meant to run inside tmux `bev_eval`.
set -e
cd /home/hanyan_arch/viewpoint/BEVFormer
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevformer-b200
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

CFG=projects/configs/bevformer/bevformer_tiny_carla.py
CKPT=work_dirs/bevformer_tiny_carla_sedan/latest.pth
TAG=tiny_sedan

echo "================ CTS START $(date) ================"
python bev_det_benchmark/eval_cts_det.py \
    --config "$CFG" --ckpt "$CKPT" --ngpu 1 --tag "$TAG" \
    2>&1 | tee bev_det_benchmark/out/cts_driver.log || echo "CTS exited non-zero"

echo "================ VP START $(date) ================"
PORT=29596 bash bev_det_benchmark/run_vp_bevformer.sh \
    --config "$CFG" --ckpt "$CKPT" \
    --frames-per-scene 2 --protocol both --tag "$TAG" \
    2>&1 | tee bev_det_benchmark/out/vp_driver.log || echo "VP exited non-zero"

echo "================ ALL DONE $(date) ================"
