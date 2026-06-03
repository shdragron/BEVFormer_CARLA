#!/usr/bin/env bash
# Launch the in-process VP robustness driver (loads BEVFormer once, loops the
# 631-cell grid). Extra args are forwarded to eval_vp_robustness_det.py.
#
#   CUDA_VISIBLE_DEVICES=1 run_vp_bevformer.sh [--frames-per-scene N --tag ... ]
#
# No `set -u`: conda's cuda-nvcc activate.d hook references unset NVCC_PREPEND_FLAGS.
set -e

BEVF_ROOT=/home/hanyan_arch/viewpoint/BEVFormer
ENV=bevformer-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
fi
cd "$BEVF_ROOT"

PORT=${PORT:-$((29590 + RANDOM % 200))}
PYTHONPATH="$BEVF_ROOT:$PYTHONPATH" python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port="$PORT" \
    bev_det_benchmark/eval_vp_robustness_det.py "$@"
