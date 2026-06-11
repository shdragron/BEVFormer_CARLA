#!/usr/bin/env bash
# Run one CAPE eval pass on a condition-specific val pkl and print the
# deterministic `[CARLA-EVAL] 6-class mAP=.. NDS=..` line the CTS/VP drivers scrape.
#
#   run_cape.sh <CONFIG> <CKPT> <NGPU> <COND_PKL> [extra cfg-options...]
#
# Same 4-arg interface as run_bevformer.sh, BUT:
#   * env = legacy-mmdet140-b200 (CAPE/PETR/DETR3D env), root = viewpoint/BEVFormer/CAPE
#   * SINGLE-GPU tools/test.py (NOT dist_test.sh): the sparse detectors hit a
#     mmcv-1.4 DataContainer scatter bug in multi_gpu_test with the
#     MultiScaleFlipAug3D test pipeline (verified on DETR3D); MMDataParallel
#     (single_gpu_test) unwraps the DataContainer cleanly. NGPU is accepted for
#     interface compatibility but ignored (always 1 GPU).
#   * PYTHONPATH=<root> so `projects.mmdet3d_plugin` imports under plain `python`
#     (plain `python tools/test.py` puts tools/ on sys.path, not the repo root).
#
# No `set -u`: conda's cuda-nvcc activate.d hook references unset NVCC_PREPEND_FLAGS.
set -e

CONFIG=$1
CKPT=$2
NGPU=$3          # ignored (single-GPU); kept for run_one() interface parity
COND_PKL=$4
shift 4

CAPE_ROOT=/home/hanyan_arch/viewpoint/BEVFormer/CAPE
ENV=legacy-mmdet140-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
fi
cd "$CAPE_ROOT"

# tools/test.py does NOT append --eval (unlike dist_test.sh), so pass it here.
# data.test.ann_file -> the condition pkl; data_root -> CAPE's carla_geobev symlink.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH="$CAPE_ROOT:${PYTHONPATH:-}" \
    python tools/test.py "$CONFIG" "$CKPT" --eval bbox \
        --cfg-options data.test.ann_file="$COND_PKL" \
                      data.test.data_root="$CAPE_ROOT/data/nuscenes/" \
                      data.workers_per_gpu=4 "$@"
