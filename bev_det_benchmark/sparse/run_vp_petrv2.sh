#!/usr/bin/env bash
# Launch the in-process SPARSE VP driver (PETRv2 (sparse)) — one shard or
# --merge. Single-GPU in-process cell loop via MMDataParallel (avoids the mmcv-1.4
# multi_gpu_test DataContainer bug). Run from the model repo so relative cam
# data_paths (data/nuscenes/...) and `projects.mmdet3d_plugin` resolve.
#
#   run_vp_cape.sh --config <cfg> --ckpt <ckpt> [--shard i/n] \
#                  [--frames-per-scene N] [--protocol both] [--batch B] [...]
#
# Pass --config/--ckpt relative to the model repo (cd'd below) or absolute.
# run_vp_petrv2.sh / run_vp_detr3d.sh are identical with ROOT swapped.
set -e

BEVF=/home/hanyan_arch/viewpoint/BEVFormer
ROOT=$BEVF/PETR                          # model repo (has data/nuscenes -> carla_geobev + projects/)
ENV=legacy-mmdet140-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
fi

export PYTHONPATH="$ROOT:$PYTHONPATH"    # so config plugin `projects.mmdet3d_plugin` imports
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

cd "$ROOT"
python "$BEVF/bev_det_benchmark/sparse/eval_vp_robustness_det_sparse.py" "$@"
