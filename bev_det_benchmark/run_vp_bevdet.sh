#!/usr/bin/env bash
# Launch the in-process BEVDet VP driver (one shard, or --merge). Direct python
# (no torch.distributed.launch -- it's a single-GPU in-process cell loop).
#
#   run_vp_bevdet.sh --config <cfg> --ckpt <ckpt> [--shard i/n] \
#                    [--frames-per-scene N] [--protocol both] [--batch B] [...]
#
# Runs from the BEVDet repo so relative cam data_paths (data/nuscenes/...) and the
# editable mmdet3d resolve. VP_STAGE_ROOT (tmpfs) staging is honoured by the driver.
#
# No `set -u`: conda's cuda-nvcc activate.d hook references unset NVCC_PREPEND_FLAGS.
set -e

BEVF=/home/hanyan_arch/viewpoint/BEVFormer
BEVDET=$BEVF/BEVDet
ENV=bevdet-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1      # torch>=2.6 weights_only default
export PYTHONPATH="$BEVDET:$PYTHONPATH"
# modest BLAS threads: decode + eval run in-process; full-data uses run_vp_full.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

cd "$BEVDET"
python "$BEVF/bev_det_benchmark/eval_vp_robustness_det_bevdet.py" "$@"
