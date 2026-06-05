#!/usr/bin/env bash
# Run one BEVDet eval pass on a condition-specific val pkl and print the
# deterministic 6-class NDS line the CTS driver scrapes
# (``[CARLA-EVAL] 6-class mAP=.. NDS=..`` + ``[CARLA-METRICS-JSON] {...}``,
# emitted by CarlaNuScenesDataset._evaluate_single).
#
#   run_bevdet.sh <CONFIG> <CKPT> <NGPU> <COND_PKL> [extra cfg-options...]
#
# COND_PKL's metadata['version']='v1.0-carla_<veh>_eval' drives which per-vehicle
# DB the GT loads from (and the visibility>=2 GT filter). For CTS the numerator
# uses the SEDAN ckpt on a TARGET condition pkl (target GT/version); the oracle
# uses the TARGET ckpt on the target pkl.
#
# No `set -u`: conda's cuda-nvcc activate.d hook references unset
# NVCC_PREPEND_FLAGS and would abort activation under nounset.
set -e

CONFIG=$1
CKPT=$2
NGPU=$3
COND_PKL=$4
shift 4

BEVDET=/home/hanyan_arch/viewpoint/BEVFormer/BEVDet
ENV=bevdet-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
fi

cd "$BEVDET"
# torch>=2.6 defaults weights_only=True; checkpoints carry non-tensor meta.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# unique port per call to avoid clashes when conditions are chained
PORT=${PORT:-$((29510 + RANDOM % 200))}

# --seed 0 --deterministic -> run-to-run identical NDS. data_root provides the
# images (relative cam data_paths) + the NuScenes eval DB; ann_file is the swapped
# condition pkl. workers_per_gpu=4 (images on Lustre; more workers thrash the FS).
# Eval batch (BS, default 1): bigger is much faster and NDS-INVARIANT (BatchNorm
# uses stored running stats in eval, so per-sample preds don't depend on batch).
# test.py forces depth_net's DeformConv im2col_step=6 so any BS*6 is divisible.
PORT=$PORT bash tools/dist_test.sh "$CONFIG" "$CKPT" "$NGPU" \
    --eval bbox --seed 0 --deterministic \
    --cfg-options data.test.ann_file="$COND_PKL" \
                  data.test.data_root="$BEVDET/data/nuscenes/" \
                  data.test.samples_per_gpu="${BS:-1}" \
                  data.workers_per_gpu="${WORKERS:-8}" "$@"
