#!/usr/bin/env bash
# Run one BEVFormer eval pass on a condition-specific val pkl and print the
# deterministic 6-class NDS line the drivers scrape.
#
#   run_bevformer.sh <CONFIG> <CKPT> <NGPU> <COND_PKL> [extra cfg-options...]
#
# NOTE: tools/dist_test.sh already appends `--eval bbox`, so we pass ONLY
# --cfg-options here (passing --eval again double-feeds argparse).
#
# No `set -u`: conda's cuda-nvcc activate.d hook references the unset
# NVCC_PREPEND_FLAGS and would abort activation under nounset.
set -e

CONFIG=$1
CKPT=$2
NGPU=$3
COND_PKL=$4
shift 4

BEVF_ROOT=/home/hanyan_arch/viewpoint/BEVFormer
ENV=bevformer-b200

# activate conda env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
fi

cd "$BEVF_ROOT"

# unique port per call to avoid clashes when runs are chained
PORT=${PORT:-$((29510 + RANDOM % 200))}

# workers_per_gpu=4: images are on Lustre (network FS); many workers thrash it.
PORT=$PORT bash tools/dist_test.sh "$CONFIG" "$CKPT" "$NGPU" \
    --cfg-options data.test.ann_file="$COND_PKL" \
                  data.test.data_root="$BEVF_ROOT/data/nuscenes/" \
                  data.workers_per_gpu=4 "$@"
