#!/usr/bin/env bash
# Run one BEVDepth eval pass on a condition-specific val pkl and print the
# deterministic 6-class NDS line the CTS driver scrapes
# (``[CARLA-EVAL] 6-class mAP=.. NDS=..`` + ``[CARLA-METRICS-JSON] {...}``,
# emitted by DetNuscEvaluator.evaluate()).
#
#   run_bevdepth.sh <EXP_PY> <CKPT> <NGPU> <COND_PKL> [extra exp args...]
#
# The exp (bevdepth/exps/nuscenes/carla/carla_<veh>.py) sets the evaluator
# version = v1.0-carla_<veh>_eval, i.e. which per-vehicle DB the GT loads from;
# CARLA_VAL_INFO injects COND_PKL as the val set (cam-field swaps) so the model
# runs unchanged. For CTS the numerator uses the SEDAN ckpt on the TARGET exp
# (target GT/version), the oracle uses the TARGET ckpt on the same target exp.
#
# No `set -u`: conda's cuda-nvcc activate.d hook references unset
# NVCC_PREPEND_FLAGS and would abort activation under nounset.
set -e

EXP=$1
CKPT=$2
NGPU=$3
COND_PKL=$4
shift 4

BEVDEPTH=/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "bevdepth-b200" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bevdepth-b200
fi

# CUDA_HOME pinned to the env (conda CUDA 12.8 matching torch cu128); PYTHONPATH
# so `import bevdepth` works when the exp is run by file path.
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$BEVDEPTH:$PYTHONPATH"
export CARLA_VAL_INFO="$COND_PKL"      # consumed by carla_base.py

cd "$BEVDEPTH"
# Eval batch (BS, default 64): bigger is faster on the B200 (it sits idle at small
# batches) and NDS-invariant -- BatchNorm runs on stored running stats in eval, so
# per-sample predictions don't depend on batch size. Override with BS=... if needed.
# Dataloader stays at 4 workers (images on Lustre; more workers thrash the FS).
python "$EXP" --ckpt_path "$CKPT" -e --gpus "$NGPU" -b "${BS:-64}" "$@"
