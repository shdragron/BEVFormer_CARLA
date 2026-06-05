#!/usr/bin/env bash
# Drive the BEVDepth CTS eval. Numerator = sedan ckpt on target under
# NORMAL/EXT/IMG/CAL; denominator = target ckpt on its own val (ORACLE).
# Uses predictable ckpt symlinks outputs/cts_ckpt_{sedan,suv,bus}.ckpt.
# Usage: _run_cts_bevdepth.sh <TAG> [extra eval_cts_det.py args...]
set -e
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200
BEVF=/home/hanyan_arch/viewpoint/BEVFormer
cd "$BEVF"
TAG=$1; shift
"$ENV/bin/python" bev_det_benchmark/eval_cts_det.py \
    --framework bevdepth \
    --ckpt BEVDepth/outputs/cts_ckpt_sedan.ckpt \
    --target-ckpt-tmpl 'BEVDepth/outputs/cts_ckpt_{}.ckpt' \
    --exp-tmpl 'bevdepth/exps/nuscenes/carla/carla_{}.py' \
    --ngpu 2 --tag "$TAG" "$@"
