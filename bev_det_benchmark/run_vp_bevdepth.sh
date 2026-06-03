#!/usr/bin/env bash
# Launch the in-process BEVDepth VP robustness driver (loads model + GT DB ONCE,
# loops the 631-cell grid). Unlike BEVFormer's VP runner, BEVDepth needs no
# torch.distributed.launch -- the driver is plain single-GPU in-process.
#
#   CUDA_VISIBLE_DEVICES=1 run_vp_bevdepth.sh \
#       --frames-per-scene 4 --protocol both --tag bevdepth_sedan [--shard i/n]
#
# cwd = BEVDepth so the exp's relative data_root='data/carla' (symlink ->
# carla_geobev) resolves; VR images carry absolute paths so they're cwd-agnostic.
# No `set -u`: conda's cuda-nvcc activate.d hook references unset NVCC_PREPEND_FLAGS.
set -e

BEVF=/home/hanyan_arch/viewpoint/BEVFormer
BEVDEPTH="$BEVF/BEVDepth"
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "bevdepth-b200" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bevdepth-b200
fi

export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$BEVDEPTH:$BEVF:$PYTHONPATH"
# Keep BLAS modest so 2-GPU shards (each: decode workers + the NuScenes eval)
# don't oversubscribe the cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

cd "$BEVDEPTH"
python "$BEVF/bev_det_benchmark/eval_vp_robustness_det_bevdepth.py" "$@"
