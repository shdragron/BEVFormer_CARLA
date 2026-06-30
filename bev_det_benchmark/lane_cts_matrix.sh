#!/usr/bin/env bash
# One GPU lane of the CTS transfer-matrix campaign (full-frame 3792, CAL only).
# 6 models, ordered for paradigm coverage by deadline (1st three = fwd/back/free).
# Sequential on ONE gpu, resumable (per-cell cache), fault-tolerant (a model
# failing does not stop the lane).
#   lane_cts_matrix.sh <GPU> <SOURCE> <TARGETS...>
#     e.g.  lane_cts_matrix.sh 0 suv sedan bus  /  lane_cts_matrix.sh 1 bus sedan suv
set +e
GPU=$1; SRC=$2; shift 2; TGT="$*"
ROOT=/home/hanyan_arch/viewpoint/BEVFormer
SCR=/tmp/claude-3292/-home-hanyan-arch-viewpoint-BEVFormer/a8f76c13-f85f-4d13-a229-6878bbff6a20/scratchpad/cts_matrix
LANE="$SCR/lane_${SRC}base.log"
export CUDA_VISIBLE_DEVICES=$GPU
cd "$ROOT"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
P=$((29600 + GPU*20))
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LANE"; }
log "===== lane GPU$GPU source=$SRC targets=[$TGT]  6 models, CAL, full3792 ====="

run_main(){ # <port> <framework> <config> <ckpt> <tgtTmpl> <tag> [extra...]
  local port=$1 fw=$2 cfg=$3 ckpt=$4 tmpl=$5 tag=$6; shift 6
  PORT=$port python bev_det_benchmark/eval_cts_det.py \
    --framework "$fw" --config "$cfg" --ckpt "$ckpt" --source "$SRC" \
    --target-ckpt-tmpl "$tmpl" --targets $TGT --conditions CAL --tag "$tag" --ngpu 1 "$@"
}

# 1) BEVFormer (backward) — bevformer-b200
conda activate bevformer-b200; log "BEVFormer ${SRC} start"
run_main $((P+1)) bevformer projects/configs/bevformer/bevformer_tiny_carla.py \
  work_dirs/bevformer_tiny_carla_${SRC}/latest.pth 'work_dirs/bevformer_tiny_carla_{}/latest.pth' \
  bevformer_${SRC}base_full >> "$SCR/bevformer_${SRC}base.log" 2>&1; log "BEVFormer ${SRC} rc=$?"

# 2) BEVDepth (forward) — bevdepth-b200 (per-target exp sets eval DB)
conda activate bevdepth-b200; log "BEVDepth ${SRC} start"
run_main $((P+2)) bevdepth dummy BEVDepth/outputs/cts_ckpt_${SRC}.ckpt 'BEVDepth/outputs/cts_ckpt_{}.ckpt' \
  bevdepth_${SRC}base_full --exp-tmpl 'bevdepth/exps/nuscenes/carla/carla_{}.py' >> "$SCR/bevdepth_${SRC}base.log" 2>&1; log "BEVDepth ${SRC} rc=$?"

# 3) CAPE (projection-free) — legacy
conda activate legacy-mmdet140-b200; log "CAPE ${SRC} start"
run_main $((P+3)) cape CAPE/projects/configs/CAPE/cape_carla_sedan.py \
  CAPE/ckpts/CAPE_ckpt/${SRC}/latest.pth 'CAPE/ckpts/CAPE_ckpt/{}/latest.pth' \
  cape_${SRC}base_full >> "$SCR/cape_${SRC}base.log" 2>&1; log "CAPE ${SRC} rc=$?"

# 4) DFA3D (backward) — bevformer-b200, own repo copy
conda activate bevformer-b200; log "DFA3D ${SRC} start"
( cd "$ROOT/3D-deformable-attention/BEVFormer_DFA3D" && \
  PORT=$((P+4)) python bev_det_benchmark/eval_cts_det.py \
    --framework bevformer --config projects/configs/bevformer/bevformer_DFA3D_carla.py \
    --ckpt work_dirs/bevformer_DFA3D_carla_${SRC}/epoch_24.pth --source $SRC \
    --target-ckpt-tmpl 'work_dirs/bevformer_DFA3D_carla_{}/epoch_24.pth' \
    --targets $TGT --conditions CAL --tag dfa3d_${SRC}base_full --ngpu 1 ) \
  >> "$SCR/dfa3d_${SRC}base.log" 2>&1; log "DFA3D ${SRC} rc=$?"

# 5) BEVDet (forward) — bevdet-b200
conda activate bevdet-b200; log "BEVDet ${SRC} start"
run_main $((P+5)) bevdet BEVDet/configs/bevdet/carla/bevdet-r50-carla.py \
  BEVDet/work_dirs/bevdet-r50-carla_${SRC}/latest.pth 'BEVDet/work_dirs/bevdet-r50-carla_{}/latest.pth' \
  bevdet_${SRC}base_full >> "$SCR/bevdet_${SRC}base.log" 2>&1; log "BEVDet ${SRC} rc=$?"

# 6) DETR3D (backward/sparse) — legacy
conda activate legacy-mmdet140-b200; log "DETR3D ${SRC} start"
run_main $((P+6)) detr3d detr3d/projects/configs/detr3d/detr3d_carla_sedan.py \
  detr3d/work_dirs/detr3d_carla_${SRC}/latest.pth 'detr3d/work_dirs/detr3d_carla_{}/latest.pth' \
  detr3d_${SRC}base_full >> "$SCR/detr3d_${SRC}base.log" 2>&1; log "DETR3D ${SRC} rc=$?"

log "===== lane GPU$GPU source=$SRC ALL 6 DONE ====="
