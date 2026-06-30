#!/usr/bin/env bash
# No-visibility-filter sedan-source CTS for all 6 detectors (full-frame 3792).
# Each eval runs ONCE per cell with CARLA_DUAL_VIS=1, so the scoring code emits
# BOTH the default visibility>=2 line ([CARLA-EVAL]) AND the all-boxes line
# ([CARLA-EVAL-VIS0]) from the SAME predictions (single inference, dual score).
# Conditions: NORMAL (sedan-input reference) + CAL (full deploy) + ORACLE (denom).
# The driver writes the vis>=2 CTS table as usual; vis0 is scraped from the logs
# by score_novis_from_logs.py afterwards.
#   lane_cts_novis.sh <GPU>          (single GPU, sequential, resumable per cell)
set +e
GPU=${1:-0}
ROOT=/home/hanyan_arch/viewpoint/BEVFormer
SCR=/tmp/claude-3292/-home-hanyan-arch-viewpoint-BEVFormer/a8f76c13-f85f-4d13-a229-6878bbff6a20/scratchpad/novis
mkdir -p "$SCR"
LANE="$SCR/lane_novis.log"
export CUDA_VISIBLE_DEVICES=$GPU
export CARLA_DUAL_VIS=1                 # <-- turns on the vis0 second scoring pass
cd "$ROOT"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
P=$((29700 + GPU*20))
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LANE"; }
log "===== no-vis sedan-source CTS (6 models, suv+bus, NORMAL+CAL) GPU$GPU ====="

run_main(){ # <port> <framework> <config> <ckpt> <tgtTmpl> <tag> [extra...]
  local port=$1 fw=$2 cfg=$3 ckpt=$4 tmpl=$5 tag=$6; shift 6
  PORT=$port python bev_det_benchmark/eval_cts_det.py \
    --framework "$fw" --config "$cfg" --ckpt "$ckpt" --source sedan \
    --target-ckpt-tmpl "$tmpl" --targets suv bus --conditions NORMAL CAL \
    --tag "$tag" --ngpu 1 "$@"
}

# 1) BEVFormer (backward) — bevformer-b200
conda activate bevformer-b200; log "BEVFormer start"
run_main $((P+1)) bevformer projects/configs/bevformer/bevformer_tiny_carla.py \
  work_dirs/bevformer_tiny_carla_sedan/latest.pth 'work_dirs/bevformer_tiny_carla_{}/latest.pth' \
  bevformer_novis >> "$SCR/bevformer_novis.log" 2>&1; log "BEVFormer rc=$?"

# 2) DFA3D (backward) — bevformer-b200, own repo copy
conda activate bevformer-b200; log "DFA3D start"
( cd "$ROOT/3D-deformable-attention/BEVFormer_DFA3D" && \
  CARLA_DUAL_VIS=1 PORT=$((P+2)) python bev_det_benchmark/eval_cts_det.py \
    --framework bevformer --config projects/configs/bevformer/bevformer_DFA3D_carla.py \
    --ckpt work_dirs/bevformer_DFA3D_carla_sedan/epoch_24.pth --source sedan \
    --target-ckpt-tmpl 'work_dirs/bevformer_DFA3D_carla_{}/epoch_24.pth' \
    --targets suv bus --conditions NORMAL CAL --tag dfa3d_novis --ngpu 1 ) \
  >> "$SCR/dfa3d_novis.log" 2>&1; log "DFA3D rc=$?"

# 3) BEVDepth (forward) — bevdepth-b200 (per-target exp sets eval DB)
conda activate bevdepth-b200; log "BEVDepth start"
run_main $((P+3)) bevdepth dummy BEVDepth/outputs/cts_ckpt_sedan.ckpt 'BEVDepth/outputs/cts_ckpt_{}.ckpt' \
  bevdepth_novis --exp-tmpl 'bevdepth/exps/nuscenes/carla/carla_{}.py' >> "$SCR/bevdepth_novis.log" 2>&1; log "BEVDepth rc=$?"

# 4) BEVDet (forward) — bevdet-b200
conda activate bevdet-b200; log "BEVDet start"
run_main $((P+4)) bevdet BEVDet/configs/bevdet/carla/bevdet-r50-carla.py \
  BEVDet/work_dirs/bevdet-r50-carla_sedan/latest.pth 'BEVDet/work_dirs/bevdet-r50-carla_{}/latest.pth' \
  bevdet_novis >> "$SCR/bevdet_novis.log" 2>&1; log "BEVDet rc=$?"

# 5) CAPE (projection-free) — legacy
conda activate legacy-mmdet140-b200; log "CAPE start"
run_main $((P+5)) cape CAPE/projects/configs/CAPE/cape_carla_sedan.py \
  CAPE/ckpts/CAPE_ckpt/sedan/latest.pth 'CAPE/ckpts/CAPE_ckpt/{}/latest.pth' \
  cape_novis >> "$SCR/cape_novis.log" 2>&1; log "CAPE rc=$?"

# 6) DETR3D (backward/sparse) — legacy
conda activate legacy-mmdet140-b200; log "DETR3D start"
run_main $((P+6)) detr3d detr3d/projects/configs/detr3d/detr3d_carla_sedan.py \
  detr3d/work_dirs/detr3d_carla_sedan/latest.pth 'detr3d/work_dirs/detr3d_carla_{}/latest.pth' \
  detr3d_novis >> "$SCR/detr3d_novis.log" 2>&1; log "DETR3D rc=$?"

log "===== ALL 6 DONE ====="
