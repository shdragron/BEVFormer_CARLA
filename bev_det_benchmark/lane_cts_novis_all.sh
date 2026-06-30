#!/usr/bin/env bash
# Full-source no-vis CTS: source platforms (sedan/suv/bus) as base, the selected
# detectors, full-frame 3792, CARLA_DUAL_VIS=1 so each cell emits BOTH the vis>=2
# ([CARLA-EVAL]) and all-boxes vis>=0 ([CARLA-EVAL-VIS0]) score from one inference.
# Conditions NORMAL+CAL (+ORACLE denom). Targets default to the two platforms !=
# source. Tags <model>_<src>_novis. Resumable per cell.
#   lane_cts_novis_all.sh <GPU> [SOURCES...]          (default sources: sedan suv bus)
#   MODELS="bevformer dfa3d ..." env selects the detector subset (default: all 6)
set +e
GPU=${1:-0}; shift
SOURCES="${*:-sedan suv bus}"
MODELS="${MODELS:-bevformer dfa3d bevdepth bevdet cape detr3d}"
ROOT=/home/hanyan_arch/viewpoint/BEVFormer
SCR=/tmp/claude-3292/-home-hanyan-arch-viewpoint-BEVFormer/a8f76c13-f85f-4d13-a229-6878bbff6a20/scratchpad/novis
mkdir -p "$SCR"
LANE="$SCR/lane_gpu${GPU}.log"
export CUDA_VISIBLE_DEVICES=$GPU
export CARLA_DUAL_VIS=1
cd "$ROOT"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LANE"; }
want(){ [[ " $MODELS " == *" $1 "* ]]; }
log "===== FULL-SOURCE no-vis CTS  sources=[$SOURCES]  models=[$MODELS]  GPU$GPU ====="

declare -A SP=( [sedan]=0 [suv]=8 [bus]=16 )

run_main(){ # <port> <framework> <config> <ckpt> <tgtTmpl> <tag> [extra...]
  local port=$1 fw=$2 cfg=$3 ckpt=$4 tmpl=$5 tag=$6; shift 6
  PORT=$port python bev_det_benchmark/eval_cts_det.py \
    --framework "$fw" --config "$cfg" --ckpt "$ckpt" --source "$SRC" \
    --target-ckpt-tmpl "$tmpl" --conditions NORMAL CAL \
    --tag "$tag" --ngpu 1 "$@"
}

for SRC in $SOURCES; do
  P=$((29600 + GPU*40 + ${SP[$SRC]}))
  log "##### SOURCE=$SRC start (port base $P) #####"

  if want bevformer; then
    conda activate bevformer-b200; log "[$SRC] BEVFormer start"
    run_main $((P+1)) bevformer projects/configs/bevformer/bevformer_tiny_carla.py \
      work_dirs/bevformer_tiny_carla_${SRC}/latest.pth 'work_dirs/bevformer_tiny_carla_{}/latest.pth' \
      bevformer_${SRC}_novis >> "$SCR/bevformer_${SRC}_novis.log" 2>&1; log "[$SRC] BEVFormer rc=$?"
  fi

  if want dfa3d; then
    conda activate bevformer-b200; log "[$SRC] DFA3D start"
    ( cd "$ROOT/3D-deformable-attention/BEVFormer_DFA3D" && \
      CARLA_DUAL_VIS=1 PORT=$((P+2)) python bev_det_benchmark/eval_cts_det.py \
        --framework bevformer --config projects/configs/bevformer/bevformer_DFA3D_carla.py \
        --ckpt work_dirs/bevformer_DFA3D_carla_${SRC}/epoch_24.pth --source $SRC \
        --target-ckpt-tmpl 'work_dirs/bevformer_DFA3D_carla_{}/epoch_24.pth' \
        --conditions NORMAL CAL --tag dfa3d_${SRC}_novis --ngpu 1 ) \
      >> "$SCR/dfa3d_${SRC}_novis.log" 2>&1; log "[$SRC] DFA3D rc=$?"
  fi

  if want bevdepth; then
    conda activate bevdepth-b200; log "[$SRC] BEVDepth start"
    run_main $((P+3)) bevdepth dummy BEVDepth/outputs/cts_ckpt_${SRC}.ckpt 'BEVDepth/outputs/cts_ckpt_{}.ckpt' \
      bevdepth_${SRC}_novis --exp-tmpl 'bevdepth/exps/nuscenes/carla/carla_{}.py' >> "$SCR/bevdepth_${SRC}_novis.log" 2>&1; log "[$SRC] BEVDepth rc=$?"
  fi

  if want bevdet; then
    conda activate bevdet-b200; log "[$SRC] BEVDet start"
    run_main $((P+4)) bevdet BEVDet/configs/bevdet/carla/bevdet-r50-carla.py \
      BEVDet/work_dirs/bevdet-r50-carla_${SRC}/latest.pth 'BEVDet/work_dirs/bevdet-r50-carla_{}/latest.pth' \
      bevdet_${SRC}_novis >> "$SCR/bevdet_${SRC}_novis.log" 2>&1; log "[$SRC] BEVDet rc=$?"
  fi

  if want cape; then
    conda activate legacy-mmdet140-b200; log "[$SRC] CAPE start"
    run_main $((P+5)) cape CAPE/projects/configs/CAPE/cape_carla_sedan.py \
      CAPE/ckpts/CAPE_ckpt/${SRC}/latest.pth 'CAPE/ckpts/CAPE_ckpt/{}/latest.pth' \
      cape_${SRC}_novis >> "$SCR/cape_${SRC}_novis.log" 2>&1; log "[$SRC] CAPE rc=$?"
  fi

  if want detr3d; then
    conda activate legacy-mmdet140-b200; log "[$SRC] DETR3D start"
    run_main $((P+6)) detr3d detr3d/projects/configs/detr3d/detr3d_carla_sedan.py \
      detr3d/work_dirs/detr3d_carla_${SRC}/latest.pth 'detr3d/work_dirs/detr3d_carla_{}/latest.pth' \
      detr3d_${SRC}_novis >> "$SCR/detr3d_${SRC}_novis.log" 2>&1; log "[$SRC] DETR3D rc=$?"
  fi

  log "##### SOURCE=$SRC models=[$MODELS] DONE #####"
done
log "===== FULL-SOURCE no-vis CTS COMPLETE (GPU$GPU, models=[$MODELS]) ====="
