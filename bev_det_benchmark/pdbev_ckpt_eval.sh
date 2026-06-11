#!/usr/bin/env bash
# Non-disruptive checkpoint eval for PD-BEV CARLA: as epoch_N.pth appears, dump val
# detections (pdbev-b200) and score with the VERIFIED BEVDet CARLA 6-class NDS
# (bevdet-b200). Does NOT touch the running training. Evals every EVERY-th epoch.
# Results -> $OUTDIR/val_nds.tsv  (epoch \t NDS \t mAP).  Logs an "EPOCH-NDS" line per eval.
# NO `set -u`: conda's cuda-nvcc activate.d hook references unset NVCC_PREPEND_FLAGS.
GEN=/home/hanyan_arch/viewpoint/BEVFormer/Generalizable-BEV
BENCH=/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark
WD=$GEN/work_dirs/pdbev-r50-cbgs-CARLA-dg
CFG=configs/bevdet_our/bevdepth-r50-cbgs-pc-carla.py   # EVAL model (BEVDepth_DG, working simple_test; loads DG ckpt)
OUTDIR=/tmp/pdbev_eval
EVERY=${EVERY:-4}                 # eval epochs that are multiples of EVERY (+ always the last seen)
GPU=${GPU:-0}
mkdir -p "$OUTDIR"; RES="$OUTDIR/val_nds.tsv"; [ -f "$RES" ] || echo -e "epoch\tNDS\tmAP" > "$RES"
CB="$(conda info --base)"

score_one () {   # $1 = epoch number, $2 = ckpt path
  local ep=$1 ck=$2 dets="$OUTDIR/epoch_${1}_dets.pkl"
  grep -qE "^$ep\b" "$RES" && return 0           # already scored
  echo "[ckpt-eval] epoch $ep: dump $(date +%H:%M:%S)"
  ( source "$CB/etc/profile.d/conda.sh"; conda activate pdbev-b200
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=$GPU
    cd "$GEN"; python "$BENCH/pdbev_dump_val.py" "$CFG" "$ck" "$dets" --batch 4 --workers 6 ) \
    > "$OUTDIR/dump_epoch_${ep}.log" 2>&1 || { echo "[ckpt-eval] epoch $ep DUMP FAIL"; return 1; }
  echo "[ckpt-eval] epoch $ep: score $(date +%H:%M:%S)"
  local out
  out=$( ( source "$CB/etc/profile.d/conda.sh"; conda activate bevdet-b200
           export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
           python "$BENCH/pdbev_score_carla.py" "$dets" --vehicle sedan ) 2>"$OUTDIR/score_epoch_${ep}.err" )
  local line; line=$(echo "$out" | grep '^NDS_RESULT')
  if [ -n "$line" ]; then
     local nds=$(echo "$line"|awk '{print $2}') map=$(echo "$line"|awk '{print $3}')
     echo -e "$ep\t$nds\t$map" >> "$RES"
     echo "EPOCH-NDS epoch=$ep NDS=$nds mAP=$map"
     rm -f "$dets"                                # free space (dets ~big)
  else echo "[ckpt-eval] epoch $ep SCORE FAIL"; tail -3 "$OUTDIR/score_epoch_${ep}.err"; fi
}

echo "==== PDBEV ckpt-eval watcher START $(date) (every $EVERY ep, GPU $GPU) ===="
while true; do
  # stop when training finished AND all expected ckpts scored
  done_train=0; grep -qaE 'train EXIT=' /tmp/bevdet_smoke/pdbev_train.log 2>/dev/null && done_train=1
  for ck in $(ls "$WD"/epoch_*.pth 2>/dev/null); do
    ep=$(basename "$ck" .pth | sed 's/epoch_//')
    if [ $((ep % EVERY)) -eq 0 ]; then score_one "$ep" "$ck"; fi
  done
  if [ "$done_train" -eq 1 ]; then
    # ensure the final epoch is scored even if not a multiple of EVERY
    last=$(ls "$WD"/epoch_*.pth 2>/dev/null | sed 's/.*epoch_//;s/.pth//' | sort -n | tail -1)
    [ -n "$last" ] && score_one "$last" "$WD/epoch_${last}.pth"
    echo "==== PDBEV ckpt-eval watcher DONE $(date) ===="; break
  fi
  sleep 600
done
