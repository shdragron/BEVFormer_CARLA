#!/usr/bin/env bash
# Full-val PD-BEV VP INFER (sedan384, fps=79 = all 3792 frames, 631 cells) with
# auto-restart: the earlier 768 run had both shards SIGTERM'd at ~cell 275, so each
# round re-launches both shards (resume skips done dets) until all 631 cells exist.
# Separate tag 'pdbev_sedan384_full' so the 768-matched results are not clobbered.
GEN=/home/hanyan_arch/viewpoint/BEVFormer/Generalizable-BEV
BENCH=/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark
CB="$(conda info --base)"
TAG=pdbev_sedan384_full
CFG=configs/bevdet_our/bevdepth-r50-cbgs-pc-carla-sedan384.py
CK=work_dirs/pdbev-r50-cbgs-CARLA-dg-sedan384/epoch_24.pth
DETS=$BENCH/out/vp_$TAG/dets
LOGD=/tmp/bevdet_smoke

infer_shard () {  # $1 gpu  $2 shard
  ( source "$CB/etc/profile.d/conda.sh"; conda activate pdbev-b200
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NVIDIA_TF32_OVERRIDE=0 TMPDIR=/tmp
    export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1
    cd "$GEN"
    python "$BENCH/pdbev_vp_infer.py" --config "$CFG" --ckpt "$CK" \
      --frames-per-scene 79 --conditions ER VR CR --protocol both \
      --batch 32 --workers 6 --tag "$TAG" --shard "$2"
  ) > "$LOGD/pdbev_vpfull_infer_${2//\//of}.log" 2>&1
}

echo "[vpfull] ==== INFER START $(date) ===="
for round in $(seq 1 12); do
  n=$(ls "$DETS" 2>/dev/null | wc -l)
  if [ "$n" -ge 631 ]; then echo "[vpfull] all 631 dets present -> infer done"; break; fi
  echo "[vpfull] round $round: dets=$n/631, launching both shards $(date)"
  infer_shard 0 0/2 & P0=$!
  infer_shard 1 1/2 & P1=$!
  wait $P0; wait $P1
  echo "[vpfull] round $round ended (shard0=$? ...) dets now $(ls "$DETS" 2>/dev/null | wc -l)/631 $(date)"
done
n=$(ls "$DETS" 2>/dev/null | wc -l)
echo "[vpfull] ==== INFER DONE dets=$n/631 $(date) ===="
