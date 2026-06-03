#!/usr/bin/env bash
# Chain BEVFormer-tiny CARLA training: wait for the already-running sedan run to
# finish (epoch 24 + GPUs free), then run suv, then bus -- sequentially on the
# same 2 GPUs. sedan keeps running in its own tmux session; this only adds
# suv -> bus afterwards.
set -u
REPO=/home/hanyan_arch/viewpoint/BEVFormer
cd "$REPO"

gpus_free() {   # true when total GPU memory in use < 5 GB (training idle)
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         | awk '{s+=$1} END{print s+0}')
  [ "$used" -lt 5000 ]
}

wait_done() {   # $1 = work_dir name; wait for final checkpoint + free GPUs
  echo "[chain] waiting for work_dirs/$1/epoch_24.pth ..."
  until [ -f "work_dirs/$1/epoch_24.pth" ]; do sleep 120; done
  echo "[chain] $1 hit epoch_24; waiting for GPUs to free ..."
  until gpus_free; do sleep 30; done
  sleep 10
  echo "[chain] GPUs free; proceeding."
}

run() {         # $1 = vehicle (suv|bus)
  local v=$1
  echo "[chain] ===== launching $v ($(date)) ====="
  PORT=28510 bash tools/dist_train.sh \
    "projects/configs/bevformer/bevformer_tiny_carla_${v}.py" 2 \
    --work-dir "work_dirs/bevformer_tiny_carla_${v}" 2>&1 \
    | tee "logs/train_tiny_${v}.log"
  echo "[chain] ===== $v finished ($(date)) ====="
}

# sedan is already training in tmux 'bev_sedan'
wait_done bevformer_tiny_carla_sedan
run suv
run bus
echo "[chain] ALL DONE: sedan -> suv -> bus ($(date))"
