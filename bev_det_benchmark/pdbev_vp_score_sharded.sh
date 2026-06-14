#!/usr/bin/env bash
# Parallel CPU scoring of a PD-BEV VP run (all dets already on disk). N shards score
# disjoint cell slices concurrently (each loads NuScenes GT once), then merge ->
# eval_vp.json with the 1/7 mVRS Table-2 row. Usage: TAG=.. N=.. bash this.sh
BENCH=/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark
CB="$(conda info --base)"
TAG=${TAG:-pdbev_sedan384}
N=${N:-6}
LOGD=/tmp/bevdet_smoke
echo "[vp-score-par] tag=$TAG shards=$N START $(date)"
pids=()
for i in $(seq 0 $((N-1))); do
  ( source "$CB/etc/profile.d/conda.sh"; conda activate bevdet-b200
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    cd /home/hanyan_arch/viewpoint/BEVFormer
    python "$BENCH/pdbev_vp_score.py" --tag "$TAG" --vehicle sedan \
      --conditions ER VR CR --protocol both --shard "$i/$N"
  ) > "$LOGD/pdbev_vpscore_${TAG}_${i}of${N}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[vp-score-par] all shards done, merging $(date)"
( source "$CB/etc/profile.d/conda.sh"; conda activate bevdet-b200
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  cd /home/hanyan_arch/viewpoint/BEVFormer
  python "$BENCH/pdbev_vp_score.py" --tag "$TAG" --vehicle sedan --merge
) > "$LOGD/pdbev_vpscore_${TAG}_merge.log" 2>&1
echo "[vp-score-par] ==== DONE $(date) ===="
cat "$BENCH/out/vp_$TAG/eval_vp_summary.txt" 2>/dev/null
