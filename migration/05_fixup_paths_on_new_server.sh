#!/usr/bin/env bash
# [NEW server] ONLY if you did NOT mirror the absolute roots. Rewrites the old
# roots to new ones across the bench code. Mirroring avoids needing this at all.
set -uo pipefail
B=${B:-/home/hanyan_arch/viewpoint/BEVFormer}
OLD_HOME=/home/hanyan_arch/viewpoint/BEVFormer ; NEW_HOME=${NEW_HOME:-$B}
OLD_NHN=/NHNHOME/WORKSPACE/0526040099_A        ; NEW_NHN=${NEW_NHN:-$OLD_NHN}
OLD_CONDA=$OLD_NHN/giyong/miniconda3           ; NEW_CONDA=${NEW_CONDA:-$OLD_NHN/giyong/miniconda3}
skip='/(out|logs|test|\.git)/'                 # never rewrite log/dump dirs
echo "== NHN/conda refs (VR_ROOT, bevdepth ENV, data roots) =="
grep -rlF "$OLD_NHN" "$B" 2>/dev/null | grep -vE "$skip" \
  | xargs -r sed -i "s#$OLD_CONDA#$NEW_CONDA#g; s#$OLD_NHN#$NEW_NHN#g"
if [ "$NEW_HOME" != "$OLD_HOME" ]; then
  echo "== home refs ($OLD_HOME -> $NEW_HOME) =="
  grep -rlF "$OLD_HOME" "$B" 2>/dev/null | grep -vE "$skip" | xargs -r sed -i "s#$OLD_HOME#$NEW_HOME#g"
fi
echo "== DONE. Manually re-check: =="
grep -n "VR_ROOT" "$B/bev_det_benchmark/build_condition_pkls.py" || true
grep -n "^ENV="   "$B/bev_det_benchmark/run_bevdepth.sh" || true
echo "  and the data/nuscenes symlink target."
