#!/usr/bin/env bash
# [SOURCE server] Package the B200 envs for transfer. Compiled CUDA ops
# (bev_pool_v2_ext, DFA3D, patched mmcv) cannot be rebuilt from a yml.
#
# Two packaging modes (auto):
#   - conda-pack present  -> relocatable <e>.tar.gz (+ run `conda-unpack` on dest)
#   - else                -> plain  <e>.tar.gz of the env dir. VALID ONLY if the
#                            new server mounts the SAME conda prefix
#                            (/NHNHOME/.../giyong/miniconda3/envs/<e>) — i.e. you
#                            mirror the path. Extract to the IDENTICAL path; no
#                            conda-unpack needed.
# Always also writes yml + explicit manifests.
set -uo pipefail
OUT=${OUT:-/tmp/robogeo_migrate/envs}; mkdir -p "$OUT"
ENVS=${ENVS:-"bevformer-b200 legacy-mmdet140-b200 bevdet-b200 bevdepth-b200 pdbev-b200 coin3d"}
source "$(conda info --base)/etc/profile.d/conda.sh"
ENVDIR="$(conda info --base)/envs"
HAVE_CP=0; python -c "import conda_pack" 2>/dev/null && HAVE_CP=1
echo "mode: $([ $HAVE_CP = 1 ] && echo conda-pack || echo 'plain-tar (mirror prefix required)')"
for e in $ENVS; do
  echo "=== $e ==="
  conda env export -n "$e"      > "$OUT/$e.yml"          2>/dev/null || true
  conda list -n "$e" --explicit > "$OUT/$e.explicit.txt" 2>/dev/null || true
  if [ $HAVE_CP = 1 ]; then
    conda pack -n "$e" -o "$OUT/$e.tar.gz" --ignore-missing-files --n-threads 8 || echo "  conda-pack FAILED $e"
  else
    tar -C "$ENVDIR" -czf "$OUT/$e.tar.gz" "$e" && echo "  tarred $(du -h "$OUT/$e.tar.gz" | cut -f1)" || echo "  tar FAILED $e"
  fi
done
echo "== done -> $OUT =="; ls -lah "$OUT"/*.tar.gz 2>/dev/null
cat <<EOF
NEW server restore:
  conda-pack tar:  mkdir -p \$CONDA/envs/<e> && tar -xzf <e>.tar.gz -C \$CONDA/envs/<e> && conda activate <e> && conda-unpack
  plain tar (MIRROR): tar -xzf <e>.tar.gz -C /NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/   # same prefix
EOF
