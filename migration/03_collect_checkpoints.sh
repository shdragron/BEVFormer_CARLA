#!/usr/bin/env bash
# [SOURCE server] Checkpoints.
#   default       : stage the 18 eval-critical ckpts (symlinks resolved) -> OUT
#   FULL=1 DEST=… : rsync ALL training work_dirs/outputs/ckpts (~141 GB) straight
#                   to the new server (mirror path) — needed to resume/continue training.
set -uo pipefail
SRC=/home/hanyan_arch/viewpoint/BEVFormer

if [ "${FULL:-0}" = 1 ]; then
  DEST=${DEST:?"FULL mode: set DEST=user@newhost:/home/hanyan_arch/viewpoint/BEVFormer"}
  for rel in work_dirs \
             3D-deformable-attention/BEVFormer_DFA3D/work_dirs \
             detr3d/work_dirs BEVDet/work_dirs CAPE/ckpts BEVDepth/outputs \
             ckpts detr3d/ckpts \
             3D-deformable-attention/BEVFormer_DFA3D/ckpts; do
    [ -e "$SRC/$rel" ] || { echo "  (skip absent $rel)"; continue; }
    echo "== rsync $rel (full, all epochs) =="
    rsync -aP --stats "$SRC/$rel/" "$DEST/$rel/"
  done
  echo "== FULL checkpoints synced (~141G incl. every training epoch + logs). =="
  exit 0
fi

# ---- eval-only staging (default) ----
OUT=${OUT:-/tmp/robogeo_migrate/ckpts}; mkdir -p "$OUT"
R="$OUT/restore_ckpts.sh"
printf '#!/usr/bin/env bash\nset -e\nB=/home/hanyan_arch/viewpoint/BEVFormer\nckptdir="$(cd "$(dirname "$0")" && pwd)"\n' > "$R"
copy(){ local rel="$1" real; real="$(readlink -f "$SRC/$rel")"
  [ -f "$real" ] || { echo "  MISSING: $rel"; return; }
  mkdir -p "$OUT/$(dirname "$rel")"; cp -v "$real" "$OUT/$rel"
  printf 'mkdir -p $B/%s; cp "$ckptdir/%s" "$B/%s"\n' "$(dirname "$rel")" "$rel" "$rel" >> "$R"; }
for P in sedan suv bus; do
  copy "work_dirs/bevformer_tiny_carla_$P/epoch_24.pth"
  copy "3D-deformable-attention/BEVFormer_DFA3D/work_dirs/bevformer_DFA3D_carla_$P/epoch_24.pth"
  copy "detr3d/work_dirs/detr3d_carla_$P/epoch_24.pth"
  copy "CAPE/ckpts/CAPE_ckpt/$P/latest.pth"
  copy "BEVDet/work_dirs/bevdet-r50-carla_$P/epoch_24.pth"
  copy "BEVDepth/outputs/cts_ckpt_$P.ckpt"
done
cat >> "$R" <<'EOF'
for P in sedan suv bus; do
  ln -sf epoch_24.pth $B/work_dirs/bevformer_tiny_carla_$P/latest.pth
  ln -sf epoch_24.pth $B/detr3d/work_dirs/detr3d_carla_$P/latest.pth
  ln -sf epoch_24.pth $B/BEVDet/work_dirs/bevdet-r50-carla_$P/latest.pth
done
echo "eval checkpoints restored (18 files + latest.pth links)."
EOF
chmod +x "$R"
echo "== staged eval ckpts -> $OUT (or use FULL=1 DEST=… for all 141G) =="; du -sh "$OUT"
