#!/usr/bin/env bash
# [SOURCE server] Tar the BEVFormer working tree (main repo + nested DFA3D/detr3d/
# BEVDet/BEVDepth/CAPE + bev_det_benchmark + results + ALL uncommitted patches),
# EXCLUDING heavy regenerable/checkpoint dirs (those go via 03 / 04).
set -uo pipefail
OUT=${OUT:-/tmp/robogeo_migrate}; mkdir -p "$OUT"
cd /home/hanyan_arch/viewpoint
tar --zstd -cf "$OUT/code_BEVFormer.tar.zst" \
  --exclude='BEVFormer/data' \
  --exclude='*/work_dirs/*' --exclude='*/outputs/*' --exclude='BEVFormer/CAPE/ckpts/*' \
  --exclude='BEVFormer/bev_det_benchmark/out' \
  --exclude='*/test/*' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='*/__pycache__/*' --exclude='*.pyc' --exclude='*/.git/objects/*' \
  BEVFormer
echo "wrote $OUT/code_BEVFormer.tar.zst"; ls -lah "$OUT/code_BEVFormer.tar.zst"
echo "NEW server:  cd /home/hanyan_arch/viewpoint && tar --zstd -xf code_BEVFormer.tar.zst"
echo "NOTE: keeps results/ + configs + bench driver; excludes ckpts (use 03) & data (use 04)."
