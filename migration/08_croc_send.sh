#!/usr/bin/env bash
# [SOURCE server] Send the 31 GB core bundle to a NAT'd school machine via croc
# (relay-based P2P — neither side needs inbound; works through double-NAT; resumable).
#   install croc once:  curl https://getcroc.schollz.com | bash   (no sudo: PREFIX=$HOME/.local bash)
set -uo pipefail
B=${BUNDLE:-/NHNHOME/robogeo_migrate}
command -v croc >/dev/null || { echo "croc not found. install: curl https://getcroc.schollz.com | bash"; exit 1; }
cd "$B"
echo "== sending core bundle from $B (note the CODE croc prints, type it on the school box) =="
# one transfer preserves the dir layout; --no-local skips LAN discovery (different LANs)
croc send --no-local code_BEVFormer.tar.zst envs ckpts SHA256SUMS migration 2>/dev/null \
  || croc send code_BEVFormer.tar.zst envs ckpts SHA256SUMS
