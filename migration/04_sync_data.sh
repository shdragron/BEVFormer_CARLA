#!/usr/bin/env bash
# [SOURCE server] FULL data sync (training + VP + all images), structure-preserving
# so the layout is IDENTICAL on the new server. ~2.4 TB. EDIT DEST (mirror path!).
# Symlinks are kept as-is (NOT -L): they point at absolute /NHNHOME paths which
# also get copied here, so they resolve on the mirrored new server (no duplication).
set -uo pipefail
J=/NHNHOME/WORKSPACE/0526040099_A/jeongtae
DEST=${DEST:?"set DEST=user@newhost:/NHNHOME/WORKSPACE/0526040099_A/jeongtae  (MIRROR the path!)"}
echo "== carla_geobev (8.5G: train+val pkls + v1.0-carla_* DBs + sweeps symlinks + split) =="
rsync -aP --stats "$J/carla_geobev/"   "$DEST/carla_geobev/"
echo "== simbev_compare (1.3TB: ALL baseline images, train+val — needed for retraining) =="
rsync -aP --stats "$J/simbev_compare/" "$DEST/simbev_compare/"
echo "== carla_VR (1.1TB: 31 variant renders + viewpoint_metadata.json — VP/mVRS eval) =="
rsync -aP --stats "$J/carla_VR/"       "$DEST/carla_VR/"
echo "== simbev/ground-truth (BEV seg GT — needed for seg training/eval) =="
rsync -aP --stats "$J/simbev/ground-truth/" "$DEST/simbev/ground-truth/"
cat <<EOF
== FULL data synced (~2.4TB). On NEW server: ==
  ln -sf $J/carla_geobev /home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes
  # the carla_geobev/sweeps/* symlinks resolve because simbev_compare is mirrored.
  # sanity: realpath \$(ls data/nuscenes/sweeps/RGB-CAM_FRONT/*.jpg | head -1)
TIP: run inside tmux; resumable (rsync -P). Consider --bwlimit / parallel dirs.
EOF
