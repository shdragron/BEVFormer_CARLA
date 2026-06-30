#!/usr/bin/env bash
# [SOURCE server] EVAL-ONLY data path: extract the exact baseline images the 3
# *_infos_val.pkl reference (~30 GB) instead of the 1.3 TB full train+val set,
# and rsync just those to the new server (mirror the path). Run carla_geobev
# sync (8.5 G, pkls+DBs) separately via 04, or it is included here.
set -uo pipefail
B=/home/hanyan_arch/viewpoint/BEVFormer
J=/NHNHOME/WORKSPACE/0526040099_A/jeongtae
DEST=${DEST:?"set DEST=user@newhost:/NHNHOME/WORKSPACE/0526040099_A/jeongtae"}
OUT=${OUT:-/tmp/robogeo_migrate}; mkdir -p "$OUT"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate bevformer-b200
echo "== extracting val image list from the 3 val pkls =="
python - "$OUT/val_images.txt" <<'PY'
import pickle, os, sys
out=sys.argv[1]; root=os.path.realpath('/NHNHOME/WORKSPACE/0526040099_A/jeongtae/simbev_compare')
seen=set()
for plat in ('sedan','suv','bus'):
    d=pickle.load(open(f'/home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes/{plat}_infos_val.pkl','rb'))['infos']
    for info in d:
        for cam in info['cams'].values():
            rp=os.path.realpath(cam['data_path'])          # resolves symlink into simbev_compare
            if rp.startswith(root) and os.path.exists(rp):
                seen.add(os.path.relpath(rp, root))
open(out,'w').write('\n'.join(sorted(seen))+'\n')
print(f'  {len(seen)} val images -> {out}')
PY
echo "== rsync carla_geobev (pkls+DBs, materialised val sweeps via -L) =="
rsync -aLP --stats "$J/carla_geobev/" "$DEST/carla_geobev/"
echo "== rsync ONLY the val baseline images from simbev_compare (--files-from) =="
rsync -aP --files-from="$OUT/val_images.txt" "$J/simbev_compare/" "$DEST/simbev_compare/"
echo "NEW server: ln -sf $J/carla_geobev $B/data/nuscenes"
echo "NOTE: this skips carla_VR (VP eval). Add VP later with 04_sync_data.sh (carla_VR, +1.1TB)."
