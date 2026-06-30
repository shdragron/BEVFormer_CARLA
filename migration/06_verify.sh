#!/usr/bin/env bash
# [NEW server] Smoke-verify the migration without retraining: per-env import
# check, then reproduce one CTS anchor. Compare to the shipped JSON.
set -uo pipefail
cd /home/hanyan_arch/viewpoint/BEVFormer
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "===== 1) per-env torch/cuda import ====="
for e in bevformer-b200 legacy-mmdet140-b200 bevdet-b200 bevdepth-b200; do
  conda activate "$e" 2>/dev/null \
    && python -c "import torch;print('  $e: torch',torch.__version__,'cuda',torch.cuda.is_available())" \
    || echo "  $e: IMPORT FAIL"
done
echo "===== 2) data + ckpt presence ====="
ls -L data/nuscenes/sedan_infos_val.pkl >/dev/null 2>&1 && echo "  data OK" || echo "  data MISSING (symlink?)"
ls work_dirs/bevformer_tiny_carla_sedan/latest.pth >/dev/null 2>&1 && echo "  bevformer ckpt OK" || echo "  ckpt MISSING"
ls -d /NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_VR >/dev/null 2>&1 && echo "  carla_VR OK" || echo "  carla_VR MISSING (VP eval will fail)"
echo "===== 3) reproduce BEVFormer sedan->{suv,bus} CAL (dual-vis) ====="
echo "  expect ~ SUV-CAL CTS vis2/vis0 = 71.9/72.5 ; BUS-CAL = 40.1/41.9"
conda activate bevformer-b200
CARLA_DUAL_VIS=1 python bev_det_benchmark/eval_cts_det.py --framework bevformer \
  --config projects/configs/bevformer/bevformer_tiny_carla.py \
  --ckpt work_dirs/bevformer_tiny_carla_sedan/latest.pth \
  --source sedan --conditions CAL --tag verify_migration --ngpu 1 2>&1 \
  | grep -E 'CARLA-EVAL|CTS|VIS0' | tail -20
echo "===== compare against results/ROBOGEO_CTS_NOVIS_FULL.json (bevformer/sedan rows) ====="
