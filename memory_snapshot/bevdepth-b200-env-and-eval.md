---
name: bevdepth-b200-env-and-eval
description: "How to run BEVDepth CARLA eval — bevdepth-b200 env build, run command, CARLA eval patch, baseline NDS, eval-identity vs BEVFormer"
metadata: 
  node_type: memory
  type: project
  originSessionId: d6e20ce2-7fd8-462d-9469-4a1b57b88da9
---

BEVDepth (repo at `BEVFormer/BEVDepth/`) had CARLA support + trained ckpts (`pth/BEVDeth/bevdepth_{sedan,suv,bus}.ckpt`, note the dir typo "BEVDeth") but **no runnable env on this machine** — ckpts were trained elsewhere. Built `bevdepth-b200` env (2026-06-03).

**Env build** (clone of [[carla-bevformer-conda-env]]'s bevformer-b200, reusing its hard-won B200 mmcv 1.7.1 / mmdet3d 0.17.1 build):
- `conda create --clone bevformer-b200 --name bevdepth-b200`
- Add PL stack (proven combo from the `cvt` env): downgrade env pip `<24.1` (PL 1.6.2 has invalid metadata `torch (>=1.8.*)`), then `pip install --no-deps pytorch_lightning==1.6.2 torchmetrics==0.7.2 tensorboardX pyDeprecate==0.3.2`. PL 1.6.2 works fine on torch 2.11.0+cu128 (BEVDepth's base_cli uses PL-1.x API `add_argparse_args`/`from_argparse_args`/`accelerator='ddp'` — needs the 1.x line).
- Compile voxel_pooling CUDA ops: `cd BEVDepth; CUDA_HOME=<env prefix> PATH=<env>/bin:$PATH TORCH_CUDA_ARCH_LIST=10.0 python setup.py build_ext --inplace`. **Must set CUDA_HOME to the env** (it has conda CUDA 12.8 matching torch's cu128) — else cpp_extension picks system `/usr/local/cuda`=13.1 and errors on version mismatch.

**Run eval** (the package is NOT pip-installed, so PYTHONPATH must include the repo root or `import bevdepth` fails when running the exp by file path):
```
cd BEVFormer/BEVDepth
CUDA_HOME=<env> PATH=<env>/bin:$PATH PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
  <env>/bin/python bevdepth/exps/nuscenes/carla/carla_sedan.py \
  --ckpt_path ../pth/BEVDeth/bevdepth_sedan.ckpt -e --gpus 1 -b 8
```
~4 min inference (474 batches) + ~30s eval on one B200.

**CARLA eval patch** (I added to `bevdepth/evaluators/det_evaluators.py`): stock devkit only knows v1.0-{mini,trainval,test}; CARLA versions failed with `KeyError eval_set_map['v1.0-carla_sedan_eval']`. Patched `_evaluate_single` to mirror BEVFormer's `CarlaNuScenesDataset`: eval against 'val' split, replace `create_splits_scenes()['val']` with the predicted scenes only (so devkit's pred==gt assertion holds), flip `nusc.version`→'v1.0-trainval' around construction to bypass load_gt's trainval assertion. Also added a `[CARLA-EVAL] 6-class mAP=.. NDS=..` + `[CARLA-METRICS-JSON]` print in `evaluate()` (matches the bev_det_benchmark scrape regex; metric prefix is `pts_bbox_NuScenes/`).

**Eval GT filter changed to visibility>=2 (2026-06-03, matches training):** Originally both evals filtered GT by num_pts>0 (the devkit "valid" rule) = 142,874 sedan boxes. But TRAINING uses visibility>=2 (BEVDepth `gt_visibility_min=2`; BEVFormer pkl `valid_flag`==visibility>=2 + `use_valid_flag=True`) = 97,090 boxes — so eval was inconsistent with training (~35% of GT was low-vis v0-40%, which the model never trained on). Fixed BOTH evals to filter GT by `visibility_token >= 2` (constant `CARLA_MIN_VISIBILITY=2`), forcing `num_pts=1` so the devkit num_pts==0 removal is a no-op (visibility is now the sole validity rule); class_range still applies. BEVDepth: custom `_carla_load_gt` swapped into `nuscenes.eval.detection.evaluate.load_gt` in `det_evaluators.py`. BEVFormer: visibility post-filter added to `patched_load_gt` in `carla_nuscenes_dataset.py`. **Verified the two paths produce byte-identical GT: 43,190 boxes after class_range (car 16430, ped 20043, truck 4007, mc 1838, bicycle 644, bus 228), per-class identical** — directly comparable.

**Sedan baseline (visibility>=2 eval, 2026-06-03):** 6-class **mAP=0.2955, NDS=0.3278** (was 0.2185/0.2881 under the old num_pts>=0 filter). Big mAVE (~3.7) still drags NDS. The 97,090 visibility>=2 count == README "sedan 97k valid boxes" == BEVFormer `valid_flag=True` count (all three match exactly). See [[carla-geobev-dataset-layout]].
