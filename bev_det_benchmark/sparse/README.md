# Sparse-detector CTS / VP robustness eval (CAPE · PETRv2 · DETR3D)

Self-contained bundle for running the CARLA geobev **CTS** (cross-platform transfer)
and **VP** (viewpoint robustness) NDS benchmarks on the *sparse* detectors. Same
metrics/protocol as BEVFormer/BEVDet/BEVDepth; the sparse detectors just need a
different runner.

## Why a separate runner

The sparse detectors run in `legacy-mmdet140-b200` (mmcv 1.4.0 / mmdet 2.14.0).
Their `MultiScaleFlipAug3D` test pipeline trips a **DataContainer scatter bug in
mmcv-1.4 `multi_gpu_test`** (verified on DETR3D), so dist_test/DDP eval crashes
with `DataContainer has no attribute size`. We therefore use the **single-GPU**
path everywhere:
- CTS: `tools/test.py` (MMDataParallel) per condition pass.
- VP: an in-process `MMDataParallel` cell loop (`eval_vp_robustness_det_sparse.py`).

The condition pkls are **identical** to BEVFormer's (same nuScenes pkl format +
cam fields), so `build_condition_pkls.py` is reused verbatim.

## Files

| file | role |
|---|---|
| `eval_cts_det.py` | CTS driver (`--framework cape/petrv2/detr3d`); copy of the bench one |
| `build_condition_pkls.py` | condition-pkl builder (CTS + VP swaps); copy of the bench one |
| `run_{cape,petrv2,detr3d}.sh` | CTS per-pass runner (single-GPU `tools/test.py` + PYTHONPATH) |
| `eval_vp_robustness_det_sparse.py` | VP in-process driver (shard + `--merge`) |
| `run_vp_{cape,petrv2,detr3d}.sh` | VP launcher (legacy env, model repo as CWD/PYTHONPATH) |

Shared deps (`eval_cts_det.py`, `build_condition_pkls.py`) are **copied** here so the
folder is portable (e.g. to the vp/ training host) — keep them in sync with the
bench copies if either changes.

## Run

CTS (sedan model → suv/bus oracles; needs the suv/bus target ckpts/DBs too):
```bash
python sparse/eval_cts_det.py --framework cape \
    --config /abs/CAPE/projects/configs/CAPE/cape_carla_sedan.py \
    --ckpt   /abs/CAPE/work_dirs/cape_carla_sedan/latest.pth \
    --ngpu 1 --tag cape_sedan
```

VP (1/5 subset, 2-GPU cell-shard + merge — same args as BEVFormer's run_vp):
```bash
CUDA_VISIBLE_DEVICES=0 bash sparse/run_vp_cape.sh \
    --config projects/configs/CAPE/cape_carla_sedan.py \
    --ckpt   work_dirs/cape_carla_sedan/latest.pth \
    --frames-per-scene 16 --protocol both --shard 0/2 --tag cape_sedan
CUDA_VISIBLE_DEVICES=1 bash sparse/run_vp_cape.sh ... --shard 1/2 --tag cape_sedan
python sparse/eval_vp_robustness_det_sparse.py --merge --tag cape_sedan --protocol both
```
(`--config`/`--ckpt` resolve relative to the model repo, which the wrapper `cd`s into.)

## Status

Statically verified (parse / import / `B.*` compatibility / dispatch). **End-to-end
inference is NOT yet validated** — pending a sparse ckpt + free GPU. The DETR3D
sedan ckpt (`viewpoint/BEVFormer/detr3d/work_dirs/detr3d_carla_sedan/latest.pth`)
is the natural first validation target (same sparse family → also vouches for CAPE).
