# BEVFormer-CARLA — viewpoint-robustness & cross-platform-transfer 3D detection (a CARLA fork of BEVFormer)

## Attribution

This repository is a **fork / modification** of
[**BEVFormer**](https://github.com/fundamentalvision/BEVFormer)
(fundamentalvision / OpenDriveLab), which is licensed under the
**Apache License 2.0**. We are **not** the original authors of BEVFormer; all
credit for the model and the upstream training code belongs to them. The
upstream `LICENSE` file is kept **unmodified** in this repo.

Per Apache-2.0 §4(b), the following is a list of the files we **added or
changed** for the CARLA viewpoint / cross-platform study. Everything else is
upstream BEVFormer, unchanged.

**Added (new files):**

- `projects/mmdet3d_plugin/datasets/carla_nuscenes_dataset.py` — `CarlaNuScenesDataset`.
- `tools/create_carla_data.py` — builds the per-vehicle info pkls.
- `tools/check_carla_projection.py` — cam-projection sanity check.
- `tools/train_carla_tiny_chain.sh` — sequential sedan→suv→bus training driver.
- `projects/configs/bevformer/bevformer_tiny_carla.py` (sedan) and
  `bevformer_tiny_carla_{suv,bus}.py`.
- `projects/configs/bevformer/bevformer_base_carla.py` (sedan) and
  `bevformer_base_carla_{suv,bus}.py`.
- `bev_det_benchmark/` — the VP (viewpoint-robustness) + CTS (cross-platform
  transfer) NDS benchmark (see [Evaluation](#7-evaluation)).

**Modified (upstream files changed):**

- `projects/mmdet3d_plugin/datasets/__init__.py` — register `CarlaNuScenesDataset`.
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` — CARLA-eval support.
- `tools/train.py`, `tools/test.py` — minor CARLA-eval / logging support.

The original upstream files (`projects/configs/bevformer/bevformer_tiny.py`,
`bevformer_base.py`, `tools/data_converter/nuscenes_converter.py`, etc.) are
**not** modified by the CARLA work.

---

## 2. What's different from upstream

CARLA adaptations + fair-comparison fixes:

1. **`CarlaNuScenesDataset`** — wraps the upstream `CustomNuScenesDataset` and
   overrides `_evaluate_single()` so the nuScenes devkit metric pipeline accepts
   our custom DB version strings (`v1.0-carla_{sedan,suv,bus}` and `..._eval`)
   and CARLA scene names. CARLA has no can-bus, so temporal is disabled.
2. **Per-vehicle info builder** (`tools/create_carla_data.py`) — produces
   `{sedan,suv,bus}_infos_{train,val}.pkl` from the per-vehicle CARLA "geobev"
   DBs. `valid_flag = visibility_token >= '2'` (≈ ≥40% visible). The same
   `split/{train,val}.txt` scene split is shared by all three vehicles; each
   sample is assigned a unique `scene_token` to force `prev_bev_exists=False`
   (temporal off without code surgery). `can_bus` is zeros, `sweeps` empty.
3. **CARLA configs** — `bevformer_tiny_carla*.py` (and `bevformer_base_carla*.py`):
   6 CARLA classes (`car, truck, bus, motorcycle, bicycle, pedestrian`),
   `use_valid_flag=True`, `use_can_bus=False`, `queue_length=2`,
   `dataset_type='CarlaNuScenesDataset'`, `data_root='data/nuscenes/'`.
4. **6-class NDS, GT filtered to visibility≥2** — `_evaluate_single` recomputes
   mAP/NDS over **exactly the 6 CARLA classes** with GT filtered to
   `visibility ≥ 2`, **identical to the training valid_flag**, so val curves and
   benchmark numbers are directly comparable. The result prints as a
   `[CARLA-EVAL] 6-class mAP=.. NDS=..` line.
5. **VP + CTS robustness benchmark** (`bev_det_benchmark/`) — NDS counterpart of
   the CVT BEV-seg benchmark. VP = viewpoint robustness (yaw/pitch/roll
   perturbations of camera extrinsics, 631-cell grid); CTS = cross-platform
   transfer of the sedan-trained model onto suv/bus. Both are 100% driven by
   cam-field swaps on a clone of the val pkl — the model runs unchanged.
6. **Fast NDS-exact eval path** — RAM-staged images (tmpfs) + shared-memory JPEG
   decode + NUMA-pinned 2-GPU sharding. Bit-identical NDS to the standard path;
   full VP grid in ~16h.

> The mmdet3d plugin **auto-loads** because the CARLA configs set
> `plugin = True` and `plugin_dir = 'projects/mmdet3d_plugin/'`. No `import`
> hacks needed — running the config registers `CarlaNuScenesDataset`.

---

## 3. Environment setup

Conda env name: **`bevformer-b200`**. Install follows the upstream BEVFormer
instructions (`docs/install.md`): CUDA-matched PyTorch, `mmcv-full` 1.x,
`mmdet==2.14.0`, `mmsegmentation==0.14.1`, and `mmdet3d==0.17.1` from source.

```bash
conda create -n bevformer-b200 python=3.8 -y
conda activate bevformer-b200

# PyTorch matching your CUDA (upstream reference: torch 1.9.1 + cu111;
# use the build that matches your GPU/driver).
pip install torch torchvision torchaudio

# OpenMMLab stack (mmcv-full 1.7.x is fine for the 1.x API used here)
pip install mmcv-full==1.7.1
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# mmdet3d 0.17.1 from source
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout v0.17.1 && python setup.py install && cd ..

# extras used by the plugin
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13 \
    typing-extensions==4.5.0 numpy==1.19.5 numba==0.48.0 \
    nuscenes-devkit setuptools==59.5.0

# Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

(`wandb` is optional; the configs add a `WandbLoggerHook`. `pip install wandb`
and `wandb login`, or remove the hook from the config to skip it.)

---

## 4. Data

The CARLA "geobev" dataset is in **nuScenes format**, with one DB per
ego-vehicle viewpoint. Top-level layout (geobev root):

```
carla_geobev/
  v1.0-carla_sedan/        v1.0-carla_sedan_eval/      # sedan train / eval DBs
  v1.0-carla_suv/          v1.0-carla_suv_eval/        # suv   train / eval DBs
  v1.0-carla_bus/          v1.0-carla_bus_eval/        # bus   train / eval DBs
  split/train.txt          split/val.txt               # scene-name split (shared)
  sweeps/                  <hash>_CAM_*.jpg            # camera images
  viewpoint_metadata.json                              # VR extrinsics (for VP eval)
```

**Symlink the geobev root into the repo as `data/nuscenes`:**

```bash
ln -s /NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev \
      data/nuscenes
```

**Build the per-vehicle info pkls** (reads in place; copies nothing):

```bash
conda activate bevformer-b200
python tools/create_carla_data.py \
    --root-path data/nuscenes \
    --out-dir   data/nuscenes \
    --vehicles  sedan suv bus \
    --workers   16
```

Produces, under `data/nuscenes/`:
`{sedan,suv,bus}_infos_train.pkl` and `{sedan,suv,bus}_infos_val.pkl`
(`valid_flag = visibility ≥ 2`).

---

## 5. Pretrained weights

- **Backbone:** ResNet-50, `torchvision://resnet50`, downloaded automatically by
  the tiny CARLA configs on first run. No manual download needed.
- **Trained checkpoints** land in `work_dirs/bevformer_tiny_carla_<veh>/`
  (e.g. `latest.pth`, `epoch_24.pth`). These are produced by training below;
  they are **not** committed (see `.gitignore`).

> Note: the base CARLA configs (`bevformer_base_carla*.py`) follow upstream and
> use the R101-DCN FCOS3D pretrain (`ckpts/r101_dcn_fcos3d_pretrain.pth`). The
> benchmark and the commands below target the **tiny** configs, which are the
> runnable ones in this fork.

---

## 6. Training

Training uses the upstream `tools/dist_train.sh` launcher
(`dist_train.sh <CONFIG> <NUM_GPUS> [extra args]`). The CARLA tiny configs use
`samples_per_gpu=8`, `total_epochs=24`.

**Sedan (the base config the others inherit):**

```bash
conda activate bevformer-b200
PORT=28509 bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_tiny_carla.py 2 \
    --work-dir work_dirs/bevformer_tiny_carla_sedan
```

**SUV:**

```bash
PORT=28510 bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_tiny_carla_suv.py 2 \
    --work-dir work_dirs/bevformer_tiny_carla_suv
```

**Bus:**

```bash
PORT=28510 bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_tiny_carla_bus.py 2 \
    --work-dir work_dirs/bevformer_tiny_carla_bus
```

To run all three sequentially on the same 2 GPUs (sedan → suv → bus), use the
chain driver (waits for `epoch_24.pth` + free GPUs between runs):

```bash
bash tools/train_carla_tiny_chain.sh
```

(`bevformer_tiny_carla_{suv,bus}.py` inherit the sedan config and only repoint
`ann_file` to the suv/bus pkls and rename the wandb run. `bevformer_base_carla*.py`
exist analogously if you train the base variant.)

---

## 7. Evaluation

**Metric:** 6-class **NDS** (car/truck/bus/motorcycle/bicycle/pedestrian), with
GT filtered to **visibility ≥ 2** — identical to the training valid_flag.
Printed as `[CARLA-EVAL] 6-class mAP=.. NDS=..`.

### Plain single-config NDS (sanity / val score)

Upstream test launcher (`dist_test.sh <CONFIG> <CKPT> <NUM_GPUS> ...`,
which already appends `--eval bbox`):

```bash
conda activate bevformer-b200
PORT=29503 bash tools/dist_test.sh \
    projects/configs/bevformer/bevformer_tiny_carla.py \
    work_dirs/bevformer_tiny_carla_sedan/latest.pth 1
```

### VP — viewpoint robustness (`eval_vp_robustness_det.py`)

Camera-extrinsic perturbations (yaw/pitch/roll × ±{4,8,12,16,20}), per-cam and
all-cam protocols. Scores: `mRRS`, `RRSALL`, `mVRS = ½(mRRS+RRSALL)`
(RRS = NDS_cell / NDS_Normal); VR (image-only) is the primary condition.

Quick single-GPU run via the launcher (loads the model once, loops the grid):

```bash
CUDA_VISIBLE_DEVICES=1 bash bev_det_benchmark/run_vp_bevformer.sh \
    --config projects/configs/bevformer/bevformer_tiny_carla.py \
    --ckpt   work_dirs/bevformer_tiny_carla_sedan/latest.pth \
    --frames-per-scene 2 --protocol both --tag tiny_sedan
```

Full-data, NDS-exact, 2-GPU NUMA-pinned fast path (~16h, then merges shards):

```bash
bash bev_det_benchmark/run_vp_full.sh 8 tiny_sedan
#   args: run_vp_full.sh [WORKERS] [TAG]
```

Outputs in `bev_det_benchmark/out/vp_<tag>/`:
`eval_vp_per_config.csv`, `eval_vp.json`, `eval_vp_summary.txt`.

### CTS — cross-platform transfer (`eval_cts_det.py`)

The sedan-trained model evaluated on the suv/bus targets under 4 conditions
(NORMAL / EXT / IMG / CAL). `CTS_c = NDS(cond c) / P_TARGET`, where `P_TARGET`
is the target-native oracle (a model trained on that platform,
`work_dirs/bevformer_tiny_carla_{suv,bus}`).

```bash
conda activate bevformer-b200
python bev_det_benchmark/eval_cts_det.py \
    --config projects/configs/bevformer/bevformer_tiny_carla.py \
    --ckpt   work_dirs/bevformer_tiny_carla_sedan/latest.pth \
    --ngpu 1 --tag tiny_sedan
# smoke test: --targets suv --conditions IMG
```

Outputs in `bev_det_benchmark/out/cts_<tag>/`:
`eval_cts.csv`, `eval_cts.json`, `eval_cts_summary.txt`. See
`bev_det_benchmark/README.md` for the full condition tables and methodology.

---

## 8. License

This project is licensed under the **Apache License 2.0**, inherited from
upstream BEVFormer; see the unmodified `LICENSE` file. Changes made by this fork
are listed in [Attribution](#attribution) per Apache-2.0 §4(b). Original
BEVFormer © its respective authors (fundamentalvision / OpenDriveLab). The CARLA
"geobev" data and `viewpoint_metadata.json` are external assets governed by their
own terms and are **not** distributed with this repository.
