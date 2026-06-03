# bev_det_benchmark — 3D detection robustness (NDS)

The 3D-object-detection counterpart of `bev_seg_benchmark` (CVT BEV-segmentation
IoU eval). Same two robustness studies, **NDS instead of IoU**, for the camera-only
detectors (BEVFormer / BEVDepth / BEVDet) on the CARLA "geobev" data:

| seg (IoU)                     | here (NDS)                       |
|-------------------------------|----------------------------------|
| `eval_vp_robustness_cvt.py`   | `eval_vp_robustness_det.py` (VP) |
| `eval_cts_cvt.py`             | `eval_cts_det.py` (CTS)          |

The metric is the **6-class NDS** recomputed inside `CarlaNuScenesDataset._evaluate_single`
(car/truck/bus/motorcycle/bicycle/pedestrian) — identical to the training-time
eval, so the numbers are directly comparable to the wandb val curves.

## Why pkl-driven (no model/dataset surgery)

Detection eval is 100% driven by the val info pkl. `CarlaNuScenesDataset` rebuilds
`lidar2img` purely from each cam's `sensor2lidar_rotation/translation` + `cam_intrinsic`;
the image bytes only feed the backbone and never touch the GT (`gt_boxes` are
ego-frame, viewpoint/platform-independent). So every robustness condition is just a
**cam-field swap on a clone of the sedan val pkl** — `build_condition_pkls.py` does the
swaps, the models run unchanged.

Invariants kept in every condition pkl: `gt_boxes/gt_names/valid_flag`, `token`,
`metadata['version']='v1.0-carla_sedan_eval'` (load-bearing: drives the NuScenes GT
DB load + the carla scene-name patch), and `cam_intrinsic` (identical across
platforms and VR variants — viewpoint perturbation is extrinsic-only).

## CTS — cross-platform transfer (`eval_cts_det.py`)

The **sedan-trained model** under 4 conditions (the **numerators**), evaluated **on
the TARGET (suv/bus) eval set** (target GT + target `valid_flag`/visibility, which
differs from sedan by mount height: sedan 97k vs suv 117k vs bus 137k valid boxes),
swapping the cam image and/or extrinsic toward the target:

| condition | image  | extrinsic | note            |
|-----------|--------|-----------|-----------------|
| NORMAL    | sedan  | sedan     | sedan-inputs reference on target (a numerator, **not** the denominator) |
| EXT       | sedan  | target    | extrinsic swapped to target |
| IMG       | target | sedan     | image swapped — **primary** |
| CAL       | target | target    | both swapped (full deploy) |

`CTS_c = NDS(cond c) / P_TARGET` per target (paper **Eq. 6**). The denominator
`P_TARGET` is the **target-native oracle** — a model *trained on* that target platform
(`work_dirs/bevformer_tiny_carla_{suv,bus}`) and evaluated on its own eval set — **not**
the sedan model's NORMAL. NORMAL is just the sedan-inputs reference, scored like every
other condition. sedan↔target cam fields join by parsed `(scene, frame)` (regex
`scene-(\d+)-frame-(\d+)`), not token. 10 runs (suv/bus × {ORACLE, NORMAL, EXT, IMG,
CAL}). Output table: `P_SUV EXT IMG CAL | P_BUS EXT IMG CAL` (the `P_*` column is the
oracle `P_TARGET`).

```bash
conda activate bevformer-b200
python bev_det_benchmark/eval_cts_det.py \
    --config projects/configs/bevformer/bevformer_tiny_carla.py \
    --ckpt   work_dirs/bevformer_tiny_carla_sedan/latest.pth \
    --ngpu 1 --tag tiny_sedan
# smoke: --targets suv --conditions IMG
```
Outputs in `out/cts_<tag>/`: `eval_cts.csv`, `eval_cts.json`, `eval_cts_summary.txt`.
Each condition shells out to `run_bevformer.sh` (→ `tools/dist_test.sh`) and scrapes
the `[CARLA-EVAL] 6-class mAP=.. NDS=..` line.

## VP — viewpoint robustness (`eval_vp_robustness_det.py`)

`carla_VR` viewpoint perturbations (31 variants = baseline + yaw/pitch/roll ×
±{4,8,12,16,20}). Conditions mirror the seg eval:

| condition | image swap | extrinsic swap |              |
|-----------|------------|----------------|--------------|
| Normal    | –          | –              | oracle / RRS denominator |
| ER        | –          | ✓              | extrinsic-only |
| VR        | ✓          | –              | **primary** (image-only) |
| CR        | ✓          | ✓              | both |

- image swap → cam `data_path` points at the VR variant JPG, baseline extrinsic kept.
- extrinsic swap → sedan image kept, `sensor2lidar_*` overwritten from
  `viewpoint_metadata.json` (quaternion → 3×3; baseline matches the sedan pkl exactly).
- protocols: **per-cam** (perturb one camera, ×6) + **all-cam** (perturb all six).

Full grid = `1 + 3 conditions × 3 axes × 10 signed-mags × 7 protocols = 631`. Running
each as its own process would reload the weights (~100 s) 631×, so the driver loads
the model **and** the NuScenes GT DB **once** and loops in-process (test.py blocks the
non-distributed path, so it still launches through `torch.distributed.launch` with one
proc). A frozen `--frames-per-scene` subset (test-mode temporal is off, so per-frame
subsetting is unbiased) keeps wall-clock sane; GT is filtered to the subset tokens so
the devkit's `pred==gt` assertion holds.

```bash
conda activate bevformer-b200
CUDA_VISIBLE_DEVICES=1 bash bev_det_benchmark/run_vp_bevformer.sh \
    --config projects/configs/bevformer/bevformer_tiny_carla.py \
    --ckpt   work_dirs/bevformer_tiny_carla_sedan/latest.pth \
    --frames-per-scene 2 --protocol both --tag tiny_sedan
# allcam-only / quick: --protocol allcam --frames-per-scene 4
```
Scores (RRS = NDS_cell / NDS_Normal): `mRRS_c` (mean over per-cam), `RRSALL_c`
(all-cam), `mVRS_c = ½(mRRS_c + RRSALL_c)`; VR primary. Outputs in `out/vp_<tag>/`:
`eval_vp_per_config.csv`, `eval_vp.json`, `eval_vp_summary.txt`.

## Adding BEVDepth / BEVDet

The condition-input mechanism is framework-agnostic. Once those envs run:
1. regenerate the same condition pkls in their info schema (their `create_data` /
   a thin adapter over the cam-field-swapped infos), then
2. add `run_bevdepth.sh` / `run_bevdet.sh` (their stock test entry, scraping their own
   6-class `NDS=` line) and point the drivers' `--framework` at them.
GT/tokens/scenes are shared, so the swap logic in `build_condition_pkls.py` is reused
verbatim.

## Files

| file | role |
|------|------|
| `build_condition_pkls.py` | cam-field swaps (CTS `make_cts_pkl`, VP `make_vp_infos`) |
| `eval_cts_det.py`         | CTS driver (shell-out per condition) |
| `eval_vp_robustness_det.py` | VP driver (model loaded once, in-process loop) |
| `run_bevformer.sh`        | one CTS eval pass → `[CARLA-EVAL]` line |
| `run_vp_bevformer.sh`     | launches the in-process VP driver |
| `out/`                    | results (`cts_<tag>/`, `vp_<tag>/`) |

Status: BEVFormer-tiny only (the one runnable env; BEVDepth/BEVDet envs pending). No
BEVFormer-base checkpoint exists yet, so eval uses the tiny sedan model.
