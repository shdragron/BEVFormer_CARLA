# BEVFormer Vehicle-Occupancy BEV Segmentation — HANDOFF

> Read this first if you are a new assistant picking up this work (esp. after a
> machine move). It is self-contained: goal, current state, every file, every
> bug already fixed, all verified geometry facts, and how to resume.
> Companion: `bevformer_seg/NOTES.md` (locked low-level facts).

Last updated: 2026-06-15, mid-training (epoch ~8/24).

---

## 1. Goal

Fill the empty **"BEVFormer (Backward-projection, seg)"** row in Table 2 of the
RoboGeo benchmark paper. We build a BEVFormer-based **vehicle-occupancy BEV
segmentation** model (binary vehicle / not-vehicle) and report in-distribution
IoU@0.5, then (later) run it through the VP/CTS robustness harness.

**Seg-task protocol** (matches CVT / LSS / GaussianLSS so it's comparable):
- Input: 6 cameras, **224×480** each.
- Output: **200×200** BEV over **±50 m** (0.5 m/cell), single **vehicle** binary
  occupancy.
- Metric: **IoU@0.5**.
- GT source: **GaussianLSS** bit-packed PNG labels.
- Visibility: **`visibility>=2`, detection-style removal** (see §5).

---

## 2. Current status (numbers)

| run | IoU@0.5 | note |
|-----|---------|------|
| **buggy single-cam** (`..._BUGGY_singlecam`) | 0.0506 | only 1 of 6 cams reached the model — DO NOT USE |
| fixed, epoch 2 (subset 400) | 0.3811 | sanity after fixes |
| fixed, **epoch 4 full-val (3792)** | **0.4055** | eval hook working |
| fixed, **epoch 6 full-val** | **0.4123** | climbing |
| target | ~0.42–0.45 | by epoch 24 |

Training is **resumed from epoch_2** and running to epoch 24 (sedan). When it
finishes, take the best-epoch full-val IoU for the Table-2 sedan cell, then
repeat for **suv** and **bus** (§8).

wandb: project **`BEVFormer_Seg`**, entity `Robust_Ex`, run
`bevformer_seg_r50_carla_sedan`.

---

## 3. Environment & repo

- Repo: `/home/hanyan_arch/viewpoint/BEVFormer`
- Conda env: **`bevformer-b200`** (B200 GPUs). Always:
  ```bash
  source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
  conda activate bevformer-b200
  export PYTHONPATH=/home/hanyan_arch/viewpoint/BEVFormer
  ```
- All standalone scripts need `PYTHONPATH=<repo>` (plugin import).

---

## 4. Files

### Created (the seg model — all untracked in git, `??`)
| file | role |
|------|------|
| `projects/configs/bevformer/bevformer_seg_r50_carla.py` | config (R50, 224×480, BEV200, 24ep, bs8×2gpu, lr4e-4, wandb) |
| `projects/mmdet3d_plugin/bevformer/detectors/bevformer_seg.py` | `BEVFormerSeg` detector + `SegHead` (resnet18 decoder), BCE+Dice |
| `projects/mmdet3d_plugin/datasets/carla_seg_dataset.py` | `CarlaSegDataset` (reads GaussianLSS jsons directly) |
| `projects/mmdet3d_plugin/datasets/pipelines/carla_seg_loading.py` | `LoadCarlaSegImages`, `FormatCarlaSeg` |
| `bevformer_seg/NOTES.md` | locked facts (bit4, lidar2img, flip) |
| `bevformer_seg/verify/reproject_check.py` | GT→image reprojection sanity (geometry proof) |
| `bevformer_seg/verify/smoke_fix.py` | forward smoke test (img shape, loss) |
| `bevformer_seg/verify/viz_pred.py` | pred-vs-GT BEV overlay diagnostic |
| `bevformer_seg/out/` | diagnostic PNGs |

### Modified (registration only — tracked, `M`)
- `projects/mmdet3d_plugin/bevformer/detectors/__init__.py` → `+ BEVFormerSeg`
- `projects/mmdet3d_plugin/datasets/__init__.py` → `+ CarlaSegDataset`
- `projects/mmdet3d_plugin/datasets/pipelines/__init__.py` → `+ LoadCarlaSegImages, FormatCarlaSeg`

---

## 5. The 4 bugs already found & fixed (CRITICAL — do not regress)

1. **Camera-dim strip (THE big one — caused IoU 0.05).**
   `forward_train/forward_test` did `img = img[:, -1]` when `img.dim()==5`,
   treating the **6-camera axis** as a temporal queue and feeding the model **1
   camera**. Fix: only strip when `img.dim()==6` (genuine queue). Verified:
   `extract_feat` now sees `(bs, 6, 3, 224, 480)`.

2. **Visibility convention — use REMOVAL, not ignore.**
   The detection models (`carla_nuscenes_dataset.py`, `CARLA_MIN_VISIBILITY=2`)
   **remove** low-vis GT: training `valid_flag = visibility>=2`, eval keeps only
   `visibility>=2` GT, so a prediction on a dropped vehicle is a **false
   positive**. We match this by **baking `vis>=2` into the GT mask** (`_gt_mask`):
   low-vis vehicle → background; loss & IoU over the **full grid** (no ignore
   mask). Confirmed bit4 is NOT pre-filtered — over 1350 frames, vehicle-cell
   visibility = `{1:155k, 2:43k, 3:42k, 4:249k}`, i.e. **31.7% are vis=1** and
   get dropped.

3. **Reprojection height (visualization only).**
   Ego origin is at the **ground** (cameras at z=+1.6 m; `ego_z≈0`). Vehicle BEV
   footprints must be projected at **z=0** to sit on the tire–ground line. (Was a
   viz bug; no effect on training.)

4. **Eval-hook crash: bs misread as 6.**
   `extract_img_feat` does `img.squeeze_()` **in-place** when `bs==1`, mutating
   `img` to `(6,3,224,480)`. `_bev_embed` then read `bs = img.size(0)` → **6**,
   so `bev_mask`/`bev_pos` got bs=6 while features were bs=1 → temporal-attention
   "size 2 vs 6". Training (bs=8) took the `reshape` branch and never hit it; only
   eval (bs=1) did. Fix: **`bs = img_feats[0].size(0)`** (read from features, not
   the mutated `img`).

---

## 6. Verified geometry / GT facts (locked)

- **vehicle = bit4**: `veh = (bev_png >> 4) & 1`. (bit0/1 = road. An earlier
  mapping claim of "bit0" was WRONG; cross-checked vs detection GT.)
- **view matrix** (per-frame, identical everywhere) `[[0,-2,100],[-2,0,100],[0,0,1]]`
  = 0.5 m/cell, ego-centered. BEV pixel→ego: `x=(100-row)/2`, `y=(100-col)/2`.
- **lidar2img = K4 @ E**, where `E = extrinsics` = **ego2cam** (4×4), and
  `K4 = eye(4); K4[:3,:3] = rescale(K)`.
- **K rescale** (resize 1600×900→480×270, then top-crop 46 → 224×480):
  `fx,cx *= 480/1600; fy,cy *= 270/900; cy -= 46`.
- **GT→BEVFormer grid**: `(veh & vis>=2)[::-1, ::-1].T`. Derived to match
  BEVFormer's `get_reference_points` (x↔col, y↔row, both increasing).
- **visibility PNG**: background = **255** sentinel; vehicle cells carry nuScenes
  bins **1–4** (1=0–40% … 4=80–100%). `vis>=2` keeps bins 2–4 + all background.
- Reprojection proof: `bevformer_seg/out/reproject_check.png` — footprints land on
  the correct camera and the tire line of each vehicle, no mirroring.

---

## 7. How to run

**Train (fresh, sedan):**
```bash
cd /home/hanyan_arch/viewpoint/BEVFormer
export CUDA_VISIBLE_DEVICES=0,1 PORT=29533 PYTHONPATH=/home/hanyan_arch/viewpoint/BEVFormer
bash tools/dist_train.sh projects/configs/bevformer/bevformer_seg_r50_carla.py 2
```
**Resume:** append `--resume-from work_dirs/bevformer_seg_r50_carla_sedan/epoch_N.pth`.

**Quick subset eval of a checkpoint** (template — see `verify/viz_pred.py` for the
forward call; use `ds._gt_mask(s,fr)` as GT, IoU over full grid).

**Smoke test after any code change:** `python bevformer_seg/verify/smoke_fix.py`
(must print `extract_feat sees img: (.,6,3,224,480)`).

> If `work_dirs/.../epoch_*.pth` did NOT come across in the machine move, just
> train fresh — it's ~6 h on 2× B200. The `_BUGGY_singlecam` backup is garbage
> (pre-fix); ignore it.

---

## 8. Pending

- [ ] Finish sedan to epoch 24; record best full-val IoU for Table 2.
- [ ] **suv** and **bus**: copy config, set `labels_dir` to `.../gaussianlss/suv`
      (+`suv_eval`) and `.../bus` (+`bus_eval`), change `work_dir`/wandb name. Same
      everything else.
- [ ] **seg VP/CTS robustness harness** — USER is bringing this; needs in-dist IoU
      first (done once sedan finishes).
- [ ] Fill the Table-2 seg BEVFormer row (sedan/suv/bus oracle IoU, then VP/CTS).

---

## 9. External data paths (on /NHNHOME shared workspace)

- GaussianLSS labels: `/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev_labels/gaussianlss/{sedan,sedan_eval,suv,suv_eval,bus,bus_eval}`
  (per scene: `scene_XXXX.json` + `scene_XXXX/{bev_*,visibility_*,aux_*}.png/npz`)
- Images: `/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev` (json `images[i]` are relative to this)
- Split: `/NHNHOME/WORKSPACE/0526040099_A/jeongtae/models/CARLA_GaussianLSS/GaussianLSS/data/splits/carla/train.txt`
  (val/eval dirs use `split_file=None` → all `scene_*.json`)

> If the new machine does not mount `/NHNHOME`, update the 3 path constants at the
> top of the config and `carla_seg_dataset.py`.
