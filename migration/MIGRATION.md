# RoboGeo — Full Server Migration Guide

Move the entire RoboGeo benchmark (code + 6 model repos + envs + data + eval
checkpoints + results) to a **new server with NO shared filesystem**.

Generated from a live inventory of the source server. Run the helper scripts in
this folder in order; this file explains *what* and *why*.

> ### ✅ CHOSEN SCOPE: **EVERYTHING (retraining-capable)**
> Full data (**~2.4 TB**: all train+val images + carla_VR) + **all** training
> `work_dirs` (**~141 GB**, every epoch) + all results + envs + code. This
> reproduces *and* lets you continue training. Total ≈ **2.6 TB + 57 GB envs**.
> Use **`04_sync_data.sh`** (full data) and **`FULL=1 03_collect_checkpoints.sh`**
> (all work_dirs). Budget days for the transfer; run inside `tmux`, rsync is
> resumable (`-P`).
>
> Retraining also needs (already covered, just don't skip): the **train pkls**
> (`*_infos_train.pkl`, in carla_geobev), **seg GT** (`simbev/ground-truth`),
> and the **ImageNet/DPT pretrained backbones** the configs `load_from` — verify
> those weights resolve on the new server (see §4).

---

## 0. The one idea that makes this easy: mirror the two absolute-path roots

The code has **hard-coded absolute paths** in two namespaces:

| Root | Holds | # refs |
|---|---|---|
| `/home/hanyan_arch/viewpoint/BEVFormer` | all code, bench scripts, configs | ~323 files |
| `/NHNHOME/WORKSPACE/0526040099_A/...` | conda envs, data, carla_VR | a handful in code + the conda env prefixes |

**If you recreate BOTH roots at the same absolute paths on the new server, you
edit almost nothing.** Username need not be `hanyan_arch`, but the *path*
`/home/hanyan_arch/viewpoint/BEVFormer` must exist (use a symlink or a bind
mount if your home differs). Same for `/NHNHOME/WORKSPACE/0526040099_A`.

If you cannot mirror the paths, run `05_fixup_paths_on_new_server.sh` (sed
rewrites) — but mirroring is strongly recommended; it is the difference between
"works immediately" and "chase 300 path edits".

This guide assumes mirroring. Deviations are flagged **[non-mirror]**.

---

## 1. Inventory (what is being moved)

### Code — `/home/hanyan_arch/viewpoint/BEVFormer`  (git: master @ 8d18148)
The single tree contains the main repo **plus all model repos nested inside it**:

| Path (under BEVFormer/) | Model | git |
|---|---|---|
| `.` (projects/, bev_det_benchmark/, tools/) | BEVFormer + benchmark driver | master, **38 dirty + 924 untracked** |
| `3D-deformable-attention/BEVFormer_DFA3D` | DFA3D | vendored, **no .git** |
| `detr3d` | DETR3D | main, 3 dirty + 21 untracked |
| `BEVDet` | BEVDet | dev3.0, 1 dirty |
| `BEVDepth` | BEVDepth | main, 11 dirty + 15 untracked |
| `CAPE` | CAPE (the one eval uses) | clean-ish |

> ⚠️ The **uncommitted** changes ARE the research (dual-vis patch, CARLA dataset
> classes, configs, the whole `bev_det_benchmark/` driver). Do **not** rely on a
> clean `git clone` — you must move the working tree. `02_pack_code.sh` tars it.
>
> Note: `/home/hanyan_arch/viewpoint/CAPE` is a *separate* clean clone; the eval
> pipeline uses the nested `BEVFormer/CAPE`. Move the nested one.

### Conda envs — `/NHNHOME/.../giyong/miniconda3/envs/`  (~9.4 GB each)
B200 builds with **compiled CUDA ops** (bev_pool_v2_ext, DFA3D ops, patched
mmcv). These **cannot be rebuilt reliably from a yml** (many manual fixes). Move
them as **conda-pack tarballs** (`01_export_envs.sh`).

| env | used by |
|---|---|
| `bevformer-b200` | BEVFormer, DFA3D |
| `legacy-mmdet140-b200` | CAPE, DETR3D |
| `bevdet-b200` | BEVDet |
| `bevdepth-b200` | BEVDepth |
| `pdbev-b200` | PD-BEV (optional, seg/DG) |
| `coin3d` | CoIn3D (optional) |

### Data — Lustre `/NHNHOME/WORKSPACE/0526040099_A/jeongtae/`  ⚠️ **~2.4 TB if copied whole**
| Path | Holds | size |
|---|---|---|
| `carla_geobev` | info pkls + nuScenes-schema DBs (`v1.0-carla_*`) + `sweeps/` (symlinks) + `split/` | 8.5 G |
| `simbev_compare/sweeps2` | **real baseline images** (geobev/sweeps symlinks point here; **train+val, all platforms**) | **1.3 TB** |
| `carla_VR` | **variant (perturbed) renders, 31 variants** + `viewpoint_metadata.json` | **1.1 TB** |
| `simbev/ground-truth` | BEV GT masks (seg only; via `ground-truth` symlink) | (seg) |

#### 🔑 DATA SCOPE — do **not** blindly copy 2.4 TB. Pick by what you reproduce:
| You want to reproduce… | Need | ~size |
|---|---|---|
| **CTS / no-vis results + CTS figures** (the recent campaign) | `carla_geobev` + **val-only** baseline images | **~30 GB** |
| **+ VP / mVRS robustness eval & VP qual rows** | + `carla_VR` (all 31 variants) | **+1.1 TB** |
| **+ retraining** | + full `simbev_compare/sweeps2` (train images) | **+1.3 TB** |

For eval-only reproduction (no retraining) you need just the **val** baseline
images, not the 1.3 TB train set. `07_image_subset.sh` extracts the exact image
list referenced by the 3 `*_infos_val.pkl` and rsyncs only those (~30 GB).
`carla_VR` is effectively all-or-nothing for VP (every val frame × 31 variants).

`carla_geobev/sweeps/RGB-*` & `DPT-*` are **symlinks into `simbev_compare`**;
`04_sync_data.sh -L` materialises them, but for the eval-only path use
`07_image_subset.sh` instead (far smaller).

`BEVFormer/data/nuscenes` is a **symlink → carla_geobev**. Recreate it (step 3).

### Eval checkpoints (per platform P ∈ {sedan, suv, bus})  — full-repro set ≈ 10–15 GB
| Model | path (latest.pth are symlinks → epoch_24) |
|---|---|
| BEVFormer | `work_dirs/bevformer_tiny_carla_<P>/epoch_24.pth` |
| DFA3D | `3D-deformable-attention/BEVFormer_DFA3D/work_dirs/bevformer_DFA3D_carla_<P>/epoch_24.pth` (480 MB) |
| DETR3D | `detr3d/work_dirs/detr3d_carla_<P>/epoch_24.pth` |
| CAPE | `CAPE/ckpts/CAPE_ckpt/<P>/latest.pth` (693 MB) |
| BEVDet | `BEVDet/work_dirs/bevdet-r50-carla_<P>/epoch_24.pth` (596 MB) |
| BEVDepth | `BEVDepth/outputs/cts_ckpt_<P>.ckpt` → `…/carla_<P>/BEVDepth-CARLA/<runid>/checkpoints/epoch=23-step=6528.ckpt` (908 MB) |

Full `work_dirs/` (all training epochs) ≈ **141 GB** — only needed if you want to
resume/continue training (scope "everything"). `03_collect_checkpoints.sh`
grabs just the eval set by default.

### Results & memory (small, carry over)
- `results/` (incl. `ROBOGEO_CTS_NOVIS_FULL.json`, `qual_grid/`) — in the code tar.
- Project memory `~/.claude/projects/-home-hanyan-arch-viewpoint-BEVFormer/memory/`
  — optional; copy to keep assistant context.

---

## 2. New-server prerequisites
- **GPU**: NVIDIA B200 (or same arch the envs were built for). Driver must
  support **CUDA 12.8** (`nvidia-smi` → CUDA Version ≥ 12.8).
- **Disk**: ≥ 250 GB (envs ~57 G + data + eval ckpts ~15 G + headroom). For the
  "everything incl. work_dirs" scope, ≥ 400 GB.
- **miniconda/anaconda** installed (any base). `conda-pack` on the *source* only.
- Same OS family (Linux x86-64). glibc ≥ source's.

---

## 3. Migration steps (run in order)

All scripts live in `BEVFormer/migration/`. **Edit the `DEST_*` variables at the
top of each script** (new-server host/paths) before running.

### Step 1 — Code
```bash
bash migration/02_pack_code.sh          # → /tmp/robogeo_migrate/code_BEVFormer.tar.zst
# transfer the tar to the new server, then on the NEW server:
mkdir -p /home/hanyan_arch/viewpoint && cd /home/hanyan_arch/viewpoint
tar --zstd -xf code_BEVFormer.tar.zst   # recreates BEVFormer/ working tree
```

### Step 2 — Conda envs (the hard, valuable part)
```bash
# SOURCE server (needs conda-pack: `pip install conda-pack` in base):
bash migration/01_export_envs.sh        # → /tmp/robogeo_migrate/envs/<name>.tar.gz  (+ .yml manifests)
# NEW server: place each env under your conda envs dir and unpack:
mkdir -p $CONDA_ROOT/envs/bevformer-b200
tar -xzf bevformer-b200.tar.gz -C $CONDA_ROOT/envs/bevformer-b200
conda activate bevformer-b200 && conda-unpack    # rewrites the prefix paths
# repeat for legacy-mmdet140-b200, bevdet-b200, bevdepth-b200 (+ pdbev-b200, coin3d)
```
**[non-mirror]** If conda root differs, `conda-unpack` fixes intra-env paths, but
`run_bevdepth.sh` still hard-codes the giyong env path → run step 5.

### Step 3 — Data
```bash
bash migration/04_sync_data.sh          # rsync -aL the 4 data roots to the new server
# on the NEW server, recreate the symlink the code expects:
ln -s /NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev \
      /home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes
```
`rsync -L` materialises the `carla_geobev/sweeps/*` image symlinks, so you do NOT
need to preserve the cross-Lustre `simbev_compare` link structure — but you DO
need `carla_VR/` (variant renders) and `viewpoint_metadata.json` for VP eval.

### Step 4 — Eval checkpoints
```bash
bash migration/03_collect_checkpoints.sh   # stages eval ckpts (symlinks resolved) → /tmp/robogeo_migrate/ckpts/
# transfer, then place each back at its original relative path under BEVFormer/
# (the script also emits restore_ckpts.sh that recreates the dirs + latest.pth links)
```

### Step 5 — Path fixups  **[only if you did NOT mirror the roots]**
```bash
bash migration/05_fixup_paths_on_new_server.sh   # sed-rewrites old roots → new ones in code
```
Hand-check these specific hard-coded refs regardless:
- `bev_det_benchmark/build_condition_pkls*.py`: `VR_ROOT = '/NHNHOME/.../carla_VR'`
- `bev_det_benchmark/run_bevdepth.sh`: `ENV=/NHNHOME/.../giyong/miniconda3/envs/bevdepth-b200`
- `BEVFormer/data/nuscenes` symlink target
- conda base path in the activation snippet (`source $(conda info --base)/...`)

### Step 6 — Verify (no retraining)
```bash
bash migration/06_verify.sh             # single-cell eval per model + reproduce one CTS number
```
Expected anchors (sedan-source, vis2/vis0):
- BEVFormer SUV-CAL CTS ≈ **71.9 / 72.5 %**
- CAPE SUV-CAL CTS ≈ **34.3 / 37.0 %**
- BEVDepth BUS-CAL CTS ≈ **16.6 / 0.4 %**
Match ⇒ migration faithful. Also re-run `bev_det_benchmark/score_novis_from_logs.py`
and diff against the shipped `results/ROBOGEO_CTS_NOVIS_FULL.json`.

---

## 4. Gotchas (learned the hard way)
- **conda-pack, not `conda env create -f yml`** — the compiled CUDA ops won't
  rebuild; the yml is a manifest only.
- **Working tree, not clean clone** — the dual-vis patch & dataset classes are
  uncommitted. Tar the tree.
- **`latest.pth` are symlinks** → `epoch_24.pth`. Copy with `-L` or recreate the
  link (the collect script does both).
- **carla_VR frame ×2 rule** is in code (`vr_image_path`), data-independent —
  survives migration as long as carla_VR images come along.
- **BEVDepth env is at the giyong path** and `run_bevdepth.sh` hard-codes it —
  the single most likely thing to break on a non-mirror move.
- **Lustre `du` is slow**; size the data on the new (local) FS after sync.
- Don't move `bev_det_benchmark/out/` and `…/test/` prediction dumps — huge,
  regenerated on eval. `02_pack_code.sh` excludes them.
- **Retraining backbones**: configs `pretrained='ckpts/resnet50_msra-5891d200.pth'`
  (ImageNet R50). Present in `{detr3d,CAPE,PETR}/ckpts/`; `FULL=1` ckpt sync
  carries `ckpts/` + `detr3d/ckpts/`. Ensure each repo that trains has its
  `ckpts/resnet50_msra-5891d200.pth`. The `ckpts/r101_dcn_fcos3d_pretrain.pth`
  symlink points at a **different old path** (`/home/hanyan_arch/BEVFormer/…`, not
  `viewpoint/BEVFormer`) — re-point or drop it; the CARLA configs use R50, not it.
- BEVDet/BEVDepth use **DPT depth pretrain** + their own R50 — those live in
  their `work_dirs`/`outputs` (covered by `FULL=1`).

---

## 5. Quick checklist
- [ ] new server: B200 + driver CUDA ≥ 12.8, ≥ 250 GB free, miniconda
- [ ] code tar extracted to `/home/hanyan_arch/viewpoint/BEVFormer`
- [ ] 4 (+2) envs unpacked + `conda-unpack`
- [ ] data rsync'd; `data/nuscenes` symlink recreated; `carla_VR` present
- [ ] eval checkpoints restored (18 files, symlinks fixed)
- [ ] path fixups (only if non-mirror)
- [ ] `06_verify.sh` matches the CTS anchors
