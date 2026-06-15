---
name: detr3d-carla-setup
description: "DETR3D CARLA port — the 3rd/last sparse detector; env, configs, dataset, R50-msra ckpt, no-DN note, what's verified vs pending"
metadata:
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

DETR3D added to the CARLA VP/CTS benchmark — the **last/6th detector**, 3rd sparse
method alongside PETRv2 + CAPE (see [[petr-cape-carla-setup]], [[bev-fair-comparison-matrix]]).
GPU-free prep DONE + verified (2026-06-04); training pending (waits for GPUs).

**Repo:** `viewpoint/BEVFormer/detr3d` — clean upstream clone of WangYueFt/detr3d
(origin still upstream → needs own fork to share, like PETR/CAPE). Same
`legacy-mmdet140-b200` env (mmcv 1.4.0 / mmdet 2.14.0 / mmdet3d 0.17.1) as PETR/CAPE.
Plumbing (all symlinks; `data` + `*.pth` already in .gitignore so not committed):
`detr3d/mmdetection3d` → `CAPE/mmdetection3d` (provides configs/_base_; shows as
git ` T` submodule→symlink type change, handle at push like PETR), `detr3d/data/nuscenes`
→ `/NHNHOME/.../jeongtae/carla_geobev`, `detr3d/ckpts/resnet50_msra-5891d200.pth`
→ `PETR/ckpts/...`. Run scripts from inside the repo (NOT /tmp — /tmp/mmcv shadows).

**KEY DIFFERENCE from PETRv2/CAPE — DETR3D has NO DN.** It's the original sparse-query
detector (predates query denoising), so "DN KEPT" doesn't apply — no-DN IS its native
paradigm and we keep it. Detr3DHead has no DN. Still fair (keep each method's native
paradigm). In the view-transform taxonomy DETR3D is **Sparse** (Detr3DCrossAtten samples
multi-view features at projected 3D ref points). Also native: multi-scale **FPN** (4-level,
KEPT — DETR3D needs it, architectural not a fairness knob) and **full-res 1600×900** input
(no IDA resize; kept as its native resolution — resolution is NOT equalized across models).

**Configs:** `detr3d/projects/configs/detr3d/detr3d_carla_{sedan,suv,bus}.py` (base:
detr3d_res101_gridmask.py). Fairness diffs: R101→**R50**+DCN ImageNet (resnet50_msra,
**style='caffe' std=[1,1,1] to_rgb=False**), out_indices=(0,1,2,3) kept for FPN,
frozen_stages=-1 + img_backbone lr_mult=0.1 (parity w/ PETR/CAPE); **load_from=None**
(drop the FCOS3D 3D-pretrain → ImageNet-only, the fairness point); use_grid_mask=False +
PhotoMetricDistortionMultiViewImage removed (all aug off); num_classes=6; fp32 grad_clip
(no fp16); custom_hooks=[] (no EMA); no CBGS. pc_range=[-51.2,-51.2,-5,51.2,51.2,3] +
post_center_range=[-61.2,..,61.2,61.2,10] (pc_range already == upstream default). KEEP
FocalLoss + DCN. `CarlaNuScenesDataset` ported into `projects/mmdet3d_plugin/datasets/`
(subclasses detr3d's CustomNuScenesDataset; identical vis≥2 + 6-class NDS eval as
BEVFormer/CAPE/PETRv2; registered in both __init__.py). ann_file `{vehicle}_infos_{train,val}.pkl`
(SAME pkls as CAPE/PETR → identical GT), use_valid_flag=True.

**lr/batch:** config default samples_per_gpu=4 (2 GPU → total batch 8 == paper) + lr=2e-4
(paper-faithful). For our batch-16 runs override via --cfg-options with **sqrt scaling**
(team convention): lr=2.83e-4 = 2e-4×√(16/8); NEVER pass --auto-scale-lr. DETR3D at
1600×900 is the **heaviest** of the 3 → batch 16 likely OOMs even on B200; fall back to
batch 8 (4/gpu) / lr 2e-4 (sqrt of 8/8). See [[petr-cape-carla-setup]] lr discussion.

**Verified (GPU-free, 2026-06-04):** build_model=Detr3D(6cls) OK; build_dataset(val)=3792;
get_data_info lidar2img=(6,4,4); Test B GT-center projection rate **1.0000**; lidar2img
**byte-identical** to canonical pkl projection (== CAPE/BEVFormer → NDS comparable); msra
ckpt loads with real-weight missing == DCN conv_offset only (18) + 53 harmless
num_batches_tracked buffers (=71 total; PETR/CAPE's "18" just didn't count buffers).
**TRAINING (sedan, started 2026-06-06 on viewpoint/BEVFormer/detr3d, env legacy-mmdet140-b200,
2×B200):** runs cleanly — batch4@1600×900 = **61GB/GPU (no OOM)**, ~0.62s/iter, loss ~7.6
(12-term: cls+bbox+d0..d4 aux), grad_norm ~42 (clipped at 35), **ETA ~9.5h for 24ep**.
Launched via `PORT=28603 bash tools/dist_train.sh detr3d_carla_sedan.py 2 --no-validate`.
**3 porting fixes (committed detr3d 1a06e7a, for mmcv1.4/mmdet2.14/torch2x):**
(1) tools/train.py accept `--local_rank` + `--local-rank` (torch distributed.launch passes
hyphen); (2) drop WandbLoggerHook `log_artifact=` kwarg (not in mmcv1.4); (3) **in-training
EvalHook is BROKEN** — mmcv-1.4 DDP scatter doesn't unwrap DETR3D's MultiScaleFlipAug3D
DataContainer → `img.size()` AttributeError in `extract_img_feat`; workaround `--no-validate`
+ eval separately via `tools/test.py` (single-GPU MMDataParallel scatter unwraps fine). So
val NDS is NOT on W&B during training (only train loss); run tools/test.py per-ckpt for NDS.
**ORACLES DONE (2026-06-07):** suv+bus oracle trains finished (chained, 2-GPU 24ep each).
Val NDS (6-class, separate tools/test.py single-GPU): **sedan 0.5336, suv 0.5602, bus 0.6019**
(mAP6 suv 0.515, bus 0.585). NB **bus > suv > sedan** — REVERSE of BEVDepth (suv 0.556 > sedan
0.541 > bus 0.428); DETR3D's full-res 1600×900 handles the high bus mount well. P_NORMAL gate
PASSED (sedan 0.5336 fine, no FCOS3D convergence issue). CTS oracle denominators: P_TARGET suv
0.5602 / bus 0.6019.
**CTS DONE (2026-06-07, full 3792 val):** suv NORMAL/EXT/IMG/CAL CTS = 0.900/0.429/**0.349**/0.769;
bus = 0.774/0.307/**0.265**/0.376 (IMG primary; CTS=NDS/oracle). DETR3D transfers to bus better than
BEVFormer (IMG bus 0.265 vs 0.180) — full-res robustness to mount gap. Results: results/DETR3D/cts/
+ README. **First end-to-end validation of the sparse/ bundle** — oracle passes reproduce direct
tools/test.py NDS EXACTLY (suv 0.5602/bus 0.6019). **Bug fixed in sparse/eval_cts_det.py:
BEVF_ROOT was dirname(HERE)=bev_det_benchmark (wrong, sparse/ adds a nesting level) -> fixed to
dirname(dirname(HERE))=BEVFormer** (the "statically verified, not end-to-end tested" caveat bit us).
Ran 2 instances (--targets suv / bus) parallel on 2 GPUs, separate tags, combined into results.
**VP DONE (2026-06-08, frame-fixed builder, 768-subset, 631-cell grid, sparse/ bundle 2-GPU shard
+ merge, ~8.8h):** ER all-cam 0.438, VR(IMG) 0.422, CR(CAL) **0.845**, Normal NDS 0.5368.
**EXT≈IMG + strongest CAL recovery (0.845, incl pitch 0.825)** — DETR3D (Sparse) patterns with
BEVFormer (Dense), NOT the Depth/LSS models. Confirms the cross-model **depth-vs-sampling family
split** (not Dense-vs-Sparse): sampling (BEVFormer,DETR3D)=EXT≈IMG+full CAL recovery; Depth(BEVDet)=
IMG<EXT+pitch-locked CAL. Results: results/DETR3D/vp/ + VP_CROSS_MODEL_ANALYSIS Finding D. Gotchas:
(1) sparse VP driver writes JSON shards to outdir RELATIVE to cwd=detr3d repo → landed in
detr3d/bev_det_benchmark/...; moved to bev_det_benchmark/sparse/out_vp_detr3d/vp_detr3d_sedan/ before
--merge. (2) summary header hardcodes "BEVDet NDS" (cosmetic copy in sparse driver) — data is DETR3D.
**STILL PENDING:** push fork. (BEVDepth VP VR/CR also pending — user said DETR3D-only.)
