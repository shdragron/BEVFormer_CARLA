---
name: vp-full-run-checkpoint-resume
description: Full-val (3792) VP no-stage runs crash ~6h in (dataloader worker OOM); driver now checkpoints per-cell + resumes; recover from logs if pre-checkpoint.
metadata: 
  node_type: memory
  type: project
  originSessionId: 09f0506b-7473-403c-8e74-a3ace5c5a83b
---

The **full-val VP** run (`eval_vp_robustness_det_bevdet.py`, `--frames-per-scene 79`
→ 3792 frames, no-stage = read carla_VR off Lustre) **crashes ~6 h in**: the 631-cell
loop builds a *fresh* dataloader per cell, so worker spawn/teardown churn leaks until a
DataLoader worker is OOM/`/dev/shm`-killed (`RuntimeError: DataLoader worker ... killed
by signal: Killed` + `ConnectionResetError`). System RAM is fine (node has ~2.2 TB);
it's the per-allocation/`/dev/shm` fd pool, same family as the BEVDepth fix in
[[bevdepth-carla-sedan-result]].

**Fixed in the driver (commit 7726f9d):** each finished cell is appended
(`flush`+`fsync`) to `out/vp_<tag>/progress_<i>of<n>.jsonl`, Normal (RRS denom) to
`normal_<i>of<n>.json`. Relaunch with the **same `--tag` and `--shard`** → `load_progress`
/`load_normal` restore them and the loop **skips done cells** (Normal not recomputed).
Also added `torch.multiprocessing.set_sharing_strategy('file_system')` (spills shared
tensors to `TMPDIR` instead of `/dev/shm`). `--merge` unchanged (final `shard_<i>of<n>.json`
assembled from resumed+new rows).

**Recover a PRE-checkpoint crash from the shard log** (this salvaged ~6 h once): the log
prints, per cell, a `[CARLA-METRICS-JSON] {…full per-class res dict…}` line immediately
followed by `[VP k/N] <cond> <axis><±mag> <proto> NDS=.. RRS=..`. Parse them paired
(metrics dict belongs to the next `[VP]` line; the `[VP] Normal NDS=..` line gets the
metrics seen just before it), take exact NDS/mAP from `pts_bbox_NuScenes/NDS` /
`…/mAP` (6-class), recompute `rrs = nds/nds_norm`, and write `normal_<sfx>.json` +
`progress_<sfx>.jsonl`. Validate: every recovered cell key must fall in its own
`shard_slice` (stray=0) before relaunch. The in-flight cell at crash (no metrics line)
is simply re-run.

**Launcher pattern** (`/tmp/vp_full_resume.sh`): per-shard retry loop — relaunch until
`shard_<i>of2.json` exists (cap ~15), `workers 4` (less churn), `TMPDIR=/tmp`,
2-GPU split (CUDA_VISIBLE_DEVICES 0/1, numactl nodes 0/1), then `--merge`. Detached via
`setsid nohup`; not a harness task, so it needs an explicit Monitor.

Sanity already established: full ≈ fps16-subset (ER cells |ΔRRS| mean 0.004), so the full
run is a more-precise BEVDet number, not a cross-model swap — the cross-model table stays
matched at 768 ([[vp-cross-model-mechanism-finding]]). Frame-2× fix is live in it
([[vp-carla-vr-frame-2x-misalignment]]).

**EXTENDED to the OTHER 4 drivers (2026-06-08):** only `eval_vp_robustness_det_bevdet.py`
had this resume (jsonl scheme above). Audited the rest and found NONE had per-cell save /
skip-done (in-memory `rows` → all-at-end write → a SIGKILL loses the whole shard). Added the
same per-cell-persist + skip-done resume to **`eval_vp_robustness_det.py` (BEVFormer VP),
`sparse/eval_vp_robustness_det_sparse.py`, `eval_cts_det.py`, `sparse/eval_cts_det.py`** — via
a DIFFERENT scheme than bevdet's jsonl: a `_cell_path` helper + per-cell `<outdir>/cells[_shard{si}]/<key>.pkl`
(VP key=cond_axis±mag_proto, CTS key=target_cond) + `normal.pkl`/`ORACLE` oracle cache. Writes are
ATOMIC (`_save_cell`: tmp+`pickle.dump`+flush+`os.fsync`+`os.replace`); reads are SAFE (`_load_cell`:
try/except → corrupt/truncated returns None + removes file → recompute). Skip gates on the cache
load so ALL ranks take the same branch (collective-safe for the DDP BEVFormer VP driver; the others
are single-process/subprocess-per-cell). Final shard*.pkl / --merge / CSV/JSON/TXT writers UNCHANGED.
All 4 py_compile + empirical-verified (8-agent workflow). (bevdet driver keeps its own jsonl scheme.)
