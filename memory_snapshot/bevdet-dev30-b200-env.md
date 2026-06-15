---
name: bevdet-dev30-b200-env
description: "How to run the local BEVDet (dev3.0, mmcv-1.x) repo on the B200 box — env clone, version/op/runtime fixes"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 09f0506b-7473-403c-8e74-a3ace5c5a83b
---

The local repo `/home/hanyan_arch/viewpoint/BEVFormer/BEVDet` is `shdragron/BEVDet_CARLA` branch **dev3.0** (HEAD c9b3e58), mmdet3d **1.0.0rc4** (mmcv-1.x stack), carla configs = **BEVDetDepth** (depth-supervised). To run it on the B200 (sm_100) box, I built env **`bevdet-b200`** = clone of `bevformer-b200`, plus these fixes (all reusable):

- `pip install mmdet==2.28.2 --no-deps` (repo wants mmdet 2.24–3.0; clone had 2.14) + `pip install mmengine` (checkpoints unpickle a mmengine ref).
- Relaxed asserts in `BEVDet/mmdet3d/__init__.py`: mmcv max 1.7.0→**1.7.1**, mmseg min 0.20→**0.14.0**.
- Built the one CUDA op: `cd BEVDet && CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=10.0 CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13 pip install -v -e . --no-deps --no-build-isolation`. **CUDA 12.8 needs gcc<14**; conda g++ is 14.3 → must use system **/usr/bin/g++-13**.
- Code fixes in the repo: `models/detectors/__init__.py` guard `from .dal import DAL` (needs spconv, unused); `tools/{train,test}.py` add `--local-rank` alias (torch 2.x launch passes hyphen); bulk-replaced deprecated numpy aliases (`np.bool/float/int/long`) across mmdet3d+tools.
- Runtime env var for every run: **`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`** (torch 2.6+ defaults weights_only=True; checkpoints have non-tensor meta).
- `tools/train.py` validation is OFF by default (it has `--validate` opt-in, NOT `--no-validate`).

bevdet-b200 baseline: torch 2.11.0+cu128, mmcv 1.7.1, mmdet 2.28.2, mmengine 0.10.7, mmdet3d 1.0.0rc4 (editable). See [[carla-bevformer-conda-env]], [[bevdet-checkpoints-provenance]].
