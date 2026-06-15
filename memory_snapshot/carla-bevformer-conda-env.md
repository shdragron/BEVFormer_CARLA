---
name: carla-bevformer-conda-env
description: Conda env to run this BEVFormer repo (mmcv/mmdet3d/nuscenes-devkit)
metadata: 
  node_type: memory
  type: reference
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

Run BEVFormer code in conda env **`bevformer-b200`** (`/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevformer-b200`). It has mmcv + mmdet3d + nuscenes-devkit + cv2. The shell's `base` python does NOT have mmcv/mmdet3d.

Use `conda run -n bevformer-b200 python ...`. Other envs with mmcv+mmdet3d also exist (`legacy-mmdet140-b200`, `streampetr-b200`) but bevformer-b200 is the one for this repo.

Related: [[carla-geobev-dataset-layout]].
