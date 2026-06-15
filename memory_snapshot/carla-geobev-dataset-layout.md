---
name: carla-geobev-dataset-layout
description: "CARLA geobev dataset layout, image location, splits, and filter decision for the BEVFormer viewpoint experiment"
metadata: 
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

BEVFormer viewpoint experiment trains/evals **per ego-vehicle (camera viewpoint)**: sedan, suv, bus.

**Data root**: `data/nuscenes` is a symlink → `/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev` (the old target `jeongtae/carla` is gone). Under it, per vehicle:
- `v1.0-carla_<veh>` = train DB (220 scenes), `v1.0-carla_<veh>_eval` = eval DB (49 scenes). train DB ∩ eval DB = ∅.
- Rule: train on a vehicle → val on the SAME vehicle's `_eval`.
- `carla_VR/` (sibling dir) = later cross-viewpoint final eval: 50 scenes × 31 camera yaw/pitch/roll variants; extrinsics in `carla_VR/viewpoint_metadata.json`.

**Split**: `carla_geobev/split/{train,val}.txt` (one scene name per line, e.g. `scene_0000`). Same split for all vehicles (scene names identical). train.txt matches each train DB exactly; val.txt ⊂ each eval DB (drops 1 extra scene). Converter filters by these.

**Images** (IMPORTANT, non-obvious): `carla_geobev/sweeps/RGB-*` etc. are symlinks. They originally pointed to `/home/hanyan_arch/data/simbev_compare/sweeps2/` which does NOT exist on this machine. Real images live at `/NHNHOME/WORKSPACE/0526040099_A/jeongtae/simbev_compare/sweeps2/`. The 54 broken symlinks were repointed there (Jun 2026). Naming: no-prefix `RGB-CAM_FRONT` = subcompact(=sedan); `RGB-bus-*`/`RGB-suv-*` = bus/suv. `LIDAR` symlink (→ simbev/sweeps/LIDAR) was already valid.

**valid_flag filter decision**: keep `visibility_token >= '2'` (40%+ visible). Keeps ~34% of annotations (vis=1 is 66%, i.e. in-frame but heavily occluded). num_lidar_pts IS populated (stock BEVFormer's `num_lidar_pts+radar>0` would keep ~49.7%) but user chose visibility>=2 deliberately. Classes: car, truck, bus, motorcycle, bicycle, pedestrian (6).

See [[carla-bevformer-conda-env]] for the runtime env. Coordinate convention (gt_boxes = [xyz, wlh, -yaw-π/2]) matches stock nuscenes converter — verified by projection (tools/check_carla_projection.py, `--db` and `--vis-compare` modes).
