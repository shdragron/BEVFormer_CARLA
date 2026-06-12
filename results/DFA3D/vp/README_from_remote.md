# DFA3D-CARLA results

BEVFormer + DFA3D (3D deformable attention, depth-aware lifting; R50 ImageNet, BEV 50x50,
single-frame, 6-class CARLA). Trained per platform (sedan/suv/bus, 24ep, 2xB200, bs16, lr4e-4),
then in-distribution + viewpoint robustness (VP) + cross-platform transfer (CTS). 6-class NDS/mAP (vis>=2).

> NOTE: eval_vp.json/summary mislabel `frames_per_scene=4` (driver default not updated); the run
> actually used **16/scene = 768 samples** (shard logs: "subset = 768 samples (16/scene x 48 scenes)").

## 1. In-distribution (P_NORMAL) — `indist_<veh>.json`
| vehicle | mAP | NDS |
|---|---|---|
| suv | 0.4694 | 0.5138 |
| sedan | 0.4354 | 0.4892 |
| bus | 0.5400 | 0.5560 |

## 2. VP viewpoint robustness (sedan), RRS(1/7) — `vp_dfa3d_sedan_axis_table.txt`
RRS = NDS_cell / NDS_Normal(0.4900). RRS(1/7) = (6*mRRS_percam + RRSALL)/7.
EXT=ER(extrinsic), IMG=VR(image, PRIMARY), CAL=CR(both). fps16 subset (768). carla_VR x2 frame map applied.

| | ROLL | PITCH | YAW | **ALL** |
|---|---|---|---|---|
| EXT | 0.9661 | 0.8932 | 0.8531 | 0.9041 |
| IMG★ | 0.8838 | 0.8382 | 0.8611 | 0.8610 |
| CAL | 0.9017 | 0.9036 | 0.9896 | 0.9317 |

(per-cam robust ~0.9+; all-cam image perturb is the weak axis — see RRSALL row in the txt.)

## 3. CTS cross-platform transfer (sedan->target), CTS=NDS/P_TARGET — `cts_dfa3d_sedan.*`
|  | P_TARGET(oracle) | EXT | IMG★ | CAL |
|---|---|---|---|---|
| suv | 0.5138 | 0.6870 | 0.3875 | 0.4747 |
| bus | 0.5560 | 0.3714 | 0.2876 | 0.4811 |

## Files
- `indist_{sedan,suv,bus}.json` + `indist_perclass.csv` — per-vehicle in-dist (epoch 24).
- `vp_dfa3d_sedan_fps16.{json,_summary.txt,_per_config.csv}` — VP eval (631 cells, ER/VR/CR, fps16/768).
- `vp_dfa3d_sedan_axis_table.txt` — VP RRS(1/7) EXT/IMG/CAL x ROLL/PITCH/YAW/ALL.
- `vp_perclass_AP_RRS.csv` — per-class AP-retention RRS(1/7).
- `cts_dfa3d_sedan.{json,csv,_summary.txt}` + `cts_perclass.csv` — CTS (full 3792/condition).
- VP Full (all 3792) still running -> `vp_dfa3d_sedan_fullframe.*` added on completion.
