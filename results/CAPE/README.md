# CAPE-CARLA results

CAPE (R50+DCN, ImageNet, single-frame, DN; 6-class CARLA) — trained per platform
(sedan/suv/bus, 24ep, 2×B200, fp32, batch16, lr4e-4), then evaluated for
in-distribution + viewpoint robustness (VP) + cross-platform transfer (CTS).

All metrics are the 6-class CARLA NDS/mAP (visibility≥2 GT).

## 1. In-distribution (P_NORMAL) — `indist_<veh>.json`
| vehicle | mAP | NDS |
|---|---|---|
| suv   | 0.5675 | 0.5964 |
| sedan | 0.5128 | 0.5544 |
| bus   | 0.2835 | 0.4490 |

## 2. VP viewpoint robustness (sedan model), RRS(1/7) — `vp_cape_sedan_axis_table.txt`, `vp_cape_sedan.json`
RRS = NDS_cell / NDS_Normal(0.5508). RRS(1/7) = (6·mRRS_percam + RRSALL)/7
(all-cam counted as a 7th camera). EXT=ER(extrinsic), IMG=VR(image, PRIMARY), CAL=CR(both).
Frozen fps=16 subset (768). carla_VR frame map fixed (geobev N → VR 2N).

| | ROLL | PITCH | YAW | **ALL** |
|---|---|---|---|---|
| EXT | 0.9939 | 0.9874 | 0.8476 | **0.9430** |
| IMG★ | 0.8691 | 0.8076 | 0.8517 | **0.8428** |
| CAL | 0.8670 | 0.8074 | 0.9949 | **0.8898** |

(per-cam robust ~0.9+; all-cam image perturb is the weak axis — see RRSALL row in the txt.)

## 3. CTS cross-platform transfer (sedan→target), CTS=NDS/P_TARGET — `cts_cape_sedan.{json,csv,_summary.txt}`
|  | P_TARGET(oracle) | EXT | IMG★ | CAL |
|---|---|---|---|---|
| suv | 0.5964 | 0.8230 | 0.3379 | 0.3433 |
| bus | 0.4490 | 0.5750 | 0.3598 | 0.3294 |

## Files
- `indist_{sedan,suv,bus}.json` — per-vehicle in-dist full metric dicts (epoch 24).
- `vp_cape_sedan.json` — full VP eval (631 cells, ER/VR/CR, x2-fixed). [added when the full VP run finishes]
- `vp_cape_sedan_axis_table.txt` — VP RRS(1/7) EXT/IMG/CAL × ROLL/PITCH/YAW/ALL.
- `vp_cape_sedan_VRCR_fixed.json` — VR/CR x2-fixed rows (interim; superseded by vp_cape_sedan.json).
- `cts_cape_sedan.*` — CTS results (full 3792/condition).

Note: an earlier VP run had a carla_VR frame-rate bug (geobev is ½-rate relabel → need ×2);
VR/CR were re-run with the fix. ER/CTS were unaffected (no carla_VR image lookup).
