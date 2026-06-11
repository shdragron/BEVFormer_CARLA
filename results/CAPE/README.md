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

### 2b. VP FULL-FRAME (no sampling, all 3792) — `vp_cape_sedan_fullframe.*`
Same VP but every frame (not the fps=16/768 subset). NDS_Normal=0.5547. Confirms
full ≈ subset (every condition within ~0.004 of the fps=16 numbers above → the
fps=16 subset was representative; RRS is a ratio so frame-count-insensitive).

| | ROLL | PITCH | YAW | **ALL** | (fps16 ALL) |
|---|---|---|---|---|---|
| EXT | 0.9915 | 0.9849 | 0.8438 | **0.9401** | 0.9430 |
| IMG★ | 0.8649 | 0.8041 | 0.8479 | **0.8390** | 0.8428 |
| CAL | 0.8627 | 0.8044 | 0.9904 | **0.8858** | 0.8898 |

## 3. CTS cross-platform transfer (sedan→target), CTS=NDS/P_TARGET — `cts_cape_sedan.{json,csv,_summary.txt}`
|  | P_TARGET(oracle) | EXT | IMG★ | CAL |
|---|---|---|---|---|
| suv | 0.5964 | 0.8230 | 0.3379 | 0.3433 |
| bus | 0.4490 | 0.5750 | 0.3598 | 0.3294 |

## Files
- `indist_{sedan,suv,bus}.json` — per-vehicle in-dist full metric dicts (epoch 24).
- `vp_cape_sedan.{json,_summary.txt,_per_config.csv}` — VP eval, **fps=16 subset** (631 cells, ER/VR/CR, x2-fixed).
- `vp_cape_sedan_axis_table.txt` — VP RRS(1/7) EXT/IMG/CAL × ROLL/PITCH/YAW/ALL (fps=16).
- `vp_cape_sedan_fullframe.{json,_summary.txt,_per_config.csv,_axis_table.txt}` — **VP FULL-FRAME** (all 3792, 631 cells, x2-fixed).
- `cts_cape_sedan.*` — CTS results (full 3792/condition).

### Per-class (6 CARLA classes) — extracted tables
The full per-class detail (AP@4 dist + 5 TP errors per class) lives inside every
`*.json` (indist_*.json directly; vp/cts under `rows[].metrics`). Clean CSV extracts:
- `indist_perclass.csv` — vehicle × class × {AP, mATE, mASE, mAOE, mAVE, mAAE}.
- `cts_perclass.csv` — platform × condition(ORACLE/NORMAL/EXT/IMG/CAL) × class × {AP, TP errors}.
- `vp_perclass_AP_RRS.csv` — condition(EXT/IMG/CAL) × class × per-axis AP-retention RRS(1/7)
  (AP-based, since NDS is not per-class; ROLL/PITCH/YAW/ALL). fps=16.
- `vp_perclass_fullframe_AP_RRS.csv` — same, full-frame (all 3792).

Note: an earlier VP run had a carla_VR frame-rate bug (geobev is ½-rate relabel → need ×2);
VR/CR were re-run with the fix. ER/CTS were unaffected (no carla_VR image lookup).
