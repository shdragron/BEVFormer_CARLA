# DFA3D CARLA — results & checkpoints

The 7th detector: fills the empty **sampling×depth quadrant** of the mechanism
taxonomy. Paradigm = **Backward / projection-sampling WITH depth** — it subclasses
BEVFormer (`BEVFormer_DFA3D`): 3D deformable attention whose sampling weights are
modulated by a predicted depth distribution (`MSDeformableAttention3D_DFA3D` +
`DepthHead_MLVGDpt`, d_bound [2,58]×0.5m). In §5/proposal terms this is the
controlled case study for **G2** (depth as sampling weight, not feature placement):
prediction = CAL recovery retained (projection-sampling) + IMG precision gain (depth).

Setup = "BEVFormer-tiny + DPT depth": R50 ImageNet, 800×450 (0.5× of 1600×900),
BEV 50×50, single-frame (queue_length=2 but scene_token==token → prev_bev=None),
6-class CARLA eval. **DPT depth maps supervise training only (`loss_dpt`); at test
time depth is predicted by the model** — so VP/CTS perturbed images need no DPT files.
Training: 24 epochs, 2-GPU DDP samples_per_gpu=8 ×2 = global batch 16, lr 4e-4
(fair match to the BEVFormer-tiny baseline). wandb project `DFA3D_CARLA`.
Code: github.com/shdragron/DFA3D_CARLA (eval port: `BEVFormer_DFA3D/bev_det_benchmark/`,
commit de4d04d).

## ckpts/  (local only, gitignored)

| file | trained on | source |
|---|---|---|
| `dfa3d_carla_sedan_epoch24.pth` | sedan | `work_dirs/bevformer_DFA3D_carla/epoch_24.pth` (md5 bcfc0523…) |
| `dfa3d_carla_suv_epoch24.pth`   | suv   | `work_dirs/bevformer_DFA3D_carla_suv/epoch_24.pth` (md5 8930b9e6…) |
| `dfa3d_carla_bus_epoch24.pth`   | bus   | `work_dirs/bevformer_DFA3D_carla_bus/epoch_24.pth` (md5 b642f5b9…) |

+ the three configs (`bevformer_DFA3D_carla{,_suv,_bus}.py`) used to train them.
Originals live in the DFA3D fork's `work_dirs/` (finished sedan 06-09, suv 06-10,
bus 06-11).

## indist/  (committed)

Final-epoch (24) in-distribution val, own platform, full 3792, vis≥2, 6-class —
scraped from the training logs (`indist_<veh>_ep24.txt` + full
`[CARLA-METRICS-JSON]` detail in `indist_<veh>_ep24_metrics.json`):

| platform | NDS6 | mAP6 |
|---|---|---|
| sedan | 0.4892 | 0.4354 |
| suv | 0.5138 | 0.4694 |
| **bus** | **0.5560** | 0.5400 |

**bus > suv > sedan** — same ordering as DETR3D (and the reverse of BEVDepth),
consistent with projection-sampling handling the higher mount natively.

## vp/, cts/  (pending)

Full-val VP (631 cells) and CTS (oracle + NORMAL/EXT/IMG/CAL × suv/bus) to be run
via the fork's `bev_det_benchmark/run_vp_full_dfa3d.sh` / `run_cts_dfa3d.sh`
(both crash-resumable). Smoke-verified 2026-06-11: VP Normal NDS 0.4841 (48-frame
subset) + VR pitch−8 all-cam RRS 0.362; CTS path SUV-oracle 1-scene NDS 0.4743
with pred/GT tokens 79/79. Waiting on GPU (BEVFormer per-cam full + PD-BEV
training occupy both B200s); merged outputs will be copied here.
