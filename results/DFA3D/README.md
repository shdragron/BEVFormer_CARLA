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

## vp/ (FULL 3792 fps79; fps16 subset also kept), cts/ (full 3792)  — full arrived 2026-06-14

**VP 1/7 mVRS (%, FULL 3792):** EXT **90.5** / IMG **86.1** / CAL **93.1**
(per-cam .9461/.9219/.9614, all-cam .6601/.4947/.7462, Normal NDS 0.4892).
per-axis(roll/pitch/yaw) EXT .9667/.8946/.8545, IMG .8831/.8391/.8603,
CAL .9009/.9028/.9883; per-cam IMG F/FL/FR/B/BL/BR
.8964/.9365/.9680/.8414/.9230/.9659. (fps16 subset Δ≤0.1pt — 대표성 재확인;
**"(sub)" 제거, 완전 full**.)
**CTS (full, oracle-normalized %):** suv EXT 68.7 / IMG **38.8** / CAL 47.5;
bus EXT 37.1 / IMG **28.8** / CAL 48.1 (oracles suv 0.5138 / bus 0.5560).

### Quadrant verdict (sampling × depth — the controlled case study)

1. **VP(같은 rig): depth 게이트는 강건성을 더한다.** vs BEVFormer(83.2/83.9/93.6
   full): EXT +7.2pt, IMG +2.2pt, CAL 동급. CAL−EXT 부호 **양성 유지(+2.8)** —
   weight-path 게이트가 투영 샘플링의 CAL 회복을 깨지 않음(설계 예측 ✓). 우려했던
   "게이트가 stale extrinsic 손상을 증폭"은 mVRS 수준에서 나타나지 않음 — 오히려
   잘못 투영된 샘플을 게이트가 죽이고 hit-count 정규화가 재가중해 **내장 outlier
   rejection**처럼 작동.
2. **CTS(새 rig): 학습 depth의 rig prior가 게이트를 오염.** bus IMG 28.8 —
   BEVDet(0.2)식 붕괴는 면했고 BEVFormer(18.0)보다도 높음(depth를 splat이 아니라
   가중치로 쓰는 것의 가치 ✓). 그러나 **CAL 회복이 제한**: suv CAL 47.5 vs
   BEVFormer 71.9, 그리고 **suv에서 CAL−EXT 부호가 음수(−21.2pt)로 반전** —
   올바른 extrinsic이 샘플 좌표는 고치지만, sedan 이미지로만 학습된 image-only
   depth head가 새 플랫폼 이미지에서 편향된 분포를 내놓아 게이트가 올바른 샘플을
   억압. 6캠 mAP가 suv IMG/CAL에서 0.039/0.094로 붕괴하는 것이 그 흔적.

**종합**: 게이트 메커니즘 자체는 검증(온-rig 강건성+CAL 보존), 전이 실패의
원인은 메커니즘이 아니라 **depth의 출처(단일 rig 학습)** — "learned modules must
not carry rig priors" 법칙의 사분면 확장이자, frozen 파운데이션 depth로 게이트
분포를 교체하는 LatentCalib 설계의 직접적 motivation 실험. 벤치마크 §5 관점:
부호 진단의 정밀화 — *부호는 extrinsic 소비 위치를 따르되, 소비 경로에 학습된
rig prior가 끼면(CTS에서) 오염된다.*
