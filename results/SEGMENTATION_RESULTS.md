# GeoBEV — BEV Segmentation Results (VP + CTS)

> **3D detection 결과와 분리된 segmentation 전용 정리.** 6개 BEV-seg 모델
> (CVT, GaussianLSS, LaRa, LSS, PointBEV, SimpleBEV), 메트릭은 **IoU**(vis≥2),
> robustness는 **mVRS**(=½·(per-cam + all-cam) RRS). CTS는 sedan→target raw IoU.
> 출처: `results/eval_results.tar.gz` → `eval_results/bevunify-*-carla/`
> (VR=viewpoint robustness, CTS=cross-platform transfer, 최신 run 기준).
> 원자료: `results/seg_vp_cts.tsv`.

## 1. Viewpoint Robustness (VP) — sedan, mVRS over IoU

| Model | Normal IoU | mVRS$_{\text{EXT}}$ | mVRS$_{\text{IMG}}$ | mVRS$_{\text{CAL}}$ | EXT all-cam | IMG all-cam | CAL all-cam |
|---|---|---|---|---|---|---|---|
| CVT         | 0.424 | 0.818 | 0.660 | 0.800 | 0.690 | 0.416 | 0.653 |
| GaussianLSS | 0.489 | 0.820 | **0.682** | 0.828 | 0.698 | **0.453** | 0.699 |
| LaRa        | 0.454 | 0.824 | 0.639 | 0.786 | 0.703 | 0.378 | 0.627 |
| LSS         | 0.445 | 0.774 | 0.625 | 0.716 | 0.619 | 0.354 | 0.510 |
| PointBEV    | 0.481 | 0.615 | 0.585 | 0.853 | 0.364 | 0.289 | 0.756 |
| SimpleBEV   | 0.504 | 0.667 | 0.556 | 0.511 | 0.539 | 0.242 | 0.183 |

- **IMG가 모든 모델에서 가장 어려운 조건**(mVRS_IMG 0.56–0.68, all-cam 0.24–0.45) — detection과 동일 경향.
- IMG-robustness 순위: GaussianLSS > CVT > LaRa > LSS > PointBEV > SimpleBEV.
- per-cam(≈0.9)에서는 모델 차이가 압축되고 **all-cam에서 갈림** — detection과 같은 구조(헤드라인 mVRS는 차이를 가림).

## 2. Cross-Platform Transfer (CTS) — sedan→{suv,bus}, raw IoU

| Model | sedan IoU | suv EXT | suv IMG | suv CAL | bus EXT | bus IMG | bus CAL |
|---|---|---|---|---|---|---|---|
| CVT         | 0.424 | 0.192 | 0.194 | 0.133 | 0.016 | 0.083 | 0.009 |
| GaussianLSS | 0.489 | 0.409 | 0.186 | 0.209 | 0.220 | 0.082 | 0.213 |
| LaRa        | 0.454 | 0.321 | 0.152 | 0.189 | 0.121 | 0.149 | 0.115 |
| LSS         | 0.445 | 0.340 | 0.123 | 0.119 | 0.178 | 0.097 | 0.097 |
| PointBEV    | 0.481 | 0.253 | 0.104 | 0.352 | 0.026 | 0.063 | 0.086 |
| SimpleBEV   | 0.504 | 0.424 | 0.063 | 0.074 | 0.105 | 0.060 | 0.017 |

- **bus(높은 마운트)로의 전이가 suv보다 훨씬 심하게 붕괴** (특히 EXT·CAL): CVT/PointBEV/SimpleBEV는 bus에서 0.01–0.11.
- IMG 조건은 suv·bus 양쪽에서 0.06–0.19로 가장 낮음 — 기운 타깃-시점 이미지가 전이를 더 깨뜨림.
- LSS 계열(LSS/GaussianLSS/SimpleBEV)은 EXT 전이는 비교적 살지만(suv 0.34–0.42) IMG에서 무너짐.

> CTS는 현재 **raw target IoU**만 기록됨(target-oracle IoU 부재 → ratio 미산출).
> detection의 CTS-ratio와 맞추려면 target-trained(suv/bus) oracle IoU가 필요하다.

## 3. Per-axis all-cam retention (IoU/Normal) — pitch / yaw / roll

| Model | Cond | pitch | yaw | roll |
|---|---|---|---|---|
| CVT | EXT | 0.746 | 0.356 | 0.967 |
| CVT | IMG | 0.405 | 0.312 | 0.530 |
| CVT | CAL | 0.611 | 0.796 | 0.551 |
| GaussianLSS | EXT | 0.922 | 0.263 | 0.907 |
| GaussianLSS | IMG | 0.530 | 0.257 | 0.571 |
| GaussianLSS | CAL | 0.566 | 0.937 | 0.594 |
| LaRa | EXT | 0.872 | 0.287 | 0.951 |
| LaRa | IMG | 0.285 | 0.288 | 0.562 |
| LaRa | CAL | 0.361 | 0.929 | 0.592 |
| LSS | EXT | 0.779 | 0.262 | 0.818 |
| LSS | IMG | 0.264 | 0.287 | 0.509 |
| LSS | CAL | 0.258 | 0.778 | 0.493 |
| PointBEV | EXT | 0.214 | 0.266 | 0.613 |
| PointBEV | IMG | 0.146 | 0.262 | 0.459 |
| PointBEV | CAL | 0.542 | 0.952 | 0.773 |
| SimpleBEV | EXT | 0.301 | 0.953 | 0.362 |
| SimpleBEV | IMG | 0.078 | 0.260 | 0.387 |
| SimpleBEV | CAL | 0.080 | 0.247 | 0.222 |

- **IMG에서 yaw가 가장 파괴적**(all-cam yaw 0.26–0.31, 거의 모든 모델 일관) — all-cam yaw 교란이 **cross-camera correspondence**를 깨 BEV lifting을 무너뜨림. pitch/roll은 지면 투영만 이동시켜 상대적으로 덜 치명적(roll 0.39–0.57).
- **CAL에서 yaw는 크게 회복**(0.78–0.95) — 일관된 yaw 교란은 ground 외형을 보존하므로. pitch/roll은 CAL에서도 낮음(추출에 박힌 기울임).
- SimpleBEV는 예외적으로 EXT-yaw가 0.953로 높음(다른 모델과 반대) — 별도 확인 필요.

---
*VP=768-frame subset(per-axis는 all-cam). CTS=full target val. CTS 두 run 중 최신(0608) 사용.
원자료 `seg_vp_cts.tsv`, 디렉토리 `eval_results/bevunify-*-carla/`.*
