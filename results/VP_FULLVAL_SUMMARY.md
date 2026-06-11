# Full-val VP 정리 (3792-frame, all-cam) — subset 대표성 검증

> sedan full-val(79 frames/scene × 48 = 3792) all-cam VP를 3개 검출 모델에 대해 재계산.
> 목적: 그동안 분석에 쓴 **768-subset all-cam mVRS가 full-val을 대표하는지** 확인.
> 출처: `bev_det_benchmark/out_vpfull/vp_tiny_sedan_fullval_allcam/` (BEVFormer),
> `out/vp_bevdet_sedan_full/`, `out/vp_bevdepth_sedan_full/`. RRS = NDS_cell / NDS_Normal.

## all-cam RRS (= mVRS all-cam) : full-val vs subset

| Model | Mech | Normal NDS | EXT (full / subset) | IMG (full / subset) | CAL (full / subset) | max Δ |
|---|---|---|---|---|---|---|
| **BEVFormer** | projection-sampling | 0.5037 | 0.425 / 0.428 | 0.425 / 0.426 | 0.779 / 0.777 | **0.003** |
| **BEVDet** | extract-then-place | 0.5166 | 0.610 / 0.610 | 0.360 / 0.364 | 0.512 / 0.521 | **0.009** |
| **BEVDepth** | extract-then-place | 0.5354 | 0.648 / 0.652 | 0.325 / 0.328 | 0.485 / 0.492 | **0.007** |
| **CAPE** | extract-then-place | 0.5547 | 0.803 / 0.811 | 0.400 / 0.407 | 0.553 / 0.560 | **0.008** |
| **DETR3D** | projection-sampling | — | 0.440 / 0.438 | 0.424 / 0.422 | 0.855 / 0.845 | **0.009** |

→ **4개 모델 모두 subset ≈ full-val (max Δ ≤ 0.009)**. 768-subset all-cam 숫자는 full-val을
신뢰성 있게 대표한다. (CAPE full-frame은 2026-06-10 cape_result.tar로 합류; 1/7 기준도
subset 94.30/84.28/88.98 vs full 94.01/83.90/88.58로 Δ≤0.4pt.)

## per-cam + mVRS (BEVDet/BEVDepth는 full-val에서 per-cam도 계산)

| Model | cond | mRRS (per-cam) | RRSALL (all-cam) | mVRS (1/7 headline) |
|---|---|---|---|---|
| BEVDet | EXT | 0.939 | 0.610 | 0.774 |
| BEVDet | **IMG** | 0.908 | 0.360 | **0.634** |
| BEVDet | CAL | 0.935 | 0.512 | 0.724 |
| BEVDepth | EXT | 0.945 | 0.648 | 0.796 |
| BEVDepth | **IMG** | 0.912 | 0.325 | **0.619** |
| BEVDepth | CAL | 0.938 | 0.485 | 0.712 |
| BEVFormer | (all-cam only) | — | EXT 0.425 / IMG 0.425 / CAL 0.779 | — |

- per-cam은 0.91–0.95로 압축(6-view 융합이 한 카메라를 outvote) → **헤드라인 mVRS가 EXT–IMG 격차를 가림**(BEVDet IMG mVRS 0.634인데 all-cam은 0.360). 분석은 all-cam이 변별력.
- BEVFormer full-val은 비용 절감으로 **all-cam만** 재계산(per-cam은 subset 사용; subset 대표성이 위에서 확인됨).

## 메커니즘이 full-val에서도 그대로 재현됨

- **gates-sampling (BEVFormer): EXT ≈ IMG** (0.425 ≈ 0.425), CAL 회복 (0.779).
- **extract-then-place (BEVDet/BEVDepth): EXT ≫ IMG** (0.61/0.65 ≫ 0.36/0.33), CAL 부분 회복 (0.51/0.49).
- 즉 §VP의 핵심 finding(EXT≠IMG가 메커니즘으로 갈림)은 **subset artifact가 아니라 full-val에서 성립**.

## 상태/남은 것 (2026-06-11 12:00 갱신)
- 완료: **검출 5종 모두 full-val all-cam 완료**(BEVFormer/BEVDet/BEVDepth/CAPE/DETR3D;
  BEVDet/BEVDepth/CAPE는 per-cam도 full). DETR3D full은 06-11 합류(위 표 행 추가,
  subset 대비 max Δ 0.009 — 5/5 모델에서 대표성 재확인).
- **DFA3D bus oracle 학습 완료**(06-11 05:25, epoch 24) → 3개 차량 oracle 완비, full
  VP/CTS 평가 대기(GPU는 현재 PD-BEV 학습 + BEVFormer per-cam이 점유).
- 실행 중: **BEVFormer per-cam full** 2-shard 262/540 셀(~7분/셀/샤드, ETA ~06-12 새벽).
- **중단: DETR3D per-cam 자동 체인** — allcam 종료 감시 워처가 06-11 kill됨. allcam은
  완료됐으나 per-cam 미시작. GPU 여유 시 같은 --tag 재실행으로 이어가면 됨(셀 resume).
- **본문 그림 반영(figures/paper/):** Fig. C의 all-cam IMG를 det 5종 모두 full로 교체
  (BEVFormer 0.425, DETR3D 0.424). 순위·ρ 불변(det +1.00 유지). 주의: BEVFormer-DETR3D
  격차는 0.4250 vs 0.4243으로 매우 얇음 — ρ=1.00의 민감 지점(뒤집히면 0.90).
- 미완: **PETRv2** 미평가.
- frame-2× fix 반영본(VR/CR), geobev 이미지 무관.

*숫자 출처: 위 3개 summary.txt. subset 값은 results/PAPER_ANALYSIS_KR.md / BENCHMARK_SUMMARY.md.*
