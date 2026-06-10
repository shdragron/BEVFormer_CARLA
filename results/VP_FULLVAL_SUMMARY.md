# Full-val VP 정리 (3792-frame, all-cam) — subset 대표성 검증

> sedan full-val(79 frames/scene × 48 = 3792) all-cam VP를 3개 검출 모델에 대해 재계산.
> 목적: 그동안 분석에 쓴 **768-subset all-cam mVRS가 full-val을 대표하는지** 확인.
> 출처: `bev_det_benchmark/out_vpfull/vp_tiny_sedan_fullval_allcam/` (BEVFormer),
> `out/vp_bevdet_sedan_full/`, `out/vp_bevdepth_sedan_full/`. RRS = NDS_cell / NDS_Normal.

## all-cam RRS (= mVRS all-cam) : full-val vs subset

| Model | Mech | Normal NDS | EXT (full / subset) | IMG (full / subset) | CAL (full / subset) | max Δ |
|---|---|---|---|---|---|---|
| **BEVFormer** | gates-sampling | 0.5037 | 0.425 / 0.428 | 0.425 / 0.426 | 0.779 / 0.777 | **0.003** |
| **BEVDet** | extract-then-place | 0.5166 | 0.610 / 0.610 | 0.360 / 0.364 | 0.512 / 0.521 | **0.009** |
| **BEVDepth** | extract-then-place | 0.5354 | 0.648 / 0.652 | 0.325 / 0.328 | 0.485 / 0.492 | **0.007** |

→ **3개 모델 모두 subset ≈ full-val (max Δ ≤ 0.009)**. 768-subset all-cam 숫자는 full-val을
신뢰성 있게 대표한다(앞서 BEVFormer Δ≤0.004 확인을 BEVDet/BEVDepth로 확장).

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

## 상태/남은 것
- 완료: BEVFormer, BEVDet, BEVDepth full-val all-cam.
- 미완: **CAPE, DETR3D** full-val 미실행(subset만; subset 대표성 입증됐으니 subset 사용 가능). **PETRv2** 미평가. **DFA3D** oracle 학습 중(bus 진행).
- frame-2× fix 반영본(VR/CR), geobev 이미지 무관.

*숫자 출처: 위 3개 summary.txt. subset 값은 results/PAPER_ANALYSIS_KR.md / BENCHMARK_SUMMARY.md.*
