# GeoBEV 분석 섹션 — 한글 초안 (3D Object Detection, Table 2 Task 2 해석)

> 논문 용어(mVRS, mVRS_IMG, CTS, SDS, NORMAL/EXT/IMG/CAL, per-camera/all-camera, P_NORMAL,
> P_TARGET)에 맞춤. **헤드라인 mVRS(1/7)는 효과를 압축하므로 분석은 all-camera 수치로 전개**한다.
> PETRv2는 미평가, 숫자는 frame-2×-fix 반영본. (BEVDet CTS는 표 vs results/ 불일치 — §확인 필요.)

## 5.1 두 프로토콜은 서로 다른 강건성 축을 측정한다

GeoBEV의 VP(viewpoint perturbation)와 CTS(cross-platform transfer)는 카메라-기하 강건성의 **서로
독립적인 두 축**을 분리한다. VP는 동일 차량(sedan)에서 카메라 기하만 교란하여 **view-transform이
extrinsic과 어떻게 결합하는가**를, CTS는 sedan 모델을 suv/bus에 배치하여 **모델의 depth/scale prior가
배치 시점(viewpoint)과 어떻게 결합하는가**를 측정한다. 아래에서 보이듯 한 축에서 강건한 모델이 다른
축에서도 강건한 것은 아니며(§5.5), 따라서 두 점수를 합치지 않고 분리 보고하는 것이 타당하다.

## 5.2 강건성을 가르는 축은 view-transform "패러다임"이 아니라 메커니즘이다

모델 코드를 직접 분석하면, 강건성 패턴은 **Forward/Backward/Sparse 분류로도, explicit depth 유무로도
설명되지 않으며**, 오직 **카메라 extrinsic이 feature "샘플링"을 게이트하는지 여부**로 설명된다.

- **extrinsic-gates-sampling** (BEVFormer, DETR3D): backbone은 extrinsic과 **무관하게** 2D feature를
  추출하고, extrinsic(lidar2img)이 3D query/reference point를 이미지에 투영하여 **그 위치에서 feature를
  샘플링**한다(BEVFormer의 deformable cross-attention, DETR3D의 grid_sample). 즉 extrinsic이 *어디를
  볼지*를 결정한다.
- **extract-then-place** (BEVDet/BEVDepth의 LSS, CAPE의 camera-view PE): backbone이 extrinsic과
  **무관하게** feature를 추출한 뒤, 기하는 *나중에* position embedding(CAPE) 또는 depth-splat(LSS)으로만
  적용된다. extrinsic은 이미 추출된 feature를 *배치*할 뿐이다.

이 축은 **Sparse 클래스를 가로지른다**(DETR3D=gates-sampling, CAPE/PETRv2=extract-then-place)는 점이
핵심이며, **depth와도 독립**이다 — CAPE는 depth network가 없음에도(camera-view PE) LSS depth 모델과
동일한 패턴을 보인다. 따라서 본 분석의 분류 기준은 paradigm/depth가 아니라 이 **메커니즘**이다.

## 5.3 Viewpoint robustness 분석

**(a) EXT는 IMG 저하를 충실히 반영하지 못한다 — 단, extract-then-place 모델에 한해서다.**
all-camera mVRS로 보면(Table A) 메커니즘에 따라 EXT와 IMG의 관계가 정반대로 갈린다.

- **gates-sampling: EXT ≈ IMG** (BEVFormer 0.428≈0.426, DETR3D 0.438≈0.422). extrinsic이 샘플링을
  게이트하므로, extrinsic을 틀리든(EXT) 그 위에서 샘플하는 이미지를 틀리든(IMG) 동일하게 손상된다.
- **extract-then-place: EXT ≫ IMG** (CAPE 0.811≫0.407, BEVDepth 0.652≫0.328, BEVDet 0.610≫0.364).
  EXT에서는 *깨끗한* 이미지에서 feature를 정상 추출하고 배치만 틀어지지만, IMG에서는 *기운 이미지가
  추출 자체를 오염*시킨다.

기존 연구가 가정한 "extrinsic-only(EXT) 평가가 viewpoint 저하를 반영한다"는 명제는 따라서 **extract-then-place
모델에서는 거짓**(EXT가 강건성을 과대평가)이고, gates-sampling 모델에서만 성립한다. 이것이 본 벤치마크의
첫 관찰(extrinsic-only는 불충분)을 *메커니즘 수준에서* 정밀화한 결과다. 특히 CAPE는 camera-view PE가
설계상 extrinsic 의존을 줄이도록 만들어졌고, 그 효과가 **EXT all-cam 0.811(전 모델 중 최고 강건)**로
직접 드러난다.

> **주의(표 구성).** 헤드라인 mVRS는 per-camera 6개 + all-camera 1개를 동일 가중(1/7)하는데, per-camera
> 점수가 모든 모델에서 ~0.9로 거의 동일해 **mVRS(1/7)에서는 EXT–IMG 격차가 압축**된다(예: BEVDet EXT
> 89.1 vs IMG 83.3). 위 효과는 **all-camera 점수에서만 극적으로 드러나므로**(BEVDet 0.610 vs 0.364, CAPE
> 0.811 vs 0.407), 본문 분석은 all-camera를 기준으로 전개하고 mVRS(1/7)는 단일 요약 점수로만 둔다.
> [권장: Table 2에 all-camera 컬럼을 추가하거나 별도 breakdown 표/그림을 둔다.]

**Table A. all-camera mVRS (axis 평균), 메커니즘별.**

| 모델 | 메커니즘 | EXT | IMG | CAL |
|---|---|---|---|---|
| CAPE | extract-then-place | 0.811 | 0.407 | 0.560 |
| BEVDepth | extract-then-place | 0.652 | 0.328 | 0.492 |
| BEVDet | extract-then-place | 0.610 | 0.364 | 0.521 |
| DETR3D | gates-sampling | 0.438 | 0.422 | 0.845 |
| BEVFormer | gates-sampling | 0.428 | 0.426 | 0.777 |

**(b) CAL(일관 교란)에서의 회복도 메커니즘을 따른다 — CAL-pitch가 가장 깨끗한 판별자다.**
이미지와 extrinsic을 *일관되게* 교란하면, gates-sampling은 투영을 end-to-end로 재정렬해 가장 어려운
축까지 회복하지만(CAL-pitch: BEVFormer 0.661, DETR3D 0.825), extract-then-place는 회복하지 못한다
(CAL-pitch: BEVDepth 0.132, CAPE 0.182, BEVDet 0.254) — 일관된 extrinsic이라도 *이미 추출에 박힌
이미지 기울임*은 되돌릴 수 없기 때문이다. 모든 모델이 CAL-yaw는 회복(≈0.92–0.98)하는데, yaw는 수직축
회전이라 지면 외형을 보존하므로 추출이 살아남기 때문이다 — 즉 **지평선을 기울이는 축(roll/pitch)만이
extract-then-place를 일관성에서도 무너뜨린다**. (부수적으로, extract-then-place 내부에서는 depth 감독이
강할수록 더 pitch-locked 된다: BEVDepth(explicit) 0.132 < BEVDet(implicit) 0.254.)

**(c) metric 수준에서 VP의 분기는 결국 recall(검출) 이야기다.**
SDS = ½·mAP + ½·(1−TP오차)를 mAP(recall)와 TP 품질로 분해하면 실패 양상이 분명해진다(Table B). **기운
이미지(IMG)는 아키텍처와 무관하게 recall을 붕괴**시키고(IMG mAP-retention 0.14–0.19), **EXT에서만
메커니즘이 recall 수준에서 갈린다** — extract-then-place는 recall을 *유지*(mAP 0.40–0.48; 깨끗한 이미지는
여전히 검출되고 배치만 틀림)하고, gates-sampling은 *상실*(mAP 0.15; 오샘플링으로 box가 제안조차 안 됨).
이것이 EXT≫IMG의 근본 원인이다. CAL에서는 gates-sampling이 recall을 회복(mAP 0.67–0.75), extract-then-place는
부분 회복(0.37–0.41)에 그친다.

**Table B. VP all-camera, mVRS(SDS-RRS) / mAP-retention.**

| 모델 | 메커니즘 | EXT | IMG | CAL |
|---|---|---|---|---|
| BEVDepth | extract-then-place | 0.65 / 0.48 | 0.33 / 0.14 | 0.49 / 0.37 |
| BEVDet | extract-then-place | 0.61 / 0.40 | 0.36 / 0.16 | 0.52 / 0.38 |
| CAPE | extract-then-place | — | 0.41 / 0.16 | 0.56 / 0.41 |
| DETR3D | gates-sampling | 0.44 / 0.15 | 0.42 / 0.15 | 0.85 / 0.75 |
| BEVFormer | gates-sampling | 0.43 / 0.19 | 0.43 / 0.19 | 0.78 / 0.67 |

**(d) per-camera 강건성은 아키텍처 불변이다(부록).** 단일 카메라 교란에서는 모든 모델이 ≤10% 저하에
그치고, *어느 카메라가 중요한지*의 순서도 아키텍처와 무관하게 동일하다(후방 카메라가 가장 결정적, 우측
카메라가 가장 덜 중요) — 6-view 융합 redundancy와 장면 객체 분포의 산물이다. 따라서 per-camera를 무겁게
섞은 집계(mVRS 1/7 포함)는 아키텍처 차이를 가린다.

## 5.4 Cross-platform transfer 분석

**CTS는 VP와 다른 축, 즉 depth 일반화로 지배된다.** 명시적/범주형 depth를 쓰는 모델은 **타깃-시점 이미지
(IMG)에서 가장 크게 붕괴**한다(BEVDet/BEVDepth bus-IMG ≈ 0.001–0.002): sedan 마운트로 학습된 monocular
depth가 높은 bus 시점에 대해 틀린 깊이를 예측해, feature가 잘못된 BEV 거리에 lift된다. 반면 depth가 없는
모델은 훨씬 잘 전이된다 — **CAPE가 최강 전이체(IMG 0.349)이며 플랫폼-강건**(bus 0.360 ≈ suv 0.338),
DETR3D/BEVFormer는 중간이다. 전이 실패는 depth 고유의 TP 지문도 남긴다: 생존 검출 중에서도 depth 모델은
추가로 *크기*가 틀린다(CTS-IMG mASE 0.81–0.87 vs depth-free 0.31–0.34).
*(BEVDet CTS 수치는 Table 2와 results/가 약 2× 차이 — 둘 중 어느 run이 최종본인지 확정 필요.)*

## 5.5 종합 — 두 축은 독립이다

두 프로토콜은 검출기를 **서로 독립적인 두 아키텍처 성질**로 분리한다: VP는 *메커니즘*(gates-sampling vs
extract-then-place)으로, CTS는 *depth 의존*(depth vs depth-free)으로 가른다. 이 둘은 같은 분할이 아니다
(Figure: mechanism × depth의 2×2). 네 칸 중 셋이 채워진다.

| | depth-free | uses depth |
|---|---|---|
| **gates-sampling** | BEVFormer, DETR3D | — |
| **extract-then-place** | **CAPE** | BEVDet, BEVDepth |

**CAPE가 결정적 증거다**: *extract-then-place*이므로 VP에서는 LSS depth 모델과 동일한 지문(EXT≫IMG,
pitch-locked CAL)을 보이지만, *depth-free*이므로 cross-platform 전이는 최고다. 따라서 한 모델의 VP 거동은
CTS 거동을 예측하지 못하며, 그 역도 성립한다. 실무적으로: **self-calibration drift(CAL)에는 gates-sampling
(BEVFormer/DETR3D)이, 순수 extrinsic 오차에는 extract-then-place(특히 CAPE)가, cross-platform 배치에는
depth-free 모델이** 가장 안전하며 — **세 가지를 모두 이기는 현행 검출기는 없다.**

---
*수치 출처: `results/{model}/{vp,cts}/`, per-cam/per-axis는 `results/vp_percam_peraxis.tsv`, 메커니즘
근거는 `results/VP_CROSS_MODEL_ANALYSIS.md`. VP=768 subset(현재; full-val all-cam 재계산 진행 중),
CTS=full 3792. PETRv2는 학습/평가 대기.*
