# RoboGeo가 처방하는 강건 모델 설계 — v2 (전 아키텍처 코드 검증판)

> 목표 지표(사용자 확정): **VP-IMG**(재렌더링 시점 변화 + 낡은 extrinsic, 모델은 모름)
> 와 **CTS-CAL**(타깃 플랫폼 이미지 + 올바른 타깃 extrinsic 제공). EXT는 분석용.
> 제약: **학습에는 normal 이미지만 사용**(재렌더링/멀티 rig 증강 금지; 2D 워핑은 허용).
> 근거: 10개 아키텍처 forward 경로 코드 분석(8-agent 서베이, 2026-06-11, file:line
> 인용 원본 = `results/_arch_geometry_survey.json`) + 벤치마크 측정값(§5 v7 검증치).
> PointBeV/LaRa/SimpleBEV는 소스가 이 머신에 없어 행동 데이터로만 포함.
> 용어는 §5와 동일(투영 샘플링/추출 후 배치, 그 외 신규 용어 없음).

---

## A. 코드 검증: 각 모델이 카메라 기하를 소비하는 지점

| 모델 | extrinsic 진입점 (해석적) | depth | 학습된 기하/rig prior (코드 확인) | 추론 시 calib 변경에 반응하는 것 |
|---|---|---|---|---|
| **BEVFormer** | `point_sampling`의 lidar2img matmul 단 한 곳 (encoder.py:88-149) | 없음 (pillar 4점 + 학습 attention이 암묵 해소) | `cams_embeds` nn.Parameter(6,256) — 카메라 슬롯 고정 (transformer.py:74-75,170-171) | 샘플링 좌표 + bev_mask(쿼리↔카메라 배정) 전부 |
| **DETR3D** | `feature_sampling`의 lidar2img → grid_sample (detr3d_transformer.py:398-417) | 없음 (학습 3D 앵커 반복 정제) | 카메라-인덱스별 attention Linear(:280) + 학습 3D 쿼리 앵커 분포 | 샘플링 좌표 + frustum 가시성 마스크 |
| **CAPE** | 쿼리측 world=R@(ref+t) 변환 (cape_transformer.py:429-432) | 없음 (LID 64빈은 PE 입력일 뿐) | **QcR/V_R = Sigmoid(Linear(R.flatten 9))** — 회전행렬을 1층 게이트에 직접 입력(:19-42) + `camera_embedding` nn.Embedding(6,512) 슬롯 고정(:287); **key측 PE는 intrinsic-only → extrinsic 오차가 이미지측에서 관측 불가** | 쿼리측 기하항 + 학습 R-게이트 출력만; key/feature는 불변 |
| **PETRv2** | coords3d = inv(lidar2img)@frustum → **학습 MLP** → key PE (petrv2_dnhead.py:378-388) | 없음 (LID 64빈 = PE 입력) | position_encoder MLP가 학습 rig의 ray 좌표 분포를 내장; 슬롯-순서 sine PE; out-of-range mask 폐기(:542) | key PE 값만 (attention 재가중); 샘플링/배치 없음 |
| **BEVDet** | `get_lidar_coor`→bev_pool splat 좌표 (view_transformer.py:174-176) | 예측 59빈 metric, **splat 가중** | **BN1d(27)+MLP+SE 게이트에 raw calib 27차원 입력**(:534-538,645-652) — 단일 rig라 학습 분산 0 | splat 좌표(해석적 ✓) + depth/context 게이트(학습, OOD ✗) |
| **BEVDepth** | `get_geometry` (base_lss_fpn.py:424-459) | 예측 112빈 metric, DPT 지도, **splat 가중** | 동일 BN(27)+MLP — **검증: 학습 rig는 6캠 전부 높이 1.6m·동일 intrinsic·프레임간 분산 0 → 새 calib에서 19–784σ 활성 → 게이트 포화 → depth softmax 퇴화** | splat 좌표(✓) + depth 분포 자체가 망가짐(✗) |
| **LSS** | `get_geometry` 순수 행렬 (models.py:166-186) | 예측 41빈 metric, 이미지-only, splat 가중 | **없음** (전부 해석적; depth conv 가중치에 rig prior 암묵 내장) | splat 좌표 전부; depth/feature 불변 |
| **GaussianLSS** | lidar2img 역투영 단 한 곳 (GaussianLSS.py:411) → 3D 가우시안 splat | 예측 64빈 분포 → **평균+공분산(soft splat)**, 이미지-only | **없음** + CARLA 변형엔 **PitchBinClassifier/BEVScorer: pitch 후보 11개(−10..+10°)를 해석적으로 적용해 BEV를 스코어링·argmax하는 자기보정 모듈 실존**(calibration.py:157-211) | 가우시안 평균 이동·공분산 회전 전부; feature/depth/opacity 불변 |
| **CVT** | 없음 — 기하는 전부 PE로: `cam_embed(E_inv 중심)`, `img_embed(ray)` 학습 Conv (encoder.py:216-262) | 없음 (방향만; 깊이는 attention이 암묵) | **cam_embed가 학습 중 정확히 6개의 입력 벡터만 봄** → 새 extrinsic(올바른 것 포함)이 OOD; BEV 그리드 2D라 높이/pitch는 학습 상호작용에 전적으로 흡수 | PE 값만; "올바른 calib가 올바른 attention을 보장한다는 구조적 근거 없음" |
| **DFA3D** | BEVFormer와 동일 + 투영점의 정규화 depth 좌표 (encoder.py:450-466) | 예측 112빈(24코드 압축), 이미지-only, **weight-path 게이트**: depth_score가 attention 가중에 곱해짐 | cams_embeds 승계; **cam_channel(=calib 인코더)은 의도적으로 비활성** — 고정 참조 텐서로 정규화하는 그 경로가 바로 안티패턴 | 샘플링 좌표 + depth 게이트 평가 위치 + bev_mask |

## B. 측정값 대응 (목표 지표만)

| 모델 | VP IMG all-cam | CTS CAL suv / bus | 한 줄 진단 |
|---|---|---|---|
| BEVFormer | 0.425 | 71.9 / 40.1 | 해석적 샘플링 = CAL 회복 원천 |
| DETR3D | 0.424 | **76.9** / 37.6 | + 고해상도(1600×900)가 bus 흡수 |
| PointBeV(seg) | 0.289 | 68.4 / 15.6 | 투영 샘플링도 bus regime엔 불충분 |
| CAPE | 0.400 | 34.3 / 32.9 | 학습 PE/게이트는 재배선 안 됨 |
| BEVDet | 0.360 | 16.1 / 22.8 | splat+조건부 퇴화 이중고 |
| BEVDepth | 0.325 | 5.7 / 16.6 | 〃 (가장 심함) |
| LSS(seg) | — | 24.2 / 17.1 | 해석적이지만 depth prior가 한계 |
| GaussianLSS(seg) | seg IMG 최고 84.5(mVRS) | 40.6 / 36.4 | soft splat 완충 + 자기보정 선례 |
| CVT(seg) | — | 30.4 / **1.8** | 학습 기하 embedding = CAL 역전 |
| DFA3D | 평가 예정 | 평가 예정 | 사분면 통제 실험 |

## C. 코드 서베이가 새로 준 발견 3개

**C-1. "조건부의 퇴화" — 조건 입력은 학습 분포 안에서 분산이 있어야 한다.**
BEVDepth/BEVDet은 depth net에 calibration을 *조건 입력으로 받는데도* 전이에 실패한다.
코드 검증 결과 이유가 정확히 나왔다: 27차원 raw calib 벡터가 BatchNorm1d+MLP를
지나는데, 단일 rig 학습에서 이 입력의 분산이 사실상 0(6캠 모두 높이 1.6m, intrinsic
동일, 프레임간 이동 분산 0)이므로, 새 calibration이 들어오면 BN 출력이 19–784σ로
폭주해 SE 게이트가 포화되고 depth 분포 자체가 퇴화한다. → **설계 규칙: rig 수준
벡터(높이·전역 R)는 학습 모듈의 입력으로 금지. 조건이 필요하면 단일 rig 안에서도
분산이 존재하는 좌표 — per-pixel ray 방향 — 로만 조건화한다.** (픽셀마다 ray가
다르므로 normal-only 학습에서도 일반화 가능한 조건부가 된다.)

**C-2. "숨은 rig prior"는 해석적 모델에도 있다 — 전수 목록.**
BEVFormer/DFA3D `cams_embeds`(슬롯 고정 6×256), DETR3D 카메라-인덱스 attention
Linear + 학습 3D 앵커, CAPE `camera_embedding`(슬롯 6×512)+QcR, PETRv2 슬롯-순서
sine PE, CVT `cam_embed`. 전부 "카메라 슬롯 인덱스"에 묶인 학습 파라미터로, 새
calibration에 무반응(또는 OOD)이다. → **설계 규칙: 슬롯-인덱스 학습 파라미터 전면
제거. 카메라 구분이 필요하면 현재 calibration에서 해석적으로 계산되는 양(ray
방향장)으로 대체.**

**C-3. 자기보정은 거의 공짜다 — 그리고 선례 코드가 이미 있다.**
조사한 10개 모델 전부에서 backbone feature는 extrinsic-독립이다(기하는 그 뒤에서
적용). 따라서 feature를 1회 추출해 캐시하면, extrinsic 후보를 바꿔가며 BEV
인코더만 재실행하는 탐색이 저비용으로 가능하다. GaussianLSS CARLA 변형의
`PitchBinClassifier/BEVScorer`(calibration.py:157-211)가 정확히 이 구조의 실존
구현이다: pitch 후보 11개(−10°..+10°, 2° 간격)를 해석적으로 적용해 만든 BEV들을
작은 CNN으로 스코어링해 argmax. → 우리 설계 C4의 코드 출발점.

## D. 설계 — 다섯 부품 (각 부품의 코드 출처 명시)

**C1 정준화 입력** — 현재 extrinsic의 회전 성분으로 매 뷰를 중력 정렬 가상 카메라로
homography 워핑(학습·추론 동일 적용). 카메라 중심 회전은 장면 무관 정확한
homography이므로 렌더링이 아니다(normal-only 제약 합치). CTS-CAL에선 타깃 rig의
pitch/roll 차이가 backbone 입력 단계에서 해석적으로 제거된다.

**C2 기하-프리 backbone + 고해상도 + 슬롯 prior 제거** — 전 모델 공통으로 backbone은
이미 기하-프리(코드 확인). 추가로 C-2의 숨은 prior를 제거: cams_embeds류 삭제,
필요 시 ray-방향장(현재 calib에서 계산)으로 대체. 해상도는 DETR3D의 bus 증거
(oracle 0.602, bus IMG CTS 26.5 최고)에 따라 예산 내 최대(1600×900).

**C3 extrinsic-게이트 샘플링 + ray-조건 depth의 weight-path 게이트** —
골격은 BEVFormer `point_sampling`(encoder.py:88-149; lidar2img가 유일한 기하
진입점이라는 성질을 보존). depth는 DFA3D처럼 **값/배치 경로가 아니라 가중치
경로**에 곱한다(multi_scale_3ddeformable_attn_function.py:286-300 — 투영 일관성
게이트로 작동). 단 DFA3D의 이미지-only metric depth head 대신: (i) 지면 위 높이로
파라미터화(미터 스케일은 extrinsic이 해석적으로 공급), (ii) per-pixel ray 방향을
조건 입력으로(C-1 규칙 — 분산 있는 조건만). BN-on-calib, 고정 참조 정규화
(DFA3D가 비활성화한 cam_channel 경로) 금지.

**C4 자기보정(Δextrinsic) — VP-IMG의 승부처** — 2단 구성.
(i) *coarse 스윕*: GaussianLSS BEVScorer를 일반화 — backbone feature 캐시 후
Δ(pitch/roll/yaw) 후보 격자를 해석적으로 적용해 BEV 인코더만 재실행, 학습된
스코어러로 argmax(C-3 근거: 저비용).
(ii) *fine 회귀*: normal 이미지의 homography 워프(회전 교란의 정확한 합성)로
학습한 ΔR 회귀 헤드 + 뷰 중첩 영역 교차뷰 일관성 자기지도. 반복 2회 적용.
IMG에선 회전 성분을 회수해 근사-CAL로(측정 헤드룸 all-cam 0.425→0.779, yaw 우선
— CAL-yaw ~0.95), CAL에선 Δ≈0을 예측해 무해. CAPE 교훈의 보완: key측이
extrinsic-free면 오차가 이미지측에서 관측 불가하므로, 스코어링은 반드시 **BEV
공간(투영 결과)** 에서 한다(BEVScorer 방식이 정확히 이것).

**C5 지면 기준 BEV refinement + 뷰 일관성 게이팅** — IMG 손상은 소실이 아닌 위치
오류(0.5m AP <20%, mATE ×1.6)이므로 지면 anchored 위치 보정 헤드. 뷰 게이팅은
BEVFormer가 이미 보유한 메커니즘(bev_mask 기반 hit-count 정규화,
spatial_cross_attention.py:169-172)을 일관성 점수로 확장.

## E. 안티패턴 (코드 인용, "하지 말 것")

1. **raw calib 벡터 → BN/MLP/게이트**: BEVDepth/BEVDet mlp_input(27)
   (base_lss_fpn.py:187-255 / view_transformer.py:534-652), CAPE QcR/V_R
   (cape_transformer.py:19-42). 단일 rig 학습에서 분산 0 → 퇴화(C-1).
2. **카메라 슬롯-인덱스 학습 파라미터**: cams_embeds, camera_embedding,
   카메라별 attention Linear, 슬롯 sine PE(C-2 목록).
3. **기하의 전부를 학습 PE로**: CVT encoder.py:216-262 — 올바른 새 calib조차 OOD
   입력이 됨(bus CAL 17.7→1.8의 코드 측 근거).
4. **이미지-only metric depth + splat 배치**: LSS/BEVDet/BEVDepth — rig prior가
   conv 가중치에 암묵 내장, CAL이 placement만 고치고 radial 오차는 못 고침.
5. **extrinsic-jitter-only 증강**(이미지 불변): 모델에게 extrinsic 불신을 학습시켜
   CAL 응답을 평탄화(CAPE형 퇴화) — Table 3 Ext.Aug가 직접 검정.

## F. 구현 계획 (base = BEVFormer_DFA3D 포크, 이미 우리가 운영)

diff 목록(작은 것부터): (1) cams_embeds 제거/ray-장 대체 [C2], (2) depth head를
ray-조건·지면높이 파라미터로 교체 [C3], (3) 입력 정준화 transform 추가(파이프라인
1개) [C1], (4) BEVScorer류 coarse 스윕 + homography-학습 ΔR 헤드 [C4], (5) 지면
refinement 헤드 [C5]. **Ablation → 벤치마크 셀 매핑**: (1)(2)는 CTS-CAL
suv/bus로, (3)은 CTS-CAL bus + VP-IMG pitch/roll로, (4)는 VP-IMG 전 축(특히
yaw)으로, (5)는 IMG의 mATE/0.5m AP로 각각 분리 측정 가능 — 벤치마크의 조건
분해가 곧 ablation 프로토콜이 된다(논문 Discussion 연결 문장).

기대치(정직): CTS-CAL은 C3만으로 현 최고선(suv ~77, bus ~40) 보장 + C1·C2가 bus
상향. VP-IMG는 C4가 회수하는 회전 성분만큼 0.425→0.779 구간 회수(yaw 거의 전부,
pitch/roll 부분). **잔여 한계** = bus 높이 시차·큰 pitch/roll의 content 성분 —
PointBeV(bus CAL 15.6)가 증명하는, normal-only 제약의 구조적 대가이며 재렌더링
데이터셋의 존재 의의로 역활용.

---

### 부기
- v2 작성 2026-06-11. 근거: 8-agent 코드 서베이(BEVFormer/DETR3D/CAPE/PETRv2/
  BEVDet/BEVDepth/GaussianLSS/CVT/LSS/DFA3D, file:line 인용 원본
  `_arch_geometry_survey.json`) + §5 v7 검증 수치. v1(G1–G4 골자)은 git 이력 보존.
- 논문 활용: C-1(조건부 퇴화)·C-3(자기보정 공짜 조건)은 Discussion 소재로도 강력
  — 특히 C-1은 "BEVDepth가 calibration-aware인데 왜 전이에 실패하는가"라는 예상
  리뷰어 질문의 정답. 영어 확정 문장 후보: "Calibration-awareness does not imply
  calibration-generalization: when the conditioning input has near-zero variance
  under a single training rig, the learned gates saturate on any novel calibration
  (19–784 sigma activations in BEVDepth's BN(27))."
- DFA3D full 결과가 들어오면 C3의 weight-path 게이트 예측(CAL 회복 유지 + IMG
  정밀도 향상)을 본 문서에 검증 기록으로 추가할 것.
