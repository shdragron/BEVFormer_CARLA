# RoboGeo가 처방하는 강건 모델 설계 (논문 제안 초안, 한글 v1)

> 용도: Discussion(§6 또는 §5 말미) 압축 문단 + 부록 확장본의 원고.
> 용어는 §5 v7과 동일(투영 샘플링/추출 후 배치, EXT/IMG/CAL, per-camera/all-camera,
> mVRS/CTS). 목표 조건은 사용자 확정대로 **VP IMG와 CTS CAL** 두 가지(EXT는 분석용).
> 톤 규칙 동일: 관찰(수치)은 단정, 설계 효과 예측은 "시사한다/기대할 수 있다".

---

## 1. 프레임: 진단 축을 거꾸로 읽으면 설계 지침이 된다

§5의 결과는 어떤 모델이 강건한지의 순위표가 아니라, 실패가 **어디서** 생기는지의
분해다. 그 분해를 거꾸로 읽으면 각 실패 지점에 대응하는 설계 지침이 나온다. 즉
RoboGeo는 평가 벤치마크인 동시에 설계 처방을 주는 진단 도구다. 아래 네 지침(G1--G4)은
모두 §5의 측정에 일대일로 대응한다.

## 2. 설계 지침 (각각 근거 수치와 함께)

**G1. extrinsic이 feature sampling을 직접 결정하게 하라 (투영 샘플링 골격).**
근거: CTS CAL에서 메커니즘 격차가 가장 크다 — bus CAL 40.1/37.6(BEVFormer/DETR3D)
대 22.8/16.6(BEVDet/BEVDepth), SUV CAL 71.9/76.9 대 16.1/5.7. CAL−EXT 부호 진단은
VP 9/10, CTS SUV/bus 10/10. 올바른 calibration이 주어졌을 때 회복하는 능력은 사후
보정으로 얻을 수 없는 구조적 성질이다.

**G2. depth는 rig-불변으로 파라미터화하고, feature를 옮기는 수단이 아니라 샘플링
가중치로 써라.** 근거: depth 기반 detector의 bus IMG 붕괴(0.2/0.1)와 CAL 부분
회복(22.8/16.6 — 투영 샘플링의 40.1/37.6에 못 미침)은 병목이 extrinsic이 아니라
source rig 높이에 학습된 monocular metric depth prior임을 보여준다(SUV IMG에서
depth 기반 mASE 0.63--0.73 대 depth-free 0.31--0.36). 처방: (i) 픽셀→미터 매핑을
직접 학습하지 말고 지면 위 높이나 ray-조건부 분포로 파라미터화해 미터 스케일은
extrinsic이 해석적으로 공급하게 하고, (ii) depth를 splat 배치가 아니라 투영 샘플링
ray 위의 가중 분포로만 사용한다. (ii)는 정확히 DFA3D 사분면(샘플링×depth)이며,
우리 벤치마크의 통제 실험이 이 처방을 직접 검증한다 — 예측: CAL 회복은 유지되고
depth의 위치 정밀도만 추가된다.

**G3. rig 전용 학습 geometry embedding을 금지하라.** 근거: CVT의 bus CAL
역전(17.7→1.8) — 기하를 학습 파라미터에 구우면 올바른 새 calibration 자체가
out-of-distribution 입력이 된다. 기하 처리는 파라미터 없는 해석적 투영/샘플링으로.

**G4. VP IMG는 자기보정(self-calibration)으로 근사-CAL로 변환하라.** 근거: IMG는
"진짜 이미지 + 낡은 extrinsic" 조건이므로, 모델이 이미지에서 Δextrinsic을 추정하면
IMG가 CAL로 바뀐다. 그 헤드룸을 우리가 정량화했다 — BEVFormer all-camera IMG 0.425
대 CAL 0.779(약 1.8배). 축 분석이 우선순위도 준다: yaw는 calibration-type이라 알기만
하면 거의 전부 회복되고(CAL-yaw ~0.95), pitch/roll도 투영 샘플링은 CAL에서 회복되므로
추정 정확도가 곧 성능이다. 핵심 결합 조건: 자기보정은 **G1 골격 위에서만 의미가
있다** — 추출 후 배치 위에서는 추정에 성공해도 CAL이 듣지 않는다.

부수 지침 두 개. (a) **정준화**: 카메라 중심 회전은 장면과 무관하게 정확히
homography이므로, extrinsic의 회전 성분으로 모든 뷰를 중력 정렬 가상 카메라로
워핑해 backbone에 넣으면 pitch/roll의 content-type 성분 일부를 해석적으로
calibration-type으로 되돌릴 수 있다(렌더링 불필요; 높이 차이의 시차만 잔여).
(b) **위치 refinement**: IMG 손상은 검출 소실이 아니라 위치 오류이므로(0.5m AP
정상 대비 20% 이하, translation error 1.6배 이상), 지면 기준 BEV refinement 헤드가
직접적인 마지막 부품이 된다.

**함정 경고 (벤치마크가 예측하는 부작용).** 이미지를 그대로 두고 extrinsic만 흔드는
증강(extrinsic-jitter-only)은 이 목적함수에서 역효과가 예상된다 — 모델에게
extrinsic을 불신하라고 가르쳐 CAL 응답을 평평하게 만들고(CAPE형 응답), CTS CAL을
이기게 해주는 바로 그 메커니즘을 약화시킨다. Table 3의 Ext.Aug baseline이 이 예측의
직접 검정이다.

## 3. 종합 스케치 (가칭, 한 문단)

G1--G4를 합치면: **BEVFormer류 deformable 투영 샘플링 골격 + ray-조건부 depth 분포
가중(DFA3D식, metric prior 없음) + 중력 정렬 가상 카메라 정준화 입력 + homography
증강으로 학습한 Δextrinsic 추정 헤드 + 지면 기준 BEV 위치 refinement.** 기대 효과를
우리 수치로 말하면, CTS CAL은 골격 선택만으로 bus 16.6→37.6급 점프가 측정돼 있고,
VP IMG는 자기보정이 성공하는 만큼 0.425→0.779 헤드룸을 회수한다. 잔여(큰 pitch/roll의
시차 성분, bus 높이 regime의 content shift)는 재렌더링/멀티 rig 데이터만이 가르칠 수
있는 몫이며, 이것이 곧 본 데이터셋의 존재 이유다.

## 4. 검증 경로 (이미 우리 파이프라인에 있는 것)

- **G2 ↔ DFA3D**(bus oracle 완료, full VP/CTS 평가 예정): 샘플링×depth 사분면의
  통제 실험. CAL 회복 유지 + IMG 정밀도 향상이 나오면 G2 채택, CAL 회복이 깨지면
  depth 가중조차 위험하다는 더 강한 결론.
- **G4 데이터 측 ↔ Table 3**(Ext.Aug / PD-BEV / EAFormer): extrinsic-jitter 증강의
  CAL 평탄화 예측(함정 경고)과 domain-generalization 계열의 CTS 이득을 분리 측정.
- **자기보정 모듈 자체는 future work**로 명시(우리가 구현하지 않음) — 단 헤드룸
  (0.425→0.779)과 회수 순서(yaw 우선)는 본 벤치마크 수치로 이미 정량화되어 있음을
  강조.

## 5. 논문 배치 제안

- **본문(Discussion, 6--8문장)**: G1--G4를 각 한 문장 + 함정 경고 한 문장 + "DFA3D와
  Table 3이 부분 검증" 한 문장. 주어는 RoboGeo/the benchmark("RoboGeo의 측정은 ...을
  시사한다").
- **부록(반 페이지)**: 본 문서 2--3절 수준의 근거 수치 표(지침 ↔ 측정 대응표).
- 영어 확정 문장 후보:
  "Read in reverse, the diagnostic axes of RoboGeo prescribe a design recipe."
  / "Correct-calibration recovery is an architectural property that post-hoc
  correction cannot supply (G1)." / "Parameterize depth in a rig-invariant way and
  use it to weight sampling rather than to place features (G2); this is exactly the
  quadrant our DFA3D case study instantiates." / "Self-calibration converts IMG into
  approximate CAL; our measurements bound the headroom at 0.425 to 0.779 in
  all-camera RRS (G4)." / "The benchmark also predicts a failure mode: extrinsic-only
  jitter augmentation teaches the model to distrust extrinsics and flattens the CAL
  response."

---

### 부기
- 작성 2026-06-11. §5 v7.1 확정 직후, 사용자 요청 "우리 논문에 robust한 모델 아이디어
  제시"에 대응. 목표 조건은 사용자 확정(EXT 제외, VP IMG + CTS CAL).
- 모든 인용 수치는 §5 v7/Table 2 검증값과 동일 출처(2026-06-10/11 전수 검증).
- n=10 상관 관찰 기반 처방임을 본문에서 한 문장으로 명시할 것(과주장 방지) — DFA3D와
  Table 3 결과가 들어오면 "부분 검증됨"으로 승격.
