# §5 Analysis — 한글 v4 (사용자 최종 개고, 신규 5.3 검증 완료)

> 본문 수치 표기: headline mVRS와 1/7 성분은 퍼센트, all-camera 분해는 0에서 1 비율.
> 그룹 명칭: 투영 샘플링(projection sampling) / 추출 후 배치(extract then place).
> SimpleBEV는 논문에서 완전 제외. VP 수치: BEVDet, BEVDepth, CAPE는 full(3792프레임,
> 631셀), BEVFormer, DETR3D와 segmentation은 768프레임 subset(full 진행 중).

## 5. Analysis

세 가지 질문을 따라 분석한다. 시점 강건성이 모델 간에 어떻게 다른가(§5.1), 플랫폼 전이는 어떻게 다른가(§5.2), 두 축이 서로를 예측하는가(§5.3). 셋을 관통하는 관찰은 하나다. 카메라 기하 강건성은 기존 view-transformation paradigm, 즉 forward, backward, sparse라는 라벨만으로 설명되지 않는다. 시점 강건성은 extrinsic이 feature sampling을 직접 결정하는지에 따라 갈리고, 플랫폼 전이는 표현이 source rig의 prior에 얼마나 묶여 있는지에 따라 갈린다. 두 task가 같은 지문을 보이며, 이 현상은 detection에서 가장 선명하다.

### 5.1. Viewpoint Robustness

먼저 per-camera perturbation과 all-camera perturbation은 서로 다른 실패 regime을 측정한다. 단일 카메라만 회전하는 per-camera 조건에서는 나머지 다섯 뷰가 정상으로 남아 있어 multi-view redundancy가 손상을 상당 부분 흡수한다. 반대로 여섯 카메라가 함께 움직이는 all-camera 조건에서는 전체 rig의 기하가 동시에 바뀌므로, 모델의 view transformation 구조가 직접 드러난다. Headline mVRS는 여섯 개 per-camera 모드와 하나의 all-camera 모드를 평균하므로 배치 수준 요약으로는 유용하지만, 구조적 실패 모드를 읽기에는 all-camera 분해가 더 진단적이다.

**IMG에서는 순위가 압축되고, CAL에서 재배열된다.** 원래 extrinsic을 유지한 채 시점만 다시 렌더링한 IMG 조건에서는 모델 간 차이가 크게 압축된다. Detection 모델들은 모두 82.8--83.9 mVRS에 모이고, segmentation 모델들도 79.6--84.5 범위에 머문다. 즉 IMG 조건의 순위는 forward, backward, sparse라는 기존 paradigm을 거의 따르지 않는다. 반면 바뀐 기하에 맞는 extrinsic을 함께 공급하는 CAL 조건에서는 순위가 크게 재배열된다. BEVFormer는 83.1에서 93.5로, DETR3D는 84.2에서 95.6으로, PointBeV는 79.4에서 92.3으로 EXT 수준을 넘어 회복한다. 반대로 CAPE는 94.0에서 88.6으로, BEVDet은 89.2에서 87.5로, BEVDepth는 90.2에서 87.4로, LSS는 88.4에서 86.3으로 오히려 EXT보다 낮아진다.

이 차이는 paradigm의 속성이 아니라 extrinsic을 소비하는 위치의 차이로 설명된다. 3D query나 BEV point를 extrinsic으로 이미지에 투영한 뒤 해당 좌표에서 feature를 읽는 모델을 **투영 샘플링(projection sampling)** 모델이라 부르자. BEVFormer, DETR3D, PointBeV가 여기에 해당한다. 이들은 올바른 extrinsic이 주어지면 샘플링 좌표가 다시 맞춰지므로 CAL에서 회복한다. 반대로 backbone이 먼저 image feature를 추출한 뒤, depth splat이나 positional encoding으로 기하를 나중에 적용하는 모델을 **추출 후 배치(extract then place)** 모델이라 부르자. LSS 계열, CVT, LaRa, CAPE가 여기에 해당한다. 이 경우 기울어진 viewpoint가 이미 image feature 안에 반영된 뒤라서, 뒤늦게 올바른 extrinsic을 넣어도 feature 자체의 오염을 되돌리기 어렵다. 결국 CAL--EXT의 부호는 모델이 extrinsic을 feature sampling 전에 쓰는지 이후에 쓰는지를 드러내는 진단 신호이며, 분석 대상 10개 모델 중 9개에서 이 구분과 일치한다. 유일한 예외는 GaussianLSS로, 추출 후 배치 계열임에도 mVRS 기준 CAL이 EXT보다 높다(92.1 대 90.7). 다만 all-camera 성분에서는 0.699 대 0.698로 사실상 경계선에 놓인다.

**EXT와 IMG의 차이는 같은 분기를 다른 방향에서 보여준다.** Fig. A에서 mVRS의 IMG 점수를 EXT 점수에 대해 그리면, 투영 샘플링 모델들은 대각선 근처에 놓여 EXT와 IMG가 거의 같은 손상을 준다. 반면 추출 후 배치 모델들은 대각선 아래에 놓이며, IMG가 EXT보다 대략 6--10%p 낮다. 즉 깨끗한 이미지를 둔 채 extrinsic만 틀리는 경우보다, 실제로 viewpoint가 바뀐 이미지를 보는 경우가 더 치명적이다. 이 gap은 uniform offset이 아니다. 모델들은 대각선 근처에 머무는 그룹과 크게 아래로 떨어지는 그룹으로 나뉘며, 중간 영역은 거의 비어 있다. 따라서 extrinsic-only perturbation은 단순히 점수를 조금 높게 만드는 것이 아니라, 어떤 모델이 강건해 보이는지 자체를 바꿀 수 있다. 이는 extrinsic-only 평가가 실제 시점 변화에서의 성능 저하를 과소보고한다는 본 벤치마크의 첫 번째 주장을 메커니즘 수준에서 정밀화한다.

**축별 분해는 yaw와 pitch/roll이 서로 다른 종류의 오류임을 보여준다.** Yaw는 주로 calibration-type error처럼 행동한다. EXT와 IMG의 차이가 작고, 올바른 extrinsic을 넣는 CAL에서 대부분 회복되기 때문이다. 반면 pitch와 roll은 content-type error에 가깝다. 실제 viewpoint가 바뀌면서 지면, 하늘, 객체 크기와 위치 관계가 image feature 자체에 반영되므로, 뒤늦게 올바른 extrinsic을 넣어도 손상이 완전히 복구되지 않는다. 이 차이는 detection에서 가장 선명하다. EXT 조건에서 추출 후 배치 검출기는 yaw에 가장 약한 반면, 투영 샘플링 검출기는 pitch에 가장 약하다. CAL에서는 yaw가 대부분 90% 후반까지 회복되지만, pitch는 투영 샘플링 검출기에서만 크게 회복되고 추출 후 배치 검출기는 80% 안팎에 머문다. 따라서 detection에서는 CAL-pitch가 가장 깨끗한 메커니즘 판별자다.

카메라별 분해는 단일 카메라 교란이 모델 구조보다 rig의 중복도를 더 강하게 반영함을 보여준다. Per-camera IMG 점수는 대부분 80% 이상에 머물며, all-camera 교란에서 보이는 큰 구조적 차이보다 훨씬 작다. 또한 5개 검출기 모두에서 중요도 순서는 거의 같다. 후방 카메라가 가장 치명적이고, 전방 카메라가 그 다음이며, 좌우 측면 카메라는 상대적으로 덜 중요하다. 이 순서는 특정 아키텍처의 성질이라기보다, 카메라 시야 중첩과 장면 내 객체 분포에서 나온다. 따라서 per-camera robustness는 모델의 기하 처리 방식보다 rig redundancy를 더 많이 측정한다. Headline mVRS는 여섯 개 per-camera perturbation과 하나의 all-camera perturbation을 평균하므로, per-camera의 높은 점수가 모델 간 차이를 압축한다. 구조적 차이를 읽으려면 headline mVRS보다 all-camera 분해를 함께 봐야 한다.

**크기별 all-camera 응답은 앞선 메커니즘 해석을 더 선명하게 만든다.** CAL 조건에서 작은 교란을 거의 흡수하는 안전 구간은 투영 샘플링 모델에만 나타난다. 4도 교란에서 BEVFormer와 DETR3D는 CAL 성능을 약 0.9 수준으로 유지하지만, BEVDet, BEVDepth, CAPE는 이미 절반 가까이 하락한다. 특히 depth-splat 기반 모델은 pitch가 커질수록 빠르게 무너져, 올바른 extrinsic만으로는 기울어진 image feature를 복구할 수 없음을 보여준다. 반면 IMG 조건의 손상은 검출 소실이라기보다 위치 오류에 가깝다. 4도 all-camera IMG에서도 느슨한 4m 기준으로는 60% 이상의 물체가 여전히 매칭되지만, 엄격한 0.5m 기준 AP는 정상 대비 20% 이하로 떨어지고 살아남은 검출의 translation error가 1.6배 이상 증가한다. 즉 시점 변화는 물체를 안 보이게 만드는 것이 아니라, BEV 공간에서 잘못된 위치에 보이게 만든다.

### 5.2. Cross-Platform Transfer

**IMG 조건에서 가장 큰 차이는 depth 의존성에서 나온다.** BEVDet과 BEVDepth 같은 depth 기반 forward detector는 bus에서 거의 붕괴한다(CTS 0.2와 0.1). 반면 depth를 직접 예측하지 않는 detector들은 훨씬 더 잘 전이된다. 이는 sedan 마운트에서 학습된 monocular depth prior가 높아진 bus 시점에서 잘못된 거리를 예측하고, image feature를 잘못된 BEV 위치로 옮기기 때문이다. 이 해석과 일관되게, SUV IMG에서 depth 기반 모델의 mASE는 0.63--0.73으로, depth-free 모델의 0.31--0.36보다 훨씬 크다. Bus에서는 검출 생존 자체가 거의 없어 mASE가 1.0으로 포화된다. CTS 기준으로는 CAPE가 bus IMG에서 가장 높지만, absolute SDS는 DETR3D와 거의 같으므로 이 순위는 oracle normalization의 영향을 함께 받는다.

CAL 조건은 다시 모델의 기하 사용 방식을 가른다. 투영 샘플링 모델은 올바른 타깃 extrinsic을 큰 전이 이득으로 바꾸지만, depth 기반 detector는 CAL에서 부분적으로만 회복한다. Bus에서 BEVDet은 IMG 0.2에서 CAL 22.8로, BEVDepth는 0.1에서 16.6으로 크게 오르지만, 여전히 BEVFormer와 DETR3D의 CAL 성능인 40.1과 37.6에는 한참 못 미친다. 이는 병목이 타깃 extrinsic의 유무만이 아니라, depth 기반 lifting에 학습된 source-rig depth prior에도 있음을 시사한다.

같은 분리는 EXT와 CAL의 부호에서도 반복된다. Depth 기반 detector는 올바른 타깃 extrinsic을 받아도 CAL이 EXT보다 낮게 남고, 같은 extract-then-place 계열인 CAPE도 같은 부호를 보인다. 반대로 투영 샘플링 모델은 CAL이 EXT를 넘어선다. Segmentation까지 포함하면 이 부호 분리는 SUV와 bus 양쪽에서 모든 분석 대상 모델에 대해 성립하며, VP에서 유일한 예외였던 GaussianLSS도 cross-platform에서는 이 규칙을 따른다. 즉 §5.1의 CAL--EXT 진단은 cross-platform transfer에서 더 깨끗하게 반복되며, extrinsic-only 평가는 이 스케일에서도 강건성을 과대평가한다.

Segmentation에서는 CVT가 가장 뚜렷한 반대 방향의 CAL 응답을 보인다. CVT는 bus에서 올바른 extrinsic을 받을 때 오히려 성능이 크게 떨어진다. 우리는 이를 source-rig-specific geometry embedding이 학습 분포에서 먼 target geometry를 더 강한 out-of-distribution 입력으로 만들기 때문이라고 해석한다. 다만 이는 embedding 분포를 직접 측정한 결과가 아니라 구조에 근거한 가설이다. 따라서 cross-platform 배치에서는 calibration을 정확히 아는 것만으로 충분하지 않으며, 모델이 extrinsic을 소비하는 방식이 더 나은 calibration이 도움이 되는지조차 좌우할 수 있다.

CTS 해석에는 한 가지 주의가 필요하다. CTS는 각 모델의 target-platform oracle로 나눈 비율이므로, oracle이 낮은 모델에는 상대적으로 유리할 수 있다. Detection의 bus oracle은 모델 간 최대 1.6배 차이가 난다. 예를 들어 CAPE와 DETR3D는 bus IMG에서 absolute SDS가 거의 같다(0.162 대 0.159). 그러나 CAPE의 bus oracle이 DETR3D보다 낮기 때문에(0.449 대 0.602), CTS는 36.0 대 26.5로 벌어진다. 따라서 depth 사용 여부에 따른 큰 그룹 차이는 견고하지만, 그룹 안의 미세한 순위는 absolute transfer performance와 함께 읽어야 한다.

**플랫폼별로 보면 bus 전이는 대체로 SUV보다 어렵다.** Detection에서는 BEVFormer와 DETR3D도 SUV보다 bus에서 더 크게 떨어지고, depth 기반 detector는 bus IMG에서 거의 붕괴한다. Segmentation에서도 IMG 성능 범위가 bus에서 더 낮아지며, 특히 CVT는 bus CAL에서 올바른 extrinsic을 받았을 때 오히려 크게 무너진다. 이는 두 target platform이 같은 종류의 shift가 아니라 서로 다른 강도의 재구성을 나타내기 때문이다. SUV는 상대적으로 온건한 layout 변화에 가깝지만, bus는 카메라 높이와 viewpoint regime까지 바꾸는 더 강한 shift다. 그 결과 sedan에서 학습된 depth prior가 더 크게 깨지고, geometry embedding도 학습 분포에서 더 멀어질 수 있으며, 실제로 관측되는 객체 분포도 달라진다. Table 1에서 sedan에서 bus로 갈 때 class-wise visibility가 9.5--19.4%p 이동하는 것이 이를 보여준다. 따라서 cross-platform robustness는 하나의 평균으로만 읽기보다, SUV와 bus를 분리해 해석해야 한다.

### 5.3. Relating Viewpoint Robustness and Cross-Platform Transfer

Fig. C는 viewpoint robustness와 cross-platform transfer가 언제 정렬되고 언제 분리되는지를 보여준다. 핵심은 headline mVRS가 아니라 all-camera IMG 성분이다. All-camera perturbation은 전체 rig가 함께 바뀌는 조건이므로, source platform 안에서 측정 가능한 가장 가까운 cross-platform proxy다. 실제로 SUV transfer에서는 all-camera IMG 순위가 CTS IMG 순위와 강하게 정렬된다. Detection에서는 all-camera IMG 순위가 SUV CTS 순위와 정확히 일치하고(Spearman ρ=1.0), segmentation에서도 높은 정렬을 보인다(ρ=0.90). 반면 per-camera 성분은 단일 카메라 오류를 나머지 뷰가 흡수하는 regime을 측정한다. detection에서는 per-camera 점수와 transfer의 순위 상관이 오히려 음수이고(ρ=-0.62), segmentation에서는 순위가 정렬되어 보이지만 per-camera 점수의 변별 폭이 0.03으로 all-camera의 0.16에 비해 5분의 1 수준이라 신호 자체가 약하다. headline mVRS는 이 per-camera를 6 대 1로 더 많이 포함하기 때문에 transfer를 예측하는 all-camera 신호를 희석한다.

이 정렬은 re-rendered viewpoint shift를 측정할 때만 나타난다. 같은 all-camera perturbation도 EXT 조건으로 평가하면, 즉 이미지는 그대로 두고 extrinsic만 바꾸면, transfer와의 관계가 약해지거나 detection에서는 오히려 뒤집힌다. Detection의 SUV transfer에서 all-camera IMG와 CTS의 순위 상관은 ρ=1.0이지만, all-camera EXT로 바꾸면 ρ=-0.70으로 반전된다. EXT는 추출 후 배치 모델의 강건성을 과대평가하고, 실제 transfer에서 강한 투영 샘플링 모델은 상대적으로 약하게 보이게 만든다. 따라서 cross-platform deployability를 예측하려면 extrinsic-only score가 아니라 re-rendered all-camera IMG score를 봐야 한다.

그럼에도 all-camera VP와 CTS가 완전히 같은 축은 아니다. CAPE는 VP CAL에서는 약하지만 bus IMG CTS 기준으로는 가장 높고, BEVFormer와 DETR3D는 VP CAL을 지배하지만 cross-platform IMG 순위에서는 CAPE에 최상위를 내준다. Segmentation에서도 PointBeV는 VP CAL에서는 강하지만 bus IMG에서는 약하다. 또한 NORMAL 성능도 geometry robustness를 안정적으로 예측하지 못한다. Segmentation에서는 NORMAL 순위와 CTS IMG 순위의 상관이 음수이며(ρ=-0.50), NORMAL 성능이 가장 낮은 CVT가 SUV와 bus 평균 CTS IMG에서는 가장 높은 전이 성능을 보인다. 즉 all-camera VP는 transfer의 유용한 early indicator이지만, 플랫폼 전이는 source-rig depth prior, geometry embedding, target-platform visibility까지 함께 반영한다. RoboGeo가 mVRS와 CTS를 하나의 점수로 합치지 않고 분리해 보고하는 이유다.

---

## 부기 (작업 기록, 논문 비수록)

0. 영어화 지침(확정).
   (a) 단정 강도. 근거가 넓은 주장(진단 규칙 10개 중 9개, depth 그룹 분리)은 단정 유지.
   단독 사례 기반 인과 설명(CVT의 embedding 역전)은 "we attribute this to"와
   "consistent with"로 한 단계 낮춤(본문 반영 완료). 확정 영어 문장:
   "In our results, the sign of the CAL-EXT gap largely tracks where the model consumes
   extrinsics." / "Extract-then-place models lie below the diagonal." /
   "This under-reporting is concentrated in extract-then-place models and is much weaker
   for projection-sampling models." / "Bus transfer is generally harder than SUV
   transfer, with the gap most visible in detection and in the CAL reversal of CVT." /
   "CAPE is top-ranked by normalized bus-IMG CTS; its absolute SDS is close to DETR3D,
   and the CTS gap partly reflects CAPE's lower bus oracle." / "We attribute this
   reversal to source-rig-specific geometry embeddings, although this remains an
   architectural hypothesis rather than a directly measured quantity."
   (b) 신조어. 그룹명은 5.1 첫 등장에서 한 문장으로 정의하고 이후 일관 사용. 영어 명칭
   확정: **projection-sampling** / **extract-then-place**(하이픈 형용사형).
   (c) 문체. RoboBEV의 장식적 어휘는 따르지 않고 구조만 차용(주장, 수치, 메커니즘,
   귀결 한 문장 순).
   (d) 수치 인용. 현재 방식 유지(상대 변화 + 모델명).
1. **본문에 판단 편입한 채굴 finding 3건**: (i) 크기별 knee와 CAL 안전 구간(5.1(4)
   전반부), (ii) IMG 손상의 위치 오류 성격(5.1(4) 후반부 — "recall 붕괴" 서사를
   정밀화), (iii) CTS의 EXT 대 CAL 역전(5.2(1) — 5.1의 결론이 cross-platform에서도
   성립함을 보임, 부록 표 Z 필요). **부록 배정**: LSS pitch 부호 비대칭(4.8에서
   6.6배), per-class 순서(pedestrian 최악, bicycle 최강건 — 크기 분할 아님), TP
   분해(translation 지배, mAOE clamp 아티팩트), CAL-yaw 전 크기 회복 상세.
2. 본문 수치는 Table 2의 1/7 mVRS로 통일. all-camera 분해는 부록 표/Fig. A 보조 버전
   권장(격차 최대 2배 증폭). 5.1(4)의 크기별 수치는 all-camera 기준임을 본문에 명시
   (부록 그림 Y 필요).
3. Fig. A는 1/7 스케일(72에서 98) SimpleBEV 제외 버전으로 재생성 완료. Fig. C 캡션 명시
   사항: VP IMG 순위는 all-camera 기준(1/7 IMG는 detection에서 1.2포인트로 압축),
   CTS IMG는 suv와 bus의 평균(bus 단독이면 LaRa가 최고).
4. 5.3의 원래 outline "same ranking"은 데이터, 서론의 different patterns 주장 모두와
   상충하여 "부분 정렬 + 축별 분리 + NORMAL 무관"으로 작성.
5. **SimpleBEV 완전 제외(2026-06-10 확정, 팩트체크 완료).** 원 논문 Simple-BEV는
   learned embedding 없는 전형적 투영 샘플링이 맞으나(이전 "embedding 기반" 분류는
   오류, 본문 제거됨), 우리가 받은 구현의 행동은 충실한 샘플러와 불일치: **CAL 곡선이
   IMG와 전 축 전 크기에서 0.01 이내 동일**(올바른 extrinsic의 이득 0), **EXT-yaw 전
   크기 0.90 이상 평탄**(yaw extrinsic 무반영). 동일 하니스의 PointBeV는 정반대(EXT-yaw
   0.12 급락, CAL-yaw 0.95 회복). 즉 규칙의 반례가 아니라 구현 아티팩트. 최종 확정은
   bevunify 측 소스 확인 필요(이 머신에 없음). 제외에 따른 통계: seg NORMAL-CTS 상관
   -0.50, seg VP-CTS 0.70, Fig. A 1/7 상관 0.30, 규칙 10개 중 9개. 원자료는
   seg_vp_cts.tsv와 eval_results/에 내부 보존.
6. 미완. DFA3D(bus oracle 학습 중, 완료 후 full VP/CTS), DSPE, PETRv2(Table 2 부재),
   Table 3 baseline(Ext.Aug, EAFormer, PD-BEV).
6b. **Table 2의 dash row 처리(리뷰 반영).** 미완 row가 많아 보이지 않도록 최종 표에서는
   dash row를 빼는 쪽이 깔끔하다. DSPE는 표에서 제거 권장(§4 목록에는 유지 가능 여부
   별도 결정). DFA3D는 bus oracle 완료 직후 full VP/CTS를 돌려 row를 채우는 것이
   1순위이고, 제출 시점까지 미완이면 그때 행 제거.
7. **Table 2 정정 필수.** BEVDet CTS 4셀은 oracle 나눗셈 누락(raw NDS x100). 정정값
   suv 17.0/16.1, bus 0.2/22.8. 본문, Fig. B, Fig. C 반영 완료. 논문 표 수정 필요.
   §4의 "seven methods"(seg)는 "six"로 수정하되, 이는 **BEVFormer(seg)를 목록에서 빼는
   편집**이다 — §4 목록에 SimpleBEV는 원래 없고(Table 2에만 있음) BEVFormer(seg)는
   Table 2에 없으므로, 양쪽을 LSS, GaussianLSS, CVT, LaRa, DSPE, PointBeV 6개로
   일치시킨다. SimpleBEV 행 삭제는 Table 2 쪽 편집.
7b. CTS가 1을 넘는 관측은 본문 제외, 부록 후보. 정의: CTS 드라이버의 NORMAL 조건
   (sedan 이미지 + sedan extrinsic 입력, 즉 원본 sedan_infos_val; GT는 타깃 플랫폼의
   visibility 기준으로 재계산된 평가 집합 — eval_cts_det.py 머리주석 참조)에서 CAPE
   sedan 모델의 6-class NDS가 0.4842로, bus oracle(bus 학습, bus 입력) 0.4490을 초과해
   비율 1.078이 된다. full bus val(3792프레임), 출처
   results/CAPE/cts/cts_cape_sedan_summary.txt.
8. **full-val 전환 진행.** BEVDet, BEVDepth, CAPE는 full(VP 631셀+CTS) 완료로 본문
   수치 교체 완료. BEVFormer per-cam full(2-shard)과 DETR3D full(all-cam 후 per-cam
   체인) 실행 중, 완료 시 두 모델 수치 교체 + Table 2의 dagger 제거. segmentation
   full은 bevunify 측 재계산 필요.
9. 부록 자산 목록: 표 X(per-camera 분해, vp_percam_peraxis.tsv), 표 Z(CTS EXT 조건,
   results/{model}/cts), 그림 Y(크기별 all-camera 곡선) = fig_magnitude_curves.png
   생성 완료, all-camera breakdown 표, per-class/TP 분해(채굴 finding), GaussianLSS
   경계선 수치.
10. 생성된 figure 현황(figures/, 스크립트 make_section5_figs.py와 _figs2.py):
   Fig. A fig_vp_ext_img_correlation.png(EXT-IMG 산점, 1/7), Fig. B
   fig_cts_img_to_cal.png(CTS IMG에서 CAL 덤벨), Fig. C fig_ranking_bump.png(순위 변화),
   **Fig. D fig_cal_ext_sign.png(CAL-EXT 부호 진단, VP와 CTS 2패널 — 5.1(1)과 5.2(1)의
   핵심 그림, 본문 메인 후보)**, Fig. Y fig_magnitude_curves.png(크기 곡선, 부록).
   부호 일관성: VP 9/10(GaussianLSS 예외), CTS suv 10/10, CTS bus 10/10.

11. **v4(사용자 개고) 신규 5.3 주장 검증 완료(2026-06-10).** 재계산 결과 전부 일치:
   det all-camera IMG 대 SUV CTS 순위상관 +1.00, all-camera EXT 대 SUV CTS -0.70,
   per-camera IMG 대 SUV CTS -0.62(약하거나 음수 주장 부합), seg all-camera IMG 대
   SUV CTS +0.90. 참고: bus는 +0.60(본문이 SUV로 한정한 것이 적절). **중요한 의존성:
   ρ=1.00은 BEVDet CTS 정정값(suv IMG 17.0) 기준에서만 성립**(구값 9.1이면 0.90) —
   Table 2 정정의 추가 근거. Fig. C는 현 bump chart가 새 5.3 framing(SUV 정렬, EXT
   반전, per-cam 희석)을 직접 보여주지 못하므로 보강 필요: 상관 막대 패널(allcam-IMG,
   percam-IMG, allcam-EXT 대 SUV CTS) 추가 또는 교체 권장.
   → **신규 Fig. C 생성 완료**: fig_vp_cts_alignment.png(make_section5_figs3.py).
   (a) allcam-IMG 대 SUV CTS 산점(det +1.00, seg +0.90), (b) 예측자별 ρ 막대(det
   +1.00/-0.62/-0.70, seg +0.90/+0.90/+0.60). 주의: seg per-camera ρ는 +0.90으로
   양수라 "약하거나 음수" 일반화가 깨짐 → 본문을 "det 음수, seg는 변별 폭 0.03으로
   신호 약함"으로 정정(2026-06-10). 기존 bump chart(fig_ranking_bump.png)는 부록행.
   seg allcam-EXT ρ=+0.60: 본문의 "약해지거나(seg) 뒤집힌다(det)" 표현과 부합.
