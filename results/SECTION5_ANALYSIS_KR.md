# §5 Analysis — 한글 v2 (1/7 mVRS 기준 서술, outline 순서)

> 본문 수치는 Table 2의 headline mVRS(%)를 기본으로 사용한다. 세부 분해(all-camera,
> per-axis 1/7)는 해당 문단에서 출처를 명시한다. 그룹 명칭은 "투영 샘플링(projection
> sampling)"과 "추출 후 배치(extract then place)"로 통일한다.

---

## 5. Analysis

### 5.1. Viewpoint Robustness

**(1) Paradigm별 분석. IMG에서는 순위가 압축되고, CAL에서 재배열된다.**

원래 extrinsic을 유지한 채 시점만 다시 렌더링한 IMG 조건에서는 모든 모델이 paradigm과
무관하게 비슷한 수준으로 떨어진다. detection의 mVRS는 83.1에서 84.3 사이(BEVDepth 83.1,
BEVFormer 83.8, CAPE 84.3), segmentation은 78.1에서 84.5 사이에 몰려, IMG 순위는
forward, backward, sparse 어느 라벨도 따르지 않는다. 반면 바뀐 기하의 올바른 extrinsic을
함께 공급하는 CAL 조건에서는 순위가 크게 재배열된다. BEVFormer는 83.1에서 93.5로,
DETR3D는 84.2에서 95.6으로, PointBeV는 79.4에서 92.3으로 EXT 수준을 넘어 회복하는 반면,
CAPE는 94.3에서 89.0으로, BEVDet은 89.1에서 87.8로, BEVDepth는 90.5에서 87.8로, LSS는
88.4에서 86.3으로 오히려 EXT보다 낮아진다.

**이 회복은 paradigm의 속성이 아니다.** 회복하는 모델은 backward인 BEVFormer와 sparse인
DETR3D, PointBeV에 걸쳐 있고, 회복하지 못하는 모델은 forward 전부와 sparse인 CAPE,
backward인 CVT, LaRa에 걸쳐 있다. 가르는 것은 extrinsic을 소비하는 방식이다. 3D query나
BEV 점을 extrinsic으로 이미지에 투영해 그 좌표에서 feature를 읽는 방식을 **투영
샘플링(projection sampling)**, backbone이 extrinsic과 무관하게 feature를 먼저 추출하고
기하를 depth splat(LSS 계열)이나 positional encoding(CVT, LaRa, CAPE)으로 나중에
적용하는 방식을 **추출 후 배치(extract then place)** 라 부른다. 투영 샘플링 모델은
올바른 extrinsic이 주어지면 투영이 다시 정렬되어 회복하고, 추출 후 배치 모델은 기울어진
이미지가 이미 추출된 feature에 박혀 있어 올바른 extrinsic으로도 되돌릴 수 없다. 정리하면,
CAL 점수가 EXT 점수보다 높으면 투영 샘플링 모델이고 낮으면 추출 후 배치 모델이며, 이
규칙은 Table 2에 수치가 있는 11개 모델 중 10개에서 성립한다. 유일한 예외는 GaussianLSS로,
추출 후 배치 계열임에도 mVRS 기준 92.1 대 90.7로 CAL이 EXT보다 높다. 다만 all-camera
성분에서는 0.699 대 0.698로 사실상 경계선에 있다(부록). SimpleBEV는 규칙 자체는
만족하지만(74.5 대 75.9) 모든 일관 교란에서 비정상적으로 무너져 별도 검증 대상으로 둔다.

**(2) EXT와 IMG의 상관 (Fig. A).**

mVRS의 IMG 점수를 EXT 점수에 대해 그리면(Fig. A, 가로축 EXT, 세로축 IMG) 같은 분기가
metadata 쪽에서도 드러난다. 투영 샘플링 모델은 대각선에 붙어 있다. BEVFormer는 83.1 대
83.8, DETR3D는 84.2 대 83.9, PointBeV는 79.4 대 79.6으로 EXT와 IMG가 사실상 같다.
샘플링에 쓰는 extrinsic을 오염시키는 것과 이미지를 오염시키는 것이 동등하게 치명적이기
때문이다. 추출 후 배치 모델은 전부 대각선 아래에 놓인다. CAPE는 94.3 대 84.3, BEVDepth는
90.5 대 83.1, LaRa는 91.1 대 82.5로, EXT가 IMG보다 6에서 10포인트 높다. 깨끗한 이미지는
extrinsic 오류에서 살아남지만 기울어진 이미지는 추출 자체를 오염시키기 때문이다.
SimpleBEV는 IMG가 EXT보다 높아(78.1 대 75.9) 유일하게 대각선보다 위쪽에 찍히며,
그림에서는 이상치로 회색 표기한다(캡션에 명시). 따라서 extrinsic만 교란하는 평가는
실제 시점 변화에서의 저하를 과소보고하며, 이는 정확히 추출 후 배치 모델에서 발생하고 투영
샘플링 모델에서는 발생하지 않는다. 본 벤치마크의 첫 번째 주장을 메커니즘 수준에서
정밀화한 결과다. 한편 이 격차는 all-camera 성분에서 최대 2배까지 벌어진다(부록).
headline mVRS는 per-camera 점수가 모든 모델에서 0.9 부근으로 비슷해 격차를 압축하기
때문이며, 그 이유는 아래 카메라 분석에서 설명한다.

**(3) 축과 카메라 분석.**

축 분석. 1/7 기준 per-axis 점수로 보면, IMG에서 가장 파괴적인 축은 task에 따라 다르다.
detection은 pitch가 최악이고(79에서 81, 전 검출기) segmentation은 yaw가 최악이다(77에서
79, 6개 모델 일관). 일관된 all-camera yaw는 BEV lifting이 의존하는 카메라 간 대응을
깨뜨리고, pitch는 detection의 거리와 크기 추정이 의존하는 지면을 기울인다. EXT에서는 최악
축 자체가 메커니즘의 지문이다. 추출 후 배치 검출기는 yaw에서 가장 무너지고(BEVDet과
BEVDepth 83, 나머지 축은 91에서 96) 투영 샘플링 검출기는 pitch에서 가장 무너진다
(BEVFormer 78, DETR3D 81). 전자는 멀쩡한 feature를 잘못 배치할 뿐이어서 BEV 방위를 가장
크게 트는 yaw가 치명적이고, 후자는 틀린 extrinsic이 이미지의 잘못된 행을 샘플링하므로
하늘과 지면을 오가는 pitch가 치명적이다. CAL에서는 두 task 모두 yaw를 거의 완전히
회복한다(detection 97에서 99, segmentation 92에서 98, SimpleBEV 제외). 일관된 yaw는
수직축에 대한 전역 회전이라 지면 외형을 보존하기 때문이다. 반면 pitch는 투영 샘플링
검출기만 회복한다(BEVFormer 90, DETR3D 95 대 추출 후 배치 79에서 81). 따라서 CAL의
pitch 점수가 detection에서 가장 깨끗한 메커니즘 판별자다.

카메라 분석. 단일 카메라 교란은 비교적 무해하고 그 양상은 아키텍처와 무관하다.
per-camera IMG 점수는 모든 검출기에서 81에서 98 사이에 머물고, 카메라 중요도 순서는
5개 검출기에서 완전히 동일하다. 후방 카메라가 가장 결정적이고(81에서 83) 전방
카메라가 그 다음이며(85에서 89) 우측 카메라들이 가장 덜 중요하다(95에서 98).
이 순서는 모델이 아니라 rig의 시야 중첩과 장면의 객체 분포에서 나온다. 여섯 개 뷰의
융합이 오염된 한 뷰를 그냥 눌러버리는 것이다. 결국 per-camera 강건성은 아키텍처가 아니라
rig의 중복도를 측정하며, headline mVRS가 per-camera를 6 대 1로 가중하기 때문에 위에서 본
압축이 생긴다. 모델 간 차이를 보려면 all-camera 성분을 읽어야 한다.

### 5.2. Cross-Platform Transfer

**(1) Paradigm별 분석. depth 의존성이 전이를 결정하고, CAL 응답이 메커니즘을 다시
가른다 (Fig. B).**

타깃 시점 이미지로 평가하는 IMG 조건에서 forward 검출기, 즉 depth 기반 모델은 bus에서
붕괴한다(BEVDet과 BEVDepth 모두 CTS 0.1). sedan 마운트로 학습된 monocular depth가 높아진
bus 시점에서 잘못된 거리를 예측해 feature가 잘못된 BEV 거리로 옮겨지기 때문이다. 이
실패는 살아남은 검출에도 흔적을 남긴다. suv IMG에서 depth 모델의 크기 오차 mASE는
0.63에서 0.73으로 depth를 쓰지 않는 모델의 0.31에서 0.36보다 두 배 가까이 크고, bus에서는
생존 자체가 거의 없어 mASE가 1.0으로 포화된다. depth를 쓰지 않는 검출기는 훨씬 잘
전이되며, 시점 교란에서는 가장 전형적인 추출 후 배치 모델이었던 CAPE가 bus IMG 36.0으로
최강 전이체가 된다. 올바른 타깃 extrinsic을 공급하는 CAL은 시점 실험과 정확히 같은
방식으로 메커니즘을 가른다(Fig. B). 투영 샘플링 모델은 올바른 extrinsic을 큰 전이 이득으로
바꾼다. suv에서 BEVFormer는 37.2에서 71.9로, DETR3D는 34.9에서 76.9로, PointBeV는
20.2에서 68.4로 오른다. depth에 묶인 검출기는 올바른 extrinsic을 받아도 붕괴 상태에
머문다(bus CAL 8.5와 16.6). 실패하는 것은 projection이 아니라 depth prior이기 때문이다.
가장 눈에 띄는 결과는 embedding 기반 segmentation 모델 두 개에서 올바른 extrinsic이
오히려 해롭다는 점이다. CVT는 bus에서 17.7에서 1.8로, SimpleBEV는 12.2에서 3.4로
떨어진다. 우리는 이를 학습된 geometry embedding이 source rig에 맞춰져 있어, 학습
분포에서 먼 진짜 타깃 기하가 낡은 sedan extrinsic보다 인코딩을 더 멀리 분포 밖으로
밀어내는 것으로 해석한다. 이 해석과 일관되게, 역전은 기하 변화가 가장 큰 bus에서만
나타난다. 정리하면 calibration을 정확히 아는 것만으로는 cross-platform 배치에 충분하지
않으며, 모델이 extrinsic을 소비하는 방식이 더 나은 calibration이 도움이 되는지조차
좌우할 수 있다.

측정상의 주의. CTS는 각 모델 자신의 플랫폼 일치 oracle로 정규화하는데, 이 oracle이
검출기 간 최대 1.6배 차이가 난다. 그래서 ratio는 oracle이 약한 모델에 유리할 수 있다.
CAPE와 DETR3D는 bus IMG의 절대 전이 성능이 사실상 같지만(SDS 0.162와 0.159) CAPE의 bus
oracle이 낮아서(0.449 대 0.602) CTS는 36.0 대 26.5로 갈린다. depth 사용 여부에 따른 그룹
분리는 절대값으로 0.001 대 0.10에서 0.16이라 이 문제와 무관하게 견고하지만, 그룹 안의
미세한 순위는 절대 전이 성능으로 읽어야 한다. Table 2가 각 CTS 옆에 oracle을 병기하는
이유다.

**(2) 플랫폼 분석. bus는 suv보다 질적으로 어렵다.**

모든 모델이 두 task와 모든 조건에서 suv보다 bus 전이에서 더 많이 잃는다. detection
IMG에서 BEVFormer는 37.2에서 18.0으로, DETR3D는 34.9에서 26.5로 떨어지고 depth
검출기들은 9에서 10 수준이 0.1로 무너진다. segmentation IMG는 suv에서 11.8에서 44.3
범위였던 것이 bus에서 11.4에서 27.7로 줄고, bus CAL은 CVT와 SimpleBEV에서 1.8과 3.4로
붕괴한다. suv는 비슷한 높이에 layout만 조금 다른 온건한 재장착인 반면, bus는 카메라 높이
레짐 자체를 바꾼다. 그 결과 첫째로 sedan에서 학습된 depth prior가 가장 세게 깨지고,
둘째로 앞의 해석을 따르면 geometry embedding이 학습 분포에서 가장 멀리 밀려나며(실제로
CAL 역전은 bus에서만 나타난다), 셋째로 보이는 것 자체가 바뀐다. sedan에서 bus로 가면 클래스별 visibility가
9.5에서 19.4포인트 이동하는데(Table 1) 이 변화는 플랫폼별 oracle이 흡수하지만 전이된
모델은 스스로 감당해야 한다. 따라서 cross-platform 강건성은 플랫폼별로 읽어야 한다.
suv는 layout 변화에 대한 내성을, bus는 시점 레짐 변화에 대한 내성을 측정한다.

### 5.3. VP와 CTS의 상관

Fig. C는 평가축이 NORMAL에서 VP IMG, VP CAL, CTS IMG로 바뀔 때 각 모델의 순위 변화를
보여준다. 여기서 CTS IMG는 suv와 bus의 평균이다(Fig. C 캡션에도 명시). 순위는 두
프로토콜이 같은 병목을 공유하는 곳에서는 부분적으로 정렬되고, 지배 성질이 다른 곳에서는
분리된다. IMG 조건끼리는 시점 순위와 전이 순위가 segmentation에서
강하게, detection에서 중간 정도로 상관된다(순위 상관 0.83과 0.50). 둘 다 그 지점에서는
같은 이미지 외형 변화가 지배하기 때문이다. 그러나 정렬은 앞서 밝힌 두 축을 따라 정확히
깨진다. CAPE는 일관된 시점 변화의 흡수에서는 하위권이지만(CAL 89.0, 추출 후 배치) depth를
쓰지 않아 전이에서는 최강이고(bus IMG 36.0), 반대로 투영 샘플링인 BEVFormer와 DETR3D는
CAL을 지배하지만(93.5와 95.6) CTS IMG에서는 중위권이다. segmentation에서는 PointBeV가
같은 패턴을 반복한다(CAL 92.3으로 최상위권이나 bus IMG는 11.4로 최하위권). 그리고 어느
축도 표준 리더보드로 예측되지 않는다. NORMAL 순위와의 상관은 detection의 VP에서 음수이고
segmentation의 CTS에서는 강한 음수다(순위 상관 -0.71). segmentation에서 NORMAL 최고
모델인 SimpleBEV가 세 강건성 축 모두에서 꼴찌이고(VP IMG 78.1, VP CAL 74.5, CTS IMG 평균
12.0), NORMAL 최하 모델인 CVT가 평균 기준 최고의 전이체다(CTS IMG 평균 31.0, bus 단독
으로는 LaRa 27.7이 가장 높다). 결국 카메라 기하 강건성은 단일한 양이 아니다. 시점 강건성은 extrinsic이 feature
샘플링을 결정하는지가(5.1), 전이는 표현이 source rig의 prior에 의존하는지가(5.2)
결정하며, in-distribution 정확도는 어느 쪽도 결정하지 않는다. RoboGeo가 mVRS와 CTS를
하나의 점수로 합치지 않고 분리해 보고하는 이유다.

---

## 부기

0. 영어화 지침(확정).
   (a) 단정 강도. 근거가 넓은 주장(EXT와 CAL의 대소 규칙, 11개 중 10개)은 단정 유지.
   2개 사례 기반 인과 설명(CVT와 SimpleBEV의 embedding 역전)은 "we attribute this to"
   또는 "consistent with"로 한 단계 낮춤(본문 반영 완료).
   (b) 신조어. 그룹명은 §5 첫 등장에서 한 문장으로 정의하고 이후 일관 사용(본문 반영
   완료). 기존 paradigm 라벨이 설명에 실패함을 먼저 보이므로 신조어 도입이 정당화됨.
   (c) 문체. RoboBEV의 장식적 어휘는 따르지 않고 구조만 차용. 각 문단은 주장, 수치,
   메커니즘, 귀결 한 문장의 순서로 구성.
   (d) 수치 인용. 현재 방식 유지(상대 변화 + 모델명, 변형 표기 불필요).
1. 그룹 영어 명칭(확정 제안). **projection-sampling** 과 **extract-then-place**
   (형용사형 하이픈 표기, 예: "projection-sampling models"). 대안이었던 geometry
   early/late는 폐기.
2. 본문 수치는 Table 2의 1/7 mVRS로 통일했고, per-axis도 1/7로 재계산해 사용(패턴은
   all-camera와 동일하게 유지됨을 확인). all-camera 분해는 부록 표나 Fig. A의 보조
   버전으로 제공 권장(격차가 최대 2배로 증폭되어 보임).
3. Fig. A는 1/7 mVRS 스케일(72에서 98)로 재생성 완료. 캡션에 두 가지 명시할 것.
   첫째, SimpleBEV는 이상치로 회색 표기(유일하게 대각선보다 위쪽). 둘째, Fig. C의
   VP IMG 순위는 분해(all-camera) 기준이고(1/7 IMG는 1포인트 안으로 압축되어 순위가
   무의미) CTS IMG는 suv와 bus의 평균이다(bus 단독이면 SimpleBEV 꼴찌 아님, CVT 최고
   아님에 유의).
4. 5.3의 원래 outline "same ranking"은 데이터와 서론의 different patterns 주장 모두와
   상충. 본문은 "IMG끼리 부분 정렬, 축별로 분리, NORMAL은 무관"으로 작성.
5. SimpleBEV 이상치(CAL 74.5, EXT yaw 역행)는 인용 전 검증 필요. CVT와 SimpleBEV의 bus
   CAL 역전은 Table 2와 원시 IoU 대조로 검증 완료, 인용 안전.
6. 미완. DFA3D(bus oracle 학습 중), DSPE, PETRv2(Table 2 부재), Table 3 baseline.
