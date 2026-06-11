# §5 Analysis — 한글 v3 최종 통합본 (전체 분석, 판단 반영)

> 본문 수치는 Table 2의 headline mVRS(%)를 기본으로 사용한다. 세부 분해(all-camera,
> per-axis 1/7, 크기별 곡선)는 출처를 명시하고 부록과 연결한다. 그룹 명칭은 "투영
> 샘플링(projection sampling)"과 "추출 후 배치(extract then place)". SimpleBEV는
> 논문에서 완전히 제외한다(본문, 표, 그림, 부록 모두 미수록 — 부기 5).
> VP 수치 기준: BEVDet, BEVDepth, CAPE는 full(프레임 3792, VP 그리드 631셀 전부),
> BEVFormer, DETR3D와 segmentation은 768프레임 subset(full 재계산 진행 중, 완료 시
> 교체 — 부기 8).

---

## 5. Analysis

세 가지 질문을 따라 분석한다. 시점 강건성이 모델 간에 어떻게 다른가(5.1), 플랫폼 전이는
어떻게 다른가(5.2), 두 축이 서로를 예측하는가(5.3). 셋을 관통하는 관찰은 하나다.
**강건성은 view transformation의 paradigm(forward, backward, sparse)이 아니라 거의
독립적인 두 구조 성질이 결정한다.** 시점 강건성은 extrinsic이 feature 샘플링을
결정하는지가, 플랫폼 전이는 표현이 source rig의 prior에 의존하는지가 가른다. 두 task가
같은 지문을 보이며, detection에서 가장 선명하다. 표기 규약: headline mVRS와 그
1/7 성분(per-axis 포함)은 퍼센트로, all-camera 분해 점수는 0에서 1 비율로 적는다.

### 5.1. Viewpoint Robustness

**(1) Paradigm별 분석. IMG에서는 순위가 압축되고, CAL에서 재배열된다.**

원래 extrinsic을 유지한 채 시점만 다시 렌더링한 IMG 조건에서는 모든 모델이 paradigm과
무관하게 비슷한 수준으로 떨어진다. detection의 mVRS는 82.8에서 83.9 사이(BEVDepth 82.8,
BEVFormer 83.8, CAPE 83.9), segmentation은 79.6에서 84.5 사이에 몰려, IMG 순위는
forward, backward, sparse 어느 라벨도 따르지 않는다. 반면 바뀐 기하의 올바른 extrinsic을
함께 공급하는 CAL 조건에서는 순위가 크게 재배열된다. detection에서는 BEVFormer가
83.1에서 93.5로, DETR3D가 84.2에서 95.6으로, segmentation에서는 PointBeV가 79.4에서
92.3으로 EXT 수준을 넘어 회복한다. 반면 detection의 CAPE는 94.0에서 88.6으로, BEVDet은
89.2에서 87.5로, BEVDepth는 90.2에서 87.4로, segmentation의 LSS는 88.4에서 86.3으로
오히려 EXT보다 낮아진다.

**이 회복은 paradigm의 속성이 아니다.** 회복하는 모델은 backward인 BEVFormer와 sparse인
DETR3D, PointBeV에 걸쳐 있고, 회복하지 못하는 모델은 forward 전부와 sparse인 CAPE,
backward인 CVT, LaRa에 걸쳐 있다. 가르는 것은 extrinsic을 소비하는 방식이다. 3D query나
BEV 점을 extrinsic으로 이미지에 투영해 그 좌표에서 feature를 읽는 방식을 **투영
샘플링(projection sampling)**, backbone이 extrinsic과 무관하게 feature를 먼저 추출하고
기하를 depth splat(LSS 계열)이나 positional encoding(CVT, LaRa, CAPE)으로 나중에
적용하는 방식을 **추출 후 배치(extract then place)** 라 부른다. 투영 샘플링 모델은
올바른 extrinsic이 주어지면 투영이 다시 정렬되어 회복하고, 추출 후 배치 모델은 기울어진
이미지가 이미 추출된 feature에 박혀 있어 올바른 extrinsic으로도 되돌릴 수 없다. 정리하면,
우리 결과에서 **CAL과 EXT 격차의 부호는 모델이 extrinsic을 어디에서 소비하는지를 대체로
반영하며**, 이 경험적 진단은 분석 대상 10개 모델 중 9개에서 들어맞는다. 유일한 예외는
GaussianLSS로, 추출 후 배치 계열임에도 mVRS 기준 92.1 대 90.7로 CAL이 EXT보다 높다.
다만 all-camera 성분에서는 0.699 대 0.698로 사실상 경계선에 있다(부록).

**(2) EXT와 IMG의 상관 (Fig. A).**

mVRS의 IMG 점수를 EXT 점수에 대해 그리면(Fig. A, 가로축 EXT, 세로축 IMG) 같은 분기가
metadata 쪽에서도 드러난다. 투영 샘플링 모델은 대각선에 붙어 있다. BEVFormer는 83.1 대
83.8, DETR3D는 84.2 대 83.9, PointBeV는 79.4 대 79.6으로 EXT와 IMG가 사실상 같다.
샘플링에 쓰는 extrinsic을 오염시키는 것과 이미지를 오염시키는 것이 동등하게 치명적이기
때문이다. 추출 후 배치 모델은 대각선 아래에 놓인다. CAPE는 94.0 대 83.9, BEVDepth는
90.2 대 82.8, LaRa는 91.1 대 82.5로, EXT가 IMG보다 6에서 10포인트가량 높다. 깨끗한
이미지는 extrinsic 오류에서 살아남지만 기울어진 이미지는 추출 자체를 오염시키기
때문이다. 따라서 extrinsic만 교란하는 평가는 실제 시점 변화에서의 저하를 과소보고하며,
이 과소보고는 주로 추출 후 배치 모델에서 나타나고 투영 샘플링 모델에서는 크게 나타나지
않는다. 본 벤치마크의 첫 번째 주장을 메커니즘 수준에서 정밀화한 결과다. 한편 이 격차는
all-camera 성분에서 최대 2배까지 벌어진다(부록). headline mVRS는 per-camera 점수가 모든
모델에서 90 부근으로 비슷해 격차를 압축하기 때문이며, 그 이유는 아래 카메라 분석에서
설명한다.

**(3) 축과 카메라 분석.**

축 분석. 1/7 기준 per-axis 점수의 요점은 넷이다. IMG에서는 detection은 pitch(79에서
81), segmentation은 다수 모델에서 yaw(77에서 79; PointBeV는 pitch가 근소 우세)가 가장
치명적이다. 일관된 yaw는 카메라 간 대응을 깨고, pitch는 거리와 크기 추정이 의존하는
지면을 기울이기 때문이다. EXT에서는 최악 축이 메커니즘의 지문이다. 추출 후 배치
검출기는 yaw(83, 나머지 축 91에서 99), 투영 샘플링 검출기는 pitch(78에서 81)에서 가장
약하다. 잘못 배치되는 쪽은 방위가 틀어지는 yaw가, 잘못 샘플링하는 쪽은 하늘과 지면을
오가는 pitch가 아픈 것이다. CAL에서는 yaw가 어느 모델에서나 거의 회복되는
반면(detection 97에서 99, segmentation 92에서 98), pitch는 투영 샘플링 검출기만
회복한다(90과 95 대 79에서 81). 따라서 **CAL의 pitch 점수가 detection에서 가장 깨끗한
메커니즘 판별자다.**

카메라 분석. 단일 카메라 교란은 비교적 무해하고 그 양상은 아키텍처와 무관하다(카메라별
분해는 부록 표 X). per-camera IMG 점수는 모든 검출기에서 대략 81에서 98 사이에
머물고(최저는 BEVDet 후방 80.7), 중요도의 계층 구조는 5개 검출기에서 동일하다. 후방
카메라가 가장 결정적이고(81에서 83) 전방 카메라가 그 다음이며(85에서 89) 좌우 측면
카메라가 가장 덜 중요하다(95에서 98). 개별 카메라 단위의 미세 순서만 모델별로 한 쌍씩
다르다. 이 순서는 모델이 아니라 rig의 시야 중첩과 장면의 객체 분포에서 나온다. 여섯 개
뷰의 융합이 오염된 한 뷰를 그냥 눌러버리는 것이다. 결국 per-camera 강건성은 아키텍처가
아니라 rig의 중복도를 측정한다. headline mVRS는 배치 환경에서 흔한 국소 마운트 교란까지
포함하는 배치 관점의 지표로서 per-camera를 6 대 1로 가중하며, 그 결과 위에서 본 압축이
생긴다. 따라서 mVRS는 배치 수준의 요약으로 두고, 모델 간 구조 차이는 all-camera 분해로
읽는 것이 적절하다.

**(4) 교란 크기와 손상의 성격.**

크기별 응답(all-camera 기준, 부록 그림 Y)은 두 가지를 더한다. 첫째, **CAL에서의 안전
구간은 투영 샘플링 모델에만 존재한다.** 가장 작은 교란인 4도에서, 같은 pitch와 roll 기준으로 투영 샘플링 검출기는 CAL
점수를 거의 유지하지만(DETR3D 0.93, BEVFormer 0.89) 추출 후 배치 검출기는 이미 절반
가까이 잃는다(BEVDet 0.55, BEVDepth 0.52, CAPE 0.57). depth splat을 쓰는 BEVDet과
BEVDepth는 pitch 12도 이상에서 사실상 0으로 떨어진다. 반면 IMG의 붕괴 지점은
메커니즘과 무관하게 비슷해서(roll과 yaw 기준 8도에서 13도 사이에 절반 상실), 메커니즘은
극단 크기의 꼬리에서만 갈린다(yaw 20도에서 투영 샘플링 0.30에서 0.40 대 추출 후 배치
0.10에서 0.16). 둘째, **IMG의 손상은 검출 소실이라기보다 위치 오류다.** 4도 all-camera
IMG에서도 검출기들은 느슨한 4.0m 매칭 기준으로는 물체의 60에서 68퍼센트를 여전히
찾지만, 엄격한 0.5m 기준 AP는 정상 대비 5에서 19퍼센트로 무너지고, 살아남은 검출의
translation 오차가 1.6에서 2.0배로 커진다(orientation 오차는 1.1에서 1.5배). 시점
변화가 물체를 안 보이게 만드는 것이 아니라 잘못된 곳에 보이게 만드는 것이며, 이것이
BEV 표현이 기하 오류에 취약한 이유다.

### 5.2. Cross-Platform Transfer

**(1) Paradigm별 분석. depth 의존성이 전이를 결정하고, CAL 응답이 메커니즘을 다시
가른다 (Fig. B).**

타깃 시점 이미지로 평가하는 IMG 조건에서 forward 검출기, 즉 depth 기반 모델은 bus에서
붕괴한다(CTS 기준 BEVDet 0.2, BEVDepth 0.1). sedan 마운트로 학습된 monocular depth가
높아진 bus 시점에서 잘못된 거리를 예측해 feature가 잘못된 BEV 거리로 옮겨지기 때문이다.
이 실패는 살아남은 검출에도 흔적을 남긴다. suv IMG에서 depth 모델의 크기 오차 mASE는
0.63에서 0.73으로 depth를 쓰지 않는 모델의 0.31에서 0.36보다 두 배 가까이 크고,
bus에서는 생존 자체가 거의 없어 mASE가 1.0으로 포화된다. depth를 쓰지 않는 검출기는
훨씬 잘 전이된다. 정규화된 CTS 기준으로는 시점 교란에서 가장 전형적인 추출 후 배치
모델이었던 CAPE가 bus IMG 36.0으로 가장 강하지만, 절대 SDS는 DETR3D와 사실상
같고(0.162 대 0.159) 그 격차의 일부는 CAPE의 낮은 bus oracle에서 온다(아래 측정상의
주의).

올바른 타깃 extrinsic을 공급하는 CAL은 시점 실험과 정확히 같은 방식으로 메커니즘을
가른다(Fig. B). 투영 샘플링 모델은 올바른 extrinsic을 큰 전이 이득으로 바꾼다. suv에서
BEVFormer는 37.2에서 71.9로, DETR3D는 34.9에서 76.9로, PointBeV는 20.2에서 68.4로
오른다. depth에 묶인 검출기는 CAL에서
부분적으로만 회복한다. bus에서 BEVDet은 IMG 0.2에서 CAL 22.8로, BEVDepth는 0.1에서
16.6으로 오르지만, 여전히 투영 샘플링 모델인 BEVFormer와 DETR3D의 CAL 성능(40.1과
37.6)에 한참 못 미친다. 병목은 타깃 extrinsic의 유무만이 아니라, depth 기반 lifting에
학습된 source rig의 depth prior로 보인다. **같은 경향은 EXT와 CAL의 부호에서도
반복된다(부록 표 Z).** depth 기반 검출기는 올바른 타깃 extrinsic을 받아도 CAL이 EXT보다
낮게 남는다(bus에서 BEVDet 22.8 대 53.9, BEVDepth 16.6 대 29.9; suv에서는 격차가 더
극적이어서 BEVDepth 5.7 대 69.5). 같은 추출 후 배치 계열인 depth-free CAPE도 같은
부호를 보인다(suv 34.3 대 82.3). 반면 투영 샘플링 모델은 CAL이 EXT를
넘어선다(bus에서 BEVFormer 40.1 대 25.1, DETR3D 37.6 대 30.7). segmentation까지
포함하면 이 부호 분리는 suv와 bus 양쪽에서 10개 모델 전부 성립한다(Fig. D). VP에서
유일한 예외였던 GaussianLSS조차 cross-platform에서는 부호를 따른다. 5.1에서 관찰한
CAL과 EXT 부호 진단이 cross-platform transfer에서 오히려 더 깨끗하게 반복되는 것이며, extrinsic-only 평가가
강건성을 과대평가한다는 결론도 이 스케일에서 그대로 성립한다. 가장 눈에 띄는 결과는 embedding 기반 segmentation 모델 CVT에서
올바른 extrinsic이 오히려 해롭다는 점이다. CVT는 bus에서 17.7에서 1.8로 떨어진다.
우리는 이를 학습된 geometry embedding이 source rig에 맞춰져 있어, 학습 분포에서 먼
진짜 타깃 기하가 낡은 sedan extrinsic보다 인코딩을 더 멀리 분포 밖으로 밀어내는 것으로
해석한다. 이 해석과 일관되게, 역전은 기하 변화가 가장 큰 bus에서만 나타난다. 다만 이는
embedding 분포를 직접 측정한 결과가 아니라 구조에 근거한 가설이다. 정리하면
calibration을 정확히 아는 것만으로는 cross-platform 배치에 충분하지 않으며, 모델이
extrinsic을 소비하는 방식이 더 나은 calibration이 도움이 되는지조차 좌우할 수 있다.

측정상의 주의. CTS는 각 모델 자신의 플랫폼 일치 oracle로 정규화하는데, 이 oracle이
검출기 간 최대 1.6배 차이가 난다. 그래서 ratio는 oracle이 약한 모델에 유리할 수 있다.
CAPE와 DETR3D는 bus IMG의 절대 전이 성능이 사실상 같지만(SDS 0.162와 0.159) CAPE의 bus
oracle이 낮아서(0.449 대 0.602) CTS는 36.0 대 26.5로 갈린다. depth 사용 여부에 따른
그룹 분리는 절대값으로 0.001 대 0.10에서 0.16이라 이 문제와 무관하게 견고하지만, 그룹
안의 미세한 순위는 절대 전이 성능으로 읽어야 한다. Table 2가 각 CTS 옆에 oracle을
병기하는 이유다.

**(2) 플랫폼 분석. bus 전이는 대체로 suv보다 어렵다.**

대부분의 모델에서 bus 전이는 suv보다 어렵고, 특히 detection과 embedding 기반
segmentation 모델 CVT의 CAL 역전에서 그 차이가 뚜렷하다. detection IMG에서 BEVFormer는
37.2에서 18.0으로, DETR3D는 34.9에서 26.5로 떨어지고 depth 검출기들은 suv의 10에서 17
수준이 bus에서 0.2 이하로 무너진다. segmentation IMG는 suv에서 20.2에서 44.3 범위였던
것이 bus에서 11.4에서 27.7로 줄고, bus CAL은 CVT에서 1.8로 붕괴한다. 두 플랫폼 모두
layout과 시점이 함께 바뀌지만 정도가 다르다. suv는 상대적으로 온건한 layout 변화에
가깝고, bus는 카메라 높이와 시점 레짐까지 바꾸는 더 강한 재구성이다. 그 결과 첫째로
sedan에서 학습된 depth prior가 가장 세게 깨지고, 둘째로 앞의 해석을 따르면 geometry
embedding이 학습 분포에서 가장 멀리 밀려나며(실제로 CAL 역전은 bus에서만 나타난다),
셋째로 보이는 것 자체가 바뀐다. sedan에서 bus로 가면 클래스별 visibility가 9.5에서
19.4포인트 이동하는데(Table 1) 이 변화는 플랫폼별 oracle이 흡수하지만 전이된 모델은
스스로 감당해야 한다. 따라서 cross-platform 강건성은 플랫폼별로 읽어야 한다. suv는
주로 온건한 layout 변화에 대한 내성을, bus는 시점 레짐 변화까지 포함한 더 강한 재구성에
대한 내성을 측정한다.

### 5.3. VP와 CTS의 상관

Fig. C는 평가축이 NORMAL에서 VP IMG, VP CAL, CTS IMG로 바뀔 때 각 모델의 순위 변화를
보여준다. 여기서 CTS IMG는 suv와 bus의 평균이다(Fig. C 캡션에도 명시). 순위는 두
프로토콜이 같은 병목을 공유하는 곳에서는 부분적으로 정렬되고, 지배 성질이 다른 곳에서는
분리된다. IMG 조건끼리는 시점 순위와 전이 순위가 segmentation에서 강하게, detection에서
중간 정도로 상관된다(순위 상관 0.70과 0.60). 둘 다 그 지점에서는 같은 이미지 외형
변화가 지배하기 때문이다. 그러나 정렬은 앞서 밝힌 두 축을 따라 정확히 깨진다. CAPE는
일관된 시점 변화의 흡수에서는 하위권이지만(CAL 88.6, 추출 후 배치) depth를 쓰지 않아
정규화된 bus IMG CTS 기준으로는 최상위이고(36.0), 반대로 투영 샘플링인 BEVFormer와
DETR3D는 CAL을 지배하지만(93.5와 95.6), normalized CTS 기준의 cross-platform IMG
순위에서는 최상위 자리를 CAPE에 내준다. segmentation에서는
PointBeV가 같은 패턴을 반복한다(CAL 92.3으로 최상위권이나 bus IMG는 11.4로 최하위권).
그리고 어느 축도 표준 리더보드로 예측되지 않는다. NORMAL 순위와의 상관은 detection의
VP에서 음수이고 segmentation의 CTS에서는 음수다(순위 상관 -0.50). segmentation에서
NORMAL 최하 모델인 CVT가 평균 기준 최고의 전이체다(CTS IMG 평균 31.0, bus 단독으로는
LaRa 27.7이 가장 높다). 결국 카메라 기하 강건성은 단일한 양이 아니다. 시점 강건성은
extrinsic이 feature 샘플링을 결정하는지가(5.1), 전이는 표현이 source rig의 prior에
의존하는지가(5.2) 결정하며, in-distribution 정확도는 어느 쪽도 결정하지 않는다.
RoboGeo가 mVRS와 CTS를 하나의 점수로 합치지 않고 분리해 보고하는 이유다.

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
