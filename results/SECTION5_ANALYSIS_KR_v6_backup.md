# §5 Analysis — 한글 v6 (벤치마크 톤 재구성: 3-finding 구조)

> 표기: headline mVRS와 1/7 성분은 퍼센트, all-camera 분해는 0에서 1 비율.
> 그룹 명칭: 투영 샘플링(projection sampling) / 추출 후 배치(extract then place) —
> v5에서 정의한 용어 그대로, 신규 용어 도입 없음.
> SimpleBEV는 논문에서 완전 제외. VP 수치: BEVDet, BEVDepth, CAPE는 full(3792프레임,
> 631셀), BEVFormer, DETR3D와 segmentation은 768프레임 subset(full 진행 중).
> 그림 대응(figures/paper/): **Fig. A** = fig_mechanism_diagnostic(통합: (a) EXT-IMG
> 산점, (b) VP CAL−EXT 부호, (c) CTS CAL−EXT 부호), **Fig. B** = fig_cts_img_to_cal
> (SUV 덤벨), **Fig. C** = fig_vp_cts_alignment(정렬 산점).

---

## 5. Analysis

이 절의 목적은 RoboGeo가 점수표 하나를 더하는 것이 아니라, 서로 다른 종류의 카메라 기하 실패를 분리해 보여주는 진단 도구임을 보이는 것이다. 네 가지 입력 조건(NORMAL, EXT, IMG, CAL), per-camera와 all-camera 분해, 그리고 VP와 CTS 두 protocol이 각각 무엇을 분리하는지를 세 가지 질문으로 보인다. 재렌더링이 무엇을 드러내는가(§5.1), 플랫폼 전이가 무엇을 드러내는가(§5.2), 두 protocol은 중복인가(§5.3). 결과를 관통하는 관찰은, 강건성의 차이가 기존 view-transformation paradigm 라벨(forward, backward, sparse)을 따르지 않고, 모델이 extrinsic을 feature sampling 전에 쓰는지 이후에 쓰는지와 함께 움직인다는 것이다. 본 절에서는 이 구분을 강건성을 해석하는 진단 축으로 사용한다.

### 5.1. What Does Re-Rendering Reveal?

먼저 per-camera perturbation과 all-camera perturbation은 서로 다른 실패를 측정한다. 단일 카메라만 회전하는 per-camera 조건에서는 나머지 다섯 뷰가 정상으로 남아 multi-view redundancy가 손상을 상당 부분 흡수하며, 5개 검출기 모두에서 후방 카메라가 가장 치명적이고 좌우 측면이 가장 덜 중요한 공통 순서가 나타난다(상세 분해는 부록 표 X). 모델 간 구조적 차이는 여섯 카메라가 함께 움직이는 all-camera 조건에서 선명하게 드러나므로, 이하의 분석은 all-camera 분해를 중심으로 한다.

**Finding 1. Extrinsic-only perturbation은 실제 시점 변화의 손상을 과소보고한다.** Fig. A(a)에서 IMG 점수를 EXT 점수에 대해 그리면, 투영 샘플링 모델(BEVFormer, DETR3D, PointBeV)은 대각선 근처에 놓여 두 조건의 손상이 거의 같다. 반면 추출 후 배치 모델은 모두 대각선 아래에 놓이며 IMG가 EXT보다 대략 6--10%p 낮고, 두 그룹 사이는 거의 비어 있다. 한편 IMG 조건 자체에서는 모델 간 차이가 크게 압축된다(detection 82.8--83.9, segmentation 79.6--84.5). 즉 이미지를 그대로 두고 extrinsic만 바꾸는 평가는 점수를 일률적으로 올리는 것이 아니라, 어떤 모델이 강건해 보이는지 자체를 바꾼다. 시점 변화를 충실히 평가하려면 metadata 교란만으로는 부족하고 재렌더링된 이미지가 필요하며, 이것이 RoboGeo가 IMG 조건을 제공하는 이유다.

**Finding 2(VP). Calibration의 이득은 조건부이며, 4-condition protocol이 그 조건을 분리한다.** 바뀐 기하에 맞는 extrinsic을 함께 공급하는 CAL 조건에서 순위는 크게 재배열된다. BEVFormer는 83.1에서 93.5로, DETR3D는 84.2에서 95.6으로, PointBeV는 79.4에서 92.3으로 EXT 수준을 넘어 회복하는 반면, CAPE는 94.0에서 88.6으로, BEVDet은 89.2에서 87.5로, BEVDepth는 90.2에서 87.4로, LSS는 88.4에서 86.3으로 오히려 EXT보다 낮아진다. 이 분기는 extrinsic을 소비하는 위치와 함께 움직인다. 3D query나 BEV point를 extrinsic으로 이미지에 투영한 뒤 해당 좌표에서 feature를 읽는 **투영 샘플링** 모델은 올바른 extrinsic이 주어지면 샘플링 좌표가 다시 맞춰진다. 반면 backbone이 먼저 image feature를 추출한 뒤 depth splat이나 positional encoding으로 기하를 나중에 적용하는 **추출 후 배치** 모델은, 기울어진 시점이 이미 feature에 반영된 뒤라 올바른 extrinsic을 넣어도 되돌리기 어렵다. 실제로 CAL−EXT의 부호는 분석 대상 10개 모델 중 9개에서 이 구분과 일치한다(Fig. A(b)). 유일한 예외는 GaussianLSS로, 추출 후 배치 계열임에도 mVRS 기준 CAL이 EXT보다 높지만(92.1 대 90.7) all-camera 성분에서는 0.699 대 0.698로 사실상 경계선이다. 이 결과는 CAL−EXT 부호가 "calibration을 더 정확히 알면 도움이 되는가"를 모델별로 가려내는 유용한 진단 신호임을 시사한다.

축별로 보면 yaw는 calibration-type error에 가깝다. EXT와 IMG의 차이가 작고 CAL에서 대부분 회복된다. 반면 pitch와 roll은 특히 추출 후 배치 모델에서 content-type error에 가까워, 올바른 extrinsic을 넣어도 손상이 복구되지 않으며, detection에서는 CAL-pitch가 가장 깨끗한 판별 조건이다(축별 상세는 표 X, 크기별 곡선은 부록). 또한 IMG 손상의 성격은 검출 소실이 아니라 위치 오류에 가깝다. 살아남은 검출의 translation error가 크게 늘어 물체가 BEV 공간의 잘못된 위치에 나타난다(상세 분해는 부록).

### 5.2. What Does Cross-Platform Transfer Reveal?

**Bus 전이는 대체로 SUV보다 어렵고, 가장 큰 실패는 depth 의존에서 나온다.** SUV가 상대적으로 온건한 layout 변화라면, bus는 카메라 높이와 시점 자체를 바꾸는 더 강한 shift이며, 실제 관측되는 객체 분포도 달라진다(Table 1의 class-wise visibility가 sedan에서 bus로 9.5--19.4%p 이동). 이 설정에서 RoboGeo가 드러내는 가장 선명한 실패는 depth 기반 detector의 bus IMG 붕괴다(BEVDet 0.2, BEVDepth 0.1). depth를 직접 예측하지 않는 detector들은 같은 조건에서 훨씬 잘 전이된다. 이는 sedan 마운트에서 학습된 depth prior가 높아진 시점에서 잘못된 거리를 예측해 feature를 잘못된 BEV 위치로 옮긴다는 해석과 일관되며, SUV IMG에서 depth 기반 모델의 mASE(0.63--0.73)가 depth-free 모델(0.31--0.36)보다 훨씬 큰 것이 이를 뒷받침한다.

**Finding 2(CTS). 올바른 calibration이 항상 해결책은 아니다.** Fig. B는 SUV에서 IMG에 타깃 extrinsic을 추가로 공급했을 때(CAL) 누가 회복하는지를 보여준다. 투영 샘플링 모델은 크게 회복한다(BEVFormer 37.2에서 71.9, DETR3D 34.9에서 76.9, PointBeV 20.2에서 68.4). depth 기반 detector는 bus에서 부분적으로만 회복한다(BEVDet 0.2에서 22.8, BEVDepth 0.1에서 16.6). 이는 BEVFormer와 DETR3D의 bus CAL인 40.1과 37.6에 한참 못 미치는 수준으로, 병목이 extrinsic의 유무만이 아님을 시사한다. CVT는 bus에서 올바른 extrinsic을 받을 때 오히려 크게 떨어진다(17.7에서 1.8). 우리는 이를 학습된 geometry embedding이 source rig에 묶여 있어 타깃 기하가 더 강한 out-of-distribution 입력이 되기 때문으로 추정하지만, 이는 직접 측정이 아닌 구조에 근거한 가설이다. 종합하면 RoboGeo의 CTS protocol은 "정확한 calibration만으로는 충분하지 않은 경우"를 모델별로 그대로 드러낸다.

같은 CAL−EXT 부호 분리는 cross-platform에서 더 깨끗하게 반복된다(Fig. A(c)). 투영 샘플링 모델은 SUV와 bus 모두에서 CAL이 EXT를 넘어서고, 추출 후 배치 모델은 모두 그 반대로, 부호 일치는 SUV 10/10, bus 10/10이다. VP에서 유일한 예외였던 GaussianLSS조차 cross-platform에서는 규칙을 따른다. 즉 extrinsic-only 평가는 이 스케일에서도 강건성을 과대평가한다.

CTS 해석에는 한 가지 주의가 필요하다. CTS는 각 모델의 target-platform oracle로 나눈 비율이므로 oracle이 낮은 모델에 상대적으로 유리할 수 있고, detection의 bus oracle은 모델 간 최대 1.6배 차이가 난다. 예컨대 CAPE와 DETR3D는 bus IMG에서 absolute SDS가 거의 같지만(0.162 대 0.159) CAPE의 bus oracle이 더 낮아(0.449 대 0.602) CTS는 36.0 대 26.5로 벌어진다. 따라서 depth 사용 여부에 따른 큰 그룹 차이는 견고하지만, 그룹 안의 미세한 순위는 absolute transfer performance와 함께 읽어야 한다.

### 5.3. Are VP and CTS Redundant?

**Finding 3. All-camera IMG는 transfer의 유용한 proxy지만, 플랫폼 전이는 별도로 측정해야 한다.** Fig. C에서 all-camera IMG 점수는 SUV CTS-IMG와 강하게 정렬된다. detection에서는 순위가 정확히 일치하고(Spearman ρ=1.00) segmentation에서도 높다(ρ=0.90). all-camera perturbation은 전체 rig가 함께 바뀌는 조건이므로, source platform 안에서 측정할 수 있는 가장 가까운 cross-platform proxy로 작동한다.

그러나 같은 VP라도 무엇을 보느냐에 따라 이 신호는 사라진다. per-camera IMG는 detection에서 순위 상관이 음수가 되고(ρ=−0.62), segmentation에서는 순위는 보존되지만 변별 폭이 0.03으로 all-camera의 0.16의 약 5분의 1에 그친다. headline mVRS가 per-camera 모드를 6 대 1로 더 많이 포함할 때 transfer 관련 신호가 희석되는 이유다. EXT 조건으로 평가하면 관계는 더 나빠져, detection에서는 반전되고(ρ=−0.70) segmentation에서도 약해진다(ρ=0.60). 즉 cross-platform 배치 가능성을 가늠하는 지표로는 extrinsic-only 점수가 아니라 재렌더링된 all-camera IMG 점수를 봐야 한다.

그럼에도 두 protocol은 교환 가능하지 않다. 같은 all-camera IMG 순위도 bus CTS-IMG와는 정렬이 약해지고(detection ρ=0.60), 개별 모델 수준에서는 분리가 나타난다. CAPE는 VP CAL에서 약하지만 normalized bus IMG CTS에서는 가장 높고, BEVFormer와 DETR3D는 VP CAL을 지배하지만 cross-platform IMG 순위에서는 최상위를 CAPE에 내준다. NORMAL 성능 역시 전이를 예측하지 못한다. segmentation에서는 NORMAL 순위와 SUV·bus 평균 CTS-IMG 순위의 상관이 음수이며(ρ=−0.50), NORMAL이 가장 낮은 CVT가 평균 CTS-IMG에서는 가장 높다. 플랫폼 전이는 source-rig depth prior, geometry embedding, target-platform visibility까지 함께 반영하므로, RoboGeo가 mVRS와 CTS를 하나의 점수로 합치지 않고 분리해 보고하는 이유가 여기에 있다.

---

## 부기 (작업 기록, 논문 비수록)

0. 영어화 지침(확정, v5에서 승계).
   (a) 단정 강도. **관찰(수치)은 단정 유지, 해석은 한 단계 하향** — v6에서 전면 적용.
   관찰 예: "the sign of the CAL-EXT gap matches the grouping in 9/10 (VP) and 10/10
   (CTS SUV/bus) models." 해석 예(확정 문장): "The results suggest that
   extrinsic-consumption style is a useful diagnostic axis for interpreting robustness."
   / "This suggests that RoboGeo can expose cases where correct calibration is not
   sufficient, especially for models with learned geometry encodings." 기존 확정 문장
   유지: "Extract-then-place models lie below the diagonal." / "This under-reporting is
   concentrated in extract-then-place models and is much weaker for projection-sampling
   models." / "Bus transfer is generally harder than SUV transfer, with the gap most
   visible in detection and in the CAL reversal of CVT." / "CAPE is top-ranked by
   normalized bus-IMG CTS; its absolute SDS is close to DETR3D, and the CTS gap partly
   reflects CAPE's lower bus oracle." / "We attribute this reversal to
   source-rig-specific geometry embeddings, although this remains an architectural
   hypothesis rather than a directly measured quantity."
   (b) 신조어 금지. 그룹명은 5.1 첫 등장에서 한 문장으로 정의 후 일관 사용:
   **projection-sampling** / **extract-then-place**. v6 추가 용어 없음("진단 축/진단
   신호"는 diagnostic axis/signal로 통일).
   (c) 문체. 주어를 가능하면 "RoboGeo/the benchmark/the protocol"로 — 벤치마크가
   드러낸다(expose/reveal/separate) 프레임. 장식 어휘 금지.
   (d) 수치 인용. 상대 변화 + 모델명 방식 유지.
1. **v6 재구성 기록(2026-06-10).** 사용자 결정: Datasets Track 톤에 맞춰 §5를 3-finding
   구조로 압축. (i) Finding 1 = re-rendering matters(EXT-only 과소보고), (ii) Finding 2
   = calibration helps conditionally(VP의 CAL 분기 + CTS의 부분 회복/CVT 역전),
   (iii) Finding 3 = VP와 CTS는 관련되나 교환 불가. **본문에서 축소·이관된 항목**:
   yaw/pitch 축 분류는 4문장으로 압축(상세 → 표 X), 크기별 knee와 CAL 안전 구간
   문단은 부록으로 이관(한 문장 요약만 유지: 위치 오류 성격), per-camera 카메라 순서
   문단은 5.1 도입부 두 문장으로 압축, CVT embedding 가설은 2문장, mASE는 1문장,
   CAPE oracle caveat은 짧은 문단 유지. **메커니즘 주장 톤 하향**: "결정한다" →
   "함께 움직인다/진단 축으로 사용한다". 수치 관찰(9/10, 10/10, ρ값, 회복 폭)은 단정
   유지. 상세 메커니즘 분석(v5 5.1(4)~(5) 전문)은 부록 "extended mechanism analysis"로
   통째 이관 예정 — v5 본문이 원고(SECTION5_ANALYSIS_KR_v5_backup.md).
2. 본문 수치는 Table 2의 1/7 mVRS로 통일. all-camera 분해는 부록 표 권장. 5.1 끝
   문단의 크기/위치 오류 요약은 all-camera 기준임을 영어판에서 명시(부록 그림 Y 연결).
3. **그림 최종(2026-06-10, figures/paper/, 범례 별도 파일).** 본문 3장:
   **Fig. A** fig_mechanism_diagnostic — (a) EXT-IMG 산점(5.1 Finding 1),
   (b) VP CAL−EXT 부호(5.1 Finding 2), (c) CTS CAL−EXT 부호(5.2, SUV 진한/bus 연한);
   (b)(c) 행 정렬 공유(메커니즘 순), 부호 일관성 VP 9/10, CTS suv 10/10, bus 10/10.
   **Fig. B** fig_cts_img_to_cal — SUV 단일 패널 덤벨(IMG 빈 원, CAL 찬 원), 행은
   detection | segmentation 구분(점선), 5.2 Finding 2. **Fig. C** fig_vp_cts_alignment
   — all-camera IMG 대 SUV CTS-IMG 산점만(ρ 막대 패널은 부록 후보로 강등,
   figures/fig_vp_cts_align.png 2패널 버전 보존). 범례 4종 별도: legend_mechanism_row /
   _2x2 / legend_img_cal / legend_suv_bus. 구버전(상위 figures/ 5장 세트, suptitle 포함
   make_section5_figs*.py)은 작업용으로 보존.
   Fig. C 캡션 명시 사항(승계): VP IMG는 all-camera 기준(1/7 IMG는 detection에서
   1.2포인트로 압축), 본문 ρ는 SUV CTS-IMG 기준.
4. 5.3의 원래 outline "same ranking"은 데이터와 상충하여 "부분 정렬 + 축별 분리 +
   NORMAL 무관"으로 유지(v4에서 확정).
5. **SimpleBEV 완전 제외(2026-06-10 확정, 팩트체크 완료).** 원 논문 Simple-BEV는
   learned embedding 없는 전형적 투영 샘플링이 맞으나(이전 "embedding 기반" 분류는
   오류), 우리가 받은 구현의 행동은 충실한 샘플러와 불일치: CAL 곡선이 IMG와 전 축 전
   크기에서 0.01 이내 동일, EXT-yaw 전 크기 0.90 이상 평탄. 동일 하니스의 PointBeV는
   정반대(EXT-yaw 0.12 급락, CAL-yaw 0.95 회복). 규칙의 반례가 아니라 구현 아티팩트.
   최종 확정은 bevunify 측 소스 확인 필요. 제외 후 통계: seg NORMAL-CTS 상관 -0.50,
   seg VP-CTS 0.70, 규칙 10개 중 9개. 원자료 seg_vp_cts.tsv, eval_results/ 보존.
6. 미완. DFA3D(bus oracle 학습 중, 완료 후 full VP/CTS — **v6 프레임에서는 "벤치마크가
   sampling×depth 두 축을 분리해 읽을 수 있음을 보이는 controlled case study"로
   재포지셔닝**), DSPE, PETRv2(Table 2 부재), Table 3 baseline(Ext.Aug, EAFormer,
   PD-BEV).
6b. Table 2의 dash row 처리(승계): 최종 표에서 dash row 제거 방침. DSPE 표에서 제거
   권장, DFA3D는 완료 시 채우고 미완이면 행 제거.
7. **Table 2 정정 필수(승계).** BEVDet CTS 4셀 oracle 나눗셈 누락 — 정정값 suv
   17.0/16.1, bus 0.2/22.8. 본문·그림 반영 완료, 논문 표 수정 필요. §4 "seven
   methods"(seg) → "six"(BEVFormer(seg)를 목록에서 제거하는 편집; SimpleBEV 행 삭제는
   Table 2 쪽).
7b. CTS>1 관측(CAPE sedan, NORMAL 조건 NDS 0.4842 > bus oracle 0.4490, 비율 1.078)은
   본문 제외, 부록 후보. 출처 results/CAPE/cts/cts_cape_sedan_summary.txt.
8. **full-val 전환 진행(승계).** BEVDet, BEVDepth, CAPE full 완료·본문 반영.
   BEVFormer per-cam full(2-shard)과 DETR3D full 체인 실행 중 — 완료 시 수치 교체 +
   Table 2 dagger 제거. segmentation full은 bevunify 측 재계산 필요.
9. 부록 자산 목록(v6 확장): 표 X(per-camera 분해), 표 Z(CTS EXT 조건), 그림
   Y(크기별 all-camera 곡선 = fig_app_magnitude), **ρ 예측자 막대(구 Fig. C (b)
   패널)**, **extended mechanism analysis(v5 5.1(4)(5) 원문)**, all-camera breakdown
   표, per-class/TP 분해, GaussianLSS 경계선 수치, LSS pitch 부호 비대칭, CAL-yaw
   회복 상세.
10. 검증 기록(승계): v4 5.3 ρ 전수 재계산 일치(+1.00/−0.62/−0.70, seg
   +0.90/+0.90/+0.60, bus +0.60). **ρ=1.00은 BEVDet CTS 정정값(suv IMG 17.0)
   기준에서만 성립**(구값이면 0.90) — Table 2 정정의 추가 근거. v5 개고 3건 검증
   일치(2026-06-10). seg per-camera ρ=+0.90이므로 "약하거나 음수" 일반화 금지 —
   "det 음수, seg는 변별 폭 0.03/0.16" 표현 유지(v6 반영).
