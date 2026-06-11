# 별도 method 논문 스펙 — 검증 완료판 (v1, 2026-06-11)

> 입력: ROBUST_MODEL_PROPOSAL_KR.md v2 (C1–C5).
> 검증: 문헌 4-에이전트(웹, venue/year 확인) + 적대 기술 리뷰 + AC 시뮬레이션
> (원본 `results/_method_paper_vetting.json`). 이 문서가 v2를 **대체**한다 —
> v2의 C1·C5는 검증에서 탈락했고, C3·C4는 수정 조건부 생존.

---

## 0. 결론 한 줄

**논문은 C4 단독 헤드라인으로 간다: "Test-time rig self-calibration for
multi-camera BEV perception" — 낡은 extrinsic에 의한 시점 손상은 대부분 복구
가능한 calibration 오차이며, BEV 공간 탐색+회귀로 oracle-calibration 격차의
X%를 frozen detector 위에 plug-in으로 회수한다.** C2+C3(수정판)는 한 절짜리
"calibration-faithful 기판"으로 강등, C1·C5는 컷. 분석 발견(BN 포화, 슬롯
prior, 부호 진단)은 방법과 독립적으로 살아남는 최강 자산이므로 분석 절로 유지.

## 1. 검증이 강제한 변경 (요지)

| 항목 | 판정 | 결정적 근거 |
|---|---|---|
| C1 정준화 | **컷** | (i) 학습 rig가 정확히 중력 정렬(검증: CAM_FRONT elev 0.000°, up_z=1.0) → train-time 정준화 = 항등 → "학습·추론 동일 적용" 주장 자체가 거짓, 테스트에서만 워프 아티팩트 = 순수 분포 이동. (ii) SUV도 중력 정렬(높이만 2.35m) → C1 무효, bus만 유효한데 그 20° pitch 제거 비용 = 픽셀 41.7% + 근거리 시야 상실. (iii) UniDrive(ICLR'25)·BEV-LaneDet(CVPR'23)이 아이디어 선점. (iv) C4의 "feature 캐시" 전제를 파괴(내부 모순). 잔존 형태: 테스트타임에 C4가 추정한 ΔR을 argmax에서 1회 재워프하는 옵션만. |
| C2 슬롯 제거 | 기판으로 유지 | EAFormer가 원리 선점(seg) → "알려진 원리를 패러다임 전반에 체계 적용+정량화"로만 주장. ray-조건은 **공허함이 증명됨**(intrinsic이 플랫폼 간 동일 + 회전 정준화 시 ray장은 픽셀 좌표의 고정 함수 = positional encoding과 등가) → "cams_embeds를 해석적 ray/yaw 인코딩으로 대체"로 재포지셔닝, ray-PE vs CoordConv vs none ablation 필수. |
| C3 높이 게이트 | **하이브리드로 수정, DFA3D 결과에 조건부** | 치명 수치: 1.6m 차량 rig에서 검출 주력 20–60m 대역이 수평선 위 — 지면 접점 elev 2.29°(40m)에서 0.1m 높이빈 = 깊이 2.5m, 차체(h≈1.2m) elev 0.57°에서 = 깊이 10m, h≈1.6m(지붕)는 관측 불능. BEVHeight 저자 스스로 "ego-vehicle에선 depth보다 2% 뒤짐"이라 명시, BEVHeight++는 depth를 도로 합침. → **가파른 ray(높이 게이트) + 수평선 부근(inverse-depth 게이트) 하이브리드** + **게이트 플로어 (1−α)+α·p** (곱셈 게이트가 stale extrinsic에서 올바른 feature를 0으로 굶기는 증폭 문제: 20° stale pitch에서 40m 샘플 높이 오차 14m). 주장 자체가 DFA3D 사분면 결과(미완) 위에 서 있으므로 **DFA3D full VP/CTS 완료 전 방법 절 집필 금지**. |
| C4 자기보정 | **헤드라인, 구조 수정** | 아래 §3. |
| C5 지면 refinement | **컷** | 앵커가 필요한 곳에서 ill-conditioned: ∂d/∂pitch = d²/h = 17.5m/° (40m, h=1.6m) → coarse 스윕 잔차 ±2°에서 무의미, 0.5m-AP급은 0.03° 필요. 손실/구조 미정의로 padding 인상. cross-view 게이팅 절반만 C3의 bev_mask 정규화에 합류. |
| "조건부 분산" 규칙 | 격하 | n=2(같은 BN(27)+MLP 혈통 = 사실상 n=1) + BatchNorm 병리와 미분리 → "설계 법칙"이 아니라 "실패 사례 보고+가설"로. 살리려면 통제 2×2 필수: {무조건 / rig-벡터+BN / rig-벡터+고정정규화 / ray-조건} 동일 아키텍처 단일 rig 학습 → CTS-CAL 비교. ray-조건이 고정정규화 rig-조건을 이겨야만 규칙으로 승격. |
| "자기보정 거의 공짜" | 폐기 | all-cam 분리형 스윕만 저렴(3축×11=33 인코더 재실행). per-cam 프로토콜은 18차원 → 좌표하강 ~200–400회/프레임 — 2자릿수 차이. 런타임 표로 대체. |
| GaussianLSS 선례 | 재포지셔닝 | calibration.py는 업스트림에 없음(확인) = 외부 선례가 아니라 **팀 내 프로토타입**. 정직하게 "우리의 예비 실험"으로 기술 — 오히려 깨끗함. |

## 2. 점유되지 않은 교차점 (novelty 주장 가능 범위 — 문헌 검증 완료)

이미 점유된 것: 위협 모델 자체(NVIDIA ICCV'23 = 철학적 원조, NVS 증강 해법),
CTS-CAL 설정(CamShift IV'25, UniDrive ICLR'25, CoIn3D CVPR'26 — 3회 출판),
"backward/샘플링형이 가장 강건" 랭킹(CamShift), 가상 카메라 워핑(UniDrive,
BEV-LaneDet), homography extrinsic 증강(DG-BEV), 해석적-PE-대체(EAFormer),
높이 파라미터(BEVHeight/++/CoBEV/HeightFormer/CHARM3R), depth-게이트(DFA3D),
Δ-extrinsic 회귀+합성 탈보정 학습(LCCNet/CalibNet — 단 LiDAR 앵커),
회전 추정의 워프 합성 지도(GeoCalib/Perspective Fields — 단 단일 이미지),
pose-then-perceive(CLGo — 단 monocular lane), BEV 공간 calibration 디코딩
(BEVCalib — 단 LiDAR-camera 등록).

**비어 있는 것(검증됨)**: ① **stale-extrinsic 위협 모델 그 자체** — RoboBEV/
RoboDrive 전 계열에 extrinsic 탈보정 corruption도, 테스트타임 보정 해법도
부재(확인); CamShift/UniDrive/CoIn3D 모두 올바른 calibration 가정. ② **카메라-only
테스트타임 extrinsic 보정을 BEV detector 안에서, BEV 공간 스코어링으로, 물리적
정합 재렌더링 평가로** 하는 조합. ③ CAPE 코드 발견이 주는 기제 정당화 — extrinsic
오차는 이미지측에서 관측 불가(key PE가 intrinsic-only)이므로 **스코어링은 BEV
공간이어야 한다**는 구조적 논거. ④ frozen detector plug-in 일반성(여러 패러다임).

**가장 위험한 경쟁자**: CoIn3D(CVPR'26 — Plücker ray+지면 depth 조건 = 구 C3와
충돌, 3DGS 렌더링 증강이라 "무렌더링" 차별화는 가능하나 약함), CamShift(IV'25 —
우리 CTS 설정+랭킹 선점), dCAP(CVPR'26 — 트레일러 한정이지만 "추정 pose를
BEVFormer에 주입"을 선점, CVPR급 수요 증명).

## 3. 최종 방법 스펙 — "BEV-공간 테스트타임 rig 자기보정" (가칭 BEV-SelfCal)

**전제**: frozen 사전학습 BEV detector(투영 샘플링 계열이면 무엇이든 — BEVFormer/
DETR3D/DFA3D에 plug-in 시연), 카메라-only, 단일 rig normal 학습 데이터.

**(M1) coarse: rig-수준 ΔR 가설 스윕** — backbone feature 1회 추출·캐시(투영
샘플링 모델은 feature가 extrinsic-독립: 코드 검증 사실, C1을 컷했으므로 이 전제
유지). 축 분리(yaw/pitch/roll × 11 후보 = 33) BEV 인코더 재실행, 후보별 BEV를
**학습된 스코어러**가 평가. per-cam 모드는 좌표하강(카메라별 순차) — 비용 정직
보고. 스코어러 학습: homography-탈보정 + **Δ=0 표본 명시 포함** + 순수 외관
변화 표본(no-harm), margin-게이트 적용(스코어 이득이 τ 미만이면 무보정).

**(M2) fine: ΔR 회귀 헤드** — 입력: 다중 뷰 feature(+수평선 대역). 지도:
normal 이미지의 회전 homography(벤치마크 VP가 카메라 중심 순수 회전임이 검증돼
in-FOV에서 정확). **shortcut 차단 필수**: 워프 유효영역 내부 crop으로만 학습
(경계 밴드가 |Δ| 라벨을 누설 — 20° pitch에선 행의 58%만 생존하므로 대각도
지도는 구조적으로 제한), Δ=0 표본에도 동일 리샘플 체인 적용(블러 비정보화).
감쇠 적용(γ·Δ, γ≈0.7), 워프는 합성(재리샘플 금지), 스코어 비개선 시 정지(2회
상한). **재렌더링 held-out 검증 필수**(homography로 학습→재렌더로 평가, 축별
오차 보고 — border-shortcut 우려에 대한 유일한 답).

**(M3) 관측 가능성의 정직한 분해(논문의 분석 절)** — pitch/roll: 수평선·교차뷰
단서로 기하적 관측 가능. **all-cam yaw는 기하적으로 관측 불가**(상대 회전 불변,
pivot 잔차 ≤0.6m < BEV 셀; 회전된 장면도 유효한 장면) → yaw 회수는 장면 prior
(도로 정렬 통계) 추정임을 명시하고 축별 분리 보고. 이 정직성이 논문의 기하
기여(identifiability 분석)가 된다. translation은 범위 밖(시차 — homography
불가)임을 명시.

**(기판, 한 절)** C2: cams_embeds류 슬롯 파라미터 제거 + 해석적 ray/yaw 인코딩
대체(ablation: ray-PE vs CoordConv vs none). C3': DFA3D weight-path 게이트 +
하이브리드 파라미터(가파른 ray=지면높이/수평선 부근=inverse-depth) + 게이트
플로어 — **DFA3D 사분면 결과 확보 후에만 주장**. 해상도는 베이스라인과 동일
고정(해상도 confound 제거 — 1600×900 묶어팔기 금지).

**headline claim 형식**: "stale-extrinsic 시점 변화는 대부분 복구 가능한
calibration 오차다: BEV-공간 스윕+homography-학습 회귀가 N개 frozen detector
위에서 oracle-calibration 격차의 X%(축별 분해 제시)를 Y ms 오버헤드로 회수하며,
올바른 calibration에선 무해(false-correction rate Z%)하다."

## 4. 부족한 것 — 채택을 위한 필수 실험 목록 (검증 종합)

1. **DFA3D full VP/CTS** (다른 서버, 최우선) — C3' 주장의 통제 실험.
2. **homography-vs-재렌더 격차 정량화** — M2를 워프로 학습, 재렌더 held-out으로
   축별 평가. 이 숫자가 논문의 성패를 가른다.
3. **warp-consistent extrinsic 증강 베이스라인** — extrinsic jitter + 정합
   homography 워프(우리 제약 하에 합법인 최강 염가 경쟁자). 이걸 못 이기면 논문
   없음. (구 안티패턴 E.5는 이미지-불변 jitter만 배제했음 — 정합 워프 변형은
   별개.)
4. 베이스라인 패널: PD-BEV/DG-BEV(레포에 이미 있음), BEVHeight(-류 파라미터),
   UniDrive식 가상 카메라, 기성 온라인 calibration→frozen detector, Tent류 TTA,
   고전 기하(교차뷰 매칭/소실점 — 순수 회전이라 well-posed, zero-training).
5. **실데이터 leg** — nuScenes 학습 + extrinsic-noise 주입 프로토콜 + 교차
   데이터셋(nuScenes→Lyft/Waymo, PD-BEV 프로토콜). 자기 벤치마크 순환성의 유일한
   해독제. plug-in 일반성(BEVFormer/DETR3D/DFA3D frozen 3종)도 같은 역할.
6. no-harm 측정: CTS-CAL·clean에서 C4 on/off, false-correction rate, margin τ
   tradeoff 곡선, 멀티시드 오차막대(우리 벤치마크 자체가 0.0007 마진으로 순위가
   뒤집힘을 보였으므로 단일런 델타는 증거가 아님).
7. 런타임 표(베이스 forward vs +coarse vs +full; per-frame vs drift-트리거 배치
   정책), per-cam 좌표하강 비용 정직 보고.
8. C-1 규칙 통제 2×2(§1), 축×프로토콜별 회수 분해, 실패 사례(야간/무텍스처/스코어
   지형 퇴화), 결합 축 교란 1셀 이상, 사용 수치 frame-fix 이후 출처 감사 명세.

## 5. 과한 것 — 삭제/수정할 주장 (그대로 쓰면 리젝 사유)

- "0.425→0.779 헤드룸 회수" → **oracle 상한**임을 명시(CAL은 GT extrinsic 전체
  제공; 회수는 회전·관측가능 성분의 일부). oracle조차 pitch 0.665/roll 0.733에서
  멈춤(내용이 FOV를 떠남) — 상한의 구성을 축별로 제시.
- "CTS-CAL은 C3만으로 suv ~77/bus ~40 보장" → 미구축 모델에 대한 교차모델 상관
  외삽. "기대"로 강등 + DFA3D 결과로 대체.
- "yaw 거의 전부 회수(CAL-yaw 0.95)" → all-cam yaw는 관측 불가(§3 M3); 최대
  헤드룸 덩어리(0.527→0.938)가 prior-주도임을 명시.
- "학습·추론 동일 정준화", "pitch/roll을 입력단에서 해석적으로 제거" → C1 컷으로
  소멸.
- "자기보정 거의 공짜" → 런타임 표.
- "per-pixel ray 조건 = 일반화 가능한 조건부" → 공허(상수 함수) — 슬롯 대체
  인코딩으로만.
- "CAL에선 Δ≈0이라 무해" → 측정 항목으로 전환(자동이 아님 — Δ=0 학습 표본+margin
  게이트가 만들어내는 성질).
- "10개 모델 backbone이 extrinsic-독립 → 캐시 탐색" → 제안 모델 자신에게도
  성립하도록 C1 컷(모순 해소됨)을 명시.

## 6. 실행 순서 (사전 검증 → 집필)

1. DFA3D full VP/CTS (외부 서버; §4-1) → C3' 채택/폐기 결정.
2. 미니 파일럿(기존 frozen BEVFormer + M1만, all-cam 3축): 스코어러가 재렌더
   VP-IMG에서 회전을 실제로 찾는지 — **이 파일럿이 실패하면 논문 중단점**.
   (PitchBinClassifier가 pitch 1축에서 이미 동작 = 팀 내 예비 증거.)
3. M2 학습 + homography-vs-재렌더 격차 측정(§4-2).
4. warp-consistent 증강 베이스라인(§4-3) — 이기는지 확인.
5. 이상 통과 시: nuScenes leg + plug-in 3종 + 본 실험. 집필은 그 후.

---

## 7. 대안 프레이밍 검증(2차 정찰)과 최종 추천 — **추천 변경**

C4 단독안에 대한 대안 3개를 추가 검증했다(원본 `_alt_idea_vetting.json`).

**A. calibration-등변성 이론+감사+수리** — 가장자리 점유, **이론+감사+수리 패키지는
공백**. 구분 필수 대상: VEDet(CVPR'23, "viewpoint equivariance" 용어 선점 — 단
단일 rig 합성 시점에 대한 soft 학습 손실이지 rig 변환 하의 형식적 성질이 아님),
군-등변 BEV 검출기들(GeqBevNet/AeDet/TED — 군이 장면에 작용, calibration
메타데이터가 아님), Wang et al.(NeurIPS'23, correct/incorrect/extrinsic
equivariance 일반 이론 — 카메라에 인스턴스화된 적 없음 = 우리가 빌릴 수학),
EAFormer(한 모델짜리 증거), CoIn3D(경험적, 무형식). 우리의 10-모델 코드 감사가
그대로 본문 자산.

**B. frozen 파운데이션 metric depth를 rig-불변 앵커로** — **공백이되 창이 닫히는
중**. 양쪽에서 조여옴: arXiv:2501.08118(frozen Metric3Dv2를 LSS에 — 단 seg,
in-dist, 데이터 효율 동기, rig 일반화 주장 전무), Hashimoto arXiv:2604.00597
(frozen DA3로 시점 강건 E2E 플래닝 — **"BEV/voxel 표현을 파운데이션 기하로 만들어
extrinsic에서 분리"를 future work로 공개 명시** = 우리 프레이밍이 공개적으로
예고된 다음 수). CHARM3R(ICCV'25)는 전제(depth가 높이 병목)를 독립 입증해주는
지원군. 가용 주장: "frozen 카메라-조건 metric-depth 파운데이션을 다중 카메라 BEV
검출기의 depth 앵커로 — 통제된 플랫폼 전이에서 최초 평가". **속도가 생명.**

**C. calibration 소비 방식 통제 연구** — **핵심 공백**. Simple-BEV(ICRA'23)가
"단일 변수 비교" 템플릿 선점(단 in-dist 정확도만), RoboBEV는 관찰적·교란,
CoIn3D(CVPR'26)가 최예리 위협(Plücker ray 조건을 해법으로 씀 — 단 교란된 method
논문, 기제 진단 없음). 가용 주장: "동일 backbone/head에 calibration 소비 기제
{해석적 투영 / rig-벡터 조건(±정규화 수리) / ray 조건 / PE형 / depth-splat /
**frozen 파운데이션-조건 depth(=B를 6번째 암으로)**}만 교체, 단일 rig 학습,
인수분해된 교차 rig 평가 → 인과적 설계 법칙". 부가 발견: temporal stereo를
rig-불변 앵커로 평가한 사람도 없음(단 stale extrinsic이 cost volume 자체를
오염시키는 상호작용 미검토 — future work 한 단락감).

**최종 추천 (변경): 합본 프레이밍 — "BEV 검출기는 calibration을 어떻게 소비해야
하는가: 진단 → 법칙 → 처방".** C를 척추로, A의 등변성 조건을 조직 원리(이론층,
Wang et al. 기계를 카메라에 인스턴스화 — zero-variance 퇴화를 그 안에서 형식화)
로, B를 구성적 결론(normal-only 제약에서 metric 격차를 메우는 유일한 암)으로.
근거: (i) 세 공백을 하나의 일관된 서사로 동시 점유, (ii) **킬스위치 없음** — C4와
달리 어느 암이 이겨도 발견이 성립(연구 설계 자체가 산출물), (iii) 보유 자산
직접 화폐화(코드 감사=감사 절, 벤치마크=계측기, BEVDepth 포크+PD-BEV=암 구현,
nuScenes→Lyft/Waymo로 외부 타당성), (iv) AC가 요구한 통제 2×2가 부속 실험이
아니라 본론이 됨. **C4 자기보정은 2호 논문으로 보류** — 공백(RoboBEV 계열에
테스트타임 보정 부재)은 빨리 닫히지 않는 반면 B의 창은 닫히는 중.

**즉시 실행 (B 선행 검증, 1–2주, 기존 인프라)**:
1. *(1일)* frozen UniDepth/Metric3Dv2를 CARLA 이미지에 zero-shot 추론 → DPT GT
   대비 depth 품질을 sedan/suv/bus 높이별로 측정. **sim-to-real 역방향 위험**
   (파운데이션은 실사로 학습 — CARLA 합성에서 깨질 수 있음)을 여기서 판정.
   깨지면 합본 논문에서 B 암은 nuScenes 쪽 실험으로만.
2. *(1주)* BEVDepth 포크의 depth 분포를 frozen 파운데이션 출력(현재 카메라
   파라미터 조건)으로 교체 → sedan 재학습 → 기존 CTS 인프라로 suv/bus 평가.
   bus IMG 0.2 / CAL 16.6이 유의미하게 오르면 합본의 처방 암 확정.
3. 이후 C의 나머지 암들(동일 골격에 기제 교체) 순차 구축.

### 부기
- 출처: 문헌 4-에이전트(각 인용 venue/year 웹 확인) + 적대 기술 리뷰(기하 수치
  도출) + AC 시뮬레이션. 원본 `results/_method_paper_vetting.json` (103KB).
- 2차 정찰(대안 3개): `results/_alt_idea_vetting.json` (2026-06-11).
- 기술 리뷰의 핵심 검증 수치: sedan/suv rig 중력 정렬(elev 0.000°), VP 변형 =
  카메라 중심 순수 회전(translation Δ=0.0000 — homography 정확성의 근거이자
  "시차 잔차" 우려의 반박), 높이 파라미터 수평선 퇴화(0.1m 빈 = 2.5–10m 깊이),
  ∂d/∂pitch = d²/h, all-cam yaw 관측 불가 논증.
- 벤치마크 논문(§5)과의 관계: 분석 발견(BN 포화·슬롯 prior·부호 진단)은 벤치마크
  논문 Discussion에 남고, 본 method 논문은 그 발견을 인용하며 C4를 전개 —
  두 논문의 분업이 자연스러움.
