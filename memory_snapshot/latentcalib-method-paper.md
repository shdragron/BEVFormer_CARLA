---
name: latentcalib-method-paper
description: "2호 논문(method) 확정안 LatentCalib — calibration을 잠재변수로, frozen 파운데이션 depth 투영-일치도 A(Δ) 하나로 자기보정/게이트/뷰신뢰도. 프로젝트 폴더, 검증 사다리, 진행 상태."
metadata: 
  node_type: memory
  type: project
  originSessionId: a8f76c13-f85f-4d13-a229-6878bbff6a20
---

**LatentCalib (가칭, 2026-06-11 확정)** — 벤치마크 논문([[vp-cross-model-mechanism-finding]])과
별도의 method 논문. 폴더 `/home/hanyan_arch/viewpoint/LatentCalib` (자체 git, identity
shdragron). 확정 스펙 = BEVFormer 레포 `results/METHOD_PAPER_SPEC_KR.md` **§8**(이전 §0/§7
권고를 대체; C1 정준화·C5 지면앵커는 검증에서 컷됨). 검증 원본 3개:
results/_method_paper_vetting.json, _alt_idea_vetting.json, _arch_geometry_survey.json.

**아이디어 한 줄**: calibration을 신뢰 입력이 아니라 프레임별 잠재변수로 — 중심 연산은
투영 일치도 **A_i(Δ) = E[D_i(u_i(Δ), d_i(Δ))]** (D=frozen 파운데이션 depth(이미지에서,
K-조건), (u,d)=가설 calibration T·Δ로 투영한 BEV-pillar 샘플의 해석적 좌표/깊이;
grid_sample이라 Δ 미분가능). 한 함수 3기능: ①Δ-추론(자기보정; coarse 후보+amortized 회귀;
학습신호=**이미지 무손상 calib-jitter 역산쌍**→homography border-shortcut 결함 회피, 앵커가
카메라별 절대신호라 all-cam yaw 관측불능도 회피) ②attention 가중치 게이트(플로어 α, splat
금지) ③뷰 신뢰도(잔여 일치도로 hit-count 다운웨이트). 구조: frozen DINOv2 공유 백본(검출
neck+depth 분기 — 비용 상쇄+backbone OOD 완화), 해석적 투영 샘플링(slot 파라미터 제거),
stock DETR decoder. 학습=단일 rig normal 이미지+Δ-역산만(depth GT/LiDAR/렌더링 불필요).
올바른 calib면 Δ≈0 구조적(무해성 내장). intrinsic 확장: canonical-K 리샘플(시차 없어 정확,
extrinsic 정준화와 달리 유효)+파운데이션 K-조건 → nuScenes→Waymo leg 가능. "훈련만 depth"
불가(증류 student는 단일 rig로 rig prior 재학습; 앵커는 정의상 테스트타임).

**contribution 3**: ①calibration-잠재변수 view transformer(첫), ②frozen 파운데이션 depth의
rig-불변 앵커(첫; "일반화는 출력이 아니라 가중치에 산다" — BN(27) 19–784σ 포화 법칙+슬롯
prior 목록은 코드 검증), ③통제 증거(같은장면 플랫폼 전이+stale/corrected 인수분해+frozen
검출기 training-free plug-in+nuScenes→Waymo).

**검증 사다리(킬스위치)**: ①파운데이션 CARLA zero-shot 품질 vs DPT GT(1-2일, 학습0; 깨지면
B-앵커는 nuScenes로만) → ②A(Δ) 지형 관문(frozen BEVFormer+파운데이션, VP-IMG 셀에서 진짜
Δ에 정점?; 평평=중단→통제연구 전환) → ③training-free plug-in(argmax-Δ 보정→VP-IMG 회복;
단독 헤드라인감) → ④full 모델+소비기제 3-4암 ablation → ⑤nuScenes→Waymo.

**경쟁/마감 압박**: Hashimoto arXiv:2604.00597이 우리 방향을 future work로 공개 명시(창
닫히는 중); CoIn3D(CVPR'26)·CamShift(IV'25)·UniDrive(ICLR'25)는 CTS-CAL 선점이나 stale-
extrinsic+테스트타임 보정은 전부 공백(RoboBEV/RoboDrive 계열에도 없음 — 확인됨).

**실험 결과 확정(2026-06-12, 전부 학습 0·frozen BEVFormer·632프레임 whole-scene 서브셋)**:
rung③ 12셀 — **roll 전 크기(±4~16) recovery 100%**, pitch 크기 비례(−16: 붕괴 0.000→
천장 0.566 100%, ±4 코스 0%). fine stage((δ×s) 0.5° 공동, 캐시 depth): pitch−8
50%→**105%**(selfcal 0.814 > caltrue 0.787 — 검출기-최적 calib ≠ 물리 참값 관찰),
+4 0→25%, +8 21→34%; **전체 평균 79%**. 잔존: +pitch(하늘쪽) 2.5–3° 편향 = 지면-대역
샘플링 관측 한계(처방: 교차뷰 항/구조 가중), s가 1.10 경계 ride(scale prior 필요).
**yaw 3종 세트**: ①지형 평탄=불식별(예측 적중) ②쓰레기 δ̂ 맹목 적용 시 0.547→0.298
악화 = margin τ의 실측 근거 ③**게이지 실험 입증: 예측 박스 −8° 사후회전만으로 RRS
0.547→0.867 (천장 0.897의 91%)** — common yaw 손상=좌표 게이지 오류, 지각은 무사.
주의: BEVFormer test.py가 jsonfile_prefix를 무시하고 test/<cfg>/<ctime>/에 예측 저장.
**exp05 no-harm 측정 완성(06-12)**: 올바른 calib(sedan normal+suv/bus CTS-CAL)에서
(δ,s) joint 스윕 — granularity 기울기가 핵심 발견: 카메라·단일프레임 fc@τ.05=2–27%
(가짜 5–10°, margin 큼) / 프레임-rig 0–10% / **풀링-rig(배포 granularity) 전 플랫폼·
전 축 Δ*=0·margin=0·false-correction 0%**. 가짜 정점은 캠·프레임 간 비정렬→집계가
으깸. 설계 규칙 확정: "보정은 rig-수준·다중프레임 집계에서만"(드리프트-트리거).
측정 주장 4종 세트 완성: 식별축 회복(roll 100%/pitch 105%)+불식별축 보호(yaw 맹목
악화 0.547→0.298)+게이지(91%)+무해(0%).
**exp07(06-12) 의무 베이스라인 완료 — arbiter 압승**: dCAP식 직접 회귀(동일 증거·
동일 jitter 지도, v1 GAP / v2 CoordConv 강화판) — v2가 합성 적합은 2.7배 좋은데
(L1 0.28°) 실제 재렌더에선 출력≈0(오차=교란 자체), OOD ±16 포화(~16°), 올바른
calib에서 2.0–6.8° 가짜 출력(기각 기제 없음). 핵심: 강할수록 전이 악화 = 약점은
용량이 아니라 합성 학습 분포 의존성; argmax-A는 기하 함수라 면역(전 크기 정확,
margin으로 0). 정직 단서: same-evidence jitter 클래스이지 dCAP 완전 재현 아님.
**1차 사이클 최종(06-12): 기하 식별 가능 12/12 셀 recovery 100%** — xview δ̂(A_depth+
A_xview, 1° 격자)가 pitch 전 크기·전 방향에서 참값 정확 → SelfCal NDS=CALtrue NDS.
coarse→fine→+xview 경로가 ablation 서사(BEV항만=소각도 편향 / +scale joint=부분·
**scale prior는 음성 판정**(λ=0.3, s만 길들고 δ̂ 편향 불변) / +교차뷰=완성). 헤드라인
그림 docs/results_recovery.png(12/12 천장 도달). **DFA3D 사분면 합류(외부 서버,
results/DFA3D)**: VP 1/7 90.4/86.1/93.2 — depth 게이트가 온-rig 강건성 더함(부호 +2.8
유지, 내장 outlier rejection); CTS suv 68.7/38.8/47.5·bus 37.1/28.8/48.1 — splat 붕괴
면함(bus 28.8)이나 단일-rig 학습 depth가 게이트 오염(suv CAL−EXT −21.2 부호반전) =
"경로는 좋고 출처가 문제" → frozen 파운데이션 교체 설계의 motivation 데이터.
**exp06(06-12): per-cam yaw 교차뷰로 식별 확정** — A_xview(이웃 파운데이션 depth
점군=rig-앵커 샘플, 동일 커널)가 yaw±8에서 argmax 정확히 참값(대비 0.67 vs 0.5);
A_depth는 평탄(예측대로). yaw 3분할 완성. 종합 문서 docs/RESULTS_V1.md + 헤드라인
그림 docs/results_recovery.png. **남은 의무 실험: dCAP식 직접 회귀 베이스라인
(학습 필요; 판별=OOD jitter+false-correction), +pitch 소각도 관측성(교차뷰 항을
pitch에도), scale prior, 시드/오차막대, 그리고 full 모델(아키텍처 다이어그램
docs/architecture.png) 학습.**

**상태(2026-06-12 새벽, 사다리 ①✓②✓③진행)**: env=latentcalib(torch 2.11+cu128,
UniDepth ViT-L; conda activate 비신뢰 → ENVPY 직접 호출 패턴). **게이트① PASS**:
UniDepth zero-shot CARLA 3플랫폼 — AbsRel .17-.20, δ1 .76-.81, med_ratio
+4.5%/−7.4%/−7.0%(sedan/suv/bus) — 높은 마운트 붕괴 없음, 단 ±7% 플랫폼 스케일
편향 → (Δ,s) 공동최적화+scale prior 필요. **게이트② GO**: A(Δ) 지형(frozen, 학습0,
6프레임) — roll 완전 식별(오차 0-2°), pitch 단봉(0-4° 편향, 방향 비대칭), yaw
불식별(예측 적중→prior 경로), (δ×s) 슬라이스: δ 식별·s ridge. **rung③ 스모크:
pitch−8 recovery 50%** — frozen BEVFormer에 δ̂=−10(참−8) 꽂아 RRS 0.237→0.514
(천장 0.787); 잔차 2°가 0.27 RRS 비용 → δ̂ 1° 정밀화 여지. 전체 12셀(pitch/roll
×±4/8/16) 실행 중. 아키텍처 다이어그램(docs/architecture.png) 검토 반영판 커밋:
a_k 기호 가족, margin τ, A-증거 전용 회귀 입력, 무해성=측정 항목. 클레임 v2 +
conjunct 점유자 표 + Hashimoto 사용규칙 + re-scout 트리거 = docs/CLAIM_AND_
POSITIONING.md. 평가 인프라: exp03 run_plugin.py(whole-scene 8씬 632프레임 서브셋,
IMG/SELFCAL/CALTRUE 3조건, dist_test 서브프로세스+[CARLA-EVAL] 스크레이프, 태그
캐시 resume).
