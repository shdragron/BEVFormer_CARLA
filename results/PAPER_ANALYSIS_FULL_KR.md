# GeoBEV — 논문 구성대로 전체 종합 분석 (seg + det 통합)

> 대상 논문: **GEOBEV: Towards Camera-Geometry Robustness in Multi-Camera BEV Perception** (WACV 2027, **Datasets Track**).
> 현재 구조: §1 Intro → §2 Related(2.1 Viewpoint robustness, 2.2 Camera-config generalization) → **§3 GeoBEV**(3.1 design / 3.2 VP protocol / 3.3 CTS protocol / 3.4 conditions NORMAL·EXT·IMG·CAL / 3.5 metrics IoU·SDS·mVRS·CTS) → §4 Setup(7 seg + 6 det baseline) → **§5 Discussion & Limitations** → §6 Conclusion → Supp §7–11.
> **메인 결과 = Table 2** (Task1 BEV-Seg + Task2 3D-Det, 각각 VP mVRS[NORMAL/EXT/IMG/CAL] + CTS[suv/bus, IMG/CAL]).
> **현재 빈 곳**: Abstract L23 "(TODO fuller analysis)", DSPE 결과 dashes, PETRv2·DFA3D 미완, Supp §7–11 stub, §5 Discussion이 분석의 자리인데 빈약.

이 문서는 "Table 2 숫자를 §5에서 어떻게 분석으로 풀지"를 논문 섹션에 매핑한다.

---

## 0. 한 줄 요약 (논문이 말해야 할 것)

> **GeoBEV의 두 축(VP=in-platform viewpoint, CTS=cross-platform)은 서로 독립된 실패 모드를 드러내며, 검출에서는 그 실패가 "extrinsic이 feature 샘플링을 게이트하는가"(VP)와 "depth에 의존하는가"(CTS)라는 두 아키텍처 성질로 갈린다. 세그멘테이션은 이 분기가 약하지만 IMG-붕괴·yaw-취약·bus-붕괴라는 공통 지문을 공유한다. 기존 robustness 기법(extrinsic aug / PD-BEV / EAFormer)으로도 이 격차는 닫히지 않으며, 빈 사분면(gates×depth)을 채우는 DFA3D도 한 축만 개선한다.**

---

## 1. §3.4 (conditions) 를 뒷받침하는 핵심 관찰 — "EXT ≠ IMG"

논문의 1차 주장("extrinsic-only 평가는 실제 viewpoint 저하를 반영하지 못한다")을 **숫자로 정밀화**한다. all-cam mVRS 기준:

| | EXT | IMG | EXT−IMG 격차 |
|---|---|---|---|
| **det / extract-then-place** CAPE | 0.811 | 0.407 | **+0.40** |
| BEVDepth | 0.652 | 0.328 | +0.32 |
| BEVDet | 0.610 | 0.364 | +0.25 |
| **det / gates-sampling** DETR3D | 0.438 | 0.422 | +0.02 |
| BEVFormer | 0.428 | 0.426 | +0.00 |
| **seg** GaussianLSS | 0.698 | 0.453 | +0.25 |
| CVT | 0.690 | 0.416 | +0.27 |
| SimpleBEV | 0.539 | 0.242 | +0.30 |

- **논문 주장 강화**: "extrinsic-only(EXT)가 viewpoint 저하를 반영한다"는 가정은 **거짓** — 거의 모든 모델에서 EXT ≫ IMG. 단 **검출의 gates-sampling(BEVFormer/DETR3D)만 EXT≈IMG**로 예외다. → §3.4가 도입한 EXT/IMG 분리가 *왜 필요한지*를 정량적으로 증명.
- **이게 §5 Discussion의 1번 finding이 되어야 함**(현재 Abstract TODO 자리).

---

## 2. §5 Discussion 강화 — 5개 finding 제안

### Finding A. VP는 "메커니즘"으로 갈린다 (검출, code-verified)

- **extrinsic-gates-sampling**(BEVFormer=deformable cross-attn, DETR3D=grid_sample): extrinsic이 3D query→이미지 투영·샘플링을 *게이트* → **EXT≈IMG**, CAL에서 회복(CAL 0.78/0.85).
- **extract-then-place**(BEVDet/BEVDepth=LSS splat, CAPE=camera-view PE): backbone이 extrinsic-무관하게 feature 추출 후 기하 적용 → **EXT≫IMG**, CAL-pitch collapse(0.13–0.25).
- **CAPE가 결정적**: explicit depth가 없는데도 LSS depth 모델과 같은 VP 지문 → **VP 분기는 depth가 아니라 메커니즘**. (5-에이전트 코드 감사, HIGH confidence.)
- → 표/그림: **mechanism × depth 2×2** (아래 §3).

### Finding B. CTS는 "depth 의존"으로 갈린다 (VP와 독립 축)

- explicit/categorical depth(BEVDet/BEVDepth)는 **bus-IMG에서 0.001–0.002로 붕괴** — sedan 마운트로 학습된 monocular depth가 높은 bus 시점에서 틀린 깊이 예측 → feature가 잘못된 BEV 거리로 lift.
- **depth-free CAPE가 최강 전이체**(CTS-IMG 0.349, bus 0.360≈suv 0.338) — VP에서는 extract-then-place인데 CTS에서는 최고. **두 축이 독립**임을 한 모델로 증명.
- TP 지문: 생존 검출도 depth 모델은 *크기*가 틀림(mASE 0.81–0.87 vs depth-free 0.31–0.34).
- → CTS IMG 순위: **CAPE 0.349 > DETR3D 0.307 > BEVFormer 0.276 ≫ BEVDet 0.086 > BEVDepth 0.052**.

### Finding C. seg ↔ det 통합 — 공통 지문 + 분기 차이 (이게 "두 task" 논문의 핵심)

**공통 (seg=det 모두):**
1. **IMG가 보편적으로 가장 어려움** — seg mVRS-IMG all-cam 0.24–0.45, det mAP-retention 0.14–0.19. 기운 이미지가 task 무관하게 검출/분할 recall을 붕괴.
2. **all-cam이 변별력** — per-cam은 ~0.9로 압축(6-view 융합이 한 카메라를 outvote). mVRS 헤드라인(1/7 가중)이 차이를 가림 → **분석은 all-cam으로**.
3. **IMG-yaw가 cross-camera correspondence를 깸** — seg IMG-yaw 0.26–0.31(전 모델 일관, 가장 낮음), det도 동일. CAL-yaw는 회복(seg 0.78–0.95, det 0.92–0.98): 일관된 yaw는 수직축 전역 회전이라 inter-camera 기하 보존.
4. **bus(높은 마운트) CTS 붕괴** — seg bus 0.01–0.22(suv보다 훨씬 심함), det bus-IMG 0.001–0.002.

**차이 (seg에는 메커니즘 분기가 약함):**
- 검출은 extract/gates가 **EXT≫IMG vs EXT≈IMG로 선명히 갈리고** EXT-worst-axis도 갈림(extract=yaw, gates=pitch). **세그멘테이션은 이 분기가 거의 없음** — 모든 seg 모델이 중간(EXT 0.62–0.82, IMG 0.42–0.45). seg는 dense pixel supervision이라 sparse-query 검출의 sampling-gate 구분이 흐려짐.
- → 논문 메시지: *"두 task가 IMG-붕괴·yaw-취약·bus-붕괴를 공유한다(벤치마크의 일반성)지만, 아키텍처 분기는 검출에서 가장 선명하다"*.

### Finding D. per-axis — yaw vs pitch의 역할 분리

- **IMG all-cam worst axis**: det=**pitch**(0.14–0.26 uniform), seg=**yaw**(0.26–0.31 uniform). 둘 다 "지면/대응 깨짐"이 원인이나 task별로 지배 축이 다름 → per-axis breakdown 표(Table 2 또는 Supp §9) 정당화.
- **EXT worst axis (검출 메커니즘 지문)**: extract-then-place=yaw(0.43–0.50), gates-sampling=pitch(0.17–0.25). → 메커니즘을 *축 수준*에서 재확인.

### Finding E. 기존 robustness 기법으로도 안 닫힌다 (신규 실험 — §5 또는 별도 §)

- **부류별 대표**: train-time **extrinsic aug**(seg/det 공통), DG 검출 **PD-BEV**(BEVDepth 위), calibration-free **EAFormer**(CVT 위).
- **예측(우리 메커니즘 기반)**: extract-then-place의 CAL-pitch/roll·held-out은 aug로 안 고쳐짐(추출에 박힌 기울임); depth-기반 DG는 depth-free 전이 수준 못 미침.
- **평가 원칙**: aug는 **default 범위**로(테스트 분포에 맞추면 train-test 겹침→무의미). PD-BEV default(yaw±23°)가 우리 yaw 테스트를 덮으므로 **held-out(pitch/roll/CAL/CTS)을 핵심 증거**로, yaw는 각주.
- → 표: 앞서 만든 1-column robustness-baseline 표(seg=IoU / det=SDS 섹션).

### Finding F. DFA3D — 빈 사분면(gates × depth) (신규 7번째 검출기)

- 메커니즘×depth 2×2에서 **gates-sampling × uses-depth**가 비어 있음. DFA3D(BEVFormer + depth-aware 3D deformable attention)가 이를 채움 = **BEVFormer 대비 단일변수(depth) 비교**.
- 예측/검증 포인트: VP에서는 gates 지문(EXT≈IMG, CAL 회복) 유지하면서, depth 추가가 **CTS를 개선하는가**(축 독립성의 인과 검증). sedan oracle val NDS 0.495(baseline 0.50 수렴), suv/bus oracle 학습 자동 체인 진행 중.

---

## 3. 권장 표/그림 (Table 2 + 보조)

- **Table 2 (메인)**: 현행 유지(seg 7 + det 6) + **all-cam 컬럼 강조**(per-cam은 차이를 가림). DFA3D 행 추가(det, Backward).
- **Fig (mechanism 2×2)**: 행=gates/extract, 열=depth-free/uses-depth. BEVFormer·DETR3D | — / CAPE | BEVDet·BEVDepth / (DFA3D가 우상단 채움). VP는 행으로, CTS는 열로 갈림을 시각화.
- **Fig (correlation)**: EXT vs IMG 산점도(all-cam ρ=−0.70 anti-corr; 1/7은 +0.10로 마스킹) + CTS EXT→IMG→CAL trajectory.
- **Table (robustness baselines)**: §Finding E, 1-column, seg(IoU)/det(SDS) 섹션, EXT/IMG/CAL + suv/bus.
- **Table (per-axis)**: Supp §9, IMG-pitch(det)/IMG-yaw(seg) 대비.

---

## 4. 현재 비어있거나 확정 필요한 것 (제출 전 체크리스트)

1. **Abstract L23 "(TODO fuller analysis)"** → Finding A/B/C 한 문단으로.
2. **PETRv2** 미평가(예측 extract-then-place, 미검증) — 학습/평가 필요.
3. **DFA3D** sedan 학습 중(~수 h), suv/bus 자동 체인 → VP/CTS.
4. **DSPE** Table 2 dashes — 포함/제외 결정.
5. **BEVDet CTS** results vs 표 ~2× 차이 — reconcile(현재 suv-IMG 0.170/bus 0.002로 일치 확인, 단 재확인 권고).
6. **seg CTS는 raw IoU만** — target-oracle(suv/bus-trained seg) IoU 없으면 **CTS-ratio 불가** → det과 같은 ratio로 맞추려면 seg oracle 필요(없으면 raw IoU로 보고 + 각주).
7. **VP 768-subset** → full-val 검증 완료(BEVFormer Δ≤0.004, subset 대표성 확인); 다른 모델도 동일 가정 명시.
8. **SimpleBEV EXT-yaw 0.953 이상치** — seg per-axis에서 단일 셀 확인.

---
*출처: results/{BENCHMARK_SUMMARY, VP_CROSS_MODEL_ANALYSIS, PAPER_ANALYSIS_KR, SEGMENTATION_RESULTS}.md + seg_vp_cts.tsv + WACV pdf. 메커니즘 [[vp-cross-model-mechanism-finding]], DFA3D [[dfa3d-carla-setup]].*
