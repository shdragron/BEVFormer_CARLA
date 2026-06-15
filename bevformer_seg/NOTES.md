# BEVFormer vehicle-occupancy seg — port notes (2026-06-14)

목표: Table 2 seg행의 빈 "BEVFormer (Backward-projection)" 채우기.
규격(선택 A, 다른 seg 모델과 공정 비교): **R50 ImageNet + 224×480 + BEV 200×200
(±50m, 0.5m/cell) + vehicle binary occupancy**, GaussianLSS GT, IoU@0.5.

## 검증으로 잠근 사실 (학습 전 필수 — 둘 다 위험점)

1. **vehicle = `(bev_png >> 4) & 1`** (bit4), NOT bit0.
   - 매핑 에이전트가 "bit0"이라 했으나 **틀림** — bit0/bit1 = road(10533px, 큰 연결영역).
   - 검출 GT 박스로 8프레임 교차검증: bit4 = px/veh ≈40-50(차 1대=36-50px), 적중수가
     차량 대수에 비례(7→6, 18→16, 20→17). bit5는 무관(0-4 적중).
   - **검증 안 했으면 도로를 차량으로 학습할 뻔.** verify/verify_alignment.py.

2. **lidar2img = `viewpad(I_rescaled) @ E`** (GaussianLSS-exact, E = ego2cam).
   - intrinsic rescale: I[0,0]*=480/1600, I[0,2]*=480/1600, I[1,1]*=270/900,
     I[1,2]*=270/900, I[1,2]-=46 (top_crop). 검증: 정면 12m점 → CAM_FRONT 중앙(u=240).
   - sensor2lidar 왕복 불필요(E가 이미 lidar2cam).

3. **GT 레이아웃 정합**: `mask_bevformer = mask_gauss[::-1,::-1].T`
   (= mask_g[199-c,199-r]). GaussianLSS row=lidarX↓/col=lidarY↓ → BEVFormer
   row=lidarY↑/col=lidarX↑. 수치 검증(glue 에이전트) + 시각 확인(블롭이 차량).
   단 BEVFormer seg head의 permute/flip이 "empirically tuned"이라 최종 정합은
   학습 중 pred-vs-GT viz로 재확인 필요.

## 이미지 전처리 (GaussianLSS-exact + R50용 추가)
- PIL resize 1600×900 → 480×270 BILINEAR → crop(0,46,480,270) → 224×480.
- **GaussianLSS는 ImageNet norm 안 함([0,1] ToTensor만)**. R50 ImageNet은 norm 필요
  → mean[123.675,116.28,103.53] std[58.395,57.12,57.375] to_rgb 추가(수치 편차 명시).

## 데이터/스플릿
- 이미지 루트: carla_geobev (sweeps/RGB-CAM_*/...jpg).
- 학습 라벨: gaussianlss/sedan (220 scene). eval: gaussianlss/sedan_eval
  (48 scene × 79 = 3792프레임 = 검출 val셋과 동일).
- split: GaussianLSS/data/splits/carla/{train(255),val(63)}.txt.
- visibility png {1,2,4,255}: 손실/평가 마스킹용(vis>=2), GT엔 안 베이킹.

## 남은 빌드
1. mmdet3d Dataset 클래스(GaussianLSS json 직접 읽기, 위 3사실 적용) — 진행 예정.
2. seg head: SegEncode outC 4→1, loss CE(softmax,4cls)→BCE+Dice(sigmoid,1ch),
   GT long→float, eval num_map_class→2 threshold 0.5 class1 IoU.
3. config bevformer_seg_r50_224x480_bev200_veh.py: R50(depth50,pytorch,no-DCN,
   with_cp F,load_from None,RGB norm), bev_h_=bev_w_=200, pc_range ±50,
   pos_enc 200, queue_length 1(single-frame), det_grid==map_grid(cropper identity).
4. 학습(R50 seg, 반나절급) → pred-vs-GT viz로 정합 최종확인 → Table2 seg행.

## 미해결 결정 (사용자 판단 필요)
- **seg VP/CTS 평가 하니스**: 다른 seg 모델(CVT/LSS/GaussianLSS)의 VP/CTS는
  bevunify/GaussianLSS 하니스(eval_results/bevunify-*)로 돌았음. BEVFormer-seg도
  (a) 그 하니스에 얹을지, (b) 우리 bev_det_benchmark에 seg VP/CTS를 새로 짤지.
  in-dist IoU(Table2)는 위 빌드로 바로 가능, VP/CTS는 이 결정 후.
