# 사출성형기 데이터 변수 상관관계 요약

## 데이터 점검

- CN7 라벨 1,211건(불량 17건, 1.40%), 비라벨 35,239건
- RG3 라벨 1,182건(불량 25건, 2.12%), 비라벨 35,941건
- 네 CSV 모두 결측 셀과 완전 중복 행이 없다.
- 라벨 데이터의 `Clamp_Open_Position`은 상수열이므로 라벨 연관성 계산에서 제외했다.
- 불량 표본이 매우 적어 라벨 연관성 수치와 p-value는 탐색적 근거로만 사용해야 한다.

## CN7 주요 변수쌍 (Pearson)

- `Mold_Temperature_3` ↔ `Mold_Temperature_4`: +0.995
- `Plasticizing_Time` ↔ `Average_Screw_RPM`: +0.984
- `Clamp_Close_Time` ↔ `Cushion_Position`: +0.978
- `Max_Screw_RPM` ↔ `Average_Screw_RPM`: +0.975
- `Cushion_Position` ↔ `Average_Screw_RPM`: +0.973
- `Cushion_Position` ↔ `Max_Back_Pressure`: +0.973
- `Plasticizing_Position` ↔ `Max_Switch_Over_Pressure`: +0.971
- `Plasticizing_Position` ↔ `Mold_Temperature_3`: +0.971

## CN7 불량 라벨 연관 변수

- `Mold_Temperature_3`: point-biserial +0.145, Spearman +0.161, 방향보정 AUC 0.894, FDR q=1.28e-06
- `Mold_Temperature_4`: point-biserial +0.163, Spearman +0.160, 방향보정 AUC 0.892, FDR q=6.61e-08
- `Plasticizing_Position`: point-biserial +0.179, Spearman +0.154, 방향보정 AUC 0.870, FDR q=4.49e-09
- `Max_Back_Pressure`: point-biserial -0.176, Spearman -0.144, 방향보정 AUC 0.852, FDR q=5.28e-09
- `Average_Back_Pressure`: point-biserial -0.160, Spearman -0.142, 방향보정 AUC 0.845, FDR q=9.97e-08
- `Plasticizing_Time`: point-biserial -0.051, Spearman -0.126, 방향보정 AUC 0.809, FDR q=0.123
- `Clamp_Close_Time`: point-biserial +0.143, Spearman +0.137, 방향보정 AUC 0.791, FDR q=1.44e-06
- `Cushion_Position`: point-biserial -0.152, Spearman -0.130, 방향보정 AUC 0.767, FDR q=4.48e-07

## RG3 주요 변수쌍 (Pearson)

- `Mold_Temperature_3` ↔ `Mold_Temperature_4`: +0.989
- `Clamp_Open_Position` ↔ `Barrel_Temperature_1`: -0.980
- `Cushion_Position` ↔ `Clamp_Open_Position`: -0.977
- `Clamp_Close_Time` ↔ `Cushion_Position`: +0.974
- `Clamp_Close_Time` ↔ `Clamp_Open_Position`: -0.973
- `Clamp_Close_Time` ↔ `Barrel_Temperature_1`: +0.965
- `Cushion_Position` ↔ `Barrel_Temperature_1`: +0.965
- `Filling_Time` ↔ `Clamp_Open_Position`: +0.965

## RG3 불량 라벨 연관 변수

- `Barrel_Temperature_5`: point-biserial +0.066, Spearman +0.068, 방향보정 AUC 0.635, FDR q=0.559
- `Plasticizing_Time`: point-biserial +0.025, Spearman +0.030, 방향보정 AUC 0.561, FDR q=0.951
- `Barrel_Temperature_3`: point-biserial -0.020, Spearman -0.024, 방향보정 AUC 0.548, FDR q=0.951
- `Cycle_Time`: point-biserial +0.001, Spearman +0.027, 방향보정 AUC 0.546, FDR q=0.967
- `Clamp_Close_Time`: point-biserial +0.027, Spearman +0.025, 방향보정 AUC 0.546, FDR q=0.951
- `Hopper_Temperature`: point-biserial -0.020, Spearman -0.022, 방향보정 AUC 0.545, FDR q=0.951
- `Max_Screw_RPM`: point-biserial +0.021, Spearman +0.021, 방향보정 AUC 0.538, FDR q=0.951
- `Average_Screw_RPM`: point-biserial +0.019, Spearman +0.017, 방향보정 AUC 0.530, FDR q=0.951

## 제품별 차이가 큰 변수쌍

- `Injection_Time` ↔ `Barrel_Temperature_3`: CN7 +0.804, RG3 -0.844, 차이 +1.649
- `Injection_Time` ↔ `Average_Back_Pressure`: CN7 +0.898, RG3 -0.728, 차이 +1.626
- `Injection_Time` ↔ `Barrel_Temperature_2`: CN7 +0.746, RG3 -0.869, 차이 +1.615
- `Injection_Time` ↔ `Max_Back_Pressure`: CN7 +0.806, RG3 -0.740, 차이 +1.546
- `Clamp_Open_Position` ↔ `Average_Back_Pressure`: CN7 +0.625, RG3 -0.862, 차이 +1.487
- `Injection_Time` ↔ `Max_Injection_Speed`: CN7 +0.758, RG3 -0.719, 차이 +1.476
- `Clamp_Open_Position` ↔ `Barrel_Temperature_1`: CN7 +0.474, RG3 -0.980, 차이 +1.453
- `Clamp_Close_Time` ↔ `Clamp_Open_Position`: CN7 +0.479, RG3 -0.973, 차이 +1.452
- `Clamp_Open_Position` ↔ `Mold_Temperature_3`: CN7 +0.476, RG3 -0.953, 차이 +1.428
- `Injection_Time` ↔ `Clamp_Close_Time`: CN7 +0.860, RG3 -0.561, 차이 +1.421

## 해석 주의사항

- 상관관계는 인과관계를 뜻하지 않는다.
- 공정변수는 동일 사이클 내에서 함께 제어되므로 높은 상관이 제어 로직이나 공정 단계 결합을 반영할 수 있다.
- 다중공선성이 강한 변수는 예측 모델 투입 전 제거·집계·정규화 또는 규제 모델 사용을 검토한다.
- 제품 간 상관 구조 차이가 커서 CN7과 RG3는 분리 모델 또는 제품 상호작용 항을 우선 검토한다.
- 불량 원인 규명에는 시간 순서, 설비 설정값, 원재료·금형·작업 조건 같은 추가 맥락이 필요하다.