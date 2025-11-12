# 🚀 **NASA Turbofan Jet Engine Data Set - FD001**

# 📂 코드 개요
```
1105_EDA_FD001.ipynb                   : EDA 및 Feature Selection
1106_LSTM_TCN_FD001.ipynb              : LSTM, TCN 학습 (기타 전처리 X)
1107_XGB_LGBM_FD001_01.ipynb           : 유닛별 전체 기초 통계량 데이터로 XGB, LSTM 학습
1107_XGB_LGBM_FD001_02.ipynb           : 시간별 누적 통계량 데이터로 XGB, LSTM 모델 학습
1111_RUL_Clipping_LSTM_TCN_FD001.ipynb : RUL Clipping 적용하여 LSTM, TCN 학습
```

# 🧠 EDA
각 센서별 이동평균의 추세를 확인하여 의미 없는 Feature 제외

### 📌 센서별 이동평균: RUL이 작아질수록 위 아래로 발산할 수록 중요한 Feature
<img width="1989" height="2390" alt="image" src="https://github.com/user-attachments/assets/aa6efd37-a5a1-4ab6-90c4-016ad3805602" />

---

### 📌 센서별 이동평균: 정규분포 형태의 센서 확인
<img width="1990" height="2390" alt="image" src="https://github.com/user-attachments/assets/b80bf040-d3ae-44d6-9171-f0e3e0a470fe" />

---

### 📌 상관계수 확인
<img width="924" height="836" alt="image" src="https://github.com/user-attachments/assets/689a0e5b-cba7-4bed-98f1-86a6fa8d3ce8" />

---

**❗ EDA 결과 `s_1, s_5, s_6, s_16, s_18, s_19` 제외**

# 🧠 LSTM, TCN
아무 전처리도 적용하지 않은 Raw Data를 크기 30인 Window를 통한 Sequence 입력 생성

### 📌 1부터 100까지의 Unit을 10개 단위로 나누어 10 Fold Validation 진행

|Model|RMSE|MAE|
|---|---|---|
|LSTM|30.544|22.556|
|TCN|28.902 ✅|20.580 ✅|

### 📌 Train Set 예측 시각화 (각 유닛별 최대 RUL 기준)
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/c85ef542-4d1e-4867-93ec-55fcabc02c5a" />

---

### 📌 Test Set 예측 결과

|Model|RMSE|MAE|
|---|---|---|
|LSTM|32.204|26.555|
|TCN|27.635 ✅|21.021 ✅|

### 📌 Test Set 예측 시각화
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/b9615c1d-1952-4079-ad95-e4f2381e5dd2" />

---

RUL Clipping을 통해 RUL이 높은 구간에 데이터가 많아지는 것을 방지하고 중요한 고장 직전 데이터 예측 성능 향상

### 📌 Train Data RUL 분포 확인
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/282ef11a-42b2-44de-844e-76eb83c78242" />

**❗ 확인 결과 125 ~ 200 사이 구간에서 수가 급격히 줄어드는 것을 확인할 수 있음.**

---

### 📌 최적 RUL Clip Value 확인
125 ~ 200 으로 RUL Clipping 후 10 Fold Validation 성능 확인

- LSTM

|Clip Value|RMSE|MAE|
|---|---|---|
|125|14.529 ✅|11.047 ✅|
|150|19.105|14.784|
|175|23.097|17.738|
|200|25.973|19.661|
|Raw|30.133|22.254|

- TCN

|Clip Value|RMSE|MAE|
|---|---|---|
|125|15.014 ✅|11.624 ✅|
|150|19.093|14.800|
|175|22.225|16.876|
|200|24.665|18.562|
|Raw|29.731|21.602|

**❗ 두 모델 모두 Clip Value 125에서 가장 좋은 성능을 보임**

---

### 📌 Train Set 예측 시각화 (각 유닛별 최대 RUL 기준, RUL Clipping 적용)
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/082825cf-d3a0-44f1-ac8f-f163a80009f6" />

Train Set에는 Max RUL이 125 이상인 데이터가 대부분 → 큰 RUL에서 성능이 떨어짐<br>
**❗ But, RUL이 큰 데이터보다 고장 직전인 데이터 예측 성능이 중요**

---

### 📌 Test Set 예측 결과 (RUL Clipping 적용)

|Model|RMSE|MAE|
|---|---|---|
|LSTM|15.144 ✅|11.397 ✅|
|TCN|16.183|11.642|

### 📌 Test Set 예측 시각화
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/35115a26-4cd9-4773-b9b0-76e4ee65f2d8" />

Test Set에는 RUL이 125 이하인 데이터가 대부분<br>
**❗ 예상대로 고장 직전인 데이터 예측 성능 대폭 향상 (약 2배)**

# 🧠 XGB, LGBM
각 유닛별 전체 통계량 데이터를 사용하여 학습
```
s_*_mean    : 모든 시점의 센서 평균값
s_*_std     : 모든 시점의 센서 표준편차
s_*_min     : 모든 시점의 센서 최소값
s_*_max     : 모든 시점의 센서 최대값
s_*_last    : 마지막 시점의 센서값
s_*_median  : 모든 시점의 센서 중앙값
s_*_trend   : 모든 시점의 센서값의 선형 추세
```

### 📌 Train Set 예측 결과 (유닛별 전체 통계량)

|Model|RMSE|MAE|
|---|---|---|
|XGB|17.483|14.206|
|LGBM|17.246 ✅|12.853 ✅|

### 📌 Train Set 예측 시각화 (유닛별 전체 통계량)
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/a25028be-bb99-4448-9e65-67416d7e940c" />

---

### 📌 Test Set 예측 결과 (유닛별 전체 통계량)

|Model|RMSE|MAE|
|---|---|---|
|XGB|217.300|212.600|
|LGBM|166.879 ✅|162.734 ✅|

### 📌 Test Set 예측 시각화 (유닛별 전체 통계량)

<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/e78d566c-3b82-41b6-8231-8b1c7adc6e64" />

주어진 Test Set은 Train Set과 달리 고장 직전 시점까지가 아닌 중간 시점이 주어짐
**❗ 따라서 정확한 고장 시점을 구하기 위해선 전체 통계량이 아닌 시점마다 생성해야함**

---

시계열 데이터의 특성을 반영하기 위해 모든 시점의 누적 통계량을 입력으로 학습하여 중간 시점의 데이터가 주어졌을 때 예측 성능 향상

### 📌 Train Set 예측 결과 (시점별 누적 통계량)

|Model|RMSE|MAE|
|---|---|---|
|XGB|4.471 ✅|2.685 ✅|
|LGBM|4.843|3.044|

### 📌 Train Set 예측 시각화 (시점별 누적 통계량)
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/1cf48fd0-5301-4463-b014-084625ad99bb" />

---

### 📌 Test Set 예측 결과 (시점별 누적 통계량)

|Model|RMSE|MAE|
|---|---|---|
|XGB|21.847|13.757|
|LGBM|20.470 ✅|13.212 ✅|

### 📌 Test Set 예측 시각화 (시점별 누적 통계량)
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/56edc12e-d5dd-4d5c-839d-c7da04c0326e" />

**❗ 시점별 누적 통계량으로 학습시킨 결과 중간 시점의 데이터에 대해서도 예측 성능 우수**

---

# 🧠 Test Set 예측 성능 종합

|Model|RMSE|MAE|
|---|---|---|
|TCN (Raw Data)|27.635|21.021|
|LSTM (RUL Clipping) ✅|15.144|11.397|
|LGBM (Total Stat)|166.879|162.734|
|XGB (Cumulative Stat) ✅|20.470|13.212|

**❗ 하이퍼 파라미터 튜닝,  LSTM (RUL Clipping) + XGB (Cumulative Stat) 고려**
