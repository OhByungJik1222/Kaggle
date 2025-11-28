## 🧠 EDA

### 📌 운전조건 분포 (op1 - 고도, op2 - 마하, op3 - 스로틀)

FD004 데이터셋은 다중 복합 조건이 존재 → 모델 입력으로 고려

<img width="764" height="439" alt="image" src="https://github.com/user-attachments/assets/8768c642-3fa3-4289-ae79-8301dba86d90" />

각 센서별 이동평균의 추세를 확인하여 의미 없는 Feature 제외

### 📌 센서별 이동평균 (s1 ~ s21)
다중 운전 조건이 혼재되어 모든 센서의 변동성이 매우 높게 나타나며 추세 확인 불가

<img width="759" height="453" alt="image" src="https://github.com/user-attachments/assets/e35ddf2d-6e30-4fdd-b6b6-8d9fce321f3b" />

그러나 각 조건 변수는 이산적으로 분포하는 것을 미루어보아<br>
각 운전 조건 별 센서의 변동성 확인필요

- 조건 변수 소수 첫째자리까지 반올림 후 조건 확인

|op1|op2|op3|count|
|---|---|---|---|
|0.0|0.0|100.0|9238|
|10.0|0.2|100.0|4686|
||0.3|100.0|4538|
|20.0|0.7|100.0|9091|
|25.0|0.6|60.0|9139|
|35.0|0.8|100.0|9162|
|42.0|0.8|100.0|15395|

- 각 7개의 조건마다 센서의 이동 평균을 다시 확인해본 결과, 추세가 명확하게 나타남. 

<img width="744" height="451" alt="image" src="https://github.com/user-attachments/assets/7c619002-76f0-488b-989b-f095b344e84e" />

FD004 데이터셋의 경우 조건 변수를 모델의 입력으로 포함해야 하며, 각 조건은 7개의 범주형 변수로 학습

**❗ EDA 결과  `s_1, s_5, s_6, s_16, s_18, s_19` 제외**

## 🧠 Feature Engineering

각 모델 별 모델에 따른 Feature Engineering 적용

- Raw Data
- 전체 통계량
- 누적 통계량
- RUL Clipping

이후 모든 단계에서 RUL Clipping 적용 전후 성능, Scaler(Standard, MinMax, Robust)에 따른 성능 비교를 위해 모든 조합의 데이터에서 모델 학습

### 📌 Linear Regression

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat (RUL Clipping)|Linear|None|17.630382 ✅|21.542402 ✅|
|Cumulative Stat (RUL Clipping)	|Linear|RobustScaler|17.630382|21.542402|
|Cumulative Stat (RUL Clipping)	|Linear|StandardScaler|17.630382|21.542402|
|Cumulative Stat (RUL Clipping)|Linear|MinMaxScaler|1.17.630382|21.542402|
|Cumulative Stat (RUL Clipping)	|Linear|RobustScaler|17.630382|21.542402|

**❗Linear Regression의 경우 전처리 기법, 스케일러에 관계없이 모두 동일한 결과**

---

### 📌 Ridge, Lasso, ElasticNet

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat (RUL Clipping)|Ridge|RobustScaler|17.715606 ✅|21.641515|
|Cumulative Stat (RUL Clipping)|Ridge|None|17.742342|21.558355 ✅|
|Cumulative Stat (RUL Clipping)|Ridge|StandardScaler|17.745798|21.696824|
|Cumulative Stat (RUL Clipping)|Ridge|MinMaxScaler|18.219899|22.233853|
|Cumulative Stat (RUL Clipping)|ElasticNet|None|21.773577|26.210920|

**❗규제 모델의 경우 모두 누적 통계량에 RUL Clipping을 적용한 데이터를 활용한 모델의 성능이 가장 가장 좋은 것으로 나타남**

---

### 📌 RandomForest, Bagging

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat (RUL Clipping)|RandomForest|None|14.085685 ✅|20.226423|
|Cumulative Stat (RUL Clipping)|Bagging|None|14.515726|21.491899|
|Raw (RUL Clipping)|Bagging|None|14.537097|20.125730 ✅|
|Raw (RUL Clipping)|RandomForest|None|14.622258|19.808460|
|Cumulative Stat|RandomForest|None|22.779234|31.577347|

**❗배깅 모델의 경우 대부분의 경우 RUL Clipping을 적용하였을 때 모델의 성능이 가장 좋았으며 또한 누적 통계량을 사용하여 학습시킨 모델의 성능이 가장 좋은 것으로 나타남**

---

### 📌 XGBoost, LightGBM, CatBoost

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat (RUL Clipping)|LightGBM|None|13.540428 ✅|18.731705|
|Cumulative Stat (RUL Clipping)|CatBoost|None|13.683681|18.547914 ✅|
|Cumulative Stat (RUL Clipping)|XGBoost|None|14.183904|19.826313|
|Raw (RUL Clipping)|XGBoost|None|14.307033|19.851915|
|Raw (RUL Clipping)|CatBoost|None|14.403875|19.742004|

**❗부스팅 모델의 경우 모든 경우 RUL Clipping을 적용하였을 때 모델의 성능이 가장 좋았으며 또한 누적 통계량을 사용하여 학습시킨 모델의 성능이 가장 좋은 것으로 나타남**

---

### 📌 LSTM, TCN

LSTM, TCN 모델의 경우 전처리, 스케일러, RUL Clipping 적용 여부 비교 뿐만 아니라 Sequence의 window 크기 (30, 40, 50) 또한 추가로 비교 

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Raw (RUL Clipping) - 50|LSTM|StandardScaler|11.412210 ✅|16.287795 ✅|
|Raw (RUL Clipping) - 50|LSTM|MinMaxScaler|12.116726|16.991025|
|Cumulative Stat (RUL Clipping) - 50|LSTM|MinMaxScaler|12.252406|17.543150|
|Raw (RUL Clipping) - 40|LSTM|StandardScaler|12.289542|17.392301|
|Raw (RUL Clipping) - 40|LSTM|RobustScaler|12.512035|17.375655|

**❗시계열 모델의 경우 모든 경우 RUL Clipping을 적용하였을 때 모델의 성능이 가장 좋았으며 또한 예외적으로 통계량을 구하지 않은 Raw Data (window size 50)을 사용하여 학습시킨 모델의 성능이 가장 좋은 것으로 나타남**

---

### 📌 Final Result

각 모델별 가장 좋은 성능을 보인 모델  앙상블 전략 수립

|Data|Model|Scaler|MAE|RMSE|RMSE - MAE|
|---|---|---|---|---|---|
|Raw (RUL Clipping) - 50|LSTM|StandardScaler|11.412210 ✅|16.287795 ✅|4.875586|
|Raw (RUL Clipping) - 50|LSTM|MinMaxScaler|12.116726|16.991025|4.874300|
|Cumulative Stat (RUL Clipping) - 50|LSTM|MinMaxScaler|12.252406|17.543150|5.290744|
|Raw (RUL Clipping) - 40|LSTM|StandardScaler|12.289542|17.392301|5.102759|
|Raw (RUL Clipping) - 40|LSTM|RobustScaler|12.512035|17.375655|4.863620 ✅|
|Cumulative Stat (RUL Clipping)|LightGBM|None|13.540428|18.731705|5.191277|
|Cumulative Stat (RUL Clipping)|CatBoost|None|13.683681|18.547914|4.864233|
|Cumulative Stat (RUL Clipping)|RandomForest|None|14.085685|20.226423|6.140737|
|Cumulative Stat (RUL Clipping)|XGBoost|None|14.183904|19.826313|5.642409|
|Raw (RUL Clipping)|XGBoost|None|14.307033|19.851915|5.544883|

**❗RMSE - MAE: 오차의 분포 추정**

### 📌 Hyperparameter Tuning + Ensemble

- 최종 결과에서 선정한 각 모델 오차 간의 상관관계 분석

<img width="634" height="528" alt="image" src="https://github.com/user-attachments/assets/5748aad8-31d9-49b0-899b-a643721bd431" />

오차 간의 상관관계 분석 결과 비시계열 모델과 시계열 모델의 오차 패턴이 비슷한 것으로 나타남.<br>
**❗따라서 비시계열 모델 + 시계열 모델 간 앙상블에서 시너지 가능성 ↓**

각 모델 별 하이퍼 파라미터 튜닝 후 앙상블<br>
각 모델의 예측 결과에 가중치를 곱하는 방식으로 앙상블 적용<br>
이때 가중치 또한 Optuna를 사용하여 튜닝

- 앙상블 결과

|Model|MAE|RMSE|RMSE - MAE|
|---|---|---|---|
|RandomForest + LSTM|26.217034|35.083977|8.866944|
|LightGBM + LSTM|23.827196 ✅|32.496967 ✅|8.669771 ✅|
|Catboost + LSTM|27.659523|36.637783|8.978260|

**❗모든 지표에서 LightGBM + LSTM 조합이 가장 성능이 좋은 것으로 나타났으나 예상대로 앙상블 전보다 성능이 떨어지는 것으로 확인되었다.**

### 📌 Conclusion

모든 지표를 고려하였을 때 Raw Data + RUL Clipping + window size 50을 적용한 LSTM 모델이 가장 좋은 모델로 선정됨.
