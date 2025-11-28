## 🧠 EDA

### 📌 운전조건 분포 (op1 - 고도, op2 - 마하, op3 - 스로틀)

FD001 데이터셋은 단일 조건으로 단일 운전 조건 하에서의 미세한 변동
→ 모델의 입력으로 의미 X

<img width="642" height="368" alt="image" src="https://github.com/user-attachments/assets/56599034-1d1f-4f17-ab00-8097137e3f8e" />

각 센서별 이동평균의 추세를 확인하여 의미 없는 Feature 제외

### 📌 센서별 이동평균 (s1 ~ s21)
RUL이 작아질수록 위 아래로 발산할 수록 중요한 Feature

<img width="757" height="455" alt="image" src="https://github.com/user-attachments/assets/433e47fc-687c-4ff6-b1c0-88c0bbd35974" />

**❗ EDA 결과 `op1, op2, op3, s_1, s_5, s_6, s_16, s_18, s_19` 제외**

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
|Cumulative Stat (RUL Clipping)|Linear|RobustScaler|15.669487 ✅|18.858083 ✅|
|Cumulative Stat|Linear|RobustScaler|15.669487|18.858083|
|Cumulative Stat|Linear|StandardScaler|15.669487|18.858083|
|Cumulative Stat (RUL Clipping)|Linear|StandardScaler|15.669487|18.858083|
|Cumulative Stat|Linear|MinMaxScaler|15.669487|18.858083|

**❗Linear Regression의 경우 전처리 기법, 스케일러에 관계없이 모두 동일한 결과**

---

### 📌 Ridge, Lasso, ElasticNet

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat|Lasso|StandardScaler|14.743519 ✅|17.918506|
|Cumulative Stat (RUL Clipping)|Lasso|StandardScaler|14.743519|17.918506|
|Cumulative Stat (RUL Clipping)|ElasticNet|StandardScaler|14.950372|17.850735 ✅|
|Cumulative Stat|ElasticNet|StandardScaler|14.950372|17.850735|
|Cumulative Stat|Lasso|RobustScaler|15.027839|18.076196|

**❗규제 모델의 경우 RUL Clipping 적용여부와 관계없이 누적 통계량을 활용한 모델의 성능이 가장 가장 좋은 것으로 나타남**

---

### 📌 RandomForest, Bagging

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat (RUL Clipping)|RandomForest|None|7.5183 ✅|10.653501 ✅|
|Cumulative Stat (RUL Clipping)|Bagging|None|8.2900|11.502852|
|Raw (RUL Clipping)|RandomForest|None|12.1630|17.186846|
|Raw (RUL Clipping)|Bagging|None|12.5320|18.605591|
|Cumulative Stat|RandomForest|None|13.1510|21.337424|

**❗배깅 모델의 경우 대부분의 경우 RUL Clipping을 적용하였을 때 모델의 성능이 가장 좋았으며 또한 누적 통계량을 사용하여 학습시킨 모델의 성능이 가장 좋은 것으로 나타남**

---

### 📌 XGBoost, LightGBM, CatBoost

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat (RUL Clipping)|CatBoost|None|7.629197 ✅|10.553240 ✅|
|Cumulative Stat (RUL Clipping)|LightGBM|None|7.670903|11.071520|
|Cumulative Stat (RUL Clipping)|XGBoost|None|8.567878|12.321973|
|Raw (RUL Clipping)|CatBoost|None|11.679642|17.032597|
|Raw (RUL Clipping)|LightGBM|None|11.754092|16.769257|

**❗부스팅 모델의 경우 모든 경우 RUL Clipping을 적용하였을 때 모델의 성능이 가장 좋았으며 또한 누적 통계량을 사용하여 학습시킨 모델의 성능이 가장 좋은 것으로 나타남**

---

### 📌 LSTM, TCN

LSTM, TCN 모델의 경우 전처리, 스케일러, RUL Clipping 적용 여부 비교 뿐만 아니라 Sequence의 window 크기 (30, 40, 50) 또한 추가로 비교 

|Data|Model|Scaler|MAE|RMSE|
|---|---|---|---|---|
|Cumulative Stat (RUL Clipping) - 50|LSTM|RobustScaler|7.834088 ✅|12.397012|
|Cumulative Stat (RUL Clipping)- 50|LSTM|MinMaxScaler|8.406834|12.200348 ✅|
|Cumulative Stat (RUL Clipping)- 30|LSTM|RobustScaler|8.521237|12.321973|
|Cumulative Stat (RUL Clipping)- 40|LSTM|RobustScaler|8.550470|13.436869|
|Raw (RUL Clipping)- 50|LSTM|MinMaxScaler|9.087166|12.664556|

**❗시계열 모델의 경우 모든 경우 RUL Clipping을 적용하였을 때 모델의 성능이 가장 좋았으며 또한 누적 통계량 (window size 50)을 사용하여 학습시킨 모델의 성능이 가장 좋은 것으로 나타남**

---

### 📌 Final Result

각 모델별 가장 좋은 성능을 보인 모델  앙상블 전략 수립

|Data|Model|Scaler|MAE|RMSE|RMSE - MAE|
|---|---|---|---|---|---|
|Cumulative Stat (RUL Clipping)|RandomForest|None|7.518300 ✅|10.653501|3.135201|
|Cumulative Stat (RUL Clipping)|CatBoost|None|7.629197|10.553240 ✅|2.924043 ✅|
|Cumulative Stat (RUL Clipping)|LightGBM|None|7.670903|11.071520|3.400618|
|Cumulative Stat (RUL Clipping)	- 50|LSTM|RobustScaler|7.834088|12.397012|4.562924|
|Cumulative Stat (RUL Clipping)|Bagging|None|8.290000|11.502852|3.212852|
|Cumulative Stat (RUL Clipping) - 50|LSTM|MinMaxScaler|8.406834|12.200348|3.793514|
|Cumulative Stat (RUL Clipping) - 30|LSTM|RobustScaler|8.521237|12.822533|4.301296|
|Cumulative Stat (RUL Clipping) - 40|LSTM|RobustScaler|8.550470|13.436869|4.886398|
|Cumulative Stat (RUL Clipping)	|XGBoost|None|8.567878|12.321973|3.754095|
|Raw (RUL Clipping) - 50|LSTM|MinMaxScaler|9.087166|12.664556|3.577391|

**❗RMSE - MAE: 오차의 분포 추정**

### 📌 Hyperparameter Tuning + Ensemble

- 최종 결과에서 선정한 각 모델 오차 간의 상관관계 분석

<img width="634" height="528" alt="image" src="https://github.com/user-attachments/assets/dbc21af9-71f5-4d50-a52d-5d3ccbcf4ff2" />

오차 간의 상관관계 분석 결과 비시계열 모델과 시계열 모델의 오차 패턴이 다르다고 볼 수 있음.<br>
**❗따라서 비시계열 모델 + 시계열 모델 간 앙상블에서 시너지 가능성 ↑**

각 모델 별 하이퍼 파라미터 튜닝 후 앙상블<br>
각 모델의 예측 결과에 가중치를 곱하는 방식으로 앙상블 적용<br>
이때 가중치 또한 Optuna를 사용하여 튜닝

- 앙상블 결과

|Model|MAE|RMSE|RMSE - MAE|
|---|---|---|---|
|Bagging + LSTM|9.427386|13.275074|3.847687|
|RandomForest + LSTM|8.724696|12.399868|3.675172|
|LightGBM + LSTM|8.669172 ✅|12.161018 ✅|3.491846 ✅|
|Catboost + LSTM|9.891600|14.025749|4.134149|

**❗모든 지표에서 LightGBM + LSTM 조합이 가장 성능이 좋은 것으로 나타났으나 앙상블 전보다 성능이 떨어지는 것으로 확인되었다.**

### 📌 Conclusion

모든 지표를 고려하였을 때 누적통계량 + RUL Clipping을 적용한 RandomForest 모델이 가장 좋은 모델로 선정됨.

### 📌 보전비용 분석 및 제안

