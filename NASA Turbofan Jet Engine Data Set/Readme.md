# 🚀 **NASA Turbofan Jet Engine Data Set**

잔존 수명 예측 기반 예지보전 시스템 구축 프로젝트<br>
캐글 공개 데이터셋(C-MAPSS)을 활용한 시계열 기반 머신러닝 모델링

## 📂 루트 디렉토리 및 주요 파일 구조
```
📂 FD001
├── EDA_FD001.ipynb # FD001 데이터 EDA
├── ~~~_Regression_FD001.ipynb # 전처리 및 모델 테스트
├── Result_FD001.ipynb # 결과 종합
└── Hyperparameter_Tuning_Ensemble_FD001.ipynb # 하이퍼파라미터 튜닝 및 앙상블 실험

📂 FD001
├── EDA_FD004.ipynb
├── ~~~_Regression_FD004.ipynb
├── Result_FD004.ipynb
└── Hyperparameter_Tuning_Ensemble_FD004.ipynb
```

## 🚀 프로젝트 개요
항공기 터보팬 제트 엔진의 센서 데이터를 분석하여<br>
각 엔진의 남은 수명(RUL, Remaining Useful Life)을 예측하고,<br>
이를 기반으로 예지보전(PdM, Predictive Maintenance) 체계를 설계합니다.

### 목적
- 고장 시점을 사전에 예측하여 정비 시점 최적화
- 불필요한 조기 교체 방지 및 비용 절감
- 안전성 확보 및 데이터 기반 의사결정 지원

## 🚀 배경

📌 공기를 압축 → 연소 → 팽창시켜 발생한 에너지로 터빈을 돌리고 배기가스를 분사해 추진력을 얻는 원리

<img width="1741" height="636" alt="image" src="https://github.com/user-attachments/assets/e740c6c3-25a7-4879-a9bf-2399dc8195e4" />


## 📖 데이터셋

- 출처: NASA C-MAPSS Dataset (Kaggle)
- 형식: 시계열 센서 데이터 (온도, 압력, 속도 등)
- 엔진 수: 100+ 개별 엔진, 다양한 운항 조건과 고장 모드
- 구성
  - train_FD00X.txt: 고장까지 전체 주기 데이터
  - test_FD00X.txt: 고장 직전까지만 수집된 데이터
  - RUL_FD00X.txt: 각 test 엔진의 실제 잔존 수명

📌 C-MAPSS(Commercial Modular Aero-Propulsion System Simulation)
     : NASA가 항공기 엔진의 열역학적 작동 과정을 가상으로 재현하기 위해 제작한 시뮬레이터

📌 1 사이클(Cycle)은 1회 비행을 의미하며, 각 사이클마다 비행 시간과 누적 손상이 다름.
  - 따라서, 잔존수명(RUL) 예측은 센서 데이터의 변화를 추적하는것이 핵심

<img width="515" height="200" alt="image" src="https://github.com/user-attachments/assets/089f0a1b-87a5-451c-ad2a-960aa4813ece" />

- Train: 엔진이 고장날 때까지의 운전 데이터
- Test: 엔진이 고장 전 중단된 시점까지의 데이터
- RUL: Test 세트 각 엔진의 실제 잔여 수명 (Remaining Useful Life)

📌 총 58개의 출력 변수 중 엔진 상태를 대표하는 21개의 센서값이 제공됨

<img width="386" height="414" alt="image" src="https://github.com/user-attachments/assets/1032e575-bf41-4f92-94cf-e87aac28a1ca" />

## 🛠️ 기술 스택

|카테고리|기술|용도|
|---|---|---|
|언어|Python 3.12+|전체 개발|
|데이터 처리|Pandas, NumPy|데이터 전처리, EDA|
|머신러닝|TensorFlow/Keras|LSTM, TCN 시계열 모델|
||Scikit-learn|비시계열 모델, 전처리, 평가 지표|
||XGBoost, LightGBM|트리 기반 모델 비교|
|모델 튜닝|Optuna|하이퍼파라미터 최적화|
|해석도구|SHAP (Shapley)|특성 중요도, 모델 설명가능성|
|시각화|Matplotlib|정적 그래프 (손실 곡선, 산점도)|
||Seaborn|통계 시각화 (히트맵, 분포도)|

## 팀 정보
- 제로베이스 DS 42기 4인 팀 프로젝트
- 기간: 2025냔 11월 (총 4주)

## 참고 문헌
- Saxena et al., "Damage Propagation Modeling for Aircraft Engine", PHM08
- C-MAPSS 공식 문서 및 유저 가이드
