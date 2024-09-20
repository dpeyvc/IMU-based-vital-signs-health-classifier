# predict.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# 1. 모델 로드
model = load_model('Vital_model.h5')

# 2. LabelEncoder 정의 (학습 시 사용한 것과 동일하게 설정)
# 예를 들어, 학습 시 'Labrador'가 0, 'Bulldog'가 1로 인코딩되었다고 가정
le_breeds = LabelEncoder()
le_breeds.classes_ = np.array(['Labrador', 'Bulldog'])  # 실제 클래스에 맞게 수정하세요

# 3. StandardScaler 정의 (학습 시 사용한 스케일러의 평균과 표준편차를 사용)
# 여기서는 예시로 임의의 평균과 표준편차를 사용합니다. 실제 값으로 교체하세요.
scaler = StandardScaler()
scaler.mean_ = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])  # 예시 값
scaler.scale_ = np.array([0.5] * 14)  # 예시 값
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = 14
scaler.feature_names_in_ = np.array([
    'breeds_encoded', 'age', 'weight',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std',
    'br_mean', 'br_max', 'br_min', 'br_std',
    'ecg_pulse_interval_mean', 'ecg_pulse_interval_std',
    'bad_ecg_total_time'
])

# 4. 건강하지 않은 새로운 데이터 생성
# 'Unhealthy' 상태를 만족하는 데이터를 생성합니다.
# 조건: age > 8 또는 weight < 5 또는 weight > 50
new_data = pd.DataFrame({
    'breeds': ['Labrador', 'Bulldog', 'Labrador'],  # 예시 종
    'age': [9, 5, 14],                             # 첫 번째와 세 번째 샘플은 age > 8
    'weight': [30, 55, 4],                          # 두 번째 샘플은 weight > 50, 세 번째 샘플은 weight < 5
    'hr_mean': [100, 105, 110],                     # 예시 심박수 평균
    'hr_max': [120, 130, 140],                      # 예시 심박수 최대
    'hr_min': [80, 90, 85],                         # 예시 심박수 최소
    'hr_std': [10, 15, 12],                         # 예시 심박수 표준편차
    'br_mean': [20, 22, 25],                        # 예시 호흡수 평균
    'br_max': [25, 27, 30],                         # 예시 호흡수 최대
    'br_min': [15, 18, 20],                         # 예시 호흡수 최소
    'br_std': [2, 3, 4],                             # 예시 호흡수 표준편차
    'ecg_pulse_interval_mean': [0.5, 0.6, 0.7],      # 예시 ECG 펄스 간격 평균
    'ecg_pulse_interval_std': [0.05, 0.06, 0.07],    # 예시 ECG 펄스 간격 표준편차
    'bad_ecg_total_time': [0, 5, 10]                # 예시 불량 ECG 총 시간
})

# 5. 범주형 변수 인코딩
new_data['breeds_encoded'] = le_breeds.transform(new_data['breeds'])

# 6. 예측에 사용할 특성 선택
feature_columns = [
    'breeds_encoded', 'age', 'weight',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std',
    'br_mean', 'br_max', 'br_min', 'br_std',
    'ecg_pulse_interval_mean', 'ecg_pulse_interval_std',
    'bad_ecg_total_time'
]

X_new = new_data[feature_columns]

# 7. 데이터 정규화
X_new_scaled = scaler.transform(X_new)

# 8. 예측 수행
y_pred_probs = model.predict(X_new_scaled)
y_pred_classes = (y_pred_probs > 0.5).astype(int).reshape(-1)

# 9. 예측 결과 매핑
health_status = {0: 'Healthy', 1: 'Unhealthy'}
new_data['HealthStatus_Predicted'] = y_pred_classes
new_data['HealthStatus'] = new_data['HealthStatus_Predicted'].map(health_status)

# 10. 예측 결과 출력
print(new_data[['breeds', 'age', 'weight', 'HealthStatus']])
