import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import time

# 저장된 모델 경로
model_file_path = 'pet_vital_model.keras'
scaler_file_path = 'scaler.save'

# 1. 모델 로드
model = load_model(model_file_path)
print("모델이 성공적으로 로드되었습니다.")

# 2. 스케일러 로드 (데이터 정규화를 위해)
scaler = joblib.load(scaler_file_path)
print("스케일러가 성공적으로 로드되었습니다.")


# 3. 임의의 데이터 수신 (1초에 한 번씩 데이터 수신하여 10초 동안 데이터 모으기)
def get_mock_data():
    """
    실제 데이터 수신이 가능한 환경에서는 이 함수를
    실제 데이터 수신 함수로 대체하면 됩니다.
    """
    # 예시로 임의의 데이터를 생성합니다.
    heart_rate = np.random.randint(60, 150)  # 60 ~ 150 범위의 심박수
    respiration_rate = np.random.randint(10, 40)  # 10 ~ 40 범위의 호흡수
    temperature = np.random.uniform(37.0, 39.5)  # 37.0 ~ 39.5 범위의 체온
    state = np.random.choice([0, 1, 2])  # 상태 (정지: 0, 걷기: 1, 뛰기: 2)
    dog_type = np.random.choice([0, 1, 2])  # 도그 타입 (0: 소형견, 1: 대형견, 2: 노령견)

    return np.array([heart_rate, respiration_rate, temperature, state, dog_type])


# 4. 10초 동안 데이터 수집 및 예측 (평균값 사용)
def collect_and_predict():
    collected_data = []

    for _ in range(10):  # 10초 동안 데이터 수신
        data = get_mock_data()  # 실제 환경에서는 실제 수신 데이터로 교체
        collected_data.append(data)
        time.sleep(1)  # 1초에 한 번 데이터 수신

    # 수집된 데이터를 NumPy 배열로 변환
    collected_data = np.array(collected_data)

    # 10개의 데이터를 평균화
    averaged_data = np.mean(collected_data, axis=0).reshape(1, -1)

    # 평균화된 데이터를 pandas DataFrame으로 변환하여 스케일링
    columns = ['HeartRate', 'RespirationRate', 'Temperature', 'State', 'DogType']
    averaged_data_df = pd.DataFrame(averaged_data, columns=columns)

    # 5. 평균화된 데이터를 스케일링
    averaged_data_scaled = scaler.transform(averaged_data_df)

    # 6. 예측 수행 (평균값으로 예측)
    prediction_prob = model.predict(averaged_data_scaled)
    prediction = (prediction_prob > 0.5).astype(int)

    # 7. 결과 출력
    print(f"평균 데이터: {averaged_data}")
    print(f"예측 확률: {prediction_prob[0][0]:.4f}")
    print(f"예측된 클래스 (정상: 0, 비정상: 1): {prediction[0][0]}")


# 예측 실행
collect_and_predict()
