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


# 4. 10초 동안 데이터 수집 및 예측 (개별적으로 예측 후 최종 판정)
def collect_and_predict():
    collected_data = []
    predictions = []  # 예측 결과를 저장할 리스트

    for _ in range(10):  # 10초 동안 데이터 수신
        data = get_mock_data()  # 실제 환경에서는 실제 수신 데이터로 교체
        collected_data.append(data)
        time.sleep(1)  # 1초에 한 번 데이터 수신

    # 수집된 데이터를 NumPy 배열로 변환
    collected_data = np.array(collected_data)

    # 5. 각각의 데이터를 pandas DataFrame으로 변환하여 스케일링 및 예측
    columns = ['HeartRate', 'RespirationRate', 'Temperature', 'State', 'DogType']

    for i, data in enumerate(collected_data):
        # 데이터를 DataFrame으로 변환
        data_df = pd.DataFrame([data], columns=columns)

        # 스케일링
        data_scaled = scaler.transform(data_df)

        # 예측 수행
        prediction_prob = model.predict(data_scaled)
        prediction = (prediction_prob > 0.5).astype(int)  # 0.5 이상이면 비정상, 이하이면 정상

        # 결과 저장
        predictions.append(prediction[0][0])

        # 결과 출력 (개별 예측 출력)
        print(f"데이터 {i + 1}: 예측 확률 = {prediction_prob[0][0]:.4f}, 예측된 클래스 (정상: 0, 비정상: 1) = {prediction[0][0]}")

    # 6. 10번의 예측 결과에서 정상/비정상 판단
    normal_count = predictions.count(0)  # 정상(0)의 개수 카운트
    abnormal_count = predictions.count(1)  # 비정상(1)의 개수 카운트

    # 과반수 판정
    if normal_count > abnormal_count:
        print(f"최종 판정: 정상 (정상 데이터 개수: {normal_count}, 비정상 데이터 개수: {abnormal_count})")
    else:
        print(f"최종 판정: 비정상 (정상 데이터 개수: {normal_count}, 비정상 데이터 개수: {abnormal_count})")


# 예측 실행
collect_and_predict()
