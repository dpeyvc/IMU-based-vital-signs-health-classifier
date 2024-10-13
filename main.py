import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# 1. 데이터 생성
# ---------------------------

# 설정
num_normal_samples = 500000  # 각 상태별 정상 샘플 수
num_abnormal_samples = 50000  # 각 상태별 비정상 샘플 수 (정상의 10%)

# 파일 경로 설정 (파일 확장자를 .keras로 변경)
data_file_path = 'pet_vital_data_with_labels.csv'
model_file_path = 'pet_vital_model.keras'
scaler_file_path = 'scaler.save'

# 데이터 생성 여부 확인
if not os.path.exists(data_file_path):
    print("데이터 파일이 존재하지 않아 데이터를 생성합니다...")

    # 랜덤 시드 설정 (재현 가능성을 위해)
    np.random.seed(42)

    # 도그 타입 설정
    dog_types = [0, 1, 2]
    dog_type_probs = [0.3, 0.5, 0.2]

    # 도그 타입 별 심박수 기준 (정지 상태 기준)
    heart_rate_ranges = {
        0: {'rest': (100, 180)},  # 어린 강아지
        1: {'rest': (100, 140)},  # 소형견
        2: {'rest': (60, 100)}  # 대형견 또는 노령견
    }

    # 상태별 심박수 증가 비율 (정지 상태 대비)
    heart_rate_multiplier = {
        0: {'rest': (1.0, 1.0), 'walk': (1.3, 1.5), 'run': (1.5, 2.0)},
        1: {'rest': (1.0, 1.0), 'walk': (1.3, 1.5), 'run': (1.5, 2.0)},
        2: {'rest': (1.0, 1.0), 'walk': (1.3, 1.5), 'run': (1.5, 2.0)}
    }

    # 상태별 호흡수 및 체온 범위 설정 (예시 값)
    respiration_rate_ranges = {
        'rest': (10, 20),
        'walk': (20, 30),
        'run': (30, 40)
    }

    temperature_ranges = {
        'rest': (37.5, 38.5),
        'walk': (38.0, 39.0),
        'run': (38.5, 39.5)
    }

    states = [0, 1, 2]  # 0: 정지, 1: 걷기, 2: 뛰기


    def get_state_name(state):
        if state == 0:
            return 'rest'
        elif state == 1:
            return 'walk'
        elif state == 2:
            return 'run'


    def generate_vital_data(state, num_samples, abnormal=False):
        # 도그 타입 할당
        dog_type = np.random.choice(dog_types, size=num_samples, p=dog_type_probs)

        # 심박수 생성
        hr = np.zeros(num_samples)
        for dt in dog_types:
            idx = dog_type == dt
            hr_min, hr_max = heart_rate_ranges[dt]['rest']
            hr_rest = np.random.uniform(hr_min, hr_max, size=idx.sum())
            if state == 0:  # 정지 상태
                if not abnormal:
                    hr[idx] = hr_rest
                else:
                    # 비정상: 정지 상태 범위를 벗어나게 생성
                    is_low = np.random.rand(idx.sum()) < 0.5
                    hr[idx] = np.where(is_low,
                                       hr_rest * np.random.uniform(0.5, 0.8, size=idx.sum()),
                                       hr_rest * np.random.uniform(1.2, 1.5, size=idx.sum()))
            elif state == 1:  # 걷기 상태
                multiplier_min, multiplier_max = heart_rate_multiplier[dt]['walk']
                hr_normal = hr_rest * np.random.uniform(multiplier_min, multiplier_max, size=idx.sum())
                if not abnormal:
                    hr[idx] = hr_normal
                else:
                    # 비정상: 걷기 상태 심박수 범위를 벗어나게 생성
                    is_low = np.random.rand(idx.sum()) < 0.5
                    hr[idx] = np.where(is_low,
                                       hr_normal * np.random.uniform(0.5, 0.8, size=idx.sum()),
                                       hr_normal * np.random.uniform(1.5, 2.5, size=idx.sum()))
            elif state == 2:  # 뛰기 상태
                multiplier_min, multiplier_max = heart_rate_multiplier[dt]['run']
                hr_normal = hr_rest * np.random.uniform(multiplier_min, multiplier_max, size=idx.sum())
                if not abnormal:
                    hr[idx] = hr_normal
                else:
                    # 비정상: 뛰기 상태 심박수 범위를 벗어나게 생성
                    is_low = np.random.rand(idx.sum()) < 0.5
                    hr[idx] = np.where(is_low,
                                       hr_normal * np.random.uniform(0.5, 0.8, size=idx.sum()),
                                       hr_normal * np.random.uniform(2.0, 3.0, size=idx.sum()))

        # 호흡수 생성
        rr_min, rr_max = respiration_rate_ranges[get_state_name(state)]
        if not abnormal:
            rr = np.random.uniform(rr_min, rr_max, size=num_samples)
        else:
            # 비정상: 호흡수 범위를 벗어나게 생성
            is_low = np.random.rand(num_samples) < 0.5
            rr = np.where(is_low,
                          np.random.uniform(rr_min * 0.5, rr_min * 0.8, size=num_samples),
                          np.random.uniform(rr_max * 1.2, rr_max * 1.5, size=num_samples))

        # 체온 생성
        temp_min, temp_max = temperature_ranges[get_state_name(state)]
        if not abnormal:
            temp = np.random.uniform(temp_min, temp_max, size=num_samples)
        else:
            # 비정상: 체온 범위를 벗어나게 생성
            is_low = np.random.rand(num_samples) < 0.5
            temp = np.where(is_low,
                            np.random.uniform(temp_min - 1.0, temp_min - 0.5, size=num_samples),
                            np.random.uniform(temp_max + 0.5, temp_max + 1.0, size=num_samples))

        # 데이터프레임 생성
        data = pd.DataFrame({
            'HeartRate': hr,
            'RespirationRate': rr,
            'Temperature': temp,
            'State': np.full(num_samples, state),
            'DogType': dog_type
        })

        # 정상: 0, 비정상: 1
        data['Label'] = 0 if not abnormal else 1
        return data


    # 정상 데이터 생성
    normal_data_list = []
    for state in states:
        print(f"정상: {get_state_name(state)} 상태의 데이터 생성 중...")
        data = generate_vital_data(state, num_normal_samples, abnormal=False)
        normal_data_list.append(data)

    normal_data = pd.concat(normal_data_list, ignore_index=True)

    # 비정상 데이터 생성
    abnormal_data_list = []
    for state in states:
        print(f"비정상: {get_state_name(state)} 상태의 데이터 생성 중...")
        data = generate_vital_data(state, num_abnormal_samples, abnormal=True)
        abnormal_data_list.append(data)

    abnormal_data = pd.concat(abnormal_data_list, ignore_index=True)

    # 정상과 비정상 데이터 합치기
    vital_data = pd.concat([normal_data, abnormal_data], ignore_index=True)

    # 필요한 경우 정수형으로 변환
    vital_data['HeartRate'] = vital_data['HeartRate'].round().astype(int)
    vital_data['RespirationRate'] = vital_data['RespirationRate'].round().astype(int)
    vital_data['Temperature'] = vital_data['Temperature'].round(1)
    vital_data['DogType'] = vital_data['DogType'].astype(int)
    vital_data['Label'] = vital_data['Label'].astype(int)

    # CSV 파일로 저장
    vital_data.to_csv(data_file_path, index=False)
    print(f"데이터 생성 완료. 파일 경로: {data_file_path}")
else:
    print("이미 데이터 파일이 존재합니다. 데이터 생성을 건너뜁니다.")

# ---------------------------
# 2. 데이터 전처리
# ---------------------------

print("데이터를 불러오고 전처리합니다...")

# 데이터 불러오기
data = pd.read_csv(data_file_path)

# 특징(features)과 레이블(labels) 분리
X = data[['HeartRate', 'RespirationRate', 'Temperature', 'State', 'DogType']]
y = data['Label']

# 데이터 분할 (훈련: 80%, 테스트: 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 데이터 정규화
if os.path.exists(scaler_file_path):
    print("기존 스케일러를 로드합니다...")
    scaler = joblib.load(scaler_file_path)
else:
    print("새로운 스케일러를 생성하고 학습시킵니다...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, scaler_file_path)
    print(f"스케일러가 저장되었습니다: {scaler_file_path}")

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 3. TensorFlow 모델 구축 및 학습
# ---------------------------

# 모델 로드 또는 새 모델 생성
if os.path.exists(model_file_path):
    print("기존 모델을 로드합니다...")
    model = load_model(model_file_path)
else:
    print("새로운 모델을 생성합니다...")
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),  # Input 레이어 추가
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# 콜백 설정
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', save_best_only=True, verbose=1)

# 모델 학습
print("모델을 학습시킵니다...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=1024,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ---------------------------
# 4. 모델 평가
# ---------------------------

print("모델을 평가합니다...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.4f}")

# 예측 및 평가 지표
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

print("분류 보고서:")
print(classification_report(y_test, y_pred))

print("혼동 행렬:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 5. 학습 지표 시각화
# ---------------------------

print("학습 지표를 시각화합니다...")


def plot_history(history):
    # 정확도 그래프
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.title('훈련 및 검증 정확도')
    plt.xlabel('에포크')
    plt.ylabel('정확도')
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('훈련 및 검증 손실')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_history(history)
