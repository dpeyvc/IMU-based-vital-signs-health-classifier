import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import ast

# 1. 데이터 로드
data_dir = 'dog_health_vitals_dataset'
csv_path = os.path.join(data_dir, 'dataset.csv')
data = pd.read_csv(csv_path)

# 2. 데이터 확인
print(data.head())
print(data.info())
print(data.isnull().sum())

# 3. 결측치 처리
data['segments_br'] = data['segments_br'].fillna('[]')
data['segments_hr'] = data['segments_hr'].fillna('[]')
data['ecg_pulses'] = data['ecg_pulses'].fillna('[]')
data['ecg_path'] = data['ecg_path'].fillna('')

# 4. 범주형 변수 인코딩
le_breeds = LabelEncoder()
data['breeds_encoded'] = le_breeds.fit_transform(data['breeds'])


# 5. 세그먼트 데이터 특성 추출
def extract_segment_features(segment_column, prefix):
    features = pd.DataFrame()
    for i, segments in enumerate(segment_column):
        try:
            segments = ast.literal_eval(segments)
            if len(segments) > 0:
                values = [seg['value'] for seg in segments]
                features.loc[i, f'{prefix}_mean'] = np.mean(values)
                features.loc[i, f'{prefix}_max'] = np.max(values)
                features.loc[i, f'{prefix}_min'] = np.min(values)
                features.loc[i, f'{prefix}_std'] = np.std(values) if len(values) > 1 else 0
            else:
                features.loc[i, f'{prefix}_mean'] = 0
                features.loc[i, f'{prefix}_max'] = 0
                features.loc[i, f'{prefix}_min'] = 0
                features.loc[i, f'{prefix}_std'] = 0
        except (ValueError, SyntaxError):
            features.loc[i, f'{prefix}_mean'] = 0
            features.loc[i, f'{prefix}_max'] = 0
            features.loc[i, f'{prefix}_min'] = 0
            features.loc[i, f'{prefix}_std'] = 0
    return features


hr_features = extract_segment_features(data['segments_hr'], 'hr')
br_features = extract_segment_features(data['segments_br'], 'br')
data = pd.concat([data, hr_features, br_features], axis=1)


# 6. ECG 펄스 특성 추출
def extract_ecg_features(ecg_pulses_column):
    features = pd.DataFrame()
    for i, pulses in enumerate(ecg_pulses_column):
        try:
            pulses = ast.literal_eval(pulses)
            if len(pulses) > 1:
                intervals = np.diff(pulses)
                features.loc[i, 'ecg_pulse_interval_mean'] = np.mean(intervals)
                features.loc[i, 'ecg_pulse_interval_std'] = np.std(intervals)
            else:
                features.loc[i, 'ecg_pulse_interval_mean'] = 0
                features.loc[i, 'ecg_pulse_interval_std'] = 0
        except (ValueError, SyntaxError):
            features.loc[i, 'ecg_pulse_interval_mean'] = 0
            features.loc[i, 'ecg_pulse_interval_std'] = 0
    return features


ecg_features = extract_ecg_features(data['ecg_pulses'])
data = pd.concat([data, ecg_features], axis=1)


# 7. 불량 ECG 세그먼트 특성 추출
def calculate_bad_ecg_duration(bad_ecg_column):
    features = pd.DataFrame()
    for i, segments in enumerate(bad_ecg_column):
        try:
            segments = ast.literal_eval(segments)
            total_bad_time = sum([seg[1] - seg[0] for seg in segments]) if len(segments) > 0 else 0
            features.loc[i, 'bad_ecg_total_time'] = total_bad_time
        except (ValueError, SyntaxError):
            features.loc[i, 'bad_ecg_total_time'] = 0
    return features


bad_ecg_features = calculate_bad_ecg_duration(data['bad_ecg'])
data = pd.concat([data, bad_ecg_features], axis=1)


# 8. 목표 변수 생성 (예시)
def create_health_label(row):
    if row['age'] > 8 or row['weight'] < 5 or row['weight'] > 50:
        return 'Unhealthy'
    else:
        return 'Healthy'


data['HealthStatus'] = data.apply(create_health_label, axis=1)

# 레이블 인코딩
le_health = LabelEncoder()
data['HealthStatus_encoded'] = le_health.fit_transform(data['HealthStatus'])

print(le_health.classes_)  # ['Healthy' 'Unhealthy']

# 9. 특성 및 레이블 분리
feature_columns = [
    'breeds_encoded', 'age', 'weight',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std',
    'br_mean', 'br_max', 'br_min', 'br_std',
    'ecg_pulse_interval_mean', 'ecg_pulse_interval_std',
    'bad_ecg_total_time'
]

X = data[feature_columns]
y = data['HealthStatus_encoded']

# 10. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 11. 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 12. DNN 모델 구축 함수 정의
def build_dnn_model(input_dim, l2_strength=1e-4):
    model = models.Sequential()

    # 첫 번째 은닉층
    model.add(layers.Dense(128, activation='relu', input_dim=input_dim,
                           kernel_regularizer=regularizers.l2(l2_strength)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # 두 번째 은닉층
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_strength)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # 세 번째 은닉층
    model.add(layers.Dense(32, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_strength)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # 출력층
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


# 13. 모델 생성 및 컴파일
input_dim = X_train_scaled.shape[1]
model = build_dnn_model(input_dim)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 14. 콜백 정의
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# 15. 모델 학습
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 16. 모델 평가
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy:.4f}')

# 17. 추가 평가 지표
y_pred_probs = model.predict(X_test_scaled)
y_pred_classes = (y_pred_probs > 0.5).astype(int).reshape(-1)

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_health.classes_, yticklabels=le_health.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 분류 보고서
print(classification_report(y_test, y_pred_classes, target_names=le_health.classes_))


# 18. 학습 곡선 시각화
def plot_history(history):
    # 손실 그래프
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


plot_history(history)

# 19. 모델 저장
model.save('Vital_model.h5')

# 20. 모델 로드 및 예측 예시
loaded_model = tf.keras.models.load_model('Vital_model.h5')

# 새로운 데이터 예시 (실제 값으로 대체)
new_data = np.array([
    [
        le_breeds.transform(['Labrador'])[0],  # breeds_encoded
        5,  # age
        25,  # weight
        80,  # hr_mean
        100,  # hr_max
        60,  # hr_min
        10,  # hr_std
        30,  # br_mean
        40,  # br_max
        20,  # br_min
        5,  # br_std
        0.8,  # ecg_pulse_interval_mean
        0.1,  # ecg_pulse_interval_std
        0  # bad_ecg_total_time
    ]
])

# 데이터 정규화
new_data_scaled = scaler.transform(new_data)

# 예측
prediction = loaded_model.predict(new_data_scaled)
predicted_class = 'Unhealthy' if prediction[0][0] > 0.5 else 'Healthy'

print(f'Predicted Health Status: {predicted_class}')
