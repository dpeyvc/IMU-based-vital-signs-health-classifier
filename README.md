## 바이탈 사인 예측 모델 개발 및 검증 보고서

다학제 캡스톤 프로젝트에서 진행한 반려견 헬스케어 시스템 Pet-i 어플리케이션에 온디바이스로 탑재된 바이탈 사인 예측 모델 개발 과정과 검증 결과

### 1. 데이터셋 소개
- **데이터셋 출처**: [Dog Health Vitals Dataset (Zenodo)](https://zenodo.org/records/8020390)  
- **데이터셋 설명**:  

- **주요 컬럼**:
  - `_id`: 각 녹음 세션의 고유 식별자  
  - `ecg_path`: ECG 파일의 경로(WAV 형식)  
  - `duration`: 녹음 세션 지속 시간(초)  
  - `pet_id`: 개의 고유 식별자  
  - `breeds`: 개의 품종  
  - `weight`: 측정 당시 개의 몸무게(kg)  
  - `age`: 측정 당시 개의 나이(년)  
  - `segments_br`: 특정 시간 구간에서의 호흡률(배열)  
  - `segments_hr`: 특정 시간 구간에서의 심박수(배열)  
  - `ecg_pulses`: ECG 신호에서 검출된 심박 타이밍(배열)  
  - `bad_ecg`: 신호 품질이 좋지 않은 시간 구간(배열)  

### 2. 데이터 전처리
- **데이터 필터링**: `bad_ecg` 구간을 제거하여 노이즈 배제  
- **심박수(HR), 호흡률(RR) 데이터 추출**: `segments_hr`, `segments_br`에서 각 구간별 평균 값을 추출하여 타임 시리즈 데이터로 변환  
- **정규화**: 모든 데이터는 [0, 1] 범위로 정규화하여 모델의 안정성을 확보  
- **결측치 처리**: 결측 구간은 선형 보간법을 사용하여 보정  

### 3. 데이터 생성 및 전처리 코드
학습 데이터가 부족하여 결측치 처리에 어려움이 있어, 기존 데이터셋으로 학습한 이후에 특정 범위의 랜덤 값을 생성하여 파인튜닝 하였습니다.

```python
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

num_normal_samples = 500000
num_abnormal_samples = 50000

# 파일 경로 설정
data_file_path = 'pet_vital_data_with_labels.csv'
model_file_path = 'pet_vital_model.keras'
scaler_file_path = 'scaler.save'

if not os.path.exists(data_file_path):
    np.random.seed(42)
    dog_types = [0, 1, 2]
    dog_type_probs = [0.3, 0.5, 0.2]

    heart_rate_ranges = {
        0: {'rest': (100, 180)},
        1: {'rest': (100, 140)},
        2: {'rest': (60, 100)}
    }

    heart_rate_multiplier = {
        0: {'rest': (1.0, 1.0), 'walk': (1.3, 1.5), 'run': (1.5, 2.0)},
        1: {'rest': (1.0, 1.0), 'walk': (1.3, 1.5), 'run': (1.5, 2.0)},
        2: {'rest': (1.0, 1.0), 'walk': (1.3, 1.5), 'run': (1.5, 2.0)}
    }

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
```

### 5. 결론 및 향후 계획
- **결론**: 본 모델은 PPG 데이터 기반으로 심박수, 산소포화도, 호흡수를 기반으로 반려견의 현재 건강상태(정상, 비정상) 예측 결과를 효과적으로 예측할 수 있습니다. 향후 더 많은 데이터셋을 사용하여 모델을 고도화할 계획입니다.
- **추가 개선 사항**:  
  - 온디바이스로 탑재된 모델의 경량화를 위한 양자화
  - 실시간 추론을 위한 모바일 환경 모델 변환
      -> ONNX 변환 예정

### 참고 문헌
- Dog Health Vitals Dataset (Zenodo)  
- TensorFlow 공식 문서  
