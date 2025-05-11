# IMU based vital signs health classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)

---

## 프로젝트 개요
이 프로젝트의 목적은 차세대통신 혁신융합대학사업단에서 주관하고, 국민대학교와 울산과학대학교가 참여한 반려견 헬스케어 시스템 Pet-i 애플리케이션에 탑재하기 위함입니다.
TensorFlow Keras 기반의 심층 신경망을 사용하며, 실제 데이터가 없을 경우 합성 데이터 생성 기능을 제공합니다.

---

## 주요 기능
- **데이터 준비**  
  - 실제 CSV 데이터(`pet_vital_data_with_labels.csv`) 로드  
  - 파일 미존재 시 합성 데이터(정상 vs. 비정상) 자동 생성  
- **전처리**  
  - `train_test_split(stratify=y)`로 훈련/테스트 분리  
  - `StandardScaler`로 특징 표준화  
- **모델 학습**  
  - 은닉층 2개 + Dropout(0.5) 적용  
  - `EarlyStopping` + `ModelCheckpoint` 콜백 활용  
- **모델 평가**  
  - 정확도, 정밀도, 재현율, F1-score 보고서 출력  
  - 혼동 행렬(confusion matrix) 시각화  
- **예측 예시**  
  - 임의로 생성된 신규 샘플 10개에 대해 확률 예측  
  - 다수결 기반 최종 상태 판정  
- **시각화**  
  - 훈련·검증 손실 및 정확도 곡선 그래프  

---

## 데이터셋
- **실제 데이터**: Zenodo “Dog Health Vitals Dataset”  
  - CSV(`pet_vital_data_with_labels.csv`) 내부에 `HeartRate`, `RespirationRate`, `Temperature`, `State`, `DogType`, `Label` 컬럼 포함  
  - 필요한 경우 [Dataset](https://zenodo.org/records/8020390)
- **합성 데이터**:  
  - 상태(휴식/걷기/뛰기)별 정상·비정상 범위 정의  
  - `num_normal_samples`, `num_abnormal_samples` 변수로 샘플 수 조정 가능  

---

## 설치 및 환경 설정
```bash
# 1. 저장소 복제
git clone https://github.com/dpeyvc/Pet-i-Vital-Model.git
cd Pet-i-Vital-Model

# 2. (선택) 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
````

## 프로젝트 구조

```
Pet-i-Vital-Model/
├── Pet-i-Vital-Model.ipynb       # 단계별 예제 노트북
├── main.py                       # 학습·평가 메인 스크립트
├── predict.py                    # 신규 샘플 예측 스크립트
├── pet_vital_data_with_labels.csv # 실제/합성 데이터셋 CSV
├── pet_vital_model.keras         # 저장된 최적 모델 가중치
├── scaler.save                   # 저장된 StandardScaler 객체
├── requirements.txt              # 파이썬 패키지 목록
└── README.md                     # 프로젝트 설명서
```

---

## 참고 자료

* Jarkoff et al., “Assessing the Accuracy of a Smart Collar for Dogs…” (bioRxiv, 2023)
* TensorFlow Keras [문서](https://www.tensorflow.org/guide/keras)
* Scikit-learn [문서](https://scikit-learn.org/stable/documentation.html)

---
