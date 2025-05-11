# Pet-i Vital Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)


## 프로젝트 개요
이 프로젝트의 목적은 차세대통신 혁신융합대학사업단에서 주관하고, 국민대학교와 울산과학대학교가 참여한 반려견 헬스케어 시스템 Pet-i 애플리케이션에 탑재하기 위함입니다.

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
