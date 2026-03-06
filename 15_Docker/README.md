# Mission 15 — Docker 기반 ML 협업 워크플로우

학생 성적 데이터(Student Performance)를 이용한 선형 회귀 모델링 및 Docker 기반 배포 실습

**Docker Hub**: `youuchul/mission15-modeling:latest`

---

## 개요
- 가상 협업 시나리오 기반의 Docker 실습 위주로 진행하였습니다.
  - EDA 및 인사이트 도출 최소화
  - Docker 이미지/컨테이너 생성/도커 Hub 업로드 등 Docker 관련 작업 터미널에서 직접 진행
  - 시나리오 기반으로 실험자2가 로컬 파일 및 가상환경 이용하지 않도록 통제/검토
  - 리포트 클로드 활용하여 html로 생성
  - 코덱스/에이전트 활용 로그 ai_history 폴더 기록

## 폴더 구조

```
15_Docker/
├── docker/
│   ├── modeling.Dockerfile            # 연구자 1 이미지 (학습)
│   └── inference-notebook.Dockerfile  # 연구자 2 이미지 (추론 + Jupyter)
├── src/
│   ├── modeling/train_model.py        # 전처리 → 학습 → model.pkl 저장
│   └── inference/run_inference.py     # 배치 추론 스크립트
├── notebook/
│   ├── modeling/eda_modeling.ipynb    # EDA + 모델링 노트북 (연구자 1)
│   └── inference/inference.ipynb      # 추론 노트북 (연구자 2)
├── data/
│   ├── raw/                           # 원본 데이터 (gitignore 대상 아님)
│   │   ├── mission15_train.csv
│   │   └── mission15_test.csv
│   └── shared/                        # 컨테이너 간 볼륨 공유 경로
│       ├── model.pkl                  # 학습된 모델 (gitignore)
│       ├── metrics.json               # 평가 지표 (gitignore)
│       └── result.csv                 # 추론 결과 (gitignore)
├── requirements/
│   ├── modeling.txt                   # 연구자 1 패키지 (버전 고정)
│   └── inference.txt                  # 연구자 2 패키지 (버전 고정)
├── report/
│   └── report.html                    # 보고서 (PDF 출력용)
├── ai_history/                        # AI 작업 이력
├── scripts/
│   └── download_data.sh               # 데이터 다운로드 스크립트
├── docker-compose.yml                 # 서비스 정의 및 볼륨 설정
├── pyproject.toml                     # 로컬 개발 환경 설정
└── .python-version                    # Python 3.11 고정
```

---

## 결과 요약

| 지표 | 값 |
|---|---|
| 모델 | LinearRegression (sklearn Pipeline) |
| RMSE | **2.010** |
| MAE | **1.591** |
| R² | **0.989** |
| 학습 데이터 | 7,000 rows (train 5,600 / valid 1,400) |
| 테스트 추론 | 3,000 rows → result.csv |

---

## 빠른 실행

### 1. 연구자 1: 모델 학습

```bash
# 이미지 빌드 + 학습 실행 (model.pkl, metrics.json 생성)
docker compose build
docker compose run --rm modeling-trainer
```

### 2. 연구자 2: 추론 노트북

```bash
# Jupyter Lab 실행
docker compose up inference-notebook

# 브라우저 접속
open http://localhost:8888
# → notebook/inference/inference.ipynb 열기 → Restart & Run All
```

### 3. Docker Hub에서 가져오기 (연구자 2 시나리오)

```bash
docker pull youuchul/mission15-modeling:latest
docker compose run --rm modeling-trainer   # model.pkl 생성
docker compose up inference-notebook
```

---

## 로컬(uv) 실행

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements/modeling.txt

# 학습
python src/modeling/train_model.py \
  --train-path data/raw/mission15_train.csv \
  --test-path data/raw/mission15_test.csv \
  --shared-dir data/shared

# 추론
python src/inference/run_inference.py \
  --model-path data/shared/model.pkl \
  --test-path data/shared/mission15_test.csv \
  --output-path data/shared/result.csv
```

---

## 환경 버전 (양 연구자 공통)

| 패키지 | 버전 |
|---|---|
| Python | 3.11 |
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| scikit-learn | 1.6.1 |

---

## 데이터 전달 전략

두 컨테이너가 `data/shared/` 디렉토리를 볼륨으로 공유합니다.

```
modeling-trainer  →  data/shared/  ←  inference-notebook
  (model.pkl 생성)    (볼륨 마운트)     (model.pkl 사용)
```

`docker cp` 대안:
```bash
docker compose run --name temp-trainer modeling-trainer
docker cp temp-trainer:/workspace/data/shared/model.pkl ./data/shared/
docker rm temp-trainer
```
