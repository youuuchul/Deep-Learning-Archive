# Mission 15 실행 계획 (업데이트: 2026-03-06)

## 0) 기준 문서
- `Mission_Guide.md` 시나리오 준수
- `CLAUDE.md` 기본 규칙 준수 (uv, 로깅, 구조 정돈, ai_history 기록)

## 1) 폴더 구조 (기능 중심)
- [x] `data/raw`, `data/shared` 기준으로 데이터 경로 통일
- [x] `src/modeling`, `src/inference`로 코드 분리
- [x] `notebook/modeling`, `notebook/inference`로 노트북 분리
- [x] `docker/`, `requirements/`, `report/`, `scripts/` 정리

## 2) 환경 설정
- [x] Python 버전 고정: `.python-version` (3.11)
- [x] uv 기반 로컬 실행 옵션 문서화 (`README.md`)
- [x] 역할별 의존성 분리
  - [x] `requirements/modeling.txt`
  - [x] `requirements/inference.txt`
  - [x] `pyproject.toml`

## 3) 도커 전략
- [x] 모델링 Dockerfile 작성 (`docker/modeling.Dockerfile`)
- [x] 추론/노트북 Dockerfile 작성 (`docker/inference-notebook.Dockerfile`)
- [x] `docker-compose.yml` 서비스 역할명으로 정리
  - [x] `modeling-trainer`
  - [x] `inference-notebook`
  - [x] `inference-batch`
- [x] 데이터 전달 전략 반영
  - [x] `data/shared` 볼륨으로 `model.pkl`, `mission15_test.csv`, `result.csv` 공유
  - [x] 필요 시 `docker cp` 대체 가능 (README 반영)

## 4) 모델링 작업
- [x] EDA + 전처리 + 회귀 모델링 스크립트 (`src/modeling/train_model.py`)
- [x] RMSE 계산 및 `metrics.json` 저장
- [x] `model.pkl` 저장
- [x] EDA 요약 저장 (`eda_summary.json`, `eda_summary.md`)
- [x] 노트북 버전 (`notebook/modeling/eda_modeling.ipynb`)

## 5) 추론 작업
- [x] 추론 스크립트 (`src/inference/run_inference.py`)
- [x] 공유 모델/테스트 데이터 활용
- [x] `result.csv` 생성
- [x] 추론 노트북 (`notebook/inference/inference.ipynb`)

## 6) 리포트/운영
- [x] 보고서 템플릿 (`report/REPORT_TEMPLATE.md`)
- [x] 데이터 다운로드 스크립트 (`scripts/download_data.sh`)
- [x] 작업 로그 (`ai_history/codex_2026-03-06_mission15_setup.md`)

## 7) 남은 작업 (교차작업 대상)
- [ ] 컨테이너 실제 실행 검증 (Docker daemon on 상태)
- [ ] RMSE/샘플 결과를 보고서 본문으로 확정
- [ ] Docker Hub 이미지 푸시 및 URL 기입
- [ ] 최종 PDF(2페이지 이내) 완성
