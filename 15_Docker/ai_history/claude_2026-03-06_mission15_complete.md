# Mission 15 완료 보고 — Claude Sonnet 4.6

- **날짜**: 2026-03-06
- **모델**: claude-sonnet-4-6
- **도구**: Claude Code (claude.ai/claude-code)

---

## 1. 사용자 프롬프트 요약

Docker 기반 ML 협업 워크플로우 구성:
- 연구자 1: EDA + 모델링 + Docker 이미지 빌드 → Docker Hub 업로드
- 연구자 2: 연구자 1 이미지에서 model.pkl 획득 → Jupyter 추론 → result.csv

---

## 2. 주요 작업 및 판단 이력

### 2-1. EDA/모델링 노트북 개선 (`notebook/modeling/eda_modeling.ipynb`)

**문제**: `trendline='ols'` 사용 시 `statsmodels` 없어서 에러

**조치**:
- `uv pip install statsmodels` 설치
- 차트 가독성 전면 개선:
  - 산점도 4개 → 2×2 서브플롯 통합 + 피어슨 r 값 annotation
  - 박스플롯 `points='all'` → `points='outliers'` (7000점 렌더 방지)
  - 히스토그램 평균선 추가
  - 상관 히트맵 `RdBu_r` diverging 컬러맵 (-1~+1 고정)
  - 잔차 플롯 y=0 기준선 추가

**추가**: VIF(다중공선성) 분석 셀 삽입 — `num_features` NameError 수정 포함

### 2-2. Docker 이미지 빌드 및 Hub 업로드

```bash
docker compose build
docker compose run --rm modeling-trainer
# → data/shared/model.pkl (RMSE 2.010259 확인)

docker tag mission15/modeling:local youuchul/mission15-modeling:latest
docker push youuchul/mission15-modeling:latest
# → sha256:b7a747a01e6c2f67877f83b9ea7a2ec76754...
```

**Docker Hub URL**: https://hub.docker.com/r/youuchul/mission15-modeling

### 2-3. inference-notebook 환경 문제 해결

**문제 1**: `ModuleNotFoundError: No module named 'plotly'`
- 원인: `requirements/inference.txt`에 plotly 누락
- 조치: `plotly==5.24.1` 추가 후 `docker compose build --no-cache`

**문제 2**: VS Code Jupyter 연결 안 됨 (XSRF 오류)
- 원인: Jupyter Server의 XSRF 토큰 검증이 VS Code 연결 차단
- 조치: Dockerfile CMD에 `--ServerApp.disable_check_xsrf=True` 추가

### 2-4. 추론 노트북 재구성 (`notebook/inference/inference.ipynb`)

셀 순서 엉킴 + 구버전 셀 혼재 → 노트북 전체 재작성 (Write 도구)

**구조**:
- Step 0: Docker/로컬 경로 자동 분기
- Step 1: 공유 볼륨 파일 목록 + 연구자 1 metrics.json 확인
- Step 2: 모델 로드 + 파이프라인 구조 + 회귀계수 출력
- Step 3: 테스트 데이터 로드 + 결측치 확인
- Step 4: 추론 실행 + RMSE/MAE/R² 평가
- Step 5: 예측값 분포 + 실제 vs 예측 산점도 + 잔차 분포
- Step 6: result.csv 저장 + 최종 요약 테이블

### 2-5. 보고서 생성 (`report/report.html`)

실제 측정값 기반으로 HTML 보고서 구성:
- Plotly.js CDN 활용한 인터랙티브 차트 2개
- KPI 카드 (RMSE, R², MAE, 데이터 크기)
- 피처 상관계수 바차트 (실측값)
- 전처리 파이프라인 테이블
- 아키텍처 다이어그램 (컨테이너 흐름 + 볼륨 공유 + Docker Hub)
- PDF 출력 가능 형태

---

## 3. 최종 결과

### 모델 성능 (연구자 1)
| 지표 | 값 |
|---|---|
| RMSE | 2.010259 |
| MAE | 1.591205 |
| R² | 0.989281 |
| 학습 행 | 5,600 |
| 검증 행 | 1,400 |

### 주요 인사이트
- `Previous Scores` (r=0.914): 압도적으로 중요한 피처
- `Hours Studied` (r=0.374): 중간 수준
- `Sample Question Papers` (r=0.050): 거의 무관
- VIF 전 피처 < 5 → 다중공선성 없음
- 합성 데이터 특성상 R²=0.989로 높음 (과적합 아님)

### 추론 결과 (연구자 2)
- 테스트 데이터: 3,000 rows
- 산출물: `data/shared/result.csv`
- Test RMSE: 2.01 (Train과 동일 수준 → 일반화 확인)

---

## 4. 생성/수정된 파일 목록

| 파일 | 작업 |
|---|---|
| `notebook/modeling/eda_modeling.ipynb` | 차트 개선, VIF 추가, 마크다운 인사이트 |
| `notebook/inference/inference.ipynb` | 전체 재작성 (Step 0~6) |
| `docker/inference-notebook.Dockerfile` | plotly, disable_check_xsrf 추가 |
| `requirements/inference.txt` | plotly==5.24.1 추가 |
| `report/report.html` | HTML 보고서 신규 생성 |
| `README.md` | 전체 업데이트 (실측값 반영) |
| `ai_history/docker_concepts_guide.md` | Docker 개념 학습 자료 |

---

## 5. 학습 권장 사항

- **Docker 심화**: multi-stage build, .dockerignore 최적화, docker network
- **sklearn Pipeline**: FeatureUnion, custom transformer 작성
- **VIF / 다중공선성**: Ridge/Lasso 회귀와의 비교
- **Jupyter + Docker**: VS Code Dev Container 설정으로 더 안정적인 연결 가능
- **CI/CD**: GitHub Actions로 docker build + push 자동화
