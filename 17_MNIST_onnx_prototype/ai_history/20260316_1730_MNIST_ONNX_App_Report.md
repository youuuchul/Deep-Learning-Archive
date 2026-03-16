# MNIST ONNX Streamlit App — Claude Sonnet 4.6

- **날짜**: 2026-03-16
- **모델**: claude-sonnet-4-6
- **도구**: Python 3.11, uv, Streamlit, ONNX Runtime, Docker, Streamlit Community Cloud

---

## 1. 사용자 프롬프트 요약

- ONNX MNIST 모델로 숫자를 인식하는 Streamlit 웹 앱 개발
- 차트 라이브러리: matplotlib → plotly 변경 요청
- 실시간 예측(버튼 제거), PNG 다운로드 저장 방식으로 UX 개선
- Streamlit Community Cloud 배포 (모노레포 구조)
- 보고서 HTML 작성 + 스크린샷 삽입 + Docker Hub URL 기재

---

## 2. 주요 작업 및 판단 이력

### 2-1. 프로젝트 구조 생성
**목표**: mission_guide.md 기반 폴더/파일 스캐폴딩
**조치**: `src/`, `data/models/`, `docker/`, `study-notes/`, `reports/` 생성. `pyproject.toml`에 plotly 포함, matplotlib 제외
**결과**: 전체 파일 트리 생성 완료

### 2-2. 저장 에러 수정 + UX 개선
**문제**: `st.rerun()` + placeholder scope 에러 / 세션 기반 저장 UX 미흡
**조치**: placeholder 패턴 제거 → `with col_right:` 직접 렌더링. `st.download_button`으로 PNG 다운로드 전환
**결과**: 에러 해결, 저장 시 예측 레이블+확률 바 합성 PNG 다운로드

### 2-3. 빈 캔버스 예측 0 처리
**문제**: `background_color="#FFFFFF"` 설정 시 alpha 채널이 처음부터 255 → 빈 캔버스도 모델에 전달됨
**조치**: `json_data["objects"]` 유무로 스트로크 감지 전환
**결과**: 입력 없으면 모든 클래스 확률 0.0 표시

### 2-4. Streamlit Community Cloud 배포
**문제**: 모노레포 구조 + requirements.txt 위치 이슈 (1차 배포 실패: plotly 못 찾음)
**조치**: `requirements.txt`를 `src/`(앱 파일과 동일 디렉토리)로 이동. `sys.path.insert`로 import 경로 해결. `model.py`에 urllib 자동 다운로드 추가
**결과**: https://minist-onnx-prototype.streamlit.app 정상 배포

### 2-5. 다크모드 버튼 가시성
**문제**: 모바일 다크모드에서 버튼 텍스트 안 보임
**조치**: `data-testid` 기반 CSS 주입 (`border + color: inherit`)
**결과**: 다크모드 대응 완료

### 2-6. 저장 이미지 합성
**목표**: 단순 28×28 PNG → 예측 결과 포함 합성 이미지
**조치**: `_to_png_bytes(image, label, probs)` 확장. PIL ImageDraw로 레이블+확률 바 합성
**결과**: 다운로드 PNG에 Prediction 텍스트 + 10클래스 확률 바 포함

---

## 3. 최종 결과

| 항목 | 결과 |
|------|------|
| 라이브 데모 | https://minist-onnx-prototype.streamlit.app |
| Docker Hub | https://hub.docker.com/r/youuchul/mnist-onnx |
| 보고서 | `reports/report.html` (PDF 저장 완료) |
| 커밋 수 | 10 commits (main 브랜치) |

---

## 4. 생성/수정된 파일 목록

| 파일 | 작업 |
|------|------|
| `src/app.py` | Streamlit 메인 — 실시간 추론, 다운로드, CSS |
| `src/model.py` | ONNX 세션 캐싱 + 자동 다운로드 |
| `src/preprocess.py` | RGBA → [1,1,28,28] 전처리 파이프라인 |
| `src/requirements.txt` | Streamlit Cloud 의존성 (앱 파일 동일 경로) |
| `docker/Dockerfile` | python:3.11-slim 기반 컨테이너 |
| `docker-compose.yml` | 로컬 실행 편의 |
| `pyproject.toml` | uv 의존성 (plotly 포함) |
| `data/models/mnist-12.onnx` | Apache 2.0 모델 (26KB) |
| `reports/report.html` | HTML 보고서 (스크린샷+배포URL 포함) |
| `reports/design_notes.md` | 가이드 대비 설계 결정 메모 |
| `screenshots/app_main.png` | 앱 전체 화면 |
| `screenshots/prediction_result.png` | 예측 결과 화면 |
| `study-notes/ONNX_Runtime_Python.md` | ONNX Runtime 학습 노트 |
| `study-notes/Streamlit_Canvas.md` | streamlit-drawable-canvas 학습 노트 |
| `README.md` | 데모 URL + Docker Hub + 폴더 구조 |
| `plan.md` | 실습 체크리스트 (외부 공유 Step 추가) |

---

## 5. 학습 권장 사항

- **ONNX 모델 직접 변환**: PyTorch/Keras 모델을 `torch.onnx.export`로 변환해보기 (Mission 16 연계)
- **Streamlit 상태 관리 심화**: `st.session_state` 패턴 vs fragment 재실행 제어 (`@st.fragment`)
- **배포 최적화**: Docker 멀티스테이지 빌드로 이미지 크기 줄이기
- **모델 교체 실험**: mnist-12 → 직접 학습한 CNN 모델로 교체해 정확도 비교
