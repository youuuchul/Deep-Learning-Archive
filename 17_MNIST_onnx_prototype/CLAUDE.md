# CLAUDE.md — MNIST ONNX Prototype

## 프로젝트 개요
ONNX MNIST 모델로 손글씨 숫자를 인식하는 Streamlit 웹 앱.
Docker Hub 배포까지 완성하는 것이 최종 목표.

## 모델 스펙
| 항목 | 값 |
|------|-----|
| 파일명 | `mnist-12.onnx` |
| 입력 노드 | `Input3` |
| 출력 노드 | `Plus214_Output_0` |
| 입력 shape | `[1, 1, 28, 28]` float32 |
| 출력 shape | `[1, 10]` logits |
| 정규화 | /255 (픽셀값 0~1) |
| 배경 규칙 | **검정 배경, 흰색 선** (캔버스는 반대 → 반전 필수) |

## 모듈 구조
```
src/
├── app.py        — Streamlit UI 진입점
├── model.py      — ONNX 세션 로드 + 캐싱 + 추론
├── preprocess.py — 캔버스 RGBA → [1,1,28,28] float32
└── storage.py    — session_state 기반 히스토리 관리
```

## 핵심 제약
- `@st.cache_resource`로 InferenceSession 캐싱 (재로드 방지)
- 캔버스 반전 필수: `255 - pixel_value`
- Plotly로 확률 막대차트 렌더링 (matplotlib 사용 금지)
- 빈 캔버스 체크: 알파채널 합 == 0 → 경고 표시

## 실행 방법
```bash
# 로컬
uv venv && source .venv/bin/activate
uv pip install -e .
streamlit run src/app.py

# Docker
docker build -f docker/Dockerfile -t mnist-onnx:latest .
docker run -p 8501:8501 mnist-onnx:latest
```

## 모델 다운로드
```bash
curl -L -o data/models/mnist-12.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
```
