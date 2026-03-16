# Mission 17 — MNIST ONNX Prototype

손글씨 숫자(0~9)를 캔버스에 그리면 ONNX 모델이 실시간으로 인식하는 Streamlit 웹 앱.

## 데모

**[▶ 라이브 데모 바로가기](https://minist-onnx-prototype.streamlit.app)**

## 실행 방법

### 로컬
```bash
uv venv && source .venv/bin/activate
uv pip install -r src/requirements.txt

streamlit run src/app.py
```

### Docker
```bash
docker build -f docker/Dockerfile -t mnist-onnx:latest .
docker run -p 8501:8501 mnist-onnx:latest
```

## 기술 스택
- **모델**: ONNX MNIST-12 (from onnx/models)
- **추론**: onnxruntime
- **UI**: Streamlit + streamlit-drawable-canvas
- **차트**: Plotly
- **패키지 관리**: uv

## 모델 출처 및 라이선스

| 항목 | 내용 |
|------|------|
| 모델 | MNIST-12 (`mnist-12.onnx`) |
| 출처 | [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist) |
| 라이선스 | [Apache License 2.0](https://github.com/onnx/models/blob/main/LICENSE) |
| 변경 여부 | 원본 그대로 사용 (no modification) |

## 폴더 구조
```
17_MNIST_onnx_prototype/
├── src/          — 앱 소스코드 + requirements.txt
├── data/models/  — mnist-12.onnx (Apache 2.0)
├── docker/       — Dockerfile
└── plan.md       — 실습 체크리스트
```
