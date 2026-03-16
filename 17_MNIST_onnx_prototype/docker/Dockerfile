FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 복사 (레이어 캐싱 활용)
COPY pyproject.toml .

RUN pip install --no-cache-dir \
    streamlit \
    streamlit-drawable-canvas \
    onnxruntime \
    numpy \
    pillow \
    pandas \
    plotly

# 소스 + 모델 복사
COPY src/ ./src/
COPY data/models/mnist-12.onnx ./data/models/mnist-12.onnx

EXPOSE 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
