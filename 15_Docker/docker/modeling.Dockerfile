FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements/modeling.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/modeling /workspace/src/modeling

CMD ["python", "/workspace/src/modeling/train_model.py", "--train-path", "/workspace/data/raw/mission15_train.csv", "--test-path", "/workspace/data/raw/mission15_test.csv", "--shared-dir", "/workspace/data/shared"]
