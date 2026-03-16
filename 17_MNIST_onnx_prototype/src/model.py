"""ONNX 모델 로드 및 세션 캐싱 모듈."""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "mnist-12.onnx"
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated"
    "/vision/classification/mnist/model/mnist-12.onnx"
)

# mnist-12.onnx 입출력 노드명
INPUT_NAME = "Input3"
OUTPUT_NAME = "Plus214_Output_0"


def _download_model() -> None:
    """모델 파일이 없으면 GitHub ONNX 저장소에서 자동 다운로드합니다."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("모델 다운로드 중: %s", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("다운로드 완료: %s (%.1f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)


@st.cache_resource
def load_session() -> ort.InferenceSession:
    """ONNX InferenceSession을 로드하고 캐싱합니다.

    모델 파일이 없으면 자동으로 다운로드합니다 (Streamlit Cloud 대응).

    Returns:
        ort.InferenceSession: 로드된 ONNX 추론 세션.
    """
    if not MODEL_PATH.exists():
        _download_model()
    logger.info("ONNX 세션 로드: %s", MODEL_PATH)
    return ort.InferenceSession(str(MODEL_PATH))


def predict(session: ort.InferenceSession, input_array: np.ndarray) -> np.ndarray:
    """ONNX 세션으로 추론을 수행합니다.

    Args:
        session: 로드된 ONNX 추론 세션.
        input_array: 전처리된 입력 배열. shape [1, 1, 28, 28], dtype float32.

    Returns:
        np.ndarray: 소프트맥스 적용된 확률 배열. shape [10].
    """
    logits = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]  # [1, 10]
    logits = logits[0]  # [10]
    # softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs
