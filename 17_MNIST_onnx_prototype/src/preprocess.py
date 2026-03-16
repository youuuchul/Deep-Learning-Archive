"""캔버스 이미지 전처리 파이프라인."""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def canvas_to_model_input(image_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """streamlit-drawable-canvas 출력을 ONNX 모델 입력으로 변환합니다.

    처리 순서:
        1. RGBA → grayscale
        2. 흰 배경(캔버스) → 검정 배경 반전 (MNIST 형식)
        3. 28×28 리사이즈
        4. float32 변환 + /255 정규화
        5. [1, 1, 28, 28] reshape

    Args:
        image_data: 캔버스 RGBA 배열. shape (H, W, 4), dtype uint8.

    Returns:
        tuple:
            - model_input: ONNX 입력 배열. shape [1, 1, 28, 28], dtype float32.
            - preview: 시각화용 28×28 grayscale 배열. shape (28, 28), dtype float32.
    """
    # RGBA → grayscale (PIL 경유)
    pil_img = Image.fromarray(image_data.astype(np.uint8), mode="RGBA")
    gray = pil_img.convert("L")  # 0=검정, 255=흰색

    # 캔버스는 흰 배경에 검정 선 → MNIST는 검정 배경에 흰 선
    gray_arr = np.array(gray, dtype=np.float32)
    inverted = 255.0 - gray_arr

    # 28×28 리사이즈
    resized = Image.fromarray(inverted).resize((28, 28), Image.LANCZOS)
    resized_arr = np.array(resized, dtype=np.float32)

    # 정규화
    normalized = resized_arr / 255.0
    preview = normalized.copy()

    # ONNX 입력 shape
    model_input = normalized.reshape(1, 1, 28, 28).astype(np.float32)

    logger.debug("전처리 완료 — min=%.3f, max=%.3f", normalized.min(), normalized.max())
    return model_input, preview
