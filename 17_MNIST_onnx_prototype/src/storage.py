"""이미지 저장소 — st.session_state 기반."""

import io
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image

HISTORY_KEY = "history"


@dataclass
class PredictionRecord:
    """단일 예측 기록."""

    image: np.ndarray    # 28×28 float32 (표시용)
    png_bytes: bytes     # PNG 바이너리 (다운로드용)
    label: int           # 예측 레이블 (0~9)
    confidence: float    # 최고 확률
    probs: np.ndarray    # 전체 확률 배열 [10]
    index: int           # 저장 순번 (파일명용)


def _to_png_bytes(image: np.ndarray) -> bytes:
    """28×28 float32 배열을 PNG bytes로 변환합니다."""
    uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(uint8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def init_storage() -> None:
    """session_state 히스토리 초기화 (최초 1회)."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []


def save_prediction(image: np.ndarray, label: int, probs: np.ndarray) -> None:
    """예측 결과를 히스토리에 추가합니다.

    Args:
        image: 28×28 float32 grayscale 배열.
        label: 예측된 숫자 레이블.
        probs: 클래스별 확률 배열 [10].
    """
    init_storage()
    index = len(st.session_state[HISTORY_KEY]) + 1
    record = PredictionRecord(
        image=image,
        png_bytes=_to_png_bytes(image),
        label=label,
        confidence=float(probs[label]),
        probs=probs,
        index=index,
    )
    st.session_state[HISTORY_KEY].append(record)


def get_history() -> list[PredictionRecord]:
    """저장된 예측 기록 전체를 반환합니다."""
    init_storage()
    return st.session_state[HISTORY_KEY]


def clear_history() -> None:
    """히스토리를 초기화합니다."""
    st.session_state[HISTORY_KEY] = []
