"""MNIST ONNX Streamlit 앱 — 메인 진입점."""

import io
import logging
import sys
from pathlib import Path

# Streamlit Community Cloud는 레포 루트에서 실행하므로 src/ 경로를 명시 추가
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from model import load_session, predict
from preprocess import canvas_to_model_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MNIST 숫자 인식기",
    page_icon="✏️",
    layout="wide",
)

# 다크모드 버튼 가시성 보정
st.markdown(
    """
    <style>
    /* secondary 버튼: 다크모드에서 테두리+텍스트 흰색으로 */
    [data-testid="stBaseButton-secondary"] {
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        color: inherit !important;
    }
    /* download 버튼도 동일 적용 */
    [data-testid="stBaseButton-downloadButton"] {
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MNIST 숫자 인식기")
st.caption("캔버스에 숫자(0~9)를 그리면 ONNX 모델이 자동으로 인식합니다.")

# ── 모델 초기화 ───────────────────────────────────────────────────────────────
try:
    session = load_session()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── session_state 초기화 ─────────────────────────────────────────────────────
for key, default in [
    ("last_preview", None),
    ("last_probs", None),
    ("last_pred_label", None),
    ("last_png_bytes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _to_png_bytes(image: np.ndarray) -> bytes:
    """28×28 float32 배열을 PNG bytes로 변환."""
    uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(uint8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _make_chart(probs: np.ndarray | None, pred_label: int | None) -> go.Figure:
    """확률 막대차트 생성. probs가 None이면 모두 0."""
    y = probs.tolist() if probs is not None else [0.0] * 10
    text = [f"{p:.1%}" for p in y] if probs is not None else [""] * 10
    colors = [
        "#EF553B" if (probs is not None and i == pred_label) else "#636EFA"
        for i in range(10)
    ]
    fig = go.Figure(
        go.Bar(
            x=[str(i) for i in range(10)],
            y=y,
            marker_color=colors,
            text=text,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="클래스별 예측 확률",
        xaxis_title="숫자",
        yaxis_title="확률",
        yaxis_range=[0, 1.1],
        height=350,
        margin={"t": 40, "b": 20},
    )
    return fig


# ── 레이아웃 ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("입력 캔버스")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# ── 자동 예측 ────────────────────────────────────────────────────────────────
# alpha 채널 합이 아닌 json_data objects로 체크
# (background_color가 흰색이면 alpha가 처음부터 255이므로 alpha 체크 불가)
has_drawing = (
    canvas_result.json_data is not None
    and len(canvas_result.json_data.get("objects", [])) > 0
)

if has_drawing:
    image_data: np.ndarray = canvas_result.image_data
    model_input, preview = canvas_to_model_input(image_data)
    probs = predict(session, model_input)
    pred_label = int(probs.argmax())

    st.session_state.last_preview = preview
    st.session_state.last_probs = probs
    st.session_state.last_pred_label = pred_label
    st.session_state.last_png_bytes = _to_png_bytes(preview)
else:
    st.session_state.last_preview = None
    st.session_state.last_probs = None
    st.session_state.last_pred_label = None
    st.session_state.last_png_bytes = None

# ── 우측: 결과 렌더링 ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("예측 결과")

    has_result = st.session_state.last_preview is not None

    # 전처리 이미지 (예측 있을 때만)
    if has_result:
        st.image(
            st.session_state.last_preview,
            caption="전처리 이미지 (28×28)",
            width=140,
            clamp=True,
        )

    # 막대차트: 항상 표시, 입력 없으면 모두 0
    st.plotly_chart(
        _make_chart(st.session_state.last_probs, st.session_state.last_pred_label),
        use_container_width=True,
    )

    # 예측 레이블 + 💾 다운로드 버튼
    if has_result:
        pred_label = st.session_state.last_pred_label
        probs = st.session_state.last_probs

        col_metric, col_dl = st.columns([1, 1])
        with col_metric:
            st.metric(
                label="예측 결과",
                value=str(pred_label),
                delta=f"확률 {probs[pred_label]:.1%}",
            )
        with col_dl:
            st.download_button(
                label="저장",
                data=st.session_state.last_png_bytes,
                file_name=f"mnist_{pred_label}.png",
                mime="image/png",
                use_container_width=True,
                type="primary",
            )
    else:
        st.caption("숫자를 그리면 예측이 시작됩니다.")
