# streamlit-drawable-canvas 사용 가이드

> 학습 날짜: <!-- 직접 기입 -->

## 기본 사용법
```python
from streamlit_drawable_canvas import st_canvas

result = st_canvas(
    fill_color="rgba(0,0,0,0)",   # 투명 채우기
    stroke_width=20,               # 펜 두께
    stroke_color="#000000",        # 펜 색상
    background_color="#FFFFFF",    # 배경색
    height=280,
    width=280,
    drawing_mode="freedraw",       # 자유 그리기
    key="canvas",
)
```

## 반환값
```python
result.image_data   # numpy ndarray — shape (H, W, 4), dtype uint8, RGBA
result.json_data    # 그리기 객체 정보 (dict)
```

## 빈 캔버스 감지
```python
is_empty = result.image_data[:, :, 3].sum() == 0
# 알파채널(A)이 전부 0 → 아무것도 안 그린 상태
```

## MNIST 전처리 시 주의사항
- 캔버스: **흰 배경(255) + 검정 선(0)**
- MNIST 모델 학습 데이터: **검정 배경(0) + 흰 선(255)**
- → 반전 필수: `inverted = 255 - gray_array`

## 학습 포인트
<!-- 실습 후 채워넣기 -->
-
-
