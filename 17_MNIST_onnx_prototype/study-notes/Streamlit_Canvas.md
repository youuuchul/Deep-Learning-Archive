# streamlit-drawable-canvas 사용 가이드

> 학습 날짜: 2026-03-16

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
result.json_data    # 그리기 객체 정보 (dict) — 스트로크 목록 포함
```

## 빈 캔버스 감지 — 올바른 방법

```python
# ❌ 잘못된 방법: 흰 배경(background_color="#FFFFFF")이면 alpha가 처음부터 255
is_empty = result.image_data[:, :, 3].sum() == 0  # 항상 False

# ✅ 올바른 방법: 실제 스트로크 객체 유무 확인
has_drawing = (
    result.json_data is not None
    and len(result.json_data.get("objects", [])) > 0
)
```

`background_color`를 흰색으로 설정하면 alpha 채널이 처음부터 255로 채워지므로
alpha 합 체크로는 빈 캔버스를 감지할 수 없다. `json_data.objects`가 신뢰할 수 있는 방법.

## MNIST 전처리 시 주의사항
- 캔버스: **흰 배경(255) + 검정 선(0)**
- MNIST 모델 학습 데이터: **검정 배경(0) + 흰 선(255)**
- → 반전 필수: `inverted = 255 - gray_array`

## 학습 포인트

- **alpha 체크 함정** — 투명 배경(`rgba(0,0,0,0)`)일 때만 alpha 합 체크가 유효. 흰 배경 사용 시엔 반드시 `json_data.objects`로 확인
- **실시간 추론 가능** — 캔버스는 획이 그려질 때마다 Streamlit rerun을 트리거하므로 별도 버튼 없이 자동 추론 가능
- **image_data는 항상 RGBA** — grayscale 변환은 PIL `.convert("L")`로 처리
- **json_data 구조** — `{"objects": [{"type": "path", "path": [...], ...}]}` 형태. 각 획이 하나의 object로 저장됨
- **key 파라미터 필수** — 같은 페이지에 여러 캔버스를 쓸 경우 고유한 key 지정 필요. 없으면 상태 충돌
