# ONNX Runtime — Python 사용 가이드

> 학습 날짜: 2026-03-16

## 핵심 개념

### InferenceSession
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```
- 모델을 메모리에 로드하고 추론 엔진을 초기화
- 생성 비용이 크므로 **캐싱 필수** (`@st.cache_resource`)

### 입출력 노드 확인
```python
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

for out in session.get_outputs():
    print(out.name, out.shape, out.type)
```

### 추론 실행
```python
outputs = session.run(
    output_names=["Plus214_Output_0"],
    input_feed={"Input3": input_array}  # numpy float32
)
logits = outputs[0]  # shape [1, 10]
```

## MNIST-12 모델 스펙
| 항목 | 값 |
|------|-----|
| 입력 노드 | `Input3` |
| 입력 shape | `[1, 1, 28, 28]` float32 |
| 출력 노드 | `Plus214_Output_0` |
| 출력 shape | `[1, 10]` (logits) |

## Softmax 직접 구현
```python
def softmax(x):
    exp = np.exp(x - x.max())  # 수치 안정성: max를 빼서 overflow 방지
    return exp / exp.sum()
```

## 학습 포인트

- **노드명은 모델마다 다르다** — `get_inputs()` / `get_outputs()`로 반드시 먼저 확인. mnist-12의 경우 `Input3`, `Plus214_Output_0`으로 직관적이지 않은 이름
- **출력은 logits** — ONNX 모델은 softmax를 포함하지 않는 경우가 많아 직접 적용해야 함. 수치 안정성을 위해 `x - x.max()` 후 exp 계산
- **dtype 주의** — 입력은 반드시 `float32`. `float64`로 넣으면 런타임 에러 발생
- **캐싱 효과** — `@st.cache_resource`로 세션을 캐싱하면 첫 요청 이후 재로드 없이 즉시 추론 가능. 미사용 시 매 rerun마다 모델 재로드
- **Cloud 배포 대응** — 모델 파일이 없을 때 `urllib.request.urlretrieve`로 자동 다운로드하면 GitHub에 바이너리 없어도 배포 가능 (fallback 패턴)
