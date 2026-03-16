# ONNX Runtime — Python 사용 가이드

> 학습 날짜: <!-- 직접 기입 -->

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
    exp = np.exp(x - x.max())  # 수치 안정성
    return exp / exp.sum()
```

## 학습 포인트
<!-- 실습 후 채워넣기 -->
-
-
