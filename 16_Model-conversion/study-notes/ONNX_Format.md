# ONNX 포맷 (Open Neural Network Exchange)

## 개요

ONNX는 ML 모델의 **중간 표현(IR)** 포맷. PyTorch/TensorFlow 등으로 학습한 모델을 프레임워크 독립적으로 배포 가능하게 함.

## 왜 ONNX인가

```
[학습 프레임워크]     [배포 런타임]
  PyTorch     ──┐
  TensorFlow  ──┤──→ ONNX ──→ onnxruntime (CPU/GPU)
  MXNet       ──┘           ──→ TensorRT (NVIDIA GPU)
                            ──→ OpenVINO (Intel)
                            ──→ CoreML (Apple)
                            ──→ Web (ONNX.js)
                            ──→ Node.js (onnxruntime-node)
```

## PyTorch → ONNX 변환

```python
torch.onnx.export(
    model,                          # 모델 (eval 모드)
    dummy_input,                    # 입력 텐서 예시 (shape 추론용)
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={                  # 동적 배치 크기 지원
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11                # 연산자 집합 버전 (11 = 안정적)
)
```

## 모델 구조 (Protobuf)

```
ModelProto
  ├── ir_version        # ONNX IR 버전
  ├── opset_imports     # 사용 연산자 집합
  └── graph (GraphProto)
        ├── node[]      # 연산 노드 (Conv, Relu, FC 등)
        ├── input[]     # 입력 텐서 명세
        ├── output[]    # 출력 텐서 명세
        └── initializer[] # 가중치 (상수 텐서)
```

## ONNX 검증

```python
import onnx
model = onnx.load('model.onnx')
onnx.checker.check_model(model)   # 유효성 검사 (통과 시 예외 없음)
```

## Opset 버전 선택 기준

| Opset | 특징 |
|-------|------|
| 9~10  | 구버전 호환성 높음, 일부 최신 연산 미지원 |
| 11    | **권장** - 안정적, 대부분 런타임 지원 |
| 13+   | 최신 연산자 지원, 구버전 런타임 미지원 가능 |

## MNIST CNN 모델 스펙

- 입력: `float32[batch, 1, 28, 28]` (NCHW)
- 출력: `float32[batch, 10]` (클래스 logit)
- Opset: 11
- 크기: ~1.7MB (Conv 가중치 포함)
