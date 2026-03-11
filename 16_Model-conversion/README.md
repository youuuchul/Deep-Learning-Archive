# Mission 16 - 모델 변환 및 Node.js 추론 (심화)

## 프로젝트 개요

MNIST CNN 모델을 **3가지 포맷**으로 변환하고, **JavaScript (Node.js)** 환경에서 ONNX 모델을 실행하는 심화 미션.
Python 학습 환경이 없어도 서버·엣지 디바이스에서 AI 모델을 실행할 수 있음을 보여줍니다.

---

## 프로젝트 구조

```
16_Model-conversion/
├── data/
│   ├── mnist_cnn.onnx                      # 심화 미션용 모델 (제공된 파일)
│   ├── models/                             # 기본 미션 생성 모델
│   │   ├── mission_16_mnist_cnn.pth            # PyTorch state_dict
│   │   ├── mission_16_mnist_cnn_quantized.pth  # 양자화 버전 (INT8)
│   │   ├── mission_16_mnist_cnn.onnx           # ONNX 변환 모델
│   │   └── onnx_inference_sample.png           # 추론 샘플 시각화
│   └── mission16_target_images/           # 심화 추론 대상 이미지 3장
│       ├── image1.png
│       ├── image2.png
│       └── image3.png
├── modeling.ipynb                          # [기본 미션 1] CNN 학습 + 3종 포맷 변환
├── inference.ipynb                         # [기본 미션 2] ONNX 추론 정확도 검증 (Python)
├── inference.js                            # [심화 미션] Node.js ONNX 추론 스크립트
├── screenshots/
│   ├── onnx_inference_sample.png           # inference.ipynb 샘플 추론 시각화
│   └── nodejs_inference_result.png         # node inference.js 실행 결과 캡쳐
├── study-notes/                            # 학습 개념 정리
│   ├── NodeJS_JavaScript.md               # Node.js/JS 기초 + 프로젝트 코드 패턴
│   ├── ONNX_Format.md                     # ONNX 포맷 구조, opset
│   ├── ONNX_Runtime_Node.md               # Node.js ONNX Runtime API
│   └── Quantization.md                    # 양자화 개념, dynamic/static 차이
├── package.json
└── README.md
```

---

## 실행 순서

### Step 1. 기본 미션 - 모델 학습 및 변환 (`modeling.ipynb`)

```bash
# Python 가상환경 생성 (uv, Python 3.11)
uv venv --python 3.11
source .venv/bin/activate

# 의존성 설치
uv pip install torch torchvision onnx onnxruntime ipykernel plotly kaleido

# Jupyter 커널 등록
python -m ipykernel install --user --name mission16 --display-name "Mission16 (Python 3.11)"

# 노트북 실행 (커널: Mission16 선택)
jupyter notebook modeling.ipynb
```

**결과물**: `data/models/` 아래 `.pth` 2개, `.onnx` 1개 생성

---

### Step 2. 기본 미션 - ONNX 추론 검증 (`inference.ipynb`)

```bash
jupyter notebook inference.ipynb
```

**결과**: MNIST test set 10,000장 추론 → 정확도 **99.09%** 달성

---

### Step 3. 심화 미션 - Node.js 추론 (`inference.js`)

```bash
# Node.js v18+ 확인
node --version

# 의존성 설치
npm install

# 추론 실행
node inference.js
```

**결과**:
```
==================================================
Mission 16 심화 - MNIST ONNX 추론 (Node.js)
==================================================

image1.png → 예측: 8
image2.png → 예측: 3
image3.png → 예측: 2

추론 완료
```

---

## 기본 미션 요약

| 포맷 | 파일명 | 설명 |
|------|--------|------|
| PyTorch | `mission_16_mnist_cnn.pth` | 일반 state_dict 저장 |
| PyTorch (양자화) | `mission_16_mnist_cnn_quantized.pth` | `quantize_dynamic` INT8 (FC 레이어) |
| ONNX | `mission_16_mnist_cnn.onnx` | `torch.onnx.export` opset 11 |

- 학습: MNIST 60,000장, 5 에폭, Adam(lr=1e-3)
- ONNX 추론 정확도: **99.09%** (기준 95% 이상)

---

## 심화 미션 - Node.js 추론

### 사용 라이브러리

| 라이브러리 | 버전 | 역할 |
|-----------|------|------|
| `onnxruntime-node` | ^1.24.3 | ONNX 모델 로드 및 추론 (macOS ARM64 지원) |
| `jimp` | ^0.22.12 | 순수 JS 이미지 처리 (PNG → float32 배열) |

### 전처리 파이프라인

```
PNG 이미지
  → Jimp.read()           # 이미지 로드
  → .resize(28, 28)       # 28×28 리사이즈
  → .grayscale()          # 그레이스케일 변환
  → pixel / 255.0         # [0, 1] 정규화
  → Float32Array [784]    # 1차원 배열
  → Tensor [1, 1, 28, 28] # NCHW 배치 텐서
  → ONNX session.run()    # 추론 → logits [1, 10]
  → argmax                # 예측 레이블 (0~9)
```

---

## 학습 개념 노트

`study-notes/` 폴더에 미션 중 정리한 개념 노트:

| 파일 | 내용 |
|------|------|
| `NodeJS_JavaScript.md` | Node.js vs JS 관계, async/await, TypedArray, npm 등 프로젝트 코드 패턴 |
| `ONNX_Format.md` | ONNX 포맷 구조, opset, IR 버전 |
| `ONNX_Runtime_Node.md` | Node.js ONNX Runtime API 상세 |
| `Quantization.md` | 양자화 개념, dynamic/static 차이 |
