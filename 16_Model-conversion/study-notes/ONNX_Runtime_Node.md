# ONNX Runtime (Node.js)

## 개요

`onnxruntime-node`는 Microsoft의 ONNX Runtime을 Node.js에서 사용하기 위한 공식 패키지. 네이티브 바인딩 방식으로 C++ ONNX Runtime 엔진을 JS에서 호출.

## 설치

```bash
npm install onnxruntime-node   # CPU 실행 (macOS ARM64, x64, Linux, Windows 지원)
```

## 핵심 API

### 세션 생성

```js
const ort = require('onnxruntime-node');

// 파일 경로로 로드
const session = await ort.InferenceSession.create('./model.onnx');

// 메타데이터 조회
console.log(session.inputNames);   // ['input']
console.log(session.outputNames);  // ['output']
```

### 텐서 생성

```js
// Tensor(type, data, dims)
const tensor = new ort.Tensor(
  'float32',                  // 데이터 타입
  new Float32Array(784),      // TypedArray
  [1, 1, 28, 28]             // shape (NCHW)
);
```

### 추론 실행

```js
const feeds = { [session.inputNames[0]]: tensor };
const results = await session.run(feeds);
const output = results[session.outputNames[0]].data;  // Float32Array
```

## jimp@0.22 이미지 전처리

```js
const Jimp = require('jimp');

const img = await Jimp.read('image.png');
img.resize(28, 28).grayscale();

const data = new Float32Array(784);
img.scan(0, 0, 28, 28, (x, y, idx) => {
  // bitmap.data: RGBA 배열, idx=픽셀 시작 인덱스
  data[y * 28 + x] = img.bitmap.data[idx] / 255.0;
});
```

**jimp 버전 주의**: v0.22 (CommonJS) vs v1.x (ESM, API 변경)
- `Jimp.read()` 반환값 구조가 v1.x에서 달라짐
- 안정적인 v0.22.x 사용 권장

## MNIST 전처리 파이프라인 상세

```
PNG → RGBA bitmap (Jimp)
  → resize(28, 28): 비율 무시 강제 리사이즈
  → grayscale(): R=G=B (평균법)
  → R channel / 255.0: [0, 1] float32 정규화
  → Float32Array[784]: 행 우선 (row-major)
  → Tensor[1, 1, 28, 28]: 배치 1, 채널 1, 28×28
```

**주의**: 이 모델은 `(pixel / 255.0)` 정규화만 사용. PyTorch 학습 시 `Normalize((0.1307,), (0.3081,))`를 적용했다면 JS에서도 동일하게 적용해야 함.

```js
// 평균/표준편차 정규화가 필요한 경우
const MEAN = 0.1307;
const STD = 0.3081;
data[y * 28 + x] = (img.bitmap.data[idx] / 255.0 - MEAN) / STD;
```

## Execution Providers

```js
// 기본: CPUExecutionProvider
const session = await ort.InferenceSession.create('./model.onnx');

// CUDA (NVIDIA GPU)
const session = await ort.InferenceSession.create('./model.onnx', {
  executionProviders: ['cuda', 'cpu']
});

// CoreML (macOS Apple Silicon)
const session = await ort.InferenceSession.create('./model.onnx', {
  executionProviders: ['coreml', 'cpu']
});
```

## 성능 특성

- CPU 추론: MNIST 28×28 단일 이미지 ~1ms 이하
- 세션 로드: 첫 로드 시 수백ms (모델 크기 비례)
- 배치 재사용: 세션을 반복 사용하면 두 번째 추론부터 빠름
