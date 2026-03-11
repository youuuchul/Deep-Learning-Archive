# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 미션 개요 (심화)

`mnist_cnn.onnx` 모델을 **JavaScript (Node.js)** 환경에서 로드하여, 제공된 이미지 3장을 추론하고 예측 결과를 출력하는 심화 미션.

## 프로젝트 구조

```
16_Model-conversion/
├── data/
│   ├── mnist_cnn.onnx                  # MNIST CNN 모델 (입력: 1×1×28×28, float32)
│   └── mission16_target_images/
│       ├── image1.png                  # 추론 대상 이미지 (28×28 grayscale)
│       ├── image2.png
│       └── image3.png
├── inference.js                        # 메인 추론 스크립트 (작성 대상)
├── package.json
├── plan.md
└── README.md                           # 제출용 (프로젝트 개요, 실행 방법)
```

## 개발 환경

```bash
node --version   # v18+ 필요
npm init -y
npm install onnxruntime-node jimp
```

- `onnxruntime-node`: ONNX 모델 로드 및 추론 (macOS ARM64 지원)
- `jimp`: 순수 JS 이미지 처리 (PNG → float32 배열 변환)

## 핵심 구현 패턴

```js
const ort = require('onnxruntime-node');
const Jimp = require('jimp');

async function infer(imagePath) {
  // 1. 이미지 로드 및 전처리
  const img = await Jimp.read(imagePath);
  img.resize(28, 28).grayscale();
  const inputData = new Float32Array(28 * 28);
  img.scan(0, 0, 28, 28, (x, y, idx) => {
    inputData[y * 28 + x] = img.bitmap.data[idx] / 255.0;  // 정규화
  });

  // 2. 텐서 생성 (NCHW: 1×1×28×28)
  const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);

  // 3. 추론
  const session = await ort.InferenceSession.create('./data/mnist_cnn.onnx');
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const output = results[session.outputNames[0]].data;

  // 4. argmax → 예측 레이블
  return Array.from(output).indexOf(Math.max(...output));
}
```

## ONNX 모델 스펙

- 입력: `float32[1, 1, 28, 28]` (NCHW 배치)
- 출력: `float32[1, 10]` (클래스별 logit)
- 전처리: grayscale, resize to 28×28, 픽셀값 / 255.0

## 제출물

- `inference.js` — 추론 스크립트
- `README.md` — 프로젝트 개요, 사용 라이브러리, 실행 방법
- 예측 결과 화면 캡처 이미지
