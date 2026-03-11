# NodeJS_ONNX_Inference

- **날짜**: 2026-03-11
- **프로젝트**: Mission 16 - 모델 변환 (inference.js)
- **난이도**: 중급

---

## 핵심 개념

`onnxruntime-node`는 C++ ONNX Runtime 엔진의 Node.js 바인딩. Python `onnxruntime`과 동일한 모델 파일을 그대로 사용하며, `jimp`로 이미지를 전처리해 `Float32Array` TypedArray로 변환한 뒤 NCHW 텐서를 만들어 추론한다. Python 없이 서버·엣지에서 AI 모델을 실행할 수 있는 대표 패턴.

---

## 코드 예시

```js
'use strict';

const path = require('path');
const ort = require('onnxruntime-node');
const Jimp = require('jimp');  // v0.22.x (CommonJS)

const MODEL_PATH = path.join(__dirname, 'data', 'mnist_cnn.onnx');
const IMG_SIZE = 28;

// 이미지 전처리 → Float32Array
async function preprocessImage(imagePath) {
  const img = await Jimp.read(imagePath);
  img.resize(IMG_SIZE, IMG_SIZE).grayscale();

  const inputData = new Float32Array(IMG_SIZE * IMG_SIZE);
  img.scan(0, 0, IMG_SIZE, IMG_SIZE, (x, y, idx) => {
    // idx: RGBA 배열에서 해당 픽셀의 R 채널 인덱스
    inputData[y * IMG_SIZE + x] = img.bitmap.data[idx] / 255.0;
  });

  return inputData;
}

// 텐서 생성 → 추론 → argmax
async function runInference(session, inputData) {
  const tensor = new ort.Tensor('float32', inputData, [1, 1, IMG_SIZE, IMG_SIZE]);

  // session.inputNames[0]으로 동적으로 피드 키 설정 (모델마다 다름)
  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);

  const output = Array.from(results[session.outputNames[0]].data);
  return output.indexOf(Math.max(...output));  // argmax
}

async function main() {
  // 세션 1회 로드, 여러 이미지 재사용
  const session = await ort.InferenceSession.create(MODEL_PATH);

  for (const filename of ['image1.png', 'image2.png', 'image3.png']) {
    const inputData = await preprocessImage(`data/mission16_target_images/${filename}`);
    const label = await runInference(session, inputData);
    console.log(`${filename} → 예측: ${label}`);
  }
}

main().catch((err) => {
  console.error('오류 발생:', err);
  process.exit(1);
});
```

---

## 주의점 / 함정 (Gotchas)

- **jimp 버전**: `jimp@0.22.x`는 CommonJS(`require`), `jimp@1.x`는 ESM(`import`) → API 구조가 달라 혼용 불가. `package.json`에 `"jimp": "^0.22.12"` 고정 권장
- **정규화 일치**: Python 학습 시 `Normalize((0.1307,), (0.3081,))`를 적용했다면 JS에서도 동일하게 적용해야 함. 이번 모델은 `/255.0`만 사용
- **Float32Array → Array 변환**: `argmax`를 위해 `Array.from()` 필요. TypedArray는 `indexOf` 없음
- **`__dirname`**: 현재 파일의 절대 경로. `path.join(__dirname, ...)`으로 상대 경로 문제 방지
- **세션 로드 비용**: `InferenceSession.create()`는 무거움 → 루프 밖에서 1회만 생성

---

## 언제 쓰고, 언제 쓰지 말 것

| 사용 O | 사용 X |
|--------|--------|
| Python 없는 서버/엣지에서 ONNX 추론 | GPU 가속 필요한 대형 모델 (CUDA 미지원) |
| 웹 백엔드(Express)에 AI 추론 통합 | 브라우저 환경 (→ `onnxruntime-web` 사용) |
| 경량 모델 CPU 추론 | 실시간 배치 처리 대량 이미지 (Python 대비 느림) |

---

## 참고 / 출처

- 실제 코드: `16_Model-conversion/inference.js`
- 개념 노트: `16_Model-conversion/study-notes/NodeJS_JavaScript.md`
- ONNX Runtime 노트: `16_Model-conversion/study-notes/ONNX_Runtime_Node.md`
