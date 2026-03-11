# 심화 미션 실행 계획

> 목표: Node.js에서 `mnist_cnn.onnx`로 이미지 3장 추론 후 예측 레이블 출력

---

## Step 1. 환경 초기화

```bash
cd 16_Model-conversion
npm init -y
npm install onnxruntime-node jimp
```

**완료 기준:** `node_modules/` 생성, `package.json`에 두 패키지 등록

---

## Step 2. 모델 입출력 구조 확인

```bash
node -e "
const ort = require('onnxruntime-node');
ort.InferenceSession.create('./data/mnist_cnn.onnx').then(s => {
  console.log('inputs:', s.inputNames);
  console.log('outputs:', s.outputNames);
});
"
```

**완료 기준:** inputNames, outputNames 콘솔 출력 확인

---

## Step 3. inference.js 작성

구현 순서:

1. `ort.InferenceSession.create()` — 모델 로드 (세션 재사용을 위해 함수 밖에서 1회만)
2. `Jimp.read()` → `resize(28,28)` → `grayscale()` — 이미지 전처리
3. 픽셀 스캔 → `Float32Array(784)` 생성, 픽셀값 / 255.0 정규화
4. `new ort.Tensor('float32', data, [1, 1, 28, 28])` — NCHW 텐서
5. `session.run({ inputName: tensor })` — 추론
6. `argmax(output)` — 예측 레이블 추출
7. 3장 순회 후 결과 출력

**완료 기준:**
```
image1.png → 예측: 7
image2.png → 예측: 2
image3.png → 예측: 1
```
(실제 숫자는 추론 결과에 따라 다름)

---

## Step 4. 실행 및 결과 캡처

```bash
node inference.js
```

터미널 출력 스크린샷 저장 → 제출용

---

## Step 5. README.md 작성

포함 내용:
- 프로젝트 개요 (1~2줄)
- 사용 언어 및 라이브러리 (Node.js, onnxruntime-node, jimp)
- 실행 방법 (`npm install` → `node inference.js`)

---

## 체크리스트

- [ ] Step 1: npm 환경 세팅
- [ ] Step 2: 모델 입출력 확인
- [ ] Step 3: inference.js 작성 및 동작 확인
- [ ] Step 4: 결과 캡처
- [ ] Step 5: README.md 작성

---

## 주의사항

- `jimp` v0.x와 v1.x API가 다름 → `npm install jimp@0.22` 으로 안정 버전 고정 권장
- 이미지가 이미 28×28이라도 `resize()` 명시적 호출 (모델 입력 보장)
- 세션은 1회 생성 후 3장 모두 재사용 (매 추론마다 create 하지 말 것)
