'use strict';

/**
 * Mission 16 심화 - MNIST CNN ONNX 추론 (Node.js)
 *
 * 입력: data/mission16_target_images/{image1,image2,image3}.png
 * 모델: data/mnist_cnn.onnx (float32[1,1,28,28] → float32[1,10])
 * 출력: 각 이미지의 예측 숫자 (0~9)
 */

const path = require('path');
const ort = require('onnxruntime-node');
const Jimp = require('jimp');

const MODEL_PATH = path.join(__dirname, 'data', 'mnist_cnn.onnx');
const IMAGE_DIR = path.join(__dirname, 'data', 'mission16_target_images');
const IMAGE_FILES = ['image1.png', 'image2.png', 'image3.png'];

const IMG_SIZE = 28;

/**
 * 이미지를 로드하여 MNIST 전처리 후 Float32Array 반환.
 *
 * @param {string} imagePath - 입력 이미지 경로
 * @returns {Promise<Float32Array>} 길이 784 (28×28) 정규화된 grayscale 픽셀값
 */
async function preprocessImage(imagePath) {
  const img = await Jimp.read(imagePath);
  img.resize(IMG_SIZE, IMG_SIZE).grayscale();

  const inputData = new Float32Array(IMG_SIZE * IMG_SIZE);
  img.scan(0, 0, IMG_SIZE, IMG_SIZE, (x, y, idx) => {
    // RGBA에서 R 채널만 사용 (grayscale 후 R=G=B)
    inputData[y * IMG_SIZE + x] = img.bitmap.data[idx] / 255.0;
  });

  return inputData;
}

/**
 * ONNX 세션으로 단일 이미지 추론.
 *
 * @param {ort.InferenceSession} session - 로드된 ONNX 세션
 * @param {Float32Array} inputData - 전처리된 픽셀값 (784개)
 * @returns {Promise<number>} 예측 레이블 (0~9)
 */
async function runInference(session, inputData) {
  const tensor = new ort.Tensor('float32', inputData, [1, 1, IMG_SIZE, IMG_SIZE]);
  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  const output = Array.from(results[session.outputNames[0]].data);

  // argmax → 예측 레이블
  return output.indexOf(Math.max(...output));
}

async function main() {
  console.log('='.repeat(50));
  console.log('Mission 16 심화 - MNIST ONNX 추론 (Node.js)');
  console.log('='.repeat(50));
  console.log(`모델: ${MODEL_PATH}`);
  console.log('');

  // ONNX 세션 1회 로드 (3장 공용)
  const session = await ort.InferenceSession.create(MODEL_PATH);
  console.log(`입력 형태: ${session.inputNames[0]} ${JSON.stringify(session.inputNames)}`);
  console.log('');

  // 3장 순차 추론
  for (const filename of IMAGE_FILES) {
    const imagePath = path.join(IMAGE_DIR, filename);
    const inputData = await preprocessImage(imagePath);
    const label = await runInference(session, inputData);
    console.log(`${filename} → 예측: ${label}`);
  }

  console.log('');
  console.log('추론 완료');
}

main().catch((err) => {
  console.error('오류 발생:', err);
  process.exit(1);
});
