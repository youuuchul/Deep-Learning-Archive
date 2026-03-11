# Mission 16 모델 변환 — claude-sonnet-4-6

- **날짜**: 2026-03-11
- **모델**: claude-sonnet-4-6
- **도구**: Python (uv, PyTorch, onnxruntime), Node.js (onnxruntime-node, jimp), Jupyter

---

## 1. 사용자 프롬프트 요약

- MNIST CNN 모델을 pth / quantized pth / onnx 3종 포맷으로 변환 및 저장
- Python으로 ONNX 추론 정확도 검증 (95% 이상 목표)
- 심화: Node.js 환경에서 ONNX 모델로 이미지 3장 추론
- README, study-notes, gitignore, ai_history 정리

---

## 2. 주요 작업 및 판단 이력

### 2-1. 기본 미션 - 모델 학습 및 변환 (`modeling.ipynb`)

**목표**: MNIST CNN 학습 후 3종 포맷 저장
**조치**: PyTorch `state_dict`, `quantize_dynamic`, `torch.onnx.export`
**결과**: `data/models/` 아래 파일 4개 생성

```
mission_16_mnist_cnn.pth
mission_16_mnist_cnn_quantized.pth
mission_16_mnist_cnn.onnx
mission_16_mnist_cnn.onnx.data
```

### 2-2. 기본 미션 - ONNX 추론 검증 (`inference.ipynb`)

**목표**: MNIST test set 10,000장 추론 → 95%+ 달성
**조치**: `onnxruntime.InferenceSession` + torchvision MNIST test loader
**결과**:
```
ONNX 모델 추론 결과
총 10,000장 중 9,909장 정답
정확도: 0.9909 (99.09%)
✓ 95% 이상 달성
```

### 2-3. 심화 미션 - Node.js 추론 (`inference.js`)

**목표**: JS 환경에서 ONNX 모델로 이미지 3장 추론
**조치**: `onnxruntime-node` + `jimp` v0.22, NCHW Float32Array 텐서 생성, argmax
**결과**:
```
image1.png → 예측: 8
image2.png → 예측: 3
image3.png → 예측: 2
추론 완료
```

### 2-4. gitignore 수정

**문제**: `data/models/`가 gitignore 포함 → 모델 파일 미업로드
**조치**: `.gitignore`에서 `data/models/` 라인 제거
**결과**: 모델 파일 git 추적 대상으로 전환

### 2-5. README 최종 정리

**내용 추가**:
- 실제 폴더 구조 (models/, screenshots/, study-notes/ 포함)
- 실행 순서 Step 1~3 명시
- 학습 개념 노트 테이블 추가

### 2-6. study-notes 추가

**신규 작성**: `study-notes/NodeJS_JavaScript.md`
- Node.js vs JavaScript 관계
- CommonJS require, async/await, path.join, Float32Array, argmax, npm 패턴
- Python 대응 코드 비교 형식으로 작성

---

## 3. 최종 결과

| 항목 | 결과 |
|------|------|
| PyTorch 모델 저장 | `mission_16_mnist_cnn.pth` |
| 양자화 모델 저장 | `mission_16_mnist_cnn_quantized.pth` |
| ONNX 변환 | `mission_16_mnist_cnn.onnx` |
| Python 추론 정확도 | **99.09%** (10,000장 기준) |
| Node.js 추론 결과 | image1→8, image2→3, image3→2 |

---

## 4. 생성/수정된 파일 목록

| 파일 | 작업 |
|------|------|
| `modeling.ipynb` | 기본 미션 - CNN 학습 + 3종 포맷 변환 |
| `inference.ipynb` | 기본 미션 - ONNX 추론 정확도 검증 |
| `inference.js` | 심화 미션 - Node.js ONNX 추론 |
| `.gitignore` | `data/models/` 제거 (모델 파일 포함) |
| `README.md` | 폴더 구조·실행 순서·결과·학습노트 최종 정리 |
| `study-notes/NodeJS_JavaScript.md` | 신규 - Node.js/JS 개념 + 코드 패턴 |
| `data/models/*.pth, *.onnx` | 생성 (gitignore 해제로 추적 가능) |

---

## 5. 학습 권장 사항

- **ONNX 모델 검증**: `onnx.checker.check_model()`로 그래프 무결성 확인 연습
- **정규화 불일치 주의**: Python 학습 시 `Normalize((0.1307,), (0.3081,))` 적용했다면 JS 추론에도 동일 적용 필요 (이번 모델은 `/255.0`만 사용)
- **jimp v1.x 마이그레이션**: API 변경으로 현재 v0.22 고정 사용 중, 장기적으로 v1.x 대응 필요
- **배치 추론 최적화**: 현재 이미지 1장씩 순차 처리 → `Promise.all`로 병렬화 가능
