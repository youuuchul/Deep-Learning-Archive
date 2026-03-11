# Node.js와 JavaScript

- **날짜**: 2026-03-11
- **미션/프로젝트**: Mission 16 - 모델 변환 (inference.js)
- **카테고리**: 언어/라이브러리

---

## 한 줄 정의

JavaScript는 언어, Node.js는 그 언어를 브라우저 밖(서버·터미널)에서 실행하는 런타임 환경.

## JavaScript vs Node.js 관계

```
JavaScript (언어 명세)
  ├── 브라우저 환경  → Chrome V8 엔진이 실행 (DOM, fetch, localStorage 등)
  └── Node.js 환경  → 동일한 V8 엔진 + 파일시스템·네트워크·npm 추가
```

- 같은 문법(let, const, async/await, 화살표 함수 등) 사용
- **Node.js는 JavaScript를 서버/터미널에서 돌리기 위한 실행 환경**
- Python : Python 인터프리터 = JavaScript : Node.js (런타임 역할이 동일)

---

## 이번 프로젝트에서 쓴 핵심 패턴

### 1. CommonJS 모듈 시스템 (`require`)

```js
// Node.js의 모듈 불러오기 방식 (Python의 import와 동일 역할)
const path = require('path');               // 내장 모듈 (설치 불필요)
const ort = require('onnxruntime-node');    // npm 패키지
const Jimp = require('jimp');              // npm 패키지
```

> Python의 `import os`, `import numpy as np`와 1:1 대응.
> `'use strict';` 는 파일 최상단에 쓰면 엄격 모드 활성화 (오타·버그 조기 검출).

---

### 2. async/await - 비동기 처리

```js
// 파일 I/O, 네트워크, 모델 로드 등 시간이 걸리는 작업은 async/await 필요
async function preprocessImage(imagePath) {
  const img = await Jimp.read(imagePath);  // 이미지 파일 읽기 완료까지 대기
  // ...
}

async function main() {
  const session = await ort.InferenceSession.create(MODEL_PATH);  // 모델 로드 대기
  const label = await runInference(session, inputData);           // 추론 완료 대기
}

main().catch((err) => {
  console.error('오류 발생:', err);
  process.exit(1);  // 에러 시 프로세스 종료 (exit code 1)
});
```

> Python의 `async def` / `await` / `asyncio.run(main())`과 동일한 개념.
> `.catch()` = Python의 `try/except`.

---

### 3. `path.join` - 경로 조합

```js
const MODEL_PATH = path.join(__dirname, 'data', 'mnist_cnn.onnx');
// __dirname: 현재 파일이 있는 디렉토리의 절대 경로 (자동 제공)
// 결과: /Users/.../16_Model-conversion/data/mnist_cnn.onnx
```

> Python의 `os.path.join(os.path.dirname(__file__), 'data', 'mnist_cnn.onnx')`와 동일.
> OS별 경로 구분자(/ vs \\) 차이를 자동 처리.

---

### 4. TypedArray - 고정 타입 배열

```js
// Python의 numpy array와 유사 - 타입이 고정된 효율적인 배열
const inputData = new Float32Array(28 * 28);  // 784개의 float32 값

// 값 채우기 (img.scan = 픽셀 순회)
img.scan(0, 0, 28, 28, (x, y, idx) => {
  inputData[y * 28 + x] = img.bitmap.data[idx] / 255.0;
  //         └─ row-major 인덱싱 (numpy의 [y, x]와 동일 순서)
});
```

| JS TypedArray | numpy dtype |
|--------------|-------------|
| `Float32Array` | `np.float32` |
| `Float64Array` | `np.float64` |
| `Int32Array` | `np.int32` |
| `Uint8Array` | `np.uint8` |

---

### 5. argmax 구현

```js
const output = Array.from(results[session.outputNames[0]].data);  // Float32Array → 일반 배열
const label = output.indexOf(Math.max(...output));
//                            └─ 스프레드 연산자로 배열 펼쳐서 Math.max에 전달
```

> Python의 `np.argmax(output)` 1줄 = JS에서는 2단계로 구현.
> `Array.from()`: TypedArray → 일반 Array 변환 (indexOf 사용 위해 필요).

---

### 6. npm / package.json

```bash
npm install onnxruntime-node jimp  # 패키지 설치 (Python의 pip install)
npm install                        # package.json 기반 일괄 설치
node inference.js                  # 스크립트 실행 (Python의 python inference.py)
```

```json
// package.json (Python의 pyproject.toml / requirements.txt 역할)
{
  "dependencies": {
    "jimp": "^0.22.12",
    "onnxruntime-node": "^1.24.3"
  }
}
```

---

## 헷갈리는 점 / 주의사항

- **`const` vs `let`**: 재할당 없으면 `const` 사용 (Python은 구분 없음)
- **jimp 버전**: v0.22.x (CommonJS) vs v1.x (ESM, API 다름) → v0.22 고정 사용
- **세미콜론**: JS는 선택사항이지만, `'use strict'` 환경에서 명시 권장
- **`__dirname`**: Node.js 전용 전역변수, 브라우저 JS에는 없음
- **비동기 필수**: 파일 읽기/모델 로드는 반드시 `await` 없으면 빈 결과 반환

---

## 관련 개념

- `ONNX_Runtime_Node.md` - onnxruntime-node 상세 API
- `ONNX_Format.md` - ONNX 모델 포맷 구조
