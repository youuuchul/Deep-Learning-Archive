# Mission 17 — 기본 미션 체크리스트

## Step 1: 환경 세팅
- [ ] uv venv 생성: `uv venv`
- [ ] 의존성 설치: `uv pip install -e .`
- [ ] ONNX 모델 다운로드
  ```bash
  curl -L -o data/models/mnist-12.onnx \
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
  ```
- [ ] 모델 파일 존재 확인: `ls -lh data/models/`

## Step 2: model.py 동작 확인
- [ ] ONNX 세션 로드 테스트 (Python REPL)
  ```python
  from src.model import load_session
  sess = load_session()
  print(sess.get_inputs(), sess.get_outputs())
  ```
- [ ] 입력 노드명 `Input3`, 출력 노드명 `Plus214_Output_0` 확인

## Step 3: preprocess.py 동작 확인
- [ ] 임시 numpy 배열로 전처리 결과 shape 확인
  ```python
  import numpy as np
  from src.preprocess import canvas_to_model_input
  dummy = np.ones((280, 280, 4), dtype=np.uint8) * 255
  inp, prev = canvas_to_model_input(dummy)
  print(inp.shape, inp.dtype)  # (1, 1, 28, 28) float32
  ```

## Step 4: 앱 실행 (로컬)
- [ ] `streamlit run src/app.py`
- [ ] http://localhost:8501 접속
- [ ] 캔버스에 숫자 그리기 → 자동 예측 → 확률 차트 확인
- [ ] 저장 버튼 → PNG 다운로드 확인

## Step 5: 외부 공유 URL 만들기

로컬 앱을 다른 사람이 웹으로 접속할 수 있게 하는 방법 두 가지.

### 방법 A — ngrok (즉시, 임시 URL)
로컬에서 앱을 켠 채로 외부에서 접속 가능한 임시 HTTPS URL을 만들어줌.
앱을 끄면 URL도 사라짐 (발표/데모용으로 적합).

```bash
# 1. ngrok 설치 (macOS)
brew install ngrok

# 2. ngrok 계정 가입 후 인증토큰 등록 (최초 1회)
#    https://dashboard.ngrok.com/get-started/your-authtoken
ngrok config add-authtoken <YOUR_TOKEN>

# 3. Streamlit 앱 실행 중인 상태에서 별도 터미널에서:
ngrok http 8501
```

- 출력된 `Forwarding` 주소(예: `https://xxxx.ngrok-free.app`)를 다른 사람에게 공유
- [ ] ngrok 설치 확인: `ngrok --version`
- [ ] 앱 실행 + ngrok 터널 열기
- [ ] 외부 브라우저에서 접속 확인

### 방법 B — Streamlit Community Cloud (영구, 무료) ← 채택
GitHub에 코드를 올린 후 Streamlit 공식 클라우드에 배포.
고정 URL 제공, 모노레포(서브폴더) 지원.

**사전 준비 (이미 완료)**
- `model.py` — 모델 파일 없으면 자동 다운로드 (urllib)
- `requirements.txt` — `17_MNIST_onnx_prototype/requirements.txt` 생성
- `app.py` — `sys.path`에 `src/` 경로 추가 (Cloud 실행 환경 대응)

**배포 순서**
```
1. 코드 커밋 & GitHub 푸시
   git add 17_MNIST_onnx_prototype/
   git commit -m "Add Mission 17 MNIST app"
   git push

2. https://share.streamlit.io 접속 → GitHub 로그인

3. "Create app" 클릭 후 설정:
   - Repository:     <your-github-id>/<repo-name>
   - Branch:         main
   - Main file path: 17_MNIST_onnx_prototype/src/app.py

4. "Deploy!" 클릭
   → 첫 실행 시 mnist-12.onnx 자동 다운로드 (약 30초)
   → 이후 캐싱되어 빠르게 로드
```

- [ ] GitHub에 코드 푸시
- [ ] share.streamlit.io 에서 배포 설정
- [ ] 배포 완료 후 공개 URL 확인 (예: `https://xxxx.streamlit.app`)
- [ ] 외부 브라우저에서 접속 확인

## Step 6: UI 스크린샷
- [ ] 전체 화면 스크린샷 → `screenshots/app_main.png`
- [ ] 예측 결과 클로즈업 → `screenshots/prediction_result.png`

## Step 7: Docker 빌드 및 로컬 테스트
- [ ] 이미지 빌드
  ```bash
  docker build -f docker/Dockerfile -t mnist-onnx:latest .
  ```
- [ ] 컨테이너 실행
  ```bash
  docker run -p 8501:8501 mnist-onnx:latest
  ```
- [ ] http://localhost:8501 에서 정상 동작 확인

## Step 8: Docker Hub 배포
- [ ] Docker Hub 로그인: `docker login`
- [ ] 태그 지정: `docker tag mnist-onnx:latest <your-id>/mnist-onnx:latest`
- [ ] 푸시: `docker push <your-id>/mnist-onnx:latest`
- [ ] Docker Hub URL 메모: `https://hub.docker.com/r/<your-id>/mnist-onnx`

## Step 9: 보고서 작성
- [ ] 프로젝트 개요
- [ ] 코드 설명 (model.py / preprocess.py / app.py)
- [ ] 전처리 파이프라인 다이어그램
- [ ] 앱 스크린샷 삽입
- [ ] 공유 URL (ngrok 또는 Streamlit Cloud) 포함
- [ ] Docker Hub URL 포함
- [ ] PDF 내보내기 → `reports/report.pdf`
