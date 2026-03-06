# Docker 개념 & 명령어 정리

> 작성일: 2026-03-06 | 미션 15 실습 기반

---

## 1. 핵심 개념 3층 구조

```
Dockerfile
    ↓ docker build
Image  (설계도 / 스냅샷)
    ↓ docker run / docker compose up
Container  (실행 중인 프로세스)
```

| 용어 | 비유 | 특징 |
|---|---|---|
| **Dockerfile** | 레시피 | 이미지 만드는 방법을 텍스트로 작성 |
| **Image** | 붕어빵 틀 | 불변(Immutable). 실행 전까지 아무것도 안 함 |
| **Container** | 붕어빵 | 이미지로 찍어낸 실행 인스턴스. 여러 개 동시 실행 가능 |

---

## 2. Dockerfile 문법

```dockerfile
# 베이스 이미지 (항상 첫 줄)
FROM python:3.11-slim

# 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 컨테이너 안 작업 디렉토리 설정
WORKDIR /workspace

# 호스트 파일 → 컨테이너로 복사
COPY requirements.txt /tmp/requirements.txt

# 이미지 빌드 시 실행할 명령어 (레이어 캐시됨)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 포트 문서화 (실제 개방은 compose에서)
EXPOSE 8888

# 컨테이너 시작 시 실행할 기본 명령어
CMD ["python", "main.py"]
```

### RUN vs CMD 차이

| | RUN | CMD |
|---|---|---|
| 실행 시점 | **빌드 시** (이미지 생성 중) | **런타임** (컨테이너 시작 시) |
| 용도 | 패키지 설치, 파일 생성 | 앱 실행 |
| 레이어 | 새 레이어 생성 | 레이어 없음 |

---

## 3. 레이어 & 캐시 원리

Dockerfile의 각 명령어는 **레이어**를 만든다.

```
FROM python:3.11-slim      ← Layer 1 (캐시됨)
WORKDIR /workspace         ← Layer 2 (캐시됨)
COPY requirements.txt ...  ← Layer 3 (requirements 변경 시 캐시 무효화)
RUN pip install ...        ← Layer 4 (Layer 3 변경 시 재실행)
COPY src/ ...              ← Layer 5
```

**핵심 규칙**: 변경이 잦은 파일(소스코드)은 아래, 변경이 적은 것(requirements)은 위에 배치 → 빌드 속도 최적화

```bash
# 캐시 무시하고 처음부터 빌드
docker build --no-cache .
```

---

## 4. Docker 명령어 치트시트

### 이미지 관련

```bash
# 이미지 빌드 (현재 디렉토리의 Dockerfile 사용)
docker build -t myimage:tag .

# 특정 Dockerfile 지정
docker build -f docker/modeling.Dockerfile -t myimage:tag .

# 이미지 목록
docker images

# 이미지 삭제
docker rmi myimage:tag

# Docker Hub에서 이미지 받기
docker pull python:3.11-slim

# Docker Hub에 이미지 올리기
docker push youuchul/myimage:latest
```

### 컨테이너 관련

```bash
# 컨테이너 실행
docker run myimage:tag

# 백그라운드 실행 (-d), 포트 포워딩 (-p), 이름 지정 (--name)
docker run -d -p 8888:8888 --name mycontainer myimage:tag

# 실행 후 자동 삭제 (--rm)
docker run --rm myimage:tag

# 볼륨 마운트 (-v)
docker run -v ./data:/workspace/data myimage:tag

# 실행 중인 컨테이너 목록
docker ps

# 모든 컨테이너 목록 (정지 포함)
docker ps -a

# 컨테이너 중지
docker stop mycontainer

# 컨테이너 삭제
docker rm mycontainer

# 실행 중인 컨테이너 안으로 접속
docker exec -it mycontainer bash

# 컨테이너 로그 확인
docker logs mycontainer
docker logs -f mycontainer  # 실시간 follow
```

### Docker Hub

```bash
# 로그인
docker login

# 이미지에 Hub 계정명 태그 붙이기
docker tag localimage:tag yourhubid/reponame:tag

# 업로드
docker push yourhubid/reponame:tag
```

---

## 5. docker-compose 원리

여러 컨테이너를 **한 번에 정의하고 관리**하는 도구.

```yaml
# docker-compose.yml 구조
services:
  서비스명:                        # 컨테이너 그룹 이름
    build:
      context: .                   # Dockerfile 탐색 기준 경로
      dockerfile: docker/xxx.Dockerfile
    image: myimage:local           # 빌드된 이미지 이름
    container_name: mycontainer    # 실제 컨테이너 이름
    volumes:
      - ./data:/workspace/data     # 호스트경로:컨테이너경로
    ports:
      - "8888:8888"                # 호스트포트:컨테이너포트
    depends_on:
      - 다른서비스명               # 의존 서비스 먼저 시작
    command: ["python", "main.py"] # CMD 오버라이드
    restart: "no"                  # 재시작 정책
```

### docker compose 명령어

```bash
# 이미지 빌드
docker compose build

# 특정 서비스만 빌드
docker compose build inference-notebook

# 캐시 없이 재빌드
docker compose build --no-cache inference-notebook

# 컨테이너 시작 (포그라운드)
docker compose up

# 백그라운드 실행
docker compose up -d

# 특정 서비스만 실행
docker compose up inference-notebook

# 실행 후 자동 삭제 (1회성 작업에 유용)
docker compose run --rm modeling-trainer

# 컨테이너 중지 + 삭제 + 네트워크 삭제
docker compose down

# 볼륨까지 삭제
docker compose down -v

# 로그 확인
docker compose logs -f inference-notebook
```

---

## 6. 볼륨(Volume) 원리

컨테이너는 **종료하면 내부 파일이 사라진다.** 데이터를 유지하려면 볼륨 필요.

```
호스트 (맥)                    컨테이너
─────────────────              ──────────────────────
./data/raw/         ←──ro──→  /workspace/data/raw/
./data/shared/      ←──rw──→  /workspace/data/shared/
./notebook/         ←──rw──→  /workspace/notebook/
```

```yaml
volumes:
  - ./data/raw:/workspace/data/raw:ro   # ro = 읽기 전용
  - ./data/shared:/workspace/data/shared  # 기본 = 읽기+쓰기
```

**이번 미션에서 볼륨 활용:**
- `modeling-trainer`가 `/workspace/data/shared`에 `model.pkl` 저장
- `inference-notebook`이 같은 경로를 마운트 → 파일 공유

---

## 7. 포트 포워딩 원리

```
맥 브라우저/VS Code
      │
localhost:8888  (맥 포트)
      │
      │  ports: "8888:8888"  ← docker-compose.yml
      │
컨테이너:8888  (컨테이너 포트)
      │
jupyter lab --port=8888
```

컨테이너는 격리된 네트워크 안에 있어서 **밖에서 직접 접근 불가**.
포트 포워딩이 "맥의 8888번 요청을 컨테이너의 8888번으로 전달"하는 다리 역할.

---

## 8. 이번 미션 15 구조 전체 흐름

```
[연구자 1]
Dockerfile (modeling)
    ↓ docker compose build
Image: mission15/modeling:local
    ↓ docker compose run --rm modeling-trainer
Container 실행
    → train.csv 읽기 (볼륨: data/raw)
    → 모델 학습
    → model.pkl 저장 (볼륨: data/shared)
    → metrics.json 저장
    → test.csv 복사
    ↓ docker tag + docker push
Docker Hub: youuchul/mission15-modeling:latest

[연구자 2]
docker-compose.yml (inference-notebook 서비스)
    ↓ docker compose up inference-notebook
Container 실행
    → jupyter lab on :8888
    → localhost:8888 접속
    → model.pkl 로드 (볼륨: data/shared)
    → test.csv 로드
    → 추론 실행
    → result.csv 저장 (볼륨: data/shared)
```

---

## 9. 자주 쓰는 디버깅 명령어

```bash
# 컨테이너 안에 직접 접속해서 확인
docker exec -it mission15-inference-notebook bash

# 컨테이너 안에서 파일 확인
ls /workspace/data/shared/

# 컨테이너 안에서 패키지 확인
pip list | grep plotly

# 실시간 로그 보기
docker compose logs -f inference-notebook

# 컨테이너 리소스 사용량
docker stats

# 이미지/컨테이너/볼륨 전체 정리 (주의: 전부 삭제)
docker system prune -a
```

---

## 10. 핵심 요약

| 상황 | 명령어 |
|---|---|
| 코드 수정 후 반영 | `docker compose build` → `docker compose up` |
| requirements 수정 | `docker compose build --no-cache` |
| 컨테이너 완전 초기화 | `docker compose down` → `docker compose up` |
| Docker Hub 배포 | `docker tag` → `docker push` |
| 1회성 스크립트 실행 | `docker compose run --rm 서비스명` |
| 컨테이너 내부 확인 | `docker exec -it 컨테이너명 bash` |
