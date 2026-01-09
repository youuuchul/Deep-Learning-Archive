# FashionMNIST Conditional GAN (cGAN)

## 0. 프로젝트 개요

- **목적**
  - FashionMNIST 데이터셋을 활용해, 각 클래스(티셔츠, 바지, 스니커즈 등)에 대해
    조건부로 이미지를 생성하는 cGAN(Conditional GAN) 모델을 구현한다.
  - 단순한 이미지 생성이 아니라, **“레이블 조건에 맞는 패션 아이템을 생성하는지”**를 확인하는 것이 핵심 목표다.

- **데이터**
  - `torchvision.datasets.FashionMNIST` 사용
  - 28×28, 흑백 이미지, 10개 클래스
  - Train: 60,000장 / Test: 10,000장

- **핵심 포인트**
  - 노이즈 벡터 `z` + 클래스 레이블 `y`(원-핫 or 임베딩)를 함께 입력하는 **조건부 Generator** 설계
  - 진짜 이미지와 레이블 쌍, 가짜 이미지와 레이블 쌍을 구분하는 **조건부 Discriminator** 설계
  - 에폭이 진행될수록, 각 클래스별로 “그럴듯한 패션 아이템”이 생성되는지 시각적으로 추적

---

## 1. 진행 플로우

1. **환경 설정 & 데이터 로드**
   - PyTorch, torchvision, matplotlib 등 기본 라이브러리 세팅
   - FashionMNIST train/test 로드, 텐서 변환 및 정규화
   - DataLoader 구성 (batch size, shuffle 등)

2. **데이터 EDA 및 시각화 (간단)**
   - 각 클래스별 샘플 이미지 몇 장씩 시각화
   - 라벨-이름 매핑 확인 (0~9 → T-shirt/top, Trouser, ...)

3. **cGAN 모델 설계**
   - **Generator (G)**
     - 입력: 노이즈 벡터 `z` + 클래스 조건 `y` (concat 또는 임베딩 후 concat)
     - 출력: 1×28×28 이미지
   - **Discriminator (D)**
     - 입력: 이미지 `x` + 클래스 조건 `y` (채널/공간 방향으로 결합)
     - 출력: 진짜/가짜 스칼라 확률
   - 손실 함수: Binary Cross Entropy (BCE) 기반 adversarial loss

4. **학습 루프 구현**
   - D 업데이트:  
     - real(x, y) → real label  
     - fake(G(z, y), y) → fake label
   - G 업데이트:  
     - fake(G(z, y), y)를 D가 real로 오판하도록 학습
   - 에폭마다:
     - loss_G, loss_D 로그
     - 고정된 `z`+`y`(여러 클래스)로 생성된 이미지 그리드 저장 → `outputs/samples/`

5. **결과 시각화 & 평가**
   - 에폭별 생성 이미지 비교 (초기 vs 후반)
   - 클래스별로 “레이블에 맞는 패션 아이템”이 잘 나오는지 정성 평가
   - (옵션) FID, IS 등 정량 평가지표 시도 가능

6. **정리**
   - cGAN에서 조건 정보가 들어가는 지점(G/D 모두)을 그림 & 텍스트로 설명
   - 학습 중 불안정성/모드 붕괴 여부, 개선 아이디어(데이터 증강, 아키텍처 변경 등) 기록
