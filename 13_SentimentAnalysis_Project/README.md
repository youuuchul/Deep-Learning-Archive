# 쇼핑몰 리뷰 감성 분석 프로젝트 (Full Fine-Tuning vs. PEFT)

본 프로젝트는 한국어 쇼핑몰 리뷰 데이터를 활용하여 감성 분석 모델을 구축하고, 전통적인 Full Fine-Tuning과 PEFT(LoRA) 방식의 자원 효율성을 비교 분석합니다.

## 1. 프로젝트 목적
- 효율적인 한국어 NLP 모델 학습 전략 수립
- 학습 속도, 모델 용량, 정확도 간의 Trade-off 확인

## 2. 작업 환경 (Hybrid Workflow)
- 로컬(Local): Apple Silicon M1 (MPS 가속) - EDA 및 코드 무결성 검증용
- 클라우드(Cloud): Google Colab (CUDA 가속) - 실제 대규모 학습 및 분석용

## 3. 폴더 구조
- data/: 원본 JSON 및 전처리 데이터
- models/: 학습된 모델 체크포인트 및 가중치
- notebooks/: 메인 작업 노트북 (.ipynb - 마크다운 논리 구조 포함)
- logs/: 학습 로그(CSV) 및 기록 파일
- scripts/: (참고용) .py 소스 코드

## 4. 주요 기술 스택 (2026 기준)
- Framework: PyTorch
- Library: Transformers, PEFT, Datasets, Evaluate, Accelerate
- Strategy: LoRA (Low-Rank Adaptation)