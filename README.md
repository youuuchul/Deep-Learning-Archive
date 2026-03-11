# 🧠 Deep Learning Study Repository

이 레포지토리는 Deep Learning과 관련된 다양한 주제의 스터디 및 실습 프로젝트를 모아둔 공간입니다.
각 폴더는 독립적인 프로젝트 혹은 실습 내용을 담고 있으며, Computer Vision, NLP, GAN 등 다양한 분야를 다루고 있습니다.

---

## 📂 Project Structure | 폴더 구조 및 개요

### 01. 🩺 [Pneumonia Diagnosis with Transfer Learning & Grad-CAM](./01_pneumonia-transfer-learning-gradcam)
- **주제**: 흉부 X-ray 이미지를 이용한 폐렴 분류 및 Grad-CAM 시각화
- **핵심 기술**: Transfer Learning (전이학습), Grad-CAM (XAI)
- **데이터셋**: Kaggle Chest X-Ray Images

### 02. 📦 [SSD300 Custom Transform Object Detection](./02_ssd300-custom-transform-object-detection)
- **주제**: SSD300 + VGG16 Backbone 기반 객체 탐지
- **핵심 기술**: Custom Data Augmentation (Random Crop, LetterBox Resize), SSD300
- **데이터셋**: Oxford-IIIT Pet Dataset

### 03. ⚽ [U-Net Linear Semantic Segmentation](./03_U-Net-football-semantic-segmentation)
- **주제**: U-Net 모델을 활용한 축구 경기장 영역 세그멘테이션
- **핵심 기술**: Semantic Segmentation, U-Net architecture
- **데이터셋**: Football Semantic Segmentation Dataset

### 09. 👗 [FashionMNIST cGAN](./09_fashionmnist-cgan)
- **주제**: Conditional GAN을 이용한 카테고리별 패션 아이템 생성
- **핵심 기술**: GAN, Conditional GAN (cGAN), Image Generation
- **데이터셋**: FashionMNIST

### 10. 📰 [NLP Text Classification](./10_NLP-Text-Classification)
- **주제**: 20 Newsgroups 데이터셋을 활용한 텍스트 분류 실습
- **핵심 기술**: NLP Preprocessing, Text Classification
- **데이터셋**: 20 Newsgroups

### 11. 🌐 [Seq2Seq Attention Translation](./11_seq2seq-attention-translation)
- **주제**: Seq2Seq + Attention 메커니즘 기반 한영 기계 번역
- **핵심 기술**: Seq2Seq, Attention Mechanism, NLP
- **데이터셋**: AI Hub 한국어-영어 번역 말뭉치

### 12. 📝 [NLP Summarization (KoBART)](./12_NLP_Summarization)
- **주제**: KoBART 기반 법률 문서 추상적 요약
- **핵심 기술**: Abstractive Summarization, KoBART, HuggingFace Transformers
- **데이터셋**: 한국어 법률 문서

### 13. 💬 [Sentiment Analysis: Full Fine-Tuning vs PEFT](./13_SentimentAnalysis_Project)
- **주제**: 한국어 쇼핑몰 리뷰 감성 분석 — Full Fine-Tuning과 LoRA(PEFT) 비교
- **핵심 기술**: BERT, LoRA, PEFT, Fine-Tuning
- **데이터셋**: 한국어 쇼핑몰 리뷰

### 14. 🏛️ [NTS Tax RAG](./14_NTS-Tax-RAG)
- **주제**: 국세청 자료 기반 연말정산 Q&A RAG 시스템 구축
- **핵심 기술**: RAG, LangChain, Vector DB (FAISS/Chroma), LLM Quantization
- **데이터셋**: 2024년 귀속 연말정산 종합 안내 PDF

### 15. 🐳 [Docker 기반 ML 협업 워크플로우](./15_Docker)
- **주제**: Docker를 활용한 ML 모델 패키징 및 협업 환경 구축 실습
- **핵심 기술**: Docker, Docker Compose, ML 모델 배포
- **데이터셋**: Student Performance Dataset

### 16. 🔄 [모델 변환 및 Node.js 추론](./16_Model-conversion)
- **주제**: MNIST CNN 모델을 3종 포맷으로 변환하고 JavaScript(Node.js) 환경에서 ONNX 추론
- **핵심 기술**: PyTorch 모델 변환 (.pth / 양자화 .pth / .onnx), ONNX Runtime, Node.js 크로스 플랫폼 추론
- **데이터셋**: MNIST

---

## ⚙️ Environment | 환경 설정
이 프로젝트들은 주로 **Jupyter Notebook** 환경에서 작성되었습니다.
- Python 3.x
- PyTorch / Scikit-learn / Pandas / NumPy / Matplotlib 등

## 🚀 Usage | 실행 방법
각 폴더 내의 `.ipynb` 파일을 참고하여 코드를 순차적으로 실행할 수 있습니다.
데이터셋이나 베이스라인 코드는 `.gitignore` 설정에 의해 포함되어 있지 않을 수 있으므로, 각 프로젝트의 README를 참고하여 필요한 데이터를 준비해주시기 바랍니다.
