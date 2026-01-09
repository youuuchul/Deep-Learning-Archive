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

---

## ⚙️ Environment | 환경 설정
이 프로젝트들은 주로 **Jupyter Notebook** 환경에서 작성되었습니다.
- Python 3.x
- PyTorch / Scikit-learn / Pandas / NumPy / Matplotlib 등

## 🚀 Usage | 실행 방법
각 폴더 내의 `.ipynb` 파일을 참고하여 코드를 순차적으로 실행할 수 있습니다.
데이터셋이나 베이스라인 코드는 `.gitignore` 설정에 의해 포함되어 있지 않을 수 있으므로, 각 프로젝트의 README를 참고하여 필요한 데이터를 준비해주시기 바랍니다.
