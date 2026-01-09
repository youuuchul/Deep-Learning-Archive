# 📰 NLP Text Classification
### 20 Newsgroups 데이터셋을 활용한 텍스트 분류 프로젝트

---

## 📘 Overview | 프로젝트 개요

이 프로젝트는 **20 Newsgroups 데이터셋**을 활용하여 전통적인 기계학습 및 딥러닝 기반의 텍스트 분류 모델을 실습하고 분석하는 것을 목표로 합니다.
NLP의 기초적인 전처리 과정(Tokenization, Padding 등)부터 모델 학습 및 평가까지의 과정을 다룹니다.

## 🗂 Dataset | 데이터셋

### 📌 Dataset: 20 Newsgroups
- 뉴스그룹 문서 데이터로, 약 20,000개의 기사가 20께의 카테고리로 분류되어 있습니다.
- 텍스트 분류 및 클러스터링 벤치마크 데이터셋으로 널리 사용됩니다.

### 구성
- **Train/Test Split**: 학습 및 테스트 데이터 분리
- **Categories**: `rec.autos`, `rec.motorcycles`, `sci.med` 등 다양한 주제

## 🛠 Tech Stack | 기술 스택
- **Language**: Python
- **Libraries**: Scikit-learn, PyTorch (or TensorFlow/Keras), NLTK/Spacy (전처리)
- **Model**: Naive Bayes, Logistic Regression, or Deep Learning models (RNN/LSTM/BERT etc. - 노트북 내용에 따라 구체화)

## 🚀 Key Features | 핵심 내용
- 텍스트 전처리 및 벡터화 (TF-IDF, Word Embedding)
- 텍스트 분류 모델 구현 및 학습
- 모델 성능 평가 (Accuracy, F1-score)

---
## 📝 Usage | 사용법
`notebooks/nlp_text_classification.ipynb` 파일을 열어 실습 내용을 확인할 수 있습니다.
