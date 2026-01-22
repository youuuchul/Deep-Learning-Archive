📝 KoBART 기반 법률 문서 요약 (Legal Document Summarization)
이 프로젝트는 한국어 요약 모델인 KoBART를 활용하여 복잡하고 긴 법률 문서를 핵심 내용으로 요약하는 자연어 처리(NLP) 태스크를 수행합니다.

🚀 프로젝트 개요
모델: SKT-AI/kobart-base-v2

태스크: 추상적 요약 (Abstractive Summarization)

주요 라이브러리: transformers, datasets, PyTorch, evaluate

📂 폴더 구조

12_NLP_Summarization/
├── notebooks/
│   └── nlp-summarization.ipynb   # 데이터 전처리 및 모델 학습 코드
├── output/                       # 학습된 모델 체크포인트 및 최종 모델 설정
│   ├── final_best_model/         # 성능이 가장 좋은 모델 설정 파일
│   └── kobart_legal_summary/     # 체크포인트별 학습 기록
└── README.md

🛠 주요 기능 및 워크플로우
데이터 전처리: 법률 문서를 KoBART 입력 규격에 맞게 토큰화 및 클리닝.

모델 학습 (Fine-tuning):

Hugging Face의 Seq2SeqTrainer를 사용하여 효율적인 학습 수행.

Rouge Score를 이용한 평가지표 모니터링.

성능 평가: ROUGE-1, ROUGE-2, ROUGE-L 지표를 활용하여 요약 품질 검증.

추론 (Inference): 학습된 모델을 통해 새로운 법률 전문을 입력받아 요약문 생성.

📊 학습 설정
Batch Size: 4 (또는 환경에 맞춰 조정)

Learning Rate: 5e-5

Epochs: 3~5 내외

Max Input Length: 1024 tokens

⚠️ 참고 사항
용량 문제로 인해 실제 모델 가중치 파일(.safetensors, .pt)은 저장소에서 제외되었습니다.

학습된 모델을 사용하려면 notebooks/의 코드를 재실행하거나, 별도로 저장된 모델 파일을 output/ 폴더에 배치해야 합니다.