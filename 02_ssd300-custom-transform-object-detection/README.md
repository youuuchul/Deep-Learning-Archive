⸻

## 📦 SSD300 Custom Transform Object Detection

SSD300 + VGG16 Backbone 기반의 객체 탐지(Object Detection) 프로젝트입니다.
본 프로젝트는 페이스 중심 Random Crop, LetterBox Resize, Bounding Box 좌표 보정 등
직접 구현한 Custom Transform Pipeline을 적용해
소규모 데이터셋에서도 안정적인 학습을 수행하는 것을 목표로 합니다.

----
## Data : https://www.kaggle.com/datasets/devdgohil/the-oxfordiiit-pet-dataset
⸻

## 🚀 Features

✅ SSD300(VGG16) 모델 사용
	•	torchvision.models.detection.ssd300_vgg16


--- 
## 회고
1. 전처리 과정에서 실제 파일 정보 - xml 어노테이션 파일 정보를 탐색하는 과정에서 혼선이 있었음. 파트를 처음부터 명확하게 쪼개서 eda 방향을 잡았어야 할듯

2. 비교용 모델 (aster R-CNN(ResNet50-FPN)) 진행하려다가 안했음

3. EDA 결과 개선 미반영
	- EDA 결과 기반한 커스텀 트랜스폼을 진행했으나, 학습에 오히려 악영향 (파싱-실제 데이터 불일치 유발로 추정)
4. eda부터 결론까지 프로젝트가 똑바로 안 돌아감...아직도 맘에 안들고 정리하면서 버린 것도 많은데 지나치게 딜레이 돼서 더 할 수가 없다 (모델 성능, 개선방향 일관성, 전처리 개선 전부 불만족..)
