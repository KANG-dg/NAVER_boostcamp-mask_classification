# NAVER_boostcamp-mask_classification

### 개요:
- COVID-19의 전염성을 막기 위해서는 사회적 거리두기와 마스크 착용이 요구된다. 그러나 모든 사람의 마스크 착용 상태를 확인하는 것은 인력적으로 어려움이 있다.
- 이를 해결하기 위해, 마스크 착용 여부를 자동으로 판별할 수 있는 시스템 개발이 필요하게 되었다.
- 본 프로젝트에서는 얼굴 이미지를 통해 성별, 연령, 마스크 착용 여부를 판별하는 시스템 개발을 목적으로 한다.

### 주요 사용기법:
- 해당 task는 mask(normal, mask, incorrect), age(<30, 30< and <60, <60), gender(male, female) 을 각각 분류하여 18개(3*3*2) 로 분류하는 multi label classification task
- 해당 task를 잘 해결하기 위해 각 label별로 최적의 모델을 만들어 해결하는 방법으로 접근 하였다.
- 위 과정에서 여러 모델(VIT-L, Convnext_Large, Efficientnet_b0 and b3) 를 실험하였고 최종적으로 Efficientnet_b0를 채택하였다.
- 입력 이미지에서 배경을 제외한 얼굴 부분에 집중하기 위해 Augmentation으로 Center Crop을 사용하였다. 최종적으로 사용된 Augmentation은 다음과 같다.
  ```python
  train_transform=A.Compose([A.Resize(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH']),
                           A.CenterCrop(300, 220, p=1),
                           A.HorizontalFlip(p=0.3),
                           A.OneOf([
                               A.MotionBlur(p=1),
                               A.OpticalDistortion(p=1),
                               A.GaussNoise(p=1)
                           ], p=0.3),
                           A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                           A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                           ToTensorV2()])


- 데이터에 class imbalance가 심하여 validation 과정에서 Stratified KFold 를 사용하였다
- 각 Fold에서 얻어진 model에 대해 soft_voting을 수행하여 최종 output을 얻어냈다
- 3개 모델에서 얻어진 output을 조합하여 18개 class로 반환하였다.

### 결과 & 회고
- 전체에서 public score기준 10등을 하였다
- baseline으로 제공된 코드 에서는 모든 lable을 입력으로 받아 하나의 model을 통해 18개의 class로 분류 하는 방법을 사용 하였는데 3개 모델을 사용하는 방법에서는 어느정도 성능을 높일 수 있는 방법들을 실험 해 보았지만 시간 부족으로 base line에서는 그러지 못했다. 모델 구성을 조금더 시간적으로 효율적으로 할 필요가 있다고 느꼈다.
- 모델 선정에서 내 경험상 비슷한 task 에서 Convnext_Large가 좋은 성능을 낸 경우가 많았기에 해당 모델을 위주로 실험하였다 하지만 예상과 다르게 더 작은 모델인 Efficientnet_b0에서 더 높은 성능이 나왔다.
-  마찬가지로 Augmentation에서 시간 부족으로 경험적으로 좋은 결과를 얻었던 기법들을 사용하였다 이 경우 역시 입력 데이터의 특성을 잘 파악하여 더 많은 augmentation을 실험했으면 하는 아쉬움이 있다.
-  도메인에 대한 파악이 이루어지고 실험을 하는게 중요하다고 다시 한번 느꼈다.
