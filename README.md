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

```
