# NAVER_boostcamp-mask_classification

### 개요:
- COVID-19의 전염성을 막기 위해서는 사회적 거리두기와 마스크 착용이 요구된다. 그러나 모든 사람의 마스크 착용 상태를 확인하는 것은 인력적으로 어려움이 있다.
- 이를 해결하기 위해, 마스크 착용 여부를 자동으로 판별할 수 있는 시스템 개발이 필요하게 되었다.
- 본 프로젝트에서는 얼굴 이미지를 통해 성별, 연령, 마스크 착용 여부를 판별하는 시스템 개발을 목적으로 한다.

### 주요 사용기법:
- 해당 task는 mask(normal, mask, incorrect), age(<30, 30< and <60, <60), gender(male, female) 을 각각 분류하여 18개(3*3*2) 로 분류하는 multi label classification task
- 해당 task를 잘 해결하기 위해 
