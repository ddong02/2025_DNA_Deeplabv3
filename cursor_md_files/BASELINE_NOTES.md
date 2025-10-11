# Baseline 코드 변경사항

## 개요

클래스 불균형 해결을 위한 고급 기법들을 제거하고, 기본적인 Cross-Entropy Loss만 사용하는 **Baseline 코드**로 변경했습니다.

이는 개선 방법 적용 전/후의 성능을 명확하게 비교하기 위한 기준점(baseline)을 확보하기 위함입니다.

## 제거된 기능들

### 1. ❌ 고급 손실 함수 제거
- **Dice Loss** (`my_utils/losses.py`의 `DiceLoss`)
- **Combined Loss** (Cross-Entropy + Dice의 가중합)
- 관련 인자: `--loss_type`, `--ce_weight`, `--dice_weight`

### 2. ❌ 클래스 가중치 제거
- **Class Weights 계산** (`calculate_class_weights` 함수)
- 다양한 가중치 계산 방법들:
  - `inverse_freq`: 역빈도 가중치
  - `sqrt_inv_freq`: 제곱근 역빈도
  - `effective_num`: Effective Number of Samples
  - `median_freq`: Median Frequency Balancing
- 관련 인자: `--use_class_weights`, `--weight_method`, `--effective_beta`

### 3. ✅ 유지된 기능들
- **Standard Cross-Entropy Loss**: `nn.CrossEntropyLoss(ignore_index=255)`
- **Early Stopping**: 조기 종료 기능은 그대로 유지
- **2-Stage Training**: Backbone freeze/unfreeze 전략 유지
- **WandB 로깅**: 실험 추적 기능 유지

## 현재 Baseline 설정

### 손실 함수
```python
criterion = nn.CrossEntropyLoss(
    ignore_index=255,      # 무시할 레이블 (배경)
    reduction='mean'       # 평균 손실
)
# 클래스 가중치 없음
# 고급 손실 함수 없음
```

### 특징
- ✅ **단순성**: 가장 기본적인 semantic segmentation 설정
- ✅ **재현성**: 표준 방법론으로 결과 재현이 용이
- ✅ **비교 기준**: 개선 방법의 효과를 명확하게 측정 가능

## 사용 방법

### 기본 훈련 명령어
```bash
python my_train.py \
    --dataset dna2025dataset \
    --data_root ./datasets/data \
    --model deeplabv3_mobilenet \
    --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
    --pretrained_num_classes 19 \
    --num_classes 19 \
    --epochs 200 \
    --unfreeze_epoch 15 \
    --lr 1e-5 \
    --batch_size 4 \
    --crop_size 1024 \
    --enable_vis \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_name "baseline-standard-ce" \
    --wandb_tags "baseline,standard-ce" \
    --save_val_results \
    --early_stop \
    --early_stop_patience 15
```

### WandB에서 확인할 내용
훈련 중 다음 지표들이 자동으로 기록됩니다:
- **Training Loss**: 에폭별 평균 훈련 손실
- **Validation Mean IoU**: 주요 평가 지표
- **Overall Accuracy**, **Mean Accuracy**: 보조 지표
- **Class별 IoU**: 각 클래스의 성능

## 개선 방법 적용 시나리오

Baseline 성능을 측정한 후, 다음과 같은 개선 방법들을 순차적으로 적용할 수 있습니다:

### 시나리오 1: Class Weights 적용
```python
# 1. calculate_class_weights import 추가
from my_utils.calculate_class_weights import calculate_class_weights

# 2. 가중치 계산
class_weights = calculate_class_weights(
    dataset=train_dst,
    num_classes=opts.num_classes,
    device=device,
    method='sqrt_inv_freq',  # 또는 'inverse_freq', 'effective_num', 'median_freq'
    ignore_index=255
)

# 3. Loss 함수에 적용
criterion = nn.CrossEntropyLoss(
    weight=class_weights,  # 추가
    ignore_index=255,
    reduction='mean'
)
```

### 시나리오 2: Dice Loss 적용
```python
# 1. Import
from my_utils.losses import DiceLoss

# 2. Dice Loss 사용
criterion = DiceLoss(
    smooth=1.0,
    ignore_index=255,
    weight=class_weights  # Optional
)
```

### 시나리오 3: Combined Loss 적용
```python
# 1. Import
from my_utils.losses import CombinedLoss

# 2. Combined Loss 사용
criterion = CombinedLoss(
    ce_weight=0.6,        # CE 비율
    dice_weight=0.4,      # Dice 비율
    class_weights=class_weights  # Optional
)
```

## 예상되는 Baseline 성능

클래스 불균형이 심한 데이터셋에서는 다음과 같은 경향이 나타날 수 있습니다:

### 잘 학습되는 클래스 (많은 샘플)
- ✅ Drivable Area, Road, Building 등
- 높은 IoU 예상 (60-80%)

### 어려운 클래스 (적은 샘플)
- ❌ Traffic Cone, Small Objects 등
- 낮은 IoU 예상 (10-30% 또는 더 낮음)

### 전체 Mean IoU
- 예상 범위: 40-60% (데이터셋에 따라 다름)
- 이 값이 개선의 기준점이 됩니다

## 개선 방법 비교 예시

| 방법 | Mean IoU | 비고 |
|------|----------|------|
| **Baseline (Standard CE)** | 45.2% | ← 현재 코드 |
| + Class Weights (sqrt_inv_freq) | 48.5% | +3.3%p |
| + Dice Loss | 47.8% | +2.6%p |
| + Combined Loss (CE+Dice) | 50.1% | +4.9%p |
| + Combined + Class Weights | 52.3% | +7.1%p |

*위 수치는 예시이며 실제 데이터셋에서 측정 필요*

## 파일 변경 사항

### 수정된 파일
1. **`my_train.py`**
   - Import 제거: `calculate_class_weights`, `DiceLoss`, `CombinedLoss`
   - Loss 설정 간소화 → Standard CE만 사용
   - WandB run name에 "baseline" 추가

2. **`my_train_for_submit.py`**
   - 향후 수정 제외 (백업용)

3. **`my_utils/training_args.py`**
   - 제거된 인자들:
     - `--loss_type`, `--ce_weight`, `--dice_weight`
     - `--use_class_weights`, `--weight_method`, `--effective_beta`

### 유지된 파일 (향후 사용 가능)
- **`my_utils/losses.py`**: DiceLoss, CombinedLoss 정의 (재사용 가능)
- **`my_utils/calculate_class_weights.py`**: 가중치 계산 함수 (재사용 가능)

## 주의사항

1. **기존 체크포인트와의 호환성**
   - Baseline 코드는 모델 구조를 변경하지 않으므로 기존 체크포인트 사용 가능
   - 단, 훈련 시 사용된 loss와 다를 수 있으므로 주의

2. **WandB 프로젝트 관리**
   - Baseline 실험은 "baseline" 태그를 붙여 관리 권장
   - 향후 개선 방법과 쉽게 비교 가능

3. **재현성**
   - `--random_seed` 옵션을 고정하여 재현성 확보
   - 동일한 하이퍼파라미터로 여러 번 실행하여 분산 측정 권장

## 다음 단계

1. ✅ **Baseline 성능 측정**
   ```bash
   python my_train.py --enable_vis --wandb_tags "baseline" ...
   ```

2. 📊 **결과 분석**
   - WandB에서 학습 곡선 확인
   - 클래스별 IoU 확인 → 어떤 클래스가 어려운지 파악

3. 🔧 **개선 방법 적용**
   - 위의 시나리오 참고하여 개선 방법 적용
   - 각 방법을 개별적으로 테스트하여 효과 측정

4. 📈 **비교 및 선택**
   - 최고 성능의 방법 선택
   - 또는 여러 방법을 조합하여 최적화

---

**참고**: 이 baseline 코드는 깔끔하고 표준적인 출발점을 제공합니다. 
개선 방법을 적용할 때는 한 번에 하나씩 추가하여 각각의 효과를 명확히 파악하는 것이 좋습니다.
