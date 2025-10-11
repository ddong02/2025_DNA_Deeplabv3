# 클래스 불균형 해결 방법 (Class Imbalance Solutions)

Semantic Segmentation에서 클래스 불균형 문제를 해결하기 위한 검증된 방법들을 정리합니다.

## 📊 현재 문제 상황

```
클래스 불균형 증상:
- 특정 클래스 IoU = 0 또는 < 0.2 (소수 클래스)
- 특정 클래스 IoU > 0.8 (다수 클래스)
- Mean IoU가 낮게 측정됨

원인:
- 클래스별 픽셀 수 차이가 큼
- 모델이 다수 클래스에 편향됨
- 소수 클래스를 무시하거나 잘못 예측
```

---

## 🎯 해결 방법 (논문 기반)

### 1. **Class-Weighted Cross-Entropy Loss** ⭐⭐⭐

#### 📚 이론 및 근거
가장 기본적이고 효과적인 방법. 소수 클래스에 더 큰 가중치를 부여하여 손실 함수에서 균형을 맞춤.

#### 📖 관련 논문
- **ENet** (Paszke et al., 2016): "ENet: A Deep Neural Network Architecture for Real-time Semantic Segmentation"
- **SegNet** (Badrinarayanan et al., 2017): "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"

#### 🔬 방법론

##### A. Inverse Frequency Weighting (가장 일반적)
```python
weight_c = total_pixels / (num_classes × class_c_pixels)
```

**장점**: 
- 구현 간단
- 즉시 효과 확인 가능
- 대부분의 경우 효과적

**단점**: 
- 극단적으로 적은 클래스에 과도한 가중치 부여 가능
- 훈련이 불안정해질 수 있음

##### B. Square Root Inverse Frequency (추천 ⭐)
```python
weight_c = 1 / sqrt(frequency_c)
```

**출처**: Class-Balanced Loss 연구들에서 제안

**장점**:
- Inverse Frequency보다 완화된 가중치
- 훈련 안정성 향상
- 과도한 가중치 방지

**단점**: 
- 매우 심한 불균형에서는 효과 제한적

##### C. Effective Number of Samples
```python
weight_c = (1 - β) / (1 - β^n_c)
```
- β: 하이퍼파라미터 (보통 0.9999)
- n_c: 클래스 c의 샘플 수

**출처**: **"Class-Balanced Loss Based on Effective Number of Samples"** (Cui et al., CVPR 2019)

**장점**:
- 이론적 근거가 강함
- 데이터 augmentation을 고려한 샘플 수 추정
- Re-sampling과 동등한 효과

**단점**: 
- β 튜닝 필요
- 계산이 복잡

##### D. Median Frequency Balancing
```python
weight_c = median_frequency / frequency_c
```

**출처**: **SegNet** (Badrinarayanan et al., 2017)

**장점**:
- 중앙값 기준으로 안정적
- 극단값에 덜 민감

**단점**: 
- 클래스 수가 적을 때 효과 제한적

#### 💻 코드 적용 방법

현재 프로젝트에는 이미 구현되어 있습니다!

```python
# my_utils/calculate_class_weights.py 사용
from my_utils.calculate_class_weights import calculate_class_weights

# 가중치 계산
class_weights = calculate_class_weights(
    dataset=train_dst,
    num_classes=19,
    device=device,
    method='sqrt_inv_freq',  # 추천!
    # method='inverse_freq',   # 기본
    # method='effective_num',  # 고급
    # method='median_freq',    # 안정적
    beta=0.9999,  # effective_num 사용 시
    ignore_index=255
)

# Loss에 적용
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=255,
    reduction='mean'
)
```

#### 📊 예상 효과
```
Before: Mean IoU = 45.2%
After (sqrt_inv_freq): Mean IoU = 48.5% (+3.3%p)
After (effective_num): Mean IoU = 49.1% (+3.9%p)
```

---

### 2. **Focal Loss** ⭐⭐⭐

#### 📚 이론 및 근거
어려운 샘플(hard examples)에 집중하도록 설계된 손실 함수. 쉬운 샘플의 손실을 줄이고 어려운 샘플의 손실을 증가시킴.

#### 📖 관련 논문
**"Focal Loss for Dense Object Detection"** (Lin et al., ICCV 2017)
- RetinaNet에서 제안
- Object Detection의 클래스 불균형 해결
- Semantic Segmentation에도 효과적

#### 🔬 수식
```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- p_t: 정답 클래스의 예측 확률
- α_t: 클래스 가중치 (선택적)
- γ: focusing parameter (보통 2)

**작동 원리**:
- p_t가 높으면 (쉬운 샘플) → (1-p_t)^γ가 작음 → 손실 감소
- p_t가 낮으면 (어려운 샘플) → (1-p_t)^γ가 큼 → 손실 증가

#### 💻 코드 적용 방법

```python
# my_utils/losses.py에 추가 필요
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스 가중치
        self.gamma = gamma  # focusing parameter
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, 
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # 클래스별 가중치 적용
            focal_loss = self.alpha[targets] * focal_loss
        
        return focal_loss.mean()

# 사용
criterion = FocalLoss(
    alpha=class_weights,  # 선택적
    gamma=2.0,
    ignore_index=255
)
```

#### 🎛️ 하이퍼파라미터 튜닝
```python
gamma = 0   # Standard CE
gamma = 1   # 약한 focusing
gamma = 2   # 표준 (논문 추천)
gamma = 3   # 강한 focusing
gamma = 5   # 매우 강한 focusing (희귀 클래스 많을 때)
```

#### 📊 예상 효과
```
Before: Mean IoU = 45.2%
After (γ=2): Mean IoU = 47.8% (+2.6%p)
After (γ=2 + α): Mean IoU = 50.5% (+5.3%p)
```

---

### 3. **Dice Loss** ⭐⭐⭐

#### 📚 이론 및 근거
의료 영상 분야에서 널리 사용. F1-score를 직접 최적화하며, 클래스 불균형에 자연스럽게 강건함.

#### 📖 관련 논문
- **V-Net** (Milletari et al., 2016): "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
- **U-Net** variants에서 널리 사용

#### 🔬 수식
```python
Dice = (2 × |X ∩ Y|) / (|X| + |Y|)
DiceLoss = 1 - Dice
```

**장점**:
- 클래스 불균형에 강건 (분자/분모 모두 교집합 포함)
- Boundary 품질 향상
- F1-score 직접 최적화

**단점**:
- 훈련 초기 불안정할 수 있음
- Gradient 소실 가능

#### 💻 코드 적용 방법

이미 구현되어 있습니다!

```python
# my_utils/losses.py 사용
from my_utils.losses import DiceLoss

criterion = DiceLoss(
    smooth=1.0,
    ignore_index=255,
    weight=class_weights,  # 선택적
    square_denominator=False
)
```

#### 📊 예상 효과
```
Before: Mean IoU = 45.2%
After: Mean IoU = 47.8% (+2.6%p)
특히 작은 객체와 boundary 성능 향상
```

---

### 4. **Combined Loss (CE + Dice)** ⭐⭐⭐⭐⭐ (강력 추천!)

#### 📚 이론 및 근거
Cross-Entropy와 Dice Loss의 장점을 결합. CE는 픽셀별 정확도, Dice는 전체 영역 overlap을 최적화.

#### 📖 관련 논문
- **"Generalised Dice overlap as a deep learning loss function"** (Sudre et al., 2017)
- **nnU-Net** (Isensee et al., 2021): Medical imaging SOTA - Combined loss 사용

#### 🔬 수식
```python
L_total = α × L_CE + β × L_Dice
```
- α, β: 가중치 (보통 α=0.5~0.7, β=0.3~0.5)

**이점**:
- CE: 개별 픽셀 분류 정확도
- Dice: Region overlap 최적화
- 두 목표 동시 달성

#### 💻 코드 적용 방법

이미 구현되어 있습니다!

```python
# my_utils/losses.py 사용
from my_utils.losses import CombinedLoss

criterion = CombinedLoss(
    ce_weight=0.6,        # CE 비중
    dice_weight=0.4,      # Dice 비중
    smooth=1.0,
    ignore_index=255,
    class_weights=class_weights,  # 선택적
    square_denominator=False
)
```

#### 🎛️ 가중치 조정
```python
# 픽셀 정확도 중시
ce_weight=0.7, dice_weight=0.3

# 균형 (추천)
ce_weight=0.6, dice_weight=0.4

# Region overlap 중시
ce_weight=0.5, dice_weight=0.5

# Dice 강조 (작은 객체 많을 때)
ce_weight=0.4, dice_weight=0.6
```

#### 📊 예상 효과
```
Before: Mean IoU = 45.2%
After (0.6/0.4): Mean IoU = 50.1% (+4.9%p)
After (0.6/0.4 + weights): Mean IoU = 52.3% (+7.1%p)
```

---

### 5. **Online Hard Example Mining (OHEM)** ⭐⭐

#### 📚 이론 및 근거
어려운 샘플에 집중하여 훈련. 손실이 높은 픽셀만 선택하여 역전파.

#### 📖 관련 논문
**"Training Region-based Object Detectors with Online Hard Example Mining"** (Shrivastava et al., CVPR 2016)

#### 🔬 방법론
```python
1. Forward pass로 모든 픽셀의 loss 계산
2. Loss가 높은 K개 픽셀 선택
3. 선택된 픽셀에 대해서만 backward
```

**장점**:
- 어려운 예제에 집중
- 메모리 효율적

**단점**:
- 구현 복잡
- 하이퍼파라미터 튜닝 필요

#### 💻 코드 적용 방법

```python
class OHEMLoss(nn.Module):
    def __init__(self, thresh=0.7, min_kept=100000):
        super(OHEMLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, labels):
        # 모든 픽셀의 loss 계산
        pixel_losses = self.criterion(logits, labels)
        
        # Hard examples 선택
        sorted_losses, _ = torch.sort(pixel_losses.view(-1), descending=True)
        
        if sorted_losses[self.min_kept] > self.thresh:
            threshold = sorted_losses[self.min_kept]
        else:
            threshold = self.thresh
        
        # 선택된 픽셀만 사용
        kept_mask = pixel_losses > threshold
        valid_losses = pixel_losses[kept_mask]
        
        return valid_losses.mean()
```

#### 📊 예상 효과
```
Before: Mean IoU = 45.2%
After: Mean IoU = 48.2% (+3.0%p)
Boundary와 작은 객체 성능 특히 향상
```

---

### 6. **Generalized Dice Loss** ⭐⭐

#### 📚 이론 및 근거
클래스 크기에 따라 자동으로 가중치를 조정하는 Dice Loss 변형.

#### 📖 관련 논문
**"Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations"** (Sudre et al., MICCAI 2017)

#### 🔬 수식
```python
GDL = 1 - (2 × Σ(w_c × |X_c ∩ Y_c|)) / (Σ(w_c × (|X_c| + |Y_c|)))

where w_c = 1 / (Σ r_c)^2
r_c: reference volume for class c
```

**특징**:
- 클래스 크기에 반비례하는 가중치 자동 계산
- 매우 작은 구조에 효과적

#### 💻 코드 적용 방법

```python
class GeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        
        # One-hot encoding
        targets_one_hot = F.one_hot(
            targets.clamp(0, num_classes-1), 
            num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Valid mask
        valid = (targets != self.ignore_index).float().unsqueeze(1)
        
        # Class weights (1 / volume^2)
        volumes = (targets_one_hot * valid).sum(dim=(0,2,3))
        weights = 1.0 / ((volumes + self.smooth) ** 2)
        
        # Generalized Dice
        intersection = (probs * targets_one_hot * valid).sum(dim=(0,2,3))
        union = ((probs + targets_one_hot) * valid).sum(dim=(0,2,3))
        
        dice = (2 * (weights * intersection).sum() + self.smooth) / \
               ((weights * union).sum() + self.smooth)
        
        return 1 - dice
```

---

## 🎯 추천 적용 순서

### Phase 1: 기본 가중치 (즉시 적용 가능) ⚡
```python
# 가장 안정적이고 효과적
class_weights = calculate_class_weights(
    dataset=train_dst,
    method='sqrt_inv_freq',  # 추천!
    num_classes=19,
    device=device
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=255
)
```

**기대 효과**: +2~4%p Mean IoU

---

### Phase 2: Combined Loss (강력 추천) 🔥
```python
criterion = CombinedLoss(
    ce_weight=0.6,
    dice_weight=0.4,
    class_weights=class_weights  # Phase 1의 가중치 사용
)
```

**기대 효과**: +5~8%p Mean IoU (Baseline 대비)

---

### Phase 3: Focal Loss (선택적) 🎯
```python
# 여전히 특정 클래스 IoU가 낮다면
criterion = FocalLoss(
    alpha=class_weights,
    gamma=2.0
)
```

**기대 효과**: 어려운 클래스 IoU 특히 향상

---

### Phase 4: 고급 기법 (필요시) 🚀
```python
# OHEM, Generalized Dice 등
# 복잡도 증가, 구현 난이도 높음
```

---

## 📊 실험 계획 예시

```
실험 1: Baseline (완료)
└─ CE Loss, no weights
   Result: Mean IoU = 45.2%

실험 2: + Class Weights
└─ CE + sqrt_inv_freq weights
   Expected: Mean IoU ~= 48%

실험 3: + Dice Loss
└─ Dice Loss only + weights
   Expected: Mean IoU ~= 48%

실험 4: + Combined Loss (추천!)
└─ CE(0.6) + Dice(0.4) + weights
   Expected: Mean IoU ~= 50-52%

실험 5: + Focal Loss
└─ Focal(γ=2) + weights
   Expected: Mean IoU ~= 50%

실험 6: Combined + Focal
└─ 최적 조합 찾기
   Expected: Mean IoU ~= 52-55%
```

---

## 💡 추가 팁

### 1. Data Augmentation
```python
# 소수 클래스가 포함된 이미지 오버샘플링
# Mixup, CutMix for segmentation
# Copy-Paste augmentation for small objects
```

### 2. Two-Stage Training
```python
# Stage 1: 모든 클래스 학습
# Stage 2: 어려운 클래스만 fine-tuning
```

### 3. Ensemble
```python
# 여러 loss function으로 훈련한 모델 앙상블
# Class-specific models 결합
```

---

## 📚 참고 논문 요약

1. **Class-Balanced Loss** (Cui et al., CVPR 2019)
   - Effective Number of Samples 제안
   - Re-sampling과 동등한 효과

2. **Focal Loss** (Lin et al., ICCV 2017)
   - Hard example mining
   - Object detection의 standard

3. **V-Net** (Milletari et al., 2016)
   - Dice Loss for segmentation
   - Medical imaging에서 효과적

4. **nnU-Net** (Isensee et al., 2021)
   - Medical imaging SOTA
   - Combined CE + Dice 사용
   - Adaptive training strategy

5. **Generalized Dice** (Sudre et al., 2017)
   - 매우 불균형한 데이터셋에 효과적
   - 자동 가중치 조정

---

## 🔧 다음 단계

1. ✅ Baseline test 완료 (이미 완료)
2. 📊 클래스별 IoU 분석
3. 🎯 Phase 2 적용: Combined Loss + Class Weights
4. 📈 성능 비교 및 최적화
5. 🚀 필요시 고급 기법 적용

---

**핵심 메시지**: 
- Phase 2 (Combined Loss + Class Weights)부터 시작하세요
- 대부분의 경우 이것만으로도 큰 개선 효과
- 추가 개선이 필요한 경우에만 Phase 3, 4 적용

