# Loss & mIoU 동시 하락 문제: 원인과 해결책

## 🚨 문제 상황

```
Combined Loss + Class Weights 적용 후:

Epoch 1: Loss=0.8, mIoU=45%
Epoch 2: Loss=0.7, mIoU=42% ⚠️
Epoch 3: Loss=0.6, mIoU=38% ⚠️
Epoch 5: Loss=0.5, mIoU=32% 🔴 문제!

예상: Loss ↓, mIoU ↑
실제: Loss ↓, mIoU ↓ (둘 다 하락!)
```

**이것은 모델이 잘못 학습되고 있다는 신호입니다!**

---

## 🔍 원인 분석

### 원인 1: 극단적인 Class Weights (가장 흔함!) ⭐⭐⭐

#### 문제
```python
# inverse_freq 사용 시
class_weights 계산 결과:
Class 0 (Road, 많음):      weight = 0.1
Class 11 (Pedestrian, 적음): weight = 500.0

→ 500배 차이!

훈련 중:
- Pedestrian 픽셀 1개 잘못 = loss × 500
- Road 픽셀 100개 잘못 = loss × 10

→ 모델이 Pedestrian만 신경쓰고 Road 무시
→ mIoU 하락!
```

#### 왜 Loss는 낮아지는가?
```
모델의 전략:
"Pedestrian(가중치 500)만 맞추면 loss가 낮아진다"

실제 학습:
- Pedestrian: IoU 80% (좋음)
- Road: IoU 30% (나쁨)
- Car: IoU 40% (나쁨)

Weighted Loss: 낮음 (Pedestrian 가중치 큼)
Mean IoU: 낮음 (Road, Car 망가짐)

→ Loss와 실제 성능의 불일치!
```

#### 해결책
```python
# ❌ 너무 극단적
weights = calculate_class_weights(method='inverse_freq')

# ✅ 완화된 가중치 (추천!)
weights = calculate_class_weights(method='sqrt_inv_freq')

# 예시 비교:
# inverse_freq:    [0.1, 500.0] (5000배 차이)
# sqrt_inv_freq:   [0.7, 22.6]  (32배 차이)
```

---

### 원인 2: Dice Loss의 초기 불안정성 ⭐⭐

#### 문제
```python
Dice = (2 × intersection) / (prediction + ground_truth)

초기 훈련:
- 예측이 매우 불확실 (random에 가까움)
- intersection이 거의 0에 가까움
- Denominator도 불안정

예시:
Dice = (2 × 0.001) / (0.1 + 0.15) = 0.008
Gradient: 매우 큰 값 또는 NaN

→ 훈련 폭발 또는 발산!
```

#### 실제 로그 예시
```
Epoch 1:
  CE Loss: 0.8 (안정)
  Dice Loss: 0.95 (매우 큼)
  Combined: 0.6×0.8 + 0.4×0.95 = 0.86

Epoch 2:
  CE Loss: 0.7
  Dice Loss: 0.98 (더 커짐!)
  Combined: 0.88 (증가!)

→ Dice가 학습을 방해
```

#### 해결책

**Solution 1: Warm-up (추천!)**
```python
# 초기에는 CE만, 나중에 Dice 추가
def get_loss_weights(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        # Warm-up: CE만
        ce_weight = 1.0
        dice_weight = 0.0
    else:
        # 점진적으로 Dice 추가
        progress = min((epoch - warmup_epochs) / 10, 1.0)
        ce_weight = 0.6 + 0.4 * (1 - progress)
        dice_weight = 0.4 * progress
    
    return ce_weight, dice_weight

# 훈련 중:
ce_w, dice_w = get_loss_weights(epoch)
criterion = CombinedLoss(ce_weight=ce_w, dice_weight=dice_w, ...)

# Epoch 1-5: CE 100%
# Epoch 6-15: CE 점진적 감소, Dice 점진적 증가
# Epoch 16+: CE 60%, Dice 40%
```

**Solution 2: Smooth 증가**
```python
# 초기에 더 큰 smooth factor
def get_dice_smooth(epoch):
    if epoch < 10:
        return 10.0  # 큰 smooth (안정)
    elif epoch < 20:
        return 5.0
    else:
        return 1.0   # 표준 smooth

smooth = get_dice_smooth(epoch)
criterion = CombinedLoss(smooth=smooth, ...)
```

**Solution 3: Square Denominator**
```python
# Dice Loss를 더 안정적으로
criterion = CombinedLoss(
    square_denominator=True,  # 안정성 향상!
    ...
)

# 수식:
# 기본: (2×intersection) / (pred + target)
# Square: (2×intersection) / (pred² + target²)
# → Gradient가 더 안정적
```

---

### 원인 3: Learning Rate 부적절 ⭐⭐⭐

#### 문제
```python
Baseline (CE only):
- Loss scale: ~0.5
- Gradient scale: ~0.01
- LR: 1e-5 (적합)

Combined + Weights:
- Loss scale: ~2.0 (4배 증가!)
- Gradient scale: ~0.05 (5배 증가!)
- LR: 1e-5 (여전히 동일)

→ Learning rate가 너무 큼!
→ Overshoot 발생
→ 발산
```

#### 실제 예시
```
LR=1e-5 (너무 큼):
Epoch 1: Loss=0.8, mIoU=45%
Epoch 2: Loss=0.6, mIoU=40% (overshoot)
Epoch 3: Loss=0.9, mIoU=35% (발산)

LR=5e-6 (적절):
Epoch 1: Loss=0.8, mIoU=45%
Epoch 2: Loss=0.75, mIoU=46% (안정적 개선)
Epoch 3: Loss=0.7, mIoU=48%
```

#### 해결책

**Solution 1: Learning Rate 감소 (즉각 효과)**
```python
# Baseline LR
baseline_lr = 1e-5

# Combined Loss 사용 시
adjusted_lr = baseline_lr * 0.5  # 또는 0.3
# = 5e-6 또는 3e-6

optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=adjusted_lr,  # 감소된 LR
    momentum=0.9
)
```

**Solution 2: Gradient Clipping (강력 추천!)**
```python
# Forward & Backward
loss = criterion(outputs, labels)
loss.backward()

# Gradient Clipping (추가!)
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # gradient norm 제한
)

optimizer.step()

# 효과:
# - Gradient explosion 방지
# - 훈련 안정화
# - 더 큰 LR 사용 가능
```

**Solution 3: LR Warm-up**
```python
def get_warmup_lr(epoch, base_lr, warmup_epochs=5):
    if epoch < warmup_epochs:
        # 선형으로 증가
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

# 훈련 중:
current_lr = get_warmup_lr(epoch, base_lr=5e-6)
for param_group in optimizer.param_groups:
    param_group['lr'] = current_lr

# Epoch 1: LR=1e-6
# Epoch 2: LR=2e-6
# Epoch 5: LR=5e-6
# Epoch 6+: LR=5e-6 (유지)
```

---

### 원인 4: Loss Scale 변화 ⭐⭐

#### 문제
```python
Baseline (CE only):
Loss = 0.5 (적절한 규모)

Combined (CE + Dice):
CE Loss = 0.5
Dice Loss = 0.8
Combined = 0.6×0.5 + 0.4×0.8 = 0.62

문제없어 보이지만...

가중치 추가:
CE Loss (weighted) = 2.5 (5배 증가!)
Dice Loss (weighted) = 1.5 (2배 증가!)
Combined = 0.6×2.5 + 0.4×1.5 = 2.1

→ Loss 규모가 4배 증가!
→ Gradient 폭발
```

#### 해결책

**Solution 1: Loss Normalization**
```python
class NormalizedCombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.6, dice_weight=0.4, ...):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(...)
        self.dice_loss = DiceLoss(...)
        
        # Moving average for normalization
        self.register_buffer('ce_scale', torch.tensor(1.0))
        self.register_buffer('dice_scale', torch.tensor(1.0))
        self.momentum = 0.99
    
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        # Update scales (EMA)
        with torch.no_grad():
            self.ce_scale = self.momentum * self.ce_scale + \
                           (1-self.momentum) * ce.detach()
            self.dice_scale = self.momentum * self.dice_scale + \
                             (1-self.momentum) * dice.detach()
        
        # Normalize
        ce_norm = ce / (self.ce_scale + 1e-8)
        dice_norm = dice / (self.dice_scale + 1e-8)
        
        # Combine
        total = self.ce_weight * ce_norm + self.dice_weight * dice_norm
        
        return total
```

**Solution 2: 가중치 정규화 확인**
```python
# calculate_class_weights.py에서
class_weights = class_weights / np.mean(class_weights)

# 확인
print(f"Weight mean: {class_weights.mean():.4f}")  # 1.0이어야 함
print(f"Weight std: {class_weights.std():.4f}")
print(f"Weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")

# 너무 극단적이면 clip
class_weights = np.clip(class_weights, 0.1, 10.0)  # 100배 차이로 제한
```

---

### 원인 5: Batch Size 문제 ⭐

#### 문제
```python
Dice Loss는 batch 내 통계 사용

작은 batch (예: batch_size=2):
- 클래스별 픽셀 수가 매우 적음
- 일부 클래스가 batch에 없을 수 있음
- Dice 계산이 불안정

예시:
Batch 1: Road만 있음 → Dice for Pedestrian = NaN
Batch 2: Pedestrian 10픽셀 → Dice 매우 불안정
```

#### 해결책

**Solution 1: Batch Size 증가**
```python
# 가능하면 batch size 증가
batch_size = 4  # 최소 4 이상 추천
# 또는 8, 16 (GPU 메모리 허용 시)

# 메모리 부족 시: Gradient Accumulation
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for i, (images, labels) in enumerate(train_loader):
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Solution 2: Dice Loss 수정**
```python
# Global Dice (모든 batch 합쳐서 계산)
class GlobalDiceLoss(nn.Module):
    def forward(self, logits, targets):
        # Batch dimension 유지하여 합산
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes)
        
        # 전체 batch에 대해 계산
        intersection = (probs * targets_one_hot).sum(dim=(0,2,3))
        union = probs.sum(dim=(0,2,3)) + targets_one_hot.sum(dim=(0,2,3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
```

---

### 원인 6: 클래스 불균형이 너무 심함 ⭐

#### 문제
```python
극단적인 불균형:
Class 0: 10,000,000 픽셀 (99.9%)
Class 18: 100 픽셀 (0.01%)

→ 10만배 차이!

sqrt_inv_freq로도:
weight_0 = 1 / sqrt(0.999) = 1.0
weight_18 = 1 / sqrt(0.0001) = 100.0

→ 여전히 100배 차이
→ 훈련 불안정
```

#### 해결책

**Solution 1: Weight Clipping**
```python
def calculate_class_weights_safe(..., max_weight_ratio=10.0):
    # 기존 계산
    class_weights = calculate_class_weights(...)
    
    # Normalize
    class_weights = class_weights / class_weights.mean()
    
    # Clip (최대/최소 비율 제한)
    min_weight = 1.0 / max_weight_ratio
    max_weight = max_weight_ratio
    
    class_weights = np.clip(class_weights, min_weight, max_weight)
    
    print(f"Weight range after clipping: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    print(f"Max ratio: {class_weights.max()/class_weights.min():.1f}x")
    
    return torch.FloatTensor(class_weights).to(device)

# 사용
weights = calculate_class_weights_safe(
    dataset=train_dst,
    method='sqrt_inv_freq',
    max_weight_ratio=10.0  # 최대 10배 차이
)
```

**Solution 2: 두 단계 접근**
```python
# Stage 1: 가벼운 가중치로 시작
weights_stage1 = calculate_class_weights(method='sqrt_inv_freq')
weights_stage1 = np.clip(weights_stage1, 0.5, 2.0)  # 4배 제한

# 20 epoch 훈련...

# Stage 2: 더 강한 가중치 적용
weights_stage2 = calculate_class_weights(method='sqrt_inv_freq')
weights_stage2 = np.clip(weights_stage2, 0.2, 5.0)  # 25배 제한

# 나머지 훈련...
```

---

## 🎯 종합 해결책 (추천)

### 방법 1: 안전한 설정 (초보자/안정성 우선)

```python
# 1. 완화된 가중치
class_weights = calculate_class_weights(
    dataset=train_dst,
    method='sqrt_inv_freq',  # inverse_freq 아님!
    num_classes=19,
    device=device
)

# 2. Weight clipping
class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
print(f"Weight range: {class_weights.min():.2f} ~ {class_weights.max():.2f}")

# 3. Warm-up 포함 Combined Loss
class SafeCombinedLoss(nn.Module):
    def __init__(self, epoch, **kwargs):
        super().__init__()
        self.epoch = epoch
        
        # Warm-up logic
        if epoch < 5:
            ce_weight = 1.0
            dice_weight = 0.0
        elif epoch < 15:
            progress = (epoch - 5) / 10
            ce_weight = 1.0 - 0.4 * progress
            dice_weight = 0.4 * progress
        else:
            ce_weight = 0.6
            dice_weight = 0.4
        
        self.criterion = CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            smooth=5.0 if epoch < 10 else 1.0,  # 초기 큰 smooth
            class_weights=kwargs.get('class_weights'),
            square_denominator=True  # 안정성!
        )
    
    def forward(self, logits, targets):
        return self.criterion(logits, targets)

# 4. 감소된 Learning Rate
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=3e-6,  # 기존 1e-5에서 감소!
    momentum=0.9
)

# 5. Gradient Clipping
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# 6. 큰 Batch Size (가능하면)
batch_size = 8  # 또는 gradient accumulation
```

---

### 방법 2: 점진적 적용 (추천!)

```python
# Phase 1: Class Weights만 (5 epochs)
criterion_phase1 = nn.CrossEntropyLoss(
    weight=class_weights_clipped,
    ignore_index=255
)
# LR = 5e-6
# → 안정성 확인

# Phase 2: CE + 약한 Dice (10 epochs)
criterion_phase2 = CombinedLoss(
    ce_weight=0.8,
    dice_weight=0.2,  # 약하게 시작
    class_weights=class_weights_clipped,
    smooth=5.0
)
# LR = 3e-6
# → Dice 효과 확인

# Phase 3: 최종 비율 (나머지)
criterion_phase3 = CombinedLoss(
    ce_weight=0.6,
    dice_weight=0.4,  # 최종 비율
    class_weights=class_weights_clipped,
    smooth=1.0
)
# LR = 3e-6
# → 최종 성능
```

---

## 📊 모니터링 체크리스트

### 훈련 중 확인사항

```python
# 1. Loss 개별 확인
print(f"CE Loss: {ce_loss.item():.4f}")
print(f"Dice Loss: {dice_loss.item():.4f}")
print(f"Combined: {total_loss.item():.4f}")

# 2. Gradient 확인
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")

# 정상: 0.1 ~ 10
# 문제: > 100 (exploding) 또는 < 0.001 (vanishing)

# 3. 클래스별 IoU
for i, iou in enumerate(class_ious):
    print(f"Class {i}: {iou:.3f}")

# 확인: 모든 클래스가 골고루 학습되는지

# 4. WandB에 로깅
wandb.log({
    'CE Loss': ce_loss,
    'Dice Loss': dice_loss,
    'Total Loss': total_loss,
    'Gradient Norm': total_norm,
    'Mean IoU': mean_iou,
    **{f'IoU/Class_{i}': iou for i, iou in enumerate(class_ious)}
})
```

---

## ⚠️ 경고 신호

### 즉시 훈련을 멈춰야 할 때:

```
🚨 Loss가 NaN 또는 Inf
→ Learning rate 너무 큼 또는 gradient explosion

🚨 Loss가 발산 (계속 증가)
→ Learning rate 너무 큼

🚨 Mean IoU가 5 epoch 연속 하락
→ 가중치가 너무 극단적

🚨 특정 클래스 IoU가 0으로 고정
→ 해당 클래스가 완전히 무시됨

🚨 Gradient norm > 1000
→ Gradient explosion 진행 중

→ 즉시 중단하고 설정 재조정!
```

---

## ✅ 성공 지표

### 올바르게 학습되고 있다면:

```
✅ Loss 안정적으로 감소
   Epoch 1: 0.8
   Epoch 5: 0.6
   Epoch 10: 0.5

✅ Mean IoU 꾸준히 증가
   Epoch 1: 45%
   Epoch 5: 48%
   Epoch 10: 51%

✅ 소수 클래스 IoU 향상
   Before: Pedestrian 15%
   After: Pedestrian 35% (+20%p)

✅ 다수 클래스 IoU 유지 또는 약간 향상
   Before: Road 85%
   After: Road 86% (+1%p)

✅ Gradient norm 안정적 (0.1 ~ 10)

✅ CE Loss와 Dice Loss 모두 감소
```

---

## 🎯 최종 추천 설정

```python
# 가장 안전하고 효과적인 조합
class_weights = calculate_class_weights(
    method='sqrt_inv_freq',
    ...
)
class_weights = torch.clamp(class_weights, 0.1, 10.0)

criterion = CombinedLoss(
    ce_weight=0.7,           # CE 비중 높임 (안정성)
    dice_weight=0.3,         # Dice 비중 낮춤 (초기)
    smooth=5.0,              # 큰 smooth (안정성)
    class_weights=class_weights,
    square_denominator=True  # 안정성!
)

optimizer = torch.optim.SGD(
    lr=3e-6,  # 감소된 LR
    momentum=0.9
)

# 훈련 루프
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

**이 설정으로 시작하고, 안정적이면 점진적으로 조정하세요!**

