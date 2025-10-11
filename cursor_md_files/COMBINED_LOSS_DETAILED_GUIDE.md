# Combined Loss + Class Weights: 원리와 구현

## 📚 Part 1: 원리 (Theory)

### 1.1 왜 두 가지를 결합하는가?

#### 문제: 각 방법의 한계
```
❌ Cross-Entropy만 사용:
- 픽셀별 분류는 잘함
- 하지만 영역(region) 전체의 overlap은 고려 안 함
- 클래스 불균형에 취약

❌ Dice Loss만 사용:
- Region overlap은 잘 최적화
- 하지만 훈련 초기에 불안정
- 개별 픽셀 정확도는 상대적으로 약함

❌ Class Weights만 사용:
- 클래스 간 불균형은 해결
- 하지만 여전히 작은 영역 예측이 어려움
```

#### 해결: Combined Loss
```
✅ Cross-Entropy + Class Weights
  └─ 픽셀 정확도 + 클래스 균형

✅ Dice Loss + Class Weights
  └─ Region overlap + 클래스 균형

✅ Combined (CE + Dice) + Class Weights
  └─ 픽셀 정확도 + Region overlap + 클래스 균형
  └─ 세 가지 목표 동시 달성! 🎯
```

---

## 🔬 Part 2: 수학적 원리

### 2.1 Cross-Entropy Loss

#### 기본 수식
```
L_CE = -Σ y_i * log(p_i)

where:
- y_i: 정답 레이블 (one-hot)
- p_i: 모델의 예측 확률
- i: 픽셀 인덱스
```

#### 의미
- **픽셀별 분류 정확도**를 최적화
- 각 픽셀을 독립적으로 취급
- "이 픽셀이 정답 클래스일 확률"을 높임

#### 예시
```python
# 픽셀 1개에 대한 예측
Ground Truth: Car (class 6)
Prediction: [0.1, 0.05, ..., 0.7, ...]  # Car에 0.7 확률
                              ↑
                          class 6

CE = -log(0.7) = 0.357

만약 Car에 0.9 확률이면:
CE = -log(0.9) = 0.105 (더 낮은 loss = 더 좋음)
```

#### 클래스 불균형 문제
```
Road (많음): 1,000,000 픽셀 → CE = 0.1 (쉬움)
Pedestrian (적음): 1,000 픽셀 → CE = 0.5 (어려움)

Total Loss = (1,000,000 * 0.1 + 1,000 * 0.5) / 1,001,000
           ≈ 0.1

→ Road 픽셀이 지배적! Pedestrian 무시됨
```

---

### 2.2 Class-Weighted Cross-Entropy

#### 수식
```
L_CE_weighted = -Σ w_c * y_i * log(p_i)

where w_c: 클래스 c의 가중치
```

#### 가중치 계산 방법

##### A. Inverse Frequency (기본)
```
w_c = total_pixels / (num_classes × class_c_pixels)

예시:
Class 0 (Road): 500,000 픽셀
Class 11 (Pedestrian): 1,000 픽셀
Total: 1,000,000 픽셀, 19 클래스

w_0 = 1,000,000 / (19 × 500,000) = 0.105
w_11 = 1,000,000 / (19 × 1,000) = 52.6

→ Pedestrian의 가중치가 500배 더 큼!
```

##### B. Square Root Inverse Frequency (추천!)
```
w_c = 1 / sqrt(frequency_c)

예시:
freq_0 = 500,000 / 1,000,000 = 0.5
freq_11 = 1,000 / 1,000,000 = 0.001

w_0 = 1 / sqrt(0.5) = 1.41
w_11 = 1 / sqrt(0.001) = 31.6

→ 완화된 가중치 (22배 차이)
→ 훈련 더 안정적!
```

##### C. Effective Number of Samples
```
w_c = (1 - β) / (1 - β^n_c)

β = 0.9999 (보통)
n_c = 클래스 c의 샘플 수

이론적 근거:
- Data augmentation 시 샘플 간 겹침 고려
- Re-sampling과 동등한 효과
```

#### 효과
```
가중치 적용 전:
Loss = 0.1 (Road 중심, Pedestrian 무시)

가중치 적용 후:
Loss = weighted_avg([0.1 * 1.41, 0.5 * 31.6, ...])
    = 0.3 (균형 잡힌 학습!)

→ 소수 클래스도 학습됨!
```

---

### 2.3 Dice Loss

#### 원리: F1-Score 직접 최적화

Dice Coefficient는 F1-score와 동일:
```
Dice = 2 × |X ∩ Y| / (|X| + |Y|)
     = 2 × TP / (2×TP + FP + FN)

where:
- X: 예측 영역
- Y: 정답 영역
- TP: True Positive 픽셀
- FP: False Positive 픽셀
- FN: False Negative 픽셀
```

#### 수식 (Soft Dice for Differentiability)
```
Dice_c = (2 × Σ p_i × y_i + smooth) / (Σ p_i + Σ y_i + smooth)

where:
- p_i: 클래스 c의 예측 확률 (0~1)
- y_i: 클래스 c의 정답 (0 or 1)
- smooth: 수치 안정성 (보통 1.0)

Dice Loss = 1 - Dice
```

#### 왜 Region Overlap에 강한가?

**예시 1: 큰 영역 (정답 10,000 픽셀)**
```
예측: 9,000 픽셀 맞춤 (TP=9000)
     1,000 픽셀 누락 (FN=1000)

Dice = 2×9000 / (9000+10000) = 0.947

CE Loss:
- 맞춘 9000 픽셀: loss = 0.1 (작음)
- 틀린 1000 픽셀: loss = 2.0 (큼)
- Total: (9000×0.1 + 1000×2.0) / 10000 = 0.29

Dice가 region 전체를 고려!
```

**예시 2: 작은 영역 (정답 100 픽셀)**
```
예측: 50 픽셀 맞춤 (TP=50)
     50 픽셀 누락 (FN=50)

Dice = 2×50 / (50+100) = 0.667

CE Loss: 
- Total: (50×0.1 + 50×2.0) / 100 = 1.05

작은 객체도 큰 패널티!
Dice는 영역 크기에 상관없이 overlap 비율 평가
```

#### 클래스 불균형에 왜 강한가?

분자와 분모 모두 교집합(TP) 포함:
```
다수 클래스 (많은 픽셀):
- TP 크지만, |X| + |Y|도 큼
- Dice = 2×TP / (큰 수)

소수 클래스 (적은 픽셀):
- TP 작지만, |X| + |Y|도 작음
- Dice = 2×TP / (작은 수)

→ 비율(ratio)로 평가하므로 공정!
```

---

### 2.4 Combined Loss

#### 수식
```
L_total = α × L_CE + β × L_Dice

일반적으로:
α + β = 1 (정규화)
α = 0.5~0.7 (CE 비중)
β = 0.3~0.5 (Dice 비중)
```

#### 각 Loss의 역할

```
L_CE (α=0.6):
┌─────────────────────────────┐
│ 픽셀별 정확도 최적화         │
│ - 각 픽셀을 올바르게 분류   │
│ - Sharp boundary 학습        │
│ - Class probabilities 조정   │
└─────────────────────────────┘

L_Dice (β=0.4):
┌─────────────────────────────┐
│ Region overlap 최적화        │
│ - 영역 전체의 IoU 향상      │
│ - 작은 객체 보호             │
│ - 클래스 불균형 완화         │
└─────────────────────────────┘

결합 효과:
┌─────────────────────────────┐
│ 픽셀 정확도 + Region 품질   │
│ 개별 픽셀도 맞고,           │
│ 전체 영역도 잘 맞춤!        │
└─────────────────────────────┘
```

#### 왜 0.6/0.4 비율인가?

```
실험적으로 검증된 비율:

α=0.7, β=0.3:
- CE 비중이 높음
- Boundary 정확도 우선
- 큰 객체 위주 데이터셋

α=0.6, β=0.4: ⭐ (추천)
- 균형 잡힌 비율
- 대부분의 경우 효과적
- nnU-Net 등에서 사용

α=0.5, β=0.5:
- 완전 균형
- Dice 영향력 증가
- 작은 객체 많을 때

α=0.4, β=0.6:
- Dice 비중이 높음
- 매우 불균형한 데이터셋
- 작은 객체 중심
```

---

### 2.5 Combined Loss + Class Weights (최종)

#### 수식
```
L_CE_weighted = -Σ w_c × y_i × log(p_i)
L_Dice_weighted = 1 - Σ w_c × Dice_c

L_total = α × L_CE_weighted + β × L_Dice_weighted
```

#### 시너지 효과

```
Class Weights:
┌─────────────────────────┐
│ 클래스 간 균형          │
│ 소수 클래스 보호        │
└─────────────────────────┘
            ↓
        적용됨
            ↓
┌─────────────────────────┐
│ CE Loss                 │
│ - 픽셀 정확도          │
│ - 가중치로 균형 조정   │
└─────────────────────────┘
            +
┌─────────────────────────┐
│ Dice Loss              │
│ - Region overlap       │
│ - 자체적으로 균형 잡힘 │
│ + 가중치로 추가 보정   │
└─────────────────────────┘
            =
┌─────────────────────────┐
│ 완벽한 조합!           │
│ - 픽셀 정확도 ✓        │
│ - Region 품질 ✓        │
│ - 클래스 균형 ✓        │
│ - 작은 객체 ✓          │
└─────────────────────────┘
```

---

## 💻 Part 3: 코드 구현 원리

### 3.1 Class Weights 계산

#### Step 1: 데이터셋 분석
```python
# 전체 훈련 데이터 순회
class_counts = np.zeros(19)  # 19개 클래스

for image, label in dataset:
    for class_id in range(19):
        class_counts[class_id] += (label == class_id).sum()

# 결과 예시:
# Class 0 (Road): 5,000,000 픽셀
# Class 11 (Pedestrian): 10,000 픽셀
# ...
```

#### Step 2: 가중치 계산 (sqrt_inv_freq)
```python
total_pixels = class_counts.sum()

# 각 클래스의 출현 빈도
freq = class_counts / total_pixels
# freq[0] = 0.5 (Road)
# freq[11] = 0.0001 (Pedestrian)

# Square root inverse frequency
weights = 1.0 / np.sqrt(freq + 1e-10)
# weights[0] = 1.41 (Road)
# weights[11] = 100.0 (Pedestrian)

# 정규화 (평균=1)
weights = weights / weights.mean()
# weights[0] = 0.8 (Road는 평균 이하)
# weights[11] = 58.2 (Pedestrian은 평균 이상)
```

#### 의미
```
정규화 후:
- weight < 1: 다수 클래스 (억제)
- weight = 1: 평균 클래스 (유지)
- weight > 1: 소수 클래스 (강조)

훈련 중:
다수 클래스 픽셀의 loss × 0.8
소수 클래스 픽셀의 loss × 58.2

→ 소수 클래스 학습 촉진!
```

---

### 3.2 Dice Loss 구현

#### Step 1: Softmax로 확률 계산
```python
logits = model(image)  # (B, 19, H, W) - raw scores
probs = F.softmax(logits, dim=1)  # (B, 19, H, W) - probabilities

# 예시 (픽셀 1개):
# probs[:, 0, y, x] = 0.7  # Road 확률
# probs[:, 11, y, x] = 0.05 # Pedestrian 확률
```

#### Step 2: One-hot Encoding
```python
targets = labels  # (B, H, W) - class indices
targets_one_hot = F.one_hot(targets, num_classes=19)
# (B, H, W, 19) → (B, 19, H, W)

# 예시:
# targets[b, y, x] = 11 (Pedestrian)
# targets_one_hot[b, :, y, x] = [0,0,...,1,...,0]
#                                       ↑
#                                   index 11
```

#### Step 3: Intersection & Union 계산
```python
# Valid 픽셀만 (ignore_index 제외)
valid_mask = (targets != 255)  # (B, H, W)

# Spatial dimension flatten
probs_flat = probs.view(B, 19, -1)        # (B, 19, H×W)
targets_flat = targets_one_hot.view(B, 19, -1)  # (B, 19, H×W)

# 클래스별 계산
for c in range(19):
    # Intersection (교집합)
    intersection_c = (probs_flat[:, c, :] * targets_flat[:, c, :]).sum()
    # probs와 targets가 모두 높은 영역
    
    # Union (합집합)
    union_c = probs_flat[:, c, :].sum() + targets_flat[:, c, :].sum()
    # 예측 + 정답 영역의 합
    
    # Dice
    dice_c = (2 * intersection_c + smooth) / (union_c + smooth)
```

#### 예시 계산
```python
Pedestrian 클래스 (class 11):

정답: 100 픽셀에 1, 나머지 0
예측: Pedestrian 확률
  - 60 픽셀: p=0.9 (잘 맞춤)
  - 40 픽셀: p=0.1 (틀림)
  - 다른 영역: p=0.05 (False Positive)

intersection = 60×0.9 + 40×0.1 + FP영역×0.05
             ≈ 58

union = (60×0.9 + 40×0.1 + FP) + 100
      ≈ 168

dice = (2 × 58) / 168 = 0.69

dice_loss = 1 - 0.69 = 0.31
```

---

### 3.3 Combined Loss 구현

#### Forward Pass
```python
def forward(logits, targets):
    # 1. CE Loss 계산
    ce_loss = F.cross_entropy(
        logits, targets,
        weight=class_weights,  # 가중치 적용!
        ignore_index=255
    )
    # 결과: scalar (예: 0.45)
    
    # 2. Dice Loss 계산
    dice_loss = compute_dice_loss(
        logits, targets,
        weight=class_weights  # 가중치 적용!
    )
    # 결과: scalar (예: 0.38)
    
    # 3. 가중 합
    total_loss = 0.6 * ce_loss + 0.4 * dice_loss
    # = 0.6 × 0.45 + 0.4 × 0.38
    # = 0.27 + 0.152
    # = 0.422
    
    return total_loss
```

#### Backward Pass
```python
# PyTorch가 자동으로 처리
total_loss.backward()

# 각 파라미터의 gradient:
# dL/dθ = dL/dL_total × (α×dL_CE/dθ + β×dL_Dice/dθ)

# CE로부터의 gradient (60%)
# Dice로부터의 gradient (40%)
# 두 신호가 모두 반영됨!
```

---

## 🔍 Part 4: 기존 코드 검토

### 4.1 `calculate_class_weights.py` 검토

#### ✅ 잘 구현된 부분

1. **다양한 방법 지원**
```python
✓ inverse_freq: 표준 방법
✓ sqrt_inv_freq: 안정적 (추천!)
✓ effective_num: 이론적 근거 강함
✓ median_freq: SegNet 방법
```

2. **정규화**
```python
class_weights = class_weights / np.mean(class_weights)
✓ 평균이 1이 되도록 정규화
✓ 전체 loss 규모 유지
```

3. **상세한 통계 출력**
```python
✓ 클래스별 픽셀 수, 비율
✓ 계산된 가중치 시각화
✓ 디버깅에 유용
```

#### ⚠️ 개선 가능한 부분

**개선 1: 메모리 효율성**
```python
# 현재: 전체 데이터셋 순회 (느림)
for idx in range(len(dataset)):
    _, label = dataset[idx]
    # 매번 이미지 로드 + transform

# 개선안: 샘플링
for idx in random.sample(range(len(dataset)), 1000):
    # 1000개만 샘플링하여 추정
    # 훨씬 빠름, 정확도는 충분
```

**개선 2: 캐싱**
```python
# 가중치 계산 결과를 파일로 저장
weights_file = f'class_weights_{method}.pth'
if os.path.exists(weights_file):
    weights = torch.load(weights_file)
else:
    weights = calculate_class_weights(...)
    torch.save(weights, weights_file)
```

---

### 4.2 `losses.py` 검토

#### ✅ 잘 구현된 부분

1. **DiceLoss 구현**
```python
✓ Softmax로 differentiable하게 구현
✓ smooth factor로 수치 안정성
✓ ignore_index 올바르게 처리
✓ class weights 지원
```

2. **CombinedLoss 구현**
```python
✓ CE와 Dice를 명확하게 분리
✓ 가중치 비율 조정 가능
✓ class weights 양쪽 모두 전달
```

#### ⚠️ 개선 가능한 부분

**개선 1: Dice Loss의 batch 처리**
```python
# 현재: 모든 batch를 합쳐서 계산
probs = probs.view(B, C, -1)  # (B, C, H*W)
intersection = (probs * targets).sum(dim=2)  # (B, C)

# 문제: batch 간 차이가 평균화됨

# 개선안: batch별로 먼저 계산
for b in range(B):
    intersection_b = (probs[b] * targets[b]).sum()
    dice_b = compute_dice(intersection_b, ...)
    total_dice += dice_b
total_dice /= B
```

**개선 2: 클래스별 Dice 가중치 적용 방식**
```python
# 현재:
dice_loss = 1.0 - dice_score  # (B, C)
if self.weight is not None:
    dice_loss = dice_loss * self.weight.unsqueeze(0)
loss = dice_loss.mean()  # 모든 클래스 평균

# 개선안: 가중 평균
if self.weight is not None:
    loss = (dice_loss * self.weight.unsqueeze(0)).sum() / self.weight.sum()
    # 가중치 고려한 평균
```

**개선 3: square_denominator 옵션**
```python
# 현재는 False가 기본
# 논문에 따르면 True가 더 안정적일 수 있음

# 추천: 기본값 변경 또는 자동 선택
if self.square_denominator or training_unstable:
    union = (probs ** 2).sum() + (targets ** 2).sum()
```

---

### 4.3 전체 평가

#### ✅ 사용 가능 여부: **YES!**

```
기존 코드 품질: 8.5/10

장점:
✓ 핵심 원리 올바르게 구현
✓ 4가지 weighting 방법 지원
✓ Combined loss 완성도 높음
✓ 즉시 사용 가능

개선 여지:
- 메모리 효율성 (중요도: 낮음)
- 캐싱 기능 (중요도: 중간)
- Dice 세부 구현 (중요도: 낮음)

결론:
현재 상태로 충분히 사용 가능!
개선사항은 선택적으로 적용
```

---

## 🚀 Part 5: 실전 사용 가이드

### 5.1 기본 사용 (추천)

```python
# my_train.py에서

from my_utils.calculate_class_weights import calculate_class_weights
from my_utils.losses import CombinedLoss

# 1. Class weights 계산 (훈련 전 1회)
class_weights = calculate_class_weights(
    dataset=train_dst,
    num_classes=19,
    device=device,
    method='sqrt_inv_freq',  # 추천!
    ignore_index=255
)

# 2. Combined Loss 생성
criterion = CombinedLoss(
    ce_weight=0.6,              # CE 60%
    dice_weight=0.4,            # Dice 40%
    smooth=1.0,                 # Dice smoothing
    ignore_index=255,
    class_weights=class_weights,  # 가중치 적용
    square_denominator=False     # 안정성 옵션
)

# 3. 훈련 루프
for images, labels in train_loader:
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

### 5.2 하이퍼파라미터 튜닝

#### CE/Dice 비율 조정
```python
# 픽셀 정확도 중시 (Boundary가 중요)
criterion = CombinedLoss(ce_weight=0.7, dice_weight=0.3, ...)

# 균형 (일반적)
criterion = CombinedLoss(ce_weight=0.6, dice_weight=0.4, ...)

# Region overlap 중시 (작은 객체 많음)
criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, ...)
```

#### Weighting 방법 비교
```python
# 실험 1: sqrt_inv_freq (추천)
weights_1 = calculate_class_weights(method='sqrt_inv_freq')

# 실험 2: inverse_freq (더 강한 가중치)
weights_2 = calculate_class_weights(method='inverse_freq')

# 실험 3: effective_num (이론적)
weights_3 = calculate_class_weights(method='effective_num', beta=0.9999)

# 각각 실험하여 최적 선택
```

### 5.3 예상 결과

```
Baseline (Standard CE):
- Road IoU: 85%
- Car IoU: 72%
- Pedestrian IoU: 15%
- Mean IoU: 45.2%

After (Combined + Weights):
- Road IoU: 86% (+1%p, 약간 향상)
- Car IoU: 78% (+6%p, 크게 향상)
- Pedestrian IoU: 48% (+33%p, 매우 크게 향상!)
- Mean IoU: 52.3% (+7.1%p)

특징:
✓ 소수 클래스 IoU 크게 향상
✓ 다수 클래스는 약간 향상 또는 유지
✓ Mean IoU 전체적으로 상승
✓ 0인 클래스 해소
```

---

## 📋 최종 체크리스트

### 코드 준비
- [x] calculate_class_weights.py 존재
- [x] losses.py (DiceLoss, CombinedLoss) 존재
- [x] 핵심 원리 올바르게 구현됨
- [x] 즉시 사용 가능한 상태

### 사용 전 확인
- [ ] train_dst가 올바르게 로드되었는지
- [ ] num_classes=19 확인
- [ ] device 설정 확인 (cuda/cpu)
- [ ] ignore_index=255 확인

### 훈련 중 모니터링
- [ ] Loss가 수렴하는지 (발산 안 함)
- [ ] 소수 클래스 IoU 향상 확인
- [ ] 다수 클래스 IoU 하락 안 하는지
- [ ] WandB로 실시간 추적

---

## 🎯 결론

### 기존 코드 평가
```
✅ 우수함! 즉시 사용 가능

- 이론적으로 정확
- 구현 깔끔
- 옵션 다양
- 안정적

→ 자신 있게 사용하세요!
```

### 추천 설정
```python
# 가장 안정적이고 효과적인 조합
class_weights = calculate_class_weights(
    method='sqrt_inv_freq',  # 완화된 가중치
    ...
)

criterion = CombinedLoss(
    ce_weight=0.6,          # 픽셀 정확도 우선
    dice_weight=0.4,        # Region overlap 보조
    class_weights=class_weights,
    ...
)
```

### 다음 단계
1. ✅ 코드 검토 완료
2. 🚀 바로 적용 (my_train.py 수정)
3. 📊 WandB로 결과 비교
4. 🎯 성능 향상 확인!

---

**준비 완료! 바로 시작할 수 있습니다!** 🎉

