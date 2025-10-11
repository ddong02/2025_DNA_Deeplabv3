# Fine-tuning 시 Learning Rate 전략

## 📌 현재 상황 분석

### 시나리오
```
Baseline 모델 (이미 수렴) → Combined Loss로 Fine-tuning
```

1. **Baseline**: 200 epochs, Standard CE Loss로 훈련 완료, 수렴 상태
2. **목표**: Loss function만 변경 (CE → Combined Loss + Class Weights)
3. **문제**: 이미 수렴된 모델을 어떤 LR로 다시 훈련시킬 것인가?

---

## 📊 학습률 선택 가이드

### 1. **일반적인 LR 범위**

| 시나리오 | Learning Rate | 설명 |
|---------|---------------|------|
| **Scratch Training** | 1e-3 ~ 1e-4 | 처음부터 학습 |
| **Transfer Learning (다른 도메인)** | 1e-4 ~ 1e-5 | ImageNet → Custom |
| **Fine-tuning (같은 도메인)** | 1e-5 ~ 1e-6 | 수렴된 모델 조정 |
| **Loss Function 변경** | **1e-6 ~ 1e-7** | ⭐ **현재 상황** |

### 2. **왜 낮은 LR이 필요한가?**

#### ❌ **너무 높은 LR (3e-6 이상)**
```
문제점:
- 이미 학습된 feature가 망가질 수 있음
- Loss function이 바뀌어서 gradient 방향이 크게 다름
- "Catastrophic forgetting" 발생 가능
- Loss와 mIoU 동시 하락 위험
```

#### ✅ **적절한 LR (1e-6 ~ 5e-7)**
```
장점:
- 기존 feature를 보존하면서 조정
- 새로운 loss에 부드럽게 적응
- 안정적인 수렴
- Baseline 성능 유지하면서 개선
```

#### ⚠️ **너무 낮은 LR (1e-8 이하)**
```
문제점:
- 학습 속도가 너무 느림
- 200 epochs로는 충분한 개선이 어려움
- Baseline 성능에서 거의 변화 없음
```

---

## 🔍 현재 코드 분석

### **현재 설정**
```python
# 명령어에서 지정: --lr 1e-5
# 코드에서 자동 조정: adjusted_lr = opts.lr * 0.3

실제 LR:
- Stage 1 (classifier): 1e-5 × 0.3 = 3e-6
- Stage 2 (backbone):   1e-5 × 0.3 / 100 = 3e-8  ⚠️ 너무 낮음!
- Stage 2 (classifier): 1e-5 × 0.3 / 10  = 3e-7
```

### **문제점**
1. **Stage 1**: 3e-6은 fine-tuning으로는 **약간 높음**
2. **Stage 2 backbone**: 3e-8은 **너무 낮아서 거의 학습 안됨**
3. Loss function 변경 상황을 고려하지 않음

---

## 💡 권장 Learning Rate 전략

### **Option 1: 안전한 Fine-tuning** ⭐ **추천**

```bash
--lr 5e-6  # 원래 LR을 낮춤
```

**실제 적용되는 LR**:
```python
Stage 1 (classifier):     5e-6 × 0.3 = 1.5e-6  ✅ 적절
Stage 2 (backbone):       5e-6 × 0.3 / 100 = 1.5e-8  ⚠️ 여전히 낮음
Stage 2 (classifier):     5e-6 × 0.3 / 10  = 1.5e-7  ✅ 적절
```

**장점**:
- Stage 1 LR이 fine-tuning에 적합
- Classifier 위주로 조정 (안전)
- Baseline 성능 보존

**단점**:
- Backbone 학습이 느릴 수 있음

---

### **Option 2: 균형잡힌 Fine-tuning**

```bash
--lr 1e-5  # 현재 설정 유지하되, 코드의 자동 감소 비율을 조정
```

**코드 수정**: Stage 2의 differential LR 비율 조정
```python
# 현재: backbone 1/100, classifier 1/10
backbone_lr = (opts.lr / 100) * 0.3    # 3e-8  → 너무 낮음
classifier_lr = (opts.lr / 10) * 0.3   # 3e-7  → 적절

# 수정: backbone 1/20, classifier 1/5
backbone_lr = (opts.lr / 20) * 0.3     # 1.5e-7  → 적절 ✅
classifier_lr = (opts.lr / 5) * 0.3    # 6e-7    → 적절 ✅
```

**장점**:
- Backbone도 충분히 학습
- 더 빠른 수렴
- 더 큰 개선 가능성

**단점**:
- 약간 더 불안정할 수 있음

---

### **Option 3: 보수적 Fine-tuning** (초안전)

```bash
--lr 2e-6
```

**실제 적용되는 LR**:
```python
Stage 1 (classifier):     2e-6 × 0.3 = 6e-7   ✅ 매우 안전
Stage 2 (backbone):       2e-6 × 0.3 / 100 = 6e-9   ⚠️ 너무 낮음
Stage 2 (classifier):     2e-6 × 0.3 / 10  = 6e-8   ⚠️ 약간 낮음
```

**장점**:
- 가장 안전
- Baseline 성능 절대 안 망가짐

**단점**:
- 개선 속도가 느릴 수 있음
- 200 epochs로 충분한 개선 어려울 수 있음

---

## 🎯 최종 추천

### **추천 방법: Option 2 (균형잡힌 Fine-tuning)**

#### **1단계: 코드 수정** (differential LR 비율 조정)
```python
# my_train.py의 Stage 2 부분
# 기존 1/100, 1/10 → 1/20, 1/5로 변경
```

#### **2단계: 실행 명령어**
```bash
--lr 1e-5  # 그대로 유지
```

#### **3단계: 최종 LR**
```
Stage 1: 3e-6       (classifier만, 충분히 학습)
Stage 2: 1.5e-7     (backbone, 적절히 학습)
Stage 2: 6e-7       (classifier, 잘 학습)
```

---

## 📈 모니터링 지표

훈련 시작 후 **처음 5 epochs** 동안 확인:

### ✅ **정상 (LR이 적절)**
```
Epoch 1: Loss 0.450 → mIoU 0.65 (Baseline과 비슷)
Epoch 2: Loss 0.440 → mIoU 0.66 (약간 상승)
Epoch 3: Loss 0.435 → mIoU 0.67 (지속 상승)
Epoch 4: Loss 0.428 → mIoU 0.68 (안정적)
Epoch 5: Loss 0.425 → mIoU 0.68 (수렴 시작)
```

### ❌ **비정상 (LR이 너무 높음)**
```
Epoch 1: Loss 0.550 → mIoU 0.55 (Baseline보다 나쁨!)
Epoch 2: Loss 0.580 → mIoU 0.52 (더 악화)
Epoch 3: Loss 0.600 → mIoU 0.50 (계속 하락)
→ 즉시 중단하고 LR 낮춰야 함
```

### ⚠️ **비정상 (LR이 너무 낮음)**
```
Epoch 1: Loss 0.455 → mIoU 0.650 (Baseline과 동일)
Epoch 2: Loss 0.454 → mIoU 0.650 (거의 변화 없음)
Epoch 3: Loss 0.454 → mIoU 0.651 (너무 느림)
Epoch 5: Loss 0.453 → mIoU 0.651 (개선이 미미)
→ LR 약간 높이는 것 고려
```

---

## 🔧 실전 체크리스트

### 훈련 시작 전
- [ ] Baseline mIoU 확인 (비교 기준)
- [ ] LR 설정 확인 (1e-5 또는 5e-6)
- [ ] WandB 활성화 (모니터링)

### 첫 5 epochs
- [ ] Loss가 Baseline 근처에서 시작하는지
- [ ] mIoU가 하락하지 않는지
- [ ] Gradient Norm < 10 유지하는지

### 15 epochs (Stage 2 시작)
- [ ] mIoU가 Baseline 대비 개선되었는지
- [ ] Loss가 안정적인지

### 전체 훈련
- [ ] Minority class IoU 개선 확인
- [ ] Early stopping 활성화 (과적합 방지)

---

## 📝 요약

| 항목 | 권장값 | 이유 |
|------|--------|------|
| **Initial LR (--lr)** | 1e-5 | 그대로 유지 |
| **Differential LR 비율** | 1/20, 1/5 | 코드 수정 필요 |
| **Stage 1 실제 LR** | 3e-6 | Fine-tuning 적합 |
| **Stage 2 Backbone** | 1.5e-7 | 충분히 학습 |
| **Stage 2 Classifier** | 6e-7 | 적절한 조정 |
| **첫 5 epochs 목표** | Loss ≈ Baseline, mIoU ≥ Baseline | 안정성 확인 |

코드를 수정하시겠습니까?

