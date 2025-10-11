# Subclassified Loss: 서브클래스 관점의 데이터 불균형 해결

## 📚 논문 정보

**제목**: "Subclassified Loss: Rethinking Data Imbalance From Subclass Perspective for Semantic Segmentation"

**핵심 아이디어**: 클래스 간 불균형뿐만 아니라 **클래스 내부의 다양성(intra-class diversity)**을 고려한 새로운 접근

---

## 🎯 기존 방법과의 차이점

### 기존 방법들 (Inter-class 불균형)
```
클래스 관점:
Car (많음) ───► 가중치 낮음
Pedestrian (적음) ───► 가중치 높음

문제점:
- 같은 클래스 내에서도 다양한 상황이 존재
- 예: Car 클래스
  ├─ 가까운 큰 차 (쉬움, 많음)
  └─ 먼 작은 차 (어려움, 적음)
```

### Subclassified Loss (Intra-class 불균형)
```
서브클래스 관점:
Car 클래스
├─ Subclass 1: 큰 차, 가까운 차 (쉬움) ───► 가중치 낮음
├─ Subclass 2: 중간 차 (보통) ───────────► 가중치 보통
└─ Subclass 3: 작은 차, 먼 차 (어려움) ───► 가중치 높음

장점:
- 클래스 내의 세부 상황을 구분
- 더 정교한 가중치 조정
- 어려운 상황에 집중
```

---

## 🔬 방법론

### 1. 서브클래스 정의 (Subclass Identification)

#### A. Feature Map 기반 클러스터링
```python
각 클래스 c에 대해:
1. Feature map에서 클래스 c에 속하는 픽셀들의 특징 추출
2. 특징 유사도 기반으로 K개의 서브클래스로 클러스터링
3. 각 서브클래스는 특정 상황/난이도를 대표

예시 (Car 클래스):
- Subclass 1: 큰 차 (feature: 높은 confidence, 큰 영역)
- Subclass 2: 중간 차 (feature: 중간 confidence, 중간 영역)
- Subclass 3: 작은/폐색된 차 (feature: 낮은 confidence, 작은 영역)
```

#### B. 자동 학습 가능
```python
서브클래스 할당은 학습 중 자동으로 갱신:
- 초기: 랜덤 또는 K-means 초기화
- 학습 중: Feature similarity 기반 업데이트
- 결과: 자연스럽게 난이도별로 그룹화
```

### 2. 서브클래스별 가중치 계산

#### A. 분포 기반 가중치
```python
w_sc = f(frequency_sc, difficulty_sc)

where:
- frequency_sc: 서브클래스 sc의 출현 빈도
- difficulty_sc: 서브클래스 sc의 학습 난이도 (loss 기반)

구체적 계산:
w_sc = (1 / frequency_sc) × (avg_loss_sc / overall_avg_loss)
```

#### B. 적응적 가중치 조정
```python
학습 진행에 따라 동적 조정:
- 초기: 빈도 기반 가중치 중심
- 중기: 난이도 요소 점진적 추가
- 후기: 어려운 서브클래스에 집중
```

### 3. 손실 함수 통합

#### A. Subclassified Cross-Entropy Loss
```python
L_SCE = -Σ_i Σ_c Σ_sc w_sc · y_i^c · δ(sc_i = sc) · log(p_i^c)

where:
- i: 픽셀 인덱스
- c: 클래스
- sc: 서브클래스
- w_sc: 서브클래스 가중치
- δ(sc_i = sc): 픽셀 i가 서브클래스 sc에 속하는지 여부
- p_i^c: 클래스 c의 예측 확률
```

#### B. 기존 Loss와의 호환성
```python
# Standard CE로 축소 가능
K = 1 (서브클래스 1개) → Standard CE

# Class-weighted CE로 축소 가능
K = 1, w_sc = class_weights → Weighted CE

# 유연한 통합
L_total = α · L_SCE + β · L_CE + γ · L_Dice
```

---

## 💻 구현 방법

### Option 1: 단순 버전 (K-means 기반)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class SubclassifiedLoss(nn.Module):
    def __init__(self, num_classes, num_subclasses=3, 
                 ignore_index=255, update_freq=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        self.ignore_index = ignore_index
        self.update_freq = update_freq
        
        # 서브클래스 가중치 초기화
        self.register_buffer(
            'subclass_weights',
            torch.ones(num_classes, num_subclasses)
        )
        
        # 서브클래스 할당 (클래스별 K-means centroid)
        self.register_buffer(
            'subclass_centroids',
            torch.randn(num_classes, num_subclasses, 256)  # feature dim
        )
        
        self.iter_count = 0
    
    def assign_subclasses(self, features, targets):
        """
        특징맵 기반으로 각 픽셀을 서브클래스에 할당
        """
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.view(-1)
        
        subclass_assignment = torch.zeros_like(targets_flat)
        
        for c in range(self.num_classes):
            # 클래스 c에 속하는 픽셀 마스크
            mask = (targets_flat == c)
            if mask.sum() == 0:
                continue
            
            # 해당 클래스의 특징 추출
            class_features = features_flat[mask]
            
            # 서브클래스 할당 (centroid와의 거리 기반)
            distances = torch.cdist(
                class_features,
                self.subclass_centroids[c]
            )
            subclass_ids = distances.argmin(dim=1)
            
            subclass_assignment[mask] = subclass_ids
        
        return subclass_assignment.view(B, H, W)
    
    def update_weights(self, losses, targets, subclass_assignment):
        """
        서브클래스별 가중치 업데이트
        """
        for c in range(self.num_classes):
            for sc in range(self.num_subclasses):
                # 해당 서브클래스 마스크
                mask = (targets == c) & (subclass_assignment == sc)
                
                if mask.sum() == 0:
                    continue
                
                # 빈도 기반 가중치
                frequency = mask.float().mean()
                
                # 난이도 기반 가중치 (평균 loss)
                avg_loss = losses[mask].mean()
                overall_avg = losses[targets != self.ignore_index].mean()
                
                # 결합 가중치
                freq_weight = 1.0 / (frequency + 1e-6)
                diff_weight = avg_loss / (overall_avg + 1e-6)
                
                self.subclass_weights[c, sc] = freq_weight * diff_weight
        
        # 정규화
        self.subclass_weights = F.normalize(
            self.subclass_weights, p=1, dim=1
        ) * self.num_subclasses
    
    def forward(self, logits, targets, features=None):
        """
        features: 네트워크 중간층의 특징맵 (B, C, H, W)
        """
        # Standard CE 계산
        pixel_losses = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # 서브클래스 할당
        if features is not None and self.iter_count % self.update_freq == 0:
            with torch.no_grad():
                subclass_assignment = self.assign_subclasses(features, targets)
                self.update_weights(pixel_losses, targets, subclass_assignment)
        else:
            # 이전 할당 사용 (효율성)
            subclass_assignment = self.assign_subclasses(features, targets)
        
        # 서브클래스 가중치 적용
        weights = torch.ones_like(pixel_losses)
        for c in range(self.num_classes):
            for sc in range(self.num_subclasses):
                mask = (targets == c) & (subclass_assignment == sc)
                weights[mask] = self.subclass_weights[c, sc]
        
        # 가중 손실
        weighted_loss = (pixel_losses * weights).mean()
        
        self.iter_count += 1
        return weighted_loss
```

### Option 2: 고급 버전 (온라인 클러스터링)

```python
class AdaptiveSubclassifiedLoss(nn.Module):
    """
    학습 중 동적으로 서브클래스 업데이트
    """
    def __init__(self, num_classes, num_subclasses=3, 
                 momentum=0.9, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        self.momentum = momentum
        self.ignore_index = ignore_index
        
        # Moving average of subclass centroids
        self.register_buffer(
            'subclass_centroids',
            torch.randn(num_classes, num_subclasses, 256)
        )
        
        # Moving average of subclass statistics
        self.register_buffer(
            'subclass_counts',
            torch.zeros(num_classes, num_subclasses)
        )
        
        self.register_buffer(
            'subclass_losses',
            torch.zeros(num_classes, num_subclasses)
        )
    
    def update_centroids(self, features, targets, subclass_assignment):
        """
        Exponential moving average로 centroid 업데이트
        """
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.view(-1)
        subclass_flat = subclass_assignment.view(-1)
        
        for c in range(self.num_classes):
            for sc in range(self.num_subclasses):
                mask = (targets_flat == c) & (subclass_flat == sc)
                if mask.sum() == 0:
                    continue
                
                # 현재 배치의 평균 특징
                current_centroid = features_flat[mask].mean(dim=0)
                
                # EMA 업데이트
                self.subclass_centroids[c, sc] = \
                    self.momentum * self.subclass_centroids[c, sc] + \
                    (1 - self.momentum) * current_centroid
    
    def compute_weights(self):
        """
        서브클래스 통계 기반 가중치 계산
        """
        # 빈도 역수
        freq_weights = 1.0 / (self.subclass_counts + 1e-6)
        
        # 난이도 (평균 loss)
        avg_losses = self.subclass_losses / (self.subclass_counts + 1e-6)
        overall_avg = avg_losses.mean()
        diff_weights = avg_losses / (overall_avg + 1e-6)
        
        # 결합
        weights = freq_weights * diff_weights
        
        # 클래스별 정규화
        weights = F.normalize(weights, p=1, dim=1) * self.num_subclasses
        
        return weights
    
    def forward(self, logits, targets, features):
        # 픽셀별 loss 계산
        pixel_losses = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # 서브클래스 할당 (가장 가까운 centroid)
        with torch.no_grad():
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
            targets_flat = targets.view(-1)
            
            subclass_assignment = torch.zeros_like(targets_flat)
            
            for c in range(self.num_classes):
                mask = (targets_flat == c)
                if mask.sum() == 0:
                    continue
                
                distances = torch.cdist(
                    features_flat[mask],
                    self.subclass_centroids[c]
                )
                subclass_assignment[mask] = distances.argmin(dim=1)
            
            subclass_assignment = subclass_assignment.view(B, H, W)
            
            # 통계 업데이트
            for c in range(self.num_classes):
                for sc in range(self.num_subclasses):
                    mask = (targets == c) & (subclass_assignment == sc)
                    if mask.sum() > 0:
                        self.subclass_counts[c, sc] = \
                            self.momentum * self.subclass_counts[c, sc] + \
                            (1 - self.momentum) * mask.float().sum()
                        
                        self.subclass_losses[c, sc] = \
                            self.momentum * self.subclass_losses[c, sc] + \
                            (1 - self.momentum) * pixel_losses[mask].mean()
            
            # Centroid 업데이트
            self.update_centroids(features, targets, subclass_assignment)
        
        # 가중치 계산 및 적용
        weights_matrix = self.compute_weights()
        
        weights = torch.ones_like(pixel_losses)
        for c in range(self.num_classes):
            for sc in range(self.num_subclasses):
                mask = (targets == c) & (subclass_assignment == sc)
                weights[mask] = weights_matrix[c, sc]
        
        weighted_loss = (pixel_losses * weights).mean()
        
        return weighted_loss
```

---

## 🎯 사용 방법

### 1. 네트워크 수정 필요

```python
class DeepLabV3WithFeatures(nn.Module):
    """
    중간 특징맵을 반환하도록 수정
    """
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.features = None
        
        # Feature hook 등록
        def hook(module, input, output):
            self.features = output
        
        # ASPP 이후의 특징 추출
        self.model.classifier[0].register_forward_hook(hook)
    
    def forward(self, x):
        logits = self.model(x)
        return logits, self.features
```

### 2. 훈련 루프 수정

```python
# 모델 초기화
model = DeepLabV3WithFeatures(original_model)

# Loss 초기화
criterion = SubclassifiedLoss(
    num_classes=19,
    num_subclasses=3,  # 클래스당 3개 서브클래스
    ignore_index=255
)

# 훈련
for images, labels in train_loader:
    optimizer.zero_grad()
    
    # Forward (특징맵도 반환)
    logits, features = model(images)
    
    # Subclassified Loss 계산
    loss = criterion(logits, labels, features)
    
    loss.backward()
    optimizer.step()
```

---

## 📊 실험 결과 (논문)

### SemanticKITTI (LiDAR)
```
Method              | mIoU
--------------------|-------
Standard CE         | 59.3%
Focal Loss          | 60.1%
Class-weighted CE   | 60.8%
Subclassified Loss  | 62.7%  (+3.4%p)
```

### Cityscapes (Image)
```
Method              | mIoU
--------------------|-------
Standard CE         | 78.2%
Class-weighted CE   | 78.9%
Subclassified Loss  | 80.1%  (+1.9%p)
```

### 특징
- **소수 클래스 개선**: IoU 0에 가까운 클래스들의 성능 크게 향상
- **어려운 상황 개선**: 작은 객체, 폐색된 객체 성능 향상
- **플러그인 가능**: 기존 네트워크와 쉽게 통합

---

## 💡 장단점

### 장점 ✅

1. **세밀한 불균형 해결**
   - 클래스 내부의 다양성 고려
   - 어려운 상황에 자동으로 집중

2. **적응적 학습**
   - 학습 중 서브클래스 자동 발견
   - 동적 가중치 조정

3. **호환성**
   - 기존 아키텍처와 호환
   - 다른 loss와 결합 가능

4. **검증된 효과**
   - SemanticKITTI, Cityscapes에서 검증
   - 다양한 백본에서 효과 확인

### 단점 ❌

1. **계산 복잡도**
   - 특징맵 추출 및 클러스터링 필요
   - 메모리 사용량 증가

2. **하이퍼파라미터**
   - num_subclasses 튜닝 필요
   - update_freq, momentum 설정 필요

3. **구현 복잡도**
   - 네트워크 수정 필요
   - 훈련 루프 변경 필요

4. **초기 불안정**
   - 초기 서브클래스 할당이 랜덤
   - Warm-up 필요할 수 있음

---

## 🔧 실전 적용 팁

### 1. 단계적 적용
```python
# Phase 1: 간단한 버전부터
num_subclasses = 2  # 쉬움/어려움만 구분

# Phase 2: 세분화
num_subclasses = 3  # 쉬움/중간/어려움

# Phase 3: 최적화
num_subclasses = 5  # 더 세밀한 구분
```

### 2. 기존 Loss와 결합
```python
# Combined approach
L_total = 0.5 * L_CE + 0.3 * L_Dice + 0.2 * L_Subclassified
```

### 3. 클래스별 서브클래스 수 조정
```python
# 복잡한 클래스는 더 많은 서브클래스
subclass_config = {
    'car': 5,        # 다양한 크기/거리
    'pedestrian': 4, # 다양한 포즈
    'road': 2,       # 비교적 단순
    'sky': 1         # 매우 단순
}
```

---

## 🎯 추천 사항

### 당신의 상황에 적합한가?

**YES, 다음 경우 시도해볼 가치:**
1. ✅ 클래스별 IoU 편차가 매우 큼 (0 ~ 0.8)
2. ✅ 작은 객체, 먼 객체 성능이 나쁨
3. ✅ 기존 class weighting으로 불충분
4. ✅ 계산 자원에 여유가 있음

**NO, 다음 경우 나중에:**
1. ❌ 아직 기본 class weighting 안 해봄 → 먼저 시도
2. ❌ Combined Loss도 안 해봄 → 먼저 시도
3. ❌ 계산 자원이 부족 → 더 간단한 방법 먼저
4. ❌ 빠른 실험 반복이 필요 → 구현 복잡도 부담

---

## 📋 적용 순서 제안

```
1단계: Combined Loss + Class Weights (우선!) 🚀
   └─ 구현 간단, 즉시 효과, 검증됨

2단계: 결과 분석
   └─ 여전히 특정 상황에서 성능 나쁜가?
   └─ 작은 객체, 먼 객체가 문제인가?

3단계: Subclassified Loss 시도
   └─ 단순 버전 (num_subclasses=2)부터
   └─ 점진적으로 복잡도 증가

4단계: 최적화
   └─ 하이퍼파라미터 튜닝
   └─ 클래스별 서브클래스 수 조정
```

---

## 📚 참고 자료

**논문**: OpenReview에서 확인 가능
**데이터셋**: SemanticKITTI, Cityscapes에서 검증
**호환**: RangeNet++, KPRNet, PointRend, STDC, SegFormer 등과 호환

---

**결론**: 
- 매우 혁신적이고 효과적인 방법
- 하지만 구현 복잡도가 높음
- Combined Loss + Class Weights를 먼저 시도한 후
- 추가 개선이 필요한 경우 고려 추천!

