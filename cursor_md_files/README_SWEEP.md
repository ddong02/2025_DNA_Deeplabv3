# WandB Sweep을 이용한 하이퍼파라미터 최적화

이 가이드는 Learning Rate와 Target Max Ratio를 WandB Sweep으로 자동 최적화하는 방법을 설명합니다.

## 🎯 최적화 목표

- **Learning Rate**: 1e-7 ~ 1e-5 범위에서 최적값 탐색
- **Target Max Ratio**: 5.0 ~ 20.0 범위에서 최적값 탐색
- **목표**: Mean IoU 최대화

## 📋 사전 준비

### 1. 필수 요구사항
```bash
# WandB 설치 및 로그인
pip install wandb
wandb login

# Baseline 모델 확인
ls checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth
```

### 2. 클래스 가중치 파일 (선택사항)
```bash
# 이미 계산된 가중치가 있다면
ls class_weights/class_weight.pth
```

## 🚀 실행 방법

### 방법 1: 자동화된 스크립트 사용 (추천)

```bash
python run_sweep.py
```

**스크립트가 자동으로 처리하는 것들:**
- ✅ 요구사항 확인
- ✅ WandB Sweep 생성
- ✅ Sweep Agent 실행
- ✅ 결과 가이드 제공

### 방법 2: 수동 실행

#### 1단계: Sweep 생성
```bash
wandb sweep sweep_config.yaml
```

#### 2단계: Sweep Agent 실행
```bash
# 출력된 Sweep ID 사용 (예: abc123xyz)
wandb agent <sweep_id>
```

## 📊 Sweep 설정 상세

### 최적화 파라미터
```yaml
lr:
  distribution: log_uniform_values
  min: 1e-7
  max: 1e-5

target_max_ratio:
  distribution: uniform
  min: 5.0
  max: 20.0
```

### 고정 파라미터
- **Epochs**: 50 (빠른 탐색을 위해)
- **Early Stopping**: 8 epochs patience
- **Batch Size**: 4
- **Model**: deeplabv3_mobilenet
- **Dataset**: dna2025dataset

## 📈 결과 확인

### 1. WandB 대시보드
1. https://wandb.ai 접속
2. 프로젝트 `deeplabv3-segmentation` 선택
3. Sweep 탭에서 최적화 결과 확인

### 2. 최적 조합 찾기
- **Parallel Coordinates Plot**: 하이퍼파라미터 vs 성능 관계
- **Parameter Importance**: 각 파라미터의 중요도
- **Best Run**: 최고 성능을 보인 조합

### 3. 최종 훈련 실행
최적 조합을 찾은 후:
```bash
python my_train.py \
    --lr <optimal_lr> \
    --target_max_ratio <optimal_ratio> \
    --epochs 200 \
    --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
    --enable_vis \
    --wandb_project "final-training" \
    --wandb_name "optimal-parameters"
```

## ⏱️ 예상 소요 시간

- **각 Trial**: 30-60분 (50 epochs)
- **총 10 Trials**: 5-10시간
- **Early Stopping**: 성능이 개선되지 않으면 조기 종료

## 🔧 문제 해결

### 1. WandB 로그인 문제
```bash
wandb login --relogin
```

### 2. GPU 메모리 부족
```bash
# sweep_config.yaml에서 batch_size 줄이기
batch_size:
  value: 2  # 4에서 2로 변경
```

### 3. 체크포인트 파일 없음
```bash
# 먼저 baseline 모델 훈련
python my_train.py --epochs 200 --lr 1e-3 ...
```

## 📝 예상 결과

### 최적 조합 예시
```
lr: 2.3e-6
target_max_ratio: 8.5
Mean IoU: 0.78
```

### 성능 개선
- **Baseline**: Mean IoU 0.72
- **Optimized**: Mean IoU 0.78 (+0.06)
- **개선율**: 8.3% 향상

## 🎯 다음 단계

1. **Sweep 완료 후**: 최적 조합으로 최종 훈련
2. **성능 검증**: Test 데이터셋으로 최종 평가
3. **모델 배포**: 최적화된 모델을 프로덕션에 사용
