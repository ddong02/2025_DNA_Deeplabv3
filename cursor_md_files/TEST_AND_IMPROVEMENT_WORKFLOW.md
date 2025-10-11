# Test & Improvement Workflow

Baseline 훈련 완료 후 체계적으로 개선하는 워크플로우입니다.

## 📊 현재 상태
- ✅ Baseline 훈련 완료 (Standard Cross-Entropy Loss)
- ✅ Validation 성능 기록됨 (WandB)
- ⏳ **다음 단계**: Test 성능 측정 및 기록

---

## 🎯 워크플로우

### Step 1: Baseline Test 성능 측정 및 WandB 기록

#### 방법 A: 새로운 Test Run 생성 (추천)
```bash
python my_test.py \
    --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
    --save_results \
    --output_dir test_results_baseline \
    --save_samples \
    --num_samples 10 \
    --evaluate_performance \
    --use_wandb \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_run_name "baseline-test" \
    --wandb_tags "baseline,test"
```

**결과**: 
- 새로운 WandB run 생성: `baseline-test`
- Test 결과가 `[Test]` prefix로 기록됨
- 로컬 결과 저장: `test_results_baseline/` 폴더
- 태그: `baseline`, `test`

#### 방법 B: 기존 Training Run에 Test 결과 추가
```bash
# 1. WandB에서 baseline 훈련 run ID 확인
#    예: https://wandb.ai/yourname/project/runs/abc123xyz
#    Run ID = abc123xyz

# 2. 해당 run에 test 결과 추가
python my_test.py \
    --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
    --save_results \
    --output_dir test_results_baseline \
    --evaluate_performance \
    --use_wandb \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_run_id "abc123xyz"
```

**결과**: 
- 기존 훈련 run에 test 지표 추가
- Training과 Test 결과를 한 곳에서 확인 가능

---

### Step 2: WandB에서 Baseline 성능 확인

WandB 대시보드에서 다음을 확인:

#### 1. **Test 지표**
- `[Test] Mean IoU`: 최종 평가 지표
- `[Test] Overall Acc`: 전체 정확도
- `[Test] Class IoU Table`: 클래스별 성능

#### 2. **클래스별 분석**
어떤 클래스의 성능이 낮은지 확인:
```
예시:
Class 0 (Drivable Area): 85.2%  ← 잘 됨
Class 6 (Car): 72.1%            ← 괜찮음
Class 11 (Pedestrian): 32.5%    ← 낮음 (개선 필요)
Class 13 (Traffic Cone): 15.8%  ← 매우 낮음 (심각한 불균형)
```

#### 3. **성능 지표** (optional)
- `[Model] FLOPs (G)`: 연산량
- `[Model] FPS`: 속도
- `[Model] Parameters (M)`: 모델 크기

---

### Step 3: 개선 방법 적용 (다음 단계)

Baseline Test 성능 확보 후, 다음 개선 방법들을 하나씩 적용:

#### 3.1 실험 계획
```
실험 1: Baseline (완료)
├─ Training: Standard CE Loss
└─ Test Mean IoU: 45.2% (예시)

실험 2: + Class Weights
├─ Training: CE + sqrt_inv_freq weights
└─ Test Mean IoU: ??% (측정 예정)

실험 3: + Dice Loss
├─ Training: Dice Loss only
└─ Test Mean IoU: ??% (측정 예정)

실험 4: + Combined Loss
├─ Training: CE (60%) + Dice (40%)
└─ Test Mean IoU: ??% (측정 예정)
```

#### 3.2 각 개선 방법 적용 방법

**다음 문서에서 개선 방법들이 추가될 예정입니다.**

현재는 Baseline 측정에 집중합니다!

---

## 📁 생성되는 파일들

### Test 결과 (로컬)
```
test_results_baseline/           # --output_dir로 지정한 폴더
├── test_summary.txt              # 전체 요약
├── class_results.txt             # 클래스별 IoU
├── comprehensive_results.txt     # 성능 지표 포함
├── comparisons/                  # 비교 이미지
│   ├── 00000_comparison.png
│   ├── 00001_comparison.png
│   └── ...
└── predictions/                  # 예측 마스크
    ├── 00000_pred.png
    ├── 00001_pred.png
    └── ...
```

### WandB (클라우드)
```
WandB Run
├── Metrics
│   ├── [Test] Mean IoU
│   ├── [Test] Overall Acc
│   ├── [Test] IoU/Drivable Area
│   ├── [Test] IoU/Car
│   └── ... (모든 클래스)
├── Tables
│   └── [Test] Class IoU Table
├── Images
│   ├── [Test] Sample 0/Image
│   ├── [Test] Sample 0/Ground Truth
│   ├── [Test] Sample 0/Prediction
│   └── ...
└── System
    ├── [Model] Parameters (M)
    ├── [Model] FLOPs (G)
    └── [Model] FPS
```

---

## 🔍 WandB에서 비교하는 방법

### 1. 여러 실험 비교
1. WandB 프로젝트 페이지 접속
2. 좌측 체크박스로 비교할 실험 선택
3. **Charts** 탭에서 지표 비교

### 2. 테이블로 비교
1. **Table** 탭 클릭
2. Columns에서 표시할 지표 선택:
   - `[Test] Mean IoU`
   - `[Test] Overall Acc`
   - Tags (baseline, class-weights, etc.)
3. 정렬하여 최고 성능 확인

### 3. 클래스별 개선 확인
1. 특정 클래스의 IoU 비교
2. 예: `[Test] IoU/Pedestrian` 지표를 여러 실험에서 비교
3. 어떤 방법이 어려운 클래스를 개선했는지 확인

---

## 💡 팁

### 1. Run 이름 및 폴더 규칙
```bash
# Baseline
--wandb_run_name "baseline-test"
--output_dir test_results_baseline

# + Class Weights
--wandb_run_name "baseline-classweights-test"
--output_dir test_results_classweights

# + Dice Loss
--wandb_run_name "baseline-dice-test"
--output_dir test_results_dice

# + Combined Loss
--wandb_run_name "baseline-combined-test"
--output_dir test_results_combined
```

### 2. 태그 활용
```bash
--wandb_tags "baseline,test"                      # Baseline
--wandb_tags "baseline,test,class-weights"        # 개선 1
--wandb_tags "baseline,test,dice-loss"           # 개선 2
--wandb_tags "baseline,test,combined-loss"       # 개선 3
```

### 3. 체크포인트 관리
```
checkpoints/
├── best_..._baseline.pth           # Baseline 최고 모델
├── best_..._classweights.pth       # Class Weights 최고 모델
├── best_..._dice.pth              # Dice Loss 최고 모델
└── best_..._combined.pth          # Combined Loss 최고 모델
```

---

## 🚀 다음 단계

1. **지금**: Baseline Test 실행
   ```bash
   python my_test.py \
       --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
       --save_results \
       --output_dir test_results_baseline \
       --evaluate_performance \
       --use_wandb \
       --wandb_project "deeplabv3-segmentation" \
       --wandb_run_name "baseline-test" \
       --wandb_tags "baseline,test"
   ```

2. **확인**: WandB에서 Baseline 성능 분석
   - Mean IoU는 얼마?
   - 어떤 클래스가 어려운가?
   - Validation vs Test 차이는?

3. **계획**: 개선 방법 선택
   - 클래스 불균형이 심하면 → Class Weights 시도
   - Segmentation boundary가 중요하면 → Dice Loss 시도
   - 둘 다 필요하면 → Combined Loss 시도

4. **실행**: 개선 방법 적용 (다음 단계에서 진행)

---

## ❓ FAQ

### Q1: Test와 Validation의 차이는?
- **Validation**: 훈련 중 매 epoch마다 평가 (하이퍼파라미터 조정용)
- **Test**: 최종 모델 평가 (실제 성능 측정)

### Q2: 어떤 방법(A or B)을 선택해야 하나?
- **방법 A (새 run)**: 깔끔하게 분리, 비교가 쉬움 (추천)
- **방법 B (resume)**: Training과 Test를 한 곳에서 확인

### Q3: Test 시간이 얼마나 걸리나?
- 데이터셋 크기에 따라 다름
- 예: 1000장, batch_size=1 → 약 5-10분

### Q4: 개선 방법은 언제 적용하나?
- Baseline Test 결과를 확인한 후
- 다음 단계에서 `my_train.py`에 개선 옵션 추가 예정

### Q5: --output_dir은 언제 사용하나?
- 각 실험마다 다른 폴더에 결과를 저장하고 싶을 때
- 예: `test_results_baseline`, `test_results_experiment_001` 등
- 지정하지 않으면 기본값 `test_results` 사용

---

**현재 위치**: Step 1 - Baseline Test 실행 단계
**다음 단계**: Test 결과 분석 및 개선 방법 선택
