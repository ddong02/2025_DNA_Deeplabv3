# WandB 사용 가이드 (Weights & Biases)

이 프로젝트는 Visdom 대신 WandB를 사용하여 훈련 과정을 시각화하고 실험 결과를 추적합니다.

## 1. 설치 및 설정

### 1.1 WandB 설치
```bash
pip install wandb
# 또는
pip install -r requirements.txt
```

### 1.2 WandB 계정 설정 및 로그인
WandB를 처음 사용하는 경우:

1. [wandb.ai](https://wandb.ai)에서 무료 계정 생성
2. 터미널에서 로그인:
```bash
wandb login
```
3. API 키 입력 (웹사이트의 Settings > API keys에서 확인 가능)

**참고**: API 키는 한 번만 입력하면 됩니다. 이후에는 자동으로 인증됩니다.

### 1.3 오프라인 모드 (선택사항)
인터넷 연결 없이 로컬에서만 로그를 저장하려면:
```bash
export WANDB_MODE=offline
# 또는 코드 실행 시
WANDB_MODE=offline python my_train.py ...
```

## 2. 기본 사용법

### 2.1 훈련 시작 (WandB 활성화)
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
    --wandb_project "my-segmentation-project" \
    --wandb_name "deeplabv3-mobilenet-experiment-1" \
    --wandb_tags "baseline,mobilenet" \
    --save_val_results \
    --loss_type combined \
    --ce_weight 0.6 \
    --dice_weight 0.4
```

### 2.2 WandB 옵션 설명

| 옵션 | 설명 | 기본값 | 예시 |
|-----|------|--------|------|
| `--enable_vis` | WandB 활성화 (필수) | False | `--enable_vis` |
| `--wandb_project` | 프로젝트 이름 | deeplabv3-semantic-segmentation | `--wandb_project "my-project"` |
| `--wandb_name` | 실험 이름 (자동 생성 가능) | None (자동) | `--wandb_name "exp-mobilenet-v1"` |
| `--wandb_notes` | 실험에 대한 메모 | None | `--wandb_notes "Testing combined loss"` |
| `--wandb_tags` | 쉼표로 구분된 태그 | None | `--wandb_tags "baseline,test,v1"` |

### 2.3 WandB 비활성화 (로컬 훈련만)
WandB를 사용하지 않으려면 `--enable_vis` 플래그를 제거하면 됩니다:
```bash
python my_train.py \
    --dataset dna2025dataset \
    --model deeplabv3_mobilenet \
    # ... 다른 옵션들 ...
    # --enable_vis 없이 실행
```

## 3. WandB에서 확인 가능한 정보

### 3.1 자동으로 기록되는 메트릭
- **Training Loss**: 에폭별 평균 훈련 손실
- **Learning Rate**: 현재 학습률 (backbone/classifier 구분)
- **[Val] Overall Acc**: 전체 정확도
- **[Val] Mean Acc**: 클래스별 평균 정확도
- **[Val] Mean IoU**: 클래스별 평균 IoU (주요 지표)
- **[Val] Class IoU**: 각 클래스별 IoU (테이블)

### 3.2 시각화 샘플
- **Validation Sample 0-3**: 검증 이미지, 정답 레이블, 예측 결과 비교
- 에폭마다 자동 업데이트

### 3.3 Config (실험 설정)
훈련에 사용된 모든 하이퍼파라미터가 자동으로 기록됩니다:
- Model architecture
- Learning rate, batch size, epochs
- Loss function type and weights
- Data augmentation parameters
- 기타 모든 command-line arguments

## 4. WandB 대시보드 활용

### 4.1 실험 비교
- 여러 실험을 동시에 실행하면 자동으로 비교 가능
- 좌측 패널에서 비교할 실험 선택
- 그래프에서 여러 실험의 메트릭을 동시에 확인

### 4.2 하이퍼파라미터 탐색
- WandB Sweeps를 사용하여 자동 하이퍼파라미터 튜닝 가능
- 자세한 내용: [WandB Sweeps 문서](https://docs.wandb.ai/guides/sweeps)

### 4.3 실험 필터링 및 검색
- Tags를 사용하여 실험 그룹화
- 프로젝트 페이지에서 태그별로 필터링
- 특정 메트릭 기준으로 정렬

## 5. 기존 Visdom 대비 장점

| 기능 | Visdom | WandB |
|------|--------|-------|
| **서버 필요** | ✗ (별도 서버 실행 필요) | ✓ (클라우드 기반) |
| **실험 추적** | 제한적 | ✓ (자동, 영구 저장) |
| **실험 비교** | 수동 | ✓ (자동, 직관적) |
| **하이퍼파라미터 기록** | 수동 | ✓ (자동) |
| **협업** | 어려움 | ✓ (쉬움) |
| **모델 아티팩트 저장** | 없음 | ✓ (지원) |
| **무료 사용** | ✓ | ✓ (개인 사용) |

## 6. 고급 기능

### 6.1 모델 체크포인트 업로드
코드에 다음을 추가하여 최고 성능 모델을 자동으로 업로드할 수 있습니다:
```python
import wandb

# 최고 성능 모델 저장 시
artifact = wandb.Artifact('best-model', type='model')
artifact.add_file('checkpoints/best_model.pth')
wandb.log_artifact(artifact)
```

### 6.2 실험 재개 (Resume)
중단된 실험을 이어서 계속하려면:
```python
# visualizer.py의 __init__에서 resume='allow' 설정 가능
vis = Visualizer(
    project='my-project',
    name='experiment-1',
    resume='allow',  # 또는 'must'
    id='unique-run-id'  # 재개할 run의 ID
)
```

## 7. 문제 해결

### 7.1 API 키 오류
```bash
wandb login --relogin
```

### 7.2 로그가 업로드되지 않음
- 인터넷 연결 확인
- `wandb sync` 명령으로 수동 동기화
```bash
wandb sync wandb/run-*
```

### 7.3 디스크 공간 부족
로컬 캐시 정리:
```bash
wandb artifact cache cleanup
```

## 8. 추가 자료

- [WandB 공식 문서](https://docs.wandb.ai/)
- [WandB Python API 레퍼런스](https://docs.wandb.ai/ref/python)
- [WandB 예제 프로젝트](https://wandb.ai/gallery)

## 9. 이전 Visdom 코드와의 차이점

### 이전 (Visdom):
```bash
# 1. Visdom 서버 시작 (별도 터미널)
python -m visdom.server -port 28333

# 2. 훈련 실행
python my_train.py --enable_vis --vis_port 28333 --vis_env main
```

### 현재 (WandB):
```bash
# 1. 로그인 (최초 1회만)
wandb login

# 2. 훈련 실행 (서버 불필요)
python my_train.py --enable_vis --wandb_project "my-project"
```

## 10. 비용 및 제한

- **무료 플랜**: 개인 사용 (100GB 저장소, 무제한 실험)
- **팀 플랜**: 협업 필요 시 (유료)
- 자세한 내용: [WandB Pricing](https://wandb.ai/site/pricing)

---

**참고**: 이 프로젝트의 모든 훈련 스크립트 (`my_train.py`, `my_test.py` 등)는 WandB를 지원합니다.
