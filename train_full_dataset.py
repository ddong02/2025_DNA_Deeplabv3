"""
Albumentations-based Advanced Augmentation Training Script for DeepLabV3+
Albumentations 기반 고급 증강 기법을 적용한 훈련 스크립트
"""

from tqdm import tqdm
import network
import utils
import os
import random
import numpy as np
import time
import wandb

from torch.utils import data
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== Import separated modules =====
from my_utils.training_args import get_argparser
from my_utils.dna2025_dataset import DNA2025Dataset
from my_utils.dna2025_dataset_combined import DNA2025CombinedDataset
from my_utils.validation import validate
from my_utils.checkpoint import save_checkpoint, load_pretrained_model, load_checkpoint
from my_utils.early_stopping import EarlyStopping
from my_utils.calculate_class_weights import calculate_class_weights
from my_utils.losses import CombinedLoss, FocalLoss
from my_utils.albumentations_augmentation import AlbumentationsSegmentationTransform, AlbumentationsValidationTransform
import torch.nn.functional as F


def get_dataset(opts):
    """Dataset And Albumentations Advanced Augmentation - DNA2025 only"""
    # Albumentations 고급 증강 파라미터 설정 (WandB config에서 가져오거나 기본값 사용)
    use_weather = getattr(opts, 'use_weather', False)
    use_noise = getattr(opts, 'use_noise', False)
    use_blur = getattr(opts, 'use_blur', False)
    use_cutout = getattr(opts, 'use_cutout', False)
    use_geometric = getattr(opts, 'use_geometric', False)
    use_color = getattr(opts, 'use_color', False)
    
    # 기본 증강 파라미터
    horizontal_flip_p = getattr(opts, 'horizontal_flip_p', 0.5)
    brightness_limit = getattr(opts, 'brightness_limit', 0.2)
    contrast_limit = getattr(opts, 'contrast_limit', 0.2)
    rotation_limit = getattr(opts, 'rotation_limit', 5)
    
    # 고급 증강 파라미터
    weather_p = getattr(opts, 'weather_p', 0.3)
    noise_p = getattr(opts, 'noise_p', 0.3)
    blur_p = getattr(opts, 'blur_p', 0.3)
    cutout_p = getattr(opts, 'cutout_p', 0.3)
    geometric_p = getattr(opts, 'geometric_p', 0.3)
    color_p = getattr(opts, 'color_p', 0.3)
    
    # 증강 파라미터 출력
    print("\n" + "="*60)
    print("🎨 ALBUMENTATIONS ADVANCED AUGMENTATION PARAMETERS")
    print("="*60)
    print(f"Horizontal Flip Probability: {horizontal_flip_p:.1%}")
    print(f"Brightness Limit:          ±{brightness_limit:.2f}")
    print(f"Contrast Limit:            ±{contrast_limit:.2f}")
    print(f"Rotation Limit:            ±{rotation_limit}°")
    print(f"Weather Simulation:       {use_weather} (p={weather_p:.1%})")
    print(f"Noise Addition:            {use_noise} (p={noise_p:.1%})")
    print(f"Blur Effects:             {use_blur} (p={blur_p:.1%})")
    print(f"CutOut Effects:           {use_cutout} (p={cutout_p:.1%})")
    print(f"Geometric Transform:      {use_geometric} (p={geometric_p:.1%})")
    print(f"Color Transform:          {use_color} (p={color_p:.1%})")
    print("="*60)
    
    # Albumentations Transform 생성
    train_transform = AlbumentationsSegmentationTransform(
        crop_size=[opts.crop_size, opts.crop_size],
        scale_range=[0.75, 1.25],
        # 기본 증강 파라미터
        horizontal_flip_p=horizontal_flip_p,
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        rotation_limit=rotation_limit,
        # Albumentations 고급 증강 설정
        use_weather=use_weather,
        weather_p=weather_p,
        use_noise=use_noise,
        noise_p=noise_p,
        use_blur=use_blur,
        blur_p=blur_p,
        use_cutout=use_cutout,
        cutout_p=cutout_p,
        use_geometric=use_geometric,
        geometric_p=geometric_p,
        use_color=use_color,
        color_p=color_p
    )
    
    # Check if we should use combined dataset
    use_combined_dataset = getattr(opts, 'use_combined_dataset', False)
    
    if use_combined_dataset:
        print("🔄 Using combined train+val dataset for training...")
        train_dst = DNA2025CombinedDataset(
            root_dir=opts.data_root,
            crop_size=[opts.crop_size, opts.crop_size],
            subset='train',
            scale_range=[0.75, 1.25],
            random_seed=opts.random_seed,
            subset_ratio=getattr(opts, 'subset_ratio', 1.0),
            combine_train_val=True,  # Combine train and val data
            # Albumentations Transform 사용
            custom_transform=train_transform
        )
        print(f"📊 Combined dataset loaded: {len(train_dst)} samples")
        print(f"   - Train samples: {train_dst.train_samples}")
        print(f"   - Val samples: {train_dst.val_samples}")
        print(f"   - Subset ratio: {getattr(opts, 'subset_ratio', 1.0):.1%}")
    else:
        train_dst = DNA2025Dataset(
            root_dir=opts.data_root,
            crop_size=[opts.crop_size, opts.crop_size],
            subset='train',
            scale_range=[0.75, 1.25],
            random_seed=opts.random_seed,
            subset_ratio=getattr(opts, 'subset_ratio', 1.0),
            # Albumentations Transform 사용
            custom_transform=train_transform
        )
        print(f"📊 Training dataset loaded: {len(train_dst)} samples")
        print(f"   - Subset ratio: {getattr(opts, 'subset_ratio', 1.0):.1%}")
    
    # Validation Transform 생성 (증강 없음)
    val_transform = AlbumentationsValidationTransform(
        crop_size=[opts.crop_size, opts.crop_size]
    )
    
    # Check if val directory exists
    val_dir = os.path.join(opts.data_root, "SemanticDataset_final", "image", "val")
    if os.path.exists(val_dir):
        print("📊 Loading validation dataset...")
        val_dst = DNA2025Dataset(
            root_dir=opts.data_root,
            crop_size=[opts.crop_size, opts.crop_size],
            subset='val',
            scale_range=None,
            random_seed=opts.random_seed,
            subset_ratio=1.0,  # Validation은 항상 전체 데이터 사용
            # Albumentations Validation Transform 사용
            custom_transform=val_transform
        )
        print(f"📊 Validation dataset loaded: {len(val_dst)} samples")
    else:
        print("⚠️ Validation directory not found, using train subset for validation...")
        # Use a small subset of train data for validation
        val_dst = DNA2025Dataset(
            root_dir=opts.data_root,
            crop_size=[opts.crop_size, opts.crop_size],
            subset='train',
            scale_range=None,
            random_seed=opts.random_seed,
            subset_ratio=0.1,  # Use 10% of train data for validation
            # Albumentations Validation Transform 사용
            custom_transform=val_transform
        )
        print(f"📊 Validation dataset (from train subset): {len(val_dst)} samples")

    return train_dst, val_dst


def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_warmup_scheduler(optimizer, total_epochs, warmup_epochs, warmup_start_lr, warmup_scheduler='linear'):
    """Create warmup scheduler with cosine annealing"""
    import math
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup phase
            if warmup_scheduler == 'linear':
                return warmup_start_lr + (1.0 - warmup_start_lr) * (epoch / warmup_epochs)
            elif warmup_scheduler == 'cosine':
                return warmup_start_lr + (1.0 - warmup_start_lr) * 0.5 * (1 + math.cos(math.pi * (1 - epoch / warmup_epochs)))
        else:
            # Cosine annealing phase
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_augmented_training_samples(train_loader, num_samples=2, device='cuda'):
    """Get augmented training samples for WandB visualization"""
    import matplotlib.pyplot as plt
    import random
    import time
    
    samples = []
    
    # Get the dataset from the loader
    dataset = train_loader.dataset
    
    # Use current time as seed for true randomness
    random.seed(int(time.time() * 1000) % 2**32)
    
    # Get random samples directly from dataset
    for i in range(num_samples):
        try:
            # Get random index from dataset
            random_idx = random.randint(0, len(dataset) - 1)
            
            # Get sample directly from dataset (this will apply augmentation)
            img_tensor, label_tensor = dataset[random_idx]
            
            # Move to CPU if needed
            if img_tensor.is_cuda:
                img_tensor = img_tensor.cpu()
            if label_tensor.is_cuda:
                label_tensor = label_tensor.cpu()
            
            # Denormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            img_denorm = img_denorm.permute(1, 2, 0).numpy()
            img_denorm = (img_denorm * 255).astype(np.uint8)
            
            # Create label visualization
            label_np = label_tensor.numpy()
            
            # Handle ignore index (255) by clipping to valid range
            label_np_clipped = np.clip(label_np, 0, 19)  # Clip to valid class range (0-19)
            
            # Create overlay with label
            color_map = plt.cm.tab20(np.linspace(0, 1, 20))
            colored_label = color_map[label_np_clipped]
            
            # Convert colored_label from RGBA to RGB (remove alpha channel)
            colored_label_rgb = colored_label[:, :, :3]  # Take only RGB channels
            
            # Create overlay image (image + label with transparency)
            overlay = img_denorm.copy()
            overlay = (overlay * 0.7 + colored_label_rgb * 255 * 0.3).astype(np.uint8)
            
            # Resize to reduce WandB overhead (512x512 -> 256x256)
            overlay_resized = torch.nn.functional.interpolate(
                torch.from_numpy(overlay).permute(2, 0, 1).unsqueeze(0).float(),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
            
            samples.append(overlay_resized)
            
        except Exception as e:
            print(f"⚠️ Error getting augmented sample {i+1}: {e}")
            # Create a dummy sample if error occurs
            dummy_sample = np.zeros((256, 256, 3), dtype=np.uint8)
            samples.append(dummy_sample)
    
    return samples


def progressive_training(opts):
    """Progressive training with increasing dataset ratios"""
    import ast
    
    # Parse progressive training parameters
    ratios = [float(x.strip()) for x in opts.progressive_ratios.split(',')]
    epochs_list = [int(x.strip()) for x in opts.progressive_epochs.split(',')]
    lrs = [float(x.strip()) for x in opts.progressive_lrs.split(',')]
    
    if len(ratios) != len(epochs_list) or len(ratios) != len(lrs):
        raise ValueError("progressive_ratios, progressive_epochs, and progressive_lrs must have the same length")
    
    print(f"\n{'='*80}")
    print(f"🚀 PROGRESSIVE TRAINING STARTED")
    print(f"{'='*80}")
    print(f"Stages: {len(ratios)}")
    print(f"Ratios: {ratios}")
    print(f"Epochs: {epochs_list}")
    print(f"Learning Rates: {lrs}")
    print(f"{'='*80}\n")
    
    current_checkpoint = opts.ckpt
    
    for stage, (ratio, epochs, lr) in enumerate(zip(ratios, epochs_list, lrs), 1):
        print(f"\n{'='*60}")
        print(f"🎯 STAGE {stage}/{len(ratios)}: Ratio={ratio:.1%}, Epochs={epochs}, LR={lr:.2e}")
        print(f"{'='*60}")
        
        # Update options for current stage
        stage_opts = opts
        stage_opts.subset_ratio = ratio
        stage_opts.epochs = epochs
        stage_opts.lr = lr
        stage_opts.ckpt = current_checkpoint
        stage_opts.experiment_name = f"{opts.experiment_name}_stage{stage}_ratio{ratio:.1%}"
        stage_opts.wandb_name = f"{opts.wandb_name}_stage{stage}_ratio{ratio:.1%}"
        
        # Run training for current stage
        print(f"Starting Stage {stage} training...")
        print(f"  Dataset ratio: {ratio:.1%}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr:.2e}")
        print(f"  Checkpoint: {current_checkpoint}")
        
        # Call main training function with stage options
        main_training(stage_opts)
        
        # Find the best checkpoint from this stage
        experiment_dir = f"checkpoints/{stage_opts.experiment_name}"
        if os.path.exists(experiment_dir):
            checkpoint_files = [f for f in os.listdir(experiment_dir) if f.endswith('_best_model.pth')]
            if checkpoint_files:
                current_checkpoint = os.path.join(experiment_dir, checkpoint_files[0])
                print(f"✓ Stage {stage} completed. Best checkpoint: {current_checkpoint}")
            else:
                print(f"⚠️ No best model found for stage {stage}")
        else:
            print(f"⚠️ Experiment directory not found: {experiment_dir}")
    
    print(f"\n{'='*80}")
    print(f"🎉 PROGRESSIVE TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Final checkpoint: {current_checkpoint}")
    print(f"{'='*80}\n")


def main():
    opts = get_argparser().parse_args()
    
    # Check if progressive training is enabled
    if opts.progressive_training:
        progressive_training(opts)
        return
    
    # Regular training
    main_training(opts)


def main_training(opts):
    """Main training function (extracted from original main function)"""

    # ===== DEBUG: WandB Status Check =====
    print("\n" + "="*60)
    print("🔍 WANDB STATUS")
    print("="*60)
    print(f"enable_vis: {opts.enable_vis}")
    print(f"wandb_project: {opts.wandb_project}")
    print(f"wandb_name: {opts.wandb_name}")
    print(f"wandb_tags: {opts.wandb_tags}")
    print("="*60)
    
    # Setup visualization with WandB
    vis = None
    if opts.enable_vis:
        print("✅ WandB ENABLED - Initializing...")
        # Parse tags if provided
        tags = None
        if opts.wandb_tags:
            tags = [tag.strip() for tag in opts.wandb_tags.split(',')]
        
        # Generate run name if not provided
        run_name = opts.wandb_name
        if run_name is None:
            # Albumentations 고급 증강 파라미터를 포함한 동적 run name 생성
            weather = getattr(opts, 'use_weather', False)
            noise = getattr(opts, 'use_noise', False)
            blur = getattr(opts, 'use_blur', False)
            cutout = getattr(opts, 'use_cutout', False)
            geometric = getattr(opts, 'use_geometric', False)
            color = getattr(opts, 'use_color', False)
            
            # 활성화된 증강 기법들로 run name 생성
            active_augs = []
            if weather: active_augs.append("weather")
            if noise: active_augs.append("noise")
            if blur: active_augs.append("blur")
            if cutout: active_augs.append("cutout")
            if geometric: active_augs.append("geo")
            if color: active_augs.append("color")
            
            if active_augs:
                run_name = f"alb_{'_'.join(active_augs[:3])}"  # 최대 3개까지만
            else:
                run_name = "alb_basic"
        
        vis = Visualizer(
            project=opts.wandb_project,
            name=run_name,
            config=vars(opts),
            notes=opts.wandb_notes,
            tags=tags
        )
        print(f"✅ WandB initialized - Project: {opts.wandb_project}, Run: {run_name}")
        
        # Albumentations 고급 증강 파라미터 로깅
        if hasattr(opts, 'use_weather'):
            # WandB에 증강 파라미터를 테이블로 로깅
            aug_params = {
                "Horizontal Flip Probability": f"{getattr(opts, 'horizontal_flip_p', 0.5):.1%}",
                "Brightness Limit": f"±{getattr(opts, 'brightness_limit', 0.2):.2f}",
                "Contrast Limit": f"±{getattr(opts, 'contrast_limit', 0.2):.2f}",
                "Rotation Limit": f"±{getattr(opts, 'rotation_limit', 5)}°",
                "Weather Simulation": f"{getattr(opts, 'use_weather', False)} (p={getattr(opts, 'weather_p', 0.3):.1%})",
                "Noise Addition": f"{getattr(opts, 'use_noise', False)} (p={getattr(opts, 'noise_p', 0.3):.1%})",
                "Blur Effects": f"{getattr(opts, 'use_blur', False)} (p={getattr(opts, 'blur_p', 0.3):.1%})",
                "CutOut Effects": f"{getattr(opts, 'use_cutout', False)} (p={getattr(opts, 'cutout_p', 0.3):.1%})",
                "Geometric Transform": f"{getattr(opts, 'use_geometric', False)} (p={getattr(opts, 'geometric_p', 0.3):.1%})",
                "Color Transform": f"{getattr(opts, 'use_color', False)} (p={getattr(opts, 'color_p', 0.3):.1%})"
            }
            
            vis.vis_table("Albumentations Advanced Augmentation Parameters", aug_params)
            print("📊 Albumentations augmentation parameters logged to WandB (table)")
    else:
        print("❌ WandB DISABLED - No logging will occur")
        print("   Use --enable_vis to enable WandB logging")
    print("="*60 + "\n")

    # Use GPU 0 by default
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed (only for reproducibility, not for augmentation)
    # Note: Augmentation will be truly random for variety
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # 실험 시작 전 Albumentations 고급 증강 파라미터 요약 출력
    print("\n" + "🚀" + "="*58 + "🚀")
    print("🎯 ALBUMENTATIONS ADVANCED AUGMENTATION EXPERIMENT SUMMARY")
    print("🚀" + "="*58 + "🚀")
    print(f"Learning Rate:     {opts.lr:.2e}")
    print(f"Weight Decay:      {opts.weight_decay:.2e}")
    print(f"Target Max Ratio:  {opts.target_max_ratio}")
    print(f"Optimizer:         {opts.optimizer}")
    print(f"Scheduler:         {opts.scheduler_type}")
    print(f"Warmup Epochs:     {opts.warmup_epochs}")
    print(f"Epochs:            {opts.epochs}")
    print(f"Batch Size:        {opts.batch_size}")
    print(f"Weather Simulation: {getattr(opts, 'use_weather', False)}")
    print(f"Noise Addition:    {getattr(opts, 'use_noise', False)}")
    print(f"Blur Effects:      {getattr(opts, 'use_blur', False)}")
    print(f"CutOut Effects:    {getattr(opts, 'use_cutout', False)}")
    print(f"Geometric Transform: {getattr(opts, 'use_geometric', False)}")
    print(f"Color Transform:   {getattr(opts, 'use_color', False)}")
    print("🚀" + "="*58 + "🚀\n")

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, 
        batch_size=opts.batch_size,
        shuffle=True, 
        num_workers=12,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = data.DataLoader(
        val_dst, 
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Calculate total iterations automatically
    opts.total_itrs = opts.epochs * len(train_loader)
    print(f"\nTraining for {opts.epochs} epochs, which is {opts.total_itrs} iterations.")

    # ===== Setup loss function: Combined Loss + Class Weights (Safe Version) =====
    metrics = StreamSegMetrics(opts.num_classes)
    
    print("\n" + "="*80)
    print("  LOSS FUNCTION SETUP")
    print("="*80)
    
    # Load or calculate class weights
    weights_method = 'sqrt_inv_freq'
    
    if opts.class_weights_file and os.path.isfile(opts.class_weights_file):
        # Load pre-calculated weights
        print(f"\n✓ Loading pre-calculated class weights from: {opts.class_weights_file}")
        try:
            weights_data = torch.load(opts.class_weights_file, map_location=device, weights_only=False)
            class_weights = weights_data['weights'].to(device)
            
            print(f"  Method: {weights_data.get('method', 'unknown')}")
            print(f"  Dataset: {weights_data.get('dataset', 'unknown')}")
            print(f"  Num classes: {weights_data.get('num_classes', len(class_weights))}")
            print(f"  Created: {weights_data.get('timestamp', 'unknown')}")
            print(f"  Weights range: [{class_weights.min().item():.2f}, {class_weights.max().item():.2f}]")
            print(f"  Weights ratio: {(class_weights.max() / class_weights.min()).item():.1f}x")
            print("✓ Class weights loaded successfully!")
            
        except Exception as e:
            print(f"✗ Failed to load class weights: {e}")
            print("  Calculating class weights instead...")
            class_weights = calculate_class_weights(
                dataset=train_dst,
                num_classes=opts.num_classes,
                device=device,
                method=weights_method,
                ignore_index=255
            )
    else:
        # Calculate class weights
        if opts.class_weights_file:
            print(f"  Class weights file not found: {opts.class_weights_file}")
            print("  Calculating class weights...")
        
        class_weights = calculate_class_weights(
            dataset=train_dst,
            num_classes=opts.num_classes,
            device=device,
            method=weights_method,
            ignore_index=255
        )
        
        # Save calculated weights for future use
        if not opts.skip_save_class_weights:
            # Create directory for class weights
            weights_dir = 'class_weights'
            utils.mkdir(weights_dir)
            
            # Generate filename if not specified
            if opts.class_weights_file:
                weights_save_path = opts.class_weights_file
            else:
                weights_save_path = os.path.join(
                    weights_dir, 
                    f"{opts.dataset}_{weights_method}_nc{opts.num_classes}.pth"
                )
            
            try:
                import datetime
                weights_data = {
                    'weights': class_weights.cpu(),
                    'method': weights_method,
                    'dataset': opts.dataset,
                    'num_classes': opts.num_classes,
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                torch.save(weights_data, weights_save_path)
                print(f"\n✓ Class weights saved to: {weights_save_path}")
                print(f"  You can reuse these weights with: --class_weights_file {weights_save_path}")
            except Exception as e:
                print(f"\n⚠ Failed to save class weights: {e}")
                print("  Training will continue without saving weights.")
    
    # Apply safety clipping with dynamic max based on target ratio
    original_min = class_weights.min().item()
    original_max = class_weights.max().item()
    
    # Set target max ratio (configurable for sweep optimization)
    target_max_ratio = opts.target_max_ratio
    min_weight_threshold = 0.1
    max_weight_threshold = min_weight_threshold * target_max_ratio  # 1.0
    
    class_weights = torch.clamp(class_weights, min=min_weight_threshold, max=max_weight_threshold)
    
    final_min = class_weights.min().item()
    final_max = class_weights.max().item()
    final_ratio = final_max / final_min
    
    print("-"*80)
    print("Class Weights Safety Check:")
    print(f"  Original range: [{original_min:.2f}, {original_max:.2f}]")
    print(f"  Original ratio: {original_max / original_min:.1f}x")
    print(f"  Target max ratio: {target_max_ratio:.1f}x")
    print(f"  After clipping: [{final_min:.2f}, {final_max:.2f}]")
    print(f"  Final ratio: {final_ratio:.1f}x")
    
    if original_max > max_weight_threshold:
        print(f"  ⚠ Clipped {(original_max - max_weight_threshold):.2f} from max weight for stability")
    
    if final_ratio <= 10:
        print(f"  ✓ Weight ratio is SAFE ({final_ratio:.1f}x ≤ 10x)")
    elif final_ratio <= 15:
        print(f"  ✓ Weight ratio is ACCEPTABLE ({final_ratio:.1f}x ≤ 15x)")
    else:
        print(f"  ⚠ Weight ratio is HIGH ({final_ratio:.1f}x > 15x) - may cause instability")
    
    # Print detailed class weights after clipping
    print("\n" + "="*80)
    print("  DETAILED CLASS WEIGHTS (After Clipping)")
    print("="*80)
    print(f"{'Class':<8} {'Weight':<10} {'Status':<15} {'Note'}")
    print("-"*80)
    
    # Print each class weight with status
    for i, weight in enumerate(class_weights):
        weight_val = weight.item()
        
        # Determine status
        if weight_val == min_weight_threshold:
            status = "CLIPPED_MIN"
            note = "Set to minimum"
        elif weight_val == max_weight_threshold:
            status = "CLIPPED_MAX"
            note = "Set to maximum"
        else:
            status = "ORIGINAL"
            note = "No clipping"
        
        print(f"{'C' + str(i):<8} {weight_val:<10.4f} {status:<15} {note}")
    
    print("-"*80)
    print(f"Total classes: {len(class_weights)}")
    print(f"Classes clipped to min: {sum(1 for w in class_weights if w.item() == min_weight_threshold)}")
    print(f"Classes clipped to max: {sum(1 for w in class_weights if w.item() == max_weight_threshold)}")
    print(f"Classes unchanged: {sum(1 for w in class_weights if w.item() not in [min_weight_threshold, max_weight_threshold])}")
    print("="*80)
    
    # Loss function: Weighted Cross Entropy with class weights
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=255
    )
    
    print("\n✓ Loss Function Configuration:")
    print(f"  Type: Weighted Cross Entropy")
    print(f"  Class Weights: Applied (sqrt_inv_freq method)")
    print(f"  Weight Ratio: {final_ratio:.1f}x (clipped to max {target_max_ratio:.0f}x)")
    print(f"  This configuration balances class importance with standard cross entropy")

    print("="*80 + "\n")

    # Set up model
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, 
        output_stride=opts.output_stride
    )
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Setup checkpoint directory with fixed experiment name
    experiment_name = opts.experiment_name or 'sweep_experiments'
    checkpoint_dir = os.path.join('checkpoints', experiment_name)
    print(f"\n✓ Experiment Name: '{experiment_name}'")
    print(f"  Checkpoints will be saved to: {checkpoint_dir}/")
    
    # Create run-specific suffix for filenames
    run_suffix = f"sweep_lr{opts.lr:.2e}_weight{opts.target_max_ratio:.1f}"
    print(f"  Run suffix: '{run_suffix}'")
    
    utils.mkdir(checkpoint_dir)
    pretrained_loaded = False
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        if opts.continue_training:
            # Continue training - will be handled later
            pass
        else:
            # Load pretrained model
            model, _ = load_pretrained_model(
                model, opts.ckpt, 
                num_classes_old=opts.pretrained_num_classes, 
                num_classes_new=opts.num_classes
            )
            pretrained_loaded = True
    else:
        print("[!] Training from scratch")
        continue_from_checkpoint = False

    # Backbone freezing setup
    if opts.freeze_backbone:
        # STAGE 1: Freeze backbone
        print("--- STAGE 1 SETUP: Training classifier only ---")
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        trainable_params_stage1 = filter(lambda p: p.requires_grad, model.parameters())
        
        # Using full learning rate with Weighted CE Loss
        print(f"Stage 1 Learning Rate: {opts.lr:.2e}")
        
        # Optimizer selection for Stage 1
        if opts.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                params=trainable_params_stage1, 
                lr=opts.lr,
                momentum=0.9, 
                weight_decay=opts.weight_decay
            )
        elif opts.optimizer == 'sgd_nesterov':
            optimizer = torch.optim.SGD(
                params=trainable_params_stage1, 
                lr=opts.lr,
                momentum=0.9, 
                weight_decay=opts.weight_decay,
                nesterov=True
            )
        elif opts.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                params=trainable_params_stage1,
                lr=opts.lr * 0.1,  # Adam 계열은 LR을 낮춤
                weight_decay=opts.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif opts.optimizer == 'radam':
            optimizer = torch.optim.RAdam(
                params=trainable_params_stage1,
                lr=opts.lr * 0.1,
                weight_decay=opts.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
    else:
        # Train entire network from the beginning
        print("--- TRAINING ENTIRE NETWORK: No backbone freezing ---")
        print(f"Learning Rate: {opts.lr:.2e}")
        
        # Optimizer selection for full network training
        if opts.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                params=model.parameters(), 
                lr=opts.lr,
                momentum=0.9, 
                weight_decay=opts.weight_decay
            )
        elif opts.optimizer == 'sgd_nesterov':
            optimizer = torch.optim.SGD(
                params=model.parameters(), 
                lr=opts.lr,
                momentum=0.9, 
                weight_decay=opts.weight_decay,
                nesterov=True
            )
        elif opts.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=opts.lr * 0.1,  # Adam 계열은 LR을 낮춤
                weight_decay=opts.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif opts.optimizer == 'radam':
            optimizer = torch.optim.RAdam(
                params=model.parameters(),
                lr=opts.lr * 0.1,
                weight_decay=opts.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )

    # Setup scheduler based on scheduler_type and warmup
    if opts.warmup_epochs > 0:
        # Warmup scheduler
        warmup_start_lr = opts.warmup_start_lr if opts.warmup_start_lr is not None else 0.1
        scheduler = get_warmup_scheduler(
            optimizer, 
            opts.epochs, 
            opts.warmup_epochs, 
            warmup_start_lr,
            opts.warmup_scheduler
        )
        print(f"🔥 Warmup Scheduler: {opts.warmup_epochs} epochs, start_lr={warmup_start_lr:.2e}, type={opts.warmup_scheduler}")
    elif opts.scheduler_type == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-8
        )
    else:
        # Use PolyLR scheduler for other types
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)

    model = nn.DataParallel(model)
    model.to(device)

    # Training setup
    best_score = 0.0
    cur_itrs = 0
    start_epoch = 1
    training_start_time = time.time()
    end_epoch = opts.epochs  # Define end_epoch variable
    continue_from_checkpoint = False  # Initialize continue_from_checkpoint
    
    # Continue training from checkpoint if specified
    if opts.continue_training and opts.ckpt is not None and os.path.isfile(opts.ckpt):
        print(f"\n=== CONTINUE TRAINING ===")
        print(f"Resuming from checkpoint: {opts.ckpt}")
        start_epoch, cur_itrs, best_score = load_checkpoint(
            opts.ckpt, model, optimizer, scheduler, device
        )
        print(f"Training will resume from epoch {start_epoch}")
        print(f"Current iteration count: {cur_itrs}")
        print(f"Best score so far: {best_score:.4f}")
        
        # Set continue_from_checkpoint flag
        continue_from_checkpoint = True
        
        # Check if training is already complete
        if start_epoch > opts.epochs:
            print(f"\n⚠️  WARNING: Training already completed!")
            print(f"   Current epoch: {start_epoch}")
            print(f"   Target epochs: {opts.epochs}")
            print(f"   No additional training needed.")
            print("="*50 + "\n")
            return
        elif start_epoch == opts.epochs:
            print(f"\n⚠️  WARNING: Training is at the final epoch!")
            print(f"   Current epoch: {start_epoch}")
            print(f"   Target epochs: {opts.epochs}")
            print(f"   Only validation will be performed.")
            print("="*50 + "\n")
        else:
            remaining_epochs = opts.epochs - start_epoch + 1
            print(f"   Remaining epochs to train: {remaining_epochs}")
            print("="*50 + "\n")
    else:
        # Set continue_from_checkpoint flag to False for new training
        continue_from_checkpoint = False
        
        # Determine training type
        if pretrained_loaded:
            print(f"\n=== STARTING NEW TRAINING ===")
            print("Training with pretrained weights")
            print("="*50 + "\n")
        else:
            print(f"\n=== STARTING NEW TRAINING ===")
            print("Training from scratch")
            print("="*50 + "\n")
    
    # Track metrics for displaying changes
    prev_metrics = {
        'loss': None,
        'Mean IoU': None,
        'Overall Acc': None,
        'Mean Acc': None,
        'Class IoU': None
    }
    
    # ===== Early Stopping Setup =====
    early_stopping = None
    
    # Hyperband 사용 시 early stopping 비활성화 (patience가 100 이상이면)
    if opts.early_stop and opts.early_stop_patience < 50:
        early_stopping = EarlyStopping(
            patience=opts.early_stop_patience,
            min_delta=opts.early_stop_min_delta,
            mode='max',  # Mean IoU, Accuracy 등은 maximize
            verbose=True
        )
        print(f"\n✓ Early Stopping enabled:")
        print(f"  Patience: {opts.early_stop_patience} epochs")
        print(f"  Min Delta: {opts.early_stop_min_delta}")
        print(f"  Monitoring: {opts.early_stop_metric}")
        print(f"  Will activate at: Epoch {opts.unfreeze_epoch} (Stage 2 - Fine-tuning)")
    elif opts.early_stop_patience >= 50:
        print(f"\n⚠️  Early Stopping disabled for Hyperband optimization:")
        print(f"  Patience: {opts.early_stop_patience} epochs (too high - using Hyperband instead)")
        print(f"  Hyperband will handle early termination at min_iter: 20")
    else:
        print(f"\n❌ Early Stopping disabled")
    # ================================
    
    vis_sample_id = np.random.randint(
        0, len(val_loader), opts.vis_num_samples, np.int32
    ) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ===== TRAINING LOOP =====
    for epoch in range(start_epoch, opts.epochs + 1):
        epoch_start_time = time.time()
        
        # STAGE 2: Unfreeze backbone (only if freeze_backbone is enabled)
        if opts.freeze_backbone and epoch == opts.unfreeze_epoch:
            print(f"\n--- STAGE 2: Unfreezing backbone at Epoch {epoch} ---")
            
            for param in model.module.backbone.parameters():
                param.requires_grad = True
            
            print("Re-creating optimizer with differential learning rates...")
            # Differential learning rates for fine-tuning converged model
            backbone_lr = opts.lr / 20      # 1/20 of base LR for stable backbone fine-tuning
            classifier_lr = opts.lr / 5     # 1/5 of base LR for classifier
            
            optimizer = torch.optim.SGD([
                {'params': model.module.backbone.parameters(), 'lr': backbone_lr},
                {'params': model.module.classifier.parameters(), 'lr': classifier_lr}
            ], momentum=0.9, weight_decay=opts.weight_decay)
            
            print(f"Backbone LR: {backbone_lr:.2e}")
            print(f"Classifier LR: {classifier_lr:.2e}")

            # Setup scheduler for Stage 2 based on scheduler_type
            if opts.scheduler_type == 'reduce':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-8
                )
            else:
                # Use PolyLR scheduler for other types
                remaining_itrs = max(1, opts.total_itrs - cur_itrs)
                scheduler = utils.PolyLR(optimizer, remaining_itrs, power=0.9)
            
            # Reset early stopping to start fresh from Stage 2
            if early_stopping is not None:
                early_stopping.reset()
                print(f"[Early Stopping] Reset for Stage 2 - will set baseline after this epoch's validation")
        
        # Handle continue training from STAGE 2 (only if freeze_backbone is enabled)
        elif opts.freeze_backbone and continue_from_checkpoint and epoch > opts.unfreeze_epoch:
            print(f"\n--- Continue training in STAGE 2 (Epoch {epoch}) ---")
            # Ensure backbone is unfrozen for continue training in Stage 2
            for param in model.module.backbone.parameters():
                param.requires_grad = True
            print("Backbone already unfrozen for Stage 2 continue training")
        
        # Print current learning rate
        if len(optimizer.param_groups) == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{end_epoch} - Current Learning Rate: {current_lr:.6f}")
            print(f"{'='*80}")
        else:
            backbone_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{end_epoch} - Learning Rates:")
            print(f"  Backbone:   {backbone_lr:.6f}")
            print(f"  Classifier: {classifier_lr:.6f}")
            print(f"{'='*80}")

        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        interval_loss = 0.0
        
        if opts.freeze_backbone:
            stage_str = '2 (Fine-tuning)' if epoch >= opts.unfreeze_epoch else '1 (Classifier only)'
        else:
            stage_str = 'Full Network Training'
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch} [Stage {stage_str}]")
        
        for images, labels in progress_bar:
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Call scheduler step (only for non-ReduceLROnPlateau schedulers)
            if opts.scheduler_type != 'reduce':
                try:
                    scheduler.step()
                except (TypeError, ValueError) as e:
                    if "complex" in str(e).lower() or "not supported" in str(e).lower():
                        # Skip scheduler step if complex number error occurs
                        # This happens when last_epoch > max_iters in PolyLR
                        pass
                    else:
                        raise e

            np_loss = loss.detach().cpu().numpy()
            epoch_loss += np_loss
            num_batches += 1
            interval_loss += np_loss

            if cur_itrs % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                progress_bar.set_postfix(loss=f"{interval_loss:.4f}")
                interval_loss = 0.0

        avg_epoch_loss = epoch_loss / num_batches
        
        print(f"\nEpoch {epoch}/{end_epoch} [{stage_str}] completed:")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        if len(optimizer.param_groups) == 1:
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"  Backbone LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Classifier LR: {optimizer.param_groups[1]['lr']:.6f}")

        # Validation
        print(f"Validation for Epoch {epoch}...")
        model.eval()

        save_images_this_epoch = False  # 이미지는 WandB에서만 확인
        
        val_score, ret_samples, confusion_mat = validate(
            opts=opts, 
            model=model, 
            loader=val_loader, 
            device=device, 
            metrics=metrics,
            ret_samples_ids=vis_sample_id,
            epoch=epoch if save_images_this_epoch else None,
            save_sample_images=save_images_this_epoch,
            denorm=denorm
        )

        # Call scheduler with validation metric (only for ReduceLROnPlateau)
        if opts.scheduler_type == 'reduce':
            scheduler.step(val_score['Mean IoU'])
        
        # Display validation results with changes from previous epoch
        print(f"\n{'='*80}")
        print(f"Validation Results (Epoch {epoch}):")
        print(f"{'='*80}")
        
        # Helper function to format change
        def format_change(current, previous, larger_is_better=True):
            if previous is None:
                return ""
            delta = current - previous
            if abs(delta) < 0.0001:
                return "  (━ ±0.0000)"
            arrow = "↑" if delta > 0 else "↓"
            color_good = (delta > 0) if larger_is_better else (delta < 0)
            sign = "+" if delta > 0 else ""
            return f"  ({arrow} {sign}{delta:.4f})"
        
        # Display metrics with changes
        current_loss = avg_epoch_loss
        current_miou = val_score['Mean IoU']
        current_oacc = val_score['Overall Acc']
        current_macc = val_score['Mean Acc']
        
        print(f"  Training Loss:   {current_loss:.6f}{format_change(current_loss, prev_metrics['loss'], larger_is_better=False)}")
        print(f"  Mean IoU:        {current_miou:.4f}{format_change(current_miou, prev_metrics['Mean IoU'])}")
        print(f"  Overall Acc:     {current_oacc:.4f}{format_change(current_oacc, prev_metrics['Overall Acc'])}")
        print(f"  Mean Acc:        {current_macc:.4f}{format_change(current_macc, prev_metrics['Mean Acc'])}")
        print(f"{'='*80}")
        
        # Show detailed class IoU with comparison
        print(f"\nDetailed Metrics:")
        print(metrics.to_str(val_score))
        
        # Show Class IoU comparison if previous epoch exists
        if prev_metrics['Class IoU'] is not None:
            print(f"\nClass IoU Changes from Previous Epoch:")
            print(f"{'Class':<20} {'Current':<10} {'Previous':<10} {'Change':<10}")
            print(f"{'-'*50}")
            
            current_class_ious = val_score['Class IoU']
            prev_class_ious = prev_metrics['Class IoU']
            
            for class_id in range(len(current_class_ious)):
                current_iou = current_class_ious[class_id]
                prev_iou = prev_class_ious[class_id]
                delta = current_iou - prev_iou
                
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "━"
                sign = "+" if delta > 0 else ""
                change_str = f"{arrow} {sign}{delta:.4f}" if abs(delta) >= 0.0001 else "━ ±0.0000"
                
                print(f"Class {class_id:<15} {current_iou:<10.4f} {prev_iou:<10.4f} {change_str}")
        
        # Update previous metrics for next epoch
        prev_metrics['loss'] = current_loss
        prev_metrics['Mean IoU'] = current_miou
        prev_metrics['Overall Acc'] = current_oacc
        prev_metrics['Mean Acc'] = current_macc
        prev_metrics['Class IoU'] = val_score['Class IoU'].copy()

        # ===== Early Stopping Check =====
        # Apply early stopping only if not using Hyperband (patience < 50)
        if early_stopping is not None and (not opts.freeze_backbone or epoch >= opts.unfreeze_epoch):
            monitor_score = val_score[opts.early_stop_metric]
            
            # Set baseline for early stopping
            if (opts.freeze_backbone and epoch == opts.unfreeze_epoch) or (not opts.freeze_backbone and epoch == 1):
                early_stopping.set_baseline(monitor_score)
                if opts.freeze_backbone:
                    print(f"[Early Stopping] Stage 2 baseline set: {monitor_score:.4f}")
                else:
                    print(f"[Early Stopping] Baseline set at epoch 1: {monitor_score:.4f}")
                print(f"[Early Stopping] Will monitor improvements from this point")
            else:
                # Check for improvements
                should_stop = early_stopping(monitor_score, epoch)
                
                if should_stop:
                    print(f"\n{'='*80}")
                    print(f"Training stopped early at epoch {epoch}/{end_epoch}")
                    print(f"Best {opts.early_stop_metric}: {early_stopping.best_score:.4f}")
                    print(f"  (Set at Stage 2 start: epoch {opts.unfreeze_epoch})")
                    print(f"{'='*80}\n")
                    
                    # Ensure final WandB logging before stopping
                    if vis is not None:
                        current_lr = optimizer.param_groups[0]['lr']
                        
                        # Calculate gradient norm for monitoring
                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        # Log final metrics with early stopping flag
                        try:
                            wandb.log({
                                'epoch': epoch,
                                'Training Loss': avg_epoch_loss,
                                'Learning Rate': current_lr,
                                '[Val] Overall Acc': val_score['Overall Acc'],
                                '[Val] Mean Acc': val_score['Mean Acc'],
                                '[Val] Mean IoU': val_score['Mean IoU'],
                                'Gradient Norm': total_norm,
                                'Early Stopped': True,
                                'Early Stop Reason': f'No improvement for {early_stopping.patience} epochs'
                            }, step=epoch)
                            print(f"✅ Final WandB log successful for early stopped epoch {epoch}")
                        except Exception as e:
                            print(f"❌ Final WandB log failed: {e}")
                        
                        # Log class IoU table
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'], step=epoch)
                    
                    # Save final checkpoint before stopping
                    early_stop_path = os.path.join(checkpoint_dir, f'{run_suffix}_early_stopped_epoch_{epoch:03d}.pth')
                    save_checkpoint(
                        early_stop_path,
                        epoch, cur_itrs, model, optimizer, scheduler, best_score
                    )
                    print(f"✓ Early stopped checkpoint saved → {os.path.basename(early_stop_path)}")
                    
                    break  # Exit training loop
        elif early_stopping is not None and epoch < opts.unfreeze_epoch:
            # In Stage 1, just print that early stopping is not active yet
            if epoch == 1:
                print(f"[Early Stopping] Waiting for Stage 2 (epoch {opts.unfreeze_epoch}) to start monitoring...")
        # ================================

        # WandB updates (Enhanced monitoring with visual validation)
        if vis is not None:
            # Get current learning rate (use first param group for logging)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate gradient norm for monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Enhanced WandB logging with visual validation
            try:
                # Get augmented training samples for visualization (reduced overhead)
                augmented_samples = get_augmented_training_samples(train_loader, num_samples=2, device=device)
                
                # Create comprehensive logging dictionary
                log_dict = {
                    'epoch': epoch,
                    'Training Loss': avg_epoch_loss,
                    'Learning Rate': current_lr,
                    'Gradient Norm': total_norm,
                    'Augmented Train Sample 1': wandb.Image(augmented_samples[0], caption=f'Epoch {epoch} - Augmented Sample 1'),
                    'Augmented Train Sample 2': wandb.Image(augmented_samples[1], caption=f'Epoch {epoch} - Augmented Sample 2'),
                }
                
                # Add validation metrics if available
                if 'val_score' in locals() and val_score is not None:
                    log_dict.update({
                        '[Val] Overall Acc': val_score['Overall Acc'],
                        '[Val] Mean Acc': val_score['Mean Acc'],
                        '[Val] Mean IoU': val_score['Mean IoU'],
                    })
                    
                    # Log class IoU table
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'], step=epoch)
                    
                    # Log confusion matrix if available
                    if 'confusion_mat' in locals() and confusion_mat is not None:
                        # Create confusion matrix heatmap
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(confusion_mat, cmap='Blues', interpolation='nearest')
                        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
                        ax.set_xlabel('Predicted Label')
                        ax.set_ylabel('True Label')
                        plt.colorbar(im, ax=ax)
                        plt.tight_layout()
                        
                        log_dict['Confusion Matrix'] = wandb.Image(fig)
                        plt.close(fig)
                else:
                    # No validation data - focus on training metrics
                    log_dict.update({
                        'Training Status': 'No Validation Data',
                        'Recovery Stage': f'Stage {stage}' if 'stage' in locals() else 'Unknown'
                    })
                
                # Log all metrics in one batch
                wandb.log(log_dict, step=epoch)
                print(f"✅ WandB log successful for epoch {epoch} (enhanced monitoring)")
                
            except Exception as e:
                print(f"❌ WandB log failed: {e}")
                # Fallback: minimal logging
                try:
                    wandb.log({
                        'epoch': epoch,
                        'Training Loss': avg_epoch_loss,
                        'Learning Rate': current_lr,
                        'Gradient Norm': total_norm,
                    }, step=epoch)
                    print(f"✅ WandB fallback log successful for epoch {epoch}")
                except Exception as e2:
                    print(f"❌ WandB fallback log also failed: {e2}")
            
            # Log confusion matrix as heatmap
            # Normalize confusion matrix by row (true labels) for better visualization
            confusion_mat_normalized = confusion_mat / (confusion_mat.sum(axis=1, keepdims=True) + 1e-10)
            
            # Create class labels
            class_labels = [f"C{i}" for i in range(opts.num_classes)]
            
            # Create confusion matrix heatmap with matplotlib
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(confusion_mat_normalized, cmap='Blues', interpolation='nearest')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Normalized Frequency', rotation=270, labelpad=20)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(opts.num_classes))
            ax.set_yticks(np.arange(opts.num_classes))
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)
            
            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations (only if not too many classes)
            if opts.num_classes <= 15:
                for i in range(opts.num_classes):
                    for j in range(opts.num_classes):
                        text = ax.text(j, i, f'{confusion_mat_normalized[i, j]:.2f}',
                                     ha="center", va="center",
                                     color="white" if confusion_mat_normalized[i, j] > 0.5 else "black",
                                     fontsize=max(6, 10 - opts.num_classes // 2))
            
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix - Epoch {epoch} (Row-Normalized)', fontsize=14)
            plt.tight_layout()
            
            # Log to WandB
            wandb.log({"[Val] Confusion Matrix": wandb.Image(fig)}, step=epoch)
            plt.close(fig)

            # Log validation images (if available)
            if ret_samples and len(ret_samples) > 0:
                samples_to_show = ret_samples[:4]
                
                # Prepare all images first
                wandb_images = {}
                for k, (img, target, lbl) in enumerate(samples_to_show):
                    img = (denorm(img) * 255).astype(np.uint8)
                    
                    # Use decode_target if available, otherwise use grayscale fallback
                    if hasattr(train_dst, 'decode_target'):
                        target_decoded = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl_decoded = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    elif hasattr(val_dst, 'decode_target'):
                        target_decoded = val_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl_decoded = val_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    else:
                        target_decoded = np.stack([target, target, target], axis=0).astype(np.uint8)
                        lbl_decoded = np.stack([lbl, lbl, lbl], axis=0).astype(np.uint8)
                    
                    concat_img = np.concatenate((img, target_decoded, lbl_decoded), axis=2)
                    
                    # Convert CHW to HWC for WandB
                    concat_img_hwc = concat_img.transpose(1, 2, 0)
                    wandb_images[f'Validation Sample {k}'] = wandb.Image(concat_img_hwc, caption=f'Sample {k}')
                
                # Log all images at once
                wandb.log(wandb_images, step=epoch)
        else:
            print(f"⚠️ WandB disabled - No logging for epoch {epoch}")

        # Save checkpoints
        current_score = val_score['Mean IoU']
        
        # 1. Save best model separately (with run suffix)
        if current_score > best_score:
            best_score = current_score
            best_model_path = os.path.join(checkpoint_dir, f'{run_suffix}_best_model.pth')
            save_checkpoint(
                best_model_path,
                epoch, cur_itrs, model, optimizer, scheduler, best_score
            )
            print(f"✓ New best model saved! (Mean IoU: {best_score:.4f}) → {best_model_path}")
            
            # Best score achieved - 이미지는 WandB에서 확인 가능
        
        # 2. Save checkpoint for every epoch (with run suffix and epoch number)
        epoch_checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'{run_suffix}_epoch_{epoch:03d}_miou_{current_score:.4f}.pth'
        )
        save_checkpoint(
            epoch_checkpoint_path,
            epoch, cur_itrs, model, optimizer, scheduler, best_score
        )
        print(f"✓ Epoch {epoch} checkpoint saved → {os.path.basename(epoch_checkpoint_path)}")
        
        # 3. Save latest model (with run suffix, overwritten each epoch)
        latest_path = os.path.join(checkpoint_dir, f'{run_suffix}_latest_model.pth')
        save_checkpoint(
            latest_path,
            epoch, cur_itrs, model, optimizer, scheduler, best_score
        )
        
        # Time tracking
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = end_epoch - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        print(f"Epoch {epoch} finished. Best Mean IoU so far: {best_score:.4f}")
        print(f"  Epoch time: {format_time(epoch_time)} | Total elapsed: {format_time(total_elapsed)}")
        print(f"  Estimated remaining: {format_time(estimated_remaining)} | "
              f"ETA: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_remaining))}")
        
        # Early stopping status
        if early_stopping is not None:
            if opts.freeze_backbone:
                if epoch > opts.unfreeze_epoch and not early_stopping.early_stop:
                    print(f"  Early stop counter: {early_stopping.counter}/{early_stopping.patience}")
                elif epoch == opts.unfreeze_epoch:
                    print(f"  Early stopping: Baseline set at {early_stopping.best_score:.4f} (monitoring starts next epoch)")
                elif epoch < opts.unfreeze_epoch:
                    print(f"  Early stopping: Inactive (Stage 1 - will activate at epoch {opts.unfreeze_epoch})")
            else:
                if epoch > 1 and not early_stopping.early_stop:
                    print(f"  Early stop counter: {early_stopping.counter}/{early_stopping.patience}")
                elif epoch == 1:
                    print(f"  Early stopping: Baseline set at {early_stopping.best_score:.4f} (monitoring starts next epoch)")
        
        print()

    # Training finished message
    total_training_time = time.time() - training_start_time
    print("="*80)
    if early_stopping is not None and early_stopping.early_stop:
        print(f"Training finished early at epoch {epoch}/{end_epoch}")
    else:
        print(f"Training finished successfully - completed all {end_epoch} epochs")
    print(f"Total training time: {format_time(total_training_time)}")
    print(f"Best Mean IoU: {best_score:.4f}")
    print("="*80)
    
    # Finish WandB run
    if vis is not None:
        print("\nFinalizing WandB logging...")
        
        vis.finish()
        print("✓ WandB run completed")


if __name__ == "__main__":
    main()

# 기본 증강 최종 훈련
# python train_albumentations_augmentation.py \
#     --data_root ./datasets/data \
#     --ckpt csweep_CE_best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --experiment_name "PC1_Final_Basic_Augmentation" \
#     --epochs 200 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 1e-5 \
#     --weight_decay 1e-5 \
#     --target_max_ratio 6.5 \
#     --optimizer sgd_nesterov \
#     --scheduler_type cosine \
#     --warmup_epochs 5 \
#     --warmup_start_lr 1e-6 \
#     --warmup_scheduler linear \
#     --crop_size 1024 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "PC1_Basic_Aug_SGD_Nesterov_Warmup" \
#     --wandb_notes "PC1: Basic augmentation with SGD+Nesterov and 5-epoch warmup" \
#     --wandb_tags "final_training,pc1,basic_aug,sgd_nesterov,warmup" \
#     --early_stop False \
#     --horizontal_flip_p 0.5 \
#     --brightness_limit 0.2 \
#     --contrast_limit 0.3 \
#     --rotation_limit 7 \
#     --subset_ratio 1.0

# 기본 + 고급 증강 최종 훈련
# python train_albumentations_augmentation.py \
#     --data_root ./datasets/data \
#     --ckpt sweep_CE_best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --experiment_name "PC2_Final_Advanced_Augmentation" \
#     --epochs 200 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 1e-4 \
#     --weight_decay 1e-5 \
#     --target_max_ratio 6.5 \
#     --optimizer adamw \
#     --warmup_epochs 8 \
#     --warmup_start_lr 1e-5 \
#     --warmup_scheduler cosine \
#     --crop_size 1024 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "PC2_Advanced_Aug_AdamW_Warmup" \
#     --wandb_notes "PC2: Basic + Advanced augmentation with AdamW and 8-epoch cosine warmup" \
#     --wandb_tags "final_training,pc2,advanced_aug,adamw,warmup,all_augmentations" \
#     --early_stop False \
#     --horizontal_flip_p 0.5 \
#     --brightness_limit 0.2 \
#     --contrast_limit 0.3 \
#     --rotation_limit 7 \
#     --use_weather True \
#     --weather_p 0.25 \
#     --use_noise True \
#     --noise_p 0.3 \
#     --use_blur True \
#     --blur_p 0.35 \
#     --use_cutout True \
#     --cutout_p 0.2 \
#     --use_geometric True \
#     --geometric_p 0.4 \
#     --use_color True \
#     --color_p 0.32 \
#     --subset_ratio 1.0

# 하이브리드
# python train_albumentations_augmentation.py \
#     --data_root ./datasets/data \
#     --ckpt sweep_CE_best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --experiment_name "PC3_Final_Hybrid_Augmentation" \
#     --epochs 200 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 5e-5 \
#     --weight_decay 1e-5 \
#     --target_max_ratio 6.5 \
#     --optimizer sgd_nesterov \
#     --warmup_epochs 6 \
#     --warmup_start_lr 5e-6 \
#     --warmup_scheduler linear \
#     --crop_size 1024 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "PC3_Hybrid_Aug_SGD_Nesterov_Warmup" \
#     --wandb_notes "PC3: Hybrid augmentation (Basic + Weather/Blur/Color) with SGD+Nesterov and 6-epoch warmup" \
#     --wandb_tags "final_training,pc3,hybrid_aug,sgd_nesterov,warmup,selective_advanced" \
#     --early_stop False \
#     --horizontal_flip_p 0.5 \
#     --brightness_limit 0.2 \
#     --contrast_limit 0.3 \
#     --rotation_limit 7 \
#     --use_weather True \
#     --weather_p 0.25 \
#     --use_noise False \
#     --use_blur True \
#     --blur_p 0.35 \
#     --use_cutout False \
#     --use_geometric False \
#     --use_color True \
#     --color_p 0.32 \
#     --subset_ratio 1.0

# 기본 + 고급 증강 최종 훈련 2
# python train_albumentations_augmentation.py \
#     --data_root ./datasets/data \
#     --ckpt sweep_CE_best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --experiment_name "PC2_Final_Advanced_Augmentation_2" \
#     --epochs 200 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 2e-4 \
#     --weight_decay 5e-4 \
#     --target_max_ratio 6.5 \
#     --optimizer adamw \
#     --warmup_epochs 5 \
#     --warmup_start_lr 1e-5 \
#     --warmup_scheduler cosine \
#     --crop_size 1024 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "PC2_Advanced_Aug_AdamW_Warmup_2" \
#     --wandb_notes "PC2: Basic + Advanced augmentation with AdamW and 8-epoch cosine warmup" \
#     --wandb_tags "final_training,pc2,advanced_aug,adamw,warmup,all_augmentations" \
#     --early_stop False \
#     --horizontal_flip_p 0.5 \
#     --brightness_limit 0.2 \
#     --contrast_limit 0.3 \
#     --rotation_limit 7 \
#     --use_weather True \
#     --weather_p 0.2 \
#     --use_noise True \
#     --noise_p 0.25 \
#     --use_cutout True \
#     --cutout_p 0.15 \
#     --use_geometric True \
#     --geometric_p 0.3 \
#     --use_color True \
#     --color_p 0.25 \
#     --subset_ratio 1.0

# 기본 + 고급 증강 최종 훈련 3
# python train_albumentations_augmentation.py \
#     --data_root ./datasets/data \
#     --ckpt sweep_CE_best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --experiment_name "PC2_Final_Advanced_Augmentation_3" \
#     --epochs 200 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 5e-4 \
#     --weight_decay 1e-3 \
#     --target_max_ratio 6.5 \
#     --optimizer adamw \
#     --warmup_epochs 10 \
#     --warmup_start_lr 1e-6 \
#     --warmup_scheduler cosine \
#     --crop_size 1024 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "PC1_Advanced_Aug_AdamW_Warmup_3(Aggressive)" \
#     --wandb_notes "PC1: Basic + Advanced augmentation with AdamW and 8-epoch cosine warmup" \
#     --wandb_tags "final_training,pc1,advanced_aug,adamw,warmup,all_augmentations" \
#     --early_stop False \
#     --horizontal_flip_p 0.5 \
#     --brightness_limit 0.25 \
#     --contrast_limit 0.35 \
#     --rotation_limit 7 \
#     --use_weather True \
#     --weather_p 0.35 \
#     --use_noise True \
#     --noise_p 0.4 \
#     --use_blur True \
#     --blur_p 0.4 \
#     --use_cutout True \
#     --cutout_p 0.3 \
#     --use_geometric True \
#     --geometric_p 0.5 \
#     --use_color True \
#     --color_p 0.4 \
#     --subset_ratio 1.0