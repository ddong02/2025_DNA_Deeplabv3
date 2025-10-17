#!/usr/bin/env python3
"""
Overfitting Recovery Training Script
과적합된 소수 클래스 집중 모델을 전체 데이터셋으로 복구하는 스크립트

This script implements a recovery strategy for overfitted models:
1. Very low learning rate to gently adjust overfitted weights
2. Progressive data inclusion to prevent sudden distribution shift
3. Enhanced regularization to prevent further overfitting
4. Combined train+val dataset for better generalization
"""

import os
import sys
import subprocess
import argparse
import torch
import datetime
from pathlib import Path

# def recalculate_class_weights(args):
#     """Recalculate class weights for combined dataset"""
#     print(f"\n{'='*80}")
#     print(f"🔄 RECALCULATING CLASS WEIGHTS FOR COMBINED DATASET")
#     print(f"{'='*80}")
#     print(f"Method: {args.class_weights_method}")
#     print(f"Dataset: Combined train+val")
#     print(f"{'='*80}\n")
#     
#     try:
#         # Import required modules
#         sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#         from my_utils.dna2025_dataset_combined import DNA2025CombinedDataset
#         from my_utils.calculate_class_weights import calculate_class_weights
#         
#         # Create combined dataset for weight calculation
#         print("📊 Creating combined dataset for class weight calculation...")
#         combined_dataset = DNA2025CombinedDataset(
#             root_dir=args.data_root,
#             crop_size=1024,
#             subset='train',
#             scale_range=[0.75, 1.25],
#             random_seed=1,
#             subset_ratio=1.0,  # Use full combined dataset
#             combine_train_val=True  # Combine train and val data
#         )
#         
#         print(f"✅ Combined dataset created: {len(combined_dataset)} samples")
#         
#         # Calculate new class weights
#         print(f"🧮 Calculating class weights using {args.class_weights_method} method...")
#         new_class_weights = calculate_class_weights(
#             dataset=combined_dataset,
#             num_classes=19,  # DNA2025 dataset has 19 classes
#             device='cpu',
#             method=args.class_weights_method,
#             ignore_index=255
#         )
#         
#         # Create new weights file path
#         weights_dir = 'class_weights'
#         os.makedirs(weights_dir, exist_ok=True)
#         
#         new_weights_file = os.path.join(
#             weights_dir, 
#             f"combined_dataset_{args.class_weights_method}_nc19_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
#         )
#         
#         # Save new weights with metadata
#         weights_data = {
#             'weights': new_class_weights.cpu(),
#             'method': args.class_weights_method,
#             'dataset': 'combined_train_val',
#             'num_classes': 19,
#             'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'total_samples': len(combined_dataset),
#             'description': 'Recalculated for overfitting recovery with combined dataset'
#         }
#         
#         torch.save(weights_data, new_weights_file)
#         
#         print(f"✅ New class weights saved to: {new_weights_file}")
#         print(f"   Method: {args.class_weights_method}")
#         print(f"   Dataset: Combined train+val ({len(combined_dataset)} samples)")
#         print(f"   Weight range: [{new_class_weights.min().item():.4f}, {new_class_weights.max().item():.4f}]")
#         print(f"   Weight ratio: {new_class_weights.max().item() / new_class_weights.min().item():.2f}x")
#         
#         return new_weights_file
#         
#     except Exception as e:
#         print(f"❌ Failed to recalculate class weights: {e}")
#         print(f"   Using original class weights file: {args.class_weights_file}")
#         return args.class_weights_file

def run_recovery_training(args):
    """Run overfitting recovery training with combined dataset"""
    print(f"\n{'='*80}")
    print(f"🔄 OVERFITTING RECOVERY TRAINING STARTED")
    print(f"{'='*80}")
    print(f"Base experiment: {args.experiment_name}")
    print(f"Recovery strategy: {args.recovery_strategy}")
    print(f"Initial LR: {args.initial_lr:.2e} (very low for recovery)")
    print(f"Final LR: {args.final_lr:.2e}")
    print(f"Stages: {len(args.ratios)}")
    print(f"Ratios: {args.ratios}")
    print(f"Epochs: {args.epochs_list}")
    print(f"Using existing class weights: class_weights/class_weight_old_data.pth")
    print(f"{'='*80}\n")
    
    # Use existing class weights file
    class_weights_file = "class_weights/class_weight_old_data.pth"
    
    current_checkpoint = args.checkpoint
    
    for stage, (ratio, epochs, lr) in enumerate(zip(args.ratios, args.epochs_list, args.lrs), 1):
        print(f"\n🎯 Starting Recovery Stage {stage}/{len(args.ratios)}")
        print(f"   Dataset ratio: {ratio:.1%}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr:.2e} (recovery-focused)")
        print(f"   Using checkpoint: {current_checkpoint}")
        
        # Build command for this recovery stage
        cmd = [
            "python", "train_full_dataset.py",
            "--data_root", args.data_root,
            "--ckpt", current_checkpoint,
            "--class_weights_file", class_weights_file,
            "--experiment_name", f"{args.experiment_name}_recovery_stage{stage}",
            "--use_combined_dataset",  # Use combined train+val dataset
            "--subset_ratio", str(ratio),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--batch_size", str(args.batch_size),
            "--val_batch_size", str(args.val_batch_size),
            "--weight_decay", str(args.weight_decay),
            "--target_max_ratio", str(args.target_max_ratio),
            "--optimizer", args.optimizer,
            "--warmup_epochs", str(args.warmup_epochs),
            "--warmup_start_lr", str(lr * 0.1),
            "--warmup_scheduler", args.warmup_scheduler,
            "--crop_size", str(args.crop_size),
            "--wandb_project", args.wandb_project,
            "--wandb_name", f"{args.wandb_name}_recovery_stage{stage}",
            "--wandb_notes", f"Overfitting Recovery Stage {stage}: {ratio:.1%} data, {epochs} epochs, LR {lr:.2e}",
            "--early_stop", "True",
            "--early_stop_patience", str(args.early_stop_patience)
        ]
        
        # Enhanced regularization for overfitting recovery
        if args.recovery_strategy == "aggressive":
            # More aggressive regularization
            cmd.extend([
                "--use_weather", "True", "--weather_p", "0.4",
                "--use_blur", "True", "--blur_p", "0.4",
                "--use_cutout", "True", "--cutout_p", "0.3",
                "--use_geometric", "True", "--geometric_p", "0.3",
                "--use_color", "True", "--color_p", "0.4"
            ])
        elif args.recovery_strategy == "conservative":
            # Conservative regularization
            cmd.extend([
                "--use_weather", "True", "--weather_p", "0.2",
                "--use_blur", "True", "--blur_p", "0.2",
                "--use_color", "True", "--color_p", "0.2"
            ])
        else:  # balanced
            # Balanced regularization
            cmd.extend([
                "--use_weather", "True", "--weather_p", "0.3",
                "--use_blur", "True", "--blur_p", "0.3",
                "--use_cutout", "True", "--cutout_p", "0.2",
                "--use_color", "True", "--color_p", "0.3"
            ])
        
        # Add WandB visualization parameters
        cmd.extend([
            "--enable_vis",
            "--vis_num_samples", "2"  # Reduced for lower overhead
        ])
        
        # Add recovery-specific WandB parameters
        base_tags = f"overfitting_recovery,stage{stage},ratio{ratio:.0%},combined_dataset,visual_monitoring"
        if args.wandb_tags:
            user_tags = args.wandb_tags.replace(" ", "").replace("'", "").replace('"', "")
            combined_tags = f"{base_tags},{user_tags}"
        else:
            combined_tags = base_tags
        
        cmd.extend([
            "--wandb_tags", combined_tags
        ])
        
        # Add basic augmentation parameters
        cmd.extend([
            "--horizontal_flip_p", str(args.horizontal_flip_p),
            "--brightness_limit", str(args.brightness_limit),
            "--contrast_limit", str(args.contrast_limit),
            "--rotation_limit", str(args.rotation_limit)
        ])
        
        print(f"Running Recovery Stage {stage} command: {' '.join(cmd)}")
        print()
        
        # Execute recovery training stage
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✅ Recovery Stage {stage} completed successfully!")
            
            # Find best checkpoint from this stage
            if stage < len(args.ratios):  # Not the last stage
                experiment_dir = f"checkpoints/{args.experiment_name}_recovery_stage{stage}"
                if os.path.exists(experiment_dir):
                    import glob
                    best_files = glob.glob(f"{experiment_dir}/*_best_model.pth")
                    if best_files:
                        current_checkpoint = max(best_files, key=os.path.getctime)
                        print(f"✓ Found best checkpoint: {current_checkpoint}")
                    else:
                        print(f"⚠️ No best model found for stage {stage}. Using previous checkpoint.")
                else:
                    print(f"⚠️ Experiment directory not found: {experiment_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Recovery Stage {stage} failed with error: {e}")
            return False
    
    print(f"\n{'='*80}")
    print(f"🎉 OVERFITTING RECOVERY TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Final checkpoint: {current_checkpoint}")
    print(f"Recovery strategy: {args.recovery_strategy}")
    print(f"{'='*80}\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Overfitting Recovery Training for Full Dataset Learning")
    
    # Required arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Overfitted checkpoint (minority class focused model)")
    parser.add_argument("--class_weights_file", type=str, required=True,
                        help="Path to class weights file")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Base experiment name")
    
    # Recovery strategy
    parser.add_argument("--recovery_strategy", type=str, default="balanced",
                        choices=["conservative", "balanced", "aggressive"],
                        help="Recovery strategy: conservative (gentle), balanced (moderate), aggressive (strong regularization)")
    
    # Class weights recalculation disabled - using existing weights
    # parser.add_argument("--recalculate_class_weights", action='store_true', default=True,
    #                     help="Recalculate class weights for combined dataset (recommended for overfitting recovery)")
    # parser.add_argument("--class_weights_method", type=str, default="sqrt_inv_freq",
    #                     choices=["inverse_freq", "sqrt_inv_freq", "effective_num", "median_freq"],
    #                     help="Method for calculating class weights")
    
    # Progressive recovery parameters
    parser.add_argument("--ratios", type=str, default="0.3,0.6,1.0",
                        help="Comma-separated dataset ratios for recovery (e.g., '0.3,0.6,1.0')")
    parser.add_argument("--epochs_list", type=str, default="50,50,80",
                        help="Comma-separated epochs for each recovery stage (e.g., '50,50,80')")
    parser.add_argument("--lrs", type=str, default="2e-5,8e-6,3e-6",
                        help="Comma-separated learning rates for recovery (e.g., '2e-5,8e-6,3e-6')")
    
    # Learning rate parameters
    parser.add_argument("--initial_lr", type=float, default=2e-5,
                        help="Initial learning rate for recovery (very low)")
    parser.add_argument("--final_lr", type=float, default=3e-6,
                        help="Final learning rate for recovery (very low)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help="Batch size for validation")
    parser.add_argument("--weight_decay", type=float, default=2e-4,
                        help="Weight decay (increased for overfitting recovery)")
    parser.add_argument("--target_max_ratio", type=float, default=2.5,
                        help="Target max ratio for class weights (reduced for stability)")
    parser.add_argument("--optimizer", type=str, default="sgd_nesterov",
                        choices=["sgd", "sgd_nesterov", "adamw", "radam"],
                        help="Optimizer type")
    parser.add_argument("--warmup_epochs", type=int, default=8,
                        help="Warmup epochs (increased for stability)")
    parser.add_argument("--warmup_scheduler", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="Warmup scheduler type")
    parser.add_argument("--crop_size", type=int, default=1024,
                        help="Crop size for training")
    
    # WandB parameters
    parser.add_argument("--wandb_project", type=str, default="deeplabv3-segmentation",
                        help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="WandB run name (default: auto-generated)")
    parser.add_argument("--wandb_tags", type=str, default="",
                        help="Comma-separated WandB tags")
    
    # Early stopping (more aggressive for overfitting recovery)
    parser.add_argument("--early_stop", type=bool, default=True,
                        help="Enable early stopping")
    parser.add_argument("--early_stop_patience", type=int, default=8,
                        help="Early stopping patience (reduced for overfitting recovery)")
    
    # Basic augmentation parameters
    parser.add_argument("--horizontal_flip_p", type=float, default=0.5,
                        help="Horizontal flip probability")
    parser.add_argument("--brightness_limit", type=float, default=0.2,
                        help="Brightness adjustment limit")
    parser.add_argument("--contrast_limit", type=float, default=0.2,
                        help="Contrast adjustment limit")
    parser.add_argument("--rotation_limit", type=int, default=10,
                        help="Rotation angle limit")
    
    args = parser.parse_args()
    
    # Set default wandb_name if not provided
    if args.wandb_name is None:
        args.wandb_name = f"Recovery_{args.experiment_name}"
    
    # Parse comma-separated values
    args.ratios = [float(x.strip()) for x in args.ratios.split(',')]
    args.epochs_list = [int(x.strip()) for x in args.epochs_list.split(',')]
    args.lrs = [float(x.strip()) for x in args.lrs.split(',')]
    
    # Validate parameters
    if len(args.ratios) != len(args.epochs_list) or len(args.ratios) != len(args.lrs):
        raise ValueError("ratios, epochs_list, and lrs must have the same length")
    
    # Validate learning rates are very low for recovery
    if max(args.lrs) > 5e-5:
        print(f"⚠️ Warning: Learning rates seem too high for overfitting recovery.")
        print(f"   Consider using lower learning rates (e.g., 2e-5, 8e-6, 3e-6)")
        print(f"   Current max LR: {max(args.lrs):.2e}")
    
    # Using existing class weights
    print(f"✅ Using existing class weights: class_weights/class_weight_old_data.pth")
    
    # Run recovery training
    success = run_recovery_training(args)
    
    if success:
        print("🎉 Overfitting recovery training completed successfully!")
        print("   The model should now have better generalization with the full dataset.")
        sys.exit(0)
    else:
        print("❌ Overfitting recovery training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
# ============================================================================
# THREE PC PARALLEL TRAINING COMMANDS
# ============================================================================

# PC Main: Conservative Strategy (Safe & Slow)
python train_overfitting_recovery.py \
    --data_root /mnt/c/Users/user/Desktop/eogus/dataset/2025dna \
    --checkpoint checkpoints/sweep_lr5.00e-04_weight6.5_epoch_040_miou_0.5045.pth \
    --class_weights_file class_weights/class_weight_old_data.pth \
    --experiment_name "PC_main_Conservative_Recovery" \
    --recovery_strategy conservative \
    --ratios "0.2,0.5,0.8,1.0" \
    --epochs_list "40,40,40,60" \
    --lrs "3e-6,1e-6,5e-7,2e-7" \
    --optimizer sgd_nesterov \
    --warmup_epochs 12 \
    --warmup_scheduler linear \
    --weight_decay 1e-4 \
    --target_max_ratio 2.0 \
    --batch_size 4 \
    --early_stop_patience 12 \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_name "PC_main_Conservative_Recovery"

# PC airlab: Balanced Strategy (Recommended)
python train_overfitting_recovery.py \
    --data_root /home/linux/deeplabv3/Deeplabv3/datasets/data \
    --checkpoint checkpoints/sweep_lr5.00e-04_weight6.5_epoch_040_miou_0.5045.pth \
    --class_weights_file class_weights/class_weight_old_data.pth \
    --experiment_name "PC_airlab_Balanced_Recovery" \
    --recovery_strategy balanced \
    --ratios "0.3,0.6,1.0" \
    --epochs_list "50,50,80" \
    --lrs "5e-6,2e-6,8e-7" \
    --optimizer sgd_nesterov \
    --warmup_epochs 10 \
    --warmup_scheduler linear \
    --weight_decay 2e-4 \
    --target_max_ratio 2.5 \
    --batch_size 4 \
    --early_stop_patience 10 \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_name "PC_airlab_Balanced_Recovery"

# PC 1: Aggressive Strategy (Fast & Risky)
python train_overfitting_recovery.py \
    --data_root /home/linux/deeplabv3_ws/DeepLabV3Plus-Pytorch/datasets/data \
    --checkpoint checkpoints/sweep_lr5.00e-04_weight6.5_epoch_040_miou_0.5045.pth \
    --class_weights_file class_weights/class_weight_old_data.pth \
    --experiment_name "PC_1_Aggressive_Recovery" \
    --recovery_strategy aggressive \
    --ratios "0.4,0.7,1.0" \
    --epochs_list "60,60,100" \
    --lrs "8e-6,3e-6,1e-6" \
    --optimizer adamw \
    --warmup_epochs 8 \
    --warmup_scheduler cosine \
    --weight_decay 3e-4 \
    --target_max_ratio 3.0 \
    --batch_size 2 \
    --early_stop_patience 8 \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_name "PC_1_Aggressive_Recovery"

# ============================================================================
# SINGLE PC TESTING COMMANDS
# ============================================================================

# Basic usage with existing class weights:
python train_overfitting_recovery.py \
    --data_root /home/linux/deeplabv3_ws/DeepLabV3Plus-Pytorch/datasets/data \
    --checkpoint overfitted_model.pth \
    --class_weights_file class_weights/class_weight_old_data.pth \
    --experiment_name "Overfitting_Recovery" \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_name "Overfitting_Recovery_Test" \
    --wandb_tags "test,recovery,existing_weights"

# Advanced usage with custom parameters:
python train_overfitting_recovery.py \
    --data_root /home/linux/deeplabv3_ws/DeepLabV3Plus-Pytorch/datasets/data \
    --checkpoint overfitted_model.pth \
    --class_weights_file class_weights/class_weight_old_data.pth \
    --experiment_name "Advanced_Recovery" \
    --recovery_strategy aggressive \
    --ratios "0.3,0.6,1.0" \
    --epochs_list "60,60,100" \
    --lrs "3e-5,1e-5,5e-6" \
    --batch_size 2 \
    --weight_decay 3e-4 \
    --target_max_ratio 3.0 \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_name "Advanced_Recovery_Test" \
    --wandb_tags "advanced,aggressive,existing_weights"

# Using existing class weights (recommended):
python train_overfitting_recovery.py \
    --data_root /home/linux/deeplabv3_ws/DeepLabV3Plus-Pytorch/datasets/data \
    --checkpoint overfitted_model.pth \
    --class_weights_file class_weights/class_weight_old_data.pth \
    --experiment_name "Existing_Weights" \
    --wandb_project "deeplabv3-segmentation" \
    --wandb_name "Existing_Weights_Test" \
    --wandb_tags "existing_weights,test"
"""

# PC Main

# python train_overfitting_recovery.py \
#     --data_root /home/linux/deeplabv3_ws/DeepLabV3Plus-Pytorch/datasets/data \
#     --checkpoint pc_main_best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --experiment_name "PC_main_Conservative_Recovery" \
#     --recovery_strategy conservative \
#     --ratios "0.2,0.5,0.8,1.0" \
#     --epochs_list "40,40,40,60" \
#     --lrs "3e-6,1e-6,5e-7,2e-7" \
#     --optimizer sgd_nesterov \
#     --warmup_epochs 12 \
#     --warmup_scheduler linear \
#     --weight_decay 1e-4 \
#     --target_max_ratio 2.0 \
#     --batch_size 4 \
#     --early_stop_patience 12 \
#     --class_weights_file class_weights/class_weight_old_data.pth \
#     --wandb_name "PC_main_Conservative" \
#     --wandb_tags "conservative,slow_recovery,sgd_nesterov"

# PC airlab

# python train_overfitting_recovery.py \
#     --data_root /home/linux/deeplabv3_ws/DeepLabV3Plus-Pytorch/datasets/data \
#     --checkpoint pc3_best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --experiment_name "PC3_Recovery_AdamW" \
#     --recovery_strategy conservative \
#     --ratios "0.3,0.6,1.0" \
#     --epochs_list "40,40,60" \
#     --lrs "3e-6,1e-6,5e-7" \
#     --optimizer adamw \
#     --warmup_epochs 8 \
#     --warmup_scheduler cosine \
#     --weight_decay 3e-4 \
#     --target_max_ratio 2.0 \
#     --batch_size 4 \
#     --class_weights_file class_weights/class_weight_old_data.pth

# PC 1

