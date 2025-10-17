# ============================================================================
# File: configs/training_args.py
# ============================================================================
"""
Training argument parser for semantic segmentation.
"""

import argparse
import network


def get_argparser():
    """Create argument parser with all training options."""
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='dna2025dataset',
                        choices=['dna2025dataset'], 
                        help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ 
                            if name.islower() and 
                            not (name.startswith("__") or name.startswith('_')) and 
                            callable(network.modeling.__dict__[name]))
    
    parser.add_argument("--model", type=str, default='deeplabv3_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs to train (default: 200)")
    parser.add_argument("--unfreeze_epoch", type=int, default=16,
                        help="epoch to unfreeze backbone (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=1024,
                        help="crop size for training (default: 1024)")

    parser.add_argument("--ckpt", required=True, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False,
                        help="continue training from checkpoint")
    parser.add_argument("--pretrained_num_classes", type=int, default=19,
                        help="number of classes in pretrained model (default: 19)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="experiment name for organizing checkpoints (e.g., 'combined_loss_safe'). "
                             "If specified, checkpoints will be saved to 'checkpoints/{experiment_name}/' "
                             "instead of 'checkpoints/'")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")

    # WandB (Weights & Biases) options
    parser.add_argument("--enable_vis", action='store_true', default=True,
                        help="use WandB for visualization and logging")
    parser.add_argument("--wandb_project", type=str, default='deeplabv3-segmentation',
                        help='WandB project name')
    parser.add_argument("--wandb_name", type=str, default=None,
                        help='WandB run name (default: auto-generated)')
    parser.add_argument("--wandb_notes", type=str, default=None,
                        help='Notes about this run')
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help='Comma-separated tags for this run (e.g., "baseline,mobilenet")')
    parser.add_argument("--vis_num_samples", type=int, default=4,
                        help='number of samples for visualization (default: 4)')
    
    # Early Stopping 관련 인자 추가
    parser.add_argument("--early_stop", type=bool, default=False,
                    help="Enable early stopping (True/False)")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0001,
                        help="Minimum improvement to count as progress")
    parser.add_argument("--early_stop_metric", type=str, default='Mean IoU',
                        choices=['Mean IoU', 'Overall Acc', 'Mean Acc'],
                        help="Metric to monitor for early stopping")
    
    # Class Weights 관련 인자
    parser.add_argument("--class_weights_file", type=str, default=None,
                        help="Path to saved class weights file (.pth). If specified, load weights from file. "
                             "If not specified, calculate and save to 'class_weights/{dataset}_{method}.pth'")
    parser.add_argument("--skip_save_class_weights", action='store_true', default=False,
                        help="Skip saving calculated class weights to file")
    
    # Dataset subset 관련 인자
    parser.add_argument("--subset_ratio", type=float, default=1.0,
                        help="Ratio of dataset to use for training (0.0-1.0, default: 1.0 for full dataset). "
                             "Useful for quick testing (e.g., 0.05 for 5% of data)")
    
    # Class weights optimization 관련 인자
    parser.add_argument("--target_max_ratio", type=float, default=6.5,
                        help="Maximum ratio for class weights clipping (default: 6.5). "
                             "Used for WandB sweep optimization.")
    
    # Optimizer 관련 인자
    parser.add_argument("--optimizer", type=str, default='sgd_nesterov',
                        choices=['sgd', 'sgd_nesterov', 'adamw', 'radam'],
                        help="Optimizer type: sgd, sgd_nesterov, adamw, radam")
    
    # Warmup 관련 인자
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Number of warmup epochs (0 to disable warmup)")
    parser.add_argument("--warmup_start_lr", type=float, default=None,
                        help="Starting learning rate for warmup (default: lr/10)")
    parser.add_argument("--warmup_scheduler", type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help="Warmup scheduler type: linear or cosine")
    
    # Learning rate scheduler 관련 인자
    parser.add_argument("--scheduler_type", type=str, default='cosine', 
                       choices=['cosine', 'reduce'], 
                       help="Learning rate scheduler type for Stage 2 (default: cosine)")
    
    # Focal Loss 관련 인자
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma parameter (focusing parameter). "
                             "Higher values focus more on hard examples. Default: 2.0")
    parser.add_argument("--use_focal_loss", action='store_true', default=True,
                        help="Use Focal Loss instead of Cross-Entropy Loss")
    
    # Backbone freezing 관련 인자
    parser.add_argument("--freeze_backbone", action='store_true', default=False,
                        help="Freeze backbone parameters during training (default: False)")
    
    # 증강 강도 파라미터들 (WandB Sweep용)
    parser.add_argument("--horizontal_flip_p", type=float, default=0.5,
                        help="Probability of horizontal flip (0.0-1.0, default: 0.5)")
    parser.add_argument("--brightness_limit", type=float, default=0.2,
                        help="Brightness adjustment limit (0.0-1.0, default: 0.2)")
    parser.add_argument("--contrast_limit", type=float, default=0.2,
                        help="Contrast adjustment limit (0.0-1.0, default: 0.2)")
    parser.add_argument("--rotation_limit", type=int, default=10,
                        help="Rotation angle limit in degrees (0-180, default: 10)")
    
    # PyTorch 기반 고급 증강 파라미터들 제거됨 (Albumentations로 대체)
    
    # Albumentations 고급 증강 파라미터들
    parser.add_argument("--use_weather", type=bool, default=False,
                        help="Use weather simulation (rain, fog, shadow)")
    parser.add_argument("--weather_p", type=float, default=0.3,
                        help="Weather simulation probability (0.0-1.0, default: 0.3)")
    
    parser.add_argument("--use_noise", type=bool, default=False,
                        help="Use noise addition (Gaussian, ISO, Multiplicative)")
    parser.add_argument("--noise_p", type=float, default=0.3,
                        help="Noise addition probability (0.0-1.0, default: 0.3)")
    
    parser.add_argument("--use_blur", type=bool, default=False,
                        help="Use blur effects (Motion, Median, Gaussian)")
    parser.add_argument("--blur_p", type=float, default=0.3,
                        help="Blur effects probability (0.0-1.0, default: 0.3)")
    
    parser.add_argument("--use_cutout", type=bool, default=False,
                        help="Use CutOut effects (CoarseDropout, GridDropout)")
    parser.add_argument("--cutout_p", type=float, default=0.3,
                        help="CutOut effects probability (0.0-1.0, default: 0.3)")
    
    parser.add_argument("--use_geometric", type=bool, default=False,
                        help="Use geometric transforms (Elastic, Grid, Optical)")
    parser.add_argument("--geometric_p", type=float, default=0.3,
                        help="Geometric transform probability (0.0-1.0, default: 0.3)")
    
    parser.add_argument("--use_color", type=bool, default=False,
                        help="Use color transforms (HueSaturation, RGBShift, ChannelShuffle)")
    parser.add_argument("--color_p", type=float, default=0.3,
                        help="Color transform probability (0.0-1.0, default: 0.3)")

    return parser