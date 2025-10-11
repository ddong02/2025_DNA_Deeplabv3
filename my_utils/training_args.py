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
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=None,
                        help="total iterations (now calculated from epochs)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs to train (default: 200)")
    parser.add_argument("--unfreeze_epoch", type=int, default=16,
                        help="epoch to unfreeze backbone (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', 
                        choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=1024)

    parser.add_argument("--ckpt", required=True, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--pretrained_num_classes", type=int, default=19,
                        help="number of classes in pretrained model (default: 19)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], 
                        help='year of VOC')

    # WandB (Weights & Biases) options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use WandB for visualization and logging")
    parser.add_argument("--wandb_project", type=str, default='deeplabv3-semantic-segmentation',
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
    parser.add_argument("--early_stop", action='store_true', default=False,
                        help="Enable early stopping")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0001,
                        help="Minimum improvement to count as progress")
    parser.add_argument("--early_stop_metric", type=str, default='Mean IoU',
                        choices=['Mean IoU', 'Overall Acc', 'Mean Acc'],
                        help="Metric to monitor for early stopping")

    return parser