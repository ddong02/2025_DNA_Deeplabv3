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
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'dna2025dataset'], 
                        help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ 
                            if name.islower() and 
                            not (name.startswith("__") or name.startswith('_')) and 
                            callable(network.modeling.__dict__[name]))
    
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=None,
                        help="total iterations (now calculated from epochs)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
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
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--pretrained_num_classes", type=int, default=21,
                        help="number of classes in pretrained model (default: 21 for VOC)")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], 
                        help="loss type (default: False)")
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

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=4,
                        help='number of samples for visualization (default: 4)')
    
    # Class weight options
    parser.add_argument("--use_class_weights", action='store_true', default=False,
                        help="use class weights for handling class imbalance")
    parser.add_argument("--weight_method", type=str, default='inverse_freq',
                        choices=['inverse_freq', 'sqrt_inv_freq', 'effective_num', 'median_freq'],
                        help="method to calculate class weights (default: inverse_freq)")
    parser.add_argument("--effective_beta", type=float, default=0.9999,
                        help="beta value for effective number method (default: 0.9999)")
    
    return parser