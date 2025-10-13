"""
Main training script - refactored for DNA2025 dataset only.
Removed VOC and Cityscapes dependencies for cleaner code.
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
from my_utils.validation import validate
from my_utils.checkpoint import save_checkpoint, load_pretrained_model, load_checkpoint_for_continue
from my_utils.early_stopping import EarlyStopping
from my_utils.calculate_class_weights import calculate_class_weights
from my_utils.losses import CombinedLoss, FocalLoss


def get_dataset(opts):
    """Dataset And Augmentation - DNA2025 only"""
    train_dst = DNA2025Dataset(
        root_dir=opts.data_root,
        crop_size=[opts.crop_size, opts.crop_size],
        subset='train',
        scale_range=[0.75, 1.25],
        random_seed=opts.random_seed
    )
    
    val_dst = DNA2025Dataset(
        root_dir=opts.data_root,
        crop_size=[opts.crop_size, opts.crop_size],
        subset='val',
        scale_range=None,
        random_seed=opts.random_seed
    )

    return train_dst, val_dst


def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    opts = get_argparser().parse_args()

    # Setup visualization with WandB
    vis = None
    if opts.enable_vis:
        # Parse tags if provided
        tags = None
        if opts.wandb_tags:
            tags = [tag.strip() for tag in opts.wandb_tags.split(',')]
        
        # Generate run name if not provided
        run_name = opts.wandb_name
        if run_name is None:
            run_name = f"{opts.model}_{opts.dataset}_os{opts.output_stride}_baseline"
        
        vis = Visualizer(
            project=opts.wandb_project,
            name=run_name,
            config=vars(opts),
            notes=opts.wandb_notes,
            tags=tags
        )
    else:
        print("WandB visualization disabled. Use --enable_vis to enable.")

    # Use GPU 0 by default
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

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
    
    # Set target max ratio (10x for more balanced training with minority classes)
    target_max_ratio = 10.0
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
    
    # Loss function selection based on opts.loss_type
    if opts.loss_type == 'focal':
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=opts.focal_gamma,
            ignore_index=255
        )
        
        print("\n✓ Loss Function Configuration:")
        print(f"  Type: Focal Loss")
        print(f"  Gamma (focusing parameter): {opts.focal_gamma}")
        print(f"  Class Weights: Applied (sqrt_inv_freq method)")
        print(f"  Weight Ratio: {final_ratio:.1f}x (clipped to max {target_max_ratio:.0f}x)")
        print(f"  How it works: Focuses on hard examples by down-weighting easy examples")
        print(f"  Expected benefit: Better performance on minority classes and hard boundaries")
    else:
        # Default: Weighted Cross-Entropy Loss
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255,
            reduction='mean'
        )
        
        print("\n✓ Loss Function Configuration:")
        print(f"  Type: Weighted Cross-Entropy Loss")
        print(f"  Class Weights: Applied (sqrt_inv_freq method)")
        print(f"  Weight Ratio: {final_ratio:.1f}x (clipped to max {target_max_ratio:.0f}x)")
        print(f"  This configuration balances class importance while maintaining stability")

    print("="*80 + "\n")

    # Set up model
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, 
        output_stride=opts.output_stride
    )
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Setup checkpoint directory based on experiment_name
    if opts.experiment_name:
        checkpoint_dir = os.path.join('checkpoints', opts.experiment_name)
        print(f"\n✓ Experiment Name: '{opts.experiment_name}'")
        print(f"  Checkpoints will be saved to: {checkpoint_dir}/")
    else:
        checkpoint_dir = 'checkpoints'
        print(f"\n✓ Using default checkpoint directory: {checkpoint_dir}/")
    
    utils.mkdir(checkpoint_dir)
    
    # Initialize training variables
    start_epoch = 1
    cur_itrs = 0
    best_score = 0.0
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # Check if this is a continue training scenario
        if opts.continue_training:
            print("[!] Continue training from checkpoint")
            # We'll load the checkpoint after setting up optimizer and scheduler
            continue_from_checkpoint = True
        else:
            print("[!] Loading pretrained model")
            model, _ = load_pretrained_model(
                model, opts.ckpt, 
                num_classes_old=opts.pretrained_num_classes, 
                num_classes_new=opts.num_classes
            )
            continue_from_checkpoint = False
    else:
        print("[!] Training from scratch")
        continue_from_checkpoint = False

    # STAGE 1: Freeze backbone
    print("--- STAGE 1 SETUP: Training classifier only ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params_stage1 = filter(lambda p: p.requires_grad, model.parameters())
    
    # Using full learning rate with Weighted CE Loss
    print(f"Stage 1 Learning Rate: {opts.lr:.2e}")
    
    optimizer = torch.optim.SGD(
        params=trainable_params_stage1, 
        lr=opts.lr,
        momentum=0.9, 
        weight_decay=opts.weight_decay
    )

    # Use PolyLR scheduler
    scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)

    model = nn.DataParallel(model)
    model.to(device)

    # Load checkpoint for continue training if needed
    if continue_from_checkpoint:
        start_epoch, cur_itrs, best_score, _ = load_checkpoint_for_continue(
            opts.ckpt, model, optimizer, scheduler
        )
        print(f"Resuming from epoch {start_epoch}, iteration {cur_itrs}")

    # Training setup
    training_start_time = time.time()
    
    # Track metrics for displaying changes
    prev_metrics = {
        'loss': None,
        'Mean IoU': None,
        'Overall Acc': None,
        'Mean Acc': None
    }
    
    # ===== Early Stopping Setup =====
    early_stopping = None
    if opts.early_stop:
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
    # ================================
    
    vis_sample_id = np.random.randint(
        0, len(val_loader), opts.vis_num_samples, np.int32
    ) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ===== TRAINING LOOP =====
    # For continue training, adjust the epoch range
    if continue_from_checkpoint:
        # Continue training: start from loaded epoch + 1, run for specified additional epochs
        end_epoch = start_epoch + opts.epochs
        print(f"Continue training: Epochs {start_epoch + 1} to {end_epoch} (additional {opts.epochs} epochs)")
    else:
        # Normal training: start from 1, run for specified epochs
        end_epoch = opts.epochs
        start_epoch = 1
    
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        
        # STAGE 2: Unfreeze backbone
        if epoch == opts.unfreeze_epoch:
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

            remaining_itrs = max(1, opts.total_itrs - cur_itrs)  # 최소 1로 보장
            # Use PolyLR scheduler for Stage 2
            scheduler = utils.PolyLR(optimizer, remaining_itrs, power=0.9)
            
            # Reset early stopping to start fresh from Stage 2
            if early_stopping is not None:
                early_stopping.reset()
                print(f"[Early Stopping] Reset for Stage 2 - will set baseline after this epoch's validation")
        
        # Handle continue training from STAGE 2
        elif continue_from_checkpoint and epoch > opts.unfreeze_epoch:
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
        
        stage_str = '2 (Fine-tuning)' if epoch >= opts.unfreeze_epoch else '1 (Classifier only)'
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
            
            # Safe scheduler step to avoid complex number errors
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

        save_images_this_epoch = opts.save_val_results and (epoch % 10 == 0)
        
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
        
        # Show detailed class IoU (original output)
        print(f"\nDetailed Metrics:")
        print(metrics.to_str(val_score))
        
        # Update previous metrics for next epoch
        prev_metrics['loss'] = current_loss
        prev_metrics['Mean IoU'] = current_miou
        prev_metrics['Overall Acc'] = current_oacc
        prev_metrics['Mean Acc'] = current_macc

        # ===== Early Stopping Check =====
        # Only apply early stopping in Stage 2 (after backbone is unfrozen)
        if early_stopping is not None and epoch >= opts.unfreeze_epoch:
            monitor_score = val_score[opts.early_stop_metric]
            
            # First epoch of Stage 2: set baseline
            if epoch == opts.unfreeze_epoch:
                early_stopping.set_baseline(monitor_score)
                print(f"[Early Stopping] Stage 2 baseline set: {monitor_score:.4f}")
                print(f"[Early Stopping] Will monitor improvements from this point")
            else:
                # From second epoch of Stage 2 onwards, check for improvements
                should_stop = early_stopping(monitor_score, epoch)
                
                if should_stop:
                    print(f"\n{'='*80}")
                    print(f"Training stopped early at epoch {epoch}/{end_epoch}")
                    print(f"Best {opts.early_stop_metric}: {early_stopping.best_score:.4f}")
                    print(f"  (Set at Stage 2 start: epoch {opts.unfreeze_epoch})")
                    print(f"{'='*80}\n")
                    
                    # Save final checkpoint before stopping
                    early_stop_path = os.path.join(checkpoint_dir, f'early_stopped_epoch_{epoch:03d}.pth')
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

        # WandB updates (Enhanced monitoring)
        if vis is not None:
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate gradient norm for monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Log all metrics in one batch to avoid step conflicts
            wandb.log({
                'epoch': epoch,
                'Training Loss': avg_epoch_loss,
                'Learning Rate': current_lr,
                '[Val] Overall Acc': val_score['Overall Acc'],
                '[Val] Mean Acc': val_score['Mean Acc'],
                '[Val] Mean IoU': val_score['Mean IoU'],
                'Gradient Norm': total_norm,
            }, step=epoch)
            
            # Log class IoU table with same step
            vis.vis_table("[Val] Class IoU", val_score['Class IoU'], step=epoch)
            
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
                visdom_samples_dir = os.path.join('results', 'visdom_samples')
                os.makedirs(visdom_samples_dir, exist_ok=True)
                
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
                    
                    # Save locally
                    save_path = os.path.join(visdom_samples_dir, 
                                           f'validation_sample_{k}_epoch_{epoch:03d}.png')
                    try:
                        Image.fromarray(concat_img_hwc).save(save_path)
                    except Exception as e:
                        print(f"Warning: Failed to save visdom sample {k}: {e}")
                
                # Log all images at once
                wandb.log(wandb_images, step=epoch)

        # Save checkpoints
        current_score = val_score['Mean IoU']
        
        # 1. Save best model separately (fixed filename)
        if current_score > best_score:
            best_score = current_score
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(
                best_model_path,
                epoch, cur_itrs, model, optimizer, scheduler, best_score
            )
            print(f"✓ New best model saved! (Mean IoU: {best_score:.4f}) → {best_model_path}")
            
            if opts.save_val_results and not save_images_this_epoch:
                print("Best score achieved! Saving 3 comparison images...")
                _, _, _ = validate(
                    opts=opts, model=model, loader=val_loader, 
                    device=device, metrics=metrics, ret_samples_ids=None,
                    epoch=f"best_epoch_{epoch}", save_sample_images=True,
                    denorm=denorm
                )
        
        # 2. Save checkpoint for every epoch (with epoch number)
        epoch_checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'epoch_{epoch:03d}_miou_{current_score:.4f}.pth'
        )
        save_checkpoint(
            epoch_checkpoint_path,
            epoch, cur_itrs, model, optimizer, scheduler, best_score
        )
        print(f"✓ Epoch {epoch} checkpoint saved → {os.path.basename(epoch_checkpoint_path)}")
        
        # 3. Save latest model (overwritten each epoch)
        latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
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
        
        # Early stopping status (only show in Stage 2)
        if early_stopping is not None and epoch > opts.unfreeze_epoch and not early_stopping.early_stop:
            print(f"  Early stop counter: {early_stopping.counter}/{early_stopping.patience}")
        elif early_stopping is not None and epoch == opts.unfreeze_epoch:
            print(f"  Early stopping: Baseline set at {early_stopping.best_score:.4f} (monitoring starts next epoch)")
        elif early_stopping is not None and epoch < opts.unfreeze_epoch:
            print(f"  Early stopping: Inactive (Stage 1 - will activate at epoch {opts.unfreeze_epoch})")
        
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

# python my_train.py \
#     --ckpt checkpoints/ce_only2/best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --data_root ./datasets/data \
#     --experiment_name "focal_loss_gamma2_conservative" \
#     --epochs 50 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 0.00001 \
#     --crop_size 1024 \
#     --loss_type focal \
#     --focal_gamma 2.0 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "focal-loss-gamma2-conservative" \
#     --wandb_notes "Focal Loss (gamma=2.0) with conservative LR=0.00001, 50 epochs test" \
#     --wandb_tags "focal_loss,gamma2,fine_tuning,from_ce,conservative_lr,test_50epochs" \
#     --early_stop \
#     --early_stop_patience 10 \
#     --early_stop_min_delta 0.001 \
#     --early_stop_metric "Mean IoU" \
#     --save_val_results

# python my_train.py \
#     --ckpt checkpoints/ce_only2/best_model.pth \
#     --class_weights_file class_weights/dna2025dataset_sqrt_inv_freq_nc19.pth \
#     --data_root ./datasets/data \
#     --experiment_name "focal_loss_gamma1_5" \
#     --epochs 100 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 0.00001 \
#     --crop_size 1024 \
#     --loss_type focal \
#     --focal_gamma 1.5 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "focal-loss-gamma1.5 for performance comparison" \
#     --wandb_notes "Focal Loss (gamma=1.5)" \
#     --wandb_tags "focal_loss,gamma1.5,fine_tuning,from_ce,conservative_lr,test_50epochs" \
#     --early_stop \
#     --early_stop_patience 15 \
#     --early_stop_min_delta 0.001 \
#     --early_stop_metric "Mean IoU" \
#     --save_val_results