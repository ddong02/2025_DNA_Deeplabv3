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

# ===== Import separated modules =====
from my_utils.training_args import get_argparser
from my_utils.dna2025_dataset import DNA2025Dataset
from my_utils.validation import validate
from my_utils.checkpoint import save_checkpoint, load_pretrained_model
from my_utils.early_stopping import EarlyStopping
from my_utils.calculate_class_weights import calculate_class_weights
from my_utils.losses import CombinedLoss


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

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
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

    if opts.total_itrs is None:
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
    
    # Set target max ratio (30x is a good balance for fine-tuning)
    target_max_ratio = 30.0
    min_weight_threshold = 0.1
    max_weight_threshold = min_weight_threshold * target_max_ratio  # 3.0
    
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
    
    if final_ratio <= 20:
        print(f"  ✓ Weight ratio is SAFE ({final_ratio:.1f}x ≤ 20x)")
    elif final_ratio <= 30:
        print(f"  ✓ Weight ratio is ACCEPTABLE ({final_ratio:.1f}x ≤ 30x)")
    else:
        print(f"  ⚠ Weight ratio is HIGH ({final_ratio:.1f}x > 30x) - may cause instability")
    print("-"*80)
    
    # Create Combined Loss with safe parameters
    criterion = CombinedLoss(
        ce_weight=0.7,              # CE 70% (stable)
        dice_weight=0.3,            # Dice 30% (conservative)
        smooth=5.0,                 # Large smooth for stability
        ignore_index=255,
        class_weights=class_weights,
        square_denominator=True     # More stable gradients
    )
    
    print("\n✓ Loss Function Configuration:")
    print(f"  Type: Combined Loss (CE + Dice)")
    print(f"  CE Weight: 0.7 (70%)")
    print(f"  Dice Weight: 0.3 (30%)")
    print(f"  Dice Smooth: 5.0 (high stability)")
    print(f"  Square Denominator: True")
    print(f"  Class Weights: Applied (sqrt_inv_freq, clipped)")
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
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        model, _ = load_pretrained_model(
            model, opts.ckpt, 
            num_classes_old=opts.pretrained_num_classes, 
            num_classes_new=opts.num_classes
        )
    else:
        print("[!] Training from scratch")

    # STAGE 1: Freeze backbone
    print("--- STAGE 1 SETUP: Training classifier only ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params_stage1 = filter(lambda p: p.requires_grad, model.parameters())
    
    # Adjusted learning rate for Combined Loss (30% of original)
    adjusted_lr = opts.lr * 0.3
    print(f"Original LR: {opts.lr:.2e}")
    print(f"Adjusted LR for Combined Loss: {adjusted_lr:.2e} (30% of original)")
    
    optimizer = torch.optim.SGD(
        params=trainable_params_stage1, 
        lr=adjusted_lr,  # Reduced LR for stability
        momentum=0.9, 
        weight_decay=opts.weight_decay
    )
    
    print(f"Stage 1 Learning Rate: {adjusted_lr:.2e}")

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.step_size, gamma=0.1
        )

    model = nn.DataParallel(model)
    model.to(device)

    # Training setup
    best_score = 0.0
    cur_itrs = 0
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
    for epoch in range(1, opts.epochs + 1):
        epoch_start_time = time.time()
        
        # STAGE 2: Unfreeze backbone
        if epoch == opts.unfreeze_epoch:
            print(f"\n--- STAGE 2: Unfreezing backbone at Epoch {epoch} ---")
            
            for param in model.module.backbone.parameters():
                param.requires_grad = True
            
            print("Re-creating optimizer with differential learning rates...")
            # Adjusted for Combined Loss (30% of original)
            # Fine-tuning from converged model: use less aggressive differential rates
            backbone_lr = (opts.lr / 20) * 0.3      # 1.5e-7 (was 3e-8, too low)
            classifier_lr = (opts.lr / 5) * 0.3     # 6e-7 (was 3e-7, slightly higher)
            
            optimizer = torch.optim.SGD([
                {'params': model.module.backbone.parameters(), 'lr': backbone_lr},
                {'params': model.module.classifier.parameters(), 'lr': classifier_lr}
            ], momentum=0.9, weight_decay=opts.weight_decay)
            
            print(f"Backbone LR: {backbone_lr:.6f} (adjusted for Combined Loss)")
            print(f"Classifier LR: {classifier_lr:.6f} (adjusted for Combined Loss)")

            remaining_itrs = opts.total_itrs - cur_itrs
            if opts.lr_policy == 'poly':
                scheduler = utils.PolyLR(optimizer, remaining_itrs, power=0.9)
            elif opts.lr_policy == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=opts.step_size, gamma=0.1
                )
            
            # Reset early stopping to start fresh from Stage 2
            if early_stopping is not None:
                early_stopping.reset()
                print(f"[Early Stopping] Reset for Stage 2 - will set baseline after this epoch's validation")
        
        # Print current learning rate
        if len(optimizer.param_groups) == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{opts.epochs} - Current Learning Rate: {current_lr:.6f}")
            print(f"{'='*80}")
        else:
            backbone_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{opts.epochs} - Learning Rates:")
            print(f"  Backbone:   {backbone_lr:.6f}")
            print(f"  Classifier: {classifier_lr:.6f}")
            print(f"{'='*80}")

        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        interval_loss = 0.0
        
        stage_str = '2 (Fine-tuning)' if epoch >= opts.unfreeze_epoch else '1 (Classifier only)'
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{opts.epochs} [Stage {stage_str}]")
        
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
            scheduler.step()

            np_loss = loss.detach().cpu().numpy()
            epoch_loss += np_loss
            num_batches += 1
            interval_loss += np_loss

            if cur_itrs % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                progress_bar.set_postfix(loss=f"{interval_loss:.4f}")
                interval_loss = 0.0

        avg_epoch_loss = epoch_loss / num_batches
        
        print(f"\nEpoch {epoch}/{opts.epochs} [{stage_str}] completed:")
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
        
        val_score, ret_samples = validate(
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
                    print(f"Training stopped early at epoch {epoch}/{opts.epochs}")
                    print(f"Best {opts.early_stop_metric}: {early_stopping.best_score:.4f}")
                    print(f"  (Set at Stage 2 start: epoch {opts.unfreeze_epoch})")
                    print(f"{'='*80}\n")
                    
                    # Save final checkpoint before stopping
                    save_checkpoint(
                        os.path.join(checkpoint_dir, 'early_stopped_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride)),
                        epoch, cur_itrs, model, optimizer, scheduler, best_score,
                        include_epoch_in_name=True
                    )
                    
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
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                os.path.join(checkpoint_dir, 'best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride)),
                epoch, cur_itrs, model, optimizer, scheduler, best_score
            )
            save_checkpoint(
                os.path.join(checkpoint_dir, 'best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride)),
                epoch, cur_itrs, model, optimizer, scheduler, best_score,
                include_epoch_in_name=True
            )
            
            if opts.save_val_results and not save_images_this_epoch:
                print("Best score achieved! Saving 3 comparison images...")
                validate(
                    opts=opts, model=model, loader=val_loader, 
                    device=device, metrics=metrics, ret_samples_ids=None,
                    epoch=f"best_epoch_{epoch}", save_sample_images=True,
                    denorm=denorm
                )
        
        save_checkpoint(
            os.path.join(checkpoint_dir, 'latest_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride)),
            epoch, cur_itrs, model, optimizer, scheduler, best_score
        )
        
        if epoch % 10 == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, 'checkpoint_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride)),
                epoch, cur_itrs, model, optimizer, scheduler, best_score,
                include_epoch_in_name=True
            )
        
        # Time tracking
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = opts.epochs - epoch
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
        print(f"Training finished early at epoch {epoch}/{opts.epochs}")
    else:
        print(f"Training finished successfully - completed all {opts.epochs} epochs")
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

# 학습 실행 예시 (Combined Loss + Class Weights - Safe Version)
# python my_train.py \
#     --dataset dna2025dataset \
#     --data_root ./datasets/data \
#     --model deeplabv3_mobilenet \
#     --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
#     --pretrained_num_classes 19 \
#     --num_classes 19 \
#     --epochs 200 \
#     --unfreeze_epoch 15 \
#     --lr 1e-5 \
#     --batch_size 4 \
#     --crop_size 1024 \
#     --experiment_name combined_loss_safe \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "combined-loss-classweights-safe" \
#     --wandb_tags "combined-loss,class-weights,safe" \
#     --save_val_results \
#     --early_stop \
#     --early_stop_patience 15

# 주요 설정:
# - Checkpoint 저장: checkpoints/combined_loss_safe/
# - Loss: Combined (CE 70% + Dice 30%)
# - Class Weights: sqrt_inv_freq, clipped (0.1~10.0)
# - LR: 자동으로 30%로 조정됨 (1e-5 → 3e-6)
# - Gradient Clipping: max_norm=1.0
# - Dice Smooth: 5.0 (high stability)
# - Square Denominator: True

# 다른 실험 예시:
# python my_train.py --experiment_name focal_loss ... (checkpoints/focal_loss/)
# python my_train.py --experiment_name baseline_v2 ... (checkpoints/baseline_v2/)
# python my_train.py ... (--experiment_name 없으면 checkpoints/에 저장)