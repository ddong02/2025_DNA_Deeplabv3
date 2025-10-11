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

    # ===== Setup loss function (Baseline: Standard Cross-Entropy) =====
    metrics = StreamSegMetrics(opts.num_classes)
    
    # Use standard Cross-Entropy Loss without class weights
    criterion = nn.CrossEntropyLoss(
        ignore_index=255,
        reduction='mean'
    )
    print(f"\n✓ Using Standard Cross-Entropy Loss (Baseline)")
    print(f"  No class weights, no advanced loss functions")
    print(f"  This is the baseline for comparison with improved methods")

    # Set up model
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, 
        output_stride=opts.output_stride
    )
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Load pretrained model if specified
    utils.mkdir('checkpoints')
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
    optimizer = torch.optim.SGD(
        params=trainable_params_stage1, 
        lr=opts.lr, 
        momentum=0.9, 
        weight_decay=opts.weight_decay
    )
    
    print(f"Initial Learning Rate: {opts.lr}")

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
            backbone_lr = opts.lr / 100
            classifier_lr = opts.lr / 10
            
            optimizer = torch.optim.SGD([
                {'params': model.module.backbone.parameters(), 'lr': backbone_lr},
                {'params': model.module.classifier.parameters(), 'lr': classifier_lr}
            ], momentum=0.9, weight_decay=opts.weight_decay)
            
            print(f"Backbone LR: {backbone_lr:.6f}, Classifier LR: {classifier_lr:.6f}")

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

        print(f"Validation Results:")
        print(metrics.to_str(val_score))

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
                        'checkpoints/early_stopped_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride),
                        epoch, cur_itrs, model, optimizer, scheduler, best_score,
                        include_epoch_in_name=True
                    )
                    
                    break  # Exit training loop
        elif early_stopping is not None and epoch < opts.unfreeze_epoch:
            # In Stage 1, just print that early stopping is not active yet
            if epoch == 1:
                print(f"[Early Stopping] Waiting for Stage 2 (epoch {opts.unfreeze_epoch}) to start monitoring...")
        # ================================

        # Visdom updates
        if vis is not None:
            current_lr = optimizer.param_groups[0]['lr']
            vis.vis_scalar('Training Loss', epoch, avg_epoch_loss)
            vis.vis_scalar('Learning Rate', epoch, current_lr)
            vis.vis_scalar("[Val] Overall Acc", epoch, val_score['Overall Acc'])
            vis.vis_scalar("[Val] Mean Acc", epoch, val_score['Mean Acc'])
            vis.vis_scalar("[Val] Mean IoU", epoch, val_score['Mean IoU'])
            vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

            if ret_samples and len(ret_samples) > 0:
                samples_to_show = ret_samples[:4]
                visdom_samples_dir = os.path.join('results', 'visdom_samples')
                os.makedirs(visdom_samples_dir, exist_ok=True)
                
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
                    vis.vis_image(f'Validation Sample {k}', concat_img)
                    
                    concat_img_hwc = concat_img.transpose(1, 2, 0)
                    save_path = os.path.join(visdom_samples_dir, 
                                           f'validation_sample_{k}_epoch_{epoch:03d}.png')
                    try:
                        Image.fromarray(concat_img_hwc).save(save_path)
                    except Exception as e:
                        print(f"Warning: Failed to save visdom sample {k}: {e}")

        # Save checkpoints
        current_score = val_score['Mean IoU']
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                'checkpoints/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride),
                epoch, cur_itrs, model, optimizer, scheduler, best_score
            )
            save_checkpoint(
                'checkpoints/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride),
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
            'checkpoints/latest_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride),
            epoch, cur_itrs, model, optimizer, scheduler, best_score
        )
        
        if epoch % 10 == 0:
            save_checkpoint(
                'checkpoints/checkpoint_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride),
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

# 학습 실행 예시 (Baseline - WandB 활성화)
# python my_train.py \
#     --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
#     --lr 1e-5 \
#     --enable_vis \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_name "baseline-standard-ce" \
#     --wandb_tags "baseline,standard-ce" \
#     --save_val_results \
#     --early_stop \
#     --early_stop_patience 15