# ============================================================================
# File: my_utils/wandb_visualization.py
# ============================================================================
"""
WandB Visualization Utilities for Training Monitoring
훈련 모니터링을 위한 WandB 시각화 유틸리티

This module provides lightweight visualization functions for monitoring
training progress without validation data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import torchvision.transforms as transforms

def log_training_metrics(epoch, loss, lr, gradient_norm, stage=None, recovery_status=None):
    """Log basic training metrics to WandB"""
    log_dict = {
        'epoch': epoch,
        'Training Loss': loss,
        'Learning Rate': lr,
        'Gradient Norm': gradient_norm,
    }
    
    if stage is not None:
        log_dict['Recovery Stage'] = stage
    
    if recovery_status is not None:
        log_dict['Recovery Status'] = recovery_status
    
    wandb.log(log_dict, step=epoch)
    return log_dict

def log_augmented_samples(train_loader, num_samples=2, device='cuda', epoch=None):
    """Log augmented training samples for visual validation"""
    try:
        samples = []
        dataset = train_loader.dataset
        
        # Get random samples
        import random
        import time
        random.seed(int(time.time() * 1000) % 2**32)
        
        for i in range(num_samples):
            try:
                random_idx = random.randint(0, len(dataset) - 1)
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
                label_np_clipped = np.clip(label_np, 0, 19)
                
                # Create overlay
                color_map = plt.cm.tab20(np.linspace(0, 1, 20))
                colored_label = color_map[label_np_clipped]
                colored_label_rgb = colored_label[:, :, :3]
                
                overlay = img_denorm.copy()
                overlay = (overlay * 0.7 + colored_label_rgb * 255 * 0.3).astype(np.uint8)
                
                # Resize for WandB
                overlay_resized = torch.nn.functional.interpolate(
                    torch.from_numpy(overlay).permute(2, 0, 1).unsqueeze(0).float(),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                
                samples.append(overlay_resized)
                
            except Exception as e:
                print(f"⚠️ Error getting sample {i+1}: {e}")
                dummy_sample = np.zeros((256, 256, 3), dtype=np.uint8)
                samples.append(dummy_sample)
        
        # Log to WandB
        log_dict = {}
        for i, sample in enumerate(samples):
            caption = f'Epoch {epoch} - Augmented Sample {i+1}' if epoch else f'Augmented Sample {i+1}'
            log_dict[f'Augmented Train Sample {i+1}'] = wandb.Image(sample, caption=caption)
        
        wandb.log(log_dict, step=epoch)
        return log_dict
        
    except Exception as e:
        print(f"❌ Failed to log augmented samples: {e}")
        return {}

def log_class_weights(class_weights, epoch=None):
    """Log class weights visualization"""
    try:
        weights_np = class_weights.cpu().numpy()
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(weights_np)), weights_np)
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Class Weights - Epoch {epoch}' if epoch else 'Class Weights')
        ax.set_xticks(range(len(weights_np)))
        ax.set_xticklabels([f'C{i}' for i in range(len(weights_np))])
        
        # Color bars by weight value
        for i, bar in enumerate(bars):
            if weights_np[i] > np.mean(weights_np):
                bar.set_color('red')  # High weights (minority classes)
            else:
                bar.set_color('blue')  # Low weights (majority classes)
        
        plt.tight_layout()
        
        # Log to WandB
        wandb.log({'Class Weights': wandb.Image(fig)}, step=epoch)
        plt.close(fig)
        
        return {'Class Weights': 'Logged successfully'}
        
    except Exception as e:
        print(f"❌ Failed to log class weights: {e}")
        return {}

def log_loss_curves(training_losses, epoch=None):
    """Log training loss curve"""
    try:
        if len(training_losses) < 2:
            return {}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(training_losses, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Loss Curve - Epoch {epoch}' if epoch else 'Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        wandb.log({'Training Loss Curve': wandb.Image(fig)}, step=epoch)
        plt.close(fig)
        
        return {'Training Loss Curve': 'Logged successfully'}
        
    except Exception as e:
        print(f"❌ Failed to log loss curve: {e}")
        return {}

def log_gradient_analysis(model, epoch=None):
    """Log gradient analysis for overfitting detection"""
    try:
        gradients = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                gradients.append(grad_norm)
                layer_names.append(name.split('.')[-1])  # Get layer name
        
        if not gradients:
            return {}
        
        # Create gradient analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gradient norms by layer
        ax1.bar(range(len(gradients)), gradients)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title(f'Gradient Norms by Layer - Epoch {epoch}' if epoch else 'Gradient Norms by Layer')
        ax1.set_xticks(range(len(gradients)))
        ax1.set_xticklabels(layer_names, rotation=45)
        
        # Gradient distribution
        ax2.hist(gradients, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Gradient Norm')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Gradient Distribution - Epoch {epoch}' if epoch else 'Gradient Distribution')
        
        plt.tight_layout()
        
        # Log to WandB
        wandb.log({'Gradient Analysis': wandb.Image(fig)}, step=epoch)
        plt.close(fig)
        
        return {'Gradient Analysis': 'Logged successfully'}
        
    except Exception as e:
        print(f"❌ Failed to log gradient analysis: {e}")
        return {}

def log_overfitting_indicators(training_losses, learning_rates, epoch=None):
    """Log overfitting indicators"""
    try:
        if len(training_losses) < 5:
            return {}
        
        # Calculate overfitting indicators
        recent_losses = training_losses[-5:]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Create overfitting indicators plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Loss trend
        ax1.plot(training_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.axhline(y=np.mean(training_losses), color='r', linestyle='--', alpha=0.7, label='Mean Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Trend Analysis - Epoch {epoch}' if epoch else 'Loss Trend Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax2.plot(learning_rates, 'g-', linewidth=2, label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title(f'Learning Rate Schedule - Epoch {epoch}' if epoch else 'Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        wandb.log({'Overfitting Indicators': wandb.Image(fig)}, step=epoch)
        plt.close(fig)
        
        # Log numerical indicators
        indicators = {
            'Loss Trend': loss_trend,
            'Current Loss': training_losses[-1],
            'Loss Variance': np.var(training_losses),
            'Learning Rate': learning_rates[-1] if learning_rates else 0
        }
        
        wandb.log(indicators, step=epoch)
        
        return {'Overfitting Indicators': 'Logged successfully', **indicators}
        
    except Exception as e:
        print(f"❌ Failed to log overfitting indicators: {e}")
        return {}

def log_recovery_progress(stage, ratio, epochs, lr, epoch=None):
    """Log recovery progress for overfitting recovery training"""
    try:
        progress_data = {
            'Recovery Stage': stage,
            'Data Ratio': ratio,
            'Epochs in Stage': epochs,
            'Learning Rate': lr,
            'Progress': (stage - 1) / 3 * 100  # Assuming 3 stages
        }
        
        # Create progress visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stages = ['Stage 1', 'Stage 2', 'Stage 3']
        progress = [33.3, 66.7, 100.0]
        current_progress = progress_data['Progress']
        
        bars = ax.bar(stages, progress, alpha=0.3, color='lightblue')
        ax.bar(stages[:stage], progress[:stage], alpha=0.8, color='green')
        
        ax.set_ylabel('Progress (%)')
        ax.set_title(f'Recovery Progress - Current: {current_progress:.1f}%')
        ax.set_ylim(0, 100)
        
        # Add current stage indicator
        ax.axvline(x=stage-1, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        
        # Log to WandB
        wandb.log({'Recovery Progress': wandb.Image(fig)}, step=epoch)
        wandb.log(progress_data, step=epoch)
        plt.close(fig)
        
        return {'Recovery Progress': 'Logged successfully', **progress_data}
        
    except Exception as e:
        print(f"❌ Failed to log recovery progress: {e}")
        return {}
