# ============================================================================
# File: utils/validation.py
# ============================================================================
"""
Validation utilities for semantic segmentation training.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def validate(opts, model, loader, device, metrics, ret_samples_ids=None, 
             epoch=None, save_sample_images=False, denorm=None):
    """
    Perform validation and optionally save sample images.
    
    Args:
        opts: Argument namespace
        model: Model to validate
        loader: Validation dataloader
        device: Device to use
        metrics: Metrics object
        ret_samples_ids: Sample IDs to return
        epoch: Current epoch number
        save_sample_images: Whether to save sample images
        denorm: Denormalization function
        
    Returns:
        score: Validation metrics (dict)
        ret_samples: List of sample (image, target, prediction) tuples
        confusion_mat: Confusion matrix (numpy array)
    """
    metrics.reset()
    ret_samples = []
    
    if save_sample_images and epoch is not None:
        base_results_dir = 'results'
        if isinstance(epoch, str):
            results_dir = os.path.join(base_results_dir, epoch)
        else:
            results_dir = os.path.join(base_results_dir, f'epoch_{epoch:03d}')
        os.makedirs(results_dir, exist_ok=True)
        
        saved_count = 0
        max_save_images = 3

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            # Save comparison images
            if save_sample_images and epoch is not None and saved_count < max_save_images:
                image = images[0].detach().cpu().numpy()
                target = targets[0]
                pred = preds[0]

                if denorm is not None:
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                
                # Decode targets if dataset supports it
                if hasattr(loader.dataset, 'decode_target'):
                    target_rgb = loader.dataset.decode_target(target).astype(np.uint8)
                    pred_rgb = loader.dataset.decode_target(pred).astype(np.uint8)
                else:
                    target_rgb = target.astype(np.uint8)
                    pred_rgb = pred.astype(np.uint8)

                # Create comparison figure
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(target_rgb)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred_rgb)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                comparison_path = os.path.join(results_dir, f'comparison_{saved_count:02d}.png')
                plt.tight_layout()
                plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
                plt.close()

                saved_count += 1

        score = metrics.get_results()
        
        # Get confusion matrix for logging
        confusion_mat = metrics.confusion_matrix.copy()
        
        # Save summary
        if save_sample_images and epoch is not None:
            summary_path = os.path.join(results_dir, 'validation_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Validation Results\n")
                f.write(f"================\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Saved Images: {saved_count}\n\n")
                f.write(metrics.to_str(score))
                f.write(f"\nBest Scores:\n")
                for key, value in score.items():
                    if isinstance(value, dict):
                        f.write(f"{key}: [Dict with {len(value)} items]\n")
                    elif isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            print(f"Validation results saved to: {results_dir} ({saved_count} comparison images)")
    
    return score, ret_samples, confusion_mat