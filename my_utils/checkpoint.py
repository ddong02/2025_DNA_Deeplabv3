# ============================================================================
# File: utils/checkpoint.py
# ============================================================================
"""
Checkpoint utilities for model saving and loading.
"""

import os
import torch


def save_checkpoint(path, epoch, cur_itrs, model, optimizer, scheduler, best_score, 
                   include_epoch_in_name=False):
    """
    Save model checkpoint.
    
    Args:
        path: Base path to save checkpoint
        epoch: Current epoch
        cur_itrs: Current iteration
        model: Model to save (DataParallel wrapper)
        optimizer: Optimizer state
        scheduler: Scheduler state
        best_score: Best validation score
        include_epoch_in_name: Whether to include epoch in filename
    """
    if include_epoch_in_name:
        base_path, ext = os.path.splitext(path)
        path = f"{base_path}_epoch{epoch:03d}{ext}"
    
    torch.save({
        "epoch": epoch,
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print(f"Model saved as {path}")


def load_pretrained_model(model, checkpoint_path, num_classes_old, num_classes_new):
    """
    Load pretrained model and adjust for different number of classes.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        num_classes_old: Number of classes in pretrained model
        num_classes_new: Number of classes in current model
        
    Returns:
        model: Model with loaded weights
        checkpoint: Full checkpoint dictionary
    """
    print(f"\n=== Loading Pretrained Model ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Adjusting from {num_classes_old} to {num_classes_new} classes")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), 
                          weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in model.state_dict():
            if model.state_dict()[key].shape == value.shape:
                new_state_dict[key] = value
            else:
                print(f"  Skipping {key} due to size mismatch: "
                      f"{value.shape} -> {model.state_dict()[key].shape}")
        else:
            print(f"  Skipping {key} as it does not exist in the current model.")

    model.load_state_dict(new_state_dict, strict=False)
    print("Pretrained weights loaded successfully!\n")
    return model, checkpoint