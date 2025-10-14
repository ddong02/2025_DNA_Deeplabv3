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
    
    # Extract current learning rates from optimizer
    current_lrs = []
    for param_group in optimizer.param_groups:
        current_lrs.append(param_group['lr'])
    
    torch.save({
        "epoch": epoch,
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "current_lrs": current_lrs,  # Save current learning rates
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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load checkpoint for resuming training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load checkpoint on
        
    Returns:
        start_epoch: Epoch to resume from
        cur_itrs: Current iteration count
        best_score: Best validation score so far
    """
    print(f"\n=== Loading Checkpoint for Resume ===")
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model_state = checkpoint["model_state"]
    
    # Handle DataParallel model loading
    if hasattr(model, 'module'):
        # Model is wrapped with DataParallel, state_dict should have 'module.' prefix
        model.load_state_dict(model_state, strict=False)
    else:
        # Model is not wrapped, remove 'module.' prefix if present
        if any(key.startswith('module.') for key in model_state.keys()):
            new_state_dict = {}
            for key, value in model_state.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(model_state, strict=False)
    
    print("✓ Model state loaded")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("✓ Optimizer state loaded")
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    print("✓ Scheduler state loaded")
    
    # Get training state
    start_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
    cur_itrs = checkpoint.get("cur_itrs", 0)
    best_score = checkpoint.get("best_score", 0.0)
    
    print(f"✓ Training state: Epoch {start_epoch}, Iterations {cur_itrs}, Best Score {best_score:.4f}")
    print("Checkpoint loaded successfully!\n")
    
    return start_epoch, cur_itrs, best_score
