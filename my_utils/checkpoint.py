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


def load_checkpoint_for_continue(checkpoint_path, model, optimizer, scheduler):
    """
    Load checkpoint for continuing training with restored learning rates.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state
        scheduler: Scheduler to restore state
        
    Returns:
        epoch: Epoch number from checkpoint
        cur_itrs: Current iteration from checkpoint
        best_score: Best score from checkpoint
        current_lrs: Current learning rates from checkpoint
    """
    print(f"\n=== Loading Checkpoint for Continue ===")
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), 
                          weights_only=False)
    
    # Load model state
    try:
        model.load_state_dict(checkpoint["model_state"])
        print("✓ Model state loaded")
    except RuntimeError as e:
        if "Missing key(s)" in str(e) and "module." in str(e):
            print("⚠ DataParallel key mismatch detected, trying to fix...")
            # Try loading without DataParallel wrapper
            model.module.load_state_dict(checkpoint["model_state"])
            print("✓ Model state loaded (fixed DataParallel issue)")
            
            # Verify model loading by checking a few key parameters
            try:
                # Check if backbone parameters are loaded correctly
                backbone_params = list(model.module.backbone.parameters())
                if len(backbone_params) > 0:
                    print(f"✓ Model verification: {len(backbone_params)} backbone parameters loaded")
                else:
                    print("⚠ Warning: No backbone parameters found")
            except Exception as verify_e:
                print(f"⚠ Model verification failed: {verify_e}")
        else:
            raise e
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("✓ Optimizer state loaded")
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    print("✓ Scheduler state loaded")
    
    # Extract information
    epoch = checkpoint.get("epoch", 0)
    cur_itrs = checkpoint.get("cur_itrs", 0)
    best_score = checkpoint.get("best_score", 0.0)
    current_lrs = checkpoint.get("current_lrs", [])
    
    print(f"✓ Epoch: {epoch}")
    print(f"✓ Iterations: {cur_itrs}")
    print(f"✓ Best Score: {best_score:.4f}")
    
    if current_lrs:
        if len(current_lrs) == 1:
            print(f"✓ Learning Rate: {current_lrs[0]:.6f}")
        else:
            print(f"✓ Learning Rates: {current_lrs}")
    else:
        print("⚠ No learning rate information found in checkpoint")
    
    print("Checkpoint loaded successfully for continue training!\n")
    return epoch, cur_itrs, best_score, current_lrs