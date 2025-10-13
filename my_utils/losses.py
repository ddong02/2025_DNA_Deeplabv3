"""
Custom loss functions for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero. Default: 1.0
        ignore_index (int): Label value to ignore. Default: 255
        weight (torch.Tensor, optional): Class weights. Shape: (num_classes,)
        square_denominator (bool): Square the denominator for stability. Default: False
    """
    
    def __init__(self, smooth=1.0, ignore_index=255, weight=None, square_denominator=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
        self.square_denominator = square_denominator
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw model outputs (before softmax)
            targets: (B, H, W) - ground truth labels
        
        Returns:
            loss: scalar tensor
        """
        num_classes = logits.shape[1]
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (targets != self.ignore_index)  # (B, H, W)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets.clamp(0, num_classes - 1),  # Clamp to valid range
            num_classes=num_classes
        )  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Flatten spatial dimensions
        probs = probs.view(probs.shape[0], probs.shape[1], -1)  # (B, C, H*W)
        targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)  # (B, C, H*W)
        
        # Calculate Dice coefficient per class
        intersection = (probs * targets_one_hot).sum(dim=2)  # (B, C)
        
        if self.square_denominator:
            # Square denominator for more stable gradients
            union = (probs ** 2).sum(dim=2) + (targets_one_hot ** 2).sum(dim=2)  # (B, C)
        else:
            union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)  # (B, C)
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        
        # Dice Loss = 1 - Dice Score
        dice_loss = 1.0 - dice_score  # (B, C)
        
        # Apply class weights if provided
        if self.weight is not None:
            # Apply weights to the loss (not the score) for more intuitive behavior
            dice_loss = dice_loss * self.weight.unsqueeze(0)  # (B, C)
        
        # Average over classes and batch
        loss = dice_loss.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss.
    
    Args:
        ce_weight (float): Weight for Cross-Entropy loss. Default: 0.5
        dice_weight (float): Weight for Dice loss. Default: 0.5
        smooth (float): Smoothing factor for Dice loss. Default: 1.0
        ignore_index (int): Label value to ignore. Default: 255
        class_weights (torch.Tensor, optional): Class weights for both losses.
        square_denominator (bool): Square denominator in Dice loss. Default: False
    """
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5, smooth=1.0, 
                 ignore_index=255, class_weights=None, square_denominator=False):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        self.dice_loss = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
            weight=class_weights,
            square_denominator=square_denominator
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw model outputs
            targets: (B, H, W) - ground truth labels
        
        Returns:
            loss: scalar tensor
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss with proper class weights support for semantic segmentation.
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha (torch.Tensor, optional): Class weights tensor. Shape: (num_classes,)
        gamma (float): Focusing parameter. Higher gamma = more focus on hard examples. Default: 2.0
        ignore_index (int): Label value to ignore in loss calculation. Default: 255
    
    References:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights tensor: (num_classes,)
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw model outputs
            targets: (B, H, W) - ground truth labels
        
        Returns:
            loss: scalar tensor
        """
        # Compute CE loss without reduction to apply focal term per-pixel
        ce_loss = F.cross_entropy(
            logits, targets, 
            reduction='none',
            ignore_index=self.ignore_index
        )  # Shape: (B, H, W)
        
        # Compute focal term: (1 - p_t)^gamma
        # p_t is the probability of the true class
        pt = torch.exp(-ce_loss)  # Shape: (B, H, W)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Create valid mask for non-ignore pixels
            valid_mask = (targets != self.ignore_index)
            
            # Clamp targets to valid range [0, num_classes-1] to avoid index out of bounds
            targets_clamped = torch.clamp(targets, 0, len(self.alpha) - 1)
            
            # Get weight for each pixel based on its class
            alpha_t = self.alpha[targets_clamped]  # Shape: (B, H, W)
            
            # Apply valid mask to weights
            alpha_t = torch.where(
                valid_mask,
                alpha_t,
                torch.zeros_like(alpha_t)
            )
            focal_loss = alpha_t * focal_loss
        
        # Average over valid pixels only
        return focal_loss.mean()