"""
Class Weight Calculation Utilities for Semantic Segmentation

This module provides various methods for calculating class weights
to handle class imbalance in semantic segmentation tasks.
"""

import numpy as np
import torch
from tqdm import tqdm


def calculate_class_weights(dataset, num_classes, device, method='inverse_freq', 
                           beta=0.9999, ignore_index=255):
    """
    Calculate class weights from dataset to handle class imbalance.
    
    Traverses all labels in the dataset, counts pixels per class,
    and computes weights using the specified method.
    
    Args:
        dataset: Dataset object with __len__ and __getitem__ methods
        num_classes (int): Number of classes
        device (torch.device): Device to put tensor on (cpu/cuda)
        method (str): Weight calculation method. Options:
            - 'inverse_freq': Standard inverse frequency weighting
            - 'sqrt_inv_freq': Square root inverse frequency (less extreme)
            - 'effective_num': Effective number of samples method
            - 'median_freq': Median frequency balancing (from SegNet paper)
        beta (float): Beta parameter for effective_num method (default: 0.9999)
        ignore_index (int): Index to ignore in labels (default: 255)
    
    Returns:
        torch.Tensor: Class weights tensor of shape (num_classes,)
    
    References:
        - Inverse Frequency: Standard practice in imbalanced learning
        - Effective Number: "Class-Balanced Loss Based on Effective Number of Samples"
        - Median Frequency: "SegNet: A Deep Convolutional Encoder-Decoder Architecture"
    """
    print("\n" + "="*80)
    print(f"  Calculating Class Weights (Method: {method})")
    print("="*80)
    
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    # Count pixels per class across entire dataset
    print("Analyzing class distribution...")
    for idx in tqdm(range(len(dataset)), desc="Processing labels"):
        _, label = dataset[idx]
        
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        # Count pixels for each class
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(label == class_id)
    
    # Total pixels (excluding ignore_index)
    total_pixels = np.sum(class_counts)
    
    # Print class distribution
    _print_class_distribution(dataset, class_counts, total_pixels)
    
    # Calculate weights based on selected method
    class_weights = _compute_weights(
        class_counts, total_pixels, num_classes, method, beta
    )
    
    # Normalize weights (mean = 1)
    class_weights = class_weights / np.mean(class_weights)
    
    # Print calculated weights
    _print_weight_statistics(dataset, class_weights)
    
    # Convert to torch tensor
    return torch.FloatTensor(class_weights).to(device)


def _compute_weights(class_counts, total_pixels, num_classes, method, beta):
    """
    Compute class weights using the specified method.
    
    Args:
        class_counts (np.array): Pixel counts per class
        total_pixels (float): Total number of pixels
        num_classes (int): Number of classes
        method (str): Weight calculation method
        beta (float): Beta parameter for effective_num
    
    Returns:
        np.array: Computed class weights
    """
    if method == 'inverse_freq':
        # Standard inverse frequency: weight = total / (num_classes * count)
        class_weights = total_pixels / (num_classes * class_counts + 1e-10)
        
    elif method == 'sqrt_inv_freq':
        # Square root inverse frequency (less extreme)
        freq = class_counts / total_pixels
        class_weights = 1.0 / (np.sqrt(freq) + 1e-10)
        
    elif method == 'effective_num':
        # Effective Number of Samples
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        effective_num = 1.0 - np.power(beta, class_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-10)
        
    elif method == 'median_freq':
        # Median Frequency Balancing
        # Paper: "SegNet: A Deep Convolutional Encoder-Decoder Architecture"
        class_freq = class_counts / total_pixels
        
        # Calculate median of non-zero frequencies
        non_zero_freq = class_freq[class_freq > 0]
        if len(non_zero_freq) > 0:
            median_freq = np.median(non_zero_freq)
        else:
            median_freq = 1.0
        
        # weight[c] = median_freq / freq[c]
        class_weights = median_freq / (class_freq + 1e-10)
        
        print(f"Median Frequency: {median_freq:.6f}")
        print(f"Raw weight range: [{np.min(class_weights):.4f}, {np.max(class_weights):.4f}]")
        print("-"*80 + "\n")
    
    else:
        raise ValueError(f"Unknown weight calculation method: {method}")
    
    return class_weights


def _print_class_distribution(dataset, class_counts, total_pixels):
    """Print class distribution statistics."""
    print("\n" + "-"*80)
    print("Class Distribution:")
    print("-"*80)
    print(f"{'ID':<4} {'Class Name':<25} {'Pixel Count':<15} {'Percentage':<12}")
    print("-"*80)
    
    if hasattr(dataset, 'class_names'):
        for i, (count, name) in enumerate(zip(class_counts, dataset.class_names)):
            percentage = (count / total_pixels) * 100
            print(f"{i:<4} {name:<25} {int(count):<15,} {percentage:>6.2f}%")
    else:
        for i, count in enumerate(class_counts):
            percentage = (count / total_pixels) * 100
            print(f"{i:<4} {'Class_' + str(i):<25} {int(count):<15,} {percentage:>6.2f}%")
    
    print("-"*80)
    print(f"Total Pixels: {int(total_pixels):,}")
    print("-"*80 + "\n")


def _print_weight_statistics(dataset, class_weights):
    """Print calculated weight statistics."""
    print("-"*80)
    print("Calculated Class Weights:")
    print("-"*80)
    print(f"{'ID':<4} {'Class Name':<25} {'Weight':<12} {'Relative Impact':<15}")
    print("-"*80)
    
    max_weight = np.max(class_weights)
    if hasattr(dataset, 'class_names'):
        for i, (weight, name) in enumerate(zip(class_weights, dataset.class_names)):
            impact_bar = '█' * int((weight / max_weight) * 20)
            print(f"{i:<4} {name:<25} {weight:>8.4f}    {impact_bar}")
    else:
        for i, weight in enumerate(class_weights):
            impact_bar = '█' * int((weight / max_weight) * 20)
            print(f"{i:<4} {'Class_' + str(i):<25} {weight:>8.4f}    {impact_bar}")
    
    print("-"*80)
    print(f"Weight Statistics:")
    print(f"  Mean: {np.mean(class_weights):.4f}")
    print(f"  Std:  {np.std(class_weights):.4f}")
    print(f"  Min:  {np.min(class_weights):.4f} (Class {np.argmin(class_weights)})")
    print(f"  Max:  {np.max(class_weights):.4f} (Class {np.argmax(class_weights)})")
    print("="*80 + "\n")


# ===== Weight Method Descriptions =====
"""
1. inverse_freq (default):
   - Most common method
   - Formula: weight = total_pixels / (num_classes * class_count)
   - Pros: Simple to implement, effective
   - Cons: Can assign extremely high weights to rare classes

2. sqrt_inv_freq:
   - Relaxed version of inverse_freq
   - Formula: weight = 1 / sqrt(frequency)
   - Pros: Prevents extreme weights, stable training
   - Cons: Limited effectiveness for severe imbalance

3. effective_num:
   - Proposed in Class-Balanced Loss paper
   - Formula: weight = (1 - beta) / (1 - beta^n)
   - Pros: Strong theoretical foundation, considers sample overlap
   - Cons: Requires beta tuning

4. median_freq:
   - Proposed in SegNet paper
   - Formula: weight = median_frequency / class_frequency
   - Pros: Balanced around median, less sensitive to outliers
   - Cons: Limited effect with few classes
   - Note: Original paper doesn't normalize weights
"""