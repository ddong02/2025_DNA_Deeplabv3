"""
Early stopping utility for training.
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve.
    
    Args:
        patience (int): How many epochs to wait after last improvement. Default: 10
        min_delta (float): Minimum change to qualify as improvement. Default: 0.0001
        mode (str): 'max' for metrics to maximize (accuracy, IoU), 
                   'min' for metrics to minimize (loss). Default: 'max'
        verbose (bool): Print messages. Default: True
    """
    
    def __init__(self, patience=10, min_delta=0.0001, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf if mode == 'min' else -np.inf
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, val_score, epoch):
        """
        Check if training should stop.
        
        Args:
            val_score (float): Current validation score
            epoch (int): Current epoch number
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        score = val_score
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            if self.verbose:
                print(f"[Early Stopping] Baseline set: {score:.4f}")
            return False
        
        # Check if improved
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:  # mode == 'min'
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            # Improvement detected
            if self.verbose:
                improvement = abs(score - self.best_score)
                print(f"[Early Stopping] Improved from {self.best_score:.4f} to {score:.4f} "
                      f"(+{improvement:.4f}). Counter reset.")
            self.best_score = score
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"[Early Stopping] No improvement for {self.counter}/{self.patience} epochs. "
                      f"Best: {self.best_score:.4f}, Current: {score:.4f}")
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"[Early Stopping] Triggered at epoch {epoch}!")
                    print(f"Best score was {self.best_score:.4f}, achieved {self.patience} epochs ago.")
                    print(f"{'='*80}\n")
                self.early_stop = True
                return True
            
            return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf if self.mode == 'min' else -np.inf
    
    def set_baseline(self, score):
        """
        Set initial baseline score without triggering early stop counter.
        Useful for starting monitoring from a specific point (e.g., after Stage 1).
        
        Args:
            score (float): Baseline score to set
        """
        self.best_score = score
        self.counter = 0
        self.early_stop = False