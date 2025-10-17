# ============================================================================
# File: my_utils/dna2025_dataset_combined.py
# ============================================================================
"""
DNA2025 Combined Dataset implementation for semantic segmentation.
Combines train and validation datasets for full dataset training.

This dataset class loads both train and val data as training data,
allowing for full dataset utilization.
"""

import os
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class DNA2025CombinedDataset(Dataset):
    """DNA2025 Combined Dataset - includes both train and val data for training"""
    
    # Class names for the dataset
    CLASS_NAMES = [
        'Drivable Area', 'Sidewalk', 'Road Marking', 'Lane', 'Curb', 'Wall/Fence',
        'Car', 'Truck', 'Bus', 'Bike/Bicycle', 'Other Vehicle', 'Pedestrian',
        'Rider', 'Traffic Cone/Pole', 'Other Vertical Object', 'Building', 
        'Traffic Sign', 'Traffic Light', 'Other'
    ]
    
    # Color palette for visualization (RGB)
    COLOR_PALETTE = np.array([
        [128, 64, 128],   # 0: Drivable Area
        [244, 35, 232],   # 1: Sidewalk
        [70, 70, 70],     # 2: Road Marking
        [102, 102, 156],  # 3: Lane
        [190, 153, 153],  # 4: Curb
        [153, 153, 153],  # 5: Wall/Fence
        [0, 0, 142],      # 6: Car
        [0, 0, 70],       # 7: Truck
        [0, 60, 100],     # 8: Bus
        [0, 80, 100],     # 9: Bike/Bicycle
        [0, 0, 230],      # 10: Other Vehicle
        [220, 20, 60],    # 11: Pedestrian
        [255, 0, 0],      # 12: Rider
        [250, 170, 30],   # 13: Traffic Cone/Pole
        [220, 220, 0],    # 14: Other Vertical Object
        [70, 130, 180],   # 15: Building
        [220, 220, 220],  # 16: Traffic Sign
        [255, 255, 0],    # 17: Traffic Light
        [128, 128, 128],  # 18: Other
    ], dtype=np.uint8)
    
    def __init__(self, root_dir, crop_size, subset, scale_range=None, random_seed=1, subset_ratio=1.0,
                 horizontal_flip_p=0.5, brightness_limit=0.2, contrast_limit=0.2, rotation_limit=10,
                 custom_transform=None, combine_train_val=False):
        """
        Args:
            root_dir: Root directory of the dataset
            crop_size: Size for cropping [height, width]
            subset: 'train', 'val', or 'test'
            scale_range: Range for random scaling (only for training)
            random_seed: Random seed for reproducibility
            subset_ratio: Ratio of dataset to use (0.0-1.0, default: 1.0 for full dataset)
            horizontal_flip_p: Probability of horizontal flip (0.0-1.0)
            brightness_limit: Brightness adjustment limit (0.0-1.0)
            contrast_limit: Contrast adjustment limit (0.0-1.0)
            rotation_limit: Rotation angle limit in degrees
            custom_transform: Custom transform (e.g., Albumentations)
            combine_train_val: If True, combine train and val data for training
        """
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.subset = subset
        self.class_names = self.CLASS_NAMES
        self.color_palette = self.COLOR_PALETTE
        self.combine_train_val = combine_train_val
        
        # Only set seed for subset sampling, not for augmentation
        if subset_ratio < 1.0:
            np.random.seed(random_seed)
        
        base_dir = os.path.join(root_dir, "SemanticDataset_final")
        self.image_paths = []
        self.label_paths = []
        
        cam_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 
                       'set1', 'set2', 'set3']
        
        print(f"\nLoading {subset} data...")
        
        if combine_train_val and subset == 'train':
            # Load both train and val data for training
            print("🔄 Combining train and validation datasets for full training...")
            
            for split_dir in ['train', 'val']:
                print(f"  Loading from '{split_dir}' directory...")
                split_count = 0
                
                for cam in cam_folders:
                    image_pattern = os.path.join(base_dir, "image", split_dir, cam, "*.*")
                    cam_images = sorted(glob(image_pattern))
                    
                    if len(cam_images) == 0:
                        continue
                    
                    for img_path in cam_images:
                        label_path = self._get_label_path(img_path, base_dir, split_dir)
                        
                        if os.path.exists(label_path):
                            self.image_paths.append(img_path)
                            self.label_paths.append(label_path)
                            split_count += 1
                    
                    print(f"    {cam}: {len(cam_images)} images loaded")
                
                print(f"  Total from {split_dir}: {split_count} images")
        else:
            # Normal loading (train/val/test split)
            if subset == 'train':
                split_dir = 'train'
            elif subset == 'val':
                split_dir = 'val'
            elif subset == 'test':
                split_dir = 'test'
            else:
                raise ValueError(f"Unknown subset: {subset}. Use 'train', 'val', or 'test'")
            
            print(f"Loading from '{split_dir}' directory...")
            
            for cam in cam_folders:
                image_pattern = os.path.join(base_dir, "image", split_dir, cam, "*.*")
                cam_images = sorted(glob(image_pattern))
                
                if len(cam_images) == 0:
                    continue
                
                for img_path in cam_images:
                    label_path = self._get_label_path(img_path, base_dir, split_dir)
                    
                    if os.path.exists(label_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)
                
                print(f"  {cam}: {len(cam_images)} images loaded")
        
        # Apply subset ratio if specified
        if subset_ratio < 1.0:
            original_count = len(self.image_paths)
            subset_size = int(len(self.image_paths) * subset_ratio)
            subset_size = max(1, subset_size)  # Ensure at least 1 sample
            
            # Randomly sample subset
            indices = np.random.choice(len(self.image_paths), subset_size, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.label_paths = [self.label_paths[i] for i in indices]
            
            print(f"Applied subset ratio {subset_ratio:.1%}: {original_count} → {len(self.image_paths)} images")
        
        # Validation
        assert len(self.image_paths) == len(self.label_paths), \
            f"Image/label count mismatch: {len(self.image_paths)} vs {len(self.label_paths)}"
        assert len(self.image_paths) > 0, \
            f"No images found for {subset} subset"
        
        print(f"✅ Total {subset} images loaded: {len(self.image_paths)}")
        
        # Set transforms
        if custom_transform is not None:
            self.transform = custom_transform
        else:
            if subset == 'train':
                from my_utils.albumentations_augmentation import AlbumentationsSegmentationTransform
                self.transform = AlbumentationsSegmentationTransform(
                    crop_size=[crop_size, crop_size],
                    scale_range=scale_range or [0.75, 1.25],
                    horizontal_flip_p=horizontal_flip_p,
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    rotation_limit=rotation_limit
                )
            else:
                from my_utils.albumentations_augmentation import AlbumentationsValidationTransform
                self.transform = AlbumentationsValidationTransform(
                    crop_size=[crop_size, crop_size]
                )
    
    def _get_label_path(self, image_path, base_dir, split_dir):
        """Get corresponding label path for an image"""
        # Extract relative path from image path
        rel_path = os.path.relpath(image_path, os.path.join(base_dir, "image", split_dir))
        
        # Replace image extension with label extension
        label_rel_path = os.path.splitext(rel_path)[0] + '.png'
        
        # Construct label path
        label_path = os.path.join(base_dir, "label", split_dir, label_rel_path)
        
        return label_path
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get item by index"""
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image and label
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image, label = self.transform(image, label)
        
        # Convert label to tensor if not already
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(np.array(label)).long()
        
        return image, label
    
    def decode_target(self, target):
        """Decode target tensor to RGB image for visualization"""
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        
        # Ensure target is in valid range
        target = np.clip(target, 0, len(self.color_palette) - 1)
        
        # Map to colors
        colored_target = self.color_palette[target]
        
        return colored_target.astype(np.uint8)
    
    def get_class_weights(self, method='inverse_freq', beta=0.9999, ignore_index=255):
        """Calculate class weights for the dataset"""
        from my_utils.calculate_class_weights import calculate_class_weights
        return calculate_class_weights(self, len(self.CLASS_NAMES), 'cpu', method, beta, ignore_index)
