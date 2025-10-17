# ============================================================================
# File: datasets/dna2025_dataset.py
# ============================================================================
"""
DNA2025 Dataset implementation for semantic segmentation.
Includes custom transforms for training and validation.
"""

import os
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class ExtSegmentationTransform:
    """Transform for training: random scale, crop, brightness/contrast, rotation, flip, normalize"""
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.5, 1.5], 
                 horizontal_flip_p=0.5, brightness_limit=0.2, contrast_limit=0.2, 
                 rotation_limit=10):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.horizontal_flip_p = horizontal_flip_p
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.rotation_limit = rotation_limit
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image, label):
        # Random scale
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        image = F.resize(image, (new_height, new_width), Image.BILINEAR)
        label = F.resize(label, (new_height, new_width), Image.NEAREST)
        
        # Pad if needed
        pad_h = max(self.crop_size[0] - new_height, 0)
        pad_w = max(self.crop_size[1] - new_width, 0)
        
        if pad_h > 0 or pad_w > 0:
            padding = (0, 0, pad_w, pad_h)
            image = F.pad(image, padding, fill=0)
            label = F.pad(label, padding, fill=255)
        
        # Random crop
        if image.size[0] >= self.crop_size[1] and image.size[1] >= self.crop_size[0]:
            w, h = image.size
            th, tw = self.crop_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            
            image = F.crop(image, i, j, th, tw)
            label = F.crop(label, i, j, th, tw)
        
        # NEW: Brightness & Contrast augmentation (only for image)
        if random.random() > 0.5:
            brightness_factor = random.uniform(1-self.brightness_limit, 1+self.brightness_limit)
            contrast_factor = random.uniform(1-self.contrast_limit, 1+self.contrast_limit)
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)
            # Note: label is not affected by brightness/contrast changes
        
        # NEW: Small rotation (both image and label)
        if random.random() > 0.5:
            angle = random.uniform(-self.rotation_limit, self.rotation_limit)
            image = F.rotate(image, angle, fill=0)  # Black fill for image
            label = F.rotate(label, angle, fill=255)  # Ignore index for label
        
        # Random horizontal flip
        if random.random() < self.horizontal_flip_p:
            image = F.hflip(image)
            label = F.hflip(label)
        
        # To Tensor & Normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        
        return image, label


class ExtValidationTransform:
    """Transform for validation: resize, center crop, normalize"""
    def __init__(self, crop_size=[1024, 1024]):
        self.crop_size = crop_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image, label):
        image = F.resize(image, self.crop_size, Image.BILINEAR)
        label = F.resize(label, self.crop_size, Image.NEAREST)
        
        image = F.center_crop(image, self.crop_size)
        label = F.center_crop(label, self.crop_size)
        
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        
        return image, label


class DNA2025Dataset(Dataset):
    """DNA2025 Semantic Segmentation Dataset"""
    
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
        [250, 170, 160],  # 17: Traffic Light
        [128, 128, 128],  # 18: Other
    ], dtype=np.uint8)
    
    def __init__(self, root_dir, crop_size, subset, scale_range=None, random_seed=1, subset_ratio=1.0,
                 horizontal_flip_p=0.5, brightness_limit=0.2, contrast_limit=0.2, rotation_limit=10):
        """
        Args:
            root_dir: Root directory of the dataset
            crop_size: Size for cropping [height, width]
            subset: 'train', 'val', or 'test'
            scale_range: Range for random scaling (only for training)
            random_seed: Random seed for reproducibility
            subset_ratio: Ratio of dataset to use (0.0-1.0, default: 1.0 for full dataset)
        """
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.subset = subset
        self.class_names = self.CLASS_NAMES
        self.color_palette = self.COLOR_PALETTE
        
        # Only set seed for subset sampling, not for augmentation
        if subset_ratio < 1.0:
            np.random.seed(random_seed)
        
        base_dir = os.path.join(root_dir, "SemanticDataset_final")
        self.image_paths = []
        self.label_paths = []
        
        cam_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 
                       'set1', 'set2', 'set3']
        
        # Map subset to directory
        if subset == 'train':
            split_dir = 'train'
        elif subset == 'val':
            split_dir = 'val'
        elif subset == 'test':
            split_dir = 'test'
        else:
            raise ValueError(f"Unknown subset: {subset}. Use 'train', 'val', or 'test'")
        
        print(f"\nLoading {subset} data from '{split_dir}' directory...")
        
        # Load image and label paths
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
                else:
                    print(f"Warning: Label not found for {img_path}")
            
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
            f"No images found for {subset} subset in {split_dir} directory"
        
        # Set transforms
        if subset == 'train' and scale_range is not None:
            self.transform = ExtSegmentationTransform(
                crop_size, scale_range, 
                horizontal_flip_p, brightness_limit, contrast_limit, rotation_limit
            )
        else:
            self.transform = ExtValidationTransform(crop_size)
        
        print(f"Successfully loaded {len(self.image_paths)} {subset} images\n")

    def _get_label_path(self, image_path, base_dir, split):
        """Convert image path to corresponding label path"""
        label_path = image_path.replace(
            os.path.join(base_dir, "image", split),
            os.path.join(base_dir, "labelmap", split)
        )
        
        dir_name = os.path.dirname(label_path)
        file_name = os.path.basename(label_path)
        base_name, ext = os.path.splitext(file_name)
        
        if "_leftImg8bit" in file_name:
            new_file_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            new_file_name = base_name + "_CategoryId.png"
        
        label_path = os.path.join(dir_name, new_file_name)
        
        if not os.path.exists(label_path):
            label_path = image_path.replace(
                os.path.join(base_dir, "image", split),
                os.path.join(base_dir, "labelmap", split)
            )
        
        return label_path

    def decode_target(self, mask):
        """Decode segmentation mask to RGB color image"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask = np.clip(mask, 0, len(self.color_palette) - 1)
        return self.color_palette[mask]

    def get_class_info(self):
        """Return class information"""
        return {
            'names': self.class_names,
            'colors': self.color_palette,
            'num_classes': len(self.class_names)
        }
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label_path = self.label_paths[idx]
            
            img = Image.open(img_path).convert("RGB")
            
            if os.path.exists(label_path):
                label = Image.open(label_path).convert("L")
            else:
                print(f"Warning: Label not found for {img_path}")
                label = Image.new('L', img.size, 0)
            
            img, label = self.transform(img, label)
            
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(np.array(label, dtype=np.uint8))
            
            return img, label.long()
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Label path: {self.label_paths[idx]}")
            raise