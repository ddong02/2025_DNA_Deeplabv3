#!/usr/bin/env python3
"""
DNA2025 Dataset Visualization Script
독립적으로 실행 가능한 데이터셋 시각화 스크립트
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from tqdm import tqdm
import argparse

class DNA2025Visualizer:
    def __init__(self, data_root):
        self.data_root = data_root
        
        # Define class names and colors for DNA2025 dataset
        self.class_names = [
            '주행가능영역', '인도', '도로노면표시', '차선', '연석', '벽,울타리',
            '승용차', '트럭', '버스', '바이크,자전거', '기타차량', '보행자',
            '라이더', '교통용콘및봉', '기타수직물체', '건물', '교통표지', '교통신호', '기타'
        ]
        
        # Define color palette for visualization (RGB values)
        self.color_palette = np.array([
            [128, 64, 128],   # 0: 주행가능영역 - 보라색
            [244, 35, 232],   # 1: 인도 - 분홍색
            [70, 70, 70],     # 2: 도로노면표시 - 회색
            [102, 102, 156],  # 3: 차선 - 연보라
            [190, 153, 153],  # 4: 연석 - 베이지
            [153, 153, 153],  # 5: 벽,울타리 - 회색
            [0, 0, 142],      # 6: 승용차 - 진파랑
            [0, 0, 70],       # 7: 트럭 - 어두운파랑  
            [0, 60, 100],     # 8: 버스 - 청록색
            [0, 80, 100],     # 9: 바이크,자전거 - 청록색
            [0, 0, 230],      # 10: 기타차량 - 파랑
            [220, 20, 60],    # 11: 보행자 - 빨강
            [255, 0, 0],      # 12: 라이더 - 밝은빨강
            [250, 170, 30],   # 13: 교통용콘및봉 - 주황
            [220, 220, 0],    # 14: 기타수직물체 - 노랑
            [70, 130, 180],   # 15: 건물 - 강철파랑
            [220, 220, 220],  # 16: 교통표지 - 연회색
            [250, 170, 160],  # 17: 교통신호 - 연분홍
            [128, 128, 128],  # 18: 기타 - 회색
        ], dtype=np.uint8)
        
        # Load file paths
        self.load_file_paths()
    
    def load_file_paths(self):
        """Load all image and label file paths"""
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        
        # Training data from SemanticDataset_final
        train_base = os.path.join(self.data_root, "SemanticDataset_final")
        if os.path.exists(train_base):
            for cam in ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set2']:
                image_pattern = os.path.join(train_base, "image", "train", cam, "*.*")
                cam_images = sorted(glob(image_pattern))
                self.train_images.extend(cam_images)
                
                # Get corresponding label paths
                for img_path in cam_images:
                    label_path = self._get_label_path(img_path, train_base, 'train')
                    self.train_labels.append(label_path)
        
        # Test data from SemanticDatasetTest
        test_base = os.path.join(self.data_root, "SemanticDatasetTest")
        if os.path.exists(test_base):
            for test_set in ['set1', 'set3']:
                image_pattern = os.path.join(test_base, "image", "test", test_set, "*.*")
                test_images = sorted(glob(image_pattern))
                self.test_images.extend(test_images)
                
                # Get corresponding label paths
                for img_path in test_images:
                    label_path = self._get_label_path(img_path, test_base, 'test')
                    self.test_labels.append(label_path)
        
        print(f"Loaded {len(self.train_images)} training images")
        print(f"Loaded {len(self.test_images)} test images")
    
    def _get_label_path(self, image_path, base_dir, split):
        """Convert image path to corresponding label path"""
        # Replace 'image' with 'labelmap' in the path
        label_path = image_path.replace(
            os.path.join(base_dir, "image", split),
            os.path.join(base_dir, "labelmap", split)
        )
        
        # Handle different file naming conventions
        dir_name = os.path.dirname(label_path)
        file_name = os.path.basename(label_path)
        base_name, ext = os.path.splitext(file_name)
        
        # Check for cityscapes naming convention
        if "_leftImg8bit" in file_name:
            new_file_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            new_file_name = base_name + "_CategoryId.png"
        
        label_path = os.path.join(dir_name, new_file_name)
        
        # If the _CategoryId version doesn't exist, try the original filename
        if not os.path.exists(label_path):
            label_path = image_path.replace(
                os.path.join(base_dir, "image", split),
                os.path.join(base_dir, "labelmap", split)
            )
        
        return label_path
    
    def decode_target(self, mask):
        """Decode segmentation mask to RGB color image"""
        if isinstance(mask, np.ndarray):
            # Create RGB image
            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            for class_id in range(len(self.color_palette)):
                rgb_mask[mask == class_id] = self.color_palette[class_id]
            
            return rgb_mask
        return mask
    
    def create_class_legend(self, save_path="class_legend.png"):
        """Create a legend showing all classes and their colors"""
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Create color patches for each class
        for i, (class_name, color) in enumerate(zip(self.class_names, self.color_palette)):
            # Create a color patch
            rect = plt.Rectangle((0, len(self.class_names) - i - 1), 1, 0.8, 
                               facecolor=color/255.0, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add class ID and name
            ax.text(1.1, len(self.class_names) - i - 0.6, f"{i:2d}: {class_name}", 
                   fontsize=10, va='center')
        
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.5, len(self.class_names) - 0.5)
        ax.set_title('DNA2025 Dataset - Class Colors', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Class legend saved to {save_path}")
        plt.close()
    
    def visualize_sample(self, img_path, label_path, save_path):
        """Visualize a single sample with image and colored mask"""
        # Load original image and label
        try:
            img = Image.open(img_path).convert("RGB")
            if os.path.exists(label_path):
                label = Image.open(label_path).convert("L")
            else:
                print(f"Warning: Label not found for {img_path}")
                label = Image.new('L', img.size, 0)
            
            # Convert to numpy
            img_np = np.array(img)
            label_np = np.array(label)
            
            # Decode label to colors
            colored_label = self.decode_target(label_np)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original Image\n{os.path.basename(img_path)}')
            axes[0].axis('off')
            
            # Colored label
            axes[1].imshow(colored_label)
            axes[1].set_title(f'Ground Truth\n{os.path.basename(label_path)}')
            axes[1].axis('off')
            
            # Overlay
            overlay = img_np.copy().astype(np.float32)
            alpha = 0.6
            overlay = (overlay * (1 - alpha) + colored_label.astype(np.float32) * alpha).astype(np.uint8)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay (Image + GT)')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
            
            # Print class statistics
            unique_classes = np.unique(label_np)
            print(f"Classes present in this image:")
            for class_id in unique_classes:
                if class_id < len(self.class_names):
                    pixels = np.sum(label_np == class_id)
                    percentage = (pixels / label_np.size) * 100
                    print(f"  {class_id}: {self.class_names[class_id]} - {pixels} pixels ({percentage:.2f}%)")
            
            plt.close()
            return True
            
        except Exception as e:
            print(f"Error visualizing {img_path}: {e}")
            return False
    
    def visualize_random_samples(self, dataset_type='train', num_samples=5, save_dir="./visualizations"):
        """Visualize random samples from the dataset"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Select images and labels based on dataset type
        if dataset_type == 'train':
            images = self.train_images
            labels = self.train_labels
        else:
            images = self.test_images
            labels = self.test_labels
        
        if len(images) == 0:
            print(f"No {dataset_type} images found!")
            return
        
        print(f"\n=== Visualizing {num_samples} random {dataset_type} samples ===")
        
        # Create class legend
        legend_path = os.path.join(save_dir, "class_legend.png")
        self.create_class_legend(legend_path)
        
        # Select random samples
        total_samples = len(images)
        random_indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        successful_visualizations = 0
        for i, idx in enumerate(random_indices):
            print(f"\nVisualizing sample {i+1}/{num_samples} (index: {idx})")
            save_path = os.path.join(save_dir, f"{dataset_type}_sample_{i+1}_idx{idx}.png")
            
            if self.visualize_sample(images[idx], labels[idx], save_path):
                successful_visualizations += 1
        
        print(f"\nSuccessfully created {successful_visualizations}/{num_samples} visualizations")
        print(f"All visualizations saved to: {save_dir}")
    
    def check_data_integrity(self):
        """Check data integrity and missing files"""
        print("\n=== Data Integrity Check ===")
        
        # Check training data
        train_missing_images = [img for img in self.train_images if not os.path.exists(img)]
        train_missing_labels = [lbl for lbl in self.train_labels if not os.path.exists(lbl)]
        
        print(f"Training data:")
        print(f"  Total images: {len(self.train_images)}")
        print(f"  Missing images: {len(train_missing_images)}")
        print(f"  Missing labels: {len(train_missing_labels)}")
        
        # Check test data
        test_missing_images = [img for img in self.test_images if not os.path.exists(img)]
        test_missing_labels = [lbl for lbl in self.test_labels if not os.path.exists(lbl)]
        
        print(f"Test data:")
        print(f"  Total images: {len(self.test_images)}")
        print(f"  Missing images: {len(test_missing_images)}")
        print(f"  Missing labels: {len(test_missing_labels)}")
        
        # Show sample missing files
        if train_missing_images:
            print(f"\nSample missing training images:")
            for img in train_missing_images[:5]:
                print(f"  {img}")
        
        if train_missing_labels:
            print(f"\nSample missing training labels:")
            for lbl in train_missing_labels[:5]:
                print(f"  {lbl}")
        
        return {
            'train_total': len(self.train_images),
            'train_missing_images': len(train_missing_images),
            'train_missing_labels': len(train_missing_labels),
            'test_total': len(self.test_images),
            'test_missing_images': len(test_missing_images),
            'test_missing_labels': len(test_missing_labels)
        }

def main():
    parser = argparse.ArgumentParser(description='DNA2025 Dataset Visualization')
    parser.add_argument('--data_root', type=str, default='./datasets/data',
                        help='Path to dataset root directory')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'test', 'both'], 
                        default='both', help='Which dataset to visualize')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize per dataset')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--check_integrity', action='store_true',
                        help='Check data integrity before visualization')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = DNA2025Visualizer(args.data_root)
    
    # Check data integrity if requested
    if args.check_integrity:
        integrity_results = visualizer.check_data_integrity()
    
    # Create visualizations
    if args.dataset_type in ['train', 'both']:
        train_save_dir = os.path.join(args.save_dir, 'train')
        visualizer.visualize_random_samples('train', args.num_samples, train_save_dir)
    
    if args.dataset_type in ['test', 'both']:
        test_save_dir = os.path.join(args.save_dir, 'test')
        visualizer.visualize_random_samples('test', args.num_samples, test_save_dir)

if __name__ == '__main__':
    main()