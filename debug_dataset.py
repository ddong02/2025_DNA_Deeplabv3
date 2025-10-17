#!/usr/bin/env python3
"""
Debug script to check dataset loading
"""

import os
import sys
from glob import glob

def check_dataset_structure(data_root):
    """Check if dataset structure is correct"""
    print(f"🔍 Checking dataset structure in: {data_root}")
    
    base_dir = os.path.join(data_root, "SemanticDataset_final")
    print(f"📁 Base directory: {base_dir}")
    print(f"📁 Base directory exists: {os.path.exists(base_dir)}")
    
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return False
    
    # Check for image and labelmap directories
    image_dir = os.path.join(base_dir, "image")
    label_dir = os.path.join(base_dir, "labelmap")
    
    print(f"📁 Image directory: {image_dir} (exists: {os.path.exists(image_dir)})")
    print(f"📁 Labelmap directory: {label_dir} (exists: {os.path.exists(label_dir)})")
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"❌ Missing image or labelmap directory")
        return False
    
    # Check for train/val splits
    for split in ['train', 'val', 'test']:
        split_image_dir = os.path.join(image_dir, split)
        split_label_dir = os.path.join(label_dir, split)
        
        print(f"📁 {split} image directory: {split_image_dir} (exists: {os.path.exists(split_image_dir)})")
        print(f"📁 {split} labelmap directory: {split_label_dir} (exists: {os.path.exists(split_label_dir)})")
        
        if os.path.exists(split_image_dir):
            # Check camera folders
            cam_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set1', 'set2', 'set3']
            total_images = 0
            
            for cam in cam_folders:
                cam_dir = os.path.join(split_image_dir, cam)
                if os.path.exists(cam_dir):
                    images = glob(os.path.join(cam_dir, "*.*"))
                    print(f"  📷 {cam}: {len(images)} images")
                    total_images += len(images)
                else:
                    print(f"  📷 {cam}: directory not found")
            
            print(f"  📊 Total {split} images: {total_images}")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_dataset.py <data_root>")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    if not os.path.exists(data_root):
        print(f"❌ Data root not found: {data_root}")
        sys.exit(1)
    
    success = check_dataset_structure(data_root)
    
    if success:
        print("✅ Dataset structure looks good!")
    else:
        print("❌ Dataset structure has issues!")

if __name__ == "__main__":
    main()
