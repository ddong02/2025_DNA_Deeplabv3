#!/usr/bin/env python3
"""
Test path matching for different camera types
"""

import os
from glob import glob

def test_path_matching():
    """Test path matching for different camera types"""
    print("🧪 Testing path matching for different camera types...")
    
    base_dir = "/mnt/c/Users/user/Desktop/eogus/dataset/2025dna/SemanticDataset_final"
    
    # Test cam0 (round pattern)
    print("\n📷 Testing cam0 (round pattern):")
    cam0_images = glob(os.path.join(base_dir, "image", "train", "cam0", "*.*"))
    if cam0_images:
        img_path = cam0_images[0]
        print(f"  Image: {os.path.basename(img_path)}")
        
        # Test current logic
        rel_path = os.path.relpath(img_path, os.path.join(base_dir, "image", "train"))
        if "_leftImg8bit" in rel_path:
            label_rel_path = rel_path.replace("_leftImg8bit", "_gtFine_CategoryId")
        else:
            label_rel_path = os.path.splitext(rel_path)[0] + '_CategoryId.png'
        
        label_path = os.path.join(base_dir, "labelmap", "train", label_rel_path)
        print(f"  Expected label: {os.path.basename(label_path)}")
        print(f"  Label exists: {os.path.exists(label_path)}")
    
    # Test set1 (Daeduk pattern)
    print("\n📷 Testing set1 (Daeduk pattern):")
    set1_images = glob(os.path.join(base_dir, "image", "train", "set1", "*.*"))
    if set1_images:
        img_path = set1_images[0]
        print(f"  Image: {os.path.basename(img_path)}")
        
        # Test current logic
        rel_path = os.path.relpath(img_path, os.path.join(base_dir, "image", "train"))
        if "_leftImg8bit" in rel_path:
            label_rel_path = rel_path.replace("_leftImg8bit", "_gtFine_CategoryId")
        else:
            label_rel_path = os.path.splitext(rel_path)[0] + '_CategoryId.png'
        
        label_path = os.path.join(base_dir, "labelmap", "train", label_rel_path)
        print(f"  Expected label: {os.path.basename(label_path)}")
        print(f"  Label exists: {os.path.exists(label_path)}")
    
    # Test actual label files
    print("\n📷 Actual label files in set1:")
    set1_labels = glob(os.path.join(base_dir, "labelmap", "train", "set1", "*.*"))
    if set1_labels:
        print(f"  First label: {os.path.basename(set1_labels[0])}")
        print(f"  Total labels: {len(set1_labels)}")

if __name__ == "__main__":
    test_path_matching()
