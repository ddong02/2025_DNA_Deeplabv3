#!/usr/bin/env python3
"""
Test dataset loading to verify image-label pairs
"""

import os
import sys
from glob import glob

def test_dataset_loading():
    """Test if dataset can be loaded correctly"""
    print("🧪 Testing dataset loading...")
    
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from my_utils.dna2025_dataset_combined import DNA2025CombinedDataset
        
        # Test with very small subset
        dataset = DNA2025CombinedDataset(
            root_dir="/mnt/c/Users/user/Desktop/eogus/dataset/2025dna",
            crop_size=1024,
            subset='train',
            scale_range=[0.75, 1.25],
            random_seed=1,
            subset_ratio=0.001,  # Very small subset for testing
            combine_train_val=True
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total samples: {len(dataset)}")
        print(f"📊 Train samples: {dataset.train_samples}")
        print(f"📊 Val samples: {dataset.val_samples}")
        
        # Test loading a sample
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"✅ Sample loaded successfully!")
            print(f"   Image shape: {image.shape if hasattr(image, 'shape') else type(image)}")
            print(f"   Label shape: {label.shape if hasattr(label, 'shape') else type(label)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("🎉 Dataset loading test passed!")
    else:
        print("💥 Dataset loading test failed!")
        sys.exit(1)
