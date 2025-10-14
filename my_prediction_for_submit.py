import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import matplotlib.cm as cm

# Import network module for DeepLabV3+ models
import network
import utils


class TestTransform:
    """Transform for test dataset - preserves original image size"""
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image):
        # To Tensor & Normalize only (no resize/crop)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        return image


class TestSegmentationDataset(Dataset):
    def __init__(self, root_dir, subset='test'):
        self.image_dir = os.path.join(root_dir, "image", subset)
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*", "*.*"), recursive=True))
        self.transform = TestTransform()
        
        # Define color palette for visualization (same as DNA2025TestDataset)
        self.color_palette = np.array([
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
            [70, 130, 180],   # 14: Building
            [220, 220, 0],    # 15: Traffic Sign
            [250, 170, 160],  # 16: Traffic Light
            [220, 220, 220],  # 17: Other Vertical Object
            [128, 128, 128]   # 18: Other
        ], dtype=np.uint8)
        
        self.class_names = [
            'Drivable Area',           # 0
            'Sidewalk',                # 1
            'Road Marking',            # 2
            'Lane',                    # 3
            'Curb',                    # 4
            'Wall/Fence',              # 5
            'Car',                     # 6
            'Truck',                   # 7
            'Bus',                     # 8
            'Bike/Bicycle',            # 9
            'Other Vehicle',           # 10
            'Pedestrian',              # 11
            'Rider',                   # 12
            'Traffic Cone/Pole',       # 13
            'Building',                # 14
            'Traffic Sign',            # 15
            'Traffic Light',           # 16
            'Other Vertical Object',   # 17
            'Other'                    # 18
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Store original size for later use
        original_size = img.size  # (width, height)
        
        tensor = self.transform(img)
        return tensor, img_path, original_size
    
    def decode_target(self, mask):
        """Decode segmentation mask to RGB color image"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask = np.clip(mask, 0, len(self.color_palette) - 1)
        return self.color_palette[mask]


def load_deeplabv3_mobilenet(weight_path, num_classes, device, output_stride=16, separable_conv=False):
    """
    Load DeepLabV3 MobileNet model using network.modeling module
    """
    print("Loading DeepLabV3 MobileNet model")
    
    # Create model using network.modeling
    model = network.modeling.__dict__['deeplabv3_mobilenet'](
        num_classes=num_classes, 
        output_stride=output_stride
    )
    
    # Apply separable conv if requested
    if separable_conv:
        network.convert_to_separable_conv(model.classifier)
    
    # Set batch norm momentum
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Load checkpoint
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
    
    print(f"Loading checkpoint: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Handle different checkpoint formats
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
    
    print("Model loaded successfully")
    
    # Wrap in DataParallel
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    return model


def save_prediction(pred, save_path, colormap_root, args, original_size, dataset):
    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
    
    # Resize prediction to original image size
    pred_img = Image.fromarray(pred_np)
    pred_img = pred_img.resize(original_size, Image.NEAREST)
    pred_np_resized = np.array(pred_img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred_img.save(save_path)

    # Use color_palette for colormap (수정된 부분)
    colored_mask = dataset.decode_target(pred_np_resized)
    rgb_img = Image.fromarray(colored_mask)

    rel_path = os.path.relpath(save_path, start=os.path.join(args.result_dir, "label"))
    cmap_path = os.path.join(colormap_root, rel_path)
    os.makedirs(os.path.dirname(cmap_path), exist_ok=True)
    rgb_img.save(cmap_path)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset without crop_size argument
    dataset = TestSegmentationDataset(
        args.dataset_dir, 
        subset='test'
    )
    print(f"Found {len(dataset)} test images")
    print(f"Processing images at original resolution (no resizing/cropping)")
    print(f"Normalization: mean={[0.485, 0.456, 0.406]}, std={[0.229, 0.224, 0.225]}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load DeepLabV3 MobileNet model
    model = load_deeplabv3_mobilenet(
        args.weight_path, 
        args.num_classes, 
        device,
        output_stride=args.output_stride,
        separable_conv=args.separable_conv
    )

    colormap_root = os.path.join(args.result_dir, "colormap")
    
    print("Starting inference...")
    for img_tensor, img_path, original_size in tqdm(dataloader, desc="Predicting"):
        img_tensor = img_tensor.to(device)
        original_size = (original_size[0].item(), original_size[1].item())  # (width, height)

        with torch.no_grad():
            output = model(img_tensor)
            
            # Handle different output formats
            if isinstance(output, dict):
                # DeepLabV3 from torchvision returns a dict with 'out' key
                output = output['out']
            # DeepLabV3+ from network.modeling returns tensor directly
            
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)

        # Save prediction
        rel_path = os.path.relpath(img_path[0], os.path.join(args.dataset_dir, "image"))
        save_path = os.path.join(args.result_dir, "label", rel_path)

        save_prediction(pred, save_path, colormap_root, args, original_size, dataset)
    
    print(f"\nInference complete! Results saved to: {args.result_dir}")
    print(f"  - Grayscale predictions: {os.path.join(args.result_dir, 'label')}")
    print(f"  - Colormap visualizations: {colormap_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLabV3 MobileNet Segmentation Testing")
    parser.add_argument("--dataset_dir", type=str, default="./SemanticDatasetTest", 
                        help="Path to test images folder")
    parser.add_argument("--weight_path", type=str, required=True,
                        help="Path to model weight (.pth)")
    parser.add_argument("--result_dir", type=str, default="./result", 
                        help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=19, 
                        help="Number of segmentation classes")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16],
                        help="Output stride for DeepLabV3+ model")
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="Apply separable conv to decoder and ASPP")

    args = parser.parse_args()
    test(args)

# python my_prediction_for_submit.py \
#     --dataset_dir /mnt/c/Users/user/Desktop/eogus/dataset/2025dna/SemanticDatasetTest \
#     --weight_path checkpoints/baseline/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
#     --result_dir ./results_deeplabv3_baseline