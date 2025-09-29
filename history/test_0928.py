from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np
import time
import statistics as stats

from torch.utils import data
from metrics import StreamSegMetrics
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from glob import glob
from torch.utils.data import Dataset

# Optional: thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. FLOPs calculation will be skipped.")
    print("Install with: pip install thop")

def get_argparser():
    parser = argparse.ArgumentParser(description='DeepLabV3+ Model Testing')

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='dna2025dataset',
                        choices=['voc', 'cityscapes', 'dna2025dataset'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, required=True,
                        help="number of classes")

    # Model Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Test Options
    parser.add_argument("--ckpt", required=True, type=str,
                        help="path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size for testing (default: 1)')
    parser.add_argument("--crop_size", type=int, default=1024,
                        help='crop size for testing')
    parser.add_argument("--save_results", action='store_true', default=False,
                        help="save test results to \"./test_results\"")
    parser.add_argument("--save_samples", action='store_true', default=False,
                        help="save sample images")
    parser.add_argument("--num_samples", type=int, default=10,
                        help='number of sample images to save (default: 10)')

    # Device Options
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    
    # Test data source
    parser.add_argument("--test_source", type=str, default='test',
                        choices=['test', 'val'], 
                        help="Use 'test' for SemanticDatasetTest or 'val' for validation split")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="validation split ratio (only used when test_source='val')")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed for validation split")

    parser.add_argument("--evaluate_performance", action='store_true', default=False,
                        help="Evaluate model performance (FLOPs, inference time, FPS)")
    parser.add_argument("--input_resolution", type=str, default="1080x1920", 
                        help="Input resolution for performance evaluation (HxW)")
    parser.add_argument("--speed_iterations", type=int, default=200,
                        help="Number of iterations for speed test")

    return parser


class ExtValidationTransform:
    def __init__(self, crop_size=[513, 513]):
        self.crop_size = crop_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image, label):
        # Resize to crop_size for validation
        image = F.resize(image, self.crop_size, Image.BILINEAR)
        label = F.resize(label, self.crop_size, Image.NEAREST)
        
        # Center crop if needed
        image = F.center_crop(image, self.crop_size)
        label = F.center_crop(label, self.crop_size)
        
        # To Tensor & Normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        
        return image, label


class DNA2025TestDataset(Dataset):
    def __init__(self, root_dir, crop_size, test_source='test', val_ratio=0.2, random_seed=1):
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.test_source = test_source
        
        # Define color palette for visualization
        self.color_palette = np.array([
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [220, 20, 60],
            [255, 0, 0], [250, 170, 30], [220, 220, 0], [70, 130, 180],
            [220, 220, 220], [250, 170, 160], [128, 128, 128]
        ], dtype=np.uint8)
        
        self.class_names = [
            'Drivable Area', 'Sidewalk', 'Road Marking', 'Lane', 'Curb', 'Wall/Fence',
            'Car', 'Truck', 'Bus', 'Bike/Bicycle', 'Other Vehicle', 'Pedestrian',
            'Rider', 'Traffic Cone/Pole', 'Other Vertical Object', 'Building', 
            'Traffic Sign', 'Traffic Light', 'Other'
        ]
        
        self.image_paths = []
        self.label_paths = []
        
        if test_source == 'test':
            # Use dedicated test set
            test_base = os.path.join(root_dir, "SemanticDatasetTest")
            for test_set in ['set1', 'set3']:
                image_pattern = os.path.join(test_base, "image", "test", test_set, "*.*")
                test_images = sorted(glob(image_pattern))
                self.image_paths.extend(test_images)
                
                for img_path in test_images:
                    label_path = self._get_test_label_path(img_path, test_base, test_set)
                    self.label_paths.append(label_path)
        
        elif test_source == 'val':
            # Use validation split from training data
            np.random.seed(random_seed)
            train_base = os.path.join(root_dir, "SemanticDataset_final")
            
            for cam in ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set2']:
                image_pattern = os.path.join(train_base, "image", "train", cam, "*.*")
                cam_images = sorted(glob(image_pattern))
                
                if len(cam_images) == 0:
                    continue
                
                # Get validation split
                num_images = len(cam_images)
                num_val = int(num_images * val_ratio)
                indices = np.random.permutation(num_images)
                val_indices = indices[:num_val]
                
                for idx in val_indices:
                    img_path = cam_images[idx]
                    label_path = self._get_label_path(img_path, train_base, 'train')
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
        
        self.transform = ExtValidationTransform(crop_size)
        
        # Validation
        assert len(self.image_paths) == len(self.label_paths), "Image/label count mismatch"
        assert len(self.image_paths) > 0, f"No images found for {test_source} data"
        
        print(f"Loaded {len(self.image_paths)} images from {test_source} set")

    def _get_test_label_path(self, image_path, test_base, test_set):
        """Convert test image path to corresponding label path"""
        # Extract filename from image path
        image_filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_filename)
        
        # For test dataset: leftImg8bit -> gtFine_CategoryId
        if "_leftImg8bit" in base_name:
            label_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            # Fallback: add _gtFine_CategoryId suffix
            label_name = base_name + "_gtFine_CategoryId.png"
        
        label_path = os.path.join(test_base, "labelmap", "test", test_set, label_name)
        return label_path

    def _get_label_path(self, image_path, train_base, split):
        """Convert training image path to corresponding label path"""
        # Extract filename and folder structure
        image_filename = os.path.basename(image_path)
        folder_structure = os.path.dirname(image_path).replace(
            os.path.join(train_base, "image", split), ""
        ).lstrip(os.sep)
        
        base_name, ext = os.path.splitext(image_filename)
        
        # For training dataset: leftImg8bit -> gtFine_CategoryId
        if "_leftImg8bit" in base_name:
            label_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            label_name = base_name + "_gtFine_CategoryId.png"
        
        label_path = os.path.join(train_base, "labelmap", split, folder_structure, label_name)
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
            return img, label.long()
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            raise


def compute_custom_miou(confusion, num_classes):
    """Compute mIoU from confusion matrix"""
    ious = []
    for cls in range(num_classes):
        TP = confusion[cls, cls]
        FP = confusion[:, cls].sum() - TP
        FN = confusion[cls, :].sum() - TP
        denom = TP + FP + FN
        if denom == 0:
            iou = float('nan')
        else:
            iou = TP / denom
        ious.append(iou)
    miou = np.nanmean(ious)
    return miou, ious


def evaluate_model_performance(model, device, opts, input_resolution="1080x1920", iterations=200):
    """Evaluate model performance: FLOPs, inference time, FPS"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Parse input resolution
    h, w = map(int, input_resolution.split('x'))
    print(f"Input resolution: {h}x{w}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1024**2:.3f} M")
    
    # Create dummy input
    torch.backends.cudnn.benchmark = True
    x = torch.randn(1, 3, h, w, device=device)
    
    # Calculate FLOPs
    if THOP_AVAILABLE:
        try:
            # Extract model name from options
            model_name = opts.model
            
            # Try to get num_classes from the loaded model
            try:
                if hasattr(model.module, 'classifier'):
                    if hasattr(model.module.classifier, '__getitem__'):
                        # For models with sequential classifier
                        num_classes = model.module.classifier[-1].out_channels
                    elif hasattr(model.module.classifier, 'out_channels'):
                        # For models with single layer classifier
                        num_classes = model.module.classifier.out_channels
                    else:
                        num_classes = opts.num_classes
                else:
                    num_classes = opts.num_classes
            except:
                num_classes = opts.num_classes
            
            print(f"Creating fresh model for FLOPs: {model_name}, classes: {num_classes}")
            
            # Create model using the network.modeling module
            flops_model = network.modeling.__dict__[model_name](
                num_classes=num_classes,
                output_stride=16
            ).to(device)
            flops_model.eval()
            
            print("Running FLOPs calculation...")
            flops, params = profile(flops_model, inputs=(x,), verbose=False)
            print(f'FLOPs: {flops/(1024*1024*1024):.2f} G')
            print(f'Parameters (thop): {params/(1024*1024):.2f} M')
            
            del flops_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
            print("Providing parameter-based estimation...")
            # Rough estimation: FLOPs ≈ 2 * params * input_pixels  
            estimated_flops = total_params * h * w * 2 / 1e9
            print(f'Estimated FLOPs: {estimated_flops:.2f} G (rough approximation)')
            flops = estimated_flops * 1e9
    else:
        flops = None
        print("FLOPs calculation skipped (thop not available)")
    
    # Ensure model is on correct device before memory test
    model.to(device)
    
    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    mem_used = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak GPU Memory: {mem_used:.2f} MB")
    
    # Speed test
    print(f"\nRunning speed test with {iterations} iterations...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Speed measurement
    results = []
    for run in range(9):  # 9 runs for statistics
        torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        torch.cuda.synchronize()
        t_end = time.time()
        
        elapsed = t_end - t_start
        latency_ms = (elapsed / iterations) * 1000
        results.append(latency_ms)
    
    mean_ms = sum(results) / len(results)
    median_ms = stats.median(results)
    fps_mean = 1000.0 / mean_ms
    fps_median = 1000.0 / median_ms
    
    print("\n========= Speed Results =========")
    print(f"Per-run latencies (ms): {[round(v, 3) for v in results]}")
    print(f"Mean: {mean_ms:.3f} ms (~{fps_mean:.2f} FPS)")
    print(f"Median: {median_ms:.3f} ms (~{fps_median:.2f} FPS)")
    
    performance_stats = {
        'total_params_M': total_params / 1024**2,
        'flops_G': flops/(1024*1024*1024) if flops else None,
        'memory_MB': mem_used,
        'latency_mean_ms': mean_ms,
        'latency_median_ms': median_ms,
        'fps_mean': fps_mean,
        'fps_median': fps_median,
        'input_resolution': input_resolution
    }
    
    return performance_stats


class SimpleSegMetrics:
    """Simple segmentation metrics implementation"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, targets, preds):
        """Update metrics with new predictions and targets"""
        # Flatten arrays
        targets = targets.flatten()
        preds = preds.flatten()
        
        # Remove ignore index (255)
        mask = targets != 255
        targets = targets[mask]
        preds = preds[mask]
        
        # Clip predictions to valid range
        preds = np.clip(preds, 0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)
        
        # Compute confusion matrix for this batch
        for t, p in zip(targets, preds):
            self.confusion_matrix[t, p] += 1
    
    def get_results(self):
        """Compute and return all metrics"""
        confusion = self.confusion_matrix
        
        # Per-class IoU
        ious = []
        for i in range(self.num_classes):
            TP = confusion[i, i]
            FP = confusion[:, i].sum() - TP
            FN = confusion[i, :].sum() - TP
            
            if TP + FP + FN == 0:
                iou = 0.0  # No samples for this class
            else:
                iou = TP / (TP + FP + FN)
            ious.append(iou)
        
        # Mean IoU
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        
        # Overall Accuracy
        total_correct = np.diag(confusion).sum()
        total_pixels = confusion.sum()
        overall_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
        
        # Frequency Weighted IoU
        freq = confusion.sum(axis=1) / confusion.sum()
        freq_weighted_iou = np.sum(freq * np.array(ious))
        
        return {
            'Mean IoU': mean_iou,
            'Class IoU': np.array(ious),
            'Overall Acc': overall_acc,
            'FreqW Acc': freq_weighted_iou
        }
    
    def to_str(self, results):
        """Convert results to string format"""
        string = f"Overall Acc\t {results['Overall Acc']:.3f}\n"
        string += f"Mean IoU\t {results['Mean IoU']:.3f}\n"
        string += f"FreqW Acc\t {results['FreqW Acc']:.3f}\n"
        
        # Class IoU details
        class_ious = results['Class IoU']
        string += f"Class IoU:\n"
        for i, iou in enumerate(class_ious):
            string += f"  Class {i}: {iou:.3f}\n"
        
        return string


def test_model(opts, model, loader, device, metrics, save_samples_ids=None):
    """Test the model and return results"""
    metrics.reset()
    ret_samples = []
    
    # Setup result directories
    if opts.save_results:
        results_dir = 'test_results'
        os.makedirs(results_dir, exist_ok=True)
        
        if opts.save_samples:
            os.makedirs(os.path.join(results_dir, 'comparisons'), exist_ok=True)
        
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_id = 0

    print("Running inference...")
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            
            # Collect samples for visualization
            if save_samples_ids is not None and i in save_samples_ids:
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

            # Save detailed results
            if opts.save_results and opts.save_samples:
                for batch_idx in range(len(images)):
                    image = images[batch_idx].detach().cpu().numpy()
                    target = targets[batch_idx]
                    pred = preds[batch_idx]

                    # Process image
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    
                    if hasattr(loader.dataset, 'decode_target'):
                        target_rgb = loader.dataset.decode_target(target).astype(np.uint8)
                        pred_rgb = loader.dataset.decode_target(pred).astype(np.uint8)
                    else:
                        target_rgb = target.astype(np.uint8)
                        pred_rgb = pred.astype(np.uint8)

                    # Create comparison image (원본 | 정답 | 예측)
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(image)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(target_rgb)
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_rgb)
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    
                    comparison_path = os.path.join(results_dir, 'comparisons', f'{img_id:05d}_comparison.png')
                    plt.tight_layout()
                    plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
                    plt.close()

                    img_id += 1

    # Get final scores
    score = metrics.get_results()
    
    # Save summary
    if opts.save_results:
        summary_path = os.path.join(results_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Test Results Summary\n")
            f.write("===================\n")
            f.write(f"Model: {opts.model}\n")
            f.write(f"Checkpoint: {opts.ckpt}\n")
            f.write(f"Dataset: {opts.dataset}\n")
            f.write(f"Test source: {opts.test_source}\n")
            f.write(f"Total images: {len(loader.dataset)}\n")
            f.write(f"Crop size: {opts.crop_size}\n\n")
            f.write(metrics.to_str(score))
            f.write(f"\n\nDetailed Metrics:\n")
            for key, value in score.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {str(value)}\n")
        
        # Save class-wise results
        if hasattr(loader.dataset, 'get_class_info'):
            class_info = loader.dataset.get_class_info()
            class_results_path = os.path.join(results_dir, 'class_results.txt')
            with open(class_results_path, 'w') as f:
                f.write("Class-wise IoU Results\n")
                f.write("=====================\n")
                class_ious = score['Class IoU']
                for i, (class_name, iou) in enumerate(zip(class_info['names'], class_ious)):
                    f.write(f"{i:2d}. {class_name:<25}: {iou:.6f}\n")
        
        print(f"Results saved to: {results_dir}")
    
    return score, ret_samples


def save_prediction_masks(opts, model, loader, device, results_dir):
    """Save prediction masks for custom mIoU evaluation"""
    print("Saving prediction masks...")
    
    # Create directories
    pred_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    img_id = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            
            # Save prediction masks
            for batch_idx in range(len(images)):
                pred = preds[batch_idx].astype(np.uint8)
                
                # Save as grayscale image
                pred_img = Image.fromarray(pred, mode='L')
                pred_path = os.path.join(pred_dir, f'{img_id:05d}_leftImg8bit.png')
                pred_img.save(pred_path)
                
                img_id += 1
    
    return pred_dir


def evaluate_custom_miou(pred_dir, label_dir, num_classes):
    """Evaluate mIoU using custom confusion matrix approach"""
    print("\n" + "="*60)
    print("CUSTOM mIOU EVALUATION")
    print("="*60)
    
    # Find prediction files
    pred_patterns = [
        "*_leftImg8bit.png",
        "*.png"
    ]
    
    pred_paths = []
    for pattern in pred_patterns:
        found_paths = glob(os.path.join(pred_dir, pattern))
        if found_paths:
            pred_paths = sorted(found_paths)
            print(f"Found {len(pred_paths)} files with pattern: {pattern}")
            break
    
    if not pred_paths:
        print("No prediction files found!")
        return None, None
    
    all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    processed_count = 0
    
    for pred_path in tqdm(pred_paths, desc="Computing confusion matrix"):
        # Extract file ID
        filename = os.path.basename(pred_path)
        if "_leftImg8bit.png" in filename:
            file_id = filename.replace("_leftImg8bit.png", "")
        else:
            file_id = os.path.splitext(filename)[0]
        
        # Find corresponding label
        possible_label_names = [
            f"{file_id}_gtFine_CategoryId.png",
            f"{file_id}_CategoryId.png",
            f"{file_id}.png"
        ]
        
        label_path = None
        for label_name in possible_label_names:
            potential_path = os.path.join(label_dir, label_name)
            if os.path.exists(potential_path):
                label_path = potential_path
                break
        
        if not label_path:
            continue
        
        try:
            # Load images
            pred = np.array(Image.open(pred_path)).astype(np.uint8)
            label = np.array(Image.open(label_path)).astype(np.uint8)
            
            # Handle different image formats
            if len(pred.shape) == 3:
                # If prediction is RGB, convert to grayscale or extract channel
                pred = pred[:, :, 0]  # Take first channel
            
            pred = pred.flatten()
            label = label.flatten()
            
            # Mask out invalid pixels
            mask = label != 255
            pred = pred[mask]
            label = label[mask]
            
            # Clip predictions to valid range
            pred = np.clip(pred, 0, num_classes - 1)
            label = np.clip(label, 0, num_classes - 1)
            
            # Compute confusion matrix for this image
            conf = confusion_matrix(label, pred, labels=list(range(num_classes)))
            all_confusion += conf
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if processed_count == 0:
        print("No valid image pairs found!")
        return None, None
    
    print(f"Processed {processed_count} image pairs")
    
    # Compute mIoU
    miou, ious = compute_custom_miou(all_confusion, num_classes)
    
    print(f"\nCustom mIoU: {miou:.4f}")
    print("\nClass-wise IoU:")
    for i, iou in enumerate(ious):
        if not np.isnan(iou):
            print(f"Class {i:2d}: {iou:.4f}")
        else:
            print(f"Class {i:2d}: NaN (no samples)")
    
    return miou, ious


def main():
    opts = get_argparser().parse_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test dataset
    if opts.dataset == 'dna2025dataset':
        test_dst = DNA2025TestDataset(
            root_dir=opts.data_root,
            crop_size=[opts.crop_size, opts.crop_size],
            test_source=opts.test_source,
            val_ratio=opts.val_ratio,
            random_seed=opts.random_seed
        )
    else:
        raise NotImplementedError(f"Dataset {opts.dataset} not implemented in test script")

    # Create data loader
    test_loader = data.DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=False, num_workers=2)

    # Setup model
    print(f"Loading model: {opts.model}")
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Load checkpoint
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {opts.ckpt}")
    
    print(f"Loading checkpoint: {opts.ckpt}")
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print("Model loaded successfully")

    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Setup metrics
    metrics = SimpleSegMetrics(opts.num_classes)

    # Determine sample indices for visualization
    save_samples_ids = None
    if opts.save_results and opts.save_samples:
        total_batches = len(test_loader)
        sample_interval = max(1, total_batches // opts.num_samples)
        save_samples_ids = list(range(0, total_batches, sample_interval))[:opts.num_samples]
        print(f"Will save {len(save_samples_ids)} sample images")

    # Performance evaluation
    performance_stats = None
    if opts.evaluate_performance:
        performance_stats = evaluate_model_performance(
            model, device, opts,
            input_resolution=opts.input_resolution,
            iterations=opts.speed_iterations
        )

    # Run testing
    print("="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)
    
    score, ret_samples = test_model(
        opts=opts,
        model=model,
        loader=test_loader,
        device=device,
        metrics=metrics,
        save_samples_ids=save_samples_ids
    )
    
    # Print results
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    print(metrics.to_str(score))
    
    # Print class-wise results with class names
    if hasattr(test_dst, 'get_class_info'):
        print("\nCLASS-WISE IoU:")
        print("-" * 40)
        class_info = test_dst.get_class_info()
        class_ious = score['Class IoU']
        for i, (class_name, iou) in enumerate(zip(class_info['names'], class_ious)):
            print(f"{i:2d}. {class_name:<25}: {iou:.4f}")
    else:
        print("\nCLASS-WISE IoU:")
        print("-" * 40)
        class_ious = score['Class IoU']
        for i, iou in enumerate(class_ious):
            print(f"{i:2d}. Class {i:<20}: {iou:.4f}")
    
    # Optional: Custom mIoU evaluation with saved prediction masks
    custom_miou = None
    custom_ious = None
    if opts.save_results:
        results_dir = 'test_results'
        
        # Save prediction masks for custom evaluation
        pred_dir = save_prediction_masks(opts, model, test_loader, device, results_dir)
        
        # Find corresponding label directory
        if opts.test_source == 'test':
            # For test set, use the test label directory
            label_base = os.path.join(opts.data_root, "SemanticDatasetTest", "labelmap", "test")
            # Try to find labels in set1 or set3 directories
            for test_set in ['set1', 'set3']:
                label_dir = os.path.join(label_base, test_set)
                if os.path.exists(label_dir) and len(os.listdir(label_dir)) > 0:
                    custom_miou, custom_ious = evaluate_custom_miou(pred_dir, label_dir, opts.num_classes)
                    break
        elif opts.test_source == 'val':
            # For validation set, this is more complex as labels are distributed across cam folders
            print("Custom mIoU evaluation for validation split is not implemented yet.")
            print("The validation data comes from multiple camera folders with different structures.")
    
    # Save comprehensive results
    if opts.save_results:
        results_dir = 'test_results'
        
        # Save comprehensive summary
        summary_path = os.path.join(results_dir, 'comprehensive_results.txt')
        with open(summary_path, 'w') as f:
            f.write("Comprehensive Test Results\n")
            f.write("=========================\n\n")
            
            # Model info
            f.write(f"Model: {opts.model}\n")
            f.write(f"Checkpoint: {opts.ckpt}\n")
            f.write(f"Dataset: {opts.dataset}\n")
            f.write(f"Test source: {opts.test_source}\n")
            f.write(f"Total images: {len(test_loader.dataset)}\n")
            f.write(f"Crop size: {opts.crop_size}\n\n")
            
            # Performance stats
            if performance_stats:
                f.write("Performance Statistics:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Parameters: {performance_stats['total_params_M']:.3f} M\n")
                if performance_stats['flops_G']:
                    f.write(f"FLOPs: {performance_stats['flops_G']:.2f} G\n")
                f.write(f"GPU Memory: {performance_stats['memory_MB']:.2f} MB\n")
                f.write(f"Inference Time (mean): {performance_stats['latency_mean_ms']:.3f} ms\n")
                f.write(f"Inference Time (median): {performance_stats['latency_median_ms']:.3f} ms\n")
                f.write(f"FPS (mean): {performance_stats['fps_mean']:.2f}\n")
                f.write(f"FPS (median): {performance_stats['fps_median']:.2f}\n")
                f.write(f"Input Resolution: {performance_stats['input_resolution']}\n\n")
            
            # Accuracy metrics
            f.write("Accuracy Metrics:\n")
            f.write("-" * 17 + "\n")
            f.write(metrics.to_str(score))
            f.write(f"\n\nDetailed Metrics:\n")
            for key, value in score.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {str(value)}\n")
            
            # Custom mIoU results
            if custom_miou is not None:
                f.write(f"\n\nCustom mIoU Evaluation:\n")
                f.write("-" * 23 + "\n")
                f.write(f"Custom mIoU: {custom_miou:.6f}\n")
                f.write("Custom Class-wise IoU:\n")
                for i, iou in enumerate(custom_ious):
                    if not np.isnan(iou):
                        f.write(f"  Class {i}: {iou:.6f}\n")
                    else:
                        f.write(f"  Class {i}: NaN (no samples)\n")
        
        print(f"Comprehensive results saved to: {summary_path}")
    
    # Final summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Mean IoU: {score['Mean IoU']:.4f}")
    print(f"Overall Accuracy: {score['Overall Acc']:.4f}")
    
    if custom_miou is not None:
        print(f"Custom mIoU: {custom_miou:.4f}")
    
    if performance_stats:
        print(f"Parameters: {performance_stats['total_params_M']:.3f} M")
        if performance_stats['flops_G']:
            print(f"FLOPs: {performance_stats['flops_G']:.2f} G")
        print(f"FPS (median): {performance_stats['fps_median']:.2f}")
        print(f"GPU Memory: {performance_stats['memory_MB']:.2f} MB")
    
    print("="*60)


if __name__ == "__main__":
    main()

# 1. 기본 테스트 + 성능 평가
# python my_test2.py \
#     --dataset dna2025dataset \
#     --data_root ./datasets/data \
#     --model deeplabv3_mobilenet \
#     --ckpt ./checkpoints/best_deeplabv3_mobilenet_dna2025dataset_os16.pth \
#     --num_classes 19 \
#     --crop_size 1024 \
#     --test_source test \
#     --save_results \
#     --save_samples \
#     --num_samples 20 \
#     --evaluate_performance \
#     --input_resolution 1080x1920

# 2. 빠른 성능 테스트 (결과 저장 없음)
# python my_test2.py \
#     --dataset dna2025dataset \
#     --data_root ./datasets/data \
#     --model deeplabv3_mobilenet \
#     --ckpt ./checkpoints/best_deeplabv3_mobilenet_dna2025dataset_os16.pth \
#     --num_classes 19 \
#     --crop_size 1024 \
#     --test_source test \
#     --evaluate_performance