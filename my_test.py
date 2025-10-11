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

# WandB for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. WandB logging will be skipped.")
    print("Install with: pip install wandb")

# Optional: thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. FLOPs calculation will be skipped.")
    print("Install with: pip install thop")

def get_argparser():
    parser = argparse.ArgumentParser(description='DeepLabV3 Model Testing')

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/SemanticDataset_final',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='dna2025dataset',
                        choices=['dna2025dataset'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes")

    # Model Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Test Options
    parser.add_argument("--ckpt", required=True, type=str,
                        help="path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size for testing (default: 1)')
    parser.add_argument("--save_results", action='store_true', default=False,
                        help="save test results")
    parser.add_argument("--output_dir", type=str, default='test_results',
                        help='directory to save test results (default: test_results)')
    parser.add_argument("--save_samples", action='store_true', default=False,
                        help="save sample images")
    parser.add_argument("--num_samples", type=int, default=10,
                        help='number of sample images to save (default: 10)')

    # Device Options
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    # Performance Evaluation Options
    parser.add_argument("--evaluate_performance", action='store_true', default=False,
                        help="Evaluate model performance (FLOPs, inference time, FPS)")
    parser.add_argument("--input_resolution", type=str, default="1080x1920", 
                        help="Input resolution for performance evaluation (HxW)")
    parser.add_argument("--speed_iterations", type=int, default=200,
                        help="Number of iterations for speed test")

    # WandB Options
    parser.add_argument("--use_wandb", action='store_true', default=False,
                        help="Log test results to WandB")
    parser.add_argument("--wandb_project", type=str, default='deeplabv3-semantic-segmentation',
                        help='WandB project name')
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help='WandB run ID to resume (if you want to add test results to existing training run)')
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help='WandB run name (default: auto-generated with "-test" suffix)')
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help='Comma-separated tags (e.g., "baseline,test")')

    return parser


class TestTransform:
    """Transform for test dataset - preserves original image size"""
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image, label):
        # To Tensor & Normalize only (no resize/crop)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        
        return image, label


class DNA2025TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Define color palette for visualization (19 classes)
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
        
        # Class names in English
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
        
        self.image_paths = []
        self.label_paths = []
        
        # Load test images from image/test folder and labels from labelmap/test folder
        image_base = os.path.join(root_dir, "image", "test")
        labelmap_base = os.path.join(root_dir, "labelmap", "test")
        
        if not os.path.exists(image_base):
            raise FileNotFoundError(f"Image test directory not found: {image_base}")
        if not os.path.exists(labelmap_base):
            raise FileNotFoundError(f"Labelmap test directory not found: {labelmap_base}")
        
        # Find all subdirectories in image/test folder
        image_subdirs = [d for d in os.listdir(image_base) 
                        if os.path.isdir(os.path.join(image_base, d))]
        
        print(f"Found test subdirectories: {image_subdirs}")
        
        # Load all images from all subdirectories
        for subdir in image_subdirs:
            image_dir = os.path.join(image_base, subdir)
            image_pattern = os.path.join(image_dir, "*.*")
            images = sorted(glob(image_pattern))
            
            # Filter for valid image extensions
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            images = [img for img in images if os.path.splitext(img)[1].lower() in valid_extensions]
            
            print(f"Found {len(images)} images in {subdir}")
            
            for img_path in images:
                label_path = self._get_label_path(img_path, subdir, labelmap_base)
                self.image_paths.append(img_path)
                self.label_paths.append(label_path)
        
        self.transform = TestTransform()
        
        # Validation
        assert len(self.image_paths) == len(self.label_paths), "Image/label count mismatch"
        assert len(self.image_paths) > 0, "No test images found"
        
        print(f"Total test images loaded: {len(self.image_paths)}")

    def _get_label_path(self, image_path, subdir, labelmap_base):
        """Convert image path to corresponding label path"""
        # Extract filename from image path
        image_filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_filename)
        
        # Remove _leftImg8bit suffix if present
        if "_leftImg8bit" in base_name:
            base_name = base_name.replace("_leftImg8bit", "")
        
        # Try different label naming conventions
        possible_label_names = [
            base_name + "_gtFine_CategoryId.png",  # For set2: 000010_gtFine_CategoryId.png
            base_name + "_CategoryId.png",          # For cam folders
            base_name + "_labelmap.png",
            base_name + ".png"
        ]
        
        # Check in the corresponding subdirectory in labelmap folder
        label_subdir = os.path.join(labelmap_base, subdir)
        
        for label_name in possible_label_names:
            label_path = os.path.join(label_subdir, label_name)
            if os.path.exists(label_path):
                return label_path
        
        # If not found in subdir, return the first possible name
        # (will be checked in __getitem__)
        return os.path.join(label_subdir, possible_label_names[0])

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
                print(f"Expected label path: {label_path}")
                label = Image.new('L', img.size, 0)
            
            img, label = self.transform(img, label)
            return img, label.long()
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Label path: {self.label_paths[idx]}")
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
            model_name = opts.model
            
            try:
                if hasattr(model.module, 'classifier'):
                    if hasattr(model.module.classifier, '__getitem__'):
                        num_classes = model.module.classifier[-1].out_channels
                    elif hasattr(model.module.classifier, 'out_channels'):
                        num_classes = model.module.classifier.out_channels
                    else:
                        num_classes = opts.num_classes
                else:
                    num_classes = opts.num_classes
            except:
                num_classes = opts.num_classes
            
            print(f"Creating fresh model for FLOPs: {model_name}, classes: {num_classes}")
            
            flops_model = network.modeling.__dict__[model_name](
                num_classes=num_classes,
                output_stride=16
            ).to(device)
            flops_model.eval()
            
            print("Running FLOPs calculation...")
            flops, params = profile(flops_model, inputs=(x,), verbose=False)
            print(f'Parameters (thop): {params/(1024*1024):.2f} M')
            result = f'FLOPs: {flops/(1024*1024*1024):.2f} G'
            print(result)

            with open("FLOPs.txt", "w") as f:
                f.write(result + "\n")
            
            del flops_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
            print("Providing parameter-based estimation...")
            estimated_flops = total_params * h * w * 2 / 1e9
            print(f'Estimated FLOPs: {estimated_flops:.2f} G (rough approximation)')
            flops = estimated_flops * 1e9
    else:
        flops = None
        print("FLOPs calculation skipped (thop not available)")
    
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
    for run in range(9):
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
        targets = targets.flatten()
        preds = preds.flatten()
        
        mask = targets != 255
        targets = targets[mask]
        preds = preds[mask]
        
        preds = np.clip(preds, 0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)
        
        for t, p in zip(targets, preds):
            self.confusion_matrix[t, p] += 1
    
    def get_results(self):
        """Compute and return all metrics"""
        confusion = self.confusion_matrix
        
        ious = []
        for i in range(self.num_classes):
            TP = confusion[i, i]
            FP = confusion[:, i].sum() - TP
            FN = confusion[i, :].sum() - TP
            
            if TP + FP + FN == 0:
                iou = 0.0
            else:
                iou = TP / (TP + FP + FN)
            ious.append(iou)
        
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        
        total_correct = np.diag(confusion).sum()
        total_pixels = confusion.sum()
        overall_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
        
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
        
        class_ious = results['Class IoU']
        string += f"Class IoU:\n"
        for i, iou in enumerate(class_ious):
            string += f"  Class {i}: {iou:.3f}\n"
        
        return string


def test_model(opts, model, loader, device, metrics, save_samples_ids=None):
    """Test the model and return results"""
    metrics.reset()
    ret_samples = []
    
    if opts.save_results:
        results_dir = opts.output_dir
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
            
            if save_samples_ids is not None and i in save_samples_ids:
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_results and opts.save_samples:
                for batch_idx in range(len(images)):
                    image = images[batch_idx].detach().cpu().numpy()
                    target = targets[batch_idx]
                    pred = preds[batch_idx]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    
                    if hasattr(loader.dataset, 'decode_target'):
                        target_rgb = loader.dataset.decode_target(target).astype(np.uint8)
                        pred_rgb = loader.dataset.decode_target(pred).astype(np.uint8)
                    else:
                        target_rgb = target.astype(np.uint8)
                        pred_rgb = pred.astype(np.uint8)

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

    score = metrics.get_results()
    
    if opts.save_results:
        summary_path = os.path.join(results_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Test Results Summary\n")
            f.write("===================\n")
            f.write(f"Model: {opts.model}\n")
            f.write(f"Checkpoint: {opts.ckpt}\n")
            f.write(f"Dataset: {opts.dataset}\n")
            f.write(f"Total images: {len(loader.dataset)}\n\n")
            f.write(metrics.to_str(score))
            f.write(f"\n\nDetailed Metrics:\n")
            for key, value in score.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {str(value)}\n")
        
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
    
    pred_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    img_id = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            
            for batch_idx in range(len(images)):
                pred = preds[batch_idx].astype(np.uint8)
                
                pred_img = Image.fromarray(pred, mode='L')
                pred_path = os.path.join(pred_dir, f'{img_id:05d}_pred.png')
                pred_img.save(pred_path)
                
                img_id += 1
    
    return pred_dir


def evaluate_custom_miou(pred_dir, label_dir, num_classes):
    """Evaluate mIoU using custom confusion matrix approach"""
    print("\n" + "="*60)
    print("CUSTOM mIOU EVALUATION")
    print("="*60)
    
    pred_paths = sorted(glob(os.path.join(pred_dir, "*.png")))
    
    if not pred_paths:
        print("No prediction files found!")
        return None, None
    
    print(f"Found {len(pred_paths)} prediction files")
    
    all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    processed_count = 0
    
    for pred_path in tqdm(pred_paths, desc="Computing confusion matrix"):
        filename = os.path.basename(pred_path)
        file_id = os.path.splitext(filename)[0].replace("_pred", "")
        
        possible_label_names = [
            f"{file_id}_gtFine_CategoryId.png",
            f"{file_id}_CategoryId.png",
            f"{file_id}_labelmap.png",
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
            pred = np.array(Image.open(pred_path)).astype(np.uint8)
            label = np.array(Image.open(label_path)).astype(np.uint8)
            
            if len(pred.shape) == 3:
                pred = pred[:, :, 0]
            
            pred = pred.flatten()
            label = label.flatten()
            
            mask = label != 255
            pred = pred[mask]
            label = label[mask]
            
            pred = np.clip(pred, 0, num_classes - 1)
            label = np.clip(label, 0, num_classes - 1)
            
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
    
    # Initialize WandB if requested
    wandb_run = None
    if opts.use_wandb:
        if not WANDB_AVAILABLE:
            print("ERROR: WandB logging requested but wandb is not installed!")
            print("Install with: pip install wandb")
            return
        
        # Parse tags
        tags = ['test']  # Always add 'test' tag
        if opts.wandb_tags:
            tags.extend([tag.strip() for tag in opts.wandb_tags.split(',')])
        
        # Generate run name
        run_name = opts.wandb_run_name
        if run_name is None:
            # Extract model info from checkpoint name if possible
            ckpt_name = os.path.basename(opts.ckpt).replace('.pth', '')
            run_name = f"{ckpt_name}-test"
        
        # Resume existing run or create new one
        if opts.wandb_run_id:
            print(f"Resuming WandB run: {opts.wandb_run_id}")
            wandb_run = wandb.init(
                project=opts.wandb_project,
                id=opts.wandb_run_id,
                resume='allow'
            )
        else:
            print(f"Creating new WandB run: {run_name}")
            wandb_run = wandb.init(
                project=opts.wandb_project,
                name=run_name,
                tags=tags,
                config={
                    'mode': 'test',
                    'model': opts.model,
                    'checkpoint': opts.ckpt,
                    'dataset': opts.dataset,
                    'num_classes': opts.num_classes,
                    'batch_size': opts.batch_size
                }
            )
        
        print(f"✓ WandB initialized: {wandb_run.url}")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if opts.dataset == 'dna2025dataset':
        test_dst = DNA2025TestDataset(root_dir=opts.data_root)
    else:
        raise NotImplementedError(f"Dataset {opts.dataset} not implemented in test script")

    test_loader = data.DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=False, num_workers=2)

    print(f"Loading model: {opts.model}")
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {opts.ckpt}")
    
    print(f"Loading checkpoint: {opts.ckpt}")
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print("Model loaded successfully")

    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    metrics = SimpleSegMetrics(opts.num_classes)

    save_samples_ids = None
    if opts.save_results and opts.save_samples:
        total_batches = len(test_loader)
        sample_interval = max(1, total_batches // opts.num_samples)
        save_samples_ids = list(range(0, total_batches, sample_interval))[:opts.num_samples]
        print(f"Will save {len(save_samples_ids)} sample images")

    performance_stats = None
    if opts.evaluate_performance:
        performance_stats = evaluate_model_performance(
            model, device, opts,
            input_resolution=opts.input_resolution,
            iterations=opts.speed_iterations
        )

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
    
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    
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
    
    custom_miou = None
    custom_ious = None
    if opts.save_results:
        results_dir = opts.output_dir
        
        pred_dir = save_prediction_masks(opts, model, test_loader, device, results_dir)
        
        # Look for labels in labelmap/test directory
        labelmap_base = os.path.join(opts.data_root, "labelmap", "test")
        if os.path.exists(labelmap_base):
            label_subdirs = [d for d in os.listdir(labelmap_base) 
                           if os.path.isdir(os.path.join(labelmap_base, d))]
            
            if label_subdirs:
                for label_subdir in label_subdirs:
                    label_dir = os.path.join(labelmap_base, label_subdir)
                    if os.path.exists(label_dir) and len(os.listdir(label_dir)) > 0:
                        print(f"\nEvaluating custom mIoU with labels from: {label_subdir}")
                        custom_miou, custom_ious = evaluate_custom_miou(pred_dir, label_dir, opts.num_classes)
                        break
    
    if opts.save_results:
        results_dir = opts.output_dir
        
        summary_path = os.path.join(results_dir, 'comprehensive_results.txt')
        with open(summary_path, 'w') as f:
            f.write("Comprehensive Test Results\n")
            f.write("=========================\n\n")
            
            f.write(f"Model: {opts.model}\n")
            f.write(f"Checkpoint: {opts.ckpt}\n")
            f.write(f"Dataset: {opts.dataset}\n")
            f.write(f"Total images: {len(test_loader.dataset)}\n\n")
            
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
            
            f.write("Accuracy Metrics:\n")
            f.write("-" * 17 + "\n")
            f.write(metrics.to_str(score))
            f.write(f"\n\nDetailed Metrics:\n")
            for key, value in score.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {str(value)}\n")
            
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
    
    # Log results to WandB
    if opts.use_wandb and wandb_run is not None:
        print("\n" + "="*60)
        print("LOGGING TO WANDB")
        print("="*60)
        
        # Prepare test metrics
        test_metrics = {
            '[Test] Mean IoU': score['Mean IoU'],
            '[Test] Overall Acc': score['Overall Acc'],
            '[Test] FreqW Acc': score['FreqW Acc']
        }
        
        # Add class-wise IoU
        class_ious = score['Class IoU']
        if hasattr(test_dst, 'get_class_info'):
            class_info = test_dst.get_class_info()
            for i, (class_name, iou) in enumerate(zip(class_info['names'], class_ious)):
                test_metrics[f'[Test] IoU/{class_name}'] = iou
        else:
            for i, iou in enumerate(class_ious):
                test_metrics[f'[Test] IoU/Class_{i}'] = iou
        
        # Add custom mIoU if available
        if custom_miou is not None:
            test_metrics['[Test] Custom mIoU'] = custom_miou
        
        # Add performance stats if available
        if performance_stats:
            test_metrics['[Model] Parameters (M)'] = performance_stats['total_params_M']
            if performance_stats['flops_G']:
                test_metrics['[Model] FLOPs (G)'] = performance_stats['flops_G']
            test_metrics['[Model] GPU Memory (MB)'] = performance_stats['memory_MB']
            test_metrics['[Model] Latency Mean (ms)'] = performance_stats['latency_mean_ms']
            test_metrics['[Model] Latency Median (ms)'] = performance_stats['latency_median_ms']
            test_metrics['[Model] FPS Mean'] = performance_stats['fps_mean']
            test_metrics['[Model] FPS Median'] = performance_stats['fps_median']
        
        # Log all metrics
        wandb.log(test_metrics)
        
        # Create summary table for class-wise IoU
        class_iou_data = []
        if hasattr(test_dst, 'get_class_info'):
            class_info = test_dst.get_class_info()
            for i, (class_name, iou) in enumerate(zip(class_info['names'], class_ious)):
                class_iou_data.append([i, class_name, iou])
        else:
            for i, iou in enumerate(class_ious):
                class_iou_data.append([i, f'Class_{i}', iou])
        
        table = wandb.Table(
            columns=["Class ID", "Class Name", "IoU"],
            data=class_iou_data
        )
        wandb.log({"[Test] Class IoU Table": table})
        
        # Log sample images if available
        if ret_samples and len(ret_samples) > 0:
            print("Logging sample images to WandB...")
            denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            for idx, (img, target, pred) in enumerate(ret_samples[:5]):  # Log up to 5 samples
                # Denormalize and convert to uint8
                img_np = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
                
                # Decode target and prediction
                if hasattr(test_dst, 'decode_target'):
                    target_rgb = test_dst.decode_target(target).astype(np.uint8)
                    pred_rgb = test_dst.decode_target(pred).astype(np.uint8)
                else:
                    target_rgb = np.stack([target, target, target], axis=-1).astype(np.uint8)
                    pred_rgb = np.stack([pred, pred, pred], axis=-1).astype(np.uint8)
                
                # Log to WandB
                wandb.log({
                    f"[Test] Sample {idx}/Image": wandb.Image(img_np, caption=f"Sample {idx} - Input"),
                    f"[Test] Sample {idx}/Ground Truth": wandb.Image(target_rgb, caption=f"Sample {idx} - GT"),
                    f"[Test] Sample {idx}/Prediction": wandb.Image(pred_rgb, caption=f"Sample {idx} - Pred")
                })
        
        print(f"✓ Test results logged to WandB: {wandb_run.url}")
        print("="*60)
        
        # Finish WandB run
        wandb.finish()
        print("✓ WandB run finished")


if __name__ == "__main__":
    main()

# ===== 사용 예시 =====

# 1. Baseline 테스트 + WandB 로깅 (새 run)
# python my_test.py \
#     --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
#     --save_results \
#     --output_dir test_results_baseline \
#     --save_samples \
#     --evaluate_performance \
#     --use_wandb \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_run_name "baseline-test" \
#     --wandb_tags "baseline,test"

# 2. 기존 훈련 run에 테스트 결과 추가 (resume)
# python my_test.py \
#     --ckpt ./checkpoints/best_deeplabv3_mobilenet_dna2025dataset_os16.pth \
#     --save_results \
#     --output_dir test_results_baseline \
#     --evaluate_performance \
#     --use_wandb \
#     --wandb_project "deeplabv3-segmentation" \
#     --wandb_run_id "YOUR_RUN_ID_HERE"

# 3. 빠른 테스트 (WandB 없음, 결과만 로컬 저장)
# python my_test.py \
#     --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
#     --save_results \
#     --output_dir test_results_quick \
#     --evaluate_performance

# 4. 커스텀 결과 폴더 지정
# python my_test.py \
#     --ckpt ./checkpoints/deeplabv3_mobilenet_dna2025dataset_baseline.pth \
#     --save_results \
#     --output_dir test_results_experiment_001 \
#     --save_samples \
#     --num_samples 20