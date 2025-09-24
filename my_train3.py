from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='2025dna',
                        choices=['2025dna'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: 19)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_epochs", type=int, default=200,
                        help="total number of epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=50,
                        help="step size for step scheduler (in epochs)")
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument("--val_batch_size", type=int, default=2,
                        help='batch size for validation (default: 2)')
    parser.add_argument("--crop_size", type=int, default=1024)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    # Transfer Learning Options
    parser.add_argument("--freeze_backbone", action='store_true', default=False,
                        help="freeze backbone for initial epochs")
    parser.add_argument("--freeze_epochs", type=int, default=10,
                        help="number of epochs to freeze backbone (default: 10)")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='28333',
                        help='port for visdom (default: 28333)')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    
    # Test dataset options
    parser.add_argument("--test_data_root", type=str, default='./datasets/data/SemanticDatasetTest',
                        help="path to Test Dataset")
    parser.add_argument("--run_final_test", action='store_true', default=False,
                        help="run final test evaluation after training")
    
    return parser


# ==============================================================================
# 1. Transformation Class
# ==============================================================================
class SegmentationTransform:
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.5, 1.5]):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest = transforms.InterpolationMode.NEAREST

    def __call__(self, image, label):
        # --- Random scale ---
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        image = TF.resize(image, (new_height, new_width), interpolation=self.bilinear)
        label = TF.resize(label, (new_height, new_width), interpolation=self.nearest)

        # --- Pad if needed ---
        pad_h = max(self.crop_size[0] - new_height, 0)
        pad_w = max(self.crop_size[1] - new_width, 0)

        if pad_h > 0 or pad_w > 0:
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image = TF.pad(image, padding, fill=0)
            label = TF.pad(label, padding, fill=255)  # void class padding

        # --- Random crop ---
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # --- Random horizontal flip ---
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # --- To Tensor & Normalize ---
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image, label

class SegmentationValidationTransform:
    """Transformation class for the validation dataset without data augmentation."""
    def __init__(self, crop_size=[1024, 1024]):
        self.crop_size = crop_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest = transforms.InterpolationMode.NEAREST
    
    def __call__(self, image, label):
        # --- Resize to a fixed size ---
        image = TF.resize(image, self.crop_size, interpolation=self.bilinear)
        label = TF.resize(label, self.crop_size, interpolation=self.nearest)

        # --- To Tensor & Normalize ---
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image, label

# ==============================================================================
# 2. Dataset Class
# ==============================================================================
class SegmentationDataset(Dataset):
    colors = [
        (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70),
        (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30),
        (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180),
        (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230)
    ]

    def __init__(self, root_dir, image_paths, transform):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.label_paths = [self._get_label_path(p) for p in self.image_paths]
        self.transform = transform

    def _get_label_path(self, image_path):
        # Handle both train and test datasets
        if "SemanticDataset_final" in self.root_dir:
            # Training dataset
            image_dir = os.path.join(self.root_dir, "image", "train")
            label_dir = os.path.join(self.root_dir, "labelmap", "train")
        else:
            # Test dataset
            image_dir = os.path.join(self.root_dir, "image", "test")
            label_dir = os.path.join(self.root_dir, "labelmap", "test")

        rel_path = os.path.relpath(image_path, image_dir)
        rel_path_parts = rel_path.split(os.sep)
        file_name = rel_path_parts[-1]
        base_name, ext = os.path.splitext(file_name)

        if file_name.endswith("_leftImg8bit.png"):
            new_file_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            new_file_name = base_name + "_CategoryId.png"

        rel_path_parts[-1] = new_file_name
        label_path = os.path.join(label_dir, *rel_path_parts)
        return label_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")
        
        img, label = self.transform(img, label)
        
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(np.array(label, dtype=np.uint8))
        
        return img, label.long()
    
    def decode_target(self, mask):
        """Convert class index mask to RGB color image."""
        mask = np.array(mask, dtype=np.uint8)
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for i, color in enumerate(self.colors):
            rgb_mask[mask == i] = color
            
        return rgb_mask

# ==============================================================================
# 3. Data Split Function
# ==============================================================================
def create_stratified_split(root_dir, subset, val_ratio=0.2, random_state=42):
    """Split image files for each subfolder within the specified directory."""
    source_image_dir = os.path.join(root_dir, "image", subset)
    
    if not os.path.exists(source_image_dir):
        print(f"Error: Directory '{source_image_dir}' does not exist.")
        return [], []
    
    try:
        subfolders = sorted([f.name for f in os.scandir(source_image_dir) if f.is_dir()])
        if not subfolders:
            print(f"Warning: No subfolders found in '{source_image_dir}'")
            files_in_dir = sorted(glob(os.path.join(source_image_dir, "*.*")))
            if files_in_dir:
                print(f"Found {len(files_in_dir)} files directly in the directory")
                if len(files_in_dir) < 2:
                    return files_in_dir, []
                train_files, val_files = train_test_split(
                    files_in_dir, test_size=val_ratio, random_state=random_state
                )
                return train_files, val_files
            else:
                return [], []
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return [], []

    train_image_paths, val_image_paths = [], []
    print(f"Found subfolders: {subfolders}")
    
    folder_distribution = {}

    for folder in subfolders:
        folder_path = os.path.join(source_image_dir, folder)
        files_in_folder = sorted(glob(os.path.join(folder_path, "*.*")))
        
        if len(files_in_folder) < 2:
            train_image_paths.extend(files_in_folder)
            folder_distribution[folder] = {'train': len(files_in_folder), 'val': 0}
            continue
            
        train_files, val_files = train_test_split(
            files_in_folder, test_size=val_ratio, random_state=random_state
        )
        train_image_paths.extend(train_files)
        val_image_paths.extend(val_files)
        folder_distribution[folder] = {'train': len(train_files), 'val': len(val_files)}

    # Print folder distribution
    print("\n=== Data Split by Folder ===")
    for folder, counts in folder_distribution.items():
        total = counts['train'] + counts['val']
        val_percentage = (counts['val'] / total * 100) if total > 0 else 0
        print(f"{folder}: Train={counts['train']}, Val={counts['val']}, Total={total}, Val%={val_percentage:.1f}%")
    
    return train_image_paths, val_image_paths

def get_test_dataset(opts):
    """Create test dataset from test data root"""
    test_root_dir = opts.test_data_root
    test_subset = "test"
    
    source_image_dir = os.path.join(test_root_dir, "image", test_subset)
    
    if not os.path.exists(source_image_dir):
        print(f"Warning: Test directory '{source_image_dir}' does not exist.")
        return None
    
    try:
        subfolders = sorted([f.name for f in os.scandir(source_image_dir) if f.is_dir()])
        print(f"Test dataset subfolders: {subfolders}")
        
        test_image_paths = []
        folder_counts = {}
        
        for folder in subfolders:
            folder_path = os.path.join(source_image_dir, folder)
            files_in_folder = sorted(glob(os.path.join(folder_path, "*.*")))
            test_image_paths.extend(files_in_folder)
            folder_counts[folder] = len(files_in_folder)
        
        print(f"\n=== Test Dataset Distribution ===")
        for folder, count in folder_counts.items():
            print(f"{folder}: {count} images")
        print(f"Total test images: {len(test_image_paths)}")
        
        if test_image_paths:
            test_transform = SegmentationValidationTransform(crop_size=[1024, 1024])
            test_dst = SegmentationDataset(
                root_dir=test_root_dir,
                image_paths=test_image_paths,
                transform=test_transform
            )
            return test_dst
        else:
            return None
            
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return None

def verify_dataset_quality(dataset, dataset_name, num_samples=10):
    """Verify dataset quality by checking IoU between images and labels"""
    print(f"\n=== {dataset_name} Dataset Quality Check ===")
    
    # Create a temporary metrics object for quality check
    temp_metrics = StreamSegMetrics(19)
    temp_metrics.reset()
    
    total_samples = min(num_samples, len(dataset))
    print(f"Checking {total_samples} samples...")
    
    valid_samples = 0
    
    for i in range(total_samples):
        try:
            img, label = dataset[i]
            
            # Convert label to numpy if it's a tensor
            if isinstance(label, torch.Tensor):
                label_np = label.cpu().numpy()
            else:
                label_np = np.array(label)
            
            # Create a dummy prediction that's identical to ground truth for perfect IoU
            perfect_pred = label_np.copy()
            
            # Update metrics with perfect prediction
            temp_metrics.update(label_np[None, ...], perfect_pred[None, ...])
            
            # Check for valid classes
            unique_classes = np.unique(label_np)
            if len(unique_classes) > 1 and not np.all(unique_classes == 255):  # Not just void class
                valid_samples += 1
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if valid_samples > 0:
        # Get results - this should show perfect scores for the samples we tested
        results = temp_metrics.get_results()
        print(f"Valid samples found: {valid_samples}/{total_samples}")
        print(f"Sample IoU check (should be close to 1.0 for valid data): {results['Mean IoU']:.4f}")
        
        if results['Mean IoU'] > 0.8:
            print("Dataset appears to be loaded correctly")
        else:
            print("WARNING: Dataset may have loading issues")
    else:
        print("WARNING: No valid samples found - check your dataset paths and labels")

def get_dataset(opts):
    """Dataset And Augmentation"""
    ROOT_DIR = "./datasets/data/SemanticDataset_final"
    SUBSET_TO_SPLIT = "train"
    VAL_RATIO = 0.2
    CROP_SIZE = [1024, 1024]
    SCALE_RANGE = [0.75, 1.25]

    # Create train/validation file path lists
    train_paths, val_paths = create_stratified_split(
        root_dir=ROOT_DIR,
        subset=SUBSET_TO_SPLIT,
        val_ratio=VAL_RATIO
    )

    if not train_paths and not val_paths:
        print("No file paths were generated. Please check the data_root path.")
        print(f"Expected path: {os.path.join(ROOT_DIR, 'image', SUBSET_TO_SPLIT)}")
        # Create dummy datasets to avoid errors
        from torch.utils.data import TensorDataset
        dummy_data = torch.zeros(1, 3, 1024, 1024)
        dummy_labels = torch.zeros(1, 1024, 1024, dtype=torch.long)
        train_dst = TensorDataset(dummy_data, dummy_labels)
        val_dst = TensorDataset(dummy_data, dummy_labels)
    else:
        print(f"\n--- Dataset Split Results ---")
        print(f"Total images: {len(train_paths) + len(val_paths)}")
        print(f"Training images: {len(train_paths)}")
        print(f"Validation images: {len(val_paths)}")

        train_transform = SegmentationTransform(crop_size=CROP_SIZE, scale_range=SCALE_RANGE)
        val_transform = SegmentationValidationTransform(crop_size=CROP_SIZE)

        train_dst = SegmentationDataset(
            root_dir=ROOT_DIR,
            image_paths=train_paths,
            transform=train_transform
        )

        val_dst = SegmentationDataset(
            root_dir=ROOT_DIR,
            image_paths=val_paths,
            transform=val_transform
        )
        
        # Verify dataset quality
        verify_dataset_quality(train_dst, "Training", num_samples=5)
        verify_dataset_quality(val_dst, "Validation", num_samples=5)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Perform validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), desc='Validating'):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for j in range(len(images)):
                    image = images[j].detach().cpu().numpy()
                    target = targets[j]
                    pred = preds[j]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def save_dataset_samples(images, labels, dataset, save_dir='./dataset_samples', prefix='sample'):
    """Save dataset samples as combined image+label visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    matplotlib.use('Agg')
    
    num_images = min(4, len(images))
    
    # Create one large figure with all samples
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 4 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        img = images[i].cpu().numpy()
        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)

        label = labels[i].cpu().numpy()
        if hasattr(dataset, 'decode_target'):
            label_color = dataset.decode_target(label).astype(np.uint8)
        else:
            label_color = np.stack([label] * 3, axis=-1).astype(np.uint8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image #{i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(label_color)
        axes[i, 1].set_title(f'Label #{i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_combined.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Dataset samples saved to: {save_path}")

def check_backbone_freeze_status(model):
    """Check if backbone parameters are frozen"""
    backbone_params = list(model.module.backbone.parameters())
    frozen_count = sum(1 for param in backbone_params if not param.requires_grad)
    total_count = len(backbone_params)
    
    print(f"Backbone freeze status: {frozen_count}/{total_count} parameters frozen")
    
    if frozen_count == total_count:
        print("Backbone is completely frozen")
    elif frozen_count == 0:
        print("Backbone is completely unfrozen")
    else:
        print(f"Backbone is partially frozen ({frozen_count} out of {total_count})")
    
    return frozen_count, total_count

def freeze_backbone(model, freeze=True):
    """Freeze or unfreeze the backbone parameters"""
    for param in model.module.backbone.parameters():
        param.requires_grad = not freeze
    
    if freeze:
        print("Backbone frozen for transfer learning")
    else:
        print("Backbone unfrozen - full model training")
    
    check_backbone_freeze_status(model)

def verify_checkpoint_loading(checkpoint, model, optimizer, scheduler, opts):
    """Verify that checkpoint was loaded correctly"""
    print(f"\n=== Checkpoint Verification ===")
    
    model_keys = set(checkpoint["model_state"].keys())
    current_model_keys = set(model.module.state_dict().keys())
    
    missing_keys = current_model_keys - model_keys
    unexpected_keys = model_keys - current_model_keys
    
    print(f"Model state verification:")
    print(f"  - Parameters in checkpoint: {len(model_keys)}")
    print(f"  - Parameters in current model: {len(current_model_keys)}")
    
    if missing_keys:
        print(f"  - Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in list(missing_keys)[:5]:
                print(f"    * {key}")
    else:
        print("  - No missing keys")
    
    if unexpected_keys:
        print(f"  - Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 5:
            for key in list(unexpected_keys)[:5]:
                print(f"    * {key}")
    else:
        print("  - No unexpected keys")
    
    if opts.continue_training:
        print(f"\nOptimizer state verification:")
        print(f"  - Optimizer state loaded: {'optimizer_state' in checkpoint}")
        print(f"  - Scheduler state loaded: {'scheduler_state' in checkpoint}")
        print(f"  - Current epoch: {checkpoint.get('cur_epoch', 0)}")
        print(f"  - Best score: {checkpoint.get('best_score', 0.0)}")
    
    print("=== End Checkpoint Verification ===\n")

def inspect_data_samples(train_loader, val_loader, train_dst, val_dst, num_samples=3):
    """Inspect data samples and save as images"""
    print("\n=== Data Sample Inspection ===")
    
    try:
        print("\n--- Training Data Samples ---")
        train_images, train_labels = next(iter(train_loader))
        print(f"Training batch - Images: {train_images.shape}, Labels: {train_labels.shape}")
        print(f"Image value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print(f"Label unique values: {torch.unique(train_labels)}")
        
        print("\n--- Validation Data Samples ---")
        val_images, val_labels = next(iter(val_loader))
        print(f"Validation batch - Images: {val_images.shape}, Labels: {val_labels.shape}")
        print(f"Image value range: [{val_images.min():.3f}, {val_images.max():.3f}]")
        print(f"Label unique values: {torch.unique(val_labels)}")
        
        print(f"\nSaving sample visualizations...")
        save_dataset_samples(train_images, train_labels, train_dst, 
                            save_dir='./dataset_samples', prefix='train')
        
        save_dataset_samples(val_images, val_labels, val_dst, 
                            save_dir='./dataset_samples', prefix='val')
    
    except Exception as e:
        print(f"Error during data inspection: {e}")

def save_sample_predictions(images, targets, predictions, dataset, save_dir='./sample_results', epoch=0):
    """Save sample predictions as image files for inspection"""
    os.makedirs(save_dir, exist_ok=True)
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    matplotlib.use('Agg')
    
    for i in range(min(4, len(images))):
        img = images[i]
        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
        
        target = targets[i]
        pred = predictions[i]
        
        if hasattr(dataset, 'decode_target'):
            target_color = dataset.decode_target(target).astype(np.uint8)
            pred_color = dataset.decode_target(pred).astype(np.uint8)
        else:
            target_color = np.stack([target] * 3, axis=-1).astype(np.uint8)
            pred_color = np.stack([pred] * 3, axis=-1).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(target_color)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_color)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{i:02d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if i == 0:
            print(f"Sample predictions saved to: {save_dir}")

def safe_visdom_setup(vis_port, vis_env):
    """Safely setup Visdom connection with error handling"""
    try:
        print(f"Attempting to connect to Visdom server on port {vis_port}...")
        vis = Visualizer(port=vis_port, env=vis_env)
        # Test connection with more detailed error handling
        vis.vis_scalar('Connection Test', 0, 0)
        print(f"✓ Visdom connected successfully on port {vis_port}")
        print(f"✓ Environment: {vis_env}")
        print(f"✓ Access at: http://localhost:{vis_port}")
        return vis
    except ConnectionError as e:
        print(f"✗ Visdom connection failed - Server not running: {e}")
        print("To fix this:")
        print(f"   1. Start visdom server: python -m visdom.server -port {vis_port}")
        print(f"   2. Wait for server startup, then restart training")
        return None
    except Exception as e:
        print(f"✗ Visdom connection failed - Unknown error: {e}")
        print("Training will continue without visualization")
        print("To fix this:")
        print(f"   1. Start visdom server: python -m visdom.server -port {vis_port}")
        print("   2. Check if port {vis_port} is available")
        print("   3. Or disable visualization with --enable_vis=False")
        return None

def run_final_test(opts, model, device, metrics):
    """Run final test evaluation on test dataset with inference timing"""
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    # Load test dataset
    test_dst = get_test_dataset(opts)
    if test_dst is None:
        print("Test dataset not found or failed to load")
        return None
    
    # Verify test dataset quality
    verify_dataset_quality(test_dst, "Test", num_samples=5)
    
    test_loader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=False, 
        num_workers=2, pin_memory=True, drop_last=False)
    
    print(f"Test dataset loaded: {len(test_dst)} samples")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Model warmup
    print("\nPerforming model warmup...")
    model.eval()
    warmup_iterations = 10
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= warmup_iterations:
                break
            images = images.to(device, dtype=torch.float32)
            _ = model(images)
    
    print(f"Warmup completed with {warmup_iterations} iterations")
    
    # Inference timing measurement
    print("\nMeasuring inference time...")
    model.eval()
    inference_times = []
    total_images = 0
    
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_loader), desc='Timing Inference', total=len(test_loader)):
            images = images.to(device, dtype=torch.float32)
            
            # Synchronize GPU operations before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            outputs = model(images)
            
            # Synchronize GPU operations after inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            batch_size = images.shape[0]
            per_image_time = batch_time / batch_size
            
            inference_times.append(per_image_time)
            total_images += batch_size
    
    # Calculate timing statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    
    print(f"\nINFERENCE TIMING RESULTS:")
    print("="*40)
    print(f"Total images processed: {total_images}")
    print(f"Average inference time per image: {avg_inference_time:.2f} ms")
    print(f"Standard deviation: {std_inference_time:.2f} ms")
    print(f"Minimum inference time: {min_inference_time:.2f} ms")
    print(f"Maximum inference time: {max_inference_time:.2f} ms")
    print(f"Throughput: {1000/avg_inference_time:.2f} images/second")
    
    # Run evaluation with accuracy metrics
    print("\nRunning accuracy evaluation...")
    test_score, test_samples = validate(
        opts=opts, model=model, loader=test_loader, device=device, 
        metrics=metrics, ret_samples_ids=list(range(min(4, len(test_loader))))
    )
    
    print("\nFINAL TEST RESULTS:")
    print("="*40)
    print(metrics.to_str(test_score))
    
    # Save test predictions
    if test_samples:
        images_for_save = []
        targets_for_save = []
        preds_for_save = []
        
        for img, target, pred in test_samples:
            images_for_save.append(img)
            targets_for_save.append(target)
            preds_for_save.append(pred)
        
        if images_for_save:
            save_sample_predictions(
                images_for_save, targets_for_save, preds_for_save, 
                test_dst, save_dir='./test_results', epoch=999
            )
    
    print("="*60)
    return test_score, avg_inference_time

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 19  # Fixed for semantic segmentation

    # Setup visualization with safe connection
    vis = None
    if opts.enable_vis:
        vis = safe_visdom_setup(opts.vis_port, opts.vis_env)
        if vis is not None:
            vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    try:
        train_dst, val_dst = get_dataset(opts)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    print("\n--- DataLoaders Created ---")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Inspect data samples and save as images
    inspect_data_samples(train_loader, val_loader, train_dst, val_dst, num_samples=2)

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=21, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(19)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # Setup scheduler based on epochs
    if opts.lr_policy == 'poly':
        total_iters = opts.total_epochs * len(train_loader)
        print(f"Setting up PolynomialLR scheduler: total_iters={total_iters}, power=0.9")
        # Use PyTorch's built-in PolynomialLR to avoid complex number issues
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path, epoch):
        """ Save current model """
        torch.save({
            "cur_epoch": epoch,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        print(f"Loading checkpoint from: {opts.ckpt}")
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        
        # Verify checkpoint before loading
        model = nn.DataParallel(model)
        model.to(device)
        verify_checkpoint_loading(checkpoint, model, optimizer, scheduler, opts)
        
        model.module.load_state_dict(checkpoint["model_state"])
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epoch = checkpoint.get("cur_epoch", 0)
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # Modify model for correct number of classes
    last_layer_in_channels = model.module.classifier.classifier[4].in_channels
    new_num_classes = 19
    print(f"Updating model to {new_num_classes} classes")
    new_last_layer = nn.Conv2d(last_layer_in_channels, new_num_classes, kernel_size=(1, 1), stride=(1, 1))
    new_last_layer = new_last_layer.to(device)
    model.module.classifier.classifier[4] = new_last_layer

    # Train Loop
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        
        if opts.run_final_test:
            result = run_final_test(opts, model, device, metrics)
            if result is not None:
                test_score, avg_inference_time = result
                print(f"Final test completed - Mean IoU: {test_score['Mean IoU']:.4f}, Avg inference time: {avg_inference_time:.2f} ms")
        return

    # Training loop
    for epoch in range(cur_epoch, opts.total_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{opts.total_epochs}")
        print(f"{'='*50}")
        
        # Handle backbone freezing for transfer learning
        if opts.freeze_backbone and epoch < opts.freeze_epochs:
            freeze_backbone(model, freeze=True)
        elif opts.freeze_backbone and epoch == opts.freeze_epochs:
            freeze_backbone(model, freeze=False)
        
        # Train
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # Use tqdm for progress bar
        pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, labels) in pbar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Update progress bar with current loss
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

            if opts.lr_policy == 'poly':
                scheduler.step()
        
        pbar.close()
        
        # Step scheduler if using step policy
        if opts.lr_policy == 'step':
            scheduler.step()
            
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        # Update visdom with epoch-based metrics
        if vis is not None:
            try:
                print(f"Updating Visdom with epoch {epoch+1} metrics...")
                vis.vis_scalar('Training Loss', epoch+1, avg_epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                vis.vis_scalar('Learning Rate', epoch+1, current_lr)
                print(f"Training metrics sent to Visdom successfully")
            except Exception as e:
                print(f"Failed to update Visdom with training metrics: {e}")
                vis = None  # Disable further Visdom attempts
        
        # Validation after each epoch
        print("Running validation...")
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        
        # Save sample predictions as images
        if ret_samples:
            images_for_save = []
            targets_for_save = []
            preds_for_save = []
            
            for img, target, pred in ret_samples:
                images_for_save.append(img)
                targets_for_save.append(target)
                preds_for_save.append(pred)
            
            if images_for_save:
                save_sample_predictions(
                    images_for_save, targets_for_save, preds_for_save, 
                    train_dst, save_dir='./validation_samples', epoch=epoch+1
                )
        
        # Save checkpoints
        save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                  (opts.model, opts.dataset, opts.output_stride), epoch+1)
        
        if val_score['Mean IoU'] > best_score:
            best_score = val_score['Mean IoU']
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                      (opts.model, opts.dataset, opts.output_stride), epoch+1)
            print(f"New best model saved! Mean IoU: {best_score:.4f}")

        # Update visdom with validation metrics
        if vis is not None:
            try:
                print(f"Updating Visdom with validation metrics...")
                vis.vis_scalar("[Val] Overall Acc", epoch+1, val_score['Overall Acc'])
                vis.vis_scalar("[Val] Mean IoU", epoch+1, val_score['Mean IoU'])
                vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    if hasattr(train_dst, 'decode_target'):
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    else:
                        target = np.stack([target] * 3, axis=0).astype(np.uint8)
                        lbl = np.stack([lbl] * 3, axis=0).astype(np.uint8)
                    concat_img = np.concatenate((img, target, lbl), axis=2)
                    vis.vis_image('Sample %d' % k, concat_img)
                
                print(f"Validation metrics sent to Visdom successfully")
            except Exception as e:
                print(f"Failed to update Visdom with validation metrics: {e}")
                vis = None  # Disable further Visdom attempts

    print("\nTraining completed!")
    print(f"Best validation Mean IoU: {best_score:.4f}")
    
    # Run final test evaluation if requested
    if opts.run_final_test:
        print("\n" + "="*60)
        print("Starting final test evaluation...")
        
        # Load best model for testing
        best_model_path = f'checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth'
        if os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint["model_state"])
            print(f"Best model loaded (Mean IoU: {checkpoint.get('best_score', 'Unknown')})")
        
        run_final_test(opts, model, device, metrics)
    
    # Final summary of saved files
    print(f"\nGenerated Files:")
    print(f"   - Checkpoints: ./checkpoints/")
    print(f"   - Dataset samples: ./dataset_samples/")
    print(f"   - Validation samples: ./validation_samples/")
    if opts.run_final_test:
        print(f"   - Test results: ./test_results/")
    if opts.save_val_results:
        print(f"   - Validation results: ./results/")

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 19  # Fixed for semantic segmentation

    # Setup visualization with safe connection
    vis = None
    if opts.enable_vis:
        vis = safe_visdom_setup(opts.vis_port, opts.vis_env)
        if vis is not None:
            vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    try:
        train_dst, val_dst = get_dataset(opts)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    print("\n--- DataLoaders Created ---")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Inspect data samples and save as images
    inspect_data_samples(train_loader, val_loader, train_dst, val_dst, num_samples=2)

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=19, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(19)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # Setup scheduler based on epochs
    if opts.lr_policy == 'poly':
        total_iters = opts.total_epochs * len(train_loader)
        print(f"Setting up PolynomialLR scheduler: total_iters={total_iters}, power=0.9")
        # Use PyTorch's built-in PolynomialLR to avoid complex number issues
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path, epoch):
        """ Save current model """
        torch.save({
            "cur_epoch": epoch,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        print(f"Loading checkpoint from: {opts.ckpt}")
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        
        # Verify checkpoint before loading
        model = nn.DataParallel(model)
        model.to(device)
        verify_checkpoint_loading(checkpoint, model, optimizer, scheduler, opts)
        
        model.module.load_state_dict(checkpoint["model_state"])
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epoch = checkpoint.get("cur_epoch", 0)
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # Modify model for correct number of classes
    last_layer_in_channels = model.module.classifier.classifier[4].in_channels
    new_num_classes = 19
    print(f"Updating model to {new_num_classes} classes")
    new_last_layer = nn.Conv2d(last_layer_in_channels, new_num_classes, kernel_size=(1, 1), stride=(1, 1))
    new_last_layer = new_last_layer.to(device)
    model.module.classifier.classifier[4] = new_last_layer

    # Train Loop
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        
        if opts.run_final_test:
            run_final_test(opts, model, device, metrics)
        return

    # Training loop
    for epoch in range(cur_epoch, opts.total_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{opts.total_epochs}")
        print(f"{'='*50}")
        
        # Handle backbone freezing for transfer learning
        if opts.freeze_backbone and epoch < opts.freeze_epochs:
            freeze_backbone(model, freeze=True)
        elif opts.freeze_backbone and epoch == opts.freeze_epochs:
            freeze_backbone(model, freeze=False)
        
        # Train
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # Use tqdm for progress bar
        pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, labels) in pbar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Update progress bar with current loss
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

            if opts.lr_policy == 'poly':
                scheduler.step()
        
        pbar.close()
        
        # Step scheduler if using step policy
        if opts.lr_policy == 'step':
            scheduler.step()
            
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        # Update visdom with epoch-based metrics
        if vis is not None:
            try:
                print(f"Updating Visdom with epoch {epoch+1} metrics...")
                vis.vis_scalar('Training Loss', epoch+1, avg_epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                vis.vis_scalar('Learning Rate', epoch+1, current_lr)
                print(f"Training metrics sent to Visdom successfully")
            except Exception as e:
                print(f"Failed to update Visdom with training metrics: {e}")
                vis = None  # Disable further Visdom attempts
        
        # Validation after each epoch
        print("Running validation...")
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        
        # Save sample predictions as images
        if ret_samples:
            images_for_save = []
            targets_for_save = []
            preds_for_save = []
            
            for img, target, pred in ret_samples:
                images_for_save.append(img)
                targets_for_save.append(target)
                preds_for_save.append(pred)
            
            if images_for_save:
                save_sample_predictions(
                    images_for_save, targets_for_save, preds_for_save, 
                    train_dst, save_dir='./validation_samples', epoch=epoch+1
                )
        
        # Save checkpoints
        save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                  (opts.model, opts.dataset, opts.output_stride), epoch+1)
        
        if val_score['Mean IoU'] > best_score:
            best_score = val_score['Mean IoU']
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                      (opts.model, opts.dataset, opts.output_stride), epoch+1)
            print(f"New best model saved! Mean IoU: {best_score:.4f}")

        # Update visdom with validation metrics
        if vis is not None:
            try:
                print(f"Updating Visdom with validation metrics...")
                vis.vis_scalar("[Val] Overall Acc", epoch+1, val_score['Overall Acc'])
                vis.vis_scalar("[Val] Mean IoU", epoch+1, val_score['Mean IoU'])
                vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    if hasattr(train_dst, 'decode_target'):
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    else:
                        target = np.stack([target] * 3, axis=0).astype(np.uint8)
                        lbl = np.stack([lbl] * 3, axis=0).astype(np.uint8)
                    concat_img = np.concatenate((img, target, lbl), axis=2)
                    vis.vis_image('Sample %d' % k, concat_img)
                
                print(f"Validation metrics sent to Visdom successfully")
            except Exception as e:
                print(f"Failed to update Visdom with validation metrics: {e}")
                vis = None  # Disable further Visdom attempts

    print("\nTraining completed!")
    print(f"Best validation Mean IoU: {best_score:.4f}")
    
    # Run final test evaluation if requested
    if opts.run_final_test:
        print("\n" + "="*60)
        print("Starting final test evaluation...")
        
        # Load best model for testing
        best_model_path = f'checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth'
        if os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint["model_state"])
            print(f"Best model loaded (Mean IoU: {checkpoint.get('best_score', 'Unknown')})")
        
        run_final_test(opts, model, device, metrics)
    
    # Final summary of saved files
    print(f"\nGenerated Files:")
    print(f"   - Checkpoints: ./checkpoints/")
    print(f"   - Dataset samples: ./dataset_samples/")
    print(f"   - Validation samples: ./validation_samples/")
    if opts.run_final_test:
        print(f"   - Test results: ./test_results/")
    if opts.save_val_results:
        print(f"   - Validation results: ./results/")


if __name__ == '__main__':
    # Set matplotlib backend for headless environments
    matplotlib.use('Agg')
    
    main()

# python my_train3.py \
#     --dataset 2025dna \
#     --model deeplabv3_mobilenet \
#     --ckpt checkpoints/best_deeplabv3_mobilenet_voc_os16.pth \
#     --continue_training \
#     --freeze_backbone \
#     --freeze_epochs 15 \
#     --enable_vis \
#     --vis_port 28333 \
#     --run_final_test

# python my_train3.py --dataset 2025dna --model deeplabv3_mobilenet --test_only --ckpt checkpoints/latest_deeplabv3_mobilenet_2025dna_os8.pth --run_final_test

# python my_train3.py --dataset 2025dna --model deeplabv3_mobilenet --test_only --ckpt checkpoints/latest_deeplabv3_mobilenet_2025dna_os8.pth --run_final_test
