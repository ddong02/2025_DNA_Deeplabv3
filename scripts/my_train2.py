from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
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


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='2025dna',
                        choices=['voc', 'cityscapes', '2025dna'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

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

    # Train Options - Modified to use epochs instead of iterations
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
                        help='batch size (default: 8)')
    parser.add_argument("--val_batch_size", type=int, default=2,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=1024)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    # Transfer Learning Options - Added backbone freeze functionality
    parser.add_argument("--freeze_backbone", action='store_true', default=False,
                        help="freeze backbone for initial epochs")
    parser.add_argument("--freeze_epochs", type=int, default=10,
                        help="number of epochs to freeze backbone (default: 10)")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval in batches (default: 10)")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
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
        image_dir = os.path.join(self.root_dir, "image")
        label_dir = os.path.join(self.root_dir, "labelmap")

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
        """
        Convert class index mask to RGB color image.
        """
        mask = np.array(mask, dtype=np.uint8)
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for i, color in enumerate(self.colors):
            rgb_mask[mask == i] = color
            
        return rgb_mask

# ==============================================================================
# 3. Stratified Data Split Function with folder distribution check
# ==============================================================================
def create_stratified_split(root_dir, subset, val_ratio=0.2, random_state=42):
    """
    Split image files for each subfolder within the specified directory.
    Includes folder distribution verification.
    """
    source_image_dir = os.path.join(root_dir, "image", subset)
    
    if not os.path.exists(source_image_dir):
        print(f"Error: Directory '{source_image_dir}' does not exist.")
        print(f"Please check if the path is correct.")
        return [], []
    
    try:
        subfolders = sorted([f.name for f in os.scandir(source_image_dir) if f.is_dir()])
        if not subfolders:
            print(f"Warning: No subfolders found in '{source_image_dir}'")
            # Try to load files directly from the directory
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
    print(f"Performing stratified split on the following subfolders: {subfolders}")
    
    # Dictionary to track folder distribution
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

    # Print folder distribution verification
    print("\n=== Data Split Verification by Folder ===")
    for folder, counts in folder_distribution.items():
        total = counts['train'] + counts['val']
        val_percentage = (counts['val'] / total * 100) if total > 0 else 0
        print(f"{folder:20s}: Train={counts['train']:3d}, Val={counts['val']:3d}, "
              f"Total={total:3d}, Val%={val_percentage:.1f}%")
    
    return train_image_paths, val_image_paths

def get_dataset(opts):
    """ Dataset And Augmentation """
    train_dst, val_dst = None, None  # Initialize variables
    
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    elif opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
        
    # Custom dataset
    elif opts.dataset == '2025dna':
        ROOT_DIR = opts.data_root
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
            print("\n--- Split Results ---")
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

    if train_dst is None or val_dst is None:
        raise ValueError(f"Unsupported dataset: {opts.dataset}")

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
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
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

def visualize_batch(images, labels, dataset, num_images=4, save_path=None, show_display=False):
    """
    Visualize batch of images and labels from the dataset.
    
    Args:
        images: Batch of images
        labels: Batch of labels
        dataset: Dataset object with decode_target method
        num_images: Number of images to visualize
        save_path: Path to save visualization (if None, saves to ./sample_visualization/)
        show_display: Whether to attempt displaying with plt.show() (for GUI environments)
    """
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    num_images = min(num_images, len(images))
    
    # Set matplotlib to non-interactive backend for headless environments
    if not show_display:
        matplotlib.use('Agg')  # Use non-interactive backend

    fig, axes = plt.subplots(num_images, 2, figsize=(12, 5 * num_images))
    fig.suptitle("Original Image vs. Label Map", fontsize=16)

    for i in range(num_images):
        img = images[i].cpu().numpy()
        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)

        label = labels[i].cpu().numpy()
        if hasattr(dataset, 'decode_target'):
            label_color = dataset.decode_target(label).astype(np.uint8)
        else:
            # Fallback for dummy datasets
            label_color = np.stack([label] * 3, axis=-1).astype(np.uint8)

        ax_img = axes[i, 0] if num_images > 1 else axes[0]
        ax_label = axes[i, 1] if num_images > 1 else axes[1]

        ax_img.imshow(img)
        ax_img.set_title(f"Original Image #{i+1}")
        ax_img.axis('off')

        ax_label.imshow(label_color)
        ax_label.set_title(f"Label Map #{i+1}")
        ax_label.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or show based on environment
    if save_path is None:
        save_path = './sample_visualization'
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f'batch_visualization_{np.random.randint(0, 9999):04d}.png')
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = save_path
    
    try:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_file}")
        
        if show_display:
            plt.show()
    except Exception as e:
        print(f"Error saving/showing visualization: {e}")
    finally:
        plt.close(fig)  # Always close the figure to free memory

def freeze_backbone(model, freeze=True):
    """Freeze or unfreeze the backbone parameters"""
    for param in model.module.backbone.parameters():
        param.requires_grad = not freeze
    
    if freeze:
        print("🧊 Backbone frozen for transfer learning")
    else:
        print("🔓 Backbone unfrozen - full model training")

def inspect_data_samples(train_loader, val_loader, train_dst, val_dst, num_samples=3):
    """Inspect some data samples from train and validation loaders"""
    print("\n=== Data Sample Inspection ===")
    
    try:
        # Training data inspection
        print("\n--- Training Data Samples ---")
        train_images, train_labels = next(iter(train_loader))
        print(f"Training batch - Images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
        print(f"Image value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print(f"Label unique values: {torch.unique(train_labels)}")
        
        # Validation data inspection
        print("\n--- Validation Data Samples ---")
        val_images, val_labels = next(iter(val_loader))
        print(f"Validation batch - Images shape: {val_images.shape}, Labels shape: {val_labels.shape}")
        print(f"Image value range: [{val_images.min():.3f}, {val_images.max():.3f}]")
        print(f"Label unique values: {torch.unique(val_labels)}")
        
        # Check if we're in a GUI environment
        has_display = check_display_environment()
        
        # Visualize some samples
        print(f"\nVisualizing {num_samples} samples from training set...")
        if has_display:
            print("GUI environment detected - displaying visualizations")
            visualize_batch(train_images, train_labels, train_dst, 
                          num_images=min(num_samples, len(train_images)), show_display=True)
        else:
            print("Headless environment detected - saving visualizations to files")
            visualize_batch(train_images, train_labels, train_dst, 
                          num_images=min(num_samples, len(train_images)), 
                          save_path='./sample_visualization/training_samples.png', show_display=False)
    
    except Exception as e:
        print(f"Error during data inspection: {e}")
        print("Skipping data visualization due to data issues.")

def check_display_environment():
    """
    Check if we're in a GUI environment where matplotlib can display images
    """
    try:
        # Check if DISPLAY environment variable is set (for X11)
        display = os.environ.get('DISPLAY')
        if display is None:
            return False
            
        # Try to connect to the display
        import subprocess
        result = subprocess.run(['xset', 'q'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        # If any error occurs, assume headless
        return False

def save_sample_predictions(images, targets, predictions, dataset, save_dir='./sample_results', epoch=0):
    """
    Save sample predictions as image files for inspection
    
    Args:
        images: Batch of input images (numpy arrays)
        targets: Ground truth labels (numpy arrays)
        predictions: Model predictions (numpy arrays)
        dataset: Dataset object with decode_target method
        save_dir: Directory to save results
        epoch: Current epoch number
    """
    os.makedirs(save_dir, exist_ok=True)
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Set matplotlib backend for headless environments
    matplotlib.use('Agg')
    
    for i in range(min(4, len(images))):  # Save up to 4 samples
        # Prepare image - data is already numpy array from ret_samples
        img = images[i]  # Already numpy array
        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
        
        # Prepare target and prediction - already numpy arrays
        target = targets[i]
        pred = predictions[i]
        
        if hasattr(dataset, 'decode_target'):
            target_color = dataset.decode_target(target).astype(np.uint8)
            pred_color = dataset.decode_target(pred).astype(np.uint8)
        else:
            target_color = np.stack([target] * 3, axis=-1).astype(np.uint8)
            pred_color = np.stack([pred] * 3, axis=-1).astype(np.uint8)
        
        # Create comparison figure
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
        
        # Save the figure
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{i:02d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if i == 0:  # Only print for the first sample
            print(f"Sample predictions saved to: {save_dir}")

def safe_visdom_setup(vis_port, vis_env):
    """Safely setup Visdom connection with error handling"""
    try:
        vis = Visualizer(port=vis_port, env=vis_env)
        # Test connection
        vis.vis_scalar('Test', 0, 0)
        print(f"✅ Visdom connected successfully on port {vis_port}")
        return vis
    except Exception as e:
        print(f"❌ Visdom connection failed: {e}")
        print("⚠️  Training will continue without visualization")
        print("💡 To fix this:")
        print(f"   1. Start visdom server: python -m visdom.server -port {vis_port}")
        print("   2. Or disable visualization with --enable_vis=False")
        return None

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization with safe connection
    vis = None
    if opts.enable_vis:
        vis = safe_visdom_setup(opts.vis_port, opts.vis_env)
        if vis is not None:  # display options
            vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    try:
        train_dst, val_dst = get_dataset(opts)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
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

    # Inspect data samples
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
        scheduler = utils.PolyLR(optimizer, total_iters, power=0.9)
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
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epoch = checkpoint.get("cur_epoch", 0)
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # Modify model for correct number of classes
    last_layer_in_channels = model.module.classifier.classifier[4].in_channels
    new_num_classes = 19
    print(f"✅ new class num: {new_num_classes}")
    new_last_layer = nn.Conv2d(last_layer_in_channels, new_num_classes, kernel_size=(1, 1), stride=(1, 1))
    new_last_layer = new_last_layer.to(device)
    model.module.classifier.classifier[4] = new_last_layer

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    # Training loop - now epoch-based
    for epoch in range(cur_epoch, opts.total_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{opts.total_epochs}")
        print(f"{'='*50}")
        
        # Handle backbone freezing for transfer learning
        if opts.freeze_backbone and epoch < opts.freeze_epochs:
            freeze_backbone(model, freeze=True)
        elif opts.freeze_backbone and epoch == opts.freeze_epochs:
            freeze_backbone(model, freeze=False)
        
        # =====  Train  =====
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epoch+1}', total=num_batches):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Print batch progress
            if (batch_idx + 1) % opts.print_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Avg Loss: {avg_loss:.4f}")

            if opts.lr_policy == 'poly':
                scheduler.step()
        
        # Step scheduler if using step policy
        if opts.lr_policy == 'step':
            scheduler.step()
            
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        # Update visdom with epoch-based metrics
        if vis is not None:
            vis.vis_scalar('Training Loss', epoch+1, avg_epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
            vis.vis_scalar('Learning Rate', epoch+1, current_lr)
        
        # =====  Validation after each epoch  =====
        print("Running validation...")
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        
        # Save sample predictions as images (for headless environments)
        if ret_samples and not check_display_environment():
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
        
        if val_score['Mean IoU'] > best_score:  # save best model
            best_score = val_score['Mean IoU']
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                      (opts.model, opts.dataset, opts.output_stride), epoch+1)
            print(f"✅ New best model saved! Mean IoU: {best_score:.4f}")

        # Update visdom with validation metrics (epoch-based)
        if vis is not None:
            vis.vis_scalar("[Val] Overall Acc", epoch+1, val_score['Overall Acc'])
            vis.vis_scalar("[Val] Mean IoU", epoch+1, val_score['Mean IoU'])
            vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

            for k, (img, target, lbl) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                if hasattr(train_dst, 'decode_target'):
                    target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                    lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                else:
                    # Fallback for datasets without decode_target
                    target = np.stack([target] * 3, axis=0).astype(np.uint8)
                    lbl = np.stack([lbl] * 3, axis=0).astype(np.uint8)
                concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                vis.vis_image('Sample %d' % k, concat_img)

    print("\n🎉 Training completed!")
    print(f"Best validation Mean IoU: {best_score:.4f}")
    
    # Final summary of saved files
    print(f"\n📁 Generated Files:")
    print(f"   - Checkpoints: ./checkpoints/")
    if not check_display_environment():
        print(f"   - Sample visualizations: ./sample_visualization/")
        print(f"   - Validation samples: ./validation_samples/")
    if opts.save_val_results:
        print(f"   - Validation results: ./results/")


def print_evaluation_metrics_explanation():
    """
    Print explanation of evaluation metrics used in semantic segmentation.
    """
    print("\n" + "="*80)
    print("📊 EVALUATION METRICS EXPLANATION")
    print("="*80)
    
    print("\n1. 📏 PIXEL ACCURACY (Overall Acc)")
    print("   - Ratio of correctly classified pixels among all pixels")
    print("   - Formula: (Correctly classified pixels) / (Total pixels)")
    print("   - Range: 0~1 (closer to 1 is better)")
    print("   - Limitation: Sensitive to class imbalance (high when background dominates)")
    
    print("\n2. 🎯 INTERSECTION OVER UNION (IoU)")
    print("   - Intersection of prediction and ground truth divided by their union")
    print("   - Formula: IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)")
    print("   - Range: 0~1 (closer to 1 is better)")
    print("   - Advantage: Less sensitive to class imbalance, better reflects object shape")
    
    print("\n3. 📊 MEAN INTERSECTION OVER UNION (Mean IoU)")
    print("   - Average IoU across all classes")
    print("   - Formula: (Sum of all class IoUs) / (Number of classes)")
    print("   - Most important metric in semantic segmentation")
    print("   - Evaluates all classes equally")
    
    print("\n4. 📈 CLASS IoU")
    print("   - Individual IoU values for each class")
    print("   - Helps identify which classes are well/poorly classified")
    print("   - Useful for per-class performance analysis and model improvement")
    
    print("\n5. 🔍 FREQUENCY WEIGHTED IoU (FWIoU)")
    print("   - IoU weighted by class frequency (pixel count)")
    print("   - Formula: Σ(Class_pixel_ratio × Class_IoU)")
    print("   - Gives higher weight to frequently appearing classes")
    
    print("\n💡 WHICH METRIC TO FOCUS ON?")
    print("   - Mean IoU: Overall model performance evaluation (most important)")
    print("   - Class IoU: Individual class performance analysis")
    print("   - Overall Acc: General accuracy reference")
    print("   - Generally, Mean IoU > 0.7 is considered good performance")
    
    print("="*80)


if __name__ == '__main__':
    print_evaluation_metrics_explanation()
    
    # Set matplotlib backend early for headless environments
    if not check_display_environment():
        print("Headless environment detected - setting non-interactive matplotlib backend")
        matplotlib.use('Agg')
    
    main()

# python -m visdom.server -port 28333

# export DISPLAY=""
# python my_train2.py \
#     --data_root ./datasets/data/SemanticDataset_final \
#     --dataset 2025dna \
#     --model deeplabv3_mobilenet \
#     --total_epochs 100 \
#     --freeze_backbone \
#     --freeze_epochs 20 \
#     --enable_vis \
#     --vis_port 28333 \
#     --ckpt checkpoints/best_deeplabv3_mobilenet_voc_os16.pth

#     --ckpt checkpoints/latest_deeplabv3_mobilenet_2025dna_os8.pth

# export DISPLAY=""
# python my_train2.py \
#     --data_root ./datasets/data/SemanticDataset_final \
#     --dataset 2025dna \
#     --model deeplabv3_mobilenet \
#     --total_epochs 100 \
#     --freeze_backbone \
#     --freeze_epochs 1 \
#     --enable_vis \
#     --vis_port 28333 \
#     --ckpt checkpoints/latest_deeplabv3_mobilenet_2025dna_os8.pth