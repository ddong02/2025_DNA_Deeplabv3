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
import torchvision.transforms.functional as F

from glob import glob
from torch.utils.data import Dataset

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'dna2025dataset'], help='Name of dataset')
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
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=None,
                        help="total iterations (now calculated from epochs)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--unfreeze_epoch", type=int, default=16,
                        help="epoch to unfreeze backbone (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--pretrained_num_classes", type=int, default=21,
                        help="number of classes in pretrained model (default: 21 for VOC)")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

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
    parser.add_argument("--vis_num_samples", type=int, default=4,
                        help='number of samples for visualization (default: 4)')
    
    # ===== 클래스 가중치 관련 옵션 추가 =====
    parser.add_argument("--use_class_weights", action='store_true', default=False,
                        help="use class weights for handling class imbalance (default: True)")
    parser.add_argument("--weight_method", type=str, default='inverse_freq',
                        choices=['inverse_freq', 'sqrt_inv_freq', 'effective_num', 'median_freq'],
                        help="method to calculate class weights (default: inverse_freq)")
    parser.add_argument("--effective_beta", type=float, default=0.9999,
                        help="beta value for effective number method (default: 0.9999)")
    
    return parser


# ===== 클래스 가중치 계산 함수 =====
def calculate_class_weights(dataset, num_classes, device, method='inverse_freq', beta=0.9999, ignore_index=255):
    """
    데이터셋의 모든 레이블을 순회하며 각 클래스의 픽셀 수를 계산하고,
    선택한 방식으로 가중치를 산출합니다.
    
    Args:
        dataset: 데이터셋 객체
        num_classes: 클래스 개수
        device: torch device (cpu/cuda)
        method: 'inverse_freq', 'sqrt_inv_freq', 'effective_num', 'median_freq' 중 선택
        beta: effective number 계산 시 사용하는 파라미터
        ignore_index: 무시할 인덱스 (기본값: 255)
    
    Returns:
        torch.Tensor: 각 클래스에 대한 가중치
    """
    print("\n" + "="*80)
    print(f"  Calculating Class Weights (Method: {method})")
    print("="*80)
    
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    # 전체 데이터셋 순회하여 클래스별 픽셀 수 계산
    print("Analyzing class distribution...")
    for idx in tqdm(range(len(dataset)), desc="Processing labels"):
        _, label = dataset[idx]
        
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        # 각 클래스의 픽셀 수 카운트
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(label == class_id)
    
    # 전체 픽셀 수 (ignore_index 제외)
    total_pixels = np.sum(class_counts)
    
    # 클래스별 빈도 출력
    print("\n" + "-"*80)
    print("Class Distribution:")
    print("-"*80)
    print(f"{'ID':<4} {'Class Name':<25} {'Pixel Count':<15} {'Percentage':<12}")
    print("-"*80)
    
    if hasattr(dataset, 'class_names'):
        for i, (count, name) in enumerate(zip(class_counts, dataset.class_names)):
            percentage = (count / total_pixels) * 100
            print(f"{i:<4} {name:<25} {int(count):<15,} {percentage:>6.2f}%")
    else:
        for i, count in enumerate(class_counts):
            percentage = (count / total_pixels) * 100
            print(f"{i:<4} {'Class_' + str(i):<25} {int(count):<15,} {percentage:>6.2f}%")
    
    print("-"*80)
    print(f"Total Pixels: {int(total_pixels):,}")
    print("-"*80 + "\n")
    
    # 가중치 계산 방법 선택
    if method == 'inverse_freq':
        # 1. Inverse Frequency: weight = total / (num_classes * count)
        class_weights = total_pixels / (num_classes * class_counts + 1e-10)
        
    elif method == 'sqrt_inv_freq':
        # 2. Square Root Inverse Frequency (덜 극단적)
        freq = class_counts / total_pixels
        class_weights = 1.0 / (np.sqrt(freq) + 1e-10)
        
    elif method == 'effective_num':
        # 3. Effective Number of Samples
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        effective_num = 1.0 - np.power(beta, class_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-10)
        
    elif method == 'median_freq':
        # 4. Median Frequency Balancing
        # Paper: "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"
        # weight[c] = median_freq / freq[c], where freq[c] = count[c] / total_pixels
        
        # 각 클래스의 빈도 계산
        class_freq = class_counts / total_pixels
        
        # 빈도의 중앙값 계산 (0이 아닌 값들만 사용)
        non_zero_freq = class_freq[class_freq > 0]
        if len(non_zero_freq) > 0:
            median_freq = np.median(non_zero_freq)
        else:
            median_freq = 1.0
        
        # 가중치 계산: median_freq / freq[c]
        class_weights = median_freq / (class_freq + 1e-10)
        
        print(f"Median Frequency: {median_freq:.6f}")
        print(f"Raw weight range: [{np.min(class_weights):.4f}, {np.max(class_weights):.4f}]")
        print("-"*80 + "\n")
    
    else:
        raise ValueError(f"Unknown weight calculation method: {method}")
    
    # 정규화 (평균이 1이 되도록)
    # median_freq는 원 논문에서는 정규화하지 않지만, 실용적으로는 정규화 권장
    class_weights = class_weights / np.mean(class_weights)
    
    # 계산된 가중치 출력
    print("-"*80)
    print("Calculated Class Weights:")
    print("-"*80)
    print(f"{'ID':<4} {'Class Name':<25} {'Weight':<12} {'Relative Impact':<15}")
    print("-"*80)
    
    max_weight = np.max(class_weights)
    if hasattr(dataset, 'class_names'):
        for i, (weight, name) in enumerate(zip(class_weights, dataset.class_names)):
            impact_bar = '█' * int((weight / max_weight) * 20)
            print(f"{i:<4} {name:<25} {weight:>8.4f}    {impact_bar}")
    else:
        for i, weight in enumerate(class_weights):
            impact_bar = '█' * int((weight / max_weight) * 20)
            print(f"{i:<4} {'Class_' + str(i):<25} {weight:>8.4f}    {impact_bar}")
    
    print("-"*80)
    print(f"Weight Statistics:")
    print(f"  Mean: {np.mean(class_weights):.4f}")
    print(f"  Std:  {np.std(class_weights):.4f}")
    print(f"  Min:  {np.min(class_weights):.4f} (Class {np.argmin(class_weights)})")
    print(f"  Max:  {np.max(class_weights):.4f} (Class {np.argmax(class_weights)})")
    print("="*80 + "\n")
    
    # torch tensor로 변환
    return torch.FloatTensor(class_weights).to(device)


class ExtSegmentationTransform:
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.5, 1.5]):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image, label):
        # --- Random scale ---
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Use ExtScale equivalent functionality
        image = F.resize(image, (new_height, new_width), Image.BILINEAR)
        label = F.resize(label, (new_height, new_width), Image.NEAREST)
        
        # --- Pad if needed ---
        pad_h = max(self.crop_size[0] - new_height, 0)
        pad_w = max(self.crop_size[1] - new_width, 0)
        
        if pad_h > 0 or pad_w > 0:
            # Padding: (left, top, right, bottom)
            padding = (0, 0, pad_w, pad_h)
            image = F.pad(image, padding, fill=0)
            label = F.pad(label, padding, fill=255)  # void class padding
        
        # --- Random crop ---
        if image.size[0] >= self.crop_size[1] and image.size[1] >= self.crop_size[0]:
            # Get random crop parameters
            w, h = image.size
            th, tw = self.crop_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            
            image = F.crop(image, i, j, th, tw)
            label = F.crop(label, i, j, th, tw)
        
        # --- Random horizontal flip ---
        if random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)
        
        # --- To Tensor & Normalize ---
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        
        return image, label

class ExtValidationTransform:
    def __init__(self, crop_size=[1024, 1024]):
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

class DNA2025Dataset(Dataset):
    def __init__(self, root_dir, crop_size, subset, scale_range, random_seed=1):
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.subset = subset
        
        # Define class names and colors for DNA2025 dataset
        self.class_names = [
            'Drivable Area', 'Sidewalk', 'Road Marking', 'Lane', 'Curb', 'Wall/Fence',
            'Car', 'Truck', 'Bus', 'Bike/Bicycle', 'Other Vehicle', 'Pedestrian',
            'Rider', 'Traffic Cone/Pole', 'Other Vertical Object', 'Building', 'Traffic Sign', 'Traffic Light', 'Other'
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
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        base_dir = os.path.join(root_dir, "SemanticDataset_final")
        self.image_paths = []
        self.label_paths = []
        
        # All camera/set folders
        cam_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set1', 'set2', 'set3']
        
        # Map subset to actual directory
        if subset == 'train':
            split_dir = 'train'
        elif subset == 'val':
            split_dir = 'val'
        elif subset == 'test':
            split_dir = 'test'
        else:
            raise ValueError(f"Unknown subset: {subset}. Use 'train', 'val', or 'test'")
        
        print(f"\nLoading {subset} data from '{split_dir}' directory...")
        
        for cam in cam_folders:
            image_pattern = os.path.join(base_dir, "image", split_dir, cam, "*.*")
            cam_images = sorted(glob(image_pattern))
            
            if len(cam_images) == 0:
                continue
            
            # Add all images from this camera folder
            for img_path in cam_images:
                label_path = self._get_label_path(img_path, base_dir, split_dir)
                
                # Verify label exists
                if os.path.exists(label_path):
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
                else:
                    print(f"Warning: Label not found for {img_path}")
            
            print(f"  {cam}: {len(cam_images)} images loaded")
        
        # 데이터 검증
        assert len(self.image_paths) == len(self.label_paths), \
            f"Image/label count mismatch: {len(self.image_paths)} vs {len(self.label_paths)}"
        assert len(self.image_paths) > 0, \
            f"No images found for {subset} subset in {split_dir} directory"
        
        # Use appropriate transform based on subset
        if subset == 'train':
            self.transform = ExtSegmentationTransform(crop_size, scale_range)
        else:  # validation or test
            self.transform = ExtValidationTransform(crop_size)
        
        print(f"Successfully loaded {len(self.image_paths)} {subset} images from {split_dir} directory\n")

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
        """Decode segmentation mask to RGB color image - Optimized version"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Clip mask values to valid range to prevent index errors
        mask = np.clip(mask, 0, len(self.color_palette) - 1)
        
        # Vectorized operation for better performance
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
            
            # Load image and label
            img = Image.open(img_path).convert("RGB")
            
            if os.path.exists(label_path):
                label = Image.open(label_path).convert("L")
            else:
                print(f"Warning: Label not found for {img_path}")
                # Create a dummy label of the same size
                label = Image.new('L', img.size, 0)
            
            # Apply transforms
            img, label = self.transform(img, label)
            
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(np.array(label, dtype=np.uint8))
            
            return img, label.long()
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Label path: {self.label_paths[idx]}")
            raise

def get_dataset(opts):
    """ Dataset And Augmentation
    """
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

    elif opts.dataset == 'dna2025dataset':
        # Use the actual train/val directories from the dataset
        train_dst = DNA2025Dataset(
            root_dir=opts.data_root,
            crop_size=[opts.crop_size, opts.crop_size],
            subset='train',
            scale_range=[0.75, 1.25],
            random_seed=opts.random_seed
        )
        
        val_dst = DNA2025Dataset(
            root_dir=opts.data_root,
            crop_size=[opts.crop_size, opts.crop_size],
            subset='val',
            scale_range=None,
            random_seed=opts.random_seed
        )

    return train_dst, val_dst

def validate(opts, model, loader, device, metrics, ret_samples_ids=None, epoch=None, save_sample_images=False):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    
    # 결과 저장을 위한 디렉토리 구조 생성 (간소화)
    if save_sample_images and epoch is not None:
        base_results_dir = 'results'
        # epoch이 문자열일 수도 있으므로 처리
        if isinstance(epoch, str):
            results_dir = os.path.join(base_results_dir, epoch)
        else:
            results_dir = os.path.join(base_results_dir, f'epoch_{epoch:03d}')
        os.makedirs(results_dir, exist_ok=True)
        
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        saved_count = 0
        max_save_images = 3  # 3개 비교 이미지만 저장

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            # 3개 비교 이미지만 저장 (10 epoch마다)
            if save_sample_images and epoch is not None and saved_count < max_save_images:
                # 배치의 첫 번째 이미지만 처리
                image = images[0].detach().cpu().numpy()
                target = targets[0]
                pred = preds[0]

                # 이미지 후처리
                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                
                # DNA2025Dataset용 색상 디코딩
                if hasattr(loader.dataset, 'decode_target'):
                    target_rgb = loader.dataset.decode_target(target).astype(np.uint8)
                    pred_rgb = loader.dataset.decode_target(pred).astype(np.uint8)
                else:
                    target_rgb = target.astype(np.uint8)
                    pred_rgb = pred.astype(np.uint8)

                # 비교 이미지 (원본 | 정답 | 예측) 생성 및 저장만
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
                
                comparison_path = os.path.join(results_dir, f'comparison_{saved_count:02d}.png')
                plt.tight_layout()
                plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
                plt.close()

                saved_count += 1

        score = metrics.get_results()
        
        # 결과 요약을 텍스트 파일로 저장
        if save_sample_images and epoch is not None:
            summary_path = os.path.join(results_dir, 'validation_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Validation Results\n")
                f.write(f"================\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Saved Images: {saved_count}\n\n")
                f.write(metrics.to_str(score))
                f.write(f"\nBest Scores:\n")
                for key, value in score.items():
                    if isinstance(value, dict):
                        f.write(f"{key}: [Dict with {len(value)} items]\n")
                    elif isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            print(f"Validation results saved to: {results_dir} ({saved_count} comparison images)")
    
    return score, ret_samples

def main():
    opts = get_argparser().parse_args()
    
    # Set num_classes based on dataset
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'dna2025dataset':
        opts.num_classes = 19

    unfreeze_epoch = 16

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, 
        batch_size=opts.batch_size,
        shuffle=True, 
        num_workers=12,  # 20코어 중 12개 사용
        drop_last=True,
        pin_memory=True,
        persistent_workers=True  # 워커 재사용으로 오버헤드 감소
    )

    val_loader = data.DataLoader(
        val_dst, 
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=8,   # validation은 조금 적게
        pin_memory=True,
        persistent_workers=True
    )

    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    if opts.total_itrs is None:
        opts.total_itrs = opts.epochs * len(train_loader)
        print(f"\nTraining for {opts.epochs} epochs, which is {opts.total_itrs} iterations.")

    # ===== Weighted Cross-Entropy Loss 설정 =====
    metrics = StreamSegMetrics(opts.num_classes)
    
    if opts.use_class_weights:
        # 클래스 가중치 계산
        class_weights = calculate_class_weights(
            dataset=train_dst,
            num_classes=opts.num_classes,
            device=device,
            method=opts.weight_method,
            beta=opts.effective_beta,
            ignore_index=255
        )
        
        # Weighted Cross-Entropy Loss
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255,
            reduction='mean'
        )
        print(f"\n✓ Using Weighted Cross-Entropy Loss (method: {opts.weight_method})")
        
    else:
        # 일반 Cross-Entropy Loss
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        print("\n✓ Using Standard Cross-Entropy Loss (no class weights)")

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Helper functions
    def save_ckpt(path, epoch, include_epoch_in_name=False):
        """ save current model """
        if include_epoch_in_name:
            base_path, ext = os.path.splitext(path)
            path = f"{base_path}_epoch{epoch:03d}{ext}"
        
        torch.save({
            "epoch": epoch,
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print(f"Model saved as {path}")

    def load_pretrained_model(model, checkpoint_path, num_classes_old, num_classes_new):
        """Load pretrained model and adjust for different number of classes"""
        print(f"\n=== Loading Pretrained Model ===")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Adjusting from {num_classes_old} to {num_classes_new} classes")
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        state_dict = checkpoint.get("model_state", checkpoint)
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key in model.state_dict():
                if model.state_dict()[key].shape == value.shape:
                    new_state_dict[key] = value
                else:
                    print(f"  Skipping {key} due to size mismatch: {value.shape} -> {model.state_dict()[key].shape}")
            else:
                print(f"  Skipping {key} as it does not exist in the current model.")

        model.load_state_dict(new_state_dict, strict=False)
        print("Pretrained weights loaded successfully!\n")
        return model, checkpoint

    # --- Model Loading and Configuration ---
    utils.mkdir('checkpoints')
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        model, _ = load_pretrained_model(model, opts.ckpt, 
                                         num_classes_old=opts.pretrained_num_classes, 
                                         num_classes_new=opts.num_classes)
    else:
        print("[!] Training from scratch")

    # --- STAGE 1 SETUP: Freeze backbone ---
    print("--- STAGE 1 SETUP: Training classifier only ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params_stage1 = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params=trainable_params_stage1, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
    print(f"Initial Learning Rate: {opts.lr}")  # 초기 학습률 출력

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    model = nn.DataParallel(model)
    model.to(device)

    # --- Training Loop ---
    best_score = 0.0
    cur_itrs = 0
    
    # 시간 추적을 위한 변수들
    import time
    training_start_time = time.time()
    
    # Visdom용 샘플 ID 생성
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for epoch in range(1, opts.epochs + 1):
        epoch_start_time = time.time()
        
        # --- STAGE 2 TRANSITION ---
        if epoch == opts.unfreeze_epoch:
            print(f"\n--- STAGE 2: Unfreezing backbone at Epoch {epoch} ---")
            
            # Unfreeze backbone
            for param in model.module.backbone.parameters():
                param.requires_grad = True
            
            # Differential Learning Rate 적용
            print("Re-creating optimizer with differential learning rates...")
            backbone_lr = opts.lr / 100  # 백본은 매우 낮은 학습률
            classifier_lr = opts.lr / 10  # Classifier는 조금 낮은 학습률
            
            optimizer = torch.optim.SGD([
                {'params': model.module.backbone.parameters(), 'lr': backbone_lr},
                {'params': model.module.classifier.parameters(), 'lr': classifier_lr}
            ], momentum=0.9, weight_decay=opts.weight_decay)
            
            print(f"Backbone LR: {backbone_lr:.6f}, Classifier LR: {classifier_lr:.6f}")

            # 새로운 scheduler 생성 (남은 iteration 고려)
            remaining_itrs = opts.total_itrs - cur_itrs
            if opts.lr_policy == 'poly':
                scheduler = utils.PolyLR(optimizer, remaining_itrs, power=0.9)
            elif opts.lr_policy == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
        
        # 현재 epoch의 학습률 출력 (epoch 시작 시)
        if len(optimizer.param_groups) == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{opts.epochs} - Current Learning Rate: {current_lr:.6f}")
            print(f"{'='*80}")
        else:
            # 미세조정 단계 (여러 param_groups)
            backbone_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{opts.epochs} - Learning Rates:")
            print(f"  Backbone:   {backbone_lr:.6f}")
            print(f"  Classifier: {classifier_lr:.6f}")
            print(f"{'='*80}")

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        interval_loss = 0.0
        
        stage_str = '2 (Fine-tuning)' if epoch >= opts.unfreeze_epoch else '1 (Classifier only)'
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{opts.epochs} [Stage {stage_str}]")
        
        for images, labels in progress_bar:
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            np_loss = loss.detach().cpu().numpy()
            epoch_loss += np_loss
            num_batches += 1
            interval_loss += np_loss

            if cur_itrs % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                progress_bar.set_postfix(loss=f"{interval_loss:.4f}")
                interval_loss = 0.0

        # Epoch 평균 loss 계산
        avg_epoch_loss = epoch_loss / num_batches
        
        # Epoch 요약 출력
        print(f"\nEpoch {epoch}/{opts.epochs} [{stage_str}] completed:")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        if len(optimizer.param_groups) == 1:
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"  Backbone LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Classifier LR: {optimizer.param_groups[1]['lr']:.6f}")

        # --- Validation ---
        print(f"Validation for Epoch {epoch}...")
        model.eval()

        # 이미지 저장 조건 (10 epoch마다만)
        save_images_this_epoch = opts.save_val_results and (epoch % 10 == 0)
        
        val_score, ret_samples = validate(
            opts=opts, 
            model=model, 
            loader=val_loader, 
            device=device, 
            metrics=metrics,
            ret_samples_ids=vis_sample_id,
            epoch=epoch if save_images_this_epoch else None,
            save_sample_images=save_images_this_epoch
        )

        print(f"Validation Results:")
        print(metrics.to_str(val_score))

        # Visdom 업데이트 (매 epoch마다)
        if vis is not None:
            # 현재 learning rate 가져오기
            current_lr = optimizer.param_groups[0]['lr']
            
            # Training Loss 업데이트 (epoch 평균) - 올바른 API 사용
            vis.vis_scalar('Training Loss', epoch, avg_epoch_loss)
            
            # Learning Rate 업데이트
            vis.vis_scalar('Learning Rate', epoch, current_lr)
            
            # Validation 지표들 업데이트 (epoch 기준)
            vis.vis_scalar("[Val] Overall Acc", epoch, val_score['Overall Acc'])
            vis.vis_scalar("[Val] Mean Acc", epoch, val_score['Mean Acc'])
            vis.vis_scalar("[Val] Mean IoU", epoch, val_score['Mean IoU'])
            vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

            # Validation 샘플 이미지 시각화 및 저장 (최근 epoch의 4개만)
            if ret_samples and len(ret_samples) > 0:
                # 최대 4개 샘플만 선택
                samples_to_show = ret_samples[:4]
                
                # Visdom 샘플 이미지 저장용 디렉토리 생성
                visdom_samples_dir = os.path.join('results', 'visdom_samples')
                os.makedirs(visdom_samples_dir, exist_ok=True)
                
                for k, (img, target, lbl) in enumerate(samples_to_show):
                    img = (denorm(img) * 255).astype(np.uint8)
                    
                    # 데이터셋에 따른 decode_target 호출
                    if hasattr(train_dst, 'decode_target'):
                        target_decoded = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl_decoded = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    elif hasattr(val_dst, 'decode_target'):
                        # DNA2025Dataset의 경우 (H,W,3) -> (3,H,W) 변환
                        target_decoded = val_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl_decoded = val_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    else:
                        # 기본 처리: grayscale을 RGB로 변환
                        target_decoded = np.stack([target, target, target], axis=0).astype(np.uint8)
                        lbl_decoded = np.stack([lbl, lbl, lbl], axis=0).astype(np.uint8)
                    
                    # 원본 코드처럼 가로로 연결
                    concat_img = np.concatenate((img, target_decoded, lbl_decoded), axis=2)
                    
                    # Visdom에 표시 (고정된 윈도우 이름 사용)
                    vis.vis_image(f'Validation Sample {k}', concat_img)
                    
                    # 파일로도 저장 (각 epoch마다 덮어쓰기)
                    # transpose back to (H, W, C) for saving
                    concat_img_hwc = concat_img.transpose(1, 2, 0)
                    save_path = os.path.join(visdom_samples_dir, f'validation_sample_{k}_epoch_{epoch:03d}.png')
                    
                    try:
                        Image.fromarray(concat_img_hwc).save(save_path)
                    except Exception as e:
                        print(f"Warning: Failed to save visdom sample {k}: {e}")
                
                try:
                    print(f"Visdom validation samples saved to: {visdom_samples_dir}")
                except:
                    pass  # 출력 실패는 무시

        # 모델 저장
        current_score = val_score['Mean IoU']
        if current_score > best_score:
            best_score = current_score
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride), epoch)
            # 최고 성능 달성 시 epoch 정보와 함께 저장
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride), 
                      epoch, include_epoch_in_name=True)
            
            # 최고 성능 달성시에도 3개 비교 이미지만 저장
            if opts.save_val_results and not save_images_this_epoch:
                print("Best score achieved! Saving 3 comparison images...")
                validate(
                    opts=opts, 
                    model=model, 
                    loader=val_loader, 
                    device=device, 
                    metrics=metrics,
                    ret_samples_ids=None,
                    epoch=f"best_epoch_{epoch}",
                    save_sample_images=True
                )
        
        # 정기적으로 저장 (원본 코드처럼 latest 저장)
        save_ckpt('checkpoints/latest_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride), epoch)
        
        # 10 epoch마다 epoch 정보와 함께 저장
        if epoch % 10 == 0:
            save_ckpt('checkpoints/checkpoint_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride), 
                      epoch, include_epoch_in_name=True)
        
        # 시간 계산 및 출력
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        
        # 남은 시간 추정
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = opts.epochs - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # 시간을 시:분:초 형태로 변환
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        print(f"Epoch {epoch} finished. Best Mean IoU so far: {best_score:.4f}")
        print(f"  Epoch time: {format_time(epoch_time)} | Total elapsed: {format_time(total_elapsed)}")
        print(f"  Estimated remaining: {format_time(estimated_remaining)} | ETA: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_remaining))}")
        print()  # 빈 줄 추가

    print("Training finished.")

if __name__ == "__main__":
    main()


# ===== 각 가중치 방법 설명 =====
# 1. inverse_freq (기본):
#    - 가장 일반적인 방법
#    - weight = total_pixels / (num_classes * class_count)
#    - 장점: 구현 간단, 효과적
#    - 단점: 매우 적은 클래스에 극단적으로 높은 가중치 부여 가능
#
# 2. sqrt_inv_freq:
#    - inverse_freq의 완화된 버전
#    - weight = 1 / sqrt(frequency)
#    - 장점: 극단적인 가중치 방지, 안정적 학습
#    - 단점: 불균형이 심한 경우 효과 제한적
#
# 3. effective_num:
#    - Class-Balanced Loss 논문에서 제안
#    - weight = (1 - beta) / (1 - beta^n)
#    - 장점: 이론적 근거가 확실, 샘플 간 중복 고려
#    - 단점: beta 값 튜닝 필요
#
# 4. median_freq (NEW!):
#    - SegNet 논문에서 제안
#    - weight = median_frequency / class_frequency
#    - 장점: 중앙값 기준으로 균형 맞춤, 극단값에 덜 민감
#    - 단점: 클래스 수가 적으면 효과 제한적
#    - 특징: 정규화하지 않는 것이 원 논문의 방식

# python my_train7.py \
#     --dataset dna2025dataset \
#     --data_root ./datasets/data \
#     --model deeplabv3_mobilenet \
#     --ckpt ./checkpoints/best_deeplabv3_mobilenet_dna2025dataset_os16.pth \
#     --pretrained_num_classes 19 \
#     --num_classes 19 \
#     --unfreeze_epoch 15 \
#     --epochs 150 \
#     --batch_size 4 \
#     --crop_size 1024 \
#     --use_class_weights \
#     --lr 0.01 \
#     --weight_method median_freq \
#     --enable_vis \
#     --vis_port 28333 \
#     --save_val_results

# 1. Visdom 서버 시작 (별도 터미널)
# python -m visdom.server -port 28333

# 2. 기본 훈련 실행
# python my_train8.py \
#     --dataset dna2025dataset \
#     --data_root ./datasets/data \
#     --model deeplabv3_mobilenet \
#     --ckpt ./checkpoints/best_deeplabv3_mobilenet_voc_os16.pth \
#     --pretrained_num_classes 21 \
#     --num_classes 19 \
#     --epochs 200 \
#     --unfreeze_epoch 20 \
#     --batch_size 4 \
#     --crop_size 1024 \
#     --enable_vis \
#     --vis_port 28333 \
#     --save_val_results