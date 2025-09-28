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
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    # Train Options
    # ❗ '--total_itrs'의 기본값을 None으로 변경하거나 이 라인을 주석 처리/삭제합니다.
    parser.add_argument("--total_itrs", type=int, default=None,
                        help="total iterations (now calculated from epochs)")
    # ❗ '--epochs' 인자를 새로 추가합니다.
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.01,
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
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    
    return parser


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
    def __init__(self, root_dir, crop_size, subset, scale_range):
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
        
        # Load image and label paths based on subset
        if subset == 'train':
            # Training data from SemanticDataset_final
            train_base = os.path.join(root_dir, "SemanticDataset_final")
            self.image_paths = []
            self.label_paths = []
            
            # Load from cam0~cam5 and set2 folders
            for cam in ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set2']:
                image_pattern = os.path.join(train_base, "image", "train", cam, "*.*")
                cam_images = sorted(glob(image_pattern))
                self.image_paths.extend(cam_images)
                
                # Get corresponding label paths
                for img_path in cam_images:
                    label_path = self._get_label_path(img_path, train_base, 'train')
                    self.label_paths.append(label_path)
                    
        elif subset == 'test':
            # Test data from SemanticDatasetTest
            test_base = os.path.join(root_dir, "SemanticDatasetTest")
            self.image_paths = []
            self.label_paths = []
            
            # Load from set1 and set3 folders
            for test_set in ['set1', 'set3']:
                image_pattern = os.path.join(test_base, "image", "test", test_set, "*.*")
                test_images = sorted(glob(image_pattern))
                self.image_paths.extend(test_images)
                
                # Get corresponding label paths
                for img_path in test_images:
                    label_path = self._get_label_path(img_path, test_base, 'test')
                    self.label_paths.append(label_path)
        
        # Use appropriate transform based on subset
        if subset == 'train':
            self.transform = ExtSegmentationTransform(crop_size, scale_range)
        else:  # test/validation
            self.transform = ExtValidationTransform(crop_size)
        
        print(f"Loaded {len(self.image_paths)} {subset} images")

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
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Create RGB image
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(len(self.color_palette)):
            rgb_mask[mask == class_id] = self.color_palette[class_id]
        
        return rgb_mask

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
        # Use the custom DNA2025Dataset with built-in transforms
        train_dst = DNA2025Dataset(
            root_dir=opts.data_root,
            crop_size=[1024, 1024],
            subset='train',
            scale_range=[0.75, 1.25]
        )
        
        val_dst = DNA2025Dataset(
            root_dir=opts.data_root,
            crop_size=[1024, 1024],
            subset='test',  # Use test data as validation
            scale_range=None
        )

    return train_dst, val_dst

def validate(opts, model, loader, device, metrics, ret_samples_ids=None, epoch=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    
    # 결과 저장을 위한 디렉토리 구조 생성
    if opts.save_val_results:
        base_results_dir = 'results'
        if epoch is not None:
            results_dir = os.path.join(base_results_dir, f'epoch_{epoch:03d}')
        else:
            results_dir = base_results_dir
            
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'targets'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'overlays'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'comparisons'), exist_ok=True)
        
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

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

            if opts.save_val_results:
                for batch_idx in range(len(images)):
                    image = images[batch_idx].detach().cpu().numpy()
                    target = targets[batch_idx]
                    pred = preds[batch_idx]

                    # 이미지 후처리
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    
                    # DNA2025Dataset용 색상 디코딩
                    if hasattr(loader.dataset, 'decode_target'):
                        target_rgb = loader.dataset.decode_target(target).astype(np.uint8)
                        pred_rgb = loader.dataset.decode_target(pred).astype(np.uint8)
                    else:
                        target_rgb = target.astype(np.uint8)
                        pred_rgb = pred.astype(np.uint8)

                    # 개별 이미지들 저장
                    Image.fromarray(image).save(os.path.join(results_dir, 'images', f'{img_id:05d}_image.png'))
                    Image.fromarray(target_rgb).save(os.path.join(results_dir, 'targets', f'{img_id:05d}_target.png'))
                    Image.fromarray(pred_rgb).save(os.path.join(results_dir, 'predictions', f'{img_id:05d}_pred.png'))

                    # Overlay 이미지 생성 및 저장
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    ax.imshow(image)
                    ax.imshow(pred_rgb, alpha=0.6)
                    ax.set_title(f'Prediction Overlay - Sample {img_id}')
                    ax.axis('off')
                    
                    overlay_path = os.path.join(results_dir, 'overlays', f'{img_id:05d}_overlay.png')
                    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0, dpi=150)
                    plt.close()

                    # 비교 이미지 (원본 | 정답 | 예측) 생성
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
        
        # 결과 요약을 텍스트 파일로 저장
        if opts.save_val_results:
            summary_path = os.path.join(results_dir, 'validation_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Validation Results\n")
                f.write(f"================\n")
                if epoch is not None:
                    f.write(f"Epoch: {epoch}\n")
                f.write(f"Total Images: {img_id}\n\n")
                f.write(metrics.to_str(score))
                f.write(f"\nBest Scores:\n")
                for key, value in score.items():
                    f.write(f"{key}: {value:.4f}\n")
            
            print(f"Validation results saved to: {results_dir}")
    
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
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    if opts.total_itrs is None:
        opts.total_itrs = opts.epochs * len(train_loader)
        print(f"\nTraining for {opts.epochs} epochs, which is {opts.total_itrs} iterations.")

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics and criterion
    metrics = StreamSegMetrics(opts.num_classes)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # Helper functions
    def save_ckpt(path, epoch, include_epoch_in_name=False):
        """ save current model """
        if include_epoch_in_name:
            # 파일 경로에서 확장자 분리
            base_path, ext = os.path.splitext(path)
            path = f"{base_path}_epoch{epoch:03d}{ext}"
        
        torch.save({
            "epoch": epoch,
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
            # Check if the key exists in the current model
            if key in model.state_dict():
                # If the shape of the weight matches, copy it
                if model.state_dict()[key].shape == value.shape:
                    new_state_dict[key] = value
                else:
                    # If the shape doesn't match (e.g., the final classifier layer),
                    # print a message and skip this weight.
                    print(f"  Skipping {key} due to size mismatch: {value.shape} -> {model.state_dict()[key].shape}")
            else:
                # If a key from the checkpoint doesn't exist in the current model, skip it.
                print(f"  Skipping {key} as it does not exist in the current model.")

        # Load the new state dict, which only contains matching weights.
        # strict=False is used as a safeguard.
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

    # --- STAGE 1 SETUP: Freeze backbone and set up optimizer for classifier ---
    print("--- STAGE 1 SETUP: Training classifier only ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params_stage1 = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params=trainable_params_stage1, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    model = nn.DataParallel(model)
    model.to(device)

    # --- Training Loop ---
    best_score = 0.0
    cur_itrs = 0

    train_losses = []
    val_ious = []
    val_accs = []

    for epoch in range(1, opts.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        stage_str = '2 (Fine-tuning)' if epoch >= unfreeze_epoch else '1 (Classifier only)'
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

            epoch_loss += loss.item()
            num_batches += 1

            if cur_itrs % opts.print_interval == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Epoch 평균 loss 계산
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)

        # Validation
        print(f"\nValidation for Epoch {epoch}...")
        model.eval()

        val_score, ret_samples = validate(
            opts=opts, 
            model=model, 
            loader=val_loader, 
            device=device, 
            metrics=metrics,
            ret_samples_ids=list(range(0, len(val_loader), 10))[:opts.vis_num_samples] if vis else None,
            epoch=epoch  # epoch 정보 전달
        )

        print(metrics.to_str(val_score))
        
        # Validation 결과 저장
        val_ious.append(val_score['Mean IoU'])
        val_accs.append(val_score['Overall Acc'])

        # Visdom 업데이트
        if vis is not None:
            # Loss 그래프 업데이트
            vis.vis_plot('Train Loss', 'line', epoch, avg_epoch_loss, 
                        opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss'))
            
            # IoU 그래프 업데이트
            vis.vis_plot('Validation IoU', 'line', epoch, val_score['Mean IoU'], 
                        opts=dict(title='Validation Mean IoU', xlabel='Epoch', ylabel='IoU'))
            
            # Accuracy 그래프 업데이트
            vis.vis_plot('Validation Accuracy', 'line', epoch, val_score['Overall Acc'], 
                        opts=dict(title='Validation Overall Accuracy', xlabel='Epoch', ylabel='Accuracy'))
            
            # Class별 IoU 테이블 업데이트
            if hasattr(val_dst, 'get_class_info'):
                class_info = val_dst.get_class_info()
                class_ious = val_score['Class IoU']
                
                # 클래스별 IoU 테이블 생성
                iou_table = []
                for i, (class_name, iou) in enumerate(zip(class_info['names'], class_ious)):
                    iou_table.append([class_name, f"{iou:.4f}"])
                
                vis.vis_table(f"Class IoU - Epoch {epoch}", iou_table)
            
            # Validation 이미지 샘플 시각화
            if ret_samples:
                vis_samples = []
                for idx, (image, target, pred) in enumerate(ret_samples):
                    # Denormalize image
                    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    
                    # Decode masks if dataset has decode_target method
                    if hasattr(val_dst, 'decode_target'):
                        target_rgb = val_dst.decode_target(target)
                        pred_rgb = val_dst.decode_target(pred)
                    else:
                        target_rgb = target
                        pred_rgb = pred
                    
                    vis_samples.append([
                        image, target_rgb, pred_rgb
                    ])
                
                vis.vis_images(f"Validation Samples - Epoch {epoch}", vis_samples, 
                            opts=dict(title=f'Val Results Epoch {epoch}', 
                                    caption=['Input', 'Target', 'Prediction']))

        # 모델 저장
        if val_score['Mean IoU'] > best_score:
            best_score = val_score['Mean IoU']
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride), epoch)
        
        save_ckpt('checkpoints/latest_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride), epoch)
        print(f"Epoch {epoch} finished. Best Mean IoU so far: {best_score:.4f}\n")    

if __name__ == "__main__":
    main()

# python -m visdom.server -port 28333

# python my_train4.py \
#     --dataset dna2025dataset \
#     --data_root ./datasets/data \
#     --model deeplabv3_mobilenet \
#     --ckpt ./checkpoints/best_deeplabv3_mobilenet_voc_os16.pth \
#     --pretrained_num_classes 21 \
#     --num_classes 19 \
#     --epochs 100 \
#     --enable_vis \
#     --vis_port 28333