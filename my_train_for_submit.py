from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
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
import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='/mnt/c/Users/user/Desktop/eogus/dataset/2025dna',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='dna2025dataset',
                        choices=['dna2025dataset'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: 19)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--total_itrs", type=int, default=None,
                        help="total iterations (now calculated from epochs)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--unfreeze_epoch", type=int, default=16,
                        help="epoch to unfreeze backbone (default: 16)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--crop_size", type=int, default=1024)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
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

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    
    # Class weights
    parser.add_argument("--use_class_weights", action='store_true', default=False,
                        help="use class weights for handling class imbalance (default: True)")
    parser.add_argument("--weight_method", type=str, default='inverse_freq',
                        choices=['inverse_freq', 'sqrt_inv_freq', 'effective_num', 'median_freq'],
                        help="method to calculate class weights (default: inverse_freq)")
    parser.add_argument("--effective_beta", type=float, default=0.9999,
                        help="beta value for effective number method (default: 0.9999)")
    
    return parser


def calculate_class_weights(dataset, num_classes, device, method='inverse_freq', beta=0.9999, ignore_index=255):
    """Calculate class weights for handling class imbalance"""
    print("\n" + "="*80)
    print(f"  Calculating Class Weights (Method: {method})")
    print("="*80)
    
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    print("Analyzing class distribution...")
    for idx in tqdm(range(len(dataset)), desc="Processing labels"):
        _, label = dataset[idx]
        
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(label == class_id)
    
    total_pixels = np.sum(class_counts)
    
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
    
    if method == 'inverse_freq':
        class_weights = total_pixels / (num_classes * class_counts + 1e-10)
    elif method == 'sqrt_inv_freq':
        freq = class_counts / total_pixels
        class_weights = 1.0 / (np.sqrt(freq) + 1e-10)
    elif method == 'effective_num':
        effective_num = 1.0 - np.power(beta, class_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-10)
    elif method == 'median_freq':
        class_freq = class_counts / total_pixels
        non_zero_freq = class_freq[class_freq > 0]
        if len(non_zero_freq) > 0:
            median_freq = np.median(non_zero_freq)
        else:
            median_freq = 1.0
        class_weights = median_freq / (class_freq + 1e-10)
    else:
        raise ValueError(f"Unknown weight calculation method: {method}")
    
    class_weights = class_weights / np.mean(class_weights)
    
    print("-"*80)
    print("Calculated Class Weights:")
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
    
    return torch.FloatTensor(class_weights).to(device)


class ExtSegmentationTransform:
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.5, 1.5]):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image, label):
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        image = F.resize(image, (new_height, new_width), Image.BILINEAR)
        label = F.resize(label, (new_height, new_width), Image.NEAREST)
        
        pad_h = max(self.crop_size[0] - new_height, 0)
        pad_w = max(self.crop_size[1] - new_width, 0)
        
        if pad_h > 0 or pad_w > 0:
            padding = (0, 0, pad_w, pad_h)
            image = F.pad(image, padding, fill=0)
            label = F.pad(label, padding, fill=255)
        
        if image.size[0] >= self.crop_size[1] and image.size[1] >= self.crop_size[0]:
            w, h = image.size
            th, tw = self.crop_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            
            image = F.crop(image, i, j, th, tw)
            label = F.crop(label, i, j, th, tw)
        
        if random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)
        
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        
        return image, label


class DNA2025Dataset(Dataset):
    def __init__(self, root_dir, crop_size, scale_range, random_seed=1):
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.random_seed = random_seed
        
        self.class_names = [
            'Drivable Area', 'Sidewalk', 'Road Marking', 'Lane', 'Curb', 'Wall/Fence',
            'Car', 'Truck', 'Bus', 'Bike/Bicycle', 'Other Vehicle', 'Pedestrian',
            'Rider', 'Traffic Cone/Pole', 'Other Vertical Object', 'Building', 
            'Traffic Sign', 'Traffic Light', 'Other'
        ]
        
        self.color_palette = np.array([
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [220, 20, 60],
            [255, 0, 0], [250, 170, 30], [220, 220, 0], [70, 130, 180],
            [220, 220, 220], [250, 170, 160], [128, 128, 128],
        ], dtype=np.uint8)
        
        np.random.seed(random_seed)
        
        train_base = os.path.join(root_dir, "SemanticDataset_final")
        self.image_paths = []
        self.label_paths = []
        
        cam_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set1', 'set2', 'set3']
        
        print("\n" + "="*80)
        print("Loading Training Dataset")
        print("="*80)
        
        for cam in cam_folders:
            image_pattern = os.path.join(train_base, "image", "train", cam, "*.*")
            cam_images = sorted(glob(image_pattern))
            
            if len(cam_images) == 0:
                print(f"  {cam}: No images found")
                continue
            
            for img_path in cam_images:
                label_path = self._get_label_path(img_path, train_base, 'train')
                self.image_paths.append(img_path)
                self.label_paths.append(label_path)
            
            print(f"  {cam}: {len(cam_images)} images loaded")
        
        assert len(self.image_paths) == len(self.label_paths), \
            f"Image/label count mismatch: {len(self.image_paths)} vs {len(self.label_paths)}"
        assert len(self.image_paths) > 0, "No images found for training"
        
        self.transform = ExtSegmentationTransform(crop_size, scale_range)
        
        print("-"*80)
        print(f"Total Training Images: {len(self.image_paths)}")
        print("="*80 + "\n")

    def _get_label_path(self, image_path, base_dir, split):
        """Convert image path to corresponding label path"""
        label_path = image_path.replace(
            os.path.join(base_dir, "image", split),
            os.path.join(base_dir, "labelmap", split)
        )
        
        dir_name = os.path.dirname(label_path)
        file_name = os.path.basename(label_path)
        base_name, ext = os.path.splitext(file_name)
        
        if "_leftImg8bit" in file_name:
            new_file_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            new_file_name = base_name + "_CategoryId.png"
        
        label_path = os.path.join(dir_name, new_file_name)
        
        if not os.path.exists(label_path):
            label_path = image_path.replace(
                os.path.join(base_dir, "image", split),
                os.path.join(base_dir, "labelmap", split)
            )
        
        return label_path

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
            
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(np.array(label, dtype=np.uint8))
            
            return img, label.long()
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Label path: {self.label_paths[idx]}")
            raise


def get_dataset(opts):
    """Load training dataset only"""
    train_dst = DNA2025Dataset(
        root_dir=opts.data_root,
        crop_size=[opts.crop_size, opts.crop_size],
        scale_range=[0.75, 1.25],
        random_seed=opts.random_seed
    )
    
    return train_dst


def main():
    opts = get_argparser().parse_args()
    
    # Setup visualization
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
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
    train_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)
    
    print(f"Training with {len(train_dst)} images, {len(train_loader)} batches per epoch")

    if opts.total_itrs is None:
        opts.total_itrs = opts.epochs * len(train_loader)
        print(f"Training for {opts.epochs} epochs, which is {opts.total_itrs} iterations.\n")

    # Setup loss function
    if opts.use_class_weights:
        class_weights = calculate_class_weights(
            dataset=train_dst,
            num_classes=opts.num_classes,
            device=device,
            method=opts.weight_method,
            beta=opts.effective_beta,
            ignore_index=255
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean')
        print(f"✓ Using Weighted Cross-Entropy Loss (method: {opts.weight_method})\n")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        print("✓ Using Standard Cross-Entropy Loss (no class weights)\n")

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    def save_ckpt(path, epoch, include_epoch_in_name=False):
        """Save current model"""
        if include_epoch_in_name:
            base_path, ext = os.path.splitext(path)
            path = f"{base_path}_epoch{epoch:03d}{ext}"
        
        torch.save({
            "epoch": epoch,
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
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
                    print(f"  Skipping {key} due to size mismatch")
            else:
                print(f"  Skipping {key} as it does not exist in the current model")

        model.load_state_dict(new_state_dict, strict=False)
        print("Pretrained weights loaded successfully!\n")
        return model, checkpoint

    # Model loading
    utils.mkdir('checkpoints')
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        model, _ = load_pretrained_model(model, opts.ckpt, 
                                         num_classes_old=opts.pretrained_num_classes, 
                                         num_classes_new=opts.num_classes)
    else:
        print("[!] Training from scratch")

    # STAGE 1: Freeze backbone
    print("--- STAGE 1 SETUP: Training classifier only ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params_stage1 = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params=trainable_params_stage1, lr=opts.lr, 
                                momentum=0.9, weight_decay=opts.weight_decay)
    
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    model = nn.DataParallel(model)
    model.to(device)

    # Training loop
    cur_itrs = 0
    training_start_time = time.time()

    for epoch in range(1, opts.epochs + 1):
        epoch_start_time = time.time()
        
        # STAGE 2: Unfreeze backbone
        if epoch == opts.unfreeze_epoch:
            print(f"\n--- STAGE 2: Unfreezing backbone at Epoch {epoch} ---")
            
            for param in model.module.backbone.parameters():
                param.requires_grad = True
            
            print("Re-creating optimizer for fine-tuning the whole model...")
            optimizer = torch.optim.SGD(
                params=model.parameters(), 
                lr=opts.lr / 10,
                momentum=0.9, 
                weight_decay=opts.weight_decay
            )
            print(f"New optimizer created with LR = {opts.lr / 10}")

            remaining_itrs = opts.total_itrs - cur_itrs
            if opts.lr_policy == 'poly':
                scheduler = utils.PolyLR(optimizer, remaining_itrs, power=0.9)
            elif opts.lr_policy == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
        
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

        avg_epoch_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{opts.epochs} [{stage_str}] completed:")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.8f}")

        # Update Visdom (loss and learning rate only)
        if vis is not None:
            vis.vis_scalar('Training Loss', epoch, avg_epoch_loss)
            vis.vis_scalar('Learning Rate', epoch, current_lr)

        # Save checkpoints
        if epoch % 10 == 0:
            save_ckpt('checkpoints/checkpoint_%s_%s_os%d.pth' % 
                     (opts.model, opts.dataset, opts.output_stride), 
                     epoch, include_epoch_in_name=True)
        
        save_ckpt('checkpoints/latest_%s_%s_os%d.pth' % 
                 (opts.model, opts.dataset, opts.output_stride), epoch)
        
        # Time estimation
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = opts.epochs - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        print(f"  Epoch time: {format_time(epoch_time)} | Total elapsed: {format_time(total_elapsed)}")
        print(f"  Estimated remaining: {format_time(estimated_remaining)} | " +
              f"ETA: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_remaining))}")
        print()

    print("Training finished.")


if __name__ == "__main__":
    main()

# Visdom 서버 시작
# python -m visdom.server -port 28333

# 학습 실행
# python my_train_for_submit.py \
#     --model deeplabv3_mobilenet \
#     --ckpt ./checkpoints/best_deeplabv3_mobilenet_dna2025dataset_os16.pth \
#     --pretrained_num_classes 19 \
#     --lr 1e-5 \
#     --epochs 100 \
#     --unfreeze_epoch 1 \
#     --batch_size 4 \
#     --crop_size 1024 \
#     --enable_vis \
#     --vis_port 28333