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

    # Datset Options
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

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=833000,
                        help="epoch number (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument("--val_batch_size", type=int, default=2,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=1024)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=8300,
                        help="print interval of loss (default: 1)")
    parser.add_argument("--val_interval", type=int, default=8300,
                        help="epoch interval for eval (default: 5)")
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



# ==============================================================================
# 1. Transformation Class (Original code preserved)
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
        # The label will be converted to a tensor in __getitem__, so return a PIL Image object here.
        # label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

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
# 2. Dataset Class (Only file discovery part is modified)
# ==============================================================================
class SegmentationDataset(Dataset):
    colors = [
        (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70),
        (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30),
        (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180),
        (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230)
    ]
    # ❗️ Modified __init__ to accept a transform object directly
    def __init__(self, root_dir, image_paths, transform):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.label_paths = [self._get_label_path(p) for p in self.image_paths]
        # ❗️ The transform is now passed from outside
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
        클래스 인덱스로 구성된 마스크를 RGB 색상 이미지로 변환합니다.
        
        Args:
            mask (np.ndarray): (H, W) 형태의 레이블 맵. 각 픽셀 값은 클래스 인덱스입니다.
            
        Returns:
            np.ndarray: (H, W, 3) 형태의 RGB 이미지.
        """
        mask = np.array(mask, dtype=np.uint8)
        # (H, W, 3) 형태의 빈 컬러 이미지를 생성합니다.
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # 각 클래스 인덱스에 해당하는 픽셀에 정의된 색상을 적용합니다.
        for i, color in enumerate(self.colors):
            rgb_mask[mask == i] = color
            
        return rgb_mask

# ==============================================================================
# 3. Stratified Data Split Function (Newly added part)
# ==============================================================================
def create_stratified_split(root_dir, subset, val_ratio=0.2, random_state=42):
    """
    Splits the list of image files for each subfolder within the specified directory.
    """
    source_image_dir = os.path.join(root_dir, "image", subset)
    
    try:
        subfolders = sorted([f.name for f in os.scandir(source_image_dir) if f.is_dir()])
        if not subfolders:
            raise FileNotFoundError(f"No subfolders found in '{source_image_dir}'")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return [], []

    train_image_paths, val_image_paths = [], []
    print(f"Performing stratified split on the following subfolders: {subfolders}")

    for folder in subfolders:
        folder_path = os.path.join(source_image_dir, folder)
        files_in_folder = sorted(glob(os.path.join(folder_path, "*.*")))
        
        if len(files_in_folder) < 2:
            train_image_paths.extend(files_in_folder)
            continue
            
        train_files, val_files = train_test_split(
            files_in_folder, test_size=val_ratio, random_state=random_state
        )
        train_image_paths.extend(train_files)
        val_image_paths.extend(val_files)

    return train_image_paths, val_image_paths

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
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

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
        
    #fix by dh: add custom dataset
    if opts.dataset == '2025dna':
        ROOT_DIR = opts.data_root
        SUBSET_TO_SPLIT = "train"
        VAL_RATIO = 0.2
        CROP_SIZE = [1024, 1024]
        SCALE_RANGE = [0.75,1.25]
        BATCH_SIZE = 4

        # --- 1. Create train/validation file path lists ---
        train_paths, val_paths = create_stratified_split(
            root_dir=ROOT_DIR,
            subset=SUBSET_TO_SPLIT,
            val_ratio=VAL_RATIO
        )

        if not train_paths and not val_paths:
            print("No file paths were generated. Please check the ROOT_DIR.")
        else:
            print("\n--- Split Results ---")
            print(f"Total images: {len(train_paths) + len(val_paths)}")
            print(f"Training images: {len(train_paths)}")
            print(f"Validation images: {len(val_paths)}")

            # ❗️ Modified: Instantiate separate transforms for training and validation
            train_transform = SegmentationTransform(crop_size=CROP_SIZE, scale_range=SCALE_RANGE)
            val_transform = SegmentationValidationTransform(crop_size=CROP_SIZE)

            # --- 2. Create Dataset instances and pass the appropriate transform ---
            train_dst = SegmentationDataset(
                root_dir=ROOT_DIR,
                image_paths=train_paths,
                transform=train_transform # Pass the training transform
            )

            val_dst = SegmentationDataset(
                root_dir=ROOT_DIR,
                image_paths=val_paths,
                transform=val_transform # Pass the validation transform
            )

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
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
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

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

def visualize_batch(images, labels, dataset, num_images=4):
    """데이터셋의 이미지와 레이블 배치를 시각화합니다."""
    # 정규화를 되돌리기 위한 Denormalize 인스턴스 생성
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 표시할 이미지 수를 배치의 크기와 비교하여 제한
    num_images = min(num_images, len(images))

    # matplotlib Figure 생성
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 5 * num_images))
    fig.suptitle("Original Image vs. Label Map", fontsize=16)

    for i in range(num_images):
        # --- 원본 이미지 처리 ---
        # (C, H, W) -> (H, W, C) 변환 및 정규화 해제
        img = images[i].cpu().numpy()
        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)

        # --- 레이블 이미지 처리 ---
        # 레이블 ID를 컬러맵으로 변환
        label = labels[i].cpu().numpy()
        label_color = dataset.decode_target(label).astype(np.uint8)

        # --- 이미지 플로팅 ---
        # 표시할 이미지가 1개일 경우와 여러 개일 경우를 모두 처리
        ax_img = axes[i, 0] if num_images > 1 else axes[0]
        ax_label = axes[i, 1] if num_images > 1 else axes[1]

        ax_img.imshow(img)
        ax_img.set_title(f"Original Image #{i+1}")
        ax_img.axis('off') # 축 숨기기

        ax_label.imshow(label_color)
        ax_label.set_title(f"Label Map #{i+1}")
        ax_label.axis('off') # 축 숨기기

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 제목과 겹치지 않도록 레이아웃 조정
    plt.show()

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
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

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    print("\n--- DataLoaders Created ---")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # --- 4. Verify DataLoader operation ---
    try:
        images, labels = next(iter(train_loader))
        print("\n--- First Training Batch Info ---")
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")

        print("\n--- Visualizing a batch from the training set... ---")
        visualize_batch(images, labels, train_dst, num_images=4)

        val_images, val_labels = next(iter(val_loader))
        print("\n--- First Validation Batch Info ---")
        print(f"Image batch shape: {val_images.shape}")
        print(f"Label batch shape: {val_labels.shape}")
    except StopIteration:
        print("DataLoader is empty. Check the dataset path and number of files.")

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=21, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    # metrics = StreamSegMetrics(opts.num_classes)
    ### ✅ fix by dh: class num 19
    metrics = StreamSegMetrics(19)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    ### ✅ fix by dh: modify classes num 21 -> 19
    last_layers = model.module.classifier.classifier[1:]
    print(f"\n### original model ###\n{last_layers}")
    last_layer_in_channels = model.module.classifier.classifier[4].in_channels
    new_num_classes = 19
    print(f"✅ new class num: {new_num_classes}")
    new_last_layer = nn.Conv2d(last_layer_in_channels, new_num_classes, kernel_size=(1, 1), stride=(1, 1))
    new_last_layer = new_last_layer.to(device)  # 이 줄 추가
    model.module.classifier.classifier[4] = new_last_layer
    new_last_layers = model.module.classifier.classifier[1:]
    print(f"### modified model ###\n{new_last_layers}")

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in tqdm(train_loader, desc='Training'):
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 8300 == 0:
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()

# python my_train.py  --data_root ./datasets/data/2025dna/SemanticDataset_final --dataset 2025dna --model deeplabv3_mobilenet --enable_vis --vis_port 28333 --ckpt checkpoints/best_deeplabv3_mobilenet_voc_os16.pth