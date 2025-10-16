import os
import network
import argparse
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
# from DDRNet import DDRNet
from torch.utils.data import Dataset, DataLoader
import matplotlib.cm as cm
from collections import OrderedDict
import torch.nn as nn
from pathlib import Path

# 테스트 데이터셋을 위한 사용자 정의 Dataset 클래스
class TestSegmentationDataset(Dataset):
    def __init__(self, root_dir, subset='test', normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.image_dir = os.path.join(root_dir, "image", subset)
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*", "*.*"), recursive=True))
        self.to_tensor = transforms.ToTensor()
        
        # 정규화 설정
        self.normalize = normalize
        if self.normalize:
            self.normalize_transform = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.to_tensor(img)
        
        # 정규화 적용
        if self.normalize:
            tensor = self.normalize_transform(tensor)
            
        return tensor, img_path

# ✍️ 수정된 부분: 단일 GPU에 최적화된 안정적인 모델 로딩 함수
def load_model(weight_path, num_classes, device):
    model = network.modeling.__dict__['deeplabv3_mobilenet'](19, 16)
    
    print(f"Loading checkpoint: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print("Model loaded successfully")

    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    return model

# 예측 결과를 이미지 파일로 저장하는 함수
def save_prediction(pred, save_path, colormap_root, num_classes):
    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

    # --- 1. 흑백 레이블 이미지 저장 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(pred_np).save(save_path)

    # --- 2. 시각화를 위한 컬러맵 이미지 저장 ---
    normed = pred_np.astype(np.float32) / (num_classes - 1) # num_classes를 사용하도록 수정
    cmap = cm.get_cmap('turbo')
    colored = cmap(normed)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb)

    # 컬러맵 이미지를 저장할 경로 계산
    # 'label' 이라는 특정 문자열 대신 result_dir의 하위 폴더 구조를 기반으로 경로를 생성하도록 개선
    try:
        rel_path = os.path.relpath(save_path, start=os.path.dirname(save_path))
        cmap_path = os.path.join(colormap_root, os.path.dirname(os.path.relpath(save_path, start=args.result_dir)), rel_path)
    except ValueError: # 다른 드라이브에 있을 경우 대비
        rel_path = Path(save_path).name
        cmap_path = os.path.join(colormap_root, rel_path)
        
    os.makedirs(os.path.dirname(cmap_path), exist_ok=True)
    rgb_img.save(cmap_path)

# 전체 테스트(추론) 과정을 실행하는 메인 함수
def test(args):
    # 단일 GPU 사용을 명시적으로 설정 (cuda:0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TestSegmentationDataset(args.dataset_dir, subset=args.subset, 
                                     normalize=args.normalize, 
                                     mean=args.normalize_mean, 
                                     std=args.normalize_std)
    if not dataset.image_paths:
        print(f"Error: No images found in '{dataset.image_dir}'. Please check the path and subset.")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model = load_model(args.weight_path, args.num_classes, device)
    colormap_root = os.path.join(args.result_dir, "colormap")
    
    with torch.inference_mode(): # torch.no_grad()보다 더 효율적인 추론 모드
        for img_tensor, img_path_tuple in tqdm(dataloader, desc="Predicting..."):
            img_path = img_path_tuple[0] # 튜플에서 문자열 경로 추출
            img_tensor = img_tensor.to(device)
            
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]

            pred = torch.argmax(output, dim=1)

            # 결과 파일을 저장할 경로 생성
            rel_path = os.path.relpath(img_path, start=os.path.join(args.dataset_dir, "image"))
            save_path = os.path.join(args.result_dir, rel_path)

            save_prediction(pred, save_path, colormap_root, args.num_classes)

# 스크립트 실행 시작점
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data", help="Path to dataset root directory")
    parser.add_argument("--weight_path", type=str, default="./output/deeplabv3_mobilenet_dna2025dataset_baseline.pth", help="Path to model weight (.pth)")
    parser.add_argument("--result_dir", type=str, default="./result_deeplab", help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=19, help="Number of segmentation classes")
    parser.add_argument("--subset", type=str, default="test", help="Which subset to run prediction on (e.g., 'test', 'val', 'train')")
    parser.add_argument("--normalize", action="store_true", default=True, help="Apply normalization to input images")
    parser.add_argument("--normalize_mean", nargs=3, type=float, default=[0.485, 0.456, 0.406], help="Normalization mean values for RGB channels")
    parser.add_argument("--normalize_std", nargs=3, type=float, default=[0.229, 0.224, 0.225], help="Normalization std values for RGB channels")
    # input_size 인자는 모델 입력 크기가 고정된 경우 보통 사용되지 않으므로 제거 (필요 시 추가)
    
    args = parser.parse_args()
    
    # 결과 저장 폴더 생성
    os.makedirs(args.result_dir, exist_ok=True)
    
    test(args)
    
    # python prediction_cw.py --result_dir result_deeplab_non_norm