import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
from tqdm import tqdm
import argparse


def save_color_palette_visualization(dataset, save_path='dataset_visualization/color_palette.png'):
    """
    클래스별 색상 팔레트를 시각화하여 저장
    
    Args:
        dataset: DNA2025Dataset 인스턴스
        save_path: 저장할 파일 경로
    """
    num_classes = len(dataset.class_names)
    
    # Figure 생성
    fig, ax = plt.subplots(figsize=(12, num_classes * 0.5))
    
    # 각 클래스별로 색상 박스와 이름 표시
    for i in range(num_classes):
        color = dataset.color_palette[i] / 255.0  # Normalize to [0, 1]
        class_name = dataset.class_names[i]
        
        # 색상 박스 그리기
        rect = mpatches.Rectangle((0, num_classes - i - 1), 2, 0.8, 
                                   facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # 클래스 ID 표시
        ax.text(-0.3, num_classes - i - 0.6, f'{i:2d}', 
                ha='right', va='center', fontsize=11, fontweight='bold')
        
        # 클래스 이름 표시
        ax.text(2.3, num_classes - i - 0.6, class_name, 
                ha='left', va='center', fontsize=11)
        
        # RGB 값 표시
        rgb_text = f'RGB({dataset.color_palette[i][0]}, {dataset.color_palette[i][1]}, {dataset.color_palette[i][2]})'
        ax.text(8, num_classes - i - 0.6, rgb_text, 
                ha='left', va='center', fontsize=9, family='monospace',
                color='gray')
    
    # 축 설정
    ax.set_xlim(-1, 14)
    ax.set_ylim(-0.2, num_classes)
    ax.axis('off')
    
    # 제목 추가
    plt.title('DNA2025 Dataset - Class Color Palette', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Color palette saved to: {save_path}")


def visualize_dataset_samples(dataset, num_samples=5, save_dir='dataset_visualization', subset_name='train'):
    """
    데이터셋 샘플을 시각화하여 저장
    
    Args:
        dataset: DNA2025Dataset 인스턴스
        num_samples: 저장할 샘플 수
        save_dir: 저장 디렉토리
        subset_name: 서브셋 이름 (train/val/test)
    """
    # 저장 디렉토리 생성
    subset_dir = os.path.join(save_dir, subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"  Visualizing {subset_name.upper()} Dataset Samples")
    print(f"{'='*80}")
    print(f"Total samples in {subset_name} set: {len(dataset)}")
    print(f"Saving {num_samples} samples to: {subset_dir}\n")
    
    # 랜덤하게 샘플 선택
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Denormalization을 위한 설정
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc=f"Saving {subset_name} samples")):
        try:
            # 원본 이미지와 레이블 로드 (transform 적용 전)
            img_path = dataset.image_paths[sample_idx]
            label_path = dataset.label_paths[sample_idx]
            
            # 원본 이미지 로드
            original_img = Image.open(img_path).convert('RGB')
            original_img_np = np.array(original_img)
            
            # 원본 레이블 로드
            original_label = Image.open(label_path).convert('L')
            original_label_np = np.array(original_label)
            
            # Transform 적용된 데이터 로드
            transformed_img, transformed_label = dataset[sample_idx]
            
            # Transformed 이미지를 다시 numpy로 변환 (denormalize)
            if isinstance(transformed_img, torch.Tensor):
                transformed_img_np = transformed_img.cpu().numpy()
                # Denormalize: (C, H, W) -> (H, W, C)
                transformed_img_np = transformed_img_np.transpose(1, 2, 0)
                transformed_img_np = (transformed_img_np * std + mean) * 255
                transformed_img_np = np.clip(transformed_img_np, 0, 255).astype(np.uint8)
            
            # Transformed 레이블을 numpy로 변환
            if isinstance(transformed_label, torch.Tensor):
                transformed_label_np = transformed_label.cpu().numpy()
            else:
                transformed_label_np = np.array(transformed_label)
            
            # 컬러맵 생성 (원본 및 transformed)
            original_colormap = dataset.decode_target(original_label_np)
            transformed_colormap = dataset.decode_target(transformed_label_np)
            
            # 시각화: 2행 3열
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.2)
            
            # 파일명 추출
            img_filename = os.path.basename(img_path)
            
            # 상단 행 - 원본 데이터
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(original_img_np)
            ax1.set_title(f'Original Image\n{img_filename}', fontsize=10)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(original_colormap)
            ax2.set_title(f'Original Label (Colormap)\nSize: {original_label_np.shape}', fontsize=10)
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(original_label_np, cmap='tab20', vmin=0, vmax=18)
            ax3.set_title(f'Original Label (Raw)\nClasses: {np.unique(original_label_np)}', fontsize=10)
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # 하단 행 - 변환된 데이터
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(transformed_img_np)
            ax4.set_title(f'Transformed Image\nSize: {transformed_img_np.shape[:2]}', fontsize=10)
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(transformed_colormap)
            ax5.set_title(f'Transformed Label (Colormap)\nSize: {transformed_label_np.shape}', fontsize=10)
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 2])
            im6 = ax6.imshow(transformed_label_np, cmap='tab20', vmin=0, vmax=18)
            ax6.set_title(f'Transformed Label (Raw)\nClasses: {np.unique(transformed_label_np)}', fontsize=10)
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, fraction=0.046)
            
            # 전체 제목
            fig.suptitle(f'{subset_name.upper()} Dataset - Sample {idx+1}/{num_samples} (Index: {sample_idx})', 
                        fontsize=14, fontweight='bold')
            
            # 저장
            save_path = os.path.join(subset_dir, f'sample_{idx:03d}_idx{sample_idx:05d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # 클래스 정보 저장
    class_info_path = os.path.join(subset_dir, 'class_info.txt')
    with open(class_info_path, 'w', encoding='utf-8') as f:
        f.write(f"{subset_name.upper()} Dataset Class Information\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Classes: {len(dataset.class_names)}\n\n")
        f.write(f"{'ID':<4} {'Class Name':<30} {'RGB Color':<20}\n")
        f.write("-"*80 + "\n")
        for i, (name, color) in enumerate(zip(dataset.class_names, dataset.color_palette)):
            f.write(f"{i:<4} {name:<30} {str(tuple(color)):<20}\n")
    
    print(f"\n✓ {subset_name.upper()} dataset visualization completed!")
    print(f"  Saved {len(sample_indices)} samples to: {subset_dir}")
    print(f"  Class information saved to: {class_info_path}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize DNA2025 Dataset')
    parser.add_argument('--data_root', type=str, default='./datasets/data',
                        help='path to dataset root')
    parser.add_argument('--crop_size', type=int, default=1024,
                        help='crop size for transforms')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples to visualize per subset')
    parser.add_argument('--save_dir', type=str, default='dataset_visualization',
                        help='directory to save visualizations')
    parser.add_argument('--subsets', type=str, nargs='+', default=['train', 'val'],
                        choices=['train', 'val', 'test'],
                        help='which subsets to visualize')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='random seed for reproducibility')
    
    args = parser.parse_args()
    
    # main.py에서 DNA2025Dataset을 import
    # 같은 디렉토리에 main.py가 있다고 가정
    from my_train8 import DNA2025Dataset
    
    print("="*80)
    print("  DNA2025 Dataset Visualization Tool")
    print("="*80)
    
    # 각 subset에 대해 시각화 수행
    for subset in args.subsets:
        print(f"\nLoading {subset} dataset...")
        
        if subset == 'train':
            scale_range = [0.75, 1.25]
        else:
            scale_range = None
        
        dataset = DNA2025Dataset(
            root_dir=args.data_root,
            crop_size=[args.crop_size, args.crop_size],
            subset=subset,
            scale_range=scale_range,
            random_seed=args.random_seed
        )
        
        # 색상 팔레트 저장 (한 번만)
        if subset == args.subsets[0]:
            save_color_palette_visualization(
                dataset=dataset,
                save_path=os.path.join(args.save_dir, 'color_palette.png')
            )
        # 샘플 시각화
        visualize_dataset_samples(
            dataset=dataset,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            subset_name=subset
        )
    
    print("\n" + "="*80)
    print("  All visualizations completed!")
    print(f"  Results saved to: {args.save_dir}")
    print("="*80)


if __name__ == '__main__':
    main()