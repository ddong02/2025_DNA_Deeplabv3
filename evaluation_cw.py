import os
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from pathlib import Path

def load_image(path):
    return np.array(Image.open(path)).astype(np.uint8)

def _find_label_path(file_basename, label_dir):
    """Find corresponding label file for a given image basename"""
    # Remove _leftImg8bit suffix if present
    if "_leftImg8bit" in file_basename:
        file_basename = file_basename.replace("_leftImg8bit", "")
    
    # Try different label naming conventions (from my_test.py)
    possible_label_names = [
        f"{file_basename}_gtFine_CategoryId.png",  # For set1: Daeduk_000009_gtFine_CategoryId.png
        f"{file_basename}_CategoryId.png",          # For cam folders: round(...)_CategoryId.png
        f"{file_basename}_labelmap.png",
        f"{file_basename}.png"
    ]
    
    # Search in all subdirectories of label_dir
    for label_name in possible_label_names:
        # Search recursively in label_dir
        label_paths = []
        label_paths.extend(glob(os.path.join(label_dir, "**", label_name), recursive=True))
        
        if label_paths:
            return label_paths[0]  # Return first match
    
    return None

def compute_miou(confusion, num_classes):
    """
    1. mIoU (All): NaN을 제외한 모든 클래스(IoU=0 포함)의 평균
    2. mIoU (>0): IoU가 0보다 큰 클래스들만의 평균
    """
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
    
    # mIoU (All Classes) 계산
    # np.nanmean은 리스트에서 NaN 값을 무시하고 평균을 계산.
    miou_all = np.nanmean(ious)
    
    # mIoU (IoU > 0 Classes Only) 계산
    # IoU가 NaN이 아니고 0보다 큰 값들만 계산.
    positive_ious = [iou for iou in ious if not np.isnan(iou) and iou > 0]
    
    # 필터링된 iou 값들의 평균을 계산합니다.
    if not positive_ious:
        miou_positive = 0.0
    else:
        miou_positive = np.mean(positive_ious) 
    
    return miou_all, miou_positive, ious

def evaluate(result_dir, label_dir, num_classes):
    # 모든 PNG, JPG 파일을 찾기
    pred_paths = []
    pred_paths.extend(glob(os.path.join(result_dir, "**", "*.png"), recursive=True))
    pred_paths.extend(glob(os.path.join(result_dir, "**", "*.jpg"), recursive=True))
    pred_paths.extend(glob(os.path.join(result_dir, "**", "*.jpeg"), recursive=True))
    pred_paths = sorted(pred_paths)
    
    print(f'Found {len(pred_paths)} segmentation result images in {result_dir}')
    
    if not pred_paths:
        print("Error: No prediction files found. Please check the 'result_dir' path and file names.")
        return

    all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred_path in tqdm(pred_paths, desc="Evaluating"):
        # 파일명에서 확장자 제거하여 기본 파일명 추출
        file_basename = os.path.splitext(os.path.basename(pred_path))[0]
        
        # my_test.py의 _get_label_path 로직을 참고하여 레이블 파일 찾기
        label_path = _find_label_path(file_basename, label_dir)
        
        if not label_path or not os.path.exists(label_path):
            print(f"Label not found for {file_basename}, skipping.")
            continue

        # 이미지 로드
        pred_img = Image.open(pred_path)
        label_img = Image.open(label_path)
        
        # 크기가 다르면 레이블 이미지를 예측 이미지 크기로 리사이즈
        if pred_img.size != label_img.size:
            print(f"Resizing label from {label_img.size} to {pred_img.size}")
            label_img = label_img.resize(pred_img.size, Image.NEAREST)
        
        # numpy 배열로 변환
        pred = np.array(pred_img).astype(np.uint8)
        label = np.array(label_img).astype(np.uint8)
        
        # 1차원으로 변환
        pred = pred.flatten()
        label = label.flatten()

        mask = label != 255
        pred = pred[mask]
        label = label[mask]
        
        pred = np.clip(pred, 0, num_classes - 1)
        label = np.clip(label, 0, num_classes - 1)

        conf = confusion_matrix(label, pred, labels=list(range(num_classes)))
        all_confusion += conf

    miou_all, miou_positive, ious = compute_miou(all_confusion, num_classes)
    
    print("\n--- Evaluation Results ---")
    print(f"📊 mIoU (All Classes, IoU=0 포함): {miou_all:.4f}")
    print(f"📊 mIoU (Positive Classes, IoU>0 제외): {miou_positive:.4f}")
    print("--------------------------")
    
    for i, iou in enumerate(ious):
        print(f"Class {i}: IoU = {iou:.4f}" if not np.isnan(iou) else f"Class {i}: IoU = NaN (ignored in mean)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mIoU for semantic segmentation results.")
    
    parser.add_argument("--result_dir", type=str, default="C:/ETRI/result/test", 
                        help="Predicted *_leftImg8bit.png files가 있는 상위 디렉토리")
    parser.add_argument("--label_dir", type=str, default="C:/ETRI/data/labelmap/test", 
                        help="정답 레이블 *_gtFine_CategoryId.png files가 있는 상위 디렉토리")
    parser.add_argument("--num_classes", type=int, default=19, help="세그먼테이션 클래스 수")

    args = parser.parse_args()
    
    evaluate(args.result_dir, args.label_dir, args.num_classes)

# python evaluation_cw.py \
#     --result_dir result_deeplab_cw/test \
#     --label_dir datasets/data/SemanticDataset_final/labelmap/test