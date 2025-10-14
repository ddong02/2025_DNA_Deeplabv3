# import os
# import argparse
# import numpy as np
# from PIL import Image
# from glob import glob
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix

# def load_image(path):
#     return np.array(Image.open(path)).astype(np.uint8)

# def compute_miou(confusion, num_classes):
#     ious = []
#     for cls in range(num_classes):
#         TP = confusion[cls, cls]
#         FP = confusion[:, cls].sum() - TP
#         FN = confusion[cls, :].sum() - TP
        
#         denom = TP + FP + FN
#         if denom == 0:
#             iou = float('nan')
#         else:
#             iou = TP / denom
#         ious.append(iou)
    
#     miou = np.nanmean(ious)
#     return miou, ious

# def evaluate(result_dir, label_dir, num_classes):
#     # ✍️ 수정된 부분: glob이 하위 폴더까지 모두 검색하도록 recursive=True 옵션 추가
#     pred_paths = sorted(glob(os.path.join(result_dir, "**", "*_leftImg8bit.png"), recursive=True))
#     print(f'Found {len(pred_paths)} segmentation result images in {result_dir}')
    
#     if not pred_paths:
#         print("Error: No prediction files found. Please check the 'result_dir' path and file names.")
#         return None, None

#     all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

#     for pred_path in tqdm(pred_paths, desc="Evaluating"):
#         # 예측 파일이 있는 폴더 이름(예: 'set1')을 가져와서 정답 경로를 만듭니다.
#         sub_folder = Path(pred_path).parent.name
#         file_id = os.path.basename(pred_path).replace("_leftImg8bit.png", "")
        
#         # 정답 레이블 파일 경로를 동적으로 생성
#         label_path = os.path.join(label_dir, sub_folder, f"{file_id}_gtFine_CategoryId.png")

#         if not os.path.exists(label_path):
#             print(f"Label not found at {label_path}, skipping.")
#             continue

#         pred = load_image(pred_path).flatten()
#         label = load_image(label_path).flatten()

#         mask = label != 255
#         pred = pred[mask]
#         label = label[mask]
        
#         # 라벨과 예측 값의 범위가 num_classes를 벗어나지 않도록 클리핑
#         pred = np.clip(pred, 0, num_classes - 1)
#         label = np.clip(label, 0, num_classes - 1)

#         conf = confusion_matrix(label, pred, labels=list(range(num_classes)))
#         all_confusion += conf

#     miou, ious = compute_miou(all_confusion, num_classes)
    
#     print(f"\n📊 mIoU: {miou:.4f}")
#     for i, iou in enumerate(ious):
#         print(f"Class {i}: IoU = {iou:.4f}" if not np.isnan(iou) else f"Class {i}: IoU = NaN (ignored)")

#     return miou, ious

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Calculate mIoU for semantic segmentation results.")
    
#     # ✍️ 수정된 부분: 기본 경로를 사용자의 폴더 구조에 맞게 수정
#     # 예측 결과가 저장된 상위 폴더 (예: C:/ETRI/result/test)
#     parser.add_argument("--result_dir", type=str, default="C:/ETRI/result/test", 
#                         help="Predicted *_leftImg8bit.png files가 있는 상위 디렉토리")
#     # 정답 레이블이 있는 상위 폴더 (예: C:/ETRI/data/labelmap/test)
#     parser.add_argument("--label_dir", type=str, default="C:/ETRI/data/labelmap/test", 
#                         help="정답 레이블 *_gtFine_CategoryId.png files가 있는 상위 디렉토리")
#     parser.add_argument("--num_classes", type=int, default=19, help="세그먼테이션 클래스 수")

#     args = parser.parse_args()
    
#     evaluate(args.result_dir, args.label_dir, args.num_classes)


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
    pred_paths = sorted(glob(os.path.join(result_dir, "**", "*_leftImg8bit.png"), recursive=True))
    print(f'Found {len(pred_paths)} segmentation result images in {result_dir}')
    
    if not pred_paths:
        print("Error: No prediction files found. Please check the 'result_dir' path and file names.")
        return

    all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred_path in tqdm(pred_paths, desc="Evaluating"):
        sub_folder = Path(pred_path).parent.name
        file_id = os.path.basename(pred_path).replace("_leftImg8bit.png", "")
        
        label_path = os.path.join(label_dir, sub_folder, f"{file_id}_gtFine_CategoryId.png")

        if not os.path.exists(label_path):
            print(f"Label not found at {label_path}, skipping.")
            continue

        pred = load_image(pred_path).flatten()
        label = load_image(label_path).flatten()

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