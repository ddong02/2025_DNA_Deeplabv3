import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import os
import argparse
from glob import glob
import cv2
import numpy as np
import time
import statistics

import network
from datasets import VOCSegmentation, Cityscapes

def get_argparser():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Interactive Prediction")
    
    # 디렉토리 기반 인자로 변경
    parser.add_argument("--input_dir", type=str, required=True,
                        help="path to the directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="path to the directory to save segmented output images")
    
    # 기존 인자 유지
    parser.add_argument("--model", type=str, required=True,
                        choices=['deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to the fine-tuned checkpoint")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes in your new dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes', choices=['voc', 'cityscapes'], 
                        help='Name of dataset for decoding colors')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    
    return parser

def main():
    opts = get_argparser().parse_args()

    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 모델 정의 및 체크포인트 로드
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {opts.ckpt}")
    
    checkpoint = torch.load(opts.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    print(f"Successfully loaded fine-tuned model from {opts.ckpt}")
    print(f"### model ###{model}")
    # 이미지 변환 로직
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.456],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    # 색상 디코딩 함수 설정
    if opts.dataset.lower() == 'voc':
        decode_fn = VOCSegmentation.decode_target
    else: # cityscapes 또는 기본값
        decode_fn = Cityscapes.decode_target

    # 입력 디렉토리에서 이미지 파일 목록 가져오기 (이름순으로 정렬)
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(sorted(glob(os.path.join(opts.input_dir, ext))))
    
    if not image_files:
        print(f"No images found in {opts.input_dir}")
        return

    # 출력 디렉토리 생성
    os.makedirs(opts.output_dir, exist_ok=True)
    
    inference_times = []

    # 각 이미지에 대해 연속 예측 수행
    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            print(f"\n--- processing image {i + 1}/{len(image_files)}: {os.path.basename(img_path)} ---")
            
            img_original = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_original).unsqueeze(0).to(device)
            
            width, height = img_original.size
            
            # 모델 추론 시간 측정
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            pred = model(img_tensor).max(1)[1].cpu()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            
            inference_times.append(inference_time)
            
            mean_sec = statistics.mean(inference_times)
            median_sec = statistics.median(inference_times)
            
            mean_ms = mean_sec * 1000
            median_ms = median_sec * 1000
            
            fps_mean = 1.0 / mean_sec if mean_sec > 0 else 0
            fps_median = 1.0 / median_sec if median_sec > 0 else 0
            
            # 결과 디코딩 및 마스크 생성
            colorized_preds = decode_fn(pred.squeeze(0).numpy()).astype('uint8')
            mask = Image.fromarray(colorized_preds)
            mask = mask.resize((width, height), resample=Image.NEAREST)
            
            # OpenCV용 이미지로 변환 및 합성
            original_cv = cv2.cvtColor(np.array(img_original), cv2.COLOR_RGB2BGR)
            mask_cv = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
            combined_view = cv2.hconcat([original_cv, mask_cv])
            # --- 여기까지 ---
            
            # 합성된 이미지 저장
            base_filename = os.path.basename(img_path)
            combined_filename = "combined_" + base_filename
            output_path_combined = os.path.join(opts.output_dir, combined_filename)
            cv2.imwrite(output_path_combined, combined_view)
            print(f"Combined view saved to {output_path_combined}")
            
            print(f"Current Inference: {inference_time*1000:.3f} ms")
            print("--------- Cumulative Stats ---------")
            print(f"Average: {mean_ms:.3f} ms   (~{fps_mean:.2f} FPS)")
            print(f"Median:  {median_ms:.3f} ms   (~{fps_median:.2f} FPS)")
            print("------------------------------------")

            # 마지막 이미지인 경우 알림 후 종료
            if i == len(image_files) - 1:
                print("\nLast image processed. All predictions are complete.")
                break

            # 사용자 입력 대기
            user_input = input("Press Enter to process the next image, or type 'q' to quit: ")
            if user_input.lower() == 'q':
                print("Exiting...")
                break
            
    print("\nScript finished.")


if __name__ == '__main__':
    main()

# python my_predict2.py \
# --input_dir datasets/data/SemanticDataset_final/image/train/set3 \
# --output_dir test_result_v3_mobilenet \
# --model deeplabv3_mobilenet \
# --ckpt checkpoints/best_deeplabv3_mobilenet_2025dna_os8.pth

# python my_predict2.py \
# --input_dir datasets/data/SemanticDatasetTest/image/test/set1 \
# --output_dir test_result_v3_mobilenet_test_images \
# --model deeplabv3_mobilenet \
# --ckpt checkpoints/best_deeplabv3_mobilenet_2025dna_os8.pth