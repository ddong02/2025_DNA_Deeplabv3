import argparse
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np


def collect_images(image_base: Path, subfolders: List[str]) -> List[Path]:
    files = []
    for sub in subfolders:
        d = image_base / sub
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                files.append(p)
    return files


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def load_mask(path: Path):
    # read as-is; support single-channel and multi-channel masks
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    # If mask has 3 channels, convert to gray by taking the first channel
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m


def colorize_mask(mask: np.ndarray, colormap=cv2.COLORMAP_JET):
    if mask is None:
        return None
    # Convert mask to 0-255 uint8
    m = mask.astype(np.float32)
    minv, maxv = float(np.min(m)), float(np.max(m))
    if maxv == minv:
        if maxv == 0:
            m8 = np.zeros_like(m, dtype=np.uint8)
        else:
            m8 = np.full_like(m, 255, dtype=np.uint8)
    else:
        m8 = ((m - minv) / (maxv - minv) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(m8, colormap)
    return colored


def overlay_image(img: np.ndarray, colored_mask: np.ndarray, alpha=0.6):
    if colored_mask is None:
        return img.copy()
    # ensure same size
    if img.shape[:2] != colored_mask.shape[:2]:
        colored_mask = cv2.resize(colored_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(img, alpha, colored_mask, 1 - alpha, 0)
    return overlay


def make_side_by_side(img, overlay, max_height=900):
    # Scale both to same height if too large
    h = img.shape[0]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(img.shape[1] * scale), max_height), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, (int(overlay.shape[1] * scale), max_height), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([img, overlay])


def find_corresponding_mask(img_path: Path, image_base: Path, mask_base: Path, subfolders: List[str]) -> Path:
    # construct relative path under image_base and map to mask_base if subfolder structure matches
    try:
        rel = img_path.relative_to(image_base)
    except Exception:
        # build candidate mask names from the image stem when the image is not under image_base
        img_stem = img_path.stem
        candidates = [f"{img_stem}_CategoryId.png", f"{img_stem}_gtFine_CategoryId.png", f"{img_stem}.png"]
        if img_stem.endswith('_leftImg8bit'):
            base = img_stem[: -len('_leftImg8bit')]
            # prefer base_gtFine_CategoryId.png for datasets that use leftImg8bit naming
            candidates.insert(0, f"{base}_gtFine_CategoryId.png")
            candidates.insert(0, f"{base}_CategoryId.png")

        for sub in subfolders:
            for mask_name in candidates:
                cand = mask_base / sub / mask_name
                if cand.exists():
                    return cand
        return Path('')

    # When image is under image_base, try a few sensible mask name patterns
    img_stem = img_path.stem
    candidates = [f"{img_stem}_CategoryId.png", f"{img_stem}_gtFine_CategoryId.png", f"{img_stem}.png"]
    if img_stem.endswith('_leftImg8bit'):
        base = img_stem[: -len('_leftImg8bit')]
        candidates.insert(0, f"{base}_gtFine_CategoryId.png")
        candidates.insert(0, f"{base}_CategoryId.png")

    rel_parent = rel.parent
    # 1) same relative parent
    for mask_name in candidates:
        cand = mask_base / rel_parent / mask_name
        if cand.exists():
            return cand

    # 2) same top-level subfolder
    if len(rel.parts) >= 1:
        sub = rel.parts[0]
        for mask_name in candidates:
            cand2 = mask_base / sub / mask_name
            if cand2.exists():
                return cand2

    # 3) direct png fallback in same parent
    cand3 = mask_base / rel_parent / (img_stem + '.png')
    if cand3.exists():
        return cand3

    # 4) search candidates in provided mask subfolders
    for sub in subfolders:
        for mask_name in candidates:
            cand4 = mask_base / sub / mask_name
            if cand4.exists():
                return cand4

    return Path('')


def parse_args():
    p = argparse.ArgumentParser(description='Save original images side-by-side with colorized mask overlays.')
    default_image_base = 'datasets/data/SemanticDataset_final/image/train'
    default_mask_base = 'datasets/data/SemanticDataset_final/labelmap/train'
    p.add_argument('--image-base', default=default_image_base, help='Base directory containing image subfolders')
    p.add_argument('--mask-base', default=default_mask_base, help='Base directory containing mask subfolders')
    p.add_argument('--subfolders', nargs='+', default=['cam0', 'cam3', 'set1', 'set2', 'set3'], help='Subfolders to include')
    p.add_argument('--alpha', type=float, default=0.6, help='Overlay alpha for original image')
    p.add_argument('--max-height', type=int, default=900, help='Max displayed height in pixels')
    p.add_argument('--step', type=int, default=10, help='Number of images to skip when processing')
    p.add_argument('--output-dir', default='overlay_output', help='Output directory for saved images')
    p.add_argument('--start-index', type=int, default=0, help='Starting index for processing')
    p.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process (None for all)')
    return p.parse_args()


def main():
    args = parse_args()
    image_base = Path(args.image_base)
    mask_base = Path(args.mask_base)
    subfolders = args.subfolders
    output_dir = Path(args.output_dir)

    if not image_base.exists():
        print(f'Image base folder not found: {image_base}', file=sys.stderr)
        sys.exit(1)
    if not mask_base.exists():
        print(f'Warning: mask base folder not found: {mask_base}', file=sys.stderr)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir}')

    images = collect_images(image_base, subfolders)
    if not images:
        print('No images found. Check the --image-base and --subfolders args.', file=sys.stderr)
        sys.exit(1)

    # Filter images based on start index and max images
    start_idx = args.start_index
    if start_idx >= len(images):
        print(f'Start index {start_idx} is greater than total images {len(images)}', file=sys.stderr)
        sys.exit(1)
    
    end_idx = len(images)
    if args.max_images is not None:
        end_idx = min(start_idx + args.max_images, len(images))
    
    # Process every 'step' images starting from start_idx
    processed_count = 0
    total_to_process = (end_idx - start_idx) // args.step + (1 if (end_idx - start_idx) % args.step > 0 else 0)
    
    print(f'Processing {total_to_process} images (every {args.step} images from index {start_idx} to {end_idx-1})')
    
    for idx in range(start_idx, end_idx, args.step):
        img_path = images[idx]
        try:
            img = load_image(img_path)
        except Exception as e:
            print(f'Error loading image {img_path}: {e}', file=sys.stderr)
            continue

        mask_path = find_corresponding_mask(img_path, image_base, mask_base, subfolders)
        mask = None
        if mask_path and mask_path.exists():
            mask = load_mask(mask_path)
        else:
            mask = None

        colored = colorize_mask(mask) if mask is not None else None
        overlay = overlay_image(img, colored, alpha=args.alpha)
        side = make_side_by_side(img, overlay, max_height=args.max_height)

        # Create output filename
        rel_path = img_path.relative_to(image_base)
        output_filename = f"{rel_path.parent.name}_{rel_path.stem}_overlay.png"
        output_path = output_dir / output_filename

        # Add text overlay with information
        display = side.copy()
        info = f'{idx + 1}/{len(images)}  {rel_path}  step={args.step}'
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if mask is None:
            cv2.putText(display, 'MASK MISSING', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Save the image
        cv2.imwrite(str(output_path), display)
        processed_count += 1
        
        print(f'[{processed_count}/{total_to_process}] Saved: {output_path}')
        
        # Progress indicator
        if processed_count % 10 == 0:
            print(f'Progress: {processed_count}/{total_to_process} images processed')

    print(f'\n✅ Processing complete!')
    print(f'   Total images processed: {processed_count}')
    print(f'   Output directory: {output_dir}')
    print(f'   Step size: {args.step}')


if __name__ == '__main__':
    main()