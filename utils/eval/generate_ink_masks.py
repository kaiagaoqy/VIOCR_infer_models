#!/usr/bin/env python3
"""
Generate ink-area mask images for each text region in TotalText 16/.

Uses the official TotalText character-level ground-truth masks (binary, white=ink)
instead of morphological approximation.  For each word-level polygon annotation:
  1. Load the corresponding GT binary mask (data/totaltext/chargt/<stem>.png)
     — GT is at the original resolution; resize to match our (possibly downsampled) image
  2. Threshold to binary, mask to the polygon interior
  3. Save as PNG to results/inkarea/<image_stem>/<idx>_<word>.png

Run from the repo root:
    python utils/eval/generate_ink_masks.py
"""
import json
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

GT_PATH    = Path('data/totaltext/anno.json')
IMAGE_BASE = Path('data/totaltext')
CHARGT_DIR = Path('data/totaltext/chargt')   # GT masks, one per image stem
OUTPUT_DIR = Path('results/inkarea')
TARGET_PREFIX = '16/'

_BAD_CHARS = re.compile(r'[^\w\-]')


def safe_name(text: str) -> str:
    return _BAD_CHARS.sub('_', text)[:30] or 'UNKNOWN'


def get_polygon_mask(polygon_coords, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], 255)
    return mask


def main():
    print(f"Loading annotations from {GT_PATH} ...")
    with open(GT_PATH) as f:
        gt = json.load(f)

    img_info = {img['id']: img for img in gt['images']}
    targets = [
        (ann, img_info[ann['image_id']])
        for ann in gt['annotations']
        if img_info.get(ann['image_id'], {}).get('file_name', '').startswith(TARGET_PREFIX)
    ]
    print(f"Found {len(targets)} annotations in {TARGET_PREFIX}")

    gt_cache: dict = {}   # stem -> binary GT mask (resized to our image size)
    saved = 0
    missing_gt = set()

    for ann, info in tqdm(targets, desc='Generating ink masks'):
        file_name = info['file_name']
        stem      = Path(file_name).stem          # e.g. '0000205'
        caption   = ann.get('caption', '')
        bbox_list = ann.get('bbox', [])

        if stem not in gt_cache:
            gt_path = CHARGT_DIR / f'{stem}.png'
            if not gt_path.exists():
                missing_gt.add(stem)
                gt_cache[stem] = None
            else:
                # Load GT and resize to match our image dimensions
                src_path = IMAGE_BASE / file_name
                src = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
                gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if src is None or gt_raw is None:
                    gt_cache[stem] = None
                else:
                    H, W = src.shape
                    if gt_raw.shape != (H, W):
                        gt_raw = cv2.resize(gt_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                    _, gt_bin = cv2.threshold(gt_raw, 127, 255, cv2.THRESH_BINARY)
                    gt_cache[stem] = gt_bin

        gt_mask_full = gt_cache.get(stem)
        if gt_mask_full is None:
            continue

        out_dir = OUTPUT_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, word in enumerate(caption.split()):
            if word == '###':
                continue
            if idx >= len(bbox_list):
                continue
            polygon = bbox_list[idx]
            if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
                continue

            poly_mask = get_polygon_mask(polygon, gt_mask_full.shape)
            ink_mask  = cv2.bitwise_and(gt_mask_full, gt_mask_full, mask=poly_mask)

            fname = out_dir / f'{idx:03d}_{safe_name(word)}.png'
            cv2.imwrite(str(fname), ink_mask)
            saved += 1

    if missing_gt:
        print(f"  Warning: no GT found for: {sorted(missing_gt)}")
    print(f"\nDone. Saved {saved} ink mask images to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
