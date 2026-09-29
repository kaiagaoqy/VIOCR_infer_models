#!/usr/bin/env python3
"""
Generate edge map images for each text region in TotalText 16/.

Uses the official TotalText character GT masks (data/totaltext/chargt/).
Edge = 1-pixel morphological boundary of the GT ink mask.

Output: results/edgemap/<image_stem>/<idx>_<word>.png

Run from the repo root:
    python utils/eval/generate_edge_maps.py
"""
import json
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

GT_PATH       = Path('data/totaltext/anno.json')
CHARGT_DIR    = Path('data/totaltext/chargt')
OUTPUT_DIR    = Path('results/edgemap')
TARGET_PREFIX = '16/'

_CROSS3    = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_BAD_CHARS = re.compile(r'[^\w\-]')


def safe_name(text: str) -> str:
    return _BAD_CHARS.sub('_', text)[:30] or 'UNKNOWN'


def get_polygon_mask(polygon_coords, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], 255)
    return mask


def ink_mask_to_edge(ink_mask):
    """1-pixel boundary of an ink mask via morphological gradient."""
    return cv2.subtract(
        cv2.dilate(ink_mask, _CROSS3, iterations=1),
        cv2.erode(ink_mask,  _CROSS3, iterations=1),
    )


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

    gt_cache: dict = {}
    saved = 0

    for ann, info in tqdm(targets, desc='Generating edge maps'):
        file_name = info['file_name']
        stem      = Path(file_name).stem
        caption   = ann.get('caption', '')
        bbox_list = ann.get('bbox', [])

        if stem not in gt_cache:
            gt_path = CHARGT_DIR / f'{stem}.png'
            if not gt_path.exists():
                print(f"  Warning: GT mask not found for {stem}")
                gt_cache[stem] = None
            else:
                raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    gt_cache[stem] = None
                else:
                    _, gt_bin = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
                    gt_cache[stem] = gt_bin

        gt_full = gt_cache[stem]
        if gt_full is None:
            continue

        out_dir = OUTPUT_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, word in enumerate(caption.split()):
            if word == '###' or idx >= len(bbox_list):
                continue
            polygon = bbox_list[idx]
            if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
                continue

            poly_mask = get_polygon_mask(polygon, gt_full.shape)
            ink_mask  = cv2.bitwise_and(gt_full, gt_full, mask=poly_mask)
            edge_map  = ink_mask_to_edge(ink_mask)

            fname = out_dir / f'{idx:03d}_{safe_name(word)}.png'
            cv2.imwrite(str(fname), edge_map)
            saved += 1

    print(f"\nDone. Saved {saved} edge map images to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
