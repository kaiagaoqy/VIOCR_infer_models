#!/usr/bin/env python3
"""
Visualize detections and ground truth for each subject and image
"""
import json
import ast
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
import numpy as np
from PIL import Image

def load_data(model_output_path, gt_path):
    """Load model output and ground truth data"""
    # Load model output
    model_output_path = Path(model_output_path)
    model_output = []
    
    if model_output_path.is_dir():
        for json_file in sorted(model_output_path.glob('*.json')):
            subject_name = json_file.stem
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for record in data:
                        record['_subject'] = subject_name
                    model_output.extend(data)
    
    # Load ground truth
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    return model_output, gt_data

def organize_detections(model_output, image_info, image_base_path):
    """Organize detections by (subject, image_id) and flip y-axis"""
    det_by_image = defaultdict(list)
    image_heights = {}  # Cache for image heights
    
    for det in model_output:
        subject = det.get('_subject', 'unknown')
        image_id = det.get('image_id')
        unique_key = (subject, image_id)
        
        # Get actual image height from file
        if image_id not in image_heights:
            img_info = image_info.get(image_id, {})
            image_path = Path(image_base_path) / img_info.get('file_name', '')
            try:
                with Image.open(image_path) as img:
                    image_heights[image_id] = img.height
            except:
                # Fallback to default
                image_heights[image_id] = 720
        
        image_height = image_heights[image_id]
        
        # Parse rec_texts
        rec_texts = det.get('rec_texts', '')
        if isinstance(rec_texts, str) and rec_texts.startswith('['):
            try:
                rec_texts = ast.literal_eval(rec_texts)
            except:
                rec_texts = [rec_texts]
        elif not isinstance(rec_texts, list):
            rec_texts = [rec_texts] if rec_texts else []
        
        # Parse polys
        polys = det.get('polys', [])
        
        for i, poly in enumerate(polys):
            if isinstance(poly, list):
                polygon_points = poly
                if polygon_points and isinstance(polygon_points[0], list) and len(polygon_points[0]) > 2:
                    polygon_points = [[p[0], p[1]] for p in polygon_points]

                # FLIP Y-AXIS for human data (y is inverted)
                if len(polygon_points) >= 1:
                    polygon_points = [[x, image_height - y] for x, y in polygon_points]

                text = rec_texts[i] if i < len(rec_texts) else ''
                det_by_image[unique_key].append({
                    'bbox': polygon_points,
                    'text': text,
                    'order': i + 1  # 1-indexed, matches CSV Detection Order
                })
    
    return det_by_image

def organize_ground_truth(gt_data):
    """Organize GT by image_id"""
    gt_images = gt_data.get('images', [])
    gt_annotations = gt_data.get('annotations', [])
    
    image_info = {img['id']: img for img in gt_images}
    gt_by_image = defaultdict(list)
    
    for ann in gt_annotations:
        image_id = ann['image_id']
        caption = ann.get('caption', '')
        words = caption.split()
        bbox_list = ann.get('bbox', [])
        
        for i, word in enumerate(words):
            if i < len(bbox_list):
                polygon = bbox_list[i]
                gt_by_image[image_id].append({
                    'bbox': polygon,
                    'text': word
                })
    
    return gt_by_image, image_info

def visualize_image(subject, image_id, detections, ground_truths, image_name, image_base_path, save_path=None):
    """Visualize detections and GT for a single image"""
    # Load the actual image
    image_path = Path(image_base_path) / image_name
    
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        img_array = None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Plot detections on original image
    ax1.set_title(f'{subject} - Image {image_id}: Detections\n{image_name}', fontsize=14, fontweight='bold')
    if img_array is not None:
        ax1.imshow(img_array)
    ax1.axis('off')
    
    no_poly_labels = []
    for i, det in enumerate(detections):
        bbox = det['bbox']
        text = det['text']
        order = det.get('order', i)
        label = f'[{order}] {text}'
        n_pts = len(bbox)

        if n_pts >= 3:
            poly = MPLPolygon(bbox, fill=False, edgecolor='red', linewidth=3, alpha=0.8)
            ax1.add_patch(poly)
            center_x = np.mean([p[0] for p in bbox])
            center_y = np.mean([p[1] for p in bbox])
            ax1.text(center_x, center_y, label,
                    fontsize=11, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                    ha='center', va='center')
        elif n_pts == 2:
            xs = [bbox[0][0], bbox[1][0]]
            ys = [bbox[0][1], bbox[1][1]]
            ax1.plot(xs, ys, color='red', linewidth=3, alpha=0.8)
            ax1.text(np.mean(xs), np.mean(ys), label,
                    fontsize=11, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                    ha='center', va='center')
        elif n_pts == 1:
            ax1.plot(bbox[0][0], bbox[0][1], 'o', color='red', markersize=10, alpha=0.8)
            ax1.text(bbox[0][0] + 5, bbox[0][1], label,
                    fontsize=11, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                    ha='left', va='center')
        else:
            no_poly_labels.append(label)

    # Draw labels for 0-point detections in bottom-left corner (axis coordinates)
    for j, lbl in enumerate(no_poly_labels):
        ax1.text(0.02, 0.03 + j * 0.04, lbl,
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', alpha=0.8),
                ha='left', va='bottom', transform=ax1.transAxes)
    
    # Plot ground truth on original image
    ax2.set_title(f'{subject} - Image {image_id}: Ground Truth\n{image_name}', fontsize=14, fontweight='bold')
    if img_array is not None:
        ax2.imshow(img_array)
    ax2.axis('off')
    
    for i, gt in enumerate(ground_truths):
        bbox = gt['bbox']
        text = gt['text']
        
        # Draw polygon
        color = 'gray' if text == '###' else 'lime'
        linestyle = '--' if text == '###' else '-'
        poly = MPLPolygon(bbox, fill=False, edgecolor=color, linewidth=3, linestyle=linestyle, alpha=0.8)
        ax2.add_patch(poly)
        
        # Add text label
        if bbox:
            center_x = np.mean([p[0] for p in bbox])
            center_y = np.mean([p[1] for p in bbox])
            bg_color = 'gray' if text == '###' else 'lime'
            ax2.text(center_x, center_y, f'{text}', 
                    fontsize=12, color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    # Configuration
    MODEL_OUTPUT_PATH = 'data/human/lowviz/converted'
    GT_PATH = 'data/totaltext/anno.json'
    IMAGE_BASE_PATH = 'data/totaltext'  # Base path for images
    OUTPUT_DIR = 'results/visualizations'
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    model_output, gt_data = load_data(MODEL_OUTPUT_PATH, GT_PATH)
    
    print("Organizing ground truth...")
    gt_by_image, image_info = organize_ground_truth(gt_data)
    
    print("Organizing detections (flipping y-axis with actual image heights)...")
    det_by_image = organize_detections(model_output, image_info, IMAGE_BASE_PATH)
    
    print(f"\nFound {len(det_by_image)} unique (subject, image_id) combinations")
    print(f"Found {len(gt_by_image)} GT images")
    
    # Visualize a few examples
    num_visualized = 0
    max_visualize = 20  # Visualize first 5 images per subject
    
    # Group by subject
    subjects = set(key[0] for key in det_by_image.keys())
    
    for subject in sorted(subjects):
        if subject == 'unknown':
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {subject}")
        print('='*80)
        
        # Get all images for this subject
        subject_images = [(key, det_by_image[key]) for key in det_by_image.keys() 
                         if key[0] == subject]
        
        for idx, (unique_key, detections) in enumerate(subject_images[:max_visualize]):
            _, image_id = unique_key
            ground_truths = gt_by_image.get(image_id, [])
            image_name = image_info.get(image_id, {}).get('file_name', f'image_{image_id}')
            
            print(f"\n{subject} - Image {image_id} ({image_name}):")
            print(f"  Detections: {len(detections)}")
            print(f"  GT: {len(ground_truths)}")
            
            # Print detection texts
            if detections:
                det_texts = [d['text'] for d in detections[:5]]
                print(f"  Detection texts (first 5): {det_texts}")
            
            # Print GT texts
            if ground_truths:
                gt_texts = [g['text'] for g in ground_truths[:5]]
                print(f"  GT texts (first 5): {gt_texts}")
            
            # Visualize image_name.split('/')[-1] image_name.split('/')[-1]
            name = '_'.join([image_name.split('/')[0],image_name.split('/')[-1].split('.')[0]])
            save_path = Path(OUTPUT_DIR) / f"{subject}_image_{name}.png"
            visualize_image(subject, image_id, detections, ground_truths, image_name, IMAGE_BASE_PATH, save_path)
            
            num_visualized += 1
    
    print(f"\n{'='*80}")
    print(f"Visualization complete!")
    print(f"Total images visualized: {num_visualized}")
    print(f"Output directory: {OUTPUT_DIR}")
    print('='*80)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for subject in sorted(subjects):
        if subject == 'unknown':
            continue
        
        subject_keys = [key for key in det_by_image.keys() if key[0] == subject]
        total_dets = sum(len(det_by_image[key]) for key in subject_keys)
        total_gts = sum(len(gt_by_image[key[1]]) for key in subject_keys)
        
        print(f"\n{subject}:")
        print(f"  Images: {len(subject_keys)}")
        print(f"  Total detections: {total_dets}")
        print(f"  Total GT words: {total_gts}")
        
        # Sample polygon info
        if subject_keys:
            sample_key = subject_keys[0]
            sample_dets = det_by_image[sample_key]
            if sample_dets:
                sample_poly = sample_dets[0]['bbox']
                print(f"  Sample polygon points: {len(sample_poly)}")
                print(f"  Sample polygon (first 3 points): {sample_poly[:3]}")

if __name__ == '__main__':
    main()

