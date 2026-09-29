"""
TotalText Detection & Recognition Evaluation - FINAL VERSION

This script evaluates text detection and recognition results against TotalText ground truth.
It handles both detection (IoU-based matching) and recognition (character/word level metrics).

Features:
- Detection matching using IoU threshold
- Character-level F1 score
- Word-level exact matching
- Edit distance calculation
- Per-subject Excel output with detailed word pairs and summary
- Support for "don't care" regions (###)
- Y-axis flipping for human data
- Detection order tracking

Usage:
    python totaltext_eval_FINAL.py

Outputs:
    - results/excel_outputs/{subject}_evaluation.xlsx (per subject)
    - Console summary of overall metrics
"""

import json
import os
import re
from pathlib import Path
from shapely.geometry import Polygon
from shapely import validation
import Levenshtein
import pandas as pd
from collections import defaultdict
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

GT_ANNO_PATH = "data/totaltext/anno.json"
MODEL_OUTPUT_PATH = "data/human/lowviz/converted"
IMAGE_BASE_PATH = "data/totaltext"  # Base path for actual image files
IOU_THRESHOLD = 0.1
RESULTS_DIR = "results/excel_outputs"

# ============================================================================
# TEXT NORMALIZATION (Based on official TotalText evaluation)
# ============================================================================

def normalize_text(text):
    """
    Normalize text for recognition evaluation.
    Based on official TotalText/SPTSv2 evaluation protocol.
    """
    # Convert to uppercase
    text = text.upper()
    
    # Handle apostrophes - keep them but normalize quotes
    text = text.replace('"', '').replace("'", "'")
    
    # Replace word separators (-, ,, ;, :, /, etc.) with space
    # This ensures "24-HOURS" becomes "24 HOURS" (two words)
    text = re.sub(r'[-,;:/\\|_]', ' ', text)
    
    # Remove remaining punctuation while keeping letters, numbers, apostrophes, spaces
    text = re.sub(r'[^A-Z0-9\s\']', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def include_in_dictionary(transcription):
    """
    Check if text should be included in evaluation dictionary.
    Based on official TotalText evaluation criteria.
    
    Returns True if the text should be evaluated, False otherwise.
    """
    # Don't care regions
    if transcription == '###':
        return False
    
    # Must contain at least one alphanumeric character
    if not re.search(r'[A-Za-z0-9]', transcription):
        return False
    
    return True


# ============================================================================
# GEOMETRY & IoU
# ============================================================================

def compute_polygon_iou(poly1_coords, poly2_coords):
    """
    Compute IoU between two polygons.
    
    Args:
        poly1_coords: List of [x, y] coordinates
        poly2_coords: List of [x, y] coordinates
    
    Returns:
        IoU value (0-1), or 0 if invalid polygons
    """
    try:
        if len(poly1_coords) < 3 or len(poly2_coords) < 3:
            return 0.0
        
        # Convert to tuples for Shapely
        poly1_points = [(float(x), float(y)) for x, y in poly1_coords]
        poly2_points = [(float(x), float(y)) for x, y in poly2_coords]
        
        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)
        
        # Fix invalid polygons
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    except Exception as e:
        print(f"Error computing IoU: {e}")
        return 0.0


# ============================================================================
# RECOGNITION METRICS
# ============================================================================

def compute_bag_of_chars_match(pred_text, gt_text):
    """
    Compute character matches using Bag of Characters approach.
    
    Counts the number of common characters between pred and gt,
    ignoring order. Each character can only be matched once.
    
    Example:
        pred="HELO", gt="HELLO" -> matches 4 (H, E, L, O)
        pred="AABB", gt="AB" -> matches 2 (one A, one B)
    
    Returns:
        Number of matched characters
    """
    from collections import Counter
    
    pred_counter = Counter(pred_text.upper())
    gt_counter = Counter(gt_text.upper())
    
    # Intersection: min count for each character
    common = pred_counter & gt_counter
    return sum(common.values())


def compute_char_level_metrics(pred_text, gt_text):
    """
    Compute character-level precision, recall, and F1.
    Texts should already be normalized.
    """
    if len(pred_text) == 0 and len(gt_text) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_text) == 0 or len(gt_text) == 0:
        return 0.0, 0.0, 0.0
    
    # Use edit distance to compute character-level metrics
    distance = Levenshtein.distance(pred_text, gt_text)
    max_len = max(len(pred_text), len(gt_text))
    
    if max_len == 0:
        return 1.0, 1.0, 1.0
    
    # Character-level F1 approximation
    accuracy = 1.0 - (distance / max_len)
    
    # For precision/recall, we approximate based on string lengths
    # This is a simplified approach
    precision = max(0.0, 1.0 - distance / len(pred_text))
    recall = max(0.0, 1.0 - distance / len(gt_text))
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def compute_word_level_metrics(pred_text, gt_text):
    """
    Compute word-level matching (exact match).
    Texts should already be normalized.
    """
    is_match = (pred_text == gt_text)
    
    # For word level: 1 if exact match, 0 otherwise
    precision = 1.0 if is_match else 0.0
    recall = 1.0 if is_match else 0.0
    f1 = 1.0 if is_match else 0.0
    
    return precision, recall, f1


# ============================================================================
# DETECTION MATCHING
# ============================================================================

def match_detections_to_gt(detections, ground_truths, iou_threshold):
    """
    Match detections to ground truth using greedy IoU-based matching (ONE-TO-ONE).
    
    Strategy:
    1. Compute IoU for all detection-GT pairs
    2. Filter pairs with IoU < threshold
    3. Greedily match: pick highest IoU, assign, remove both from pool
    4. Repeat until no more matches possible
    
    Returns:
        matches: List of (det_idx, gt_idx, iou) tuples
    """
    matches = []
    
    if len(detections) == 0 or len(ground_truths) == 0:
        return matches
    
    # Compute all IoU pairs
    iou_matrix = []
    for det_idx, det in enumerate(detections):
        for gt_idx, gt in enumerate(ground_truths):
            iou = compute_polygon_iou(det['polygon'], gt['polygon'])
            if iou >= iou_threshold:
                iou_matrix.append((iou, det_idx, gt_idx))
    
    # Sort by IoU descending
    iou_matrix.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching
    used_dets = set()
    used_gts = set()
    
    for iou, det_idx, gt_idx in iou_matrix:
        if det_idx not in used_dets and gt_idx not in used_gts:
            matches.append((det_idx, gt_idx, iou))
            used_dets.add(det_idx)
            used_gts.add(gt_idx)
    
    return matches


def match_detections_to_gt_one_to_many(detections, ground_truths, iou_threshold):
    """
    Match detections to ground truth allowing ONE-TO-MANY matching.
    
    A single large circle drawn by subject can cover multiple GT words.
    Each GT word can only be matched once (to the detection with highest IoU).
    
    Returns:
        dict: {det_idx: [(gt_idx, iou), ...]} mapping each detection to its matched GT words
    """
    if len(detections) == 0 or len(ground_truths) == 0:
        return {}
    
    # For each GT, find the best matching detection (if any)
    gt_to_best_det = {}  # gt_idx -> (det_idx, iou)
    
    for gt_idx, gt in enumerate(ground_truths):
        best_det_idx = None
        best_iou = 0
        
        for det_idx, det in enumerate(detections):
            iou = compute_polygon_iou(det['polygon'], gt['polygon'])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_det_idx = det_idx
        
        if best_det_idx is not None:
            gt_to_best_det[gt_idx] = (best_det_idx, best_iou)
    
    # Invert: group GT indices by their matched detection
    det_to_gts = {}
    for gt_idx, (det_idx, iou) in gt_to_best_det.items():
        if det_idx not in det_to_gts:
            det_to_gts[det_idx] = []
        det_to_gts[det_idx].append((gt_idx, iou))
    
    return det_to_gts


def split_text_into_words(text):
    """Split text into individual words after normalization."""
    text = normalize_text(text)
    words = text.split()
    return [w for w in words if w]


def compute_word_match_no_reuse(pred_words, gt_words):
    """
    Compute word-level exact matches without reusing predicted words.
    
    Returns:
        Number of exactly matched words
    """
    if not pred_words or not gt_words:
        return 0
    
    pred_words_upper = [w.upper() for w in pred_words]
    gt_words_upper = [w.upper() for w in gt_words]
    
    matched = 0
    used_pred_indices = set()
    
    for gt_word in gt_words_upper:
        for i, pred_word in enumerate(pred_words_upper):
            if i not in used_pred_indices and pred_word == gt_word:
                matched += 1
                used_pred_indices.add(i)
                break
    
    return matched


def compute_char_match_no_reuse(pred_words, gt_words):
    """
    Compute character-level matches using Bag of Characters approach.
    
    Strategy:
    1. First find exact word matches and count all their characters
    2. For remaining unmatched GT words, find best matching pred word by Bag of Characters
    
    Each pred word can only be used once.
    Bag of Characters: count common characters ignoring order.
    
    Returns:
        Total number of matched characters
    """
    from collections import Counter
    
    if not pred_words or not gt_words:
        return 0
    
    # Work with uppercase copies
    remaining_gt = [w.upper() for w in gt_words]
    remaining_pred = [w.upper() for w in pred_words]
    total_char_matches = 0
    
    # Step 1: Find exact word matches first
    matched_gt_indices = []
    matched_pred_indices = []
    
    for gt_idx, gt_word in enumerate(remaining_gt):
        for pred_idx, pred_word in enumerate(remaining_pred):
            if pred_idx not in matched_pred_indices and gt_word == pred_word:
                # Exact match - all characters match
                total_char_matches += len(gt_word)
                matched_gt_indices.append(gt_idx)
                matched_pred_indices.append(pred_idx)
                break
    
    # Remove matched words
    remaining_gt = [w for i, w in enumerate(remaining_gt) if i not in matched_gt_indices]
    remaining_pred = [w for i, w in enumerate(remaining_pred) if i not in matched_pred_indices]
    
    # Step 2: For remaining GT words, find best match using Bag of Characters
    for gt_word in remaining_gt:
        if not remaining_pred:
            break
        
        best_match_idx = -1
        best_char_matches = 0
        
        for pred_idx, pred_word in enumerate(remaining_pred):
            # Bag of Characters: count common characters
            gt_counter = Counter(gt_word)
            pred_counter = Counter(pred_word)
            common = gt_counter & pred_counter
            char_matches = sum(common.values())
            
            if char_matches > best_char_matches:
                best_char_matches = char_matches
                best_match_idx = pred_idx
        
        if best_match_idx >= 0 and best_char_matches > 0:
            total_char_matches += best_char_matches
            remaining_pred.pop(best_match_idx)
    
    return total_char_matches


# ============================================================================
# IMAGE-LEVEL EVALUATION
# ============================================================================

def evaluate_image(detections, ground_truths, iou_threshold):
    """
    Evaluate detections against ground truth for a single image.
    
    Returns:
        dict with detection metrics, recognition metrics, and matched word pairs
    """
    # Match detections to GT
    matches = match_detections_to_gt(detections, ground_truths, iou_threshold)
    

    # Detection metrics
    num_detections = len(detections)
    num_gt = len(ground_truths)
    num_matches = len(matches)
    num_care_gt = sum(1 for gt in ground_truths if not gt.get('dont_care', False))
    
    # Count care matches (excluding dont_care GT)
    num_care_matches_det = sum(1 for _, gt_idx, _ in matches if not ground_truths[gt_idx].get('dont_care', False))
    
    det_precision = num_matches / num_detections if num_detections > 0 else 0.0
    det_recall = num_care_matches_det / num_care_gt if num_care_gt > 0 else 0.0  # Use care GT only
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0.0
    
    # Recognition metrics (only for care regions)
    char_precisions = []
    char_recalls = []
    char_f1s = []
    word_precisions = []
    word_recalls = []
    word_f1s = []
    ious = []
    edit_distances = []
    
    matched_word_pairs = []
    
    num_care_matches = 0
    
    for det_idx, gt_idx, iou in matches:
        det = detections[det_idx]
        gt = ground_truths[gt_idx]
        
        # Get texts
        det_text_raw = det.get('text', '')
        gt_text_raw = gt.get('text', '')
        
        # Check if GT is don't care
        is_dont_care = gt.get('dont_care', False)
        
        # Apply filtering and normalization
        det_included = include_in_dictionary(det_text_raw)
        gt_included = include_in_dictionary(gt_text_raw) if not is_dont_care else False
        
        # Normalize texts
        det_text = normalize_text(det_text_raw) if det_included else ""
        gt_text = normalize_text(gt_text_raw) if gt_included else ""
        
        # Store word pair info
        word_pair_info = {
            'det_idx': det_idx,
            'gt_idx': gt_idx,
            'text_no': gt.get('text_no', -1),
            'det_text_raw': det_text_raw,
            'gt_text_raw': gt_text_raw,
            'det_text': det_text,
            'gt_text': gt_text,
            'iou': iou,
            'dont_care': is_dont_care,
            'detection_order': det.get('detection_order', -1),
            'det_score': det.get('det_score', 0),
            'rec_score': det.get('rec_score', 0)
        }
        
        # Only compute recognition metrics for care regions that pass filtering
        if not is_dont_care and det_included and gt_included:
            num_care_matches += 1
            
            # Character-level
            char_p, char_r, char_f1 = compute_char_level_metrics(det_text, gt_text)
            char_precisions.append(char_p)
            char_recalls.append(char_r)
            char_f1s.append(char_f1)
            
            # Word-level (exact match)
            word_p, word_r, word_f1 = compute_word_level_metrics(det_text, gt_text)
            word_precisions.append(word_p)
            word_recalls.append(word_r)
            word_f1s.append(word_f1)
            
            # Edit distance
            edit_dist = Levenshtein.distance(det_text, gt_text)
            edit_distances.append(edit_dist)
            
            ious.append(iou)
            
            word_pair_info['char_f1'] = char_f1
            word_pair_info['word_match'] = (word_f1 == 1.0)
            word_pair_info['edit_distance'] = edit_dist
            word_pair_info['gt_chars'] = len(gt_text)
            # Character matches using Bag of Characters (order-independent)
            word_pair_info['char_matches'] = compute_bag_of_chars_match(det_text, gt_text)
        
        matched_word_pairs.append(word_pair_info)
    
    # Aggregate recognition metrics (average over matched pairs)
    avg_char_precision = sum(char_precisions) / len(char_precisions) if char_precisions else 0.0
    avg_char_recall = sum(char_recalls) / len(char_recalls) if char_recalls else 0.0
    avg_char_f1 = sum(char_f1s) / len(char_f1s) if char_f1s else 0.0
    
    avg_word_precision = sum(word_precisions) / len(word_precisions) if word_precisions else 0.0
    avg_word_recall = sum(word_recalls) / len(word_recalls) if word_recalls else 0.0
    avg_word_f1 = sum(word_f1s) / len(word_f1s) if word_f1s else 0.0
    
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    avg_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else 0.0
    
    # Count total GT chars for all care GT (including unmatched)
    total_gt_chars = sum(
        len(normalize_text(gt.get('text', ''))) 
        for gt in ground_truths 
        if not gt.get('dont_care', False) and include_in_dictionary(gt.get('text', ''))
    )
    
    # Recognition rate = correct matches / all GT
    # word_f1s contains 1.0 for exact match, 0.0 for non-match
    num_word_matches = int(sum(word_f1s))  # Count of exact word matches
    total_char_matches = sum(p.get('char_matches', 0) for p in matched_word_pairs)
    
    # Recognition rate using all care GT as denominator
    recognition_rate_word = num_word_matches / num_care_gt if num_care_gt > 0 else 0.0
    recognition_rate_char = total_char_matches / total_gt_chars if total_gt_chars > 0 else 0.0
    
    return {
        # Detection
        'num_detections': num_detections,
        'num_gt': num_gt,
        'num_matches': num_matches,
        'det_precision': det_precision,
        'det_recall': det_recall,
        'det_f1': det_f1,
        
        # Recognition (average over matched pairs only)
        'char_precision': avg_char_precision,
        'char_recall': avg_char_recall,
        'char_f1': avg_char_f1,
        'word_precision': avg_word_precision,
        'word_recall': avg_word_recall,
        'word_f1': avg_word_f1,
        
        # Recognition rate (matches / all GT)
        'recognition_rate_word': recognition_rate_word,
        'recognition_rate_char': recognition_rate_char,
        'num_word_matches': num_word_matches,
        'total_char_matches': total_char_matches,
        'total_gt_chars': total_gt_chars,
        
        # Other
        'avg_iou': avg_iou,
        'avg_edit_distance': avg_edit_distance,
        'num_care_matches': num_care_matches,
        'num_care_gt': num_care_gt,
        
        # Word pairs for detailed output
        'matched_word_pairs': matched_word_pairs
    }


def evaluate_image_one_to_many(detections, ground_truths, iou_threshold):
    """
    Evaluate detections against ground truth for a single image using ONE-TO-MANY matching.
    
    A single detection (large circle) can match multiple GT words.
    Subject's reported text is split into words for matching.
    
    Returns:
        dict with detection metrics, recognition metrics, and matched word pairs
    """
    # Match detections to GT (one-to-many)
    det_to_gts = match_detections_to_gt_one_to_many(detections, ground_truths, iou_threshold)
    
    # Count detected GT words (unique GT indices that were matched)
    detected_gt_indices = set()
    for gt_list in det_to_gts.values():
        for gt_idx, iou in gt_list:
            detected_gt_indices.add(gt_idx)
    
    # Filter out don't care regions
    care_gt_indices = [i for i, gt in enumerate(ground_truths) if not gt.get('dont_care', False)]
    num_care_gt = len(care_gt_indices)
    
    detected_care_gt = detected_gt_indices.intersection(care_gt_indices)
    num_detected_care = len(detected_care_gt)
    
    # Detection metrics
    num_detections = len(detections)
    num_gt = len(ground_truths)
    
    det_recall = num_detected_care / num_care_gt if num_care_gt > 0 else 0.0
    
    # Count ALL GT words/chars (for recognition rate denominator)
    all_gt_words = 0
    all_gt_chars = 0
    for gt_idx in care_gt_indices:
        gt = ground_truths[gt_idx]
        gt_text_raw = gt.get('text', '')
        if include_in_dictionary(gt_text_raw):
            words = split_text_into_words(gt_text_raw)
            all_gt_words += len(words)
            all_gt_chars += sum(len(w) for w in words)
    
    # Recognition evaluation
    total_word_matches = 0
    total_char_matches = 0
    total_gt_chars_detected = 0  # chars in detected GT only (for reference)
    total_gt_words_detected = 0  # words in detected GT only (for reference)
    
    matched_word_pairs = []
    
    for det_idx, gt_list in det_to_gts.items():
        det = detections[det_idx]
        det_text_raw = det.get('text', '')
        
        # Split detection text into words
        pred_words = split_text_into_words(det_text_raw)
        
        # Collect GT words for this detection (excluding don't care)
        gt_words = []
        gt_words_raw = []
        for gt_idx, iou in gt_list:
            gt = ground_truths[gt_idx]
            if gt.get('dont_care', False):
                continue
            gt_text_raw = gt.get('text', '')
            if include_in_dictionary(gt_text_raw):
                gt_words_raw.append(gt_text_raw)
                gt_words.extend(split_text_into_words(gt_text_raw))
        
        if not gt_words:
            continue
        
        # Filter pred_words
        pred_words_filtered = [w for w in pred_words if include_in_dictionary(w)]
        
        # Word-level matching
        word_matches = compute_word_match_no_reuse(pred_words_filtered, gt_words)
        total_word_matches += word_matches
        total_gt_words_detected += len(gt_words)
        
        # Character-level matching (no reuse)
        char_matches = compute_char_match_no_reuse(pred_words_filtered, gt_words)
        total_char_matches += char_matches
        total_gt_chars_detected += sum(len(w) for w in gt_words)
        
        # Store per-GT-word entries (one row per matched GT word)
        for gt_idx, iou in gt_list:
            gt = ground_truths[gt_idx]
            if gt.get('dont_care', False):
                continue
            gt_text_raw = gt.get('text', '')
            if not include_in_dictionary(gt_text_raw):
                continue
            gt_norm = normalize_text(gt_text_raw)
            det_norm = normalize_text(det_text_raw) if det_text_raw else ''
            word_match = (gt_norm in det_norm) if (gt_norm and det_norm) else False
            matched_word_pairs.append({
                'det_idx': det_idx,
                'gt_idx': gt_idx,
                'text_no': gt.get('text_no', -1),
                'det_text_raw': det_text_raw,
                'gt_text_raw': gt_text_raw,
                'iou': iou,
                'dont_care': False,
                'word_match': word_match,
                'detection_order': det.get('detection_order', -1),
                'det_score': det.get('det_score', 0),
                'rec_score': det.get('rec_score', 0),
            })
    
    # Calculate rates using ALL GT words/chars as denominator
    recognition_rate_word = total_word_matches / all_gt_words if all_gt_words > 0 else 0.0
    recognition_rate_char = total_char_matches / all_gt_chars if all_gt_chars > 0 else 0.0
    
    return {
        'num_detections': num_detections,
        'num_gt': num_gt,
        'num_care_gt': num_care_gt,
        'num_detected_care': num_detected_care,
        'det_recall': det_recall,
        'total_word_matches': total_word_matches,
        'total_gt_words': all_gt_words,           # Use all GT words as denominator
        'total_gt_words_detected': total_gt_words_detected,  # For reference
        'total_char_matches': total_char_matches,
        'total_gt_chars': all_gt_chars,           # Use all GT chars as denominator
        'total_gt_chars_detected': total_gt_chars_detected,  # For reference
        'recognition_rate_word': recognition_rate_word,
        'recognition_rate_char': recognition_rate_char,
        'word_f1': recognition_rate_word,  # Simplified: same as recognition rate
        'char_f1': recognition_rate_char,  # Simplified: same as recognition rate
        'matched_word_pairs': matched_word_pairs
    }


def evaluate_image_wordspotting(detections, ground_truths):
    """
    Evaluate detections against ground truth using WORD SPOTTING (no localization).
    
    This is a bag-of-words approach: we compare all detected words against all GT words
    without considering their spatial locations (IoU matching).
    
    Returns:
        dict with word spotting metrics
    """
    from collections import Counter
    
    # Collect all GT words (excluding don't care)
    all_gt_words = []
    for gt in ground_truths:
        if gt.get('dont_care', False):
            continue
        gt_text = gt.get('text', '')
        if include_in_dictionary(gt_text):
            words = split_text_into_words(gt_text)
            all_gt_words.extend(words)
    
    # Collect all detected words
    all_pred_words = []
    for det in detections:
        det_text = det.get('text', '')
        if include_in_dictionary(det_text):
            words = split_text_into_words(det_text)
            all_pred_words.extend(words)
    
    # Count words
    num_gt_words = len(all_gt_words)
    num_pred_words = len(all_pred_words)
    
    # Bag of words matching using Counter intersection
    counter_gt = Counter(all_gt_words)
    counter_pred = Counter(all_pred_words)
    common_words = counter_gt & counter_pred
    num_word_matches = sum(common_words.values())
    
    # Character counts
    num_gt_chars = sum(len(w) for w in all_gt_words)
    num_pred_chars = sum(len(w) for w in all_pred_words)
    
    # Character matching using bag of characters approach
    # First, exact word matches contribute all their characters
    # Then, remaining words use bag of characters matching
    gt_words_list = [w for word, freq in counter_gt.items() for w in [word] * freq]
    pred_words_list = [w for word, freq in counter_pred.items() for w in [word] * freq]
    num_char_matches = compute_char_match_no_reuse(pred_words_list, gt_words_list)
    
    # Calculate rates
    recognition_rate_word = num_word_matches / num_gt_words if num_gt_words > 0 else 0.0
    recognition_rate_char = num_char_matches / num_gt_chars if num_gt_chars > 0 else 0.0
    
    # Precision and recall
    precision_word = num_word_matches / num_pred_words if num_pred_words > 0 else 0.0
    recall_word = num_word_matches / num_gt_words if num_gt_words > 0 else 0.0
    f1_word = 2 * precision_word * recall_word / (precision_word + recall_word) if (precision_word + recall_word) > 0 else 0.0
    
    precision_char = num_char_matches / num_pred_chars if num_pred_chars > 0 else 0.0
    recall_char = num_char_matches / num_gt_chars if num_gt_chars > 0 else 0.0
    f1_char = 2 * precision_char * recall_char / (precision_char + recall_char) if (precision_char + recall_char) > 0 else 0.0
    
    return {
        'num_gt_words': num_gt_words,
        'num_pred_words': num_pred_words,
        'num_word_matches': num_word_matches,
        'num_gt_chars': num_gt_chars,
        'num_pred_chars': num_pred_chars,
        'num_char_matches': num_char_matches,
        'precision_word': precision_word,
        'recall_word': recall_word,
        'f1_word': f1_word,
        'precision_char': precision_char,
        'recall_char': recall_char,
        'f1_char': f1_char,
        'recognition_rate_word': recognition_rate_word,
        'recognition_rate_char': recognition_rate_char,
        'gt_words': all_gt_words,
        'pred_words': all_pred_words,
        'matched_words': list(common_words.elements()),
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ground_truth(gt_path):
    """
    Load TotalText ground truth annotations (COCO format).
    
    Returns:
        gt_by_image: dict {image_id: [list of GT objects]}
        image_info: dict {image_id: image_info_dict}
    """
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    # Build image info mapping
    gt_images = data.get('images', [])
    gt_annotations = data.get('annotations', [])
    
    image_info = {img['id']: img for img in gt_images}
    gt_by_image = defaultdict(list)

    # Global counter for non-dont-care words (matches features.py Text.No)
    text_no = 1

    # Process annotations
    for ann in gt_annotations:
        image_id = ann['image_id']
        image_name = image_info.get(image_id, {}).get('file_name', '')

        # Filter for 16/ images
        if not image_name.startswith('16/'):
            continue

        # Get caption and bbox
        caption = ann.get('caption', '')
        bbox_list = ann.get('bbox', [])

        # Split caption by spaces
        words = caption.split()

        # Each word gets a corresponding bbox
        for idx, word in enumerate(words):
            if idx < len(bbox_list):
                polygon = bbox_list[idx]

                # Mark ### as don't care
                is_dont_care = (word == '###')

                gt_entry = {
                    'text': word,
                    'polygon': polygon,
                    'dont_care': is_dont_care,
                    'image_name': image_name
                }

                if not is_dont_care and polygon and isinstance(polygon, list) and len(polygon) >= 3:
                    gt_entry['text_no'] = text_no
                    text_no += 1

                gt_by_image[image_id].append(gt_entry)

    return gt_by_image, image_info


def load_model_output(output_path, image_info, image_base_path):
    """
    Load model detection and recognition outputs.
    Handles both directory of JSON files and single JSON file.
    
    Args:
        output_path: Path to model output JSON file(s)
        image_info: Dict mapping image_id to image info (from GT)
        image_base_path: Base path for actual image files
    
    Returns:
        dict: {(subject, image_id): [list of detection objects]}
    """
    detections_by_image = defaultdict(list)
    image_heights = {}  # Cache for image heights
    
    if os.path.isdir(output_path):
        json_files = list(Path(output_path).glob("*.json"))
    else:
        json_files = [Path(output_path)]
    
    for json_file in json_files:
        # Extract subject from filename (e.g., Sub161.json -> Sub161)
        subject = json_file.stem
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            image_id = item.get('image_id')
            if image_id is None:
                continue
            
            key = (subject, image_id)
            
            # Get actual image height from file for y-axis flipping
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
            rec_texts = item.get('rec_texts', '')
            det_scores = item.get('det_score', [])
            rec_scores = item.get('rec_score', [])
            if isinstance(rec_texts, str) and rec_texts.startswith('['):
                try:
                    rec_texts = json.loads(rec_texts)
                except:
                    rec_texts = [rec_texts]
            elif not isinstance(rec_texts, list):
                rec_texts = [rec_texts] if rec_texts else []
            
            # Parse polys
            polys = item.get('polys', [])
            
            # Handle nested structure if present
            if isinstance(polys, str):
                try:
                    polys = json.loads(polys)
                except:
                    polys = []
            
            # Create detection objects (keep all, including < 3 points)
            for idx, poly in enumerate(polys):
                if isinstance(poly, list):
                    polygon_points = poly
                    if polygon_points and isinstance(polygon_points[0], list) and len(polygon_points[0]) > 2:
                        polygon_points = [[p[0], p[1]] for p in polygon_points]

                    # FLIP Y-AXIS for human data (y is inverted)
                    if len(polygon_points) >= 1:
                        polygon_points = [[x, image_height - y] for x, y in polygon_points]

                    text = rec_texts[idx] if idx < len(rec_texts) else ''
                    det_score = det_scores[idx] if idx < len(det_scores) else 0
                    rec_score = rec_scores[idx] if idx < len(rec_scores) else 0

                    detections_by_image[key].append({
                        'polygon': polygon_points,
                        'text': text,
                        'detection_order': idx + 1,  # 1-indexed
                        'det_score': det_score,
                        'rec_score': rec_score
                    })
    
    return detections_by_image


# ============================================================================
# EXCEL OUTPUT
# ============================================================================

def save_subject_excel(subject, results_one_to_one, results_one_to_many, results_wordspotting, output_dir):
    """
    Save per-subject evaluation results to Excel with 6 sheets:
    
    Sheet 1 (1to1 Word Pairs): One-to-one matching detailed word pairs
    Sheet 2 (1to1 Summary): One-to-one matching summary
    Sheet 3 (1toN Word Pairs): One-to-many matching detailed word pairs
    Sheet 4 (1toN Summary): One-to-many matching summary
    Sheet 5 (WordSpot Details): Word spotting detailed info
    Sheet 6 (WordSpot Summary): Word spotting summary
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject}_evaluation.xlsx")
    
    # =========================================================================
    # ONE-TO-ONE MATCHING SHEETS
    # =========================================================================
    
    # === Sheet 1: One-to-One Word Pairs ===
    word_pairs_data_1to1 = []
    
    for (subj, img_id), metrics in results_one_to_one.items():
        if subj != subject:
            continue
        
        for pair in metrics['matched_word_pairs']:
            word_pairs_data_1to1.append({
                'Image': img_id,
                'Detection Order': pair.get('detection_order', -1),
                'GT Text': pair['gt_text_raw'],
                'GT Index': pair.get('gt_idx', -1),
                'Text.No': pair.get('text_no', -1),
                'Predicted Text': pair['det_text_raw'],
                'Det Score': round(pair.get('det_score', 0), 2),
                'Rec Score': round(pair.get('rec_score', 0), 2),
                'IoU': round(pair['iou'], 4),
                'Edit Distance': pair.get('edit_distance', ''),
                'Word Match': 'Yes' if pair.get('word_match', False) else 'No' if not pair['dont_care'] else 'N/A',
                'Char F1': round(pair.get('char_f1', 0), 4) if not pair['dont_care'] else 'N/A',
                'Dont Care': 'Yes' if pair['dont_care'] else 'No'
            })
    
    df_word_pairs_1to1 = pd.DataFrame(word_pairs_data_1to1)
    
    # === Sheet 2: One-to-One Summary ===
    summary_data_1to1 = []
    
    for (subj, img_id), metrics in results_one_to_one.items():
        if subj != subject:
            continue
        
        summary_data_1to1.append({
            'Image': img_id,
            'Type': 'Image',
            'Word Precision': round(metrics['word_precision'], 4),
            'Word Recall': round(metrics['word_recall'], 4),
            'Word F1': round(metrics['word_f1'], 4),
            'Char Precision': round(metrics['char_precision'], 4),
            'Char Recall': round(metrics['char_recall'], 4),
            'Char F1': round(metrics['char_f1'], 4),
            'Avg IoU': round(metrics['avg_iou'], 4),
            'Matched Words': metrics['num_care_matches'],
            'Total GT Words': metrics['num_care_gt'],
            'Total Detections': metrics['num_detections'],
            'Avg Edit Distance': round(metrics['avg_edit_distance'], 2) if metrics['avg_edit_distance'] > 0 else 0
        })
    
    if summary_data_1to1:
        overall_1to1 = {
            'Image': 'OVERALL',
            'Type': 'Summary',
            'Word Precision': round(sum(d['Word Precision'] for d in summary_data_1to1) / len(summary_data_1to1), 4),
            'Word Recall': round(sum(d['Word Recall'] for d in summary_data_1to1) / len(summary_data_1to1), 4),
            'Word F1': round(sum(d['Word F1'] for d in summary_data_1to1) / len(summary_data_1to1), 4),
            'Char Precision': round(sum(d['Char Precision'] for d in summary_data_1to1) / len(summary_data_1to1), 4),
            'Char Recall': round(sum(d['Char Recall'] for d in summary_data_1to1) / len(summary_data_1to1), 4),
            'Char F1': round(sum(d['Char F1'] for d in summary_data_1to1) / len(summary_data_1to1), 4),
            'Avg IoU': round(sum(d['Avg IoU'] for d in summary_data_1to1) / len(summary_data_1to1), 4),
            'Matched Words': sum(d['Matched Words'] for d in summary_data_1to1),
            'Total GT Words': sum(d['Total GT Words'] for d in summary_data_1to1),
            'Total Detections': sum(d['Total Detections'] for d in summary_data_1to1),
            'Avg Edit Distance': round(sum(d['Avg Edit Distance'] for d in summary_data_1to1) / len(summary_data_1to1), 2)
        }
        summary_data_1to1.append(overall_1to1)
    
    df_summary_1to1 = pd.DataFrame(summary_data_1to1)
    
    # =========================================================================
    # ONE-TO-MANY MATCHING SHEETS
    # =========================================================================
    
    # === Sheet 3: One-to-Many Word Pairs (per GT word) ===
    word_pairs_data_1toN = []

    for (subj, img_id), metrics in results_one_to_many.items():
        if subj != subject:
            continue

        for pair in metrics['matched_word_pairs']:
            word_pairs_data_1toN.append({
                'Image': img_id,
                'Detection Order': pair.get('detection_order', -1),
                'GT Text': pair['gt_text_raw'],
                'GT Index': pair.get('gt_idx', -1),
                'Text.No': pair.get('text_no', -1),
                'Predicted Text': pair.get('det_text_raw', ''),
                'Det Score': round(pair.get('det_score', 0), 2),
                'Rec Score': round(pair.get('rec_score', 0), 2),
                'IoU': round(pair['iou'], 4),
                'Word Match': 'Yes' if pair.get('word_match', False) else 'No',
                'Dont Care': 'Yes' if pair.get('dont_care', False) else 'No',
            })

    df_word_pairs_1toN = pd.DataFrame(word_pairs_data_1toN)
    
    # === Sheet 4: One-to-Many Summary ===
    summary_data_1toN = []
    
    for (subj, img_id), metrics in results_one_to_many.items():
        if subj != subject:
            continue
        
        summary_data_1toN.append({
            'Image': img_id,
            'Type': 'Image',
            'Detection Rate': round(metrics['det_recall'], 4),
            'Word F1': round(metrics['word_f1'], 4),
            'Char F1': round(metrics['char_f1'], 4),
            'Recognition Rate (Word)': round(metrics['recognition_rate_word'], 4),
            'Recognition Rate (Char)': round(metrics['recognition_rate_char'], 4),
            'Detected GT Words': metrics['num_detected_care'],
            'Total GT Words': metrics['num_care_gt'],
            'Total Word Matches': metrics['total_word_matches'],
            'Total Char Matches': metrics['total_char_matches'],
        })
    
    if summary_data_1toN:
        # Aggregate counts for overall
        total_detected = sum(d['Detected GT Words'] for d in summary_data_1toN)
        total_gt = sum(d['Total GT Words'] for d in summary_data_1toN)
        total_word_matches = sum(d['Total Word Matches'] for d in summary_data_1toN)
        total_char_matches = sum(d['Total Char Matches'] for d in summary_data_1toN)
        total_gt_words_for_rec = sum(metrics['total_gt_words'] for (subj, _), metrics in results_one_to_many.items() if subj == subject)
        total_gt_chars_for_rec = sum(metrics['total_gt_chars'] for (subj, _), metrics in results_one_to_many.items() if subj == subject)
        
        overall_1toN = {
            'Image': 'OVERALL',
            'Type': 'Summary',
            'Detection Rate': round(total_detected / total_gt, 4) if total_gt > 0 else 0,
            'Word F1': round(total_word_matches / total_gt_words_for_rec, 4) if total_gt_words_for_rec > 0 else 0,
            'Char F1': round(total_char_matches / total_gt_chars_for_rec, 4) if total_gt_chars_for_rec > 0 else 0,
            'Recognition Rate (Word)': round(total_word_matches / total_gt_words_for_rec, 4) if total_gt_words_for_rec > 0 else 0,
            'Recognition Rate (Char)': round(total_char_matches / total_gt_chars_for_rec, 4) if total_gt_chars_for_rec > 0 else 0,
            'Detected GT Words': total_detected,
            'Total GT Words': total_gt,
            'Total Word Matches': total_word_matches,
            'Total Char Matches': total_char_matches,
        }
        summary_data_1toN.append(overall_1toN)
    
    df_summary_1toN = pd.DataFrame(summary_data_1toN)
    
    # =========================================================================
    # WORD SPOTTING SHEETS (No localization - bag of words)
    # =========================================================================
    
    # === Sheet 5: Word Spotting Details ===
    wordspot_details = []
    
    for (subj, img_id), metrics in results_wordspotting.items():
        if subj != subject:
            continue
        
        wordspot_details.append({
            'Image': img_id,
            'GT Words': ', '.join(metrics.get('gt_words', [])),
            'Pred Words': ', '.join(metrics.get('pred_words', [])),
            'Matched Words': ', '.join(metrics.get('matched_words', [])),
            'Num GT Words': metrics['num_gt_words'],
            'Num Pred Words': metrics['num_pred_words'],
            'Word Matches': metrics['num_word_matches'],
            'Num GT Chars': metrics['num_gt_chars'],
            'Num Pred Chars': metrics['num_pred_chars'],
            'Char Matches': metrics['num_char_matches'],
        })
    
    df_wordspot_details = pd.DataFrame(wordspot_details)
    
    # === Sheet 6: Word Spotting Summary ===
    wordspot_summary = []
    
    for (subj, img_id), metrics in results_wordspotting.items():
        if subj != subject:
            continue
        
        wordspot_summary.append({
            'Image': img_id,
            'Type': 'Image',
            'Precision (Word)': round(metrics['precision_word'], 4),
            'Recall (Word)': round(metrics['recall_word'], 4),
            'F1 (Word)': round(metrics['f1_word'], 4),
            'Precision (Char)': round(metrics['precision_char'], 4),
            'Recall (Char)': round(metrics['recall_char'], 4),
            'F1 (Char)': round(metrics['f1_char'], 4),
            'Recognition Rate (Word)': round(metrics['recognition_rate_word'], 4),
            'Recognition Rate (Char)': round(metrics['recognition_rate_char'], 4),
            'Word Matches': metrics['num_word_matches'],
            'Total GT Words': metrics['num_gt_words'],
            'Char Matches': metrics['num_char_matches'],
            'Total GT Chars': metrics['num_gt_chars'],
        })
    
    if wordspot_summary:
        # Aggregate for overall
        total_word_matches_ws = sum(d['Word Matches'] for d in wordspot_summary)
        total_gt_words_ws = sum(d['Total GT Words'] for d in wordspot_summary)
        total_char_matches_ws = sum(d['Char Matches'] for d in wordspot_summary)
        total_gt_chars_ws = sum(d['Total GT Chars'] for d in wordspot_summary)
        total_pred_words_ws = sum(metrics['num_pred_words'] for (subj, _), metrics in results_wordspotting.items() if subj == subject)
        total_pred_chars_ws = sum(metrics['num_pred_chars'] for (subj, _), metrics in results_wordspotting.items() if subj == subject)
        
        prec_word = total_word_matches_ws / total_pred_words_ws if total_pred_words_ws > 0 else 0
        rec_word = total_word_matches_ws / total_gt_words_ws if total_gt_words_ws > 0 else 0
        f1_word_overall = 2 * prec_word * rec_word / (prec_word + rec_word) if (prec_word + rec_word) > 0 else 0
        
        prec_char = total_char_matches_ws / total_pred_chars_ws if total_pred_chars_ws > 0 else 0
        rec_char = total_char_matches_ws / total_gt_chars_ws if total_gt_chars_ws > 0 else 0
        f1_char_overall = 2 * prec_char * rec_char / (prec_char + rec_char) if (prec_char + rec_char) > 0 else 0
        
        overall_ws = {
            'Image': 'OVERALL',
            'Type': 'Summary',
            'Precision (Word)': round(prec_word, 4),
            'Recall (Word)': round(rec_word, 4),
            'F1 (Word)': round(f1_word_overall, 4),
            'Precision (Char)': round(prec_char, 4),
            'Recall (Char)': round(rec_char, 4),
            'F1 (Char)': round(f1_char_overall, 4),
            'Recognition Rate (Word)': round(rec_word, 4),
            'Recognition Rate (Char)': round(rec_char, 4),
            'Word Matches': total_word_matches_ws,
            'Total GT Words': total_gt_words_ws,
            'Char Matches': total_char_matches_ws,
            'Total GT Chars': total_gt_chars_ws,
        }
        wordspot_summary.append(overall_ws)
    
    df_wordspot_summary = pd.DataFrame(wordspot_summary)
    
    # Write to Excel with 6 sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_word_pairs_1to1.to_excel(writer, sheet_name='1to1 Word Pairs', index=False)
        df_summary_1to1.to_excel(writer, sheet_name='1to1 Summary', index=False)
        df_word_pairs_1toN.to_excel(writer, sheet_name='1toN Word Pairs', index=False)
        df_summary_1toN.to_excel(writer, sheet_name='1toN Summary', index=False)
        df_wordspot_details.to_excel(writer, sheet_name='WordSpot Details', index=False)
        df_wordspot_summary.to_excel(writer, sheet_name='WordSpot Summary', index=False)
    
    print(f"  Saved: {output_path}")


def save_subject_excel_old(subject, results_per_image, output_dir):
    """
    [DEPRECATED] Old version - Save per-subject evaluation results to Excel with two sheets.
    
    Sheet 1 (Word Pairs): Detailed word-level information
    Sheet 2 (Summary): Image-level and overall summary
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject}_evaluation.xlsx")
    
    # === Sheet 1: Word Pairs ===
    word_pairs_data = []
    
    for (subj, img_id), metrics in results_per_image.items():
        if subj != subject:
            continue
        
        for pair in metrics['matched_word_pairs']:
            word_pairs_data.append({
                'Image': img_id,
                'Detection Order': pair.get('detection_order', -1),
                'GT Text': pair['gt_text_raw'],
                'GT Index': pair.get('gt_idx', -1),
                'Text.No': pair.get('text_no', -1),
                'Predicted Text': pair['det_text_raw'],
                'Det Score': round(pair.get('det_score', 0), 2),
                'Rec Score': round(pair.get('rec_score', 0), 2),
                'IoU': round(pair['iou'], 4),
                'Edit Distance': pair.get('edit_distance', ''),
                'Word Match': 'Yes' if pair.get('word_match', False) else 'No' if not pair['dont_care'] else 'N/A',
                'Char F1': round(pair.get('char_f1', 0), 4) if not pair['dont_care'] else 'N/A',
                'Dont Care': 'Yes' if pair['dont_care'] else 'No'
            })
    
    df_word_pairs = pd.DataFrame(word_pairs_data)
    
    # === Sheet 2: Summary ===
    summary_data = []
    
    # Per-image summaries
    for (subj, img_id), metrics in results_per_image.items():
        if subj != subject:
            continue
        
        summary_data.append({
            'Image': img_id,
            'Type': 'Image',
            'Word Precision': round(metrics['word_precision'], 4),
            'Word Recall': round(metrics['word_recall'], 4),
            'Word F1': round(metrics['word_f1'], 4),
            'Char Precision': round(metrics['char_precision'], 4),
            'Char Recall': round(metrics['char_recall'], 4),
            'Char F1': round(metrics['char_f1'], 4),
            'Avg IoU': round(metrics['avg_iou'], 4),
            'Matched Words': metrics['num_care_matches'],
            'Total GT Words': metrics['num_care_gt'],
            'Total Detections': metrics['num_detections'],
            'Avg Edit Distance': round(metrics['avg_edit_distance'], 2) if metrics['avg_edit_distance'] > 0 else 0
        })
    
    # Overall summary
    if summary_data:
        overall = {
            'Image': 'OVERALL',
            'Type': 'Summary',
            'Word Precision': round(sum(d['Word Precision'] for d in summary_data) / len(summary_data), 4),
            'Word Recall': round(sum(d['Word Recall'] for d in summary_data) / len(summary_data), 4),
            'Word F1': round(sum(d['Word F1'] for d in summary_data) / len(summary_data), 4),
            'Char Precision': round(sum(d['Char Precision'] for d in summary_data) / len(summary_data), 4),
            'Char Recall': round(sum(d['Char Recall'] for d in summary_data) / len(summary_data), 4),
            'Char F1': round(sum(d['Char F1'] for d in summary_data) / len(summary_data), 4),
            'Avg IoU': round(sum(d['Avg IoU'] for d in summary_data) / len(summary_data), 4),
            'Matched Words': sum(d['Matched Words'] for d in summary_data),
            'Total GT Words': sum(d['Total GT Words'] for d in summary_data),
            'Total Detections': sum(d['Total Detections'] for d in summary_data),
            'Avg Edit Distance': round(sum(d['Avg Edit Distance'] for d in summary_data) / len(summary_data), 2)
        }
        summary_data.append(overall)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_word_pairs.to_excel(writer, sheet_name='Word Pairs', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_dataset():
    """Main evaluation function."""
    
    print("="*80)
    print("TotalText Detection & Recognition Evaluation - FINAL")
    print("(With both One-to-One and One-to-Many matching)")
    print("="*80)
    
    # Load data
    print("\n[Loading Data]")
    print(f"Ground Truth: {GT_ANNO_PATH}")
    print(f"Model Output: {MODEL_OUTPUT_PATH}")
    print(f"Image Base Path: {IMAGE_BASE_PATH}")
    
    gt_by_image, image_info = load_ground_truth(GT_ANNO_PATH)
    det_by_image = load_model_output(MODEL_OUTPUT_PATH, image_info, IMAGE_BASE_PATH)
    
    print(f"Loaded {len(gt_by_image)} images with ground truth")
    print(f"Loaded {len(det_by_image)} images with detections")
    
    # Evaluate each image with BOTH methods
    print("\n[Evaluating Images - One-to-One Matching]")
    results_one_to_one = {}
    
    for key in det_by_image.keys():
        subject, image_id = key
        detections = det_by_image[key]
        ground_truths = gt_by_image.get(image_id, [])
        
        if len(ground_truths) == 0:
            print(f"Warning: No GT for {subject}/{image_id}")
            continue
        
        metrics = evaluate_image(detections, ground_truths, IOU_THRESHOLD)
        results_one_to_one[key] = metrics
    
    print("\n[Evaluating Images - One-to-Many Matching]")
    results_one_to_many = {}
    
    for key in det_by_image.keys():
        subject, image_id = key
        detections = det_by_image[key]
        ground_truths = gt_by_image.get(image_id, [])
        
        if len(ground_truths) == 0:
            continue
        
        metrics = evaluate_image_one_to_many(detections, ground_truths, IOU_THRESHOLD)
        results_one_to_many[key] = metrics
    
    print("\n[Evaluating Images - Word Spotting (No Localization)]")
    results_wordspotting = {}
    
    for key in det_by_image.keys():
        subject, image_id = key
        detections = det_by_image[key]
        ground_truths = gt_by_image.get(image_id, [])
        
        if len(ground_truths) == 0:
            continue
        
        metrics = evaluate_image_wordspotting(detections, ground_truths)
        results_wordspotting[key] = metrics
    
    # Organize by subject
    results_by_subject = defaultdict(list)
    for key in results_one_to_one.keys():
        subject, image_id = key
        results_by_subject[subject].append(key)
    
    # Generate Excel outputs per subject (with all 3 methods in separate sheets)
    print(f"\n[Generating Excel Outputs]")
    print(f"Output Directory: {RESULTS_DIR}")
    
    for subject in results_by_subject.keys():
        save_subject_excel(subject, results_one_to_one, results_one_to_many, results_wordspotting, RESULTS_DIR)
    
    # Overall summary for ONE-TO-ONE
    print("\n" + "="*80)
    print("OVERALL SUMMARY - ONE-TO-ONE MATCHING")
    print("="*80)
    
    all_metrics_1to1 = list(results_one_to_one.values())
    
    avg_det_precision = sum(m['det_precision'] for m in all_metrics_1to1) / len(all_metrics_1to1)
    avg_det_recall = sum(m['det_recall'] for m in all_metrics_1to1) / len(all_metrics_1to1)
    avg_det_f1 = sum(m['det_f1'] for m in all_metrics_1to1) / len(all_metrics_1to1)
    
    avg_char_f1 = sum(m['char_f1'] for m in all_metrics_1to1) / len(all_metrics_1to1)
    avg_word_f1 = sum(m['word_f1'] for m in all_metrics_1to1) / len(all_metrics_1to1)
    
    total_matched = sum(m['num_care_matches'] for m in all_metrics_1to1)
    total_gt = sum(m['num_care_gt'] for m in all_metrics_1to1)
    
    print(f"\nImages Evaluated: {len(all_metrics_1to1)}")
    print(f"Subjects: {len(results_by_subject)}")
    
    print(f"\n[Detection Metrics]")
    print(f"Precision: {avg_det_precision:.4f}")
    print(f"Recall: {avg_det_recall:.4f}")
    print(f"F1: {avg_det_f1:.4f}")
    
    print(f"\n[Recognition Metrics]")
    print(f"Word F1: {avg_word_f1:.4f}")
    print(f"Char F1: {avg_char_f1:.4f}")
    
    print(f"\n[Counts]")
    print(f"Total Matched Words: {total_matched}")
    print(f"Total GT Words: {total_gt}")
    
    # Overall summary for ONE-TO-MANY
    print("\n" + "="*80)
    print("OVERALL SUMMARY - ONE-TO-MANY MATCHING")
    print("="*80)
    
    all_metrics_1toN = list(results_one_to_many.values())
    
    total_detected = sum(m['num_detected_care'] for m in all_metrics_1toN)
    total_gt_1toN = sum(m['num_care_gt'] for m in all_metrics_1toN)
    total_word_matches = sum(m['total_word_matches'] for m in all_metrics_1toN)
    total_gt_words = sum(m['total_gt_words'] for m in all_metrics_1toN)
    total_char_matches = sum(m['total_char_matches'] for m in all_metrics_1toN)
    total_gt_chars = sum(m['total_gt_chars'] for m in all_metrics_1toN)
    
    detection_rate = total_detected / total_gt_1toN if total_gt_1toN > 0 else 0
    recognition_rate_word = total_word_matches / total_gt_words if total_gt_words > 0 else 0
    recognition_rate_char = total_char_matches / total_gt_chars if total_gt_chars > 0 else 0
    
    print(f"\n[Detection Metrics]")
    print(f"Detection Rate: {detection_rate:.4f}")
    print(f"Detected GT Words: {total_detected} / {total_gt_1toN}")
    
    print(f"\n[Recognition Metrics]")
    print(f"Recognition Rate (Word): {recognition_rate_word:.4f}")
    print(f"Recognition Rate (Char): {recognition_rate_char:.4f}")
    print(f"Word Matches: {total_word_matches} / {total_gt_words}")
    print(f"Char Matches: {total_char_matches} / {total_gt_chars}")
    
    print("\n" + "="*80)
    print(f"Excel outputs saved to: {RESULTS_DIR}/")
    print("Each file contains 4 sheets:")
    print("  - 1to1 Word Pairs: One-to-one matching details")
    print("  - 1to1 Summary: One-to-one matching summary")
    print("  - 1toN Word Pairs: One-to-many matching details")
    print("  - 1toN Summary: One-to-many matching summary")
    print("="*80)


if __name__ == "__main__":
    evaluate_dataset()

