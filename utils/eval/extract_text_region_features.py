#!/usr/bin/env python3
"""
Extract visual features from text regions in TotalText dataset
Specifically for images in the 16/ directory

Features extracted:
- polygon_size          : area of the text polygon (pixels²)
- edge_density          : ratio of edge pixels to polygon area
- luminance_std         : std of pixel intensities within polygon (local contrast)
- luminance_mean        : mean pixel brightness within polygon
- avg_curvature         : average interior-angle-style value at polygon vertices
                          (degrees; |180 - angle of consecutive edge vectors|)
- spatial_frequency     : overall activity level (row/col intensity gradient RMS)
- perimetric_complexity : (edge_length²) / (ink_area × N characters)
                         (word-level, length-normalized Pelli complexity)
- contrast_michelson    : |L_bg - L_text| / (L_bg + L_text)
                          text pixels = GT ink mask, bg = remaining polygon pixels
- contrast_wcag         : WCAG 2.1 contrast ratio (L_lighter+0.05)/(L_darker+0.05)
                          using linearised sRGB luminance; AA≥4.5, AAA≥7
"""
import json
import sys
import numpy as np
import cv2
from pathlib import Path
from shapely.geometry import Polygon
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from itti_saliency import compute_itti_saliency

# GT character masks (binary, white=ink), one PNG per image stem
_CHARGT_DIR = Path('data/totaltext/chargt')
_CROSS3     = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def load_image(image_path):
    img_gray  = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return img_gray, img_color


# ── polygon geometry ─────────────────────────────────────────────────────────

def compute_polygon_area(polygon_coords):
    try:
        return Polygon(polygon_coords).area
    except Exception:
        return 0.0


def compute_polygon_curvatures(polygon_coords):
    """Interior-angle-style value at every polygon vertex (degrees).

    Each value is calculated from the same consecutive three-point window
    ``(previous, current, next)`` used by ``compute_polygon_curvature``:
    ``abs(180 - angle(current-previous, next-current))``. Thus a locally
    straight path is near 180 degrees. This is the historical feature formula;
    it is not curvature normalized by arc length.
    Keeping this intermediate public lets visualizations show the values that
    were actually averaged for the CSV feature.
    """
    if len(polygon_coords) < 3:
        return np.array([], dtype=np.float64)
    points = np.asarray(polygon_coords, dtype=np.float64)
    n = len(points)
    curvatures = []
    for i in range(n):
        p1, p2, p3 = points[(i-1) % n], points[i], points[(i+1) % n]
        v1, v2 = p2 - p1, p3 - p2
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 > 0 and l2 > 0:
            cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
            curvatures.append(abs(180.0 - np.degrees(np.arccos(cos_a))))
        else:
            curvatures.append(np.nan)
    return np.asarray(curvatures, dtype=np.float64)


def compute_polygon_curvature(polygon_coords):
    """Average of the finite per-vertex interior-angle-style values."""
    curvatures = compute_polygon_curvatures(polygon_coords)
    curvatures = curvatures[np.isfinite(curvatures)]
    return float(np.mean(curvatures)) if len(curvatures) else 0.0


def get_polygon_mask(polygon_coords, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], 255)
    return mask


# ── GT ink mask ──────────────────────────────────────────────────────────────

def load_gt_ink_mask(image_stem, polygon_coords, image_shape):
    """
    Load the official TotalText character GT mask for one image, masked to
    the given word polygon.

    GT files live in data/totaltext/chargt/<stem>.png (binary, white=ink).
    Returns None when the GT file is missing.
    """
    gt_path = _CHARGT_DIR / f'{image_stem}.png'
    if not gt_path.exists():
        return None
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    mask = get_polygon_mask(polygon_coords, image_shape)
    return cv2.bitwise_and(gt_bin, gt_bin, mask=mask)


def ink_mask_to_edge(ink_mask):
    """1-pixel boundary of an ink mask via morphological gradient."""
    return cv2.subtract(
        cv2.dilate(ink_mask, _CROSS3, iterations=1),
        cv2.erode(ink_mask,  _CROSS3, iterations=1),
    )


# ── edge-map-based metrics ───────────────────────────────────────────────────

def compute_edge_density(edge_map, mask):
    """Ratio of edge pixels to total polygon pixels."""
    n_total = int(np.count_nonzero(mask))
    if n_total == 0:
        return 0.0
    return float(np.count_nonzero(edge_map)) / n_total


def compute_perimetric_complexity(edge_map, ink_mask, n_characters=1):
    """
    Word-level PC = perimeter² / (ink_area × N characters).

    perimeter = boundary pixels of GT ink mask
    ink_area  = white pixels in GT ink mask
    N         = number of characters in the annotated word

    Pelli et al. (2006) define perimeter² / ink_area for a single glyph.
    Dividing the whole-word value by N removes its first-order inflation with
    word length while retaining the original value when N = 1.
    """
    perimeter = int(np.count_nonzero(edge_map))
    ink_area  = int(np.count_nonzero(ink_mask))
    n_characters = int(n_characters)
    if ink_area == 0 or perimeter == 0 or n_characters <= 0:
        return 0.0
    return float((perimeter ** 2) / (ink_area * n_characters))


# ── intensity-based metrics ──────────────────────────────────────────────────

def compute_local_contrast(img_gray, mask):
    """Standard deviation of pixel intensities within the polygon."""
    pixels = img_gray[mask > 0]
    return float(np.std(pixels)) if len(pixels) > 0 else 0.0


def compute_average_luminance(img_gray, mask):
    """Mean pixel brightness within the polygon."""
    pixels = img_gray[mask > 0]
    return float(np.mean(pixels)) if len(pixels) > 0 else 0.0


def compute_spatial_frequency_detail(image, mask):
    """
    Overall activity level and the exact contributing pixel differences.

    RF = sqrt(mean((I[i,j] - I[i,j-1])²))   # row frequency
    CF = sqrt(mean((I[i,j] - I[i-1,j])²))   # col frequency
    SF = sqrt(RF² + CF²)

    Returns ``(sf, detail_map, valid_map)``. ``detail_map`` places each valid
    horizontal/vertical difference at its second pixel and combines them as
    sqrt(dx² + dy²).  The scalar ``sf`` is still computed exactly from the
    separate RF and CF means above; the map is the spatial intermediate, not
    an alternative formula for the scalar.
    """
    gray = image.astype(np.float64) / 255.0
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        empty = np.zeros(image.shape, dtype=np.float32)
        return 0.0, empty, np.zeros(image.shape, dtype=np.uint8)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    roi      = gray[r0:r1+1, c0:c1+1]
    roi_mask = mask[r0:r1+1, c0:c1+1]
    if roi.shape[0] < 2 or roi.shape[1] < 2:
        empty = np.zeros(image.shape, dtype=np.float32)
        return 0.0, empty, np.zeros(image.shape, dtype=np.uint8)

    row_diff = np.diff(roi, axis=1)
    row_m    = roi_mask[:, 1:] & roi_mask[:, :-1]
    rf = float(np.sqrt(np.mean(row_diff[row_m > 0] ** 2))) if np.sum(row_m) > 0 else 0.0

    col_diff = np.diff(roi, axis=0)
    col_m    = roi_mask[1:, :] & roi_mask[:-1, :]
    cf = float(np.sqrt(np.mean(col_diff[col_m > 0] ** 2))) if np.sum(col_m) > 0 else 0.0

    dx2 = np.zeros(roi.shape, dtype=np.float64)
    dy2 = np.zeros(roi.shape, dtype=np.float64)
    dx2[:, 1:][row_m > 0] = row_diff[row_m > 0] ** 2
    dy2[1:, :][col_m > 0] = col_diff[col_m > 0] ** 2
    detail_roi = np.sqrt(dx2 + dy2).astype(np.float32)
    valid_roi = np.zeros(roi.shape, dtype=np.uint8)
    valid_roi[:, 1:][row_m > 0] = 255
    valid_roi[1:, :][col_m > 0] = 255

    detail = np.zeros(image.shape, dtype=np.float32)
    valid = np.zeros(image.shape, dtype=np.uint8)
    detail[r0:r1+1, c0:c1+1] = detail_roi
    valid[r0:r1+1, c0:c1+1] = valid_roi
    return float(np.sqrt(rf ** 2 + cf ** 2)), detail, valid


def compute_spatial_frequency(image, mask):
    """Return the scalar from ``compute_spatial_frequency_detail``."""
    value, _, _ = compute_spatial_frequency_detail(image, mask)
    return value


# ── contrast metrics ─────────────────────────────────────────────────────────

def compute_rms_contrast_ink(img_gray, ink_mask):
    """RMS contrast of pixel intensities within the GT ink area.

    RMS = std(I/255) over ink pixels only, range [0, 1].
    Captures internal luminance variation within the ink strokes
    (e.g. shading, texture, uneven illumination on the text itself).
    """
    pixels = img_gray[ink_mask > 0].astype(np.float64) / 255.0
    if len(pixels) == 0:
        return 0.0
    return float(np.std(pixels))


def compute_image_saliency(img_color):
    """Spectral residual saliency map (Hou & Zhang 2007), float32 [0, 1].

    SR = IFFT( exp(log|F| - avg_filter(log|F|) + i*phase(F)) )²
    Implemented without opencv-contrib.
    """
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f    = np.fft.fft2(gray)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase   = np.angle(f)
    # average filter on log amplitude (spectral residual)
    avg_log = cv2.blur(log_amp, (3, 3))
    sr      = log_amp - avg_log
    # reconstruct and compute saliency
    recon  = np.fft.ifft2(np.exp(sr + 1j * phase)).real
    sal    = cv2.GaussianBlur(recon ** 2, (9, 9), 2.5)
    # normalise to [0, 1]
    sal_min, sal_max = sal.min(), sal.max()
    if sal_max > sal_min:
        sal = (sal - sal_min) / (sal_max - sal_min)
    return sal.astype(np.float32)


def compute_saliency_ink(sal_map, ink_mask):
    """Mean spectral-residual saliency over GT ink pixels."""
    vals = sal_map[ink_mask > 0]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def compute_itti_ink(itti_result, ink_mask):
    """Extract mean Itti saliency and per-channel contributions over ink pixels.

    Returns (itti_saliency, itti_intensity, itti_color, itti_orientation).
    """
    idx = ink_mask > 0
    if not np.any(idx):
        return 0.0, 0.0, 0.0, 0.0
    sal = itti_result['saliency_map'][idx]
    ci  = itti_result['channel_maps']['intensity'][idx]
    cc  = itti_result['channel_maps']['color'][idx]
    co  = itti_result['channel_maps']['orientation'][idx]
    return float(sal.mean()), float(ci.mean()), float(cc.mean()), float(co.mean())

def compute_contrast_weber(img_gray, mask, ink_mask):
    """Weber contrast between text and background (Pelli & Bex 2013).

    C_W = |L_text - L_bg| / L_bg,  range [0, inf)
    Preferred over Michelson for letter stimuli on larger backgrounds.
    """
    text_pixels = img_gray[(mask > 0) & (ink_mask > 0)]
    bg_pixels   = img_gray[(mask > 0) & (ink_mask == 0)]
    if len(text_pixels) == 0 or len(bg_pixels) == 0:
        return 0.0
    L_text = float(text_pixels.mean())
    L_bg   = float(bg_pixels.mean())
    return float(abs(L_text - L_bg) / L_bg) if L_bg > 0 else 0.0


def compute_rms_contrast_bg(img_gray, mask, ink_mask):
    """RMS contrast of background pixels within the polygon.

    C_RMS_bg = std(I/255) over non-ink polygon pixels, range [0, 1].
    Captures background texture complexity (Scharff & Ahumada 2003).
    """
    bg_pixels = img_gray[(mask > 0) & (ink_mask == 0)].astype(np.float64) / 255.0
    if len(bg_pixels) == 0:
        return 0.0
    return float(np.std(bg_pixels))


def compute_masking_index(contrast_weber, rms_bg):
    """Global Masking Index (Scharff & Ahumada 2003).

    C_M = C_Weber / sqrt(1 + (C_RMS_bg / 0.05)^2)
    r = 0.79 with reading speed on textured backgrounds,
    vs r = 0.43 for text contrast alone.
    """
    return float(contrast_weber / np.sqrt(1.0 + (rms_bg / 0.05) ** 2))


def compute_deviation_map(img_gray, sample_mask):
    """Absolute pixel deviation from the mean used by an RMS/std feature."""
    pixels = img_gray[sample_mask > 0].astype(np.float32)
    center = float(pixels.mean()) if len(pixels) else 0.0
    return np.abs(img_gray.astype(np.float32) - center)


def compute_text_background_difference_map(img_gray, mask, ink_mask, linear_srgb=False):
    """Spatial text/background luminance differences used for contrast overlays.

    Ink pixels are compared with mean background luminance and background
    pixels with mean ink luminance.  ``linear_srgb=True`` uses the same
    conversion as the WCAG scalar.
    """
    ink_idx = (mask > 0) & (ink_mask > 0)
    bg_idx = (mask > 0) & (ink_mask == 0)
    output = np.zeros(img_gray.shape, dtype=np.float32)
    if not np.any(ink_idx) or not np.any(bg_idx):
        return output
    values = (
        _srgb_to_linear(img_gray).astype(np.float32)
        if linear_srgb
        else img_gray.astype(np.float32) / 255.0
    )
    ink_mean = float(values[ink_idx].mean())
    bg_mean = float(values[bg_idx].mean())
    output[ink_idx] = np.abs(values[ink_idx] - bg_mean)
    output[bg_idx] = np.abs(values[bg_idx] - ink_mean)
    return output


def compute_contrast_michelson(img_gray, mask, ink_mask):
    """
    Michelson contrast between text and background within the polygon.

    text pixels  = GT ink mask > 0  (actual character fills)
    bg pixels    = polygon interior NOT covered by ink

    C = |L_bg - L_text| / (L_bg + L_text),  range [0, 1]
    """
    text_pixels = img_gray[(mask > 0) & (ink_mask > 0)]
    bg_pixels   = img_gray[(mask > 0) & (ink_mask == 0)]

    if len(text_pixels) == 0 or len(bg_pixels) == 0:
        return 0.0

    L_text = float(text_pixels.mean())
    L_bg   = float(bg_pixels.mean())
    denom  = L_bg + L_text
    return float(abs(L_bg - L_text) / denom) if denom > 0 else 0.0


def _srgb_to_linear(values_uint8):
    """Convert uint8 grayscale [0, 255] to linearised sRGB luminance [0, 1]."""
    c = values_uint8.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def compute_contrast_wcag(img_gray, mask, ink_mask):
    """
    WCAG 2.1 contrast ratio.

    ratio = (L_lighter + 0.05) / (L_darker + 0.05)
    where L is mean relative luminance (linearised sRGB) of text / bg pixels.

    Range [1, 21].  AA threshold: 4.5,  AAA threshold: 7.
    """
    text_pixels = img_gray[(mask > 0) & (ink_mask > 0)]
    bg_pixels   = img_gray[(mask > 0) & (ink_mask == 0)]

    if len(text_pixels) == 0 or len(bg_pixels) == 0:
        return 1.0

    L_text = float(_srgb_to_linear(text_pixels).mean())
    L_bg   = float(_srgb_to_linear(bg_pixels).mean())

    L_lighter = max(L_text, L_bg)
    L_darker  = min(L_text, L_bg)
    return float((L_lighter + 0.05) / (L_darker + 0.05))


# ── combined feature extraction ──────────────────────────────────────────────

def extract_features_from_region(
    img_gray,
    polygon_coords,
    ink_mask,
    sal_map=None,
    itti_result=None,
    n_characters=1,
    return_details=False,
):
    """
    Extract all visual features for one text region.

    ink_mask    : binary GT character mask (white=ink), already masked to polygon.
    sal_map     : full-image spectral residual saliency map (float32 [0,1]), optional.
    itti_result : dict returned by compute_itti_saliency(), optional.
    n_characters: character count N used to length-normalize whole-word PC.
    """
    polygon_size     = compute_polygon_area(polygon_coords)
    curvatures       = compute_polygon_curvatures(polygon_coords)
    finite_curvature = curvatures[np.isfinite(curvatures)]
    avg_curvature    = float(np.mean(finite_curvature)) if len(finite_curvature) else 0.0
    mask             = get_polygon_mask(polygon_coords, img_gray.shape)
    edge_map         = ink_mask_to_edge(ink_mask)
    ink_area         = int(np.count_nonzero(ink_mask))
    spatial_frequency, sf_detail, sf_valid = compute_spatial_frequency_detail(img_gray, mask)

    itti_sal = itti_i = itti_c = itti_o = None
    if itti_result is not None:
        itti_sal, itti_i, itti_c, itti_o = compute_itti_ink(itti_result, ink_mask)

    c_weber  = compute_contrast_weber(img_gray, mask, ink_mask)
    c_rms_bg = compute_rms_contrast_bg(img_gray, mask, ink_mask)

    result = {
        'polygon_size':          polygon_size,
        'ink_area':              ink_area,
        'edge_density':          compute_edge_density(edge_map, mask),
        'luminance_std':         compute_local_contrast(img_gray, mask),
        'luminance_mean':        compute_average_luminance(img_gray, mask),
        'avg_curvature':         avg_curvature,
        'spatial_frequency':     spatial_frequency,
        'perimetric_complexity': compute_perimetric_complexity(
            edge_map, ink_mask, n_characters=n_characters),
        'contrast_michelson':    compute_contrast_michelson(img_gray, mask, ink_mask),
        'contrast_wcag':         compute_contrast_wcag(img_gray, mask, ink_mask),
        'contrast_rms_ink':      compute_rms_contrast_ink(img_gray, ink_mask),
        'contrast_weber':        c_weber,
        'contrast_rms_bg':       c_rms_bg,
        'masking_index':         compute_masking_index(c_weber, c_rms_bg),
        'saliency_ink':          compute_saliency_ink(sal_map, ink_mask) if sal_map is not None else None,
        'itti_saliency_ink':     itti_sal,
        'itti_intensity_ink':    itti_i,
        'itti_color_ink':        itti_c,
        'itti_orientation_ink':  itti_o,
    }
    if return_details:
        background_mask = cv2.bitwise_and(mask, cv2.bitwise_not(ink_mask))
        result['_details'] = {
            # These are the exact masks/maps sampled by the scalar functions.
            'polygon_mask':             mask,
            'ink_mask':                 ink_mask,
            'background_mask':          background_mask,
            'edge_map':                 edge_map,
            'curvature_by_vertex':       curvatures,
            'luminance_map':             img_gray,
            'polygon_luminance_deviation': compute_deviation_map(img_gray, mask),
            'ink_luminance_deviation':   compute_deviation_map(img_gray, ink_mask),
            'background_luminance_deviation': compute_deviation_map(img_gray, background_mask),
            'text_background_difference': compute_text_background_difference_map(
                img_gray, mask, ink_mask,
            ),
            'text_background_difference_linear': compute_text_background_difference_map(
                img_gray, mask, ink_mask, linear_srgb=True,
            ),
            'spatial_frequency_map':     sf_detail,
            'spatial_frequency_valid':   sf_valid,
            'spectral_saliency_map':     sal_map,
            'itti_saliency_map':         itti_result['saliency_map'] if itti_result is not None else None,
            'itti_channel_maps':         itti_result['channel_maps'] if itti_result is not None else None,
        }
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    GT_PATH        = 'data/totaltext/anno.json'
    IMAGE_BASE     = 'data/totaltext'
    OUTPUT_CSV     = 'results/totaltext_16_text_region_features.csv'
    TARGET_DIR     = '16/'

    print("=" * 80)
    print("TotalText Text Region Feature Extraction")
    print(f"Target Directory: {TARGET_DIR}")
    print("=" * 80)

    with open(GT_PATH) as f:
        gt_data = json.load(f)

    gt_images = {img['id']: img for img in gt_data['images']}

    annotations_16 = [
        {'annotation': ann, 'image_info': gt_images[ann['image_id']]}
        for ann in gt_data['annotations']
        if gt_images.get(ann['image_id'], {}).get('file_name', '').startswith(TARGET_DIR)
    ]
    print(f"Found {len(annotations_16)} annotations in {TARGET_DIR}")

    image_cache   = {}   # image_id -> img_gray
    sal_cache     = {}   # image_id -> spectral residual saliency map
    itti_cache    = {}   # image_id -> itti_result dict
    gt_cache      = {}   # stem -> full-image GT binary mask
    results       = []
    SAL_DIR       = Path('results/saliency')
    SAL_DIR.mkdir(parents=True, exist_ok=True)

    for item in tqdm(annotations_16):
        ann      = item['annotation']
        img_info = item['image_info']
        file_name = img_info.get('file_name', '')
        caption   = ann.get('caption', '')
        bbox_list = ann.get('bbox', [])
        stem      = Path(file_name).stem

        if ann['image_id'] not in image_cache:
            img_path = Path(IMAGE_BASE) / file_name
            if not img_path.exists():
                print(f"Warning: image not found: {img_path}")
                continue
            img_gray, img_color = load_image(img_path)
            sal_map_full = compute_image_saliency(img_color)
            img_rgb      = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            itti_result  = compute_itti_saliency(img_rgb)
            image_cache[ann['image_id']] = img_gray
            sal_cache[ann['image_id']]   = sal_map_full
            itti_cache[ann['image_id']]  = itti_result
            sal_png = SAL_DIR / f'{Path(file_name).stem}.png'
            cv2.imwrite(str(sal_png), (sal_map_full * 255).astype(np.uint8))
        img_gray    = image_cache[ann['image_id']]
        sal_map     = sal_cache[ann['image_id']]
        itti_result = itti_cache[ann['image_id']]

        if stem not in gt_cache:
            gt_path = _CHARGT_DIR / f'{stem}.png'
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
            if gt is not None:
                _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                gt_cache[stem] = gt_bin
            else:
                gt_cache[stem] = None
                print(f"Warning: GT mask not found for {stem}")
        gt_full = gt_cache[stem]

        for i, word in enumerate(caption.split()):
            if i >= len(bbox_list) or word == '###':
                continue
            polygon = bbox_list[i]
            if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
                continue
            if gt_full is None:
                continue
            poly_mask = get_polygon_mask(polygon, img_gray.shape)
            ink_mask  = cv2.bitwise_and(gt_full, gt_full, mask=poly_mask)
            try:
                feats = extract_features_from_region(
                    img_gray, polygon, ink_mask, sal_map, itti_result,
                    n_characters=len(word),
                )
                H, W  = img_gray.shape
                results.append({
                    'image_name':           file_name,
                    'H': H, 'W': W,
                    'text':                 word,
                    'polygon_size':         round(feats['polygon_size'],         2),
                    'ink_area':             feats['ink_area'],
                    'edge_density':         round(feats['edge_density'],         4),
                    'luminance_std':        round(feats['luminance_std'],        2),
                    'luminance_mean':       round(feats['luminance_mean'],       2),
                    'avg_curvature':        round(feats['avg_curvature'],        2),
                    'spatial_frequency':    round(feats['spatial_frequency'],    4),
                    'perimetric_complexity': round(feats['perimetric_complexity'], 2),
                    'contrast_michelson':   round(feats['contrast_michelson'],   4),
                    'contrast_wcag':        round(feats['contrast_wcag'],        4),
                    'contrast_rms_ink':     round(feats['contrast_rms_ink'],          4),
                    'contrast_weber':       round(feats['contrast_weber'],            4),
                    'contrast_rms_bg':      round(feats['contrast_rms_bg'],           4),
                    'masking_index':        round(feats['masking_index'],             4),
                    'saliency_ink':         round(feats['saliency_ink'],              4),
                    'itti_saliency_ink':    round(feats['itti_saliency_ink'],         4),
                    'itti_intensity_ink':   round(feats['itti_intensity_ink'],        4),
                    'itti_color_ink':       round(feats['itti_color_ink'],            4),
                    'itti_orientation_ink': round(feats['itti_orientation_ink'],      4),
                })
            except Exception as e:
                print(f"Error processing {file_name} – {word}: {e}")

    df = pd.DataFrame(results)
    print(f"\nProcessed {len(df)} text regions across {df['image_name'].nunique()} images.")

    print("\nFeature statistics:")
    for col in ['edge_density', 'luminance_std', 'luminance_mean',
                'perimetric_complexity', 'contrast_michelson', 'contrast_wcag',
                'contrast_weber', 'contrast_rms_bg', 'masking_index']:
        print(f"  {col:28s}  mean={df[col].mean():.4f}  std={df[col].std():.4f}"
              f"  min={df[col].min():.4f}  max={df[col].max():.4f}")

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {Path(OUTPUT_CSV).absolute()}")
    print(df.head(5).to_string())


if __name__ == '__main__':
    main()
