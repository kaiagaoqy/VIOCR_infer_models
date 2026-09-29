"""
Itti-style saliency map following the paper's still-image feature equations.

Implements the color and intensity stages described in
Itti, Koch, and Niebur (1998):

    1. Build 9-level OpenCV Gaussian pyramids with cv2.pyrDown.
    2. Compute six center-surround maps for scales (2,5), (2,6),
       (3,6), (3,7), (4,7), and (4,8).
    3. Normalize RGB by intensity above the low-luminance threshold, create
       R/G/B/Y channels, and compute paper-style double-opponent RG/BY maps.
    4. Use fixed-parameter cv2.getGaborKernel orientation filters.
    5. Normalize feature maps and combine conspicuity maps at pyramid scale 4.

Public API
----------
compute_itti_saliency(image, mask=None) -> dict
    image : RGB np.ndarray (H, W, 3)
    mask  : optional bool/uint8 mask; when given, scalar summaries are
            computed over mask pixels only.

Returned dict keys
------------------
saliency_map          : (H, W) float32 total saliency in [0,1]
mean_saliency         : float
max_saliency          : float
std_saliency          : float
saliency_concentration: float  -- fraction of pixels above the 90th percentile
mean_itti_intensity   : float  -- WEIGHT_INTENSITY * N(conspicuity_I), mean over mask
mean_itti_color       : float  -- WEIGHT_COLOR     * N(conspicuity_C), mean over mask
mean_itti_orientation : float  -- WEIGHT_ORIENTATION * N(conspicuity_O), mean over mask
num_feature_maps      : int
channel_maps          : dict with keys 'intensity', 'color', 'orientation'
                        -- full-resolution (H, W) weighted channel maps,
                        -- useful for per-region stats without recomputing the pyramid
conspicuity_maps      : dict with keys 'intensity', 'color', 'orientation'
                        -- raw (un-normalized, un-weighted) full-resolution maps
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


NUM_PYRAMID_LEVELS = 9
LOCAL_MAX_NEIGHBORHOOD = 16
SALIENCY_SCALE = 4
ORIENTATIONS = (0, 45, 90, 135)
WEIGHT_INTENSITY = 1.0 / 3.0
WEIGHT_COLOR = 1.0 / 3.0
WEIGHT_ORIENTATION = 1.0 / 3.0

GABOR_KERNEL_SIZE = 9
GABOR_SIGMA = 1.1
GABOR_LAMBDA = 3.75
GABOR_GAMMA = 0.3
GABOR_PSI = 0.0


def _create_gabor_kernels() -> Dict[int, np.ndarray]:
    kernels = {}
    for theta in ORIENTATIONS:
        kernels[theta] = cv2.getGaborKernel(
            (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
            GABOR_SIGMA,
            np.deg2rad(theta),
            GABOR_LAMBDA,
            GABOR_GAMMA,
            GABOR_PSI,
            ktype=cv2.CV_32F,
        )
    return kernels


GABOR_KERNELS = _create_gabor_kernels()


def _prepare_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("compute_itti_saliency expects an RGB image with shape (H, W, 3).")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def _extract_rgb_i(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    intensity = (r + g + b) / 3.0
    return r, g, b, intensity


def _create_gaussian_pyramid(src: np.ndarray) -> List[np.ndarray]:
    pyramid = [src.astype(np.float32)]
    for _ in range(1, NUM_PYRAMID_LEVELS):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _center_surround_diff(gaussian_maps: List[np.ndarray]) -> List[np.ndarray]:
    feature_maps = []
    for scale in range(2, 5):
        height, width = gaussian_maps[scale].shape[:2]
        target_size = (width, height)
        surround_3 = cv2.resize(gaussian_maps[scale + 3], target_size, interpolation=cv2.INTER_LINEAR)
        feature_maps.append(cv2.absdiff(gaussian_maps[scale], surround_3))
        surround_4 = cv2.resize(gaussian_maps[scale + 4], target_size, interpolation=cv2.INTER_LINEAR)
        feature_maps.append(cv2.absdiff(gaussian_maps[scale], surround_4))
    return feature_maps


def _gaussian_pyramid_center_surround(src: np.ndarray) -> List[np.ndarray]:
    return _center_surround_diff(_create_gaussian_pyramid(src))


def _intensity_feature_maps(intensity: np.ndarray) -> List[np.ndarray]:
    return _gaussian_pyramid_center_surround(intensity)


def _create_color_channels(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, intensity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    threshold = 0.1 * float(np.max(intensity))
    visible = intensity > threshold
    denominator = np.where(visible, intensity, 1.0)

    r_norm = np.where(visible, r / denominator, 0.0)
    g_norm = np.where(visible, g / denominator, 0.0)
    b_norm = np.where(visible, b / denominator, 0.0)

    red    = np.maximum(r_norm - (g_norm + b_norm) / 2.0, 0.0)
    green  = np.maximum(g_norm - (r_norm + b_norm) / 2.0, 0.0)
    blue   = np.maximum(b_norm - (r_norm + g_norm) / 2.0, 0.0)
    yellow = np.maximum(
        (r_norm + g_norm) / 2.0 - np.abs(r_norm - g_norm) / 2.0 - b_norm, 0.0,
    )
    return red, green, blue, yellow


def _color_feature_maps(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, intensity: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    red, green, blue, yellow = _create_color_channels(r, g, b, intensity)
    red_p    = _create_gaussian_pyramid(red)
    green_p  = _create_gaussian_pyramid(green)
    blue_p   = _create_gaussian_pyramid(blue)
    yellow_p = _create_gaussian_pyramid(yellow)

    rg_maps, by_maps = [], []
    for scale in range(2, 5):
        height, width = red_p[scale].shape[:2]
        target_size = (width, height)
        for delta in (3, 4):
            s = scale + delta
            rg_maps.append(np.abs(
                (red_p[scale] - green_p[scale]) -
                (cv2.resize(green_p[s], target_size, interpolation=cv2.INTER_LINEAR) -
                 cv2.resize(red_p[s],   target_size, interpolation=cv2.INTER_LINEAR))
            ))
            by_maps.append(np.abs(
                (blue_p[scale] - yellow_p[scale]) -
                (cv2.resize(yellow_p[s], target_size, interpolation=cv2.INTER_LINEAR) -
                 cv2.resize(blue_p[s],   target_size, interpolation=cv2.INTER_LINEAR))
            ))
    return rg_maps, by_maps


def _orientation_feature_maps(intensity: np.ndarray) -> List[np.ndarray]:
    gaussian_intensity = _create_gaussian_pyramid(intensity)
    orientation_maps = []
    for theta in ORIENTATIONS:
        gabor_output = [np.empty((1, 1), dtype=np.float32), np.empty((1, 1), dtype=np.float32)]
        for level in range(2, NUM_PYRAMID_LEVELS):
            filtered = cv2.filter2D(gaussian_intensity[level], cv2.CV_32F, GABOR_KERNELS[theta])
            gabor_output.append(filtered)
        orientation_maps.extend(_center_surround_diff(gabor_output))
    return orientation_maps


def _range_normalize(src: np.ndarray) -> np.ndarray:
    min_val, max_val, _, _ = cv2.minMaxLoc(src.astype(np.float32))
    if max_val != min_val:
        return (src - min_val) / (max_val - min_val)
    return src - min_val


def _avg_local_max_except_global(src: np.ndarray) -> float:
    kernel = np.ones((LOCAL_MAX_NEIGHBORHOOD, LOCAL_MAX_NEIGHBORHOOD), dtype=np.uint8)
    dilated = cv2.dilate(src.astype(np.float32), kernel)
    local_max_mask = (src == dilated) & (src > 0)
    if not np.any(local_max_mask):
        return 0.0
    global_max = float(np.max(src))
    local_maxima = src[local_max_mask]
    other_local_maxima = local_maxima[local_maxima < global_max]
    return float(np.mean(other_local_maxima)) if len(other_local_maxima) > 0 else 0.0


def _normalization(src: np.ndarray) -> np.ndarray:
    normalized = _range_normalize(src.astype(np.float32))
    local_max_mean = _avg_local_max_except_global(normalized)
    coefficient = (1.0 - local_max_mean) ** 2
    return normalized * coefficient


def _across_scale_addition(feature_maps: List[np.ndarray], target_shape: Tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    combined = np.zeros((target_height, target_width), dtype=np.float32)
    for feature_map in feature_maps[:6]:
        normalized = _normalization(feature_map)
        resized = cv2.resize(normalized, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        combined += resized.astype(np.float32)
    return combined


def _intensity_conspicuity_map(feature_maps, target_shape):
    return _across_scale_addition(feature_maps, target_shape)


def _color_conspicuity_map(rg_maps, by_maps, target_shape):
    return _intensity_conspicuity_map(rg_maps, target_shape) + _intensity_conspicuity_map(by_maps, target_shape)


def _orientation_conspicuity_map(orientation_feature_maps, target_shape):
    conspicuity = np.zeros(target_shape, dtype=np.float32)
    for index in range(len(ORIENTATIONS)):
        angle_maps = orientation_feature_maps[index * 6:(index + 1) * 6]
        conspicuity += _normalization(_intensity_conspicuity_map(angle_maps, target_shape))
    return conspicuity


def _resize_full(src: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(src.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def compute_itti_saliency(image: np.ndarray, mask: np.ndarray = None) -> Dict:
    """
    Compute an Itti-style saliency map with paper-style color features.

    Args:
        image: RGB image with shape (H, W, 3).
        mask:  Optional bool/uint8 mask; scalar summaries use only masked pixels.

    Returns dict with 'saliency_map', scalar summaries, 'channel_maps'
    (per-channel weighted full-resolution maps for per-region sampling),
    and 'conspicuity_maps' (raw unweighted maps for visualization).
    """
    img = _prepare_rgb_image(image)
    height, width = img.shape[:2]
    r, g, b, intensity = _extract_rgb_i(img)

    intensity_maps   = _intensity_feature_maps(intensity)
    rg_maps, by_maps = _color_feature_maps(r, g, b, intensity)
    orientation_maps = _orientation_feature_maps(intensity)
    scale4_shape     = _create_gaussian_pyramid(intensity)[SALIENCY_SCALE].shape

    conspicuity_i_s4 = _intensity_conspicuity_map(intensity_maps, scale4_shape)
    conspicuity_c_s4 = _color_conspicuity_map(rg_maps, by_maps, scale4_shape)
    conspicuity_o_s4 = _orientation_conspicuity_map(orientation_maps, scale4_shape)

    channel_i_s4 = WEIGHT_INTENSITY    * _normalization(conspicuity_i_s4)
    channel_c_s4 = WEIGHT_COLOR        * _normalization(conspicuity_c_s4)
    channel_o_s4 = WEIGHT_ORIENTATION  * _normalization(conspicuity_o_s4)

    saliency_s4  = channel_i_s4 + channel_c_s4 + channel_o_s4
    saliency     = _resize_full(saliency_s4,  width, height)
    channel_i    = _resize_full(channel_i_s4, width, height)
    channel_c    = _resize_full(channel_c_s4, width, height)
    channel_o    = _resize_full(channel_o_s4, width, height)

    conspicuity_i = _resize_full(conspicuity_i_s4, width, height)
    conspicuity_c = _resize_full(conspicuity_c_s4, width, height)
    conspicuity_o = _resize_full(conspicuity_o_s4, width, height)

    if mask is not None:
        idx = mask.astype(bool)
    else:
        idx = np.ones((height, width), dtype=bool)

    sv  = saliency[idx]
    civ = channel_i[idx]
    ccv = channel_c[idx]
    cov = channel_o[idx]

    if len(sv) == 0:
        sv = civ = ccv = cov = np.array([0.0], dtype=np.float32)

    return {
        "saliency_map":           saliency,
        "mean_saliency":          float(np.mean(sv)),
        "max_saliency":           float(np.max(sv)),
        "std_saliency":           float(np.std(sv)),
        "saliency_concentration": float(np.sum(sv > np.percentile(sv, 90)) / len(sv)),
        "mean_itti_intensity":    float(np.mean(civ)),
        "mean_itti_color":        float(np.mean(ccv)),
        "mean_itti_orientation":  float(np.mean(cov)),
        "num_feature_maps":       len(intensity_maps) + len(rg_maps) + len(by_maps) + len(orientation_maps),
        "channel_maps": {
            "intensity":   channel_i,
            "color":       channel_c,
            "orientation": channel_o,
        },
        "conspicuity_maps": {
            "intensity":   conspicuity_i,
            "color":       conspicuity_c,
            "orientation": conspicuity_o,
        },
    }


def compute_itti_saliency_detailed(image: np.ndarray, mask: np.ndarray = None) -> Dict:
    """Alias for compute_itti_saliency to preserve existing API."""
    return compute_itti_saliency(image, mask)
