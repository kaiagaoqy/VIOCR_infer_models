#!/usr/bin/env python3
"""Visualize every spatially interpretable text-region feature on source images.

The CSV stores one scalar per text region.  This script recovers the matching
TotalText polygons from ``anno.json`` and, where possible, also redraws the
pixel/vertex-level quantity from which that scalar was computed.

Outputs are written to::

    results/feat/<feature_name>/<image_stem>.jpg

Every image contains:
  * either a whole-image feature overlay or a binary map at source resolution;
  * text polygons outlined with a dataset-wide scalar color scale;
  * the scalar value and GT word in a side panel (never over the text);
  * a spatial overlay for features with a meaningful intermediate map.

Examples:
  * ``avg_curvature`` shows the degree value for every 3-point vertex window,
    not only its mean;
  * ``edge_density`` shows the actual GT-ink boundary pixels;
  * Itti saliency shows the total, intensity, color, and orientation maps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_text_region_features import (  # noqa: E402
    compute_image_saliency,
    compute_spatial_frequency_detail,
    extract_features_from_region,
    get_polygon_mask,
    load_image,
)
from itti_saliency import compute_itti_saliency  # noqa: E402


FEATURES = [
    "polygon_size",
    "ink_area",
    "edge_density",
    "luminance_std",
    "luminance_mean",
    "avg_curvature",
    "spatial_frequency",
    "perimetric_complexity",
    "contrast_michelson",
    "contrast_wcag",
    "contrast_rms_ink",
    "contrast_weber",
    "contrast_rms_bg",
    "masking_index",
    "saliency_ink",
    "itti_saliency_ink",
    "itti_intensity_ink",
    "itti_color_ink",
    "itti_orientation_ink",
]

MAP_FEATURES = {
    "ink_area",
    "edge_density",
    "luminance_std",
    "luminance_mean",
    "avg_curvature",
    "spatial_frequency",
    "perimetric_complexity",
    "contrast_michelson",
    "contrast_wcag",
    "contrast_rms_ink",
    "contrast_weber",
    "contrast_rms_bg",
    "masking_index",
    "saliency_ink",
    "itti_saliency_ink",
    "itti_intensity_ink",
    "itti_color_ink",
    "itti_orientation_ink",
}

BINARY_MAP_FEATURES = {
    "ink_area",
    "edge_density",
    "perimetric_complexity",
}

FULL_IMAGE_OVERLAY_FEATURES = {
    "luminance_mean",
    "spatial_frequency",
    "saliency_ink",
    "itti_saliency_ink",
    "itti_intensity_ink",
    "itti_color_ink",
    "itti_orientation_ink",
}

FEATURE_DECIMALS = {
    "polygon_size": 2,
    "ink_area": 0,
    "edge_density": 4,
    "luminance_std": 2,
    "luminance_mean": 2,
    "avg_curvature": 2,
    "spatial_frequency": 4,
    "perimetric_complexity": 2,
    "contrast_michelson": 4,
    "contrast_wcag": 4,
    "contrast_rms_ink": 4,
    "contrast_weber": 4,
    "contrast_rms_bg": 4,
    "masking_index": 4,
    "saliency_ink": 4,
    "itti_saliency_ink": 4,
    "itti_intensity_ink": 4,
    "itti_color_ink": 4,
    "itti_orientation_ink": 4,
}

HEADER_HEIGHT = 54
LEGEND_WIDTH = 310
COLORMAP = cv2.COLORMAP_TURBO


@dataclass
class Region:
    polygon: np.ndarray
    text: str
    row: pd.Series
    poly_mask: np.ndarray
    ink_mask: np.ndarray
    edge_map: np.ndarray
    features: Mapping
    details: Mapping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/totaltext_16_text_region_features.csv"),
    )
    parser.add_argument("--annotations", type=Path, default=Path("data/totaltext/anno.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/totaltext"))
    parser.add_argument("--chargt-dir", type=Path, default=Path("data/totaltext/chargt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/feat"))
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURES,
        default=FEATURES,
        help="Subset to render (default: all).",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image stem or CSV image_name; repeat to select multiple images.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--strict-csv",
        action="store_true",
        help="Abort at the first CSV/current-extractor mismatch.",
    )
    return parser.parse_args()


def _annotation_regions(annotation_path: Path, target_names: Iterable[str]) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """Reproduce the extractor's annotation/word traversal order."""
    target_names = set(target_names)
    with annotation_path.open() as handle:
        data = json.load(handle)
    images = {image["id"]: image for image in data["images"]}
    result: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for annotation in data["annotations"]:
        info = images.get(annotation["image_id"], {})
        file_name = info.get("file_name", "")
        if file_name not in target_names:
            continue
        boxes = annotation.get("bbox", [])
        for index, word in enumerate(annotation.get("caption", "").split()):
            if index >= len(boxes) or word == "###":
                continue
            polygon = boxes[index]
            if isinstance(polygon, list) and len(polygon) >= 3:
                # Preserve the original numeric coordinates. The production
                # extractor uses floats for Shapely area/curvature and casts
                # only when constructing an OpenCV raster mask.
                result[file_name].append((word, np.asarray(polygon, dtype=np.float64)))
    return result


def _match_rows(
    frame: pd.DataFrame,
    polygon_records: Mapping[str, Sequence[Tuple[str, np.ndarray]]],
) -> Dict[str, List[Tuple[pd.Series, np.ndarray]]]:
    """Pair CSV rows and polygons, failing loudly if the source data diverged."""
    matched: Dict[str, List[Tuple[pd.Series, np.ndarray]]] = {}
    for image_name, rows in frame.groupby("image_name", sort=False):
        rows = [row for _, row in rows.iterrows()]
        polygons = list(polygon_records.get(image_name, []))
        if len(rows) != len(polygons):
            raise ValueError(
                f"{image_name}: CSV has {len(rows)} regions but annotations have "
                f"{len(polygons)} valid regions."
            )
        pairs = []
        for index, (row, (word, polygon)) in enumerate(zip(rows, polygons)):
            if str(row["text"]) != word:
                raise ValueError(
                    f"{image_name}, region {index}: CSV text {row['text']!r} "
                    f"does not match annotation text {word!r}."
                )
            pairs.append((row, polygon))
        matched[image_name] = pairs
    return matched


def _feature_limits(frame: pd.DataFrame, feature: str) -> Tuple[float, float]:
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _normalize_scalar(value: float, limits: Tuple[float, float]) -> float:
    low, high = limits
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _turbo_color(value01: float) -> Tuple[int, int, int]:
    pixel = np.asarray([[round(np.clip(value01, 0.0, 1.0) * 255)]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.applyColorMap(pixel, COLORMAP)[0, 0])


def _put_text_with_halo(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.42,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    x, y = origin
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA,
    )


def _blend_color(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    idx = mask > 0
    if not np.any(idx):
        return
    color_array = np.asarray(color, dtype=np.float32)
    image[idx] = np.clip(
        (1.0 - alpha) * image[idx].astype(np.float32) + alpha * color_array,
        0, 255,
    ).astype(np.uint8)


def _blend_heatmap(
    image: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.62,
    limits: Tuple[float, float] | None = None,
) -> None:
    idx = mask > 0
    if not np.any(idx):
        return
    selected = values[idx].astype(np.float32)
    finite = np.isfinite(selected)
    if not np.any(finite):
        return
    if limits is None:
        low, high = np.percentile(selected[finite], [2, 98])
    else:
        low, high = limits
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((values.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), COLORMAP)
    image[idx] = np.clip(
        (1.0 - alpha) * image[idx].astype(np.float32)
        + alpha * heat[idx].astype(np.float32),
        0, 255,
    ).astype(np.uint8)


def _make_regions(
    pairs: Sequence[Tuple[pd.Series, np.ndarray]],
    gray: np.ndarray,
    gt_ink: np.ndarray,
    saliency: np.ndarray,
    itti: Mapping,
    mismatches: List[Dict],
    strict_csv: bool,
) -> List[Region]:
    regions = []
    for row, polygon in pairs:
        # One call to the production extractor supplies both the scalar values
        # and the exact intermediate arrays used below.
        poly_mask = get_polygon_mask(polygon, gray.shape)
        ink_mask = cv2.bitwise_and(gt_ink, gt_ink, mask=poly_mask)
        extracted = extract_features_from_region(
            gray,
            polygon,
            ink_mask,
            sal_map=saliency,
            itti_result=itti,
            n_characters=len(str(row["text"])),
            return_details=True,
        )
        details = extracted["_details"]
        for feature, decimals in FEATURE_DECIMALS.items():
            expected = float(row[feature])
            actual = float(extracted[feature])
            tolerance = 0.5 * (10.0 ** -decimals) + 1e-9
            if not np.isclose(expected, actual, rtol=0.0, atol=tolerance):
                mismatch = {
                    "image_name": row["image_name"],
                    "text": row["text"],
                    "feature": feature,
                    "csv_value": expected,
                    "current_code_value": actual,
                    "difference": actual - expected,
                }
                mismatches.append(mismatch)
                if strict_csv:
                    raise ValueError(
                        f"{row['image_name']} / {row['text']} / {feature}: "
                        f"CSV={expected}, recomputed={actual}."
                    )
        regions.append(
            Region(
                polygon=polygon.astype(np.int32),
                text=str(row["text"]),
                row=row,
                poly_mask=details["polygon_mask"],
                ink_mask=details["ink_mask"],
                edge_map=details["edge_map"],
                features=extracted,
                details=details,
            )
        )
    return regions


def _draw_spatial_detail(
    image: np.ndarray,
    feature: str,
    region: Region,
) -> None:
    poly = region.poly_mask
    ink = region.ink_mask
    detail = region.details
    background = detail["background_mask"]

    if feature == "luminance_std":
        _blend_heatmap(image, detail["polygon_luminance_deviation"], poly)
    elif feature in {"contrast_michelson", "contrast_weber"}:
        _blend_heatmap(image, detail["text_background_difference"], poly)
        cv2.polylines(image, [region.polygon], True, (255, 255, 255), 1, cv2.LINE_AA)
    elif feature == "contrast_wcag":
        _blend_heatmap(image, detail["text_background_difference_linear"], poly)
        cv2.polylines(image, [region.polygon], True, (255, 255, 255), 1, cv2.LINE_AA)
    elif feature == "contrast_rms_ink":
        _blend_heatmap(image, detail["ink_luminance_deviation"], ink)
    elif feature == "contrast_rms_bg":
        _blend_heatmap(image, detail["background_luminance_deviation"], background)
    elif feature == "masking_index":
        _blend_heatmap(
            image, detail["background_luminance_deviation"], background, alpha=0.52,
        )
        cv2.polylines(image, [region.polygon], True, (255, 255, 255), 1, cv2.LINE_AA)
        _blend_color(image, region.edge_map, (0, 255, 255), 0.85)


def _make_feature_background(
    feature: str,
    color: np.ndarray,
    gray: np.ndarray,
    saliency: np.ndarray,
    itti: Mapping,
    full_sf_map: np.ndarray,
    full_sf_valid: np.ndarray,
    regions: Sequence[Region],
) -> np.ndarray:
    """Create the whole-image base appropriate for one feature."""
    if feature in BINARY_MAP_FEATURES:
        rendered = np.zeros_like(color)
        for region in regions:
            binary = region.ink_mask if feature == "ink_area" else region.edge_map
            rendered[binary > 0] = (255, 255, 255)
        return rendered

    rendered = color.copy()
    full_mask = np.full(gray.shape, 255, dtype=np.uint8)
    if feature == "luminance_mean":
        _blend_heatmap(rendered, gray.astype(np.float32), full_mask, limits=(0.0, 255.0))
    elif feature == "spatial_frequency":
        _blend_heatmap(rendered, full_sf_map, full_sf_valid)
    elif feature == "saliency_ink":
        _blend_heatmap(rendered, saliency, full_mask, limits=(0.0, 1.0))
    elif feature == "itti_saliency_ink":
        _blend_heatmap(rendered, itti["saliency_map"], full_mask)
    elif feature.startswith("itti_"):
        channel_name = {
            "itti_intensity_ink": "intensity",
            "itti_color_ink": "color",
            "itti_orientation_ink": "orientation",
        }[feature]
        _blend_heatmap(rendered, itti["channel_maps"][channel_name], full_mask)
    return rendered


def _draw_curvature_vertices(image: np.ndarray, region: Region) -> None:
    values = region.details["curvature_by_vertex"]
    for point, value in zip(region.polygon, values):
        if not np.isfinite(value):
            continue
        x, y = int(point[0]), int(point[1])
        color = _turbo_color(float(value) / 180.0)
        cv2.circle(image, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        _put_text_with_halo(image, f"{value:.0f}", (x + 5, y - 5), scale=0.34)


def _draw_scalar_region(
    image: np.ndarray,
    feature: str,
    region: Region,
    limits: Tuple[float, float],
    has_detail: bool,
) -> None:
    # The visualization and label always use the current production extractor
    # result. The source CSV is checked separately and only supplies a stable
    # dataset-wide color range.
    value = float(region.features[feature])
    color = _turbo_color(_normalize_scalar(value, limits))
    if feature in BINARY_MAP_FEATURES:
        return
    if not has_detail:
        _blend_color(image, region.poly_mask, color, 0.45)
    cv2.polylines(image, [region.polygon], True, color, 3, cv2.LINE_AA)


def _region_value_label(feature: str, region: Region) -> str:
    if feature == "edge_density":
        return f"{region.text[:13]}  density={region.features['edge_density']:.4f}"
    if feature == "ink_area":
        return f"{region.text[:13]}  ink={int(region.features['ink_area'])} px"
    if feature == "perimetric_complexity":
        return (
            f"{region.text[:10]}  PC={region.features['perimetric_complexity']:.3g}, "
            f"ink={int(region.features['ink_area'])} px"
        )
    return f"{region.text[:15]}  {region.features[feature]:.4g}"


def _compose_canvas(
    visualization: np.ndarray,
    feature: str,
    limits: Tuple[float, float],
    detail_note: str,
    regions: Sequence[Region],
) -> np.ndarray:
    height, width = visualization.shape[:2]
    canvas = np.full(
        (height + HEADER_HEIGHT, width + LEGEND_WIDTH, 3),
        (245, 245, 245),
        dtype=np.uint8,
    )
    canvas[HEADER_HEIGHT:, :width] = visualization
    cv2.putText(
        canvas, feature, (14, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, (25, 25, 25), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, detail_note, (14, 46), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (60, 60, 60), 1, cv2.LINE_AA,
    )

    panel_x = width + 20
    if feature in BINARY_MAP_FEATURES:
        cv2.putText(
            canvas, "region annotations", (panel_x, HEADER_HEIGHT + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA,
        )
        list_y0 = HEADER_HEIGHT + 58
        available = max(1, height + HEADER_HEIGHT - list_y0 - 12)
        row_step = min(28, max(17, available // max(1, len(regions))))
        for index, region in enumerate(regions, start=1):
            y = list_y0 + (index - 1) * row_step
            label = f"{index:02d}  {_region_value_label(feature, region)}"
            cv2.putText(
                canvas, label, (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.39, (30, 30, 30), 1, cv2.LINE_AA,
            )
        return canvas

    bar_x0, bar_x1 = panel_x, width + LEGEND_WIDTH - 20
    bar_y0, bar_y1 = HEADER_HEIGHT + 35, HEADER_HEIGHT + 53
    gradient = np.linspace(0, 255, max(1, bar_x1 - bar_x0), dtype=np.uint8)[None, :]
    gradient = cv2.applyColorMap(gradient, COLORMAP)
    canvas[bar_y0:bar_y1, bar_x0:bar_x1] = gradient
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), (40, 40, 40), 1)
    low, high = limits
    cv2.putText(
        canvas, f"{low:.4g}", (bar_x0, bar_y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA,
    )
    high_text = f"{high:.4g}"
    high_width = cv2.getTextSize(high_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(
        canvas, high_text, (bar_x1 - high_width, bar_y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "region value (outline color)", (panel_x, HEADER_HEIGHT + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA,
    )

    list_y0 = HEADER_HEIGHT + 98
    available = max(1, height + HEADER_HEIGHT - list_y0 - 12)
    row_step = min(28, max(17, available // max(1, len(regions))))
    for index, region in enumerate(regions, start=1):
        value = float(region.features[feature])
        color = _turbo_color(_normalize_scalar(value, limits))
        y = list_y0 + (index - 1) * row_step
        cv2.rectangle(canvas, (panel_x, y - 11), (panel_x + 15, y + 4), color, -1)
        cv2.rectangle(canvas, (panel_x, y - 11), (panel_x + 15, y + 4), (30, 30, 30), 1)
        label = f"{index:02d}  {_region_value_label(feature, region)}"
        cv2.putText(
            canvas, label, (panel_x + 24, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.41, (30, 30, 30), 1, cv2.LINE_AA,
        )
    return canvas


DETAIL_NOTES = {
    "polygon_size": "polygon fill + outline: region scalar",
    "ink_area": "binary full-image GT ink mask (white=ink, black=background); no source overlay",
    "edge_density": "binary full-image GT-ink edge map; side labels show region edge density",
    "luminance_std": "interior heat: |pixel luminance - region mean|",
    "luminance_mean": "whole-image pixel luminance heatmap overlaid on source image",
    "avg_curvature": "vertex value: |180 - adjacent-edge angle| in degrees (not arc-length curvature)",
    "spatial_frequency": "whole-image sqrt(horizontal_diff^2 + vertical_diff^2) overlay",
    "perimetric_complexity": "binary full-image perimeter map; labels show PC and ink area",
    "contrast_michelson": "interior heat: pixel distance from opposite-class mean",
    "contrast_wcag": "interior heat: linear-sRGB distance from opposite-class mean",
    "contrast_rms_ink": "ink heat: |ink luminance - mean ink luminance|",
    "contrast_weber": "interior heat: pixel distance from opposite-class mean",
    "contrast_rms_bg": "background heat: |background luminance - mean background|",
    "masking_index": "background texture heat + yellow ink boundary",
    "saliency_ink": "whole-image spectral-residual saliency map overlaid on source image",
    "itti_saliency_ink": "whole-image total Itti saliency map overlaid on source image",
    "itti_intensity_ink": "whole-image weighted Itti intensity-channel overlay",
    "itti_color_ink": "whole-image weighted Itti color-channel overlay",
    "itti_orientation_ink": "whole-image weighted Itti orientation-channel overlay",
}


def _write_manifest(output_dir: Path, counts: Mapping[str, int]) -> None:
    lines = [
        "# Text-region feature visualizations",
        "",
        "Each feature directory contains a source-resolution visualization.",
        "Saliency/Itti/luminance/spatial-frequency maps cover the whole image and",
        "are overlaid on the source. Ink, edge, and perimeter outputs are standalone",
        "black/white maps without a source-image overlay.",
        "Polygon outline color and the printed value come from a fresh call to the",
        "same production extractor used to generate the CSV. `csv_consistency.csv`",
        "lists any differences between the historical CSV and current code.",
        "The interior visualization shows the pixel/vertex-level computation described",
        "in the image header. The shared scalar legend is clipped to the dataset-wide",
        "2nd–98th percentiles so outliers do not flatten the color scale.",
        "",
        "| feature | images | spatial detail |",
        "|---|---:|---|",
    ]
    for feature in counts:
        lines.append(f"| `{feature}` | {counts[feature]} | {DETAIL_NOTES[feature]} |")
    lines.extend(
        [
            "",
            "Regenerate:",
            "",
            "```bash",
            "python utils/eval/visualize_text_region_features.py",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    frame = pd.read_csv(args.csv)
    missing = [feature for feature in args.features if feature not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing}")

    if args.image:
        requested = set(args.image)
        frame = frame[
            frame["image_name"].map(
                lambda name: name in requested or Path(str(name)).stem in requested
            )
        ]
        if frame.empty:
            raise ValueError(f"No CSV rows matched --image {sorted(requested)}")

    image_names = list(frame["image_name"].drop_duplicates())
    annotations = _annotation_regions(args.annotations, image_names)
    matched = _match_rows(frame, annotations)
    limits = {feature: _feature_limits(frame, feature) for feature in args.features}
    for feature in args.features:
        (args.output_dir / feature).mkdir(parents=True, exist_ok=True)

    counts = {feature: 0 for feature in args.features}
    mismatches: List[Dict] = []
    progress = tqdm(image_names, desc="Rendering source images")
    for image_name in progress:
        source_path = args.image_root / image_name
        # Use the extractor's loader: OpenCV's direct grayscale JPEG decode can
        # differ slightly from BGR decode followed by cvtColor.
        gray, color = load_image(source_path)
        if color is None or gray is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")
        stem = Path(image_name).stem

        gt_path = args.chargt_dir / f"{stem}.png"
        gt_ink = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_ink is None:
            raise FileNotFoundError(f"Could not read GT ink mask: {gt_path}")
        _, gt_ink = cv2.threshold(gt_ink, 127, 255, cv2.THRESH_BINARY)

        saliency = compute_image_saliency(color)
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        itti = compute_itti_saliency(rgb)
        full_mask = np.full(gray.shape, 255, dtype=np.uint8)
        _, full_sf_map, full_sf_valid = compute_spatial_frequency_detail(gray, full_mask)
        regions = _make_regions(
            matched[image_name],
            gray,
            gt_ink,
            saliency,
            itti,
            mismatches,
            args.strict_csv,
        )

        for feature in args.features:
            rendered = _make_feature_background(
                feature,
                color,
                gray,
                saliency,
                itti,
                full_sf_map,
                full_sf_valid,
                regions,
            )
            has_detail = feature in MAP_FEATURES
            for region in regions:
                if (
                    has_detail
                    and feature != "avg_curvature"
                    and feature not in BINARY_MAP_FEATURES
                    and feature not in FULL_IMAGE_OVERLAY_FEATURES
                ):
                    _draw_spatial_detail(rendered, feature, region)
                _draw_scalar_region(rendered, feature, region, limits[feature], has_detail)
                if feature == "avg_curvature":
                    _draw_curvature_vertices(rendered, region)

            canvas = _compose_canvas(
                rendered, feature, limits[feature], DETAIL_NOTES[feature], regions,
            )
            output_path = args.output_dir / feature / f"{stem}.jpg"
            ok = cv2.imwrite(
                str(output_path), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
            )
            if not ok:
                raise OSError(f"Failed to write {output_path}")
            counts[feature] += 1

    _write_manifest(args.output_dir, counts)
    mismatch_columns = [
        "image_name", "text", "feature", "csv_value",
        "current_code_value", "difference",
    ]
    pd.DataFrame(mismatches, columns=mismatch_columns).to_csv(
        args.output_dir / "csv_consistency.csv", index=False,
    )
    print(f"Rendered {sum(counts.values())} overlays under {args.output_dir}")
    print(f"CSV/current-code mismatches: {len(mismatches)}")
    for feature, count in counts.items():
        print(f"  {feature:28s} {count:3d} images")


if __name__ == "__main__":
    main()
