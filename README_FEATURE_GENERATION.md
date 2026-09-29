# Feature generation and data layout

Run commands from the repository root. The human response conversion and image feature extraction are independent inputs to the later evaluation and analysis. The commands below use the current low vision cohort and the 20 TotalText images under `16/`.

## What each script does

| Stage | Script | Input | Output |
| --- | --- | --- | --- |
| Human response conversion | `utils/convert_human_data_by_subject.py` | Per-subject JSON files in `data/human/lowviz/Sub*/` and `data/totaltext/anno.json` | `data/human/lowviz/converted/Sub*.json` |
| Image text-region features | `utils/eval/extract_text_region_features.py` | `data/totaltext/anno.json`, images in `data/totaltext/16/`, character masks in `data/totaltext/chargt/` | `results/totaltext_16_text_region_features.csv` and `results/saliency/*.png` |
| Detection inspection | `utils/eval/visualize_detections_and_gt.py` | Converted human JSON, annotation JSON, images | `results/visualizations/*.png` |
| Human detection and recognition evaluation | `utils/eval/totaltext_eval_FINAL.py` | Converted human JSON, annotation JSON, images | `results/excel_outputs/*_evaluation.xlsx` |
| Analysis-ready text features | `R Code For Qingying/scripts/features.py` (`build_lv_text_features`) | Feature CSV, annotation JSON, `R Code For Qingying/GT_Reference.csv` | DataFrame with scaled size, angular size, spacing and word identifiers; `run_analysis.py` consumes it |

The first two rows generate the core input tables. The visualization and evaluation scripts inspect or score human detections; they do **not** calculate the image feature CSV. The analysis feature builder adds further variables to that CSV, so the four commands alone do not cover every feature used in the analysis. The Itti saliency implementation called by the image extractor is in `utils/eval/itti_saliency.py`.

## Commands

```bash
python utils/convert_human_data_by_subject.py \
  --input_root data/human/lowviz \
  --output_dir data/human/lowviz/converted \
  --rec_texts_array --group --fail_on_missing

python utils/eval/extract_text_region_features.py
python utils/eval/visualize_detections_and_gt.py
python utils/eval/totaltext_eval_FINAL.py

PYTHONPATH="R Code For Qingying" python -c 'from scripts.features import build_lv_text_features; build_lv_text_features().to_csv("results/LVTextFeatures_regenerated.csv", index=False)'
```

`--rec_texts_array` emits JSON arrays, and `--group` combines entries by `image_id` within each subject. `--fail_on_missing` makes unresolved image IDs visible after files are written; inspect any `_warn` entries before using the output. The converter uses `data/totaltext/anno.json` by default even when the input root is `lowviz`. The other three scripts have their input and output paths set in the source. The feature extractor only processes annotation image names beginning with `16/` and skips `###`, invalid polygons, and words whose character mask is missing. The final command exports the additional analysis-ready text features; it needs `GT_Reference.csv` in the stated location.

The feature extractor needs `numpy`, `opencv-python`, `pandas`, `shapely`, and `tqdm`. The inspection script also needs `matplotlib` and `Pillow`; the evaluator also needs `python-Levenshtein`, `openpyxl`, `shapely`, `pandas`, and `Pillow`. Install these in a suitable Python environment before running the commands.

## Feature inventory

The image extractor writes one row per GT word with `image_name`, image height `H`, width `W`, and `text`, followed by 19 numeric features:

| Group | Columns |
| --- | --- |
| Geometry and ink | `polygon_size`, `ink_area`, `avg_curvature`, `edge_density`, `perimetric_complexity` |
| Luminance and spatial detail | `luminance_std`, `luminance_mean`, `spatial_frequency` |
| Contrast and masking | `contrast_michelson`, `contrast_wcag`, `contrast_rms_ink`, `contrast_weber`, `contrast_rms_bg`, `masking_index` |
| Saliency | `saliency_ink`, `itti_saliency_ink`, `itti_intensity_ink`, `itti_color_ink`, `itti_orientation_ink` |

`R Code For Qingying/scripts/features.py` reads the CSV and annotation polygons to calculate `char_count`/`word_length`, `polygon_size_scale`, `x_height_scale`, `l2lspacing_scale`, `l2lspacing_xheight_ratio`, `x_height_deg`, and `l2lspacing_deg`. It uses screen dimensions and viewing distance from `R Code For Qingying/scripts/config.py`, and maps words to `Text.No` using `GT_Reference.csv`. Its `x_height` value is derived from the polygon's vertical extent. The analysis also reads person-level visual acuity and contrast sensitivity (`VA_OU_logMAR`, `CS_OU`) from `LVTextScreening_Corr.csv`; these are measured participant data, not image-extracted features. `R Code For Qingying/scripts/merge.py` joins them with text features and human evaluation results.

Optional visual checks use `utils/eval/visualize_text_region_features.py` (writes `results/feat/`) and `utils/eval/generate_ink_masks.py` / `generate_edge_maps.py` (write `results/inkarea/` and `results/edgemap/`). The extractor computes its ink and edge maps directly from `chargt`; those generated PNG folders are **not** prerequisites for the feature CSV.

## Data to transfer

Preserve these relative paths when placing an archive in another cloud drive. The source images and participant data are not included in this code branch.

| Path | Why it is needed |
| --- | --- |
| `data/totaltext/anno.json` | Image IDs, filter numbers, GT text and polygons for all stages |
| `data/totaltext/16/` | The 20 images used for the feature CSV and the current analysis |
| `data/totaltext/chargt/` | Per-image binary character masks needed for ink, edge, contrast and saliency features |
| `data/human/lowviz/Sub*/` | Raw low vision subject JSON, required to rerun conversion |
| `R Code For Qingying/GT_Reference.csv` | Word identifiers for `build_lv_text_features` |
| `R Code For Qingying/LVTextScreening_Corr.csv` | Participant vision variables for downstream analysis |
| `R Code For Qingying/LVTextResults_0623.csv` | Curated human outcome table used by `run_analysis.py` |

Transfer `data/human/lowviz/converted/` if the recipient should run visualization or evaluation without reconverting. Transfer `results/totaltext_16_text_region_features.csv` if they should run analysis without extracting features again. The broader `data/totaltext/` tree contains other image conditions; retain it when evaluation or visualization must cover those image IDs too. `R Code For Qingying/Model Performance/` is needed only for the separate model-performance portion of `run_analysis.py`.

Before sharing participant data, choose an access level appropriate for that dataset. Once the files are uploaded, add the archive URL and any access instructions here:

**Data download:** _link to be supplied_

## Reproduction checks

The current local snapshot contains 30 low vision subject folders, 30 converted subject JSON files, 20 images in `data/totaltext/16/`, and 20 character masks. Its existing feature CSV contains 111 word rows and 23 columns (four identifiers plus 19 numeric features). These counts describe this snapshot, not a fixed requirement of the scripts.
