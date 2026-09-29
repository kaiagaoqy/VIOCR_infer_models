"""Shared paths, constants, and feature-column lists."""

from pathlib import Path

HERE       = Path(__file__).resolve().parent.parent   # R Code For Qingying/
REPO       = HERE.parent                              # repo root
ANNO_PATH  = REPO / 'data/totaltext/anno.json'
NEW_FEAT   = REPO / 'results/totaltext_16_text_region_features.csv'
SCREEN_CSV = HERE / 'LVTextScreening_Corr.csv'
RESULTS_CSV  = HERE / 'LVTextResults_0623.csv'
MODEL_DIR  = HERE / 'Model Performance'
OUT_DIR    = HERE / 'figures'
OUT_DIR.mkdir(exist_ok=True)

RESO_X_SCREEN = 1920
RESO_Y_SCREEN = 1200
MONITOR_W_CM  = 34.44
VIEW_DIST_CM  = 40.0

IMAGE_NO_MAP = {
    '16/0000002.jpg':  0,  '16/0000018.jpg': 16,  '16/0000024.jpg': 32,
    '16/0000061.jpg': 48,  '16/0000095.jpg': 64,  '16/0000099.jpg': 80,
    '16/0000114.jpg': 96,  '16/0000117.jpg':112,  '16/0000146.jpg':128,
    '16/0000166.jpg':144,  '16/0000181.jpg':160,  '16/0000184.jpg':176,
    '16/0000205.jpg':192,  '16/0000210.jpg':208,  '16/0000245.jpg':224,
    '16/0000264.jpg':240,  '16/0000268.jpg':256,  '16/0000276.jpg':272,
    '16/0000277.jpg':288,  '16/0000299.jpg':304,
}

TEXT_FEAT_COLS = [
    'x_height_deg',
    'l2lspacing_xheight_ratio',
    'edge_density',
    'perimetric_complexity',
    'itti_saliency_ink',
    'contrast_weber',
]

ALL_TEXT_FEATURES = [
    ('x_height_deg',             'x-height (°)'),
    ('l2lspacing_xheight_ratio', 'Spacing-to-x-height ratio'),
    ('edge_density',             'Edge density'),
    ('perimetric_complexity',    'Perimetric complexity'),
    ('itti_saliency_ink',        'Overall Itti saliency'),
    ('contrast_weber',           'Weber contrast'),
]

MODEL_DISPLAY_NAMES = {
    'gemini-2.0-flash': 'Gemini 2.0 Flash',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-2.5-pro':   'Gemini 2.5 Pro',
    'gpt-4o':           'GPT 4o',
    'gpt-4o-mini':      'GPT 4o mini',
}

MODEL_COLORS = {
    'gemini-2.5-pro':   '#1a6b3c',
    'gemini-2.5-flash': '#2ca05a',
    'gemini-2.0-flash': '#7ecfa0',
    'gpt-4o':           '#c0392b',
    'gpt-4o-mini':      '#e8877e',
}
HUMAN_COLOR = 'steelblue'
