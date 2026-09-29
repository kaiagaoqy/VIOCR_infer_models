"""Build updated LVTextFeatures from new CSV + polygon geometry."""

import json
import numpy as np
import pandas as pd

from .config import (
    ANNO_PATH, NEW_FEAT, IMAGE_NO_MAP, HERE,
    RESO_X_SCREEN, RESO_Y_SCREEN, MONITOR_W_CM, VIEW_DIST_CM,
)


def build_polygon_geometry(anno_path):
    with open(anno_path) as f:
        gt = json.load(f)
    img_info = {img['id']: img for img in gt['images']}

    rows = []
    for ann in gt['annotations']:
        info = img_info.get(ann['image_id'], {})
        fname = info.get('file_name', '')
        if not fname.startswith('16/'):
            continue
        caption   = ann.get('caption', '')
        bbox_list = ann.get('bbox', [])
        for i, word in enumerate(caption.split()):
            if word == '###' or i >= len(bbox_list):
                continue
            poly = bbox_list[i]
            if not poly or not isinstance(poly, list) or len(poly) < 3:
                continue
            pts = np.array(poly, dtype=float)
            xs, ys = pts[:, 0], pts[:, 1]
            rows.append({
                'image_name': fname,
                'text':       word,
                'X_Top':      ys.min(),
                'X_bottom':   ys.max(),
                'Width_Poly': xs.max() - xs.min(),
            })
    return pd.DataFrame(rows)


def build_lv_text_features():
    feat = pd.read_csv(NEW_FEAT)
    geom = build_polygon_geometry(ANNO_PATH)

    feat['text_lower'] = feat['text'].str.lower()
    geom['text_lower'] = geom['text'].str.lower()
    feat['_occ'] = feat.groupby(['image_name', 'text_lower']).cumcount()
    geom['_occ'] = geom.groupby(['image_name', 'text_lower']).cumcount()
    feat = feat.merge(
        geom[['image_name', 'text_lower', '_occ', 'X_Top', 'X_bottom', 'Width_Poly']],
        on=['image_name', 'text_lower', '_occ'], how='left',
    ).drop(columns=['text_lower', '_occ'])

    feat['Image.No']        = feat['image_name'].map(IMAGE_NO_MAP)
    feat['char_count']      = feat['text'].str.len()
    feat['word_length']     = feat['char_count']
    feat['reso_x']          = feat['W']
    feat['reso_y']          = feat['H']
    feat['reso_x_screen']   = RESO_X_SCREEN
    feat['reso_y_screen']   = RESO_Y_SCREEN

    feat['scale_ratio'] = feat.apply(
        lambda r: min(RESO_X_SCREEN / r['reso_x'], RESO_Y_SCREEN / r['reso_y']),
        axis=1,
    )
    feat['polygon_size_scale'] = feat['polygon_size'] * feat['scale_ratio'] ** 2
    feat['x_height_scale']     = (feat['X_bottom'] - feat['X_Top']) * feat['scale_ratio']
    feat['l2lspacing_scale']   = (feat['Width_Poly'] / feat['char_count']) * feat['scale_ratio']
    feat['l2lspacing_xheight_ratio'] = (
        feat['l2lspacing_scale'] / feat['x_height_scale']
    ).replace([np.inf, -np.inf], np.nan)

    half_angle_per_px = np.arctan(MONITOR_W_CM / 2 / VIEW_DIST_CM) / (RESO_X_SCREEN / 2)
    feat['x_height_deg']   = feat['x_height_scale']   * half_angle_per_px * 180 / np.pi
    feat['l2lspacing_deg'] = feat['l2lspacing_scale'] * half_angle_per_px * 180 / np.pi

    feat = feat.rename(columns={'text': 'GT.Text'})
    feat['GT.Text'] = feat['GT.Text'].str.lower()

    gt_ref = pd.read_csv(HERE / 'GT_Reference.csv')
    gt_ref['GT.Text.lower'] = gt_ref['GT Text'].str.lower()
    gt_ref['_occ'] = gt_ref.groupby(['Image.No', 'GT.Text.lower']).cumcount()
    feat['_occ2'] = feat.groupby(['Image.No', 'GT.Text']).cumcount()
    feat = feat.merge(gt_ref[['Image.No', 'GT.Text.lower', '_occ', 'Text.No']].rename(
        columns={'GT.Text.lower': 'GT.Text', '_occ': '_occ2'}),
        on=['Image.No', 'GT.Text', '_occ2'], how='left').drop(columns='_occ2')

    return feat
