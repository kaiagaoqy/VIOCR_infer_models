"""Merge results + features + screening; per-person and per-text summaries."""

import numpy as np
import pandas as pd


def build_lv_text_all(results, features, screening):
    """Build full subject x word matrix via Text.No."""
    subids = results['SubID'].unique()

    feat_rep = pd.concat(
        [features.assign(SubID=sid) for sid in subids], ignore_index=True)
    feat_rep['GT.Text'] = feat_rep['GT.Text'].str.lower()

    matched = results[results['Text.No'] > 0].copy()
    merge_cols = ['SubID', 'Text.No', 'Detection Order',
                  'Predicted Text (Raw)', 'Det.Score', 'Rec.Score',
                  'IoU', 'Word.Match.New', 'Detected']
    merge_cols = [c for c in merge_cols if c in matched.columns]

    all_data = feat_rep.merge(matched[merge_cols], on=['SubID', 'Text.No'], how='left')

    all_data['Detected'] = all_data['Detected'].fillna(False)
    all_data['Word.Match.New'] = all_data['Word.Match.New'].fillna('No')

    # One GT word may have been matched by multiple detections per subject.
    # Keep exactly one row per (SubID, Text.No): prefer Yes over No, then highest IoU.
    sort_cols = []
    all_data['_wm'] = all_data['Word.Match.New'].map({'Yes': 0, 'No': 1}).fillna(1)
    sort_cols.append('_wm')
    if 'IoU' in all_data.columns:
        all_data['_iou_neg'] = -pd.to_numeric(all_data['IoU'], errors='coerce').fillna(0)
        sort_cols.append('_iou_neg')
    all_data = (all_data.sort_values(sort_cols)
                .drop_duplicates(subset=['SubID', 'Text.No'], keep='first')
                .drop(columns=[c for c in ['_wm', '_iou_neg'] if c in all_data.columns]))

    all_data = all_data.merge(screening, on='SubID', how='left')
    return all_data


def summarise_person(all_data):
    def agg(g):
        n = len(g)
        n_det = g['Detected'].sum()
        n_rec = (g['Word.Match.New'] == 'Yes').sum()
        return pd.Series({
            'p.Det': n_det / n if n > 0 else np.nan,
            'p.Rec': n_rec / n if n > 0 else np.nan,
        })
    grp_cols = [c for c in ['SubID', 'CFL', 'VA_OU_logMAR', 'CS_OU']
                if c in all_data.columns]
    return all_data.groupby(grp_cols).apply(agg, include_groups=False).reset_index()


def summarise_text(all_data, feat_cols):
    feat_cols = [c for c in feat_cols if c in all_data.columns]
    id_cols   = ['Text.No', 'Image.No', 'GT.Text']

    feat_first = all_data[id_cols + feat_cols].drop_duplicates(subset=id_cols)

    def agg(g):
        n = g['SubID'].nunique()
        return pd.Series({
            'mean.order': g['Detection Order'].mean() if 'Detection Order' in g.columns else np.nan,
            'p.Det':      g['Detected'].sum() / n,
            'p.Rec':      (g['Word.Match.New'] == 'Yes').sum() / n,
        })
    stats = all_data.groupby(id_cols, dropna=False).apply(
        agg, include_groups=False).reset_index()
    return stats.merge(feat_first, on=id_cols, how='left')
