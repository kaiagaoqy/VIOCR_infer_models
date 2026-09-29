#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert "human data" into per-subject JSON files.

Input structure:
totaltext/
  Sub01/
    16_0000002.json
    16_0000003.json
  Sub02/
    ...

Each source JSON has keys: "file" (list[str]), "annotations" (list[dict]), "points" (list[list[[x, y, z], ...]])

Output:
For each subject folder (e.g., Sub01), produce one JSON file named "Sub01.json" under --output_dir,
containing a list of entries with the target schema:

{
  "image_id": <int>,
  "category_id": 1,
  "polys": [[x, y], ...],
  "rec_texts": "['RAILWAY', 'MELBOURNE']",
  "rec_score": 0.0,
  "det_score": 0.0,
  "filter": 14
}

- image_id and filter_no are derived from anno.json::images[].
- By default, rec_texts is a *stringified* Python list to match the user's example.
  Use --rec_texts_array to emit a true JSON array instead.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

def build_anno_maps(anno_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno = json.load(f)

    images = anno.get('images', [])
    id_map: Dict[str, int] = {}
    filter_map: Dict[str, int] = {}

    for item in images:
        file_name = item.get('file_name', '')
        image_id = item.get('id', None)
        filter_no = item.get('Filter_no', None)

        if file_name is None or image_id is None:
            continue

        variants = set()
        variants.add(file_name)

        if file_name.startswith('data/'):
            variants.add(file_name[len('data/'):])

        basename = os.path.basename(file_name)
        dirname = os.path.dirname(file_name)
        if basename:
            variants.add(basename)
        if dirname and basename:
            variants.add(os.path.join(dirname, basename))

        variants.add(os.path.join('data', file_name).replace('\\', '/'))

        for v in variants:
            key = v.replace('\\', '/')
            id_map[key] = image_id
            if filter_no is not None:
                filter_map[key] = filter_no

    return id_map, filter_map

def normalize_file_key(path_str: str) -> List[str]:
    p = path_str.replace('\\', '/')
    keys = [p]

    if p.startswith('data/'):
        keys.append(p[len('data/'):])

    basename = os.path.basename(p)
    dirname = os.path.dirname(p)
    if basename:
        keys.append(basename)
    if dirname and basename:
        keys.append(os.path.join(dirname, basename).replace('\\', '/'))

    if not p.startswith('data/'):
        keys.append(os.path.join('data', p).replace('\\', '/'))

    seen = set()
    ordered = []
    for k in keys:
        if k not in seen:
            ordered.append(k)
            seen.add(k)
    return ordered

def lookup_image_and_filter(file_entry: str, id_map: Dict[str, int], filter_map: Dict[str, int]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    for key in normalize_file_key(file_entry):
        if key in id_map:
            return id_map.get(key), filter_map.get(key), key
    return None, None, None

def convert_one_source(json_path: str, id_map: Dict[str, int], filter_map: Dict[str, int], category_id: int, rec_texts_array: bool) -> List[dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    files = data.get('file', [])
    annos = data.get('annotations', [])
    points = data.get('points', [])

    n = min(len(files), len(annos), len(points))

    results = []
    for i in range(n):
        file_entry = files[i]
        anno = annos[i] if i < len(annos) else {}
        pts3d = points[i] if i < len(points) else []

        image_id, filter_no, _ = lookup_image_and_filter(file_entry, id_map, filter_map)

        det_score = anno.get('detection_confidence', 0)
        rec_score = anno.get('recognition_confidence', 0)

        # polys: drop z
        # pts3d is [[points]], need to extract the points array
        polys_2d = []
        if isinstance(pts3d, list) and len(pts3d) > 0:
            # Get the first (and should be only) polygon from the nested structure
            polygon_points = pts3d[0] if isinstance(pts3d[0], list) else pts3d
            for p in polygon_points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    polys_2d.append([p[0], p[1]])

        # Recognition text: keep even if polygon is empty
        if anno.get('recognized') and anno.get('recognized_name'):
            rec_list = [str(anno.get('recognized_name'))]
        else:
            rec_list = [""]
        
        rec_texts_val = rec_list if rec_texts_array else str(rec_list)

        out = {
            "image_id": int(image_id) if image_id is not None else None,
            "category_id": int(category_id),
            "polys": polys_2d,
            "rec_texts": rec_texts_val,
            "rec_score": float(rec_score) if rec_score is not None else 0.0,
            "det_score": float(det_score) if det_score is not None else 0.0,
            "filter": int(filter_no) if filter_no is not None else None,
        }

        if image_id is None:
            out["_warn"] = {
                "reason": "image_id not found in anno.json",
                "file_entry": file_entry,
                "tried_keys": normalize_file_key(file_entry)
            }

        results.append(out)

    return results
def group_entries_by_image(entries: List[dict], rec_texts_array: bool) -> List[dict]:
    buckets: Dict[Any, dict] = {}
    for e in entries:
        key = e.get("image_id")
        if key not in buckets:
            # Initialize with lists for polys and scores; rec_texts as list in memory
            buckets[key] = {
                "image_id": key,
                "category_id": e.get("category_id", 1),
                "polys": [],
                "rec_texts": [],  # keep as list internally; stringify later if needed
                "rec_score": [],
                "det_score": [],
                "filter": e.get("filter"),
                "_warn": []
            }

        grp = buckets[key]

        # filter consistency
        f_cur = e.get("filter")
        if grp.get("filter") is None and f_cur is not None:
            grp["filter"] = f_cur
        elif f_cur is not None and grp.get("filter") is not None and f_cur != grp["filter"]:
            grp["_warn"].append({"reason": "inconsistent filter in group", "filter_values": [grp["filter"], f_cur]})

        # Always append polygon and rec_texts (even if polygon is empty)
        polys = e.get("polys", [])
        grp["polys"].append(polys if isinstance(polys, list) else [])
        grp["rec_score"].append(e.get("rec_score", 0.0))
        grp["det_score"].append(e.get("det_score", 0.0))

        rec = e.get("rec_texts", [])
        if isinstance(rec, str):
            try:
                parsed = json.loads(rec.replace("'", '"'))
                if isinstance(parsed, list):
                    rec = parsed
                else:
                    rec = [str(parsed)]
            except Exception:
                rec = [s.strip() for s in rec.strip("[]").split(",") if s.strip()]
        if isinstance(rec, list):
            grp["rec_texts"].extend([str(x) for x in rec if x is not None])

        # propagate warn
        if "_warn" in e:
            grp["_warn"].append(e["_warn"])

    # finalize
    grouped = []
    for key, grp in buckets.items():
        out = {
            "image_id": grp["image_id"],
            "category_id": grp["category_id"],
            "polys": grp["polys"],
            "rec_texts": grp["rec_texts"] if rec_texts_array else str(grp["rec_texts"]),
            "rec_score": grp["rec_score"],
            "det_score": grp["det_score"],
            "filter": grp["filter"],
        }
        if grp["_warn"]:
            out["_warn"] = grp["_warn"]
        grouped.append(out)

    # stable sort by image_id (None at end)
    grouped.sort(key=lambda x: (x["image_id"] is None, x["image_id"] if x["image_id"] is not None else 0))
    return grouped

def main():
    parser = argparse.ArgumentParser(description="Convert human data JSONs into per-subject files (Subxx.json).")
    parser.add_argument("--input_root", help="Path to 'totaltext' root folder.", default='data/human/totaltext')
    parser.add_argument("--anno", help="Path to anno.json file (with 'images' list).", default='data/totaltext/anno.json')
    parser.add_argument("--output_dir", help="Directory to write subject JSON files.", default='data/human/totaltext/converted')
    parser.add_argument("--category_id", type=int, default=1, help="Value for 'category_id'. Default: 1")
    parser.add_argument("--rec_texts_array", action="store_true", help="Emit rec_texts as a JSON array instead of a stringified list.")
    parser.add_argument("--fail_on_missing", action="store_true", help="Raise error when any entry cannot be mapped to image_id.")
    parser.add_argument("--group", action="store_true", help="Group entries by image_id within each subject.")
    args = parser.parse_args()

    id_map, filter_map = build_anno_maps(args.anno)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect subject → list of entries
    buckets: Dict[str, List[dict]] = defaultdict(list)
    missing_count = 0

    # Expect folders like totaltext/Sub01, Sub02, ...
    for subj_name in sorted(os.listdir(args.input_root)):
        if 'converted' in subj_name:
            continue
        subj_path = os.path.join(args.input_root, subj_name)
        if not os.path.isdir(subj_path):
            continue

        # Walk each subject folder only
        for root, _, files in os.walk(subj_path):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                # Skip anno.json just in case
                if name == os.path.basename(args.anno):
                    continue

                json_path = os.path.join(root, name)
                try:
                    converted = convert_one_source(
                        json_path,
                        id_map,
                        filter_map,
                        category_id=args.category_id,
                        rec_texts_array=args.rec_texts_array
                    )
                    for item in converted:
                        if item.get("image_id") is None:
                            missing_count += 1
                    buckets[subj_name].extend(converted)
                except Exception as e:
                    buckets[subj_name].append({
                        "_error": str(e),
                        "_source": json_path
                    })

    # Write one file per subject
    for subj_name, entries in buckets.items():
        if args.group:
            entries = group_entries_by_image(entries, rec_texts_array=args.rec_texts_array)

        out_path = os.path.join(args.output_dir, f"{subj_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    if args.fail_on_missing and missing_count > 0:
        raise SystemExit(f"Completed with {missing_count} entries missing image_id mapping (files still written). Rerun without --fail_on_missing to suppress error.")

    print(f"Done. Subjects written: {len(buckets)}. Missing image_id count: {missing_count}")

if __name__ == "__main__":
    main()
