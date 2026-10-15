#!/usr/bin/env python3
# [MODULE DOC] CLI that inspects a Hugging Face Hub dataset through the public Datasets Server
# REST API (no download) and reports whether it fits image classification, object detection
# or SAM/SAM2 segmentation training with the sibling training scripts in this folder.
# Behavior: parses CLI args, fetches the split list and a few sample rows over HTTPS, derives
# column names and feature types, runs three compatibility checkers, then prints either a JSON
# report or a human-readable report (sections per task, optional preprocessing snippet for
# object detection, and a summary). The leading comment block after this one is PEP 723 inline
# script metadata (Python 3.10 or newer, no third-party dependencies) used by "uv run".
# CLI arguments: --dataset (required, Hub dataset id); --split (default "train"); --config
# (default "default"); --preview (default 150, max characters per previewed field); --samples
# (default 5, number of rows fetched); --json-output (flag, print JSON instead of text).
# Env vars: none read. No authentication is sent, so only public datasets can be inspected.
# Constants: none at module level; the API base URL is hard-coded inside the request helpers.
# Top-level executed code: when run as a script, calls main; a keyboard interrupt exits with
# code 0 and any other uncaught exception prints "ERROR: ..." to stderr and exits with code 1.
# Side effects: network requests to datasets-server.huggingface.co; output to stdout.
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Dataset Format Inspector for Vision Model Training

Inspects Hugging Face datasets to determine compatibility with object detection
and image classification training.
Uses Datasets Server API for instant results - no dataset download needed!

ULTRA-EFFICIENT: Uses HF Datasets Server API - completes in <2 seconds.

Usage with HF Jobs:
    hf_jobs("uv", {
        "script": "path/to/dataset_inspector.py",
        "script_args": ["--dataset", "your/dataset", "--split", "train"]
    })
"""

import argparse
import math
import sys
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Tuple


def parse_args():
    # [FUNCTION DOC] HELPER: Define and parse the command-line arguments of the inspector.
    # Behavior: builds an argument parser with the options listed in the module block and parses
    # the process arguments.
    # Inputs: none (reads the process argument list).
    # Outputs: a namespace with attributes dataset, split, config, preview, samples, json_output.
    # Exceptions: the parser prints usage and exits with code 2 on invalid or missing arguments.
    # Side effects: reads the process argument list; may write usage text to stderr.
    parser = argparse.ArgumentParser(description="Inspect dataset format for vision model training")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--config", type=str, default="default", help="Dataset config name (default: default)")
    parser.add_argument("--preview", type=int, default=150, help="Max chars per field preview")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to fetch (default: 5)")
    parser.add_argument("--json-output", action="store_true", help="Output as JSON")
    return parser.parse_args()


def api_request(url: str) -> Dict:
    # [FUNCTION DOC] HELPER: Perform an HTTP GET against the Datasets Server and decode JSON.
    # Behavior: opens the URL with a 10 second timeout, reads the body, decodes it as text and parses
    # it as JSON. An HTTP 404 yields "nothing"; any other HTTP error or failure is re-raised as a
    # generic exception with an "API request failed: ..." message.
    # Inputs: url (string) - full request URL.
    # Outputs: the parsed JSON object (normally a mapping), or nothing (None) on HTTP 404.
    # Exceptions: raises a generic Exception for non-404 HTTP errors, network errors, timeouts and
    # invalid JSON.
    # Side effects: one outbound network request.
    # KNOWN ISSUE: the generic Exception raised for a non-404 HTTP error is itself caught by the
    # second catch-all handler and re-wrapped, so the message reads "API request failed: API request
    # failed: <code> <reason>".
    # KNOWN ISSUE: the return annotation promises a mapping but None is returned on 404.
    """Make API request to Datasets Server"""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise Exception(f"API request failed: {e.code} {e.reason}")
    except Exception as e:
        raise Exception(f"API request failed: {str(e)}")


def get_splits(dataset: str) -> Dict:
    # [FUNCTION DOC] HELPER: Fetch the list of configs/splits of a dataset from Datasets Server.
    # Behavior: builds the "/splits" endpoint URL with the URL-quoted dataset id and delegates to the
    # request helper.
    # Inputs: dataset (string) - Hub dataset id such as "owner/name".
    # Outputs: the parsed JSON response (expected to contain a "splits" list of entries with config,
    # split and optionally num_examples), or nothing if the dataset is not found (404).
    # Exceptions: propagates the request helper's exceptions.
    # Side effects: one network request.
    """Get available splits for dataset"""
    url = f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(dataset)}"
    return api_request(url)


def get_rows(dataset: str, config: str, split: str, offset: int = 0, length: int = 5) -> Dict:
    # [FUNCTION DOC] HELPER: Fetch a page of rows of one config/split from Datasets Server.
    # Behavior: builds the "/rows" endpoint URL with dataset, config, split, offset and length query
    # parameters and delegates to the request helper.
    # Inputs: dataset (string) - Hub dataset id; config (string) - config name; split (string) -
    # split name; offset (integer, default 0) - first row index; length (integer, default 5) -
    # number of rows to return.
    # Outputs: the parsed JSON response (expected keys "rows", each item holding a "row" mapping,
    # and "features"), or nothing on 404.
    # Exceptions: propagates the request helper's exceptions.
    # Side effects: one network request.
    # KNOWN ISSUE: only the dataset id is URL-quoted; config and split names containing special
    # characters (spaces, ampersands, etc.) produce a malformed query string.
    """Get rows from dataset"""
    url = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset)}&config={config}&split={split}&offset={offset}&length={length}"
    return api_request(url)


def find_columns(columns: List[str], patterns: List[str]) -> List[str]:
    # [FUNCTION DOC] HELPER: Select column names that contain any of the given substrings.
    # Behavior: for each column, lowercases its name and keeps it if any pattern occurs as a
    # substring; preserves the original order and casing of the columns.
    # Inputs: columns (list of strings) - column names; patterns (list of lowercase strings) -
    # substrings to look for.
    # Outputs: list of matching column names (may be empty).
    # Exceptions: none.
    # Side effects: none.
    # KNOWN ISSUE: plain substring matching yields false positives (for example "image_id" counts as
    # an image column, and the pattern "ann" used by the detection checker matches "channel").
    """Find columns matching patterns"""
    return [c for c in columns if any(p in c.lower() for p in patterns)]


def detect_bbox_format(bbox: List[float], image_size: Tuple[int, int] = None) -> str:
    # [FUNCTION DOC] HELPER: Guess the coordinate convention of a single bounding box.
    # Behavior: returns "unknown (not 4 values)" unless there are exactly four numbers a, b, c, d.
    # Marks the box as normalized when all four values lie between 0 and 1. If c is smaller than a
    # or d smaller than b the box cannot be corner-based, so it returns "xywh_normalized" (when
    # normalized) or "xywh (COCO style)". Otherwise, when an image size is given, it checks which
    # interpretation overshoots the image by more than 5 percent: if only the width/height reading
    # overshoots it returns "xyxy (Pascal VOC style)", if only the corner reading overshoots it
    # returns "xywh (COCO style)". Falling through, it returns "xyxy_normalized" or
    # "xyxy (Pascal VOC style)".
    # Inputs: bbox (list of numbers) - candidate box; image_size (pair of integers width, height,
    # default none) - optional image dimensions used to disambiguate.
    # Outputs: a descriptive format string as listed above.
    # Exceptions: none for well-formed numeric input; non-numeric values would raise a type error.
    # Side effects: none.
    # KNOWN ISSUE: the docstring lists the center-based YOLO format (cxcywh) but the code never
    # returns it; center-based boxes are reported as xyxy or xywh.
    """
    Detect bounding box format based on values and optionally image dimensions.
    Common formats:
    - [x_min, y_min, x_max, y_max] - XYXY (Pascal VOC)
    - [x_min, y_min, width, height] - XYWH (COCO)
    - [x_center, y_center, width, height] - CXCYWH (YOLO normalized)
    """
    if len(bbox) != 4:
        return "unknown (not 4 values)"

    a, b, c, d = bbox

    is_normalized = all(0 <= v <= 1 for v in bbox)

    if c < a or d < b:
        if is_normalized:
            return "xywh_normalized"
        return "xywh (COCO style)"

    # c > a and d > b — ambiguous between xyxy and xywh.
    # Use image dimensions to disambiguate when available.
    if image_size is not None:
        img_w, img_h = image_size
        # If interpreting as xywh, right edge = a + c; if that overshoots the
        # image while c alone fits, the format is more likely xyxy.
        xywh_exceeds = (a + c > img_w * 1.05) or (b + d > img_h * 1.05)
        xyxy_exceeds = (c > img_w * 1.05) or (d > img_h * 1.05)
        if xywh_exceeds and not xyxy_exceeds:
            return "xyxy (Pascal VOC style)"
        if xyxy_exceeds and not xywh_exceeds:
            return "xywh (COCO style)"

    if is_normalized:
        return "xyxy_normalized"
    return "xyxy (Pascal VOC style)"


def _extract_image_size(row: Dict) -> Tuple[int, int] | None:
    # [FUNCTION DOC] HELPER: Read image width and height from a Datasets Server row, if present.
    # Behavior: looks at the columns "image", "img", "picture", "photo" in that order; for the first
    # one whose value is a mapping holding numeric "width" and "height", returns them as integers.
    # Inputs: row (mapping) - the "row" payload of one Datasets Server row.
    # Outputs: a (width, height) pair of integers, or nothing if no such column/metadata exists.
    # Exceptions: none.
    # Side effects: none.
    """Try to extract (width, height) from the image column returned by Datasets Server."""
    for col in ("image", "img", "picture", "photo"):
        img = row.get(col)
        if isinstance(img, dict):
            w = img.get("width")
            h = img.get("height")
            if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                return (int(w), int(h))
    return None


def analyze_annotations(sample_rows: List[Dict], annotation_cols: List[str]) -> Dict[str, Any]:
    # [FUNCTION DOC] Summarize the structure of the object-detection annotation column in samples.
    # Behavior: if no annotation columns are given returns a mapping with found false. Otherwise uses
    # only the first annotation column and, for every sample row whose value is non-empty:
    #   - if the value is a mapping (Hub "objects" style), records its keys; looks for a "bbox" or
    #     "bboxes" key; if it holds a list of boxes counts them and detects the format of the first
    #     one, if it holds a single box counts one object; then collects category values from any of
    #     the keys category, categories, label, labels, class, classes, category_id.
    #   - if the value is a list of per-object mappings, records its length/item keys, counts the
    #     objects, detects the format of the first object's "bbox" and collects category values
    #     from the first of category/label/class/category_id keys present on the first item.
    # Image size from the row is used to disambiguate box formats. Afterwards computes the average
    # objects per valid sample (rounded to 2 decimals), resets an untouched minimum to 0, de-duplicates
    # categories (order not preserved), counts classes and picks the most common box format.
    # Inputs: sample_rows (list of Datasets Server row entries, each with a "row" mapping);
    # annotation_cols (list of strings) - candidate annotation columns, first one is used.
    # Outputs: a mapping with keys found, column, sample_structures, bbox_formats, categories_found,
    # avg_objects_per_image, max_objects, min_objects, num_classes and, when any box was seen,
    # primary_bbox_format.
    # Exceptions: none deliberately; malformed values may raise type errors.
    # Side effects: none.
    # KNOWN ISSUE: for list-style annotations, if several category keys are present on an item the
    # categories of that object are collected once per matching key, and only keys of the first item
    # are consulted.
    """Analyze annotation structure from sample rows"""
    if not annotation_cols:
        return {"found": False}

    annotation_col = annotation_cols[0]
    annotations_info = {
        "found": True,
        "column": annotation_col,
        "sample_structures": [],
        "bbox_formats": [],
        "categories_found": [],
        "avg_objects_per_image": 0,
        "max_objects": 0,
        "min_objects": float('inf'),
    }

    total_objects = 0
    valid_samples = 0

    for row in sample_rows:
        ann = row["row"].get(annotation_col)
        if not ann:
            continue

        valid_samples += 1
        image_size = _extract_image_size(row["row"])

        # Check if it's a list of annotations or a dict
        if isinstance(ann, dict):
            # COCO-style or structured annotation
            sample_structure = {
                "type": "dict",
                "keys": list(ann.keys())
            }

            # Check for bounding boxes
            if "bbox" in ann or "bboxes" in ann:
                bbox_key = "bbox" if "bbox" in ann else "bboxes"
                bboxes = ann[bbox_key]
                if isinstance(bboxes, list) and len(bboxes) > 0:
                    if isinstance(bboxes[0], list):
                        # Multiple bboxes
                        num_objects = len(bboxes)
                        total_objects += num_objects
                        annotations_info["max_objects"] = max(annotations_info["max_objects"], num_objects)
                        annotations_info["min_objects"] = min(annotations_info["min_objects"], num_objects)

                        # Analyze first bbox format
                        bbox_format = detect_bbox_format(bboxes[0], image_size)
                        annotations_info["bbox_formats"].append(bbox_format)
                    else:
                        # Single bbox
                        total_objects += 1
                        annotations_info["max_objects"] = max(annotations_info["max_objects"], 1)
                        annotations_info["min_objects"] = min(annotations_info["min_objects"], 1)
                        bbox_format = detect_bbox_format(bboxes, image_size)
                        annotations_info["bbox_formats"].append(bbox_format)

            # Check for categories/classes
            for key in ["category", "categories", "label", "labels", "class", "classes", "category_id"]:
                if key in ann:
                    cats = ann[key]
                    if isinstance(cats, list):
                        annotations_info["categories_found"].extend([str(c) for c in cats])
                    else:
                        annotations_info["categories_found"].append(str(cats))

            annotations_info["sample_structures"].append(sample_structure)

        elif isinstance(ann, list):
            # List of annotation dicts
            sample_structure = {
                "type": "list",
                "length": len(ann),
                "item_type": type(ann[0]).__name__ if ann else None
            }

            if ann and isinstance(ann[0], dict):
                sample_structure["item_keys"] = list(ann[0].keys())

                # Count objects
                num_objects = len(ann)
                total_objects += num_objects
                annotations_info["max_objects"] = max(annotations_info["max_objects"], num_objects)
                annotations_info["min_objects"] = min(annotations_info["min_objects"], num_objects)

                # Check first annotation
                first_ann = ann[0]
                if "bbox" in first_ann:
                    bbox_format = detect_bbox_format(first_ann["bbox"], image_size)
                    annotations_info["bbox_formats"].append(bbox_format)

                # Check for categories
                for key in ["category", "label", "class", "category_id"]:
                    if key in first_ann:
                        for item in ann:
                            if key in item:
                                annotations_info["categories_found"].append(str(item[key]))

            annotations_info["sample_structures"].append(sample_structure)

    if valid_samples > 0:
        annotations_info["avg_objects_per_image"] = round(total_objects / valid_samples, 2)

    if annotations_info["min_objects"] == float('inf'):
        annotations_info["min_objects"] = 0

    # Get unique categories
    annotations_info["categories_found"] = list(set(annotations_info["categories_found"]))
    annotations_info["num_classes"] = len(annotations_info["categories_found"])

    # Get most common bbox format
    if annotations_info["bbox_formats"]:
        from collections import Counter
        format_counts = Counter(annotations_info["bbox_formats"])
        annotations_info["primary_bbox_format"] = format_counts.most_common(1)[0][0]

    return annotations_info


def check_image_classification_compatibility(columns: List[str], sample_rows: List[Dict], features: List[Dict]) -> Dict[str, Any]:
    # [FUNCTION DOC] Check whether a dataset looks usable for image classification training.
    # Behavior: finds image columns (names containing image/img/picture/photo) and label columns
    # (names containing label/labels/class/fine_label/coarse_label). If a label column exists, uses
    # the first one and inspects its feature type: a ClassLabel yields type, class count and up to 20
    # class names (with a truncated flag); integer or string dtypes record type "int" or "string".
    # If no class count was obtained, collects the unique non-missing label values in the sample rows
    # (up to 20 shown, sorted by string form) and their count. Ready means both an image and a label
    # column were found.
    # Inputs: columns (list of strings); sample_rows (list of row entries with a "row" mapping);
    # features (list of feature descriptors with "name" and "type" from Datasets Server).
    # Outputs: a mapping with ready, has_image, image_columns, has_label, label_columns, label_info.
    # Exceptions: none deliberately; unhashable label values would raise a type error.
    # Side effects: none.
    """Check image classification dataset compatibility"""

    image_cols = find_columns(columns, ["image", "img", "picture", "photo"])
    has_image = len(image_cols) > 0

    label_cols = find_columns(columns, ["label", "labels", "class", "fine_label", "coarse_label"])
    has_label = len(label_cols) > 0

    label_info: Dict[str, Any] = {"found": has_label}

    if has_label:
        label_col = label_cols[0]
        label_info["column"] = label_col

        # Detect whether label is ClassLabel (int with names) or plain int/string
        for f in features:
            if f.get("name") == label_col:
                ftype = f.get("type", "")
                if isinstance(ftype, dict) and ftype.get("_type") == "ClassLabel":
                    label_info["type"] = "ClassLabel"
                    names = ftype.get("names", [])
                    label_info["num_classes"] = len(names)
                    label_info["class_names"] = names[:20]
                    if len(names) > 20:
                        label_info["class_names_truncated"] = True
                elif isinstance(ftype, dict) and ftype.get("dtype") in ("int64", "int32", "int8"):
                    label_info["type"] = "int"
                elif isinstance(ftype, dict) and ftype.get("dtype") == "string":
                    label_info["type"] = "string"
                break

        # Discover unique labels from samples if ClassLabel info wasn't in features
        if "num_classes" not in label_info:
            unique = set()
            for row in sample_rows:
                val = row["row"].get(label_col)
                if val is not None:
                    unique.add(val)
            label_info["sample_unique_labels"] = sorted(unique, key=str)[:20]
            label_info["sample_unique_count"] = len(unique)

    ready = has_image and has_label
    return {
        "ready": ready,
        "has_image": has_image,
        "image_columns": image_cols,
        "has_label": has_label,
        "label_columns": label_cols,
        "label_info": label_info,
    }


def check_object_detection_compatibility(columns: List[str], sample_rows: List[Dict]) -> Dict[str, Any]:
    # [FUNCTION DOC] Check whether a dataset looks usable for object detection training.
    # Behavior: finds image columns, annotation columns (names containing objects, annotations, ann,
    # bbox, bboxes, detection), analyzes the annotation column, and also looks for separate box
    # columns (bbox/bboxes/boxes) and category columns (category/label/class and plurals). Ready
    # means an image column exists and either an annotation column or both a separate box column and
    # a separate category column exist.
    # Inputs: columns (list of strings); sample_rows (list of row entries with a "row" mapping).
    # Outputs: a mapping with ready, has_image, image_columns, has_annotations, annotation_columns,
    # separate_bbox_columns, separate_category_columns, annotations_info.
    # Exceptions: none deliberately.
    # Side effects: none.
    # KNOWN ISSUE: readiness only requires a column whose name matches an annotation pattern; it does
    # not require that boxes or categories were actually found inside it.
    """Check object detection dataset compatibility"""

    # Find image column
    image_cols = find_columns(columns, ["image", "img", "picture", "photo"])
    has_image = len(image_cols) > 0

    # Find annotation columns
    annotation_cols = find_columns(columns, ["objects", "annotations", "ann", "bbox", "bboxes", "detection"])
    has_annotations = len(annotation_cols) > 0

    # Analyze annotations
    annotations_info = analyze_annotations(sample_rows, annotation_cols) if has_annotations else {"found": False}

    # Check for separate bbox and category columns
    bbox_cols = find_columns(columns, ["bbox", "bboxes", "boxes"])
    category_cols = find_columns(columns, ["category", "label", "class", "categories", "labels", "classes"])

    # Determine readiness
    ready = has_image and (has_annotations or (len(bbox_cols) > 0 and len(category_cols) > 0))

    return {
        "ready": ready,
        "has_image": has_image,
        "image_columns": image_cols,
        "has_annotations": has_annotations,
        "annotation_columns": annotation_cols,
        "separate_bbox_columns": bbox_cols,
        "separate_category_columns": category_cols,
        "annotations_info": annotations_info,
    }


def check_sam_segmentation_compatibility(columns: List[str], sample_rows: List[Dict], features: List[Dict]) -> Dict[str, Any]:
    # [FUNCTION DOC] Check whether a dataset looks usable for SAM/SAM2 prompted segmentation.
    # Behavior: finds image columns, mask columns (names containing mask/segmentation/alpha/matte),
    # prompt columns (names containing "prompt"), and columns named exactly bbox/bboxes/box/boxes or
    # point/points/input_point/input_points. First tries the first prompt column: for each sample,
    # parses it as JSON when it is a string; the first mapping with a bbox/box key marks a bbox
    # prompt (and validates that box against the row's image size), one with point/points marks a
    # point prompt. If none was found falls back to a dedicated box column (validating the first
    # non-missing box), then to a dedicated point column. Ready means image, mask and prompt found.
    # Inputs: columns (list of strings); sample_rows (list of row entries with a "row" mapping);
    # features (list of feature descriptors) - accepted but unused.
    # Outputs: a mapping with ready, has_image, image_columns, has_mask, mask_columns,
    # prompt_columns, bbox_columns, point_columns and prompt_info (has_prompt, prompt_type, source,
    # bbox_valid diagnostics or nothing).
    # Exceptions: none deliberately; invalid JSON prompts are skipped.
    # Side effects: none.
    """Check SAM/SAM2 segmentation dataset compatibility.

    A valid SAM segmentation dataset needs:
    - An image column
    - A mask column (binary ground-truth segmentation mask)
    - A prompt: either a bbox prompt or point prompt (in a JSON prompt column, or dedicated columns)
    """

    image_cols = find_columns(columns, ["image", "img", "picture", "photo"])
    has_image = len(image_cols) > 0

    mask_cols = find_columns(columns, ["mask", "segmentation", "alpha", "matte"])
    has_mask = len(mask_cols) > 0

    prompt_cols = find_columns(columns, ["prompt"])
    bbox_cols = [c for c in columns if c in ("bbox", "bboxes", "box", "boxes")]
    point_cols = [c for c in columns if c in ("point", "points", "input_point", "input_points")]

    prompt_info: Dict[str, Any] = {
        "has_prompt": False,
        "prompt_type": None,
        "source": None,
        "bbox_valid": None,
    }

    # Try JSON prompt column first
    if prompt_cols:
        for row in sample_rows:
            raw = row["row"].get(prompt_cols[0])
            if raw is None:
                continue
            parsed = raw if isinstance(raw, dict) else _try_json(raw)
            if parsed is None:
                continue

            if isinstance(parsed, dict):
                if "bbox" in parsed or "box" in parsed:
                    prompt_info["has_prompt"] = True
                    prompt_info["prompt_type"] = "bbox"
                    prompt_info["source"] = f"JSON column '{prompt_cols[0]}'"
                    bbox = parsed.get("bbox") or parsed.get("box")
                    prompt_info["bbox_valid"] = _validate_bbox(bbox, _extract_image_size(row["row"]))
                    break
                elif "point" in parsed or "points" in parsed:
                    prompt_info["has_prompt"] = True
                    prompt_info["prompt_type"] = "point"
                    prompt_info["source"] = f"JSON column '{prompt_cols[0]}'"
                    break

    if not prompt_info["has_prompt"] and bbox_cols:
        prompt_info["has_prompt"] = True
        prompt_info["prompt_type"] = "bbox"
        prompt_info["source"] = f"column '{bbox_cols[0]}'"
        for row in sample_rows:
            bbox = row["row"].get(bbox_cols[0])
            if bbox is not None:
                prompt_info["bbox_valid"] = _validate_bbox(bbox, _extract_image_size(row["row"]))
                break

    if not prompt_info["has_prompt"] and point_cols:
        prompt_info["has_prompt"] = True
        prompt_info["prompt_type"] = "point"
        prompt_info["source"] = f"column '{point_cols[0]}'"

    ready = has_image and has_mask and prompt_info["has_prompt"]

    return {
        "ready": ready,
        "has_image": has_image,
        "image_columns": image_cols,
        "has_mask": has_mask,
        "mask_columns": mask_cols,
        "prompt_columns": prompt_cols,
        "bbox_columns": bbox_cols,
        "point_columns": point_cols,
        "prompt_info": prompt_info,
    }


def _try_json(value) -> Any:
    # [FUNCTION DOC] HELPER: Parse a string as JSON, returning nothing on failure.
    # Behavior: returns nothing for non-string input; otherwise attempts JSON parsing.
    # Inputs: value (any) - candidate JSON text.
    # Outputs: the parsed value, or nothing if the input is not a string or is not valid JSON.
    # Exceptions: JSON decode and type errors are caught and swallowed.
    # Side effects: none.
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _validate_bbox(bbox, image_size=None) -> Dict[str, Any]:
    # [FUNCTION DOC] HELPER: Validate one bounding box and return diagnostic information.
    # Behavior: rejects non-sequences, sequences not of length 4, non-numeric values and non-finite
    # values, each with an "error" message. Treating the values as x0, y0, x1, y1: if the box would
    # have non-positive extent but the 3rd and 4th values are positive, adds format_hint
    # "likely xywh"; if those are not positive, reports a degenerate-area error; otherwise adds
    # format_hint "likely xyxy". With an image size, warns when any value exceeds 1.5 times the
    # larger image dimension. Successful boxes get valid true and the float values.
    # Inputs: bbox (any) - candidate box; image_size (pair width, height, default none).
    # Outputs: a mapping with valid (bool) plus error, or format_hint/values and optional warning.
    # Exceptions: type and value errors from numeric conversion are caught and reported as errors.
    # Side effects: none.
    """Validate a single bounding box and return diagnostics."""
    result: Dict[str, Any] = {"valid": False}
    if not isinstance(bbox, (list, tuple)):
        result["error"] = "bbox is not a list"
        return result
    if len(bbox) != 4:
        result["error"] = f"expected 4 values, got {len(bbox)}"
        return result
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        result["error"] = "non-numeric values"
        return result

    if not all(math.isfinite(v) for v in vals):
        result["error"] = "contains non-finite values"
        return result

    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        if vals[2] > 0 and vals[3] > 0:
            result["format_hint"] = "likely xywh"
        else:
            result["error"] = "degenerate bbox (zero or negative area)"
            return result
    else:
        result["format_hint"] = "likely xyxy"

    if image_size is not None:
        img_w, img_h = image_size
        if any(v > max(img_w, img_h) * 1.5 for v in vals):
            result["warning"] = "coordinates exceed image bounds"

    result["valid"] = True
    result["values"] = vals
    return result


def generate_mapping_code(info: Dict[str, Any]) -> str:
    # [FUNCTION DOC] Produce a Python preprocessing snippet (as text) for object detection datasets.
    # Behavior: if the detection result is ready and annotations were analyzed: when the primary box
    # format mentions coco or xywh, returns a comment-only note that the data can be used directly;
    # when it mentions xyxy, returns a snippet defining a map function that converts corner boxes to
    # x/y/width/height for list- or mapping-style annotations. If the result is not ready and both
    # separate box and category columns exist, returns a snippet that zips them into an "objects"
    # list column. In all other cases returns nothing.
    # Inputs: info (mapping) - result of the object detection compatibility check.
    # Outputs: snippet text, or nothing.
    # Exceptions: none deliberately.
    # Side effects: none.
    # KNOWN ISSUE: "xywh_normalized" contains "xywh", so normalized boxes are reported as ready-to-use
    # COCO data although COCO expects absolute pixel coordinates.
    # KNOWN ISSUE: the "not ready" branch is effectively reachable only when there is no image column
    # (separate box and category columns plus an image already make the dataset ready), and the ready
    # branch never emits the combine snippet when only separate columns exist.
    """Generate mapping code if needed"""
    if info["ready"]:
        ann_info = info["annotations_info"]
        if not ann_info.get("found"):
            return None

        # Check if format conversion is needed
        ann_col = ann_info.get("column")
        bbox_format = ann_info.get("primary_bbox_format", "unknown")

        if "coco" in bbox_format.lower() or "xywh" in bbox_format.lower():
            # Already COCO format
            return f"""# Dataset appears to be in COCO format (xywh)
# Image column: {info['image_columns'][0] if info['image_columns'] else 'image'}
# Annotation column: {ann_col}
# Use directly with transformers object detection models"""
        elif "xyxy" in bbox_format.lower():
            # Need to convert from XYXY to XYWH
            return f"""# Convert from XYXY (Pascal VOC) to XYWH (COCO) format
def convert_to_coco_format(example):
    annotations = example['{ann_col}']
    if isinstance(annotations, list):
        for ann in annotations:
            if 'bbox' in ann:
                x_min, y_min, x_max, y_max = ann['bbox']
                ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
    elif isinstance(annotations, dict) and 'bbox' in annotations:
        bbox = annotations['bbox']
        if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], list):
            for i, box in enumerate(bbox):
                x_min, y_min, x_max, y_max = box
                bbox[i] = [x_min, y_min, x_max - x_min, y_max - y_min]
    return example

dataset = dataset.map(convert_to_coco_format)"""

    elif not info["ready"]:
        # Need to create annotations structure
        if info["separate_bbox_columns"] and info["separate_category_columns"]:
            bbox_col = info["separate_bbox_columns"][0]
            cat_col = info["separate_category_columns"][0]

            return f"""# Combine separate bbox and category columns
def create_annotations(example):
    bboxes = example['{bbox_col}']
    categories = example['{cat_col}']

    if not isinstance(bboxes, list):
        bboxes = [bboxes]
    if not isinstance(categories, list):
        categories = [categories]

    annotations = []
    for bbox, cat in zip(bboxes, categories):
        annotations.append({{'bbox': bbox, 'category': cat}})

    example['objects'] = annotations
    return example

dataset = dataset.map(create_annotations)"""

    return None


def format_value_preview(value: Any, max_chars: int) -> str:
    # [FUNCTION DOC] HELPER: Render a short human-readable preview of a cell value.
    # Behavior: missing values become "None"; strings are truncated to the limit with "..."; mappings
    # show their key count and first 5 keys; empty lists show "[]"; lists of mappings show the item
    # count and first item's keys; lists of lists show the count and first item; other lists and
    # values are converted to text and truncated.
    # Inputs: value (any) - cell value; max_chars (integer) - truncation length.
    # Outputs: the preview text.
    # Exceptions: none.
    # Side effects: none.
    """Format value for preview"""
    if value is None:
        return "None"
    elif isinstance(value, str):
        return value[:max_chars] + ("..." if len(value) > max_chars else "")
    elif isinstance(value, dict):
        keys = list(value.keys())
        return f"{{dict with {len(keys)} keys: {', '.join(keys[:5])}}}"
    elif isinstance(value, list):
        if len(value) == 0:
            return "[]"
        elif isinstance(value[0], dict):
            return f"[{len(value)} items] First item keys: {list(value[0].keys())}"
        elif isinstance(value[0], list):
            return f"[{len(value)} items] First item: {value[0]}"
        else:
            preview = str(value)
            return preview[:max_chars] + ("..." if len(preview) > max_chars else "")
    else:
        preview = str(value)
        return preview[:max_chars] + ("..." if len(preview) > max_chars else "")


def main():
    # [FUNCTION DOC] Entry point: fetch metadata and samples, run checks and print the report.
    # Behavior: parses arguments; fetches splits, exiting 1 with an error if unavailable. Records all
    # configs; if the requested config/split pair is absent, switches to an arbitrary available
    # config. Fetches the requested number of sample rows (exit 1 if none), takes the columns from the
    # first row and features from the response, and looks up the total example count (formatted with
    # thousands separators) for the chosen config/split. Any exception in this phase prints an error
    # and exits 1. Then runs the detection, classification and SAM checks. In JSON mode prints a
    # single indented JSON document (dataset, config, split, total_examples, columns, features and the
    # three check results) and exits 0. Otherwise prints a banner, dataset facts, column types,
    # previews of the first row, a section per task with status, detected columns, label/annotation/
    # prompt details and suggested training-script flags, an optional object-detection preprocessing
    # snippet, and a summary, then exits 0.
    # Inputs: none directly (CLI arguments; see module block).
    # Outputs: report text or JSON on stdout; exit code 0 on success, 1 on fetch/parse failure.
    # Exceptions: exceptions during fetching are caught and turned into exit code 1; process exits
    # are raised via system exit.
    # Side effects: network requests; stdout output; terminates the process.
    # KNOWN ISSUE: the fallback config is chosen from an unordered set, so which config is used when
    # the requested one is missing is arbitrary, and the requested split is not checked to exist in
    # the fallback config before fetching rows.
    args = parse_args()

    print(f"Fetching dataset info via Datasets Server API...")

    try:
        # Get splits info
        splits_data = get_splits(args.dataset)
        if not splits_data or "splits" not in splits_data:
            print(f"ERROR: Could not fetch splits for dataset '{args.dataset}'")
            print(f"       Dataset may not exist or is not accessible via Datasets Server API")
            sys.exit(1)

        # Find the right config
        available_configs = set()
        split_found = False
        config_to_use = args.config

        for split_info in splits_data["splits"]:
            available_configs.add(split_info["config"])
            if split_info["config"] == args.config and split_info["split"] == args.split:
                split_found = True

        # If default config not found, try first available
        if not split_found and available_configs:
            config_to_use = list(available_configs)[0]
            print(f"Config '{args.config}' not found, trying '{config_to_use}'...")

        # Get rows
        rows_data = get_rows(args.dataset, config_to_use, args.split, offset=0, length=args.samples)

        if not rows_data or "rows" not in rows_data:
            print(f"ERROR: Could not fetch rows for dataset '{args.dataset}'")
            print(f"       Split '{args.split}' may not exist")
            print(f"       Available configs: {', '.join(sorted(available_configs))}")
            sys.exit(1)

        rows = rows_data["rows"]
        if not rows:
            print(f"ERROR: No rows found in split '{args.split}'")
            sys.exit(1)

        # Extract column info from first row
        first_row = rows[0]["row"]
        columns = list(first_row.keys())
        features = rows_data.get("features", [])

        # Get total count if available
        total_examples = "Unknown"
        for split_info in splits_data["splits"]:
            if split_info["config"] == config_to_use and split_info["split"] == args.split:
                total_examples = f"{split_info.get('num_examples', 'Unknown'):,}" if isinstance(split_info.get('num_examples'), int) else "Unknown"
                break

    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

    # Run compatibility checks
    od_info = check_object_detection_compatibility(columns, rows)
    ic_info = check_image_classification_compatibility(columns, rows, features)
    sam_info = check_sam_segmentation_compatibility(columns, rows, features)

    # JSON output mode
    if args.json_output:
        result = {
            "dataset": args.dataset,
            "config": config_to_use,
            "split": args.split,
            "total_examples": total_examples,
            "columns": columns,
            "features": [{"name": f["name"], "type": f["type"]} for f in features] if features else [],
            "object_detection_compatibility": od_info,
            "image_classification_compatibility": ic_info,
            "sam_segmentation_compatibility": sam_info,
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # Human-readable output optimized for LLM parsing
    print("=" * 80)
    print(f"VISION DATASET INSPECTION")
    print("=" * 80)

    print(f"\nDataset: {args.dataset}")
    print(f"Config: {config_to_use}")
    print(f"Split: {args.split}")
    print(f"Total examples: {total_examples}")
    print(f"Samples fetched: {len(rows)}")

    print(f"\n{'COLUMNS':-<80}")
    if features:
        for feature in features:
            print(f"  {feature['name']}: {feature['type']}")
    else:
        for col in columns:
            print(f"  {col}: (type info not available)")

    print(f"\n{'EXAMPLE DATA':-<80}")
    example = first_row
    for col in columns:
        value = example.get(col)
        display = format_value_preview(value, args.preview)
        print(f"\n{col}:")
        print(f"  {display}")

    # --- Image Classification ---
    print(f"\n{'IMAGE CLASSIFICATION COMPATIBILITY':-<80}")
    print(f"\n[STATUS] {'✓ READY' if ic_info['ready'] else '✗ NOT COMPATIBLE'}")

    print(f"\nImage Column:")
    if ic_info["has_image"]:
        print(f"  ✓ Found: {', '.join(ic_info['image_columns'])}")
    else:
        print(f"  ✗ No image column detected")

    print(f"\nLabel Column:")
    if ic_info["has_label"]:
        print(f"  ✓ Found: {', '.join(ic_info['label_columns'])}")
        li = ic_info["label_info"]
        if li.get("type"):
            print(f"    • Type: {li['type']}")
        if li.get("num_classes"):
            print(f"    • Number of Classes: {li['num_classes']}")
        if li.get("class_names"):
            names = li["class_names"]
            display = ", ".join(str(n) for n in names[:10])
            if len(names) > 10:
                display += f" ... ({li['num_classes']} total)"
            print(f"    • Classes: {display}")
        elif li.get("sample_unique_labels"):
            labels = li["sample_unique_labels"]
            display = ", ".join(str(l) for l in labels[:10])
            if li.get("sample_unique_count", 0) > 10:
                display += f" ... ({li['sample_unique_count']}+ from sample)"
            print(f"    • Sample labels: {display}")
    else:
        print(f"  ✗ No label column detected")
        print(f"  Expected column names: 'label', 'labels', 'class', 'fine_label'")

    if ic_info["ready"]:
        lc = ic_info["label_info"].get("column", "label")
        print(f"\n  Use with: scripts/image_classification_training.py")
        print(f"    --image_column_name {ic_info['image_columns'][0]} --label_column_name {lc}")

    # --- Object Detection ---
    print(f"\n{'OBJECT DETECTION COMPATIBILITY':-<80}")
    print(f"\n[STATUS] {'✓ READY' if od_info['ready'] else '✗ NOT COMPATIBLE'}")

    print(f"\nImage Column:")
    if od_info["has_image"]:
        print(f"  ✓ Found: {', '.join(od_info['image_columns'])}")
    else:
        print(f"  ✗ No image column detected")
        print(f"  Expected column names: 'image', 'img', 'picture', 'photo'")

    print(f"\nAnnotations:")
    if od_info["has_annotations"]:
        print(f"  ✓ Found: {', '.join(od_info['annotation_columns'])}")
        ann_info = od_info["annotations_info"]
        if ann_info.get("found"):
            print(f"\n  Annotation Details:")
            print(f"    • Column: {ann_info['column']}")
            if ann_info.get("primary_bbox_format"):
                print(f"    • BBox Format: {ann_info['primary_bbox_format']}")
            if ann_info.get("num_classes", 0) > 0:
                print(f"    • Number of Classes: {ann_info['num_classes']}")
                print(f"    • Classes: {', '.join(ann_info['categories_found'][:10])}")
                if len(ann_info['categories_found']) > 10:
                    print(f"      (showing first 10 of {len(ann_info['categories_found'])})")
            print(f"    • Avg Objects/Image: {ann_info['avg_objects_per_image']}")
            print(f"    • Min Objects: {ann_info['min_objects']}")
            print(f"    • Max Objects: {ann_info['max_objects']}")
    elif od_info["separate_bbox_columns"] and od_info["separate_category_columns"]:
        print(f"  ⚠ Separate bbox and category columns found:")
        print(f"    BBox columns: {', '.join(od_info['separate_bbox_columns'])}")
        print(f"    Category columns: {', '.join(od_info['separate_category_columns'])}")
        print(f"  Action: These need to be combined (see mapping code below)")
    else:
        print(f"  ✗ No annotation columns detected")
        print(f"  Expected: 'objects', 'annotations', 'bbox'/'bboxes' + 'category'/'label'")

    # --- SAM Segmentation ---
    print(f"\n{'SAM SEGMENTATION COMPATIBILITY':-<80}")
    print(f"\n[STATUS] {'✓ READY' if sam_info['ready'] else '✗ NOT COMPATIBLE'}")

    print(f"\nImage Column:")
    if sam_info["has_image"]:
        print(f"  ✓ Found: {', '.join(sam_info['image_columns'])}")
    else:
        print(f"  ✗ No image column detected")

    print(f"\nMask Column:")
    if sam_info["has_mask"]:
        print(f"  ✓ Found: {', '.join(sam_info['mask_columns'])}")
    else:
        print(f"  ✗ No mask column detected")
        print(f"  Expected column names: 'mask', 'segmentation', 'alpha', 'matte'")

    print(f"\nPrompt:")
    pi = sam_info["prompt_info"]
    if pi["has_prompt"]:
        print(f"  ✓ Type: {pi['prompt_type']} (from {pi['source']})")
        if pi.get("bbox_valid"):
            bv = pi["bbox_valid"]
            if bv["valid"]:
                print(f"    • BBox values: {bv.get('values')}")
                if bv.get("format_hint"):
                    print(f"    • Format: {bv['format_hint']}")
                if bv.get("warning"):
                    print(f"    ⚠ {bv['warning']}")
            else:
                print(f"    ✗ Invalid bbox: {bv.get('error', 'unknown error')}")
    else:
        print(f"  ✗ No prompt detected")
        print(f"  Expected: 'prompt' column (JSON with bbox/point), or 'bbox'/'point' column")

    if sam_info["ready"]:
        pc = sam_info["prompt_columns"][0] if sam_info["prompt_columns"] else None
        args_hint = f"--prompt_type {pi['prompt_type']}"
        if pc:
            args_hint += f" --prompt_column_name {pc}"
        print(f"\n  Use with: scripts/sam_segmentation_training.py")
        print(f"    {args_hint}")

    # Mapping code (OD only)
    mapping_code = generate_mapping_code(od_info)

    if mapping_code:
        print(f"\n{'OD PREPROCESSING CODE':-<80}")
        print(mapping_code)
    elif od_info["ready"]:
        print(f"\n  ✓ No OD preprocessing needed.")

    # --- Summary ---
    print(f"\n{'SUMMARY':-<80}")
    if ic_info["ready"]:
        num_cls = ic_info["label_info"].get("num_classes") or ic_info["label_info"].get("sample_unique_count", "?")
        print(f"✓ Image Classification: READY ({num_cls} classes)")
    else:
        print(f"✗ Image Classification: not compatible")

    if od_info["ready"]:
        ann_info = od_info["annotations_info"]
        fmt = ann_info.get("primary_bbox_format", "")
        cls = ann_info.get("num_classes", "?")
        print(f"✓ Object Detection: READY ({cls} classes, {fmt})")
    else:
        print(f"✗ Object Detection: not compatible")

    if sam_info["ready"]:
        print(f"✓ SAM Segmentation: READY (prompt: {pi['prompt_type']})")
    else:
        print(f"✗ SAM Segmentation: not compatible")

    print(f"\nNote: Used Datasets Server API (instant, no download required)")

    print("\n" + "=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
