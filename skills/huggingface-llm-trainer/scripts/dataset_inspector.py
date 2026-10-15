#!/usr/bin/env python3
# [MODULE DOC] CLI that inspects a Hub dataset via the Datasets Server API and reports its
# compatibility with TRL training methods (SFT, DPO, GRPO, KTO), plus suggested mapping code.
# Behavior: fetches split metadata and the first rows of a split over HTTPS (no dataset
# download), derives column names from the first row, runs heuristic compatibility checks and
# prints either a human-readable report or a JSON document.
# CLI arguments: --dataset (required, dataset ID); --split (default "train"); --config
# (default "default"); --preview (int, max chars per field preview, default 150); --samples
# (int, rows to fetch, default 5); --json-output (flag, print JSON instead of text).
# Env vars: none. Constants: none (API base URL is inlined in helper functions).
# Top-level executed code: when run as a script, calls main; a keyboard interrupt exits 0 and
# any other uncaught exception is printed to stderr with exit code 1.
# Outputs: report on stdout; exit code 0 on success, 1 on errors.
# Side effects: network requests to datasets-server.huggingface.co.
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Dataset Format Inspector for TRL Training (LLM-Optimized Output)

Inspects Hugging Face datasets to determine TRL training compatibility.
Uses Datasets Server API for instant results - no dataset download needed!

ULTRA-EFFICIENT: Uses HF Datasets Server API - completes in <2 seconds.

Usage with HF Jobs:
    hf_jobs("uv", {
        "script": "https://huggingface.co/datasets/evalstate/trl-helpers/raw/main/dataset_inspector.py",
        "script_args": ["--dataset", "your/dataset", "--split", "train"]
    })
"""

import argparse
import sys
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any


def parse_args():
    # [FUNCTION DOC] HELPER: Define and parse the command-line arguments.
    # Behavior: builds an argument parser with the options listed in the module doc and parses
    # the process arguments.
    # Inputs: none (reads command-line arguments).
    # Outputs: the parsed arguments namespace.
    # Exceptions: argparse exits with code 2 on invalid arguments.
    # Side effects: none beyond reading the command line.
    parser = argparse.ArgumentParser(description="Inspect dataset format for TRL training")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--config", type=str, default="default", help="Dataset config name (default: default)")
    parser.add_argument("--preview", type=int, default=150, help="Max chars per field preview")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to fetch (default: 5)")
    parser.add_argument("--json-output", action="store_true", help="Output as JSON")
    return parser.parse_args()


def api_request(url: str) -> Dict:
    # [FUNCTION DOC] HELPER: GET a Datasets Server URL and decode the JSON body.
    # Behavior: opens the URL with a 10 second timeout, reads and decodes the body as JSON.
    # Inputs: url (string) - full request URL.
    # Outputs: the decoded JSON object (mapping), or nothing (None) on HTTP 404.
    # Exceptions: other HTTP errors are re-raised as a generic exception with code and reason;
    # any other error (network, JSON decode) is re-raised as a generic exception with its message.
    # Side effects: network request.
    # KNOWN ISSUE: the non-404 HTTP error message is re-wrapped by the outer generic handler,
    # giving a doubled "API request failed" prefix; the return annotation omits the None case.
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
    # [FUNCTION DOC] HELPER: Fetch the list of configs/splits for a dataset.
    # Behavior: builds the Datasets Server "splits" URL with the URL-encoded dataset name and
    # delegates to the API request helper.
    # Inputs: dataset (string) - dataset ID.
    # Outputs: decoded JSON mapping (expected to contain "splits"), or nothing on 404.
    # Exceptions: propagates errors from the request helper.
    # Side effects: network request.
    """Get available splits for dataset"""
    url = f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(dataset)}"
    return api_request(url)


def get_rows(dataset: str, config: str, split: str, offset: int = 0, length: int = 5) -> Dict:
    # [FUNCTION DOC] HELPER: Fetch a window of rows from a dataset split.
    # Behavior: builds the Datasets Server "rows" URL from dataset (URL-encoded), config, split,
    # offset and length, and delegates to the API request helper.
    # Inputs: dataset (string); config (string); split (string); offset (int, default 0);
    # length (int, default 5).
    # Outputs: decoded JSON mapping (expected keys "rows" and "features"), or nothing on 404.
    # Exceptions: propagates errors from the request helper.
    # Side effects: network request.
    # KNOWN ISSUE: config and split are inserted without URL-encoding.
    """Get rows from dataset"""
    url = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset)}&config={config}&split={split}&offset={offset}&length={length}"
    return api_request(url)


def find_columns(columns: List[str], patterns: List[str]) -> List[str]:
    # [FUNCTION DOC] HELPER: Return columns whose lowercased name contains any of the patterns.
    # Behavior: keeps the original order of columns, case-insensitive substring match.
    # Inputs: columns (list of strings); patterns (list of lowercase substrings).
    # Outputs: list of matching column names.
    # Exceptions: none.
    # Side effects: none.
    """Find columns matching patterns"""
    return [c for c in columns if any(p in c.lower() for p in patterns)]


def check_sft_compatibility(columns: List[str]) -> Dict[str, Any]:
    # [FUNCTION DOC] HELPER: Assess whether the columns suit supervised fine-tuning.
    # Behavior: ready if a "messages" column, a "text" column, or both "prompt" and "completion"
    # exist; reason names the first matching form in that order. Also guesses the first column
    # resembling a prompt (prompt/instruction/question/input) and a response
    # (response/completion/output/answer), and notes whether a "context" column exists.
    # Inputs: columns (list of strings).
    # Outputs: mapping with ready (bool), reason (string or none), possible_prompt,
    # possible_response (string or none) and has_context (bool).
    # Exceptions: none.
    # Side effects: none.
    """Check SFT compatibility"""
    has_messages = "messages" in columns
    has_text = "text" in columns
    has_prompt_completion = "prompt" in columns and "completion" in columns
    
    ready = has_messages or has_text or has_prompt_completion
    
    possible_prompt = find_columns(columns, ["prompt", "instruction", "question", "input"])
    possible_response = find_columns(columns, ["response", "completion", "output", "answer"])
    
    return {
        "ready": ready,
        "reason": "messages" if has_messages else "text" if has_text else "prompt+completion" if has_prompt_completion else None,
        "possible_prompt": possible_prompt[0] if possible_prompt else None,
        "possible_response": possible_response[0] if possible_response else None,
        "has_context": "context" in columns,
    }


def check_dpo_compatibility(columns: List[str]) -> Dict[str, Any]:
    # [FUNCTION DOC] HELPER: Assess whether the columns suit DPO preference training.
    # Behavior: ready if "prompt", "chosen" and "rejected" columns all exist; guesses prompt,
    # chosen (chosen/preferred/winner) and rejected (rejected/dispreferred/loser) columns by
    # substring and marks can_map when all three guesses exist.
    # Inputs: columns (list of strings).
    # Outputs: mapping with ready, can_map (bools) and prompt_col, chosen_col, rejected_col.
    # Exceptions: none.
    # Side effects: none.
    """Check DPO compatibility"""
    has_standard = "prompt" in columns and "chosen" in columns and "rejected" in columns
    
    possible_prompt = find_columns(columns, ["prompt", "instruction", "question", "input"])
    possible_chosen = find_columns(columns, ["chosen", "preferred", "winner"])
    possible_rejected = find_columns(columns, ["rejected", "dispreferred", "loser"])
    
    can_map = bool(possible_prompt and possible_chosen and possible_rejected)
    
    return {
        "ready": has_standard,
        "can_map": can_map,
        "prompt_col": possible_prompt[0] if possible_prompt else None,
        "chosen_col": possible_chosen[0] if possible_chosen else None,
        "rejected_col": possible_rejected[0] if possible_rejected else None,
    }


def check_grpo_compatibility(columns: List[str]) -> Dict[str, Any]:
    # [FUNCTION DOC] HELPER: Assess whether the columns suit GRPO (prompt-only) training.
    # Behavior: ready if a "prompt" column exists and neither "chosen" nor "rejected" exists;
    # can_map if some prompt-like column exists and there are no chosen/rejected columns.
    # Inputs: columns (list of strings).
    # Outputs: mapping with ready, can_map (bools) and prompt_col (string or none).
    # Exceptions: none.
    # Side effects: none.
    """Check GRPO compatibility"""
    has_prompt = "prompt" in columns
    has_no_responses = "chosen" not in columns and "rejected" not in columns
    
    possible_prompt = find_columns(columns, ["prompt", "instruction", "question", "input"])
    
    return {
        "ready": has_prompt and has_no_responses,
        "can_map": bool(possible_prompt) and has_no_responses,
        "prompt_col": possible_prompt[0] if possible_prompt else None,
    }


def check_kto_compatibility(columns: List[str]) -> Dict[str, Any]:
    # [FUNCTION DOC] HELPER: Assess whether the columns suit KTO training.
    # Behavior: ready only if "prompt", "completion" and "label" columns all exist.
    # Inputs: columns (list of strings).
    # Outputs: mapping with a single ready boolean.
    # Exceptions: none.
    # Side effects: none.
    """Check KTO compatibility"""
    return {"ready": "prompt" in columns and "completion" in columns and "label" in columns}


def generate_mapping_code(method: str, info: Dict[str, Any]) -> str:
    # [FUNCTION DOC] HELPER: Produce Python source text that maps a dataset to a method's format.
    # Behavior: for "SFT": nothing if ready or no prompt-like column; otherwise a snippet that
    # builds a "text" field from instruction, optional context and response columns (if a
    # context column and response exist), from prompt and response joined by a blank line (if
    # only a response exists), or from the prompt alone. For "DPO"/"GRPO": nothing if ready or not
    # mappable; otherwise a snippet renaming the detected columns to prompt/chosen/rejected or to
    # prompt. Each snippet ends with a dataset map call removing the original columns.
    # Inputs: method (string: "SFT", "DPO" or "GRPO"); info (mapping from the matching
    # compatibility check).
    # Outputs: the snippet as a string, or nothing (None) when no mapping is needed/possible or
    # the method is unknown.
    # Exceptions: none expected.
    # Side effects: none.
    # KNOWN ISSUE: the newline escapes inside the generating f-strings become real line breaks
    # within single-quoted string literals of the emitted snippet, so the printed SFT code is not
    # valid Python as shown.
    """Generate mapping code for a training method"""
    if method == "SFT":
        if info["ready"]:
            return None
        
        prompt_col = info.get("possible_prompt")
        response_col = info.get("possible_response")
        has_context = info.get("has_context", False)
        
        if not prompt_col:
            return None
        
        if has_context and response_col:
            return f"""def format_for_sft(example):
    text = f"Instruction: {{example['{prompt_col}']}}\n\n"
    if example.get('context'):
        text += f"Context: {{example['context']}}\n\n"
    text += f"Response: {{example['{response_col}']}}"
    return {{'text': text}}

dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)"""
        elif response_col:
            return f"""def format_for_sft(example):
    return {{'text': f"{{example['{prompt_col}']}}\n\n{{example['{response_col}']}}}}

dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)"""
        else:
            return f"""def format_for_sft(example):
    return {{'text': example['{prompt_col}']}}

dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)"""
    
    elif method == "DPO":
        if info["ready"] or not info["can_map"]:
            return None
        
        return f"""def format_for_dpo(example):
    return {{
        'prompt': example['{info['prompt_col']}'],
        'chosen': example['{info['chosen_col']}'],
        'rejected': example['{info['rejected_col']}'],
    }}

dataset = dataset.map(format_for_dpo, remove_columns=dataset.column_names)"""
    
    elif method == "GRPO":
        if info["ready"] or not info["can_map"]:
            return None
        
        return f"""def format_for_grpo(example):
    return {{'prompt': example['{info['prompt_col']}']}}

dataset = dataset.map(format_for_grpo, remove_columns=dataset.column_names)"""
    
    return None


def format_value_preview(value: Any, max_chars: int) -> str:
    # [FUNCTION DOC] HELPER: Render a field value as a short preview string.
    # Behavior: none-values become "None"; strings are truncated to max_chars with "..." added
    # when cut; lists of mappings are summarised as item count plus the first item's keys; other
    # lists and values are stringified and truncated the same way.
    # Inputs: value (any); max_chars (int) - truncation length.
    # Outputs: preview string.
    # Exceptions: none.
    # Side effects: none.
    """Format value for preview"""
    if value is None:
        return "None"
    elif isinstance(value, str):
        return value[:max_chars] + ("..." if len(value) > max_chars else "")
    elif isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], dict):
            return f"[{len(value)} items] Keys: {list(value[0].keys())}"
        preview = str(value)
        return preview[:max_chars] + ("..." if len(preview) > max_chars else "")
    else:
        preview = str(value)
        return preview[:max_chars] + ("..." if len(preview) > max_chars else "")


def main():
    # [FUNCTION DOC] Fetch dataset metadata/rows, run compatibility checks and print the report.
    # Behavior: 1) parse arguments; 2) fetch splits, exiting 1 if missing; 3) collect configs and
    # check that the requested config/split pair exists, otherwise switch to an arbitrary
    # available config; 4) fetch the first rows, exiting 1 if missing or empty; 5) take columns
    # from the first row, features from the response and the split's example count (formatted with
    # thousands separators, else "Unknown"); any exception in steps 2-5 prints an error and exits
    # 1; 6) run the SFT/DPO/GRPO/KTO checks and build the recommended method list (with "(needs
    # mapping)" suffixes); 7) with JSON output print a JSON document (dataset, config, split,
    # total_examples, columns, features, compatibility, recommended_methods) and exit 0;
    # 8) otherwise print sections for columns/types, first-row previews, per-method status,
    # mapping snippets (or a no-mapping message) and a summary, then exit 0.
    # Inputs: none directly; reads command-line arguments.
    # Outputs: report on stdout; exit code 0 on success, 1 on fetch errors.
    # Exceptions: catches all exceptions during fetching (prints and exits 1).
    # Side effects: network requests; terminates the process via exit.
    # KNOWN ISSUE: the fallback config is taken from an unordered set, so the choice is arbitrary
    # and the requested split is not verified to exist in it.
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
    sft_info = check_sft_compatibility(columns)
    dpo_info = check_dpo_compatibility(columns)
    grpo_info = check_grpo_compatibility(columns)
    kto_info = check_kto_compatibility(columns)
    
    # Determine recommended methods
    recommended = []
    if sft_info["ready"]:
        recommended.append("SFT")
    elif sft_info["possible_prompt"]:
        recommended.append("SFT (needs mapping)")
    
    if dpo_info["ready"]:
        recommended.append("DPO")
    elif dpo_info["can_map"]:
        recommended.append("DPO (needs mapping)")
    
    if grpo_info["ready"]:
        recommended.append("GRPO")
    elif grpo_info["can_map"]:
        recommended.append("GRPO (needs mapping)")
    
    if kto_info["ready"]:
        recommended.append("KTO")
    
    # JSON output mode
    if args.json_output:
        result = {
            "dataset": args.dataset,
            "config": config_to_use,
            "split": args.split,
            "total_examples": total_examples,
            "columns": columns,
            "features": [{"name": f["name"], "type": f["type"]} for f in features] if features else [],
            "compatibility": {
                "SFT": sft_info,
                "DPO": dpo_info,
                "GRPO": grpo_info,
                "KTO": kto_info,
            },
            "recommended_methods": recommended,
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)
    
    # Human-readable output optimized for LLM parsing
    print("=" * 80)
    print(f"DATASET INSPECTION RESULTS")
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
    
    print(f"\n{'TRAINING METHOD COMPATIBILITY':-<80}")
    
    # SFT
    print(f"\n[SFT] {'✓ READY' if sft_info['ready'] else '✗ NEEDS MAPPING'}")
    if sft_info["ready"]:
        print(f"  Reason: Dataset has '{sft_info['reason']}' field")
        print(f"  Action: Use directly with SFTTrainer")
    elif sft_info["possible_prompt"]:
        print(f"  Detected: prompt='{sft_info['possible_prompt']}' response='{sft_info['possible_response']}'")
        print(f"  Action: Apply mapping code (see below)")
    else:
        print(f"  Status: Cannot determine mapping - manual inspection needed")
    
    # DPO
    print(f"\n[DPO] {'✓ READY' if dpo_info['ready'] else '✗ NEEDS MAPPING' if dpo_info['can_map'] else '✗ INCOMPATIBLE'}")
    if dpo_info["ready"]:
        print(f"  Reason: Dataset has 'prompt', 'chosen', 'rejected' fields")
        print(f"  Action: Use directly with DPOTrainer")
    elif dpo_info["can_map"]:
        print(f"  Detected: prompt='{dpo_info['prompt_col']}' chosen='{dpo_info['chosen_col']}' rejected='{dpo_info['rejected_col']}'")
        print(f"  Action: Apply mapping code (see below)")
    else:
        print(f"  Status: Missing required fields (prompt + chosen + rejected)")
    
    # GRPO
    print(f"\n[GRPO] {'✓ READY' if grpo_info['ready'] else '✗ NEEDS MAPPING' if grpo_info['can_map'] else '✗ INCOMPATIBLE'}")
    if grpo_info["ready"]:
        print(f"  Reason: Dataset has 'prompt' field")
        print(f"  Action: Use directly with GRPOTrainer")
    elif grpo_info["can_map"]:
        print(f"  Detected: prompt='{grpo_info['prompt_col']}'")
        print(f"  Action: Apply mapping code (see below)")
    else:
        print(f"  Status: Missing prompt field")
    
    # KTO
    print(f"\n[KTO] {'✓ READY' if kto_info['ready'] else '✗ INCOMPATIBLE'}")
    if kto_info["ready"]:
        print(f"  Reason: Dataset has 'prompt', 'completion', 'label' fields")
        print(f"  Action: Use directly with KTOTrainer")
    else:
        print(f"  Status: Missing required fields (prompt + completion + label)")
    
    # Mapping code
    print(f"\n{'MAPPING CODE (if needed)':-<80}")
    
    mapping_needed = False
    
    sft_mapping = generate_mapping_code("SFT", sft_info)
    if sft_mapping:
        print(f"\n# For SFT Training:")
        print(sft_mapping)
        mapping_needed = True
    
    dpo_mapping = generate_mapping_code("DPO", dpo_info)
    if dpo_mapping:
        print(f"\n# For DPO Training:")
        print(dpo_mapping)
        mapping_needed = True
    
    grpo_mapping = generate_mapping_code("GRPO", grpo_info)
    if grpo_mapping:
        print(f"\n# For GRPO Training:")
        print(grpo_mapping)
        mapping_needed = True
    
    if not mapping_needed:
        print("\nNo mapping needed - dataset is ready for training!")
    
    print(f"\n{'SUMMARY':-<80}")
    print(f"Recommended training methods: {', '.join(recommended) if recommended else 'None (dataset needs formatting)'}")
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