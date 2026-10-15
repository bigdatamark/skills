#!/usr/bin/env python3
# [MODULE DOC] CLI script that collects benchmark scores for trending text-generation models on the Hub.
# Behavior: fetches the trending model list, keeps text-generation models, and for each reads
# the model card YAML front matter (README.md) looking for model-index results that match MMLU,
# BigCodeBench or ARC MC. If the main branch has none, it scans the model's open/closed pull
# requests (newest first) and uses the first PR whose card has matching scores. Results are
# flattened to one record per model and benchmark, printed, saved as JSON locally and
# optionally uploaded to a Hugging Face dataset as JSONL plus metadata.
# Constants: API_BASE (Hub API root), PIPELINE_FILTER (text-generation), TRENDING_LIMIT (50
# models kept), TRENDING_FETCH_LIMIT (100 models requested), PR_SCAN_LIMIT (40 discussions
# requested), USER_AGENT (sent with every request), BENCHMARKS (mapping from benchmark key to
# its spec: mmlu, bigcodebench, arc_mc with display labels and alias strings).
# CLI arguments: --push-to-hub (flag, upload results), --output (local JSON path, default
# leaderboard.json), --repo-id (target dataset, default hf-skills/evals-leaderboard).
# Env vars: HF_TOKEN (optional; used for authenticated API calls, downloads and uploads).
# Top-level executed code: BENCHMARKS is built on import; main runs when executed as a script.
# Side effects: network access to huggingface.co, README downloads into the HF cache, writes
# the output JSON file, optional dataset repo creation and file uploads, console output.
"""
Collect evaluation scores from trending models' model-index metadata.

Scans trending text-generation models on the Hub and extracts benchmark
scores from their model-index metadata or open pull requests.

Results are saved to a dataset for the evals leaderboard.

Usage:
    python collect_evals.py [--push-to-hub]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

API_BASE = "https://huggingface.co/api"
PIPELINE_FILTER = "text-generation"
TRENDING_LIMIT = 50
TRENDING_FETCH_LIMIT = 100
PR_SCAN_LIMIT = 40
USER_AGENT = "skills-evals-leaderboard/0.3"


def _normalize(text: Optional[str]) -> str:
    # [FUNCTION DOC] HELPER: Normalize text for fuzzy benchmark matching.
    # Behavior: returns empty string for missing/empty input; otherwise lowercases, replaces every
    # run of characters other than a-z and 0-9 with a single space, and trims ends.
    # Inputs: text (optional string).
    # Outputs: the normalized string (possibly empty).
    # Exceptions: none.
    # Side effects: none.
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _coerce_score(value: Any) -> Optional[float]:
    # [FUNCTION DOC] HELPER: Convert a raw metric value into a float score when possible.
    # Behavior: missing value gives nothing; numbers (including booleans, which are ints) become
    # floats; strings are trimmed, a trailing percent sign is removed and the rest parsed as a
    # float, giving nothing if parsing fails; any other type gives nothing.
    # Inputs: value (any) - the metric value from model-index.
    # Outputs: a float, or nothing (None) when not convertible.
    # Exceptions: catches the float-parsing error for strings and returns nothing.
    # Side effects: none.
    # KNOWN ISSUE: boolean values are accepted as 1.0/0.0 because booleans count as integers.
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("%"):
            candidate = candidate[:-1]
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class BenchmarkSpec:
    # [CLASS DOC] Immutable description of one target benchmark.
    # Fields: key (internal id), label (display name), aliases (tuple of names that identify it).
    # Behavior: frozen dataclass; provides alias matching against normalized fields.
    # Side effects: none.
    key: str
    label: str
    aliases: tuple[str, ...]

    def matches(self, fields: List[str]) -> bool:
        # [FUNCTION DOC] Decide whether any alias of this benchmark appears in the given normalized fields.
        # Behavior: for each alias, normalize it (skip if empty); return true as soon as the
        # normalized alias is a substring of any field; otherwise return false.
        # Inputs: fields (list of already-normalized strings: dataset name/type, metric name/type).
        # Outputs: boolean.
        # Exceptions: none.
        # Side effects: none.
        # KNOWN ISSUE: plain substring matching is loose, e.g. alias 'arc c' matches any field
        # containing 'arc c...' (such as 'arc challenge' but also unrelated words starting with c),
        # and 'mmlu' also matches variants like MMLU-Pro or MMLU-Redux.
        for alias in self.aliases:
            alias_norm = _normalize(alias)
            if not alias_norm:
                continue
            for field in fields:
                if alias_norm in field:
                    return True
        return False


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "mmlu": BenchmarkSpec(
        key="mmlu",
        label="MMLU",
        aliases=("mmlu", "massive multitask language understanding"),
    ),
    "bigcodebench": BenchmarkSpec(
        key="bigcodebench",
        label="BigCodeBench",
        aliases=("bigcodebench", "big code bench"),
    ),
    "arc_mc": BenchmarkSpec(
        key="arc_mc",
        label="ARC MC",
        aliases=(
            "arc mc",
            "arc-challenge",
            "arc challenge",
            "arc multiple choice",
            "arc c",
        ),
    ),
}


class EvalsCollector:
    # [CLASS DOC] Collects benchmark scores for trending models and exports them.
    # Behavior: holds an HTTP session (User-Agent and optional bearer token), a list of log
    # messages and a list of flat result records; methods fetch trending models, read model cards
    # and PR cards, extract matching scores, and save/push the leaderboard.
    # Side effects: network access, console printing, file writes (see methods).
    """Collects evaluation scores from model-index metadata."""

    def __init__(self, token: str | None = None) -> None:
        # [FUNCTION DOC] Initialize the collector.
        # Behavior: stores the token, creates a requests session with the User-Agent header and, if a
        # token is given, an Authorization bearer header; initializes empty logs and results lists.
        # Inputs: token (optional string, default none) - Hugging Face access token.
        # Outputs: none.
        # Exceptions: none.
        # Side effects: creates an HTTP session object (no network yet).
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.logs: List[str] = []
        self.results: List[Dict[str, Any]] = []

    def log(self, message: str) -> None:
        # [FUNCTION DOC] HELPER: Print a message and remember it in the log list.
        # Inputs: message (string).
        # Outputs: prints the message to stdout.
        # Exceptions: none.
        # Side effects: appends to the instance's logs list; writes to stdout.
        """Add a log message."""
        print(message)
        self.logs.append(message)

    def collect_all(self) -> List[Dict[str, Any]]:
        # [FUNCTION DOC] Collect scores for all trending text-generation models.
        # Behavior: fetches trending models; for each, takes modelId or id (skipping if absent),
        # collects scores, and if any were found appends their flattened records to results. Logs the
        # total entry count.
        # Inputs: none.
        # Outputs: the accumulated results list (same object stored on the instance).
        # Exceptions: propagates errors from the trending request and from model card reading
        # (non-HTTP errors).
        # Side effects: network requests, logging, mutates results (calling twice appends duplicates).
        """Collect evaluation scores from trending models."""
        self.log("🔍 Fetching trending text-generation models...")
        trending = self._fetch_trending_models()

        for entry in trending:
            repo_id = entry.get("modelId") or entry.get("id")
            if not repo_id:
                continue
            scores = self._collect_scores(repo_id)
            if scores["scores"]:
                self.results.extend(self._format_scores(repo_id, scores["scores"]))

        self.log(f"✅ Collected {len(self.results)} evaluation entries")
        return self.results

    def _fetch_trending_models(self) -> List[Dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Fetch trending models and keep text-generation ones.
        # Behavior: GETs the models endpoint sorted by trendingScore with limit TRENDING_FETCH_LIMIT
        # (30 s timeout); fails on HTTP errors; requires a JSON list; keeps models whose pipeline tag
        # is text-generation or whose tags contain it; logs and returns empty if none; otherwise keeps
        # the first TRENDING_LIMIT and logs the count.
        # Inputs: none.
        # Outputs: list of model info mappings.
        # Exceptions: raises a value error if the response is not a list; propagates request errors.
        # Side effects: one network request; logging.
        params = {"sort": "trendingScore", "limit": TRENDING_FETCH_LIMIT}
        response = self.session.get(
            f"{API_BASE}/models",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected trending response.")
        filtered = [
            model
            for model in data
            if (model.get("pipeline_tag") == PIPELINE_FILTER or PIPELINE_FILTER in (model.get("tags") or []))
        ]
        if not filtered:
            self.log("⚠️ No text-generation models in trending feed.")
            return []
        limited = filtered[:TRENDING_LIMIT]
        self.log(f"📊 Found {len(limited)} trending text-generation models")
        return limited

    def _collect_scores(self, repo_id: str) -> Dict[str, Any]:
        # [FUNCTION DOC] HELPER: Find benchmark scores for one model from its card or its pull requests.
        # Behavior: reads the main-branch card front matter; if it has model-index, extracts scores
        # credited to the repo owner as source model-card at the model page URL, returning them if any.
        # Otherwise lists PRs (newest first) and for each reads the card at refs/pr/<num>; skips PRs
        # without model-index; extracts scores credited to the PR author (name, else fullname, else
        # unknown-author) with the discussion URL as source; returns the first non-empty result.
        # Logs when nothing is found.
        # Inputs: repo_id (string, owner/name).
        # Outputs: mapping with model_id and scores (mapping benchmark key to score payload, possibly
        # empty).
        # Exceptions: propagates non-HTTP errors from card reading; a PR lacking a number would raise
        # a key error.
        # Side effects: README downloads, PR list request, logging.
        # KNOWN ISSUE: logs 'model card metadata found' whenever model-index exists, even if none of
        # the target benchmarks are extracted from it.
        owner = repo_id.split("/")[0]
        card_meta = self._read_model_card(repo_id)
        model_index = card_meta.get("model-index")
        if model_index:
            self.log(f"✅ {repo_id}: model card metadata found.")
            scores = self._extract_scores(
                repo_id=repo_id,
                model_index=model_index,
                contributor=owner,
                source_type="model-card",
                source_url=f"https://huggingface.co/{repo_id}",
                revision="main",
            )
            if scores:
                return {"model_id": repo_id, "scores": scores}

        prs = self._fetch_pull_requests(repo_id)
        for pr in prs:
            revision = f"refs/pr/{pr['num']}"
            pr_meta = self._read_model_card(repo_id, revision=revision)
            pr_index = pr_meta.get("model-index")
            if not pr_index:
                continue
            author_info = pr.get("author", {}) or {}
            contributor = author_info.get("name") or author_info.get("fullname") or "unknown-author"
            discussion_path = f"{repo_id}/discussions/{pr['num']}"
            source_url = f"https://huggingface.co/{discussion_path}"
            scores = self._extract_scores(
                repo_id=repo_id,
                model_index=pr_index,
                contributor=contributor,
                source_type="pull-request",
                source_url=source_url,
                revision=revision,
            )
            if scores:
                note = f"📝 {repo_id}: PR #{pr['num']} by {contributor}."
                self.log(note)
                return {"model_id": repo_id, "scores": scores}

        self.log(f"⚠️ {repo_id}: no target benchmarks located.")
        return {"model_id": repo_id, "scores": {}}

    def _read_model_card(
        self,
        repo_id: str,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC] HELPER: Download a model's README.md and parse its YAML front matter.
        # Behavior: downloads README.md at the given revision via hf_hub_download using the token;
        # on a Hub HTTP error logs and returns an empty mapping; otherwise reads the file as UTF-8
        # (ignoring bad bytes) and parses the front matter.
        # Inputs: repo_id (string); revision (optional string, default none meaning main branch).
        # Outputs: mapping of front matter keys (empty if absent/invalid or download failed).
        # Exceptions: swallows Hub HTTP errors; other errors (e.g. connection errors) propagate.
        # Side effects: network download into the local Hugging Face cache; file read; logging.
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                revision=revision,
                token=self.token,
            )
        except HfHubHTTPError as err:
            ctx = f"{repo_id} ({revision or 'main'})"
            self.log(f"🚫 {ctx}: README download failed ({err}).")
            return {}
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return self._parse_front_matter(text)

    @staticmethod
    def _parse_front_matter(content: str) -> Dict[str, Any]:
        # [FUNCTION DOC] HELPER: Extract and parse the YAML front matter block from markdown text (static method).
        # Behavior: strips a leading byte-order mark; if text does not start with three dashes
        # returns empty; finds the next line that is exactly three dashes (after trimming); if none,
        # returns empty; safely loads the YAML between them, returning it only if it is a mapping.
        # Inputs: content (string) - full README text.
        # Outputs: mapping (empty on any failure).
        # Exceptions: catches YAML parse errors and returns an empty mapping.
        # Side effects: none.
        content = content.lstrip("\ufeff")
        if not content.startswith("---"):
            return {}
        lines = content.splitlines()
        end_idx = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            return {}
        front_matter = "\n".join(lines[1:end_idx])
        try:
            data = yaml.safe_load(front_matter) or {}
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}

    def _fetch_pull_requests(self, repo_id: str) -> List[Dict[str, Any]]:
        # [FUNCTION DOC] HELPER: List a model's pull requests, newest first.
        # Behavior: GETs the model's discussions endpoint with limit PR_SCAN_LIMIT (30 s timeout); on
        # request failure logs and returns empty; keeps discussions flagged as pull requests, sorts by
        # createdAt descending, logs the count if any.
        # Inputs: repo_id (string).
        # Outputs: list of PR discussion mappings.
        # Exceptions: swallows request errors; JSON decoding errors propagate.
        # Side effects: one network request; logging.
        # KNOWN ISSUE: the Hub discussions endpoint is paginated and may ignore the limit parameter,
        # so only the first page of discussions is scanned.
        url = f"{API_BASE}/models/{repo_id}/discussions"
        try:
            response = self.session.get(
                url,
                params={"limit": PR_SCAN_LIMIT},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as err:
            self.log(f"🚫 {repo_id}: PR list request failed ({err}).")
            return []

        payload = response.json()
        discussions = payload.get("discussions", [])
        prs = [disc for disc in discussions if disc.get("isPullRequest")]
        prs.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
        if prs:
            self.log(f"📬 {repo_id}: scanning {len(prs)} pull requests.")
        return prs

    def _extract_scores(
        self,
        repo_id: str,
        model_index: Any,
        contributor: str,
        source_type: str,
        source_url: str,
        revision: str,
    ) -> Dict[str, Dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Pull target benchmark scores out of a model-index structure.
        # Behavior: requires model-index to be a list; for each mapping entry (model name from entry,
        # else repo name) and each result, reads dataset name/type and task type; for each metric
        # matches a benchmark, coerces the value (skipping unmatched or non-numeric), sets unit to
        # the metric unit or percent if the raw string ended with a percent sign, and builds a payload
        # (model, model_name, benchmark key/label, value, unit, dataset, task_type, metric_name,
        # contributor, source_type, source_url, revision). Keeps the highest value per benchmark.
        # Inputs: repo_id, model_index (any), contributor, source_type, source_url, revision (strings).
        # Outputs: mapping from benchmark key to payload mapping (possibly empty).
        # Exceptions: none caught; malformed nested items (non-mapping results/metrics) would raise.
        # Side effects: none.
        # KNOWN ISSUE: the max-per-benchmark comparison ignores units, so a percent value (e.g. 70)
        # always beats a fraction value (e.g. 0.75) for the same benchmark.
        if not isinstance(model_index, list):
            return {}
        scores: Dict[str, Dict[str, Any]] = {}
        for entry in model_index:
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("name") or repo_id.split("/")[-1]
            for result in entry.get("results", []):
                dataset_info = result.get("dataset") or {}
                dataset_name = dataset_info.get("name")
                dataset_type = dataset_info.get("type")
                task_info = result.get("task") or {}
                task_type = task_info.get("type")
                for metric in result.get("metrics", []):
                    benchmark_key = self._match_benchmark(
                        dataset_name,
                        dataset_type,
                        metric,
                    )
                    if not benchmark_key:
                        continue
                    raw_value = metric.get("value")
                    value = _coerce_score(raw_value)
                    if value is None:
                        continue
                    unit = metric.get("unit") or ""
                    is_pct = isinstance(raw_value, str) and raw_value.strip().endswith("%")
                    if not unit and is_pct:
                        unit = "%"
                    metric_name = metric.get("name") or metric.get("type") or ""
                    payload = {
                        "model": repo_id,
                        "model_name": model_name,
                        "benchmark_key": benchmark_key,
                        "benchmark_label": BENCHMARKS[benchmark_key].label,
                        "value": value,
                        "unit": unit,
                        "dataset": dataset_name or dataset_type or "",
                        "task_type": task_type or "",
                        "metric_name": metric_name,
                        "contributor": contributor,
                        "source_type": source_type,
                        "source_url": source_url,
                        "revision": revision,
                    }
                    existing = scores.get(benchmark_key)
                    if not existing or value > existing["value"]:
                        scores[benchmark_key] = payload
        return scores

    def _match_benchmark(
        self,
        dataset_name: Optional[str],
        dataset_type: Optional[str],
        metric: Dict[str, Any],
    ) -> Optional[str]:
        # [FUNCTION DOC] HELPER: Determine which target benchmark a metric refers to.
        # Behavior: normalizes dataset name, dataset type, metric name and metric type, drops empty
        # ones, and returns the first BENCHMARKS key (in definition order) whose spec matches.
        # Inputs: dataset_name (optional string); dataset_type (optional string); metric (mapping).
        # Outputs: benchmark key string, or nothing if no match.
        # Exceptions: none.
        # Side effects: none.
        fields = [
            _normalize(dataset_name),
            _normalize(dataset_type),
            _normalize(metric.get("name")),
            _normalize(metric.get("type")),
        ]
        fields = [field for field in fields if field]
        for key, spec in BENCHMARKS.items():
            if spec.matches(fields):
                return key
        return None

    def _format_scores(self, model_id: str, scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Flatten a model's score payloads into dataset records.
        # Behavior: for each benchmark payload creates a record with model_id, benchmark label,
        # benchmark_key, score rounded to 2 decimals, source_type, source_url, contributor and the
        # current UTC time as collected_at.
        # Inputs: model_id (string); scores (mapping from benchmark key to payload).
        # Outputs: list of record mappings.
        # Exceptions: none.
        # Side effects: reads the clock.
        """Format scores as flat records for the dataset."""
        rows = []
        for benchmark_key, payload in scores.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark": payload["benchmark_label"],
                    "benchmark_key": benchmark_key,
                    "score": round(payload["value"], 2),
                    "source_type": payload["source_type"],
                    "source_url": payload["source_url"],
                    "contributor": payload["contributor"],
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return rows

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        # [FUNCTION DOC] Return all results sorted by score, highest first.
        # Inputs: none.
        # Outputs: new sorted list of record mappings (results are not modified).
        # Exceptions: none.
        # Side effects: none.
        # KNOWN ISSUE: scores of different benchmarks and units are ranked together in one ordering.
        """Get results sorted by score descending."""
        return sorted(self.results, key=lambda x: x["score"], reverse=True)

    def save_json(self, filepath: str) -> None:
        # [FUNCTION DOC] Write the leaderboard to a local JSON file.
        # Behavior: writes an object with generated_at (UTC now), total_entries, benchmarks (list of
        # keys) and leaderboard (sorted records), indented by 2; logs the path.
        # Inputs: filepath (string).
        # Outputs: the file written at filepath (overwritten).
        # Exceptions: propagates file system errors.
        # Side effects: file write; logging.
        """Save the leaderboard to a JSON file."""
        leaderboard = self.get_leaderboard()
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(leaderboard),
            "benchmarks": list(BENCHMARKS.keys()),
            "leaderboard": leaderboard,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        self.log(f"💾 Saved leaderboard to {filepath}")

    def push_to_hub(self, repo_id: str = "hf-skills/evals-leaderboard") -> None:
        # [FUNCTION DOC] Upload the leaderboard and metadata to a Hugging Face dataset repo.
        # Behavior: imports HfApi (logs and returns if missing); builds JSONL of sorted records and a
        # metadata object (generated_at, total_entries, models_with_scores as distinct model count,
        # benchmarks); creates the dataset repo if needed, uploads data/leaderboard.jsonl and
        # data/metadata.json with timestamped commit messages, and logs success.
        # Inputs: repo_id (string, default hf-skills/evals-leaderboard).
        # Outputs: none.
        # Exceptions: catches ImportError and any upload/creation exception, logging them only.
        # Side effects: network calls creating a repo and two commits; logging.
        """Push the leaderboard data to a HF dataset."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            self.log("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
            return

        api = HfApi(token=self.token)
        leaderboard = self.get_leaderboard()

        # Create dataset as JSONL
        jsonl_content = "\n".join(json.dumps(row) for row in leaderboard)

        # Create metadata file
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(leaderboard),
            "models_with_scores": len(set(r["model_id"] for r in leaderboard)),
            "benchmarks": list(BENCHMARKS.keys()),
        }

        try:
            # Create repo if it doesn't exist
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            self.log(f"📁 Ensured dataset repo exists: {repo_id}")

            # Upload leaderboard data
            api.upload_file(
                path_or_fileobj=jsonl_content.encode(),
                path_in_repo="data/leaderboard.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update leaderboard - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            )

            # Upload metadata
            api.upload_file(
                path_or_fileobj=json.dumps(metadata, indent=2).encode(),
                path_in_repo="data/metadata.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update metadata - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            )

            self.log(f"🚀 Pushed leaderboard to {repo_id}")
        except Exception as e:
            self.log(f"❌ Failed to push to hub: {e}")


def main() -> None:
    # [FUNCTION DOC] CLI entry point: collect, print, save and optionally push the evals leaderboard.
    # Behavior: parses --push-to-hub, --output and --repo-id; reads HF_TOKEN (warns if absent);
    # collects all scores; prints a banner and up to 20 entries (model, benchmark, score), a
    # count of remaining entries, totals of entries and distinct models; saves JSON to --output;
    # pushes to --repo-id if requested.
    # Inputs: command-line arguments (see module doc).
    # Outputs: console output; JSON file; optional dataset upload. Exit code 0 unless an
    # uncaught exception occurs (argparse exits 2 on bad arguments).
    # Exceptions: none caught here.
    # Side effects: env var read, network, file write, stdout.
    parser = argparse.ArgumentParser(description="Collect evaluation scores from model-index metadata")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push results to HF dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="leaderboard.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="hf-skills/evals-leaderboard",
        help="HF dataset repo ID for pushing",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️ No HF_TOKEN found. Some requests may be rate-limited.")

    collector = EvalsCollector(token=token)
    collector.collect_all()

    # Print leaderboard summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION LEADERBOARD")
    print("=" * 60)

    leaderboard = collector.get_leaderboard()
    for entry in leaderboard[:20]:
        print(f"{entry['model_id']:40} | {entry['benchmark']:12} | {entry['score']:6.2f}")

    if len(leaderboard) > 20:
        print(f"   ... and {len(leaderboard) - 20} more entries")

    print("=" * 60)
    print(f"Total entries: {len(leaderboard)}")
    print(f"Models with scores: {len(set(r['model_id'] for r in leaderboard))}")

    # Save locally
    collector.save_json(args.output)

    # Push to hub if requested
    if args.push_to_hub:
        collector.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()
