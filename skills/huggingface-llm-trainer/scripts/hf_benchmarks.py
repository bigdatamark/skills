#!/usr/bin/env -S uv run --script
# [MODULE DOC] CLI to search official Hugging Face benchmark datasets and fetch their leaderboards.
# Behavior overview: "search" lists datasets tagged benchmark:official, scores each against
#   free-text queries, domain aliases, tasks and modalities, and prints ranked matches;
#   "leaderboard" fetches the leaderboard rows of one or more dataset repos and normalizes them.
#   Output can be a fixed-width table, pretty JSON, or NDJSON so commands can be chained
#   (search output piped into "leaderboard --stdin").
# CLI arguments: subcommand "search" with repeatable --query, --alias, --task, --modality, plus
#   --limit (int, default 20) and --format (table, json or ndjson; default table). Subcommand
#   "leaderboard" with positional dataset ids, --stdin flag, --task-id (optional string),
#   --top (optional int) and --format (default table).
# Env vars: HF_TOKEN, if set, is sent as a bearer token on every request.
# Constants: BASE_URL is the Hub root URL; DEFAULT_TIMEOUT is 30 seconds per request; ALIASES maps
#   a domain alias name (ocr, coding, math, retrieval, agents, asr) to a list of search terms.
# Top-level executed code: when run as a script, calls main and exits with its return code.
# Exit codes: 0 success, 1 on API or validation (value) errors, 2 when no dataset ids are given
#   to leaderboard (argparse usage errors also exit 2).
# Side effects: HTTPS GETs to huggingface.co; reads stdin when requested; writes stdout/stderr.
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Search Hugging Face benchmark datasets and fetch leaderboard results.

This script is designed to be pipeline-friendly:
  - search benchmark datasets by free text, alias, task, and modality
  - fetch dataset leaderboards in normalized JSON / NDJSON / table form
  - optionally read dataset ids from stdin for chaining

It uses HF_TOKEN automatically when present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable


BASE_URL = "https://huggingface.co"
DEFAULT_TIMEOUT = 30

ALIASES: dict[str, list[str]] = {
    "ocr": [
        "ocr",
        "olmocr",
        "pdf",
        "image-to-text",
        "screen",
        "screenspot",
        "markdown",
        "text recognition",
    ],
    "coding": [
        "code",
        "coding",
        "software engineering",
        "programming",
        "swe",
        "terminal",
        "patch",
        "bug",
        "cuda",
    ],
    "math": [
        "math",
        "reasoning",
        "gsm8k",
        "mmlu",
        "gpqa",
        "aime",
        "hmmt",
    ],
    "retrieval": [
        "retrieval",
        "search",
        "mteb",
        "arguana",
        "bright",
    ],
    "agents": [
        "agent",
        "agents",
        "terminal",
        "screen",
        "computer use",
        "tool use",
    ],
    "asr": [
        "asr",
        "speech",
        "audio",
        "transcribe",
        "transcription",
    ],
}


class HfApiError(RuntimeError):
    # [CLASS DOC] Error raised for any failed or malformed Hugging Face API call.
    # Behavior: plain subclass of the runtime error type with no extra fields or methods.
    pass


class FullHelpArgumentParser(argparse.ArgumentParser):
    # [CLASS DOC] HELPER: Argument parser whose top-level help also includes each subcommand's help.
    # Behavior: stores optional references to the search and leaderboard subparsers (set by
    #   build_parser after creation) and appends their full help text to the main help output.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Initialize the parser and clear the subparser references.
        # Behavior: forwards all arguments to the base parser, then sets the stored search and
        #   leaderboard subparser references to empty (none).
        # Inputs: any positional and keyword arguments accepted by the base argument parser.
        # Outputs: none.
        # Exceptions: none beyond those of the base constructor.
        # Side effects: sets two instance attributes.
        super().__init__(*args, **kwargs)
        self._search_parser: argparse.ArgumentParser | None = None
        self._leaderboard_parser: argparse.ArgumentParser | None = None

    def format_help(self) -> str:
        # [FUNCTION DOC] HELPER: Build the help text, appending subcommand option sections.
        # Behavior: gets the base help text; for each stored subparser that is set, appends a header line
        #   ("search command options:" / "leaderboard command options:") followed by that subparser's
        #   help, stripped and indented by two spaces; sections are joined by newlines and appended with
        #   a leading and trailing newline only if at least one exists.
        # Inputs: none.
        # Outputs: the combined help string.
        # Exceptions: none.
        # Side effects: none.
        text = super().format_help()
        extra_sections: list[str] = []

        if self._search_parser is not None:
            extra_sections.append(
                "\nsearch command options:\n"
                + textwrap.indent(self._search_parser.format_help().strip(), "  ")
            )

        if self._leaderboard_parser is not None:
            extra_sections.append(
                "\nleaderboard command options:\n"
                + textwrap.indent(self._leaderboard_parser.format_help().strip(), "  ")
            )

        if extra_sections:
            text += "\n" + "\n".join(extra_sections) + "\n"
        return text


def auth_headers() -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Build HTTP auth headers from the HF_TOKEN environment variable.
    # Behavior: if HF_TOKEN is set and non-empty, returns a mapping with an Authorization bearer
    #   header; otherwise an empty mapping.
    # Inputs: none.
    # Outputs: mapping of header name to value.
    # Exceptions: none.
    # Side effects: reads an environment variable.
    token = os.getenv("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def http_get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    # [FUNCTION DOC] HELPER: GET a Hub API path and decode the JSON response.
    # Behavior: joins BASE_URL and the path; if params are given, builds a query string skipping
    #   params whose value is missing, expanding list/tuple values into repeated keys and converting
    #   everything else to strings; sends the request with auth headers and DEFAULT_TIMEOUT; decodes
    #   the body as UTF-8 JSON.
    # Inputs: path (string beginning with a slash); params (optional mapping of name to value).
    # Outputs: the parsed JSON value (any type).
    # Exceptions: HTTP errors are re-raised as HfApiError containing status, reason, URL and the first
    #   500 chars of the body; connection/URL errors are re-raised as HfApiError. JSON decode errors
    #   are not caught.
    # Side effects: network request.
    url = f"{BASE_URL}{path}"
    if params:
        pairs: list[tuple[str, str]] = []
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    pairs.append((key, str(item)))
            else:
                pairs.append((key, str(value)))
        if pairs:
            url = f"{url}?{urllib.parse.urlencode(pairs)}"

    req = urllib.request.Request(url, headers=auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HfApiError(f"{exc.code} {exc.reason} for {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise HfApiError(f"Request failed for {url}: {exc}") from exc


def shorten(text: str, width: int) -> str:
    # [FUNCTION DOC] HELPER: Collapse whitespace and truncate text to a display width.
    # Behavior: treats a missing/empty value as empty, collapses all whitespace runs to single spaces;
    #   if the result fits within width it is returned, otherwise cut to width minus one characters
    #   plus an ellipsis character.
    # Inputs: text (string, may be empty); width (int, maximum length).
    # Outputs: the shortened string.
    # Exceptions: none.
    # Side effects: none.
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def first_text(value: Any) -> str:
    # [FUNCTION DOC] HELPER: Flatten an arbitrary JSON-like value into a single text string.
    # Behavior: missing value gives empty string; strings returned as-is; lists and mappings are
    #   flattened recursively (mapping values only) and joined with spaces; anything else is
    #   converted to its string form.
    # Inputs: value (any JSON-like value).
    # Outputs: a string.
    # Exceptions: none.
    # Side effects: none.
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(first_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(first_text(v) for v in value.values())
    return str(value)


def benchmark_catalog(limit: int = 500) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Fetch the list of official benchmark datasets from the Hub.
    # Behavior: calls the datasets API with filter benchmark:official, the given limit and full
    #   metadata enabled; checks the response is a list.
    # Inputs: limit (int, default 500) maximum datasets to request.
    # Outputs: list of dataset metadata mappings.
    # Exceptions: HfApiError if the response is not a list (plus any from http_get_json).
    # Side effects: network request.
    data = http_get_json(
        "/api/datasets",
        params={"filter": "benchmark:official", "limit": limit, "full": "true"},
    )
    if not isinstance(data, list):
        raise HfApiError("Unexpected response while listing benchmark datasets")
    return data


def dataset_search_blob(dataset: dict[str, Any]) -> str:
    # [FUNCTION DOC] HELPER: Build one lowercase text blob of a dataset's searchable metadata.
    # Behavior: concatenates (space separated) the id, description, top-level tags, and from the card
    #   data the pretty name, tags, task categories and task ids, each flattened to text; lowercases.
    # Inputs: dataset (metadata mapping from the Hub).
    # Outputs: lowercase string.
    # Exceptions: none (missing card data treated as empty).
    # Side effects: none.
    card = dataset.get("cardData") or {}
    parts = [
        dataset.get("id", ""),
        dataset.get("description", ""),
        first_text(dataset.get("tags")),
        first_text(card.get("pretty_name")),
        first_text(card.get("tags")),
        first_text(card.get("task_categories")),
        first_text(card.get("task_ids")),
    ]
    return " ".join(parts).lower()


def dataset_search_fields(dataset: dict[str, Any]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Extract lowercase searchable fields of a dataset, kept separate.
    # Behavior: returns four lowercase strings: "id"; "pretty_name" from card data; "tags" combining
    #   top-level tags with card tags, task categories, task ids and modality; and "description".
    # Inputs: dataset (metadata mapping).
    # Outputs: mapping from field name to lowercase text.
    # Exceptions: none.
    # Side effects: none.
    card = dataset.get("cardData") or {}
    return {
        "id": first_text(dataset.get("id")).lower(),
        "pretty_name": first_text(card.get("pretty_name")).lower(),
        "tags": " ".join(
            [
                first_text(dataset.get("tags")),
                first_text(card.get("tags")),
                first_text(card.get("task_categories")),
                first_text(card.get("task_ids")),
                first_text(card.get("modality")),
            ]
        ).lower(),
        "description": first_text(dataset.get("description")).lower(),
    }


def collect_prefixed_tags(dataset: dict[str, Any], prefixes: Iterable[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Collect a dataset's tags that start with any of the given prefixes.
    # Behavior: 1) keeps top-level string tags starting with any prefix. 2) for card-data lists
    #   task_categories, task_ids and modality, synthesizes "<key>:<value>" tags and keeps those
    #   matching a prefix. 3) removes duplicates preserving first-seen order.
    # Inputs: dataset (metadata mapping); prefixes (iterable of tag prefix strings).
    # Outputs: list of unique tag strings.
    # Exceptions: none.
    # Side effects: none.
    prefixes = tuple(prefixes)
    tags = dataset.get("tags") or []
    card = dataset.get("cardData") or {}

    out: list[str] = []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefixes):
            out.append(tag)

    for key, prefix in (
        ("task_categories", "task_categories:"),
        ("task_ids", "task_ids:"),
        ("modality", "modality:"),
    ):
        values = card.get(key)
        if isinstance(values, list):
            for value in values:
                full_tag = f"{prefix}{value}"
                if full_tag.startswith(prefixes):
                    out.append(full_tag)

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in out:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


def expand_aliases(aliases: list[str]) -> dict[str, list[str]]:
    # [FUNCTION DOC] HELPER: Map each user alias to its list of search terms.
    # Behavior: looks up the lowercased alias in ALIASES; unknown aliases map to a one-item list
    #   containing the alias itself. Keys keep the user's original spelling.
    # Inputs: aliases (list of strings).
    # Outputs: mapping from alias to list of terms.
    # Exceptions: none.
    # Side effects: none.
    expanded: dict[str, list[str]] = {}
    for alias in aliases:
        terms = ALIASES.get(alias.lower(), [alias])
        expanded[alias] = terms
    return expanded


def matches_term(blob: str, term: str) -> bool:
    # [FUNCTION DOC] HELPER: Test whether a search term occurs in a text blob.
    # Behavior: lowercases and trims the term; empty means no match. Terms made only of lowercase
    #   letters, digits and underscores must match as a whole word (not adjacent to other such
    #   characters); other terms (with spaces, hyphens, etc.) match as a plain substring.
    # Inputs: blob (lowercase text to search); term (string).
    # Outputs: true if matched, else false.
    # Exceptions: none.
    # Side effects: none.
    candidate = term.lower().strip()
    if not candidate:
        return False
    if re.fullmatch(r"[a-z0-9_]+", candidate):
        return re.search(rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])", blob) is not None
    return candidate in blob


def score_dataset(
    dataset: dict[str, Any],
    queries: list[str],
    aliases: dict[str, list[str]],
    tasks: list[str],
    modalities: list[str],
) -> dict[str, Any]:
    # [FUNCTION DOC] Score one dataset against the search criteria and build its result row.
    # Behavior: computes the blob, fields, task tags and modality tags. Scoring: each query matching
    #   any field adds 3 ("query:<q>" reason). For each alias, each term adds 2 if it matches id,
    #   pretty name or tags, else 1 if it matches the description; any match adds the alias score and
    #   a reason listing up to 5 matched terms. Each task adds 5 if it exactly equals a
    #   task_categories/task_ids tag, else 2 if it matches the blob as a term. Each modality adds 4
    #   if the exact "modality:<m>" tag is present. Blank tasks/modalities are skipped.
    # Inputs: dataset (metadata mapping); queries (list of strings); aliases (mapping alias to terms);
    #   tasks (list of strings); modalities (list of strings).
    # Outputs: mapping with dataset_id, score, reasons, task_tags, modality_tags, benchmark_tags,
    #   pretty_name, downloads, and whitespace-collapsed description.
    # Exceptions: none.
    # Side effects: none.
    blob = dataset_search_blob(dataset)
    fields = dataset_search_fields(dataset)
    task_tags = collect_prefixed_tags(dataset, ["task_categories:", "task_ids:"])
    modality_tags = collect_prefixed_tags(dataset, ["modality:"])

    score = 0
    reasons: list[str] = []

    for query in queries:
        q = query.lower().strip()
        if q and any(matches_term(value, q) for value in fields.values()):
            score += 3
            reasons.append(f"query:{query}")

    for alias_name, terms in aliases.items():
        matched_terms: list[str] = []
        alias_score = 0
        for term in terms:
            strong_match = any(
                matches_term(fields[field_name], term)
                for field_name in ("id", "pretty_name", "tags")
            )
            desc_match = matches_term(fields["description"], term)
            if strong_match:
                alias_score += 2
                matched_terms.append(term)
            elif desc_match:
                alias_score += 1
                matched_terms.append(term)
        if matched_terms:
            score += alias_score
            reasons.append(f"alias:{alias_name}=" + ",".join(matched_terms[:5]))

    lower_task_tags = [t.lower() for t in task_tags]
    for task in tasks:
        task = task.lower().strip()
        if not task:
            continue
        exact = [
            tag
            for tag in lower_task_tags
            if tag == f"task_categories:{task}" or tag == f"task_ids:{task}"
        ]
        fuzzy = matches_term(blob, task)
        if exact:
            score += 5
            reasons.append(f"task:{task}")
        elif fuzzy:
            score += 2
            reasons.append(f"task~:{task}")

    lower_modality_tags = [m.lower() for m in modality_tags]
    for modality in modalities:
        modality = modality.lower().strip()
        if not modality:
            continue
        if f"modality:{modality}" in lower_modality_tags:
            score += 4
            reasons.append(f"modality:{modality}")

    return {
        "dataset_id": dataset.get("id"),
        "score": score,
        "reasons": reasons,
        "task_tags": task_tags,
        "modality_tags": modality_tags,
        "benchmark_tags": collect_prefixed_tags(dataset, ["benchmark:"]),
        "pretty_name": (dataset.get("cardData") or {}).get("pretty_name"),
        "downloads": dataset.get("downloads"),
        "description": " ".join((dataset.get("description") or "").split()),
    }


def search_benchmarks(
    queries: list[str],
    aliases: list[str],
    tasks: list[str],
    modalities: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Search and rank official benchmark datasets.
    # Behavior: fetches up to 500 benchmark datasets, expands aliases, scores each; if any filter was
    #   given, drops rows scoring below 2; sorts by score descending, then downloads descending, then
    #   dataset id ascending; returns the first "limit" rows.
    # Inputs: queries, aliases, tasks, modalities (lists of strings); limit (int, max rows).
    # Outputs: list of result row mappings (see score_dataset).
    # Exceptions: HfApiError from the catalog fetch.
    # Side effects: network request.
    # KNOWN ISSUE: the catalog is capped at 500 datasets, so benchmarks beyond the first 500 returned
    #   by the API can never be found.
    datasets = benchmark_catalog(limit=500)
    alias_map = expand_aliases(aliases)

    results = [score_dataset(ds, queries, alias_map, tasks, modalities) for ds in datasets]

    active_filters = bool(queries or aliases or tasks or modalities)
    if active_filters:
        results = [row for row in results if row["score"] >= 2]

    results.sort(
        key=lambda row: (
            -row["score"],
            -(row["downloads"] or 0),
            row["dataset_id"] or "",
        )
    )
    return results[:limit]


def parse_repo_id(repo_id: str) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Split a "namespace/repo" id into its two parts.
    # Behavior: requires a slash; splits on the first slash only.
    # Inputs: repo_id (string).
    # Outputs: pair of namespace and repo name.
    # Exceptions: ValueError if no slash is present.
    # Side effects: none.
    if "/" not in repo_id:
        raise ValueError(f"Expected <namespace>/<repo>, got: {repo_id}")
    namespace, repo = repo_id.split("/", 1)
    return namespace, repo


def get_leaderboard(repo_id: str, task_id: str | None = None) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Fetch and normalize a dataset's leaderboard rows.
    # Behavior: validates the repo id, calls the dataset leaderboard API (adding task_id as a query
    #   parameter when given), checks the response is a list, and converts each row to a flat
    #   mapping with dataset_id, task_id, rank, model_id, value, verified, lower_is_better, filename,
    #   notes, pull_request and source name/url/is-external (from the nested source object).
    # Inputs: repo_id (namespace/repo string); task_id (optional string, default none).
    # Outputs: list of normalized row mappings, in API order.
    # Exceptions: ValueError for a bad repo id; HfApiError for request failures or a non-list reply.
    # Side effects: network request.
    namespace, repo = parse_repo_id(repo_id)
    params = {"task_id": task_id} if task_id else None
    data = http_get_json(f"/api/datasets/{namespace}/{repo}/leaderboard", params=params)
    if not isinstance(data, list):
        raise HfApiError(f"Unexpected leaderboard response for {repo_id}")

    normalized: list[dict[str, Any]] = []
    for row in data:
        source = row.get("source") or {}
        normalized.append(
            {
                "dataset_id": repo_id,
                "task_id": task_id,
                "rank": row.get("rank"),
                "model_id": row.get("modelId"),
                "value": row.get("value"),
                "verified": row.get("verified"),
                "lower_is_better": row.get("lower_is_better"),
                "filename": row.get("filename"),
                "notes": row.get("notes"),
                "pull_request": row.get("pullRequest"),
                "source_name": source.get("name"),
                "source_url": source.get("url"),
                "source_is_external": source.get("isExternal"),
            }
        )
    return normalized


def read_repo_ids_from_stdin() -> list[str]:
    # [FUNCTION DOC] HELPER: Read dataset repo ids from piped stdin.
    # Behavior: returns nothing if stdin is an interactive terminal. Otherwise, for each non-blank
    #   line: if it starts with an opening brace it is parsed as JSON (invalid lines skipped) and its
    #   dataset_id or id field is used if it is a string containing a slash; otherwise the line itself
    #   is used if it contains a slash.
    # Inputs: none (reads standard input).
    # Outputs: list of repo id strings (may contain duplicates).
    # Exceptions: JSON decode errors are caught and the line skipped.
    # Side effects: consumes stdin.
    if sys.stdin.isatty():
        return []

    repo_ids: list[str] = []
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate = obj.get("dataset_id") or obj.get("id")
            if isinstance(candidate, str) and "/" in candidate:
                repo_ids.append(candidate)
            continue
        if "/" in line:
            repo_ids.append(line)
    return repo_ids


def print_json(data: Any) -> None:
    # [FUNCTION DOC] HELPER: Write data to stdout as indented JSON.
    # Behavior: dumps with 2-space indent and non-ASCII kept as-is, then writes a newline.
    # Inputs: data (any JSON-serializable value).
    # Outputs: none returned; JSON on stdout.
    # Exceptions: serialization errors propagate.
    # Side effects: stdout write.
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def print_ndjson(rows: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC] HELPER: Write rows to stdout as newline-delimited JSON.
    # Behavior: one compact JSON object per line, non-ASCII kept as-is.
    # Inputs: rows (list of mappings).
    # Outputs: none returned; NDJSON on stdout.
    # Exceptions: serialization errors propagate.
    # Side effects: stdout write.
    for row in rows:
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_search_table(rows: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC] HELPER: Print search results as a fixed-width text table.
    # Behavior: if no rows, prints "No benchmark datasets matched." Otherwise prints a header of
    #   dataset_id, score, modalities, tasks, reasons, description with widths 34, 5, 18, 24, 30, 68,
    #   a dashed separator, then one line per row with list values comma-joined and each cell
    #   shortened to its width and left-padded to align; columns separated by two spaces.
    # Inputs: rows (list of search result mappings).
    # Outputs: none returned; table on stdout.
    # Exceptions: none.
    # Side effects: stdout write.
    if not rows:
        print("No benchmark datasets matched.")
        return

    headers = ["dataset_id", "score", "modalities", "tasks", "reasons", "description"]
    widths = [34, 5, 18, 24, 30, 68]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        values = [
            shorten(row.get("dataset_id") or "", widths[0]),
            str(row.get("score", "")),
            shorten(", ".join(row.get("modality_tags") or []), widths[2]),
            shorten(", ".join(row.get("task_tags") or []), widths[3]),
            shorten(", ".join(row.get("reasons") or []), widths[4]),
            shorten(row.get("description") or "", widths[5]),
        ]
        print("  ".join(v.ljust(w) for v, w in zip(values, widths)))


def print_leaderboard_table(rows: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC] HELPER: Print leaderboard rows as a fixed-width text table.
    # Behavior: if no rows, prints "No leaderboard rows returned." Otherwise prints header dataset_id,
    #   rank, model_id, value, verified, source (widths 30, 5, 38, 10, 8, 28), a dashed separator, and
    #   one aligned line per row, shortening cells to their widths.
    # Inputs: rows (list of normalized leaderboard mappings).
    # Outputs: none returned; table on stdout.
    # Exceptions: none.
    # Side effects: stdout write.
    # KNOWN ISSUE: falsy values are blanked, so a rank or score value of 0 prints as an empty cell.
    if not rows:
        print("No leaderboard rows returned.")
        return

    headers = ["dataset_id", "rank", "model_id", "value", "verified", "source"]
    widths = [30, 5, 38, 10, 8, 28]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        values = [
            shorten(str(row.get("dataset_id") or ""), widths[0]),
            str(row.get("rank") or ""),
            shorten(str(row.get("model_id") or ""), widths[2]),
            shorten(str(row.get("value") or ""), widths[3]),
            str(row.get("verified")),
            shorten(str(row.get("source_name") or ""), widths[5]),
        ]
        print("  ".join(v.ljust(w) for v, w in zip(values, widths)))


def build_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] Build the command-line parser with search and leaderboard subcommands.
    # Behavior: creates a FullHelpArgumentParser with a raw-formatted description containing workflow
    #   examples; adds a required subcommand. "search" gets repeatable --query, --alias (help lists
    #   known ALIASES), --task, --modality (each defaulting to an empty list), --limit (int, 20) and
    #   --format (table/json/ndjson, table). "leaderboard" gets positional datasets (zero or more),
    #   --stdin flag, --task-id (default none), --top (int, default none) and --format. Stores both
    #   subparsers on the main parser so full help includes them.
    # Inputs: none.
    # Outputs: the configured parser.
    # Exceptions: none.
    # Side effects: none.
    parser = FullHelpArgumentParser(
        prog="hf_benchmarks.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Search benchmark datasets and fetch leaderboard results from the Hugging Face Hub.

            Workflow ideas:
              1) Discover candidate benchmarks:
                   hf_benchmarks.py search --alias ocr
                   hf_benchmarks.py search --alias coding
                   hf_benchmarks.py search --task image-to-text --modality document

              2) Inspect a leaderboard:
                   hf_benchmarks.py leaderboard allenai/olmOCR-bench --top 10

              3) Chain search -> leaderboard:
                   hf_benchmarks.py search --alias coding --format ndjson \\
                     | hf_benchmarks.py leaderboard --stdin --top 5 --format table
            """
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search",
        help="Search benchmark datasets by query, alias, task, and modality",
    )
    search_parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Free-text query to match against benchmark dataset metadata. Repeatable.",
    )
    search_parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help=(
            "Convenience alias for common benchmark domains. Known aliases: "
            + ", ".join(sorted(ALIASES))
            + ". Repeatable."
        ),
    )
    search_parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task to match, e.g. text-generation, image-to-text, question-answering. Repeatable.",
    )
    search_parser.add_argument(
        "--modality",
        action="append",
        default=[],
        help="Modality to match, e.g. text, image, document, audio. Repeatable.",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to print (default: 20).",
    )
    search_parser.add_argument(
        "--format",
        choices=["table", "json", "ndjson"],
        default="table",
        help="Output format (default: table).",
    )

    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help="Fetch normalized leaderboard rows for one or more benchmark datasets",
    )
    leaderboard_parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset repo ids (<namespace>/<repo>). Can also be supplied via stdin with --stdin.",
    )
    leaderboard_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read dataset ids from stdin. Accepts plain repo ids or NDJSON with dataset_id/id fields.",
    )
    leaderboard_parser.add_argument(
        "--task-id",
        default=None,
        help="Optional leaderboard task_id query parameter.",
    )
    leaderboard_parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only keep the top N results per leaderboard.",
    )
    leaderboard_parser.add_argument(
        "--format",
        choices=["table", "json", "ndjson"],
        default="table",
        help="Output format (default: table).",
    )

    parser._search_parser = search_parser
    parser._leaderboard_parser = leaderboard_parser

    return parser


def run_search(args: argparse.Namespace) -> int:
    # [FUNCTION DOC] Execute the "search" subcommand.
    # Behavior: runs search_benchmarks with the parsed query, alias, task, modality and limit values,
    #   then prints the rows as JSON, NDJSON or a table depending on --format.
    # Inputs: args (parsed namespace for the search subcommand).
    # Outputs: returns exit code 0; results on stdout.
    # Exceptions: HfApiError propagates to main.
    # Side effects: network request, stdout write.
    rows = search_benchmarks(
        queries=args.query,
        aliases=args.alias,
        tasks=args.task,
        modalities=args.modality,
        limit=args.limit,
    )

    if args.format == "json":
        print_json(rows)
    elif args.format == "ndjson":
        print_ndjson(rows)
    else:
        print_search_table(rows)
    return 0


def run_leaderboard(args: argparse.Namespace) -> int:
    # [FUNCTION DOC] Execute the "leaderboard" subcommand.
    # Behavior: collects positional dataset ids plus, with --stdin, ids read from stdin; removes
    #   duplicates preserving order; if none, prints an error to stderr and returns 2. Fetches each
    #   leaderboard (with --task-id), keeps only the first --top rows per dataset when given,
    #   concatenates all rows and prints them as JSON, NDJSON or a table.
    # Inputs: args (parsed namespace for the leaderboard subcommand).
    # Outputs: exit code 0 on success or 2 when no ids; results on stdout.
    # Exceptions: ValueError / HfApiError propagate to main; one failing dataset aborts the whole run.
    # Side effects: stdin read, network requests, stdout/stderr writes.
    repo_ids = list(args.datasets)
    if args.stdin:
        repo_ids.extend(read_repo_ids_from_stdin())

    deduped: list[str] = []
    seen: set[str] = set()
    for repo_id in repo_ids:
        if repo_id not in seen:
            deduped.append(repo_id)
            seen.add(repo_id)
    repo_ids = deduped

    if not repo_ids:
        print("Error: provide dataset ids or use --stdin.", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for repo_id in repo_ids:
        dataset_rows = get_leaderboard(repo_id, task_id=args.task_id)
        if args.top is not None:
            dataset_rows = dataset_rows[: args.top]
        rows.extend(dataset_rows)

    if args.format == "json":
        print_json(rows)
    elif args.format == "ndjson":
        print_ndjson(rows)
    else:
        print_leaderboard_table(rows)
    return 0


def main() -> int:
    # [FUNCTION DOC] Program entry point: parse arguments and dispatch to the subcommand.
    # Behavior: builds the parser, parses argv, runs run_search or run_leaderboard by command name;
    #   an unknown command triggers a parser usage error.
    # Inputs: none (reads command-line arguments).
    # Outputs: the subcommand's exit code; 1 on handled errors.
    # Exceptions: catches HfApiError and ValueError, printing "Error: <message>" to stderr and
    #   returning 1. Argparse errors exit the process with code 2.
    # Side effects: as the subcommands; stderr write on error.
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "search":
            return run_search(args)
        if args.command == "leaderboard":
            return run_leaderboard(args)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except HfApiError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
