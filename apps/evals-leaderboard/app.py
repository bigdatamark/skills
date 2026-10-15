#!/usr/bin/env python3
# [MODULE DOC] Gradio web app that displays the evaluations leaderboard (model benchmark scores).
# Behavior: at import time builds a Gradio Blocks page with a title/intro markdown, a status
# markdown box, a read-only table (Model, Benchmark, Score, Source) and a footer of links. On
# page load it calls the refresh handler, which downloads leaderboard rows and metadata from the
# public Hugging Face dataset and fills the status box and table. Data is produced separately
# by collect_evals.py.
# Constants: TABLE_HEADERS (four column titles), TABLE_DATATYPES (markdown, text, number,
# markdown per column), DATASET_REPO (the dataset id hf-skills/evals-leaderboard),
# LEADERBOARD_URL and METADATA_URL (raw file URLs for data/leaderboard.jsonl and
# data/metadata.json on the main branch of that dataset).
# Top-level executed code: the Gradio Blocks layout is built on import (module-level 'demo');
# when run as a script, the app server is launched.
# CLI arguments: none. Env vars: none read directly (Gradio may honor its own env vars).
# Side effects: network requests to huggingface.co on each page load; starts a web server when
# run as a script.
"""
Evals Leaderboard - Gradio app for displaying model evaluation scores.

Reads leaderboard data from the hf-skills/evals-leaderboard dataset.
Run collect_evals.py separately to update the dataset.

Usage:
    python app.py
"""

from __future__ import annotations

import json

import gradio as gr
import requests

TABLE_HEADERS = [
    "Model",
    "Benchmark",
    "Score",
    "Source",
]

TABLE_DATATYPES = [
    "markdown",
    "text",
    "number",
    "markdown",
]


DATASET_REPO = "hf-skills/evals-leaderboard"
LEADERBOARD_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/leaderboard.jsonl"
METADATA_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/metadata.json"


def format_model_link(model_id: str) -> str:
    # [FUNCTION DOC] HELPER: Build a markdown link to a model's Hugging Face page.
    # Behavior: returns markdown link text whose label is the model id and whose target is
    # https://huggingface.co/ followed by the model id.
    # Inputs: model_id (string) - repository id such as owner/name.
    # Outputs: the markdown link string.
    # Exceptions: none.
    # Side effects: none.
    """Format model ID as a clickable link."""
    return f"[{model_id}](https://huggingface.co/{model_id})"


def format_source_link(source_type: str, contributor: str, source_url: str) -> str:
    # [FUNCTION DOC] HELPER: Build the markdown text describing where a score came from.
    # Behavior: returns '<source type> by [<contributor>](<source url>)'.
    # Inputs: source_type (string, e.g. model-card or pull-request); contributor (string, user or
    # org name shown as link text); source_url (string, link target).
    # Outputs: the formatted markdown string.
    # Exceptions: none.
    # Side effects: none.
    """Format source as a clickable link."""
    return f"{source_type} by [{contributor}]({source_url})"


def fetch_leaderboard() -> tuple[list[dict], dict]:
    # [FUNCTION DOC] Download the leaderboard rows and the metadata file from the dataset.
    # Behavior: GETs LEADERBOARD_URL (30 s timeout), fails on HTTP error status, strips the body,
    # splits it into lines and parses each non-empty line as JSON into a list of records. Then
    # GETs METADATA_URL (30 s timeout), fails on HTTP error, and parses the body as JSON.
    # Inputs: none.
    # Outputs: a pair of (list of leaderboard record mappings, metadata mapping).
    # Exceptions: propagates request errors (connection, timeout, HTTP error status) and JSON
    # decoding errors.
    # Side effects: two network requests to huggingface.co.
    """Fetch leaderboard data from the HF dataset."""
    # Fetch leaderboard JSONL
    resp = requests.get(LEADERBOARD_URL, timeout=30)
    resp.raise_for_status()
    leaderboard = [json.loads(line) for line in resp.text.strip().split("\n") if line]

    # Fetch metadata
    resp = requests.get(METADATA_URL, timeout=30)
    resp.raise_for_status()
    metadata = resp.json()

    return leaderboard, metadata


def refresh_handler() -> tuple[str, list[list]]:
    # [FUNCTION DOC] Load data and produce the status markdown and table rows for the UI.
    # Behavior: calls fetch_leaderboard; for each record builds a row of [model link, benchmark
    # label, score, source description] in the order received (no sorting here); builds a
    # four-line markdown status with a dataset link, metadata generated_at, models_with_scores and
    # total_entries (falling back to 'unknown' or to the number of rows when missing).
    # Inputs: none.
    # Outputs: a pair of (status markdown string, list of rows). On any failure returns an error
    # message string prefixed with a cross mark and an empty row list.
    # Exceptions: catches and swallows every exception (including missing record keys), reporting
    # it only in the returned status text.
    # Side effects: network requests via fetch_leaderboard.
    # KNOWN ISSUE: scores collected from different sources may mix units (percent vs fraction), so
    # the Score column is not necessarily comparable across rows.
    """Refresh the leaderboard data from the dataset."""
    try:
        leaderboard, metadata = fetch_leaderboard()

        # Build table rows
        rows = []
        for entry in leaderboard:
            rows.append(
                [
                    format_model_link(entry["model_id"]),
                    entry["benchmark"],
                    entry["score"],
                    format_source_link(
                        entry["source_type"],
                        entry["contributor"],
                        entry["source_url"],
                    ),
                ]
            )

        status = "\n".join(
            [
                f"**Data from:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})",
                f"**Last updated:** {metadata.get('generated_at', 'unknown')}",
                f"**Models with scores:** {metadata.get('models_with_scores', 'unknown')}",
                f"**Total entries:** {metadata.get('total_entries', len(leaderboard))}",
            ]
        )

        return status, rows

    except Exception as e:
        return f"❌ Failed to load leaderboard: {e}", []


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 📊 HF Evaluation Leaderboard
        
        Shows MMLU, BigCodeBench, and ARC MC scores pulled from model-index
        metadata or their pull requests for trending text-generation models.
        """
    )

    status_box = gr.Markdown("Loading leaderboard...")

    leaderboard_table = gr.Dataframe(
        headers=TABLE_HEADERS,
        datatype=TABLE_DATATYPES,
        interactive=False,
        wrap=True,
    )

    demo.load(
        refresh_handler,
        outputs=[status_box, leaderboard_table],
    )

    gr.Markdown(
        f"""
        ---
        
        **Links:**
        - [Dataset: {DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})
        - [GitHub Repository](https://github.com/huggingface/skills)
        """
    )


if __name__ == "__main__":
    demo.launch()
