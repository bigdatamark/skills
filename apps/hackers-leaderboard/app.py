#!/usr/bin/env python3
# [MODULE DOC] Gradio web app that displays the hf-skills hackathon engagement leaderboard.
# Behavior: at import time builds a Gradio Blocks page with an HTML banner and title, a
# read-only table, a status markdown box and a footer of links. On page load it calls the
# refresh handler, which downloads leaderboard rows and metadata from the public Hugging Face
# dataset and fills the table and status box. Data is produced separately by collect_points.py.
# Constants: TABLE_HEADERS (Rank, Username, Points, Discussions), TABLE_DATATYPES (number,
# markdown, number), DATASET_REPO (hf-skills/hackers-leaderboard), LEADERBOARD_URL and
# METADATA_URL (raw URLs of data/leaderboard.jsonl and data/metadata.json on main).
# Top-level executed code: builds the module-level 'demo' Blocks; launches it when run as a
# script.
# CLI arguments: none. Env vars: none read directly.
# Side effects: network requests to huggingface.co per page load; web server when run.
# KNOWN ISSUE: TABLE_HEADERS has four columns but TABLE_DATATYPES and each row have only three
# values, so the Discussions column has no data/type.
# KNOWN ISSUE: the initial status text says 'Click refresh' but there is no refresh button; data
# loads automatically on page load.
"""
Hackers Leaderboard - Gradio app for displaying engagement from hf-skills org.

Reads leaderboard data from the hf-skills/hackers-leaderboard dataset.
Run collect_points.py separately to update the dataset.

Usage:
    python app.py
"""

from __future__ import annotations

import json

import gradio as gr
import requests

TABLE_HEADERS = [
    "Rank",
    "Username",
    "Points",
    "💬 Discussions",
]

TABLE_DATATYPES = [
    "number",
    "markdown",
    "number",
]


DATASET_REPO = "hf-skills/hackers-leaderboard"
LEADERBOARD_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/leaderboard.jsonl"
METADATA_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/metadata.json"


def format_username(username: str) -> str:
    # [FUNCTION DOC] HELPER: Build a markdown link to a user's Hugging Face profile.
    # Behavior: returns a link labelled with the username pointing at https://huggingface.co/<name>.
    # Inputs: username (string).
    # Outputs: markdown link string.
    # Exceptions: none.
    # Side effects: none.
    """Format username as a clickable link."""
    return f"[{username}](https://huggingface.co/{username})"


def fetch_leaderboard() -> tuple[list[dict], dict]:
    # [FUNCTION DOC] Download the leaderboard rows and the metadata file from the dataset.
    # Behavior: GETs LEADERBOARD_URL (30 s timeout), fails on HTTP error status, splits the trimmed
    # body into lines and parses each non-empty line as JSON; then GETs METADATA_URL the same way
    # and parses it as JSON.
    # Inputs: none.
    # Outputs: a pair of (list of user record mappings, metadata mapping).
    # Exceptions: propagates request, HTTP status and JSON decoding errors.
    # Side effects: two network requests.
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
    # Behavior: fetches the leaderboard; for each record (in received order, ranked from 1) builds
    # a row of [rank, username link, prs_opened]; builds a markdown status with the dataset link,
    # generated_at, total_participants (fallback: row count) and total_points (fallback: sum of
    # rows' total_points).
    # Inputs: none.
    # Outputs: pair of (status markdown, list of rows); on any failure an error message prefixed
    # with a cross mark and an empty list.
    # Exceptions: catches and swallows every exception, reporting it in the status text.
    # Side effects: network requests via fetch_leaderboard.
    # KNOWN ISSUE: the Points column shows prs_opened instead of total_points, while ranking follows
    # the stored order, which is sorted by total_points.
    """Refresh the leaderboard data from the dataset."""
    try:
        leaderboard, metadata = fetch_leaderboard()

        # Build table rows
        rows = []
        for i, entry in enumerate(leaderboard, 1):
            rows.append(
                [
                    i,
                    format_username(entry["username"]),
                    entry["prs_opened"],
                ]
            )

        status = "\n".join(
            [
                f"**Data from:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})",
                f"**Last updated:** {metadata.get('generated_at', 'unknown')}",
                f"**Participants:** {metadata.get('total_participants', len(leaderboard))}",
                f"**Total points:** {metadata.get('total_points', sum(e['total_points'] for e in leaderboard))}",
            ]
        )

        return status, rows

    except Exception as e:
        return f"❌ Failed to load leaderboard: {e}", []


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div class="subtitle">
            <img src="https://github.com/huggingface/skills/raw/main/assets/banner.png" alt="Humanity's Last Hackathon (of 2025)" width="100%">
        </div>
        <div class="leaderboard-title"><h1>🏆 Humanity's Last Hackathon Leaderboard</h1></div>
        """
    )

    leaderboard_table = gr.Dataframe(
        headers=TABLE_HEADERS,
        datatype=TABLE_DATATYPES,
        interactive=False,
        wrap=True,
    )

    status_box = gr.Markdown("Click refresh to load the leaderboard...")
    
    demo.load(
        refresh_handler,
        outputs=[status_box, leaderboard_table],
    )

    gr.Markdown(
        """
        ---
        
        **Links:**
        - [Join hf-skills](https://huggingface.co/organizations/hf-skills/share/KrqrmBxkETjvevFbfkXeezcyMbgMjjMaOp)
        - [Quest Instructions](https://github.com/huggingface/skills/tree/main/apps/quests)
        - [GitHub Repository](https://github.com/huggingface/skills)
        """
    )

if __name__ == "__main__":
    demo.launch()
