#!/usr/bin/env python3
# [MODULE DOC] Minimal reference CLI that lists Hugging Face Hub models and prints the raw JSON.
# Purpose: smallest possible baseline for building HF API tools; uses only the standard library.
# CLI arguments: one optional positional "limit" (a string of digits, default "3") giving how many
#   models to request; "--help" as the first argument prints usage and exits 0.
# Env vars: HF_TOKEN, if set, is sent as a bearer token in the Authorization header.
# Constants: none. Top-level executed code: when run as a script, calls main and exits the process
#   with its return value.
# Side effects: one HTTPS GET to the huggingface.co models API; writes to stdout/stderr.
"""
Ultra-simple Hugging Face API example (Python).

Fetches a small list of models from the HF API and prints raw JSON.
Uses HF_TOKEN for auth if the environment variable is set.
"""

from __future__ import annotations

import os
import sys
import urllib.request


def show_help() -> None:
    # [FUNCTION DOC] HELPER: Print the usage/help text for this script.
    # Behavior: prints a fixed multi-line block with usage, description and examples.
    # Inputs: none.
    # Outputs: returns nothing; writes the help text to stdout.
    # Exceptions: none.
    # Side effects: stdout write.
    print(
        """Ultra-simple Hugging Face API example (Python)

Usage:
  baseline_hf_api.py [limit]
  baseline_hf_api.py --help

Description:
  Fetches a small list of models from the HF API and prints raw JSON.
  Uses HF_TOKEN for auth if the environment variable is set.

Examples:
  baseline_hf_api.py
  baseline_hf_api.py 5
  HF_TOKEN=your_token baseline_hf_api.py 10
"""
    )


def main() -> int:
    # [FUNCTION DOC] Entry point: parse argv, fetch the model list and print the raw response.
    # Behavior: 1) if the first argument is "--help", print help and return 0. 2) take the first
    #   argument as the limit (default "3"); if it is not all digits, print an error to stderr and
    #   return 1. 3) build an Authorization bearer header from HF_TOKEN when set. 4) GET
    #   https://huggingface.co/api/models?limit=<limit> and write the decoded UTF-8 body to stdout
    #   unchanged (no trailing newline added). 5) return 0.
    # Inputs: none directly; reads command-line arguments and the HF_TOKEN env var.
    # Outputs: exit code 0 on success or help, 1 on a non-numeric limit; JSON text on stdout.
    # Exceptions: network/HTTP errors from the request are not caught and propagate as a traceback.
    # Side effects: network request, env var read, stdout/stderr writes.
    # KNOWN ISSUE: no timeout on the request and no handling of HTTP or connection errors, so
    #   failures surface as an uncaught traceback rather than a clean error message.
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
        return 0

    limit = sys.argv[1] if len(sys.argv) > 1 else "3"
    if not limit.isdigit():
        print("Error: limit must be a number", file=sys.stderr)
        return 1

    token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/api/models?limit={limit}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        sys.stdout.write(resp.read().decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
