# [MODULE DOC] CLI runner for inspect-ai evaluations against Hugging Face inference providers.
# Behavior: parses CLI arguments, mirrors the HF token into alternate env var names, normalizes
# the task name, resolves a working directory for task files, then runs the external
# "inspect eval" command as a subprocess with fixed options (model prefixed with
# "hf-inference-providers/", log level info, max connections 1, temperature 0.001).
# The leading uv inline-script metadata comment block declares the Python version and deps.
# CLI arguments: --model (required, Hub model ID); --task (required, inspect-ai task name, or a
# lighteval-style "suite|task|shots" string); --limit (integer, optional sample cap);
# --tasks-root (optional directory used as the subprocess working directory; defaults to the
# installed inspect_evals package directory); --sandbox (sandbox backend, default "local").
# Env vars: reads HF_TOKEN; if set, sets HUGGING_FACE_HUB_TOKEN and HF_HUB_TOKEN when absent.
# Top-level executed code: calls main when run as a script. Constants: none.
# Outputs: prints "Evaluation complete." on success; on failure prints the exit code to stderr
# and re-raises the subprocess error (non-zero exit with traceback).
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai>=0.3.0",
#     "inspect-evals",
#     "openai",
# ]
# ///

"""
Entry point script for running inspect-ai evaluations against Hugging Face inference providers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _inspect_evals_tasks_root() -> Optional[Path]:
    # [FUNCTION DOC] HELPER: Locate the directory of the installed inspect_evals package.
    # Behavior: tries to import inspect_evals and returns the parent directory of its module file.
    # Inputs: none.
    # Outputs: a filesystem path to the package directory, or nothing (None) if import fails.
    # Exceptions: any exception during import or path resolution is caught and swallowed.
    # Side effects: imports the inspect_evals package (module import side effects).
    """Return the installed inspect_evals package path if available."""
    try:
        import inspect_evals

        return Path(inspect_evals.__file__).parent
    except Exception:
        return None


def _normalize_task(task: str) -> str:
    # [FUNCTION DOC] HELPER: Reduce a lighteval-style "suite|task|shots" string to its task name.
    # Behavior: if the string contains a pipe, splits on pipes and returns the second piece when
    # there are at least two pieces and the second is non-empty; otherwise returns the input as is.
    # Inputs: task (string) - task name or pipe-delimited task specification.
    # Outputs: the task name string.
    # Exceptions: none.
    # Side effects: none.
    """Allow lighteval-style `suite|task|shots` strings by keeping the task name."""
    if "|" in task:
        parts = task.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return task


def main() -> None:
    # [FUNCTION DOC] Parse CLI arguments and run "inspect eval" for the requested model and task.
    # Behavior: 1) parse --model, --task, --limit, --tasks-root, --sandbox; 2) if HF_TOKEN is set,
    # default HUGGING_FACE_HUB_TOKEN and HF_HUB_TOKEN to it; 3) normalize the task name; 4) choose
    # the working directory: --tasks-root if given else the inspect_evals package dir, dropped if
    # it does not exist; 5) build the command "inspect eval <task> --model
    # hf-inference-providers/<model> --log-level info --max-connections 1 --temperature 0.001",
    # appending --sandbox when non-empty and --limit when truthy; 6) run it with check enabled.
    # Inputs: none directly; reads command-line arguments.
    # Outputs: prints "Evaluation complete." on success; returns nothing.
    # Exceptions: on a non-zero exit, prints the exit code (and cwd if set) to stderr and re-raises
    # the subprocess error; argparse exits with code 2 on invalid arguments.
    # Side effects: modifies process env vars, launches a subprocess, network access via the tool.
    # KNOWN ISSUE: a --limit of 0 is treated as no limit because only truthy limits are passed on.
    parser = argparse.ArgumentParser(description="Inspect-ai job runner")
    parser.add_argument("--model", required=True, help="Model ID on Hugging Face Hub")
    parser.add_argument("--task", required=True, help="inspect-ai task to execute")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument(
        "--tasks-root",
        default=None,
        help="Optional path to inspect task files. Defaults to the installed inspect_evals package.",
    )
    parser.add_argument(
        "--sandbox",
        default="local",
        help="Sandbox backend to use (default: local for HF jobs without Docker).",
    )
    args = parser.parse_args()

    # Ensure downstream libraries can read the token passed as a secret
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)

    task = _normalize_task(args.task)
    tasks_root = Path(args.tasks_root) if args.tasks_root else _inspect_evals_tasks_root()
    if tasks_root and not tasks_root.exists():
        tasks_root = None

    cmd = [
        "inspect",
        "eval",
        task,
        "--model",
        f"hf-inference-providers/{args.model}",
        "--log-level",
        "info",
        # Reduce batch size to avoid OOM errors (default is 32)
        "--max-connections",
        "1",
        # Set a small positive temperature (HF doesn't allow temperature=0)
        "--temperature",
        "0.001",
    ]

    if args.sandbox:
        cmd.extend(["--sandbox", args.sandbox])

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    try:
        subprocess.run(cmd, check=True, cwd=tasks_root)
        print("Evaluation complete.")
    except subprocess.CalledProcessError as exc:
        location = f" (cwd={tasks_root})" if tasks_root else ""
        print(f"Evaluation failed with exit code {exc.returncode}{location}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

