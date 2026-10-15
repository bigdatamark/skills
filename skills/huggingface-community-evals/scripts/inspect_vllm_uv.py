# [MODULE DOC] CLI runner for inspect-ai evaluations on local GPU via vLLM or HF Transformers.
# Behavior: parses CLI arguments and, depending on --backend, builds and runs an external
# "inspect eval" command with model spec "vllm/<model>" or "hf/<model>" as a subprocess.
# The leading uv inline-script metadata comment block declares the Python version and deps.
# CLI arguments: --model (required); --task (required); --backend ("vllm" default or "hf");
# --limit (int, optional); --max-connections (int; default 4 for vllm, 1 for hf);
# --temperature (float; default 0.0 for vllm, 0.001 for hf); --tensor-parallel-size (int,
# default 1, vllm only); --gpu-memory-utilization (float, default 0.8, vllm only); --dtype
# (auto, float16, bfloat16, float32; default auto); --device (default auto, hf only);
# --trust-remote-code (flag); --log-level (debug, info, warning, error; default info).
# Env vars: reads HF_TOKEN and defaults HUGGING_FACE_HUB_TOKEN and HF_HUB_TOKEN to it.
# Top-level executed code: calls main when run as a script. Constants: none.
# Outputs: prints the command and a completion message; exits with the subprocess exit code
# on failure.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai>=0.3.0",
#     "inspect-evals",
#     "vllm>=0.4.0",
#     "torch>=2.0.0",
#     "transformers>=4.40.0",
# ]
# ///

"""
Entry point script for running inspect-ai evaluations with vLLM or HuggingFace Transformers backend.

This script runs evaluations on custom HuggingFace models using local GPU inference,
separate from inference provider scripts (which use external APIs).

Usage (standalone):
    uv run scripts/inspect_vllm_uv.py --model "meta-llama/Llama-3.2-1B" --task "mmlu"

Model backends:
    - vllm: Fast inference with vLLM (recommended for large models)
    - hf: HuggingFace Transformers backend (broader model compatibility)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional


def setup_environment() -> None:
    # [FUNCTION DOC] HELPER: Mirror HF_TOKEN into the alternate Hugging Face token env var names.
    # Behavior: if HF_TOKEN is set and non-empty, sets HUGGING_FACE_HUB_TOKEN and HF_HUB_TOKEN to
    # its value only when they are not already set.
    # Inputs: none.
    # Outputs: none.
    # Exceptions: none.
    # Side effects: mutates the process environment variables.
    """Configure environment variables for HuggingFace authentication."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)


def run_inspect_vllm(
    model_id: str,
    task: str,
    limit: Optional[int] = None,
    max_connections: int = 4,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    log_level: str = "info",
) -> None:
    # [FUNCTION DOC] Run an inspect-ai evaluation using the vLLM model backend.
    # Behavior: 1) set up token env vars; 2) build "inspect eval <task> --model vllm/<model_id>
    # --log-level <log_level> --max-connections <n> --temperature <t>"; 3) append
    # --tensor-parallel-size if not 1, --gpu-memory-utilization if not 0.8, --dtype if not "auto",
    # --trust-remote-code if enabled, --limit if truthy; 4) print and run the command.
    # Inputs: model_id (string, Hub ID); task (string, inspect task); limit (optional int, sample
    # cap, default none); max_connections (int, default 4); temperature (float, default 0.0);
    # tensor_parallel_size (int, default 1); gpu_memory_utilization (float, default 0.8);
    # dtype (string, default "auto"); trust_remote_code (bool, default false);
    # log_level (string, default "info").
    # Outputs: prints "Running: <command>" and "Evaluation complete." on success; returns nothing.
    # Exceptions: on a non-zero subprocess exit, prints the code to stderr and exits the process
    # with that code.
    # Side effects: env var mutation, launches a subprocess (GPU work, model downloads).
    # KNOWN ISSUE: tensor-parallel-size, gpu-memory-utilization, dtype and trust-remote-code are
    # passed as top-level "inspect eval" flags, which inspect-ai does not define (model options go
    # through model args), so non-default values likely make the command fail; the nearby comment
    # even says model args are not used.
    # KNOWN ISSUE: a limit of 0 is treated as no limit.
    """
    Run inspect-ai evaluation with vLLM backend.

    Args:
        model_id: HuggingFace model ID
        task: inspect-ai task to execute (e.g., "mmlu", "gsm8k")
        limit: Limit number of samples to evaluate
        max_connections: Maximum concurrent connections
        temperature: Sampling temperature
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory fraction
        dtype: Data type (auto, float16, bfloat16)
        trust_remote_code: Allow remote code execution
        log_level: Logging level
    """
    setup_environment()

    model_spec = f"vllm/{model_id}"
    cmd = [
        "inspect",
        "eval",
        task,
        "--model",
        model_spec,
        "--log-level",
        log_level,
        "--max-connections",
        str(max_connections),
    ]

    # vLLM supports temperature=0 unlike HF inference providers
    cmd.extend(["--temperature", str(temperature)])

    # Older inspect-ai CLI versions do not support --model-args; rely on defaults
    # and let vLLM choose sensible settings for small models.
    if tensor_parallel_size != 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if gpu_memory_utilization != 0.8:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if dtype != "auto":
        cmd.extend(["--dtype", dtype])
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("Evaluation complete.")
    except subprocess.CalledProcessError as exc:
        print(f"Evaluation failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def run_inspect_hf(
    model_id: str,
    task: str,
    limit: Optional[int] = None,
    max_connections: int = 1,
    temperature: float = 0.001,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    log_level: str = "info",
) -> None:
    # [FUNCTION DOC] Run an inspect-ai evaluation using the Hugging Face Transformers backend.
    # Behavior: 1) set up token env vars; 2) build "inspect eval <task> --model hf/<model_id>
    # --log-level <log_level> --max-connections <n> --temperature <t>"; 3) append --device if not
    # "auto", --dtype if not "auto", --trust-remote-code if enabled, --limit if truthy;
    # 4) print and run the command.
    # Inputs: model_id (string); task (string); limit (optional int, default none);
    # max_connections (int, default 1); temperature (float, default 0.001); device (string,
    # default "auto"); dtype (string, default "auto"); trust_remote_code (bool, default false);
    # log_level (string, default "info").
    # Outputs: prints the command and "Evaluation complete." on success; returns nothing.
    # Exceptions: on a non-zero subprocess exit, prints the code to stderr and exits with it.
    # Side effects: env var mutation, launches a subprocess (model downloads, GPU/CPU work).
    # KNOWN ISSUE: device, dtype and trust-remote-code are passed as top-level "inspect eval"
    # flags that inspect-ai does not define, so non-default values likely fail.
    """
    Run inspect-ai evaluation with HuggingFace Transformers backend.

    Use this when vLLM doesn't support the model architecture.

    Args:
        model_id: HuggingFace model ID
        task: inspect-ai task to execute
        limit: Limit number of samples
        max_connections: Maximum concurrent connections (keep low for memory)
        temperature: Sampling temperature
        device: Device to use (auto, cuda, cpu)
        dtype: Data type
        trust_remote_code: Allow remote code execution
        log_level: Logging level
    """
    setup_environment()

    model_spec = f"hf/{model_id}"

    cmd = [
        "inspect",
        "eval",
        task,
        "--model",
        model_spec,
        "--log-level",
        log_level,
        "--max-connections",
        str(max_connections),
        "--temperature",
        str(temperature),
    ]

    if device != "auto":
        cmd.extend(["--device", device])
    if dtype != "auto":
        cmd.extend(["--dtype", dtype])
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("Evaluation complete.")
    except subprocess.CalledProcessError as exc:
        print(f"Evaluation failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def main() -> None:
    # [FUNCTION DOC] Parse CLI arguments and dispatch to the vLLM or HF evaluation runner.
    # Behavior: defines the arguments listed in the module doc, parses them, and calls the vLLM
    # runner when backend is "vllm" (max connections defaulting to 4, temperature to 0.0) or the
    # HF runner otherwise (max connections defaulting to 1, temperature to 0.001), forwarding the
    # remaining options relevant to each backend.
    # Inputs: none directly; reads command-line arguments.
    # Outputs: none beyond what the runners print.
    # Exceptions: argparse exits with code 2 on invalid arguments; runner exits propagate.
    # Side effects: those of the called runner.
    # KNOWN ISSUE: a --max-connections of 0 is silently replaced by the backend default.
    parser = argparse.ArgumentParser(
        description="Run inspect-ai evaluations with vLLM or HuggingFace Transformers on custom models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MMLU with vLLM backend
  uv run scripts/inspect_vllm_uv.py --model meta-llama/Llama-3.2-1B --task mmlu

  # Run with HuggingFace Transformers backend
  uv run scripts/inspect_vllm_uv.py --model meta-llama/Llama-3.2-1B --task mmlu --backend hf

  # Run with limited samples for testing
  uv run scripts/inspect_vllm_uv.py --model meta-llama/Llama-3.2-1B --task mmlu --limit 10

  # Run on multiple GPUs with tensor parallelism
  uv run scripts/inspect_vllm_uv.py --model meta-llama/Llama-3.2-70B --task mmlu --tensor-parallel-size 4

Available tasks (from inspect-evals):
  - mmlu: Massive Multitask Language Understanding
  - gsm8k: Grade School Math
  - hellaswag: Common sense reasoning
  - arc_challenge: AI2 Reasoning Challenge
  - truthfulqa: TruthfulQA benchmark
  - winogrande: Winograd Schema Challenge
  - humaneval: Code generation (HumanEval)

        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="inspect-ai task to execute (e.g., mmlu, gsm8k)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default="vllm",
        help="Model backend (default: vllm)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Maximum concurrent connections (default: 4 for vllm, 1 for hf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0 for vllm, 0.001 for hf)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM only, default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory fraction to use (vLLM only, default: 0.8)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for HF backend (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing remote code from model repository",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    if args.backend == "vllm":
        run_inspect_vllm(
            model_id=args.model,
            task=args.task,
            limit=args.limit,
            max_connections=args.max_connections or 4,
            temperature=args.temperature if args.temperature is not None else 0.0,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            log_level=args.log_level,
        )
    else:
        run_inspect_hf(
            model_id=args.model,
            task=args.task,
            limit=args.limit,
            max_connections=args.max_connections or 1,
            temperature=args.temperature if args.temperature is not None else 0.001,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
