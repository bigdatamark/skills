# [MODULE DOC] CLI runner for lighteval evaluations on local GPU via vLLM or accelerate.
# Behavior: parses CLI arguments and runs "lighteval vllm ..." or "lighteval accelerate ..."
# as a subprocess for the given model and task specification.
# The leading uv inline-script metadata comment block declares the Python version and deps.
# CLI arguments: --model (required); --tasks (required, "suite|task|num_fewshot", comma
# separated for several); --backend ("vllm" default or "accelerate"); --output-dir (optional);
# --max-samples (int, optional); --batch-size (int, default 1); --tensor-parallel-size (int,
# default 1, vllm only); --gpu-memory-utilization (float, default 0.8, vllm only); --dtype
# (auto, float16, bfloat16, float32; default auto, mapped to bfloat16 for accelerate);
# --trust-remote-code (flag); --use-chat-template (flag); --system-prompt (optional string).
# Env vars: reads HF_TOKEN and defaults HUGGING_FACE_HUB_TOKEN and HF_HUB_TOKEN to it.
# Top-level executed code: calls main when run as a script. Constants: none.
# Outputs: prints the command and a completion message; exits with the subprocess exit code
# on failure; result files are written by lighteval to the output directory if given.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lighteval[accelerate,vllm]>=0.6.0",
#     "torch>=2.0.0",
#     "transformers>=4.40.0",
#     "accelerate>=0.30.0",
#     "vllm>=0.4.0",
# ]
# ///

"""
Entry point script for running lighteval evaluations with local GPU backends.

This script runs evaluations using vLLM or accelerate on custom HuggingFace models.
It is separate from inference provider scripts and evaluates models directly on local hardware.

Usage (standalone):
    uv run scripts/lighteval_vllm_uv.py --model "meta-llama/Llama-3.2-1B" --tasks "leaderboard|mmlu|5"

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


def run_lighteval_vllm(
    model_id: str,
    tasks: str,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
) -> None:
    # [FUNCTION DOC] Run lighteval with the vLLM backend.
    # Behavior: 1) set up token env vars; 2) build "lighteval vllm <model_id> <tasks> --batch-size
    # <n> --tensor-parallel-size <n> --gpu-memory-utilization <f> --dtype <dtype>"; 3) append
    # --output-dir, --max-samples, --trust-remote-code, --use-chat-template and --system-prompt when
    # set/truthy; 4) print and run the command.
    # Inputs: model_id (string); tasks (string task spec); output_dir (optional string, default
    # none); max_samples (optional int, default none); batch_size (int, default 1);
    # tensor_parallel_size (int, default 1); gpu_memory_utilization (float, default 0.8); dtype
    # (string, default "auto"); trust_remote_code (bool, default false); use_chat_template (bool,
    # default false); system_prompt (optional string, default none).
    # Outputs: prints the command and "Evaluation complete." on success; returns nothing.
    # Exceptions: on a non-zero subprocess exit, prints the code to stderr and exits with it.
    # Side effects: env var mutation, launches a subprocess (GPU work, downloads, result files).
    # KNOWN ISSUE: recent lighteval CLIs take model settings as a single model-args string rather
    # than separate flags such as batch-size, dtype and tensor-parallel-size, so this command may
    # be rejected depending on the installed lighteval version.
    """
    Run lighteval with vLLM backend for efficient GPU inference.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B")
        tasks: Task specification (e.g., "leaderboard|mmlu|5" or "lighteval|hellaswag|0")
        output_dir: Directory for evaluation results
        max_samples: Limit number of samples per task
        batch_size: Batch size for evaluation
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory fraction to use (0.0-1.0)
        dtype: Data type for model weights (auto, float16, bfloat16)
        trust_remote_code: Allow executing remote code from model repo
        use_chat_template: Apply chat template for conversational models
        system_prompt: System prompt for chat models
    """
    setup_environment()

    # Build lighteval vllm command
    cmd = [
        "lighteval",
        "vllm",
        model_id,
        tasks,
        "--batch-size", str(batch_size),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if use_chat_template:
        cmd.append("--use-chat-template")

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("Evaluation complete.")
    except subprocess.CalledProcessError as exc:
        print(f"Evaluation failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def run_lighteval_accelerate(
    model_id: str,
    tasks: str,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
) -> None:
    # [FUNCTION DOC] Run lighteval with the accelerate (Transformers) backend.
    # Behavior: 1) set up token env vars; 2) build "lighteval accelerate <model_id> <tasks>
    # --batch-size <n> --dtype <dtype>"; 3) append --output-dir, --max-samples,
    # --trust-remote-code, --use-chat-template and --system-prompt when set/truthy; 4) print and
    # run the command.
    # Inputs: model_id (string); tasks (string); output_dir (optional string, default none);
    # max_samples (optional int, default none); batch_size (int, default 1); dtype (string,
    # default "bfloat16"); trust_remote_code (bool, default false); use_chat_template (bool,
    # default false); system_prompt (optional string, default none).
    # Outputs: prints the command and "Evaluation complete." on success; returns nothing.
    # Exceptions: on a non-zero subprocess exit, prints the code to stderr and exits with it.
    # Side effects: env var mutation, launches a subprocess (GPU work, downloads, result files).
    # KNOWN ISSUE: same flag-style concern as the vLLM runner for recent lighteval versions.
    """
    Run lighteval with accelerate backend for multi-GPU distributed inference.

    Use this backend when vLLM is not available or for models not supported by vLLM.

    Args:
        model_id: HuggingFace model ID
        tasks: Task specification
        output_dir: Directory for evaluation results
        max_samples: Limit number of samples per task
        batch_size: Batch size for evaluation
        dtype: Data type for model weights
        trust_remote_code: Allow executing remote code
        use_chat_template: Apply chat template
        system_prompt: System prompt for chat models
    """
    setup_environment()

    # Build lighteval accelerate command
    cmd = [
        "lighteval",
        "accelerate",
        model_id,
        tasks,
        "--batch-size", str(batch_size),
        "--dtype", dtype,
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if use_chat_template:
        cmd.append("--use-chat-template")

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("Evaluation complete.")
    except subprocess.CalledProcessError as exc:
        print(f"Evaluation failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def main() -> None:
    # [FUNCTION DOC] Parse CLI arguments and dispatch to the vLLM or accelerate lighteval runner.
    # Behavior: defines the arguments listed in the module doc, parses them, and calls the vLLM
    # runner when backend is "vllm"; otherwise calls the accelerate runner, replacing a dtype of
    # "auto" with "bfloat16" and not forwarding the vLLM-only options.
    # Inputs: none directly; reads command-line arguments.
    # Outputs: none beyond what the runners print.
    # Exceptions: argparse exits with code 2 on invalid arguments; runner exits propagate.
    # Side effects: those of the called runner.
    parser = argparse.ArgumentParser(
        description="Run lighteval evaluations with vLLM or accelerate backend on custom HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MMLU evaluation with vLLM
  uv run scripts/lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B --tasks "leaderboard|mmlu|5"

  # Run with accelerate backend instead of vLLM
  uv run scripts/lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B --tasks "leaderboard|mmlu|5" --backend accelerate

  # Run with chat template for instruction-tuned models
  uv run scripts/lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B-Instruct --tasks "leaderboard|mmlu|5" --use-chat-template

  # Run with limited samples for testing
  uv run scripts/lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B --tasks "leaderboard|mmlu|5" --max-samples 10

Task format:
  Tasks use the format: "suite|task|num_fewshot"
  - leaderboard|mmlu|5 (MMLU with 5-shot)
  - lighteval|hellaswag|0 (HellaSwag zero-shot)
  - leaderboard|gsm8k|5 (GSM8K with 5-shot)
  - Multiple tasks: "leaderboard|mmlu|5,leaderboard|gsm8k|5"
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Task specification (e.g., 'leaderboard|mmlu|5')",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "accelerate"],
        default="vllm",
        help="Inference backend to use (default: vllm)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples per task (useful for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)",
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
        "--trust-remote-code",
        action="store_true",
        help="Allow executing remote code from model repository",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Apply chat template for instruction-tuned/chat models",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt for chat models",
    )

    args = parser.parse_args()

    if args.backend == "vllm":
        run_lighteval_vllm(
            model_id=args.model,
            tasks=args.tasks,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )
    else:
        run_lighteval_accelerate(
            model_id=args.model,
            tasks=args.tasks,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            dtype=args.dtype if args.dtype != "auto" else "bfloat16",
            trust_remote_code=args.trust_remote_code,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )


if __name__ == "__main__":
    main()
