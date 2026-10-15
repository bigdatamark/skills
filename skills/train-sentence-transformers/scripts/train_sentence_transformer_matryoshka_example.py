#!/usr/bin/env python3
# [MODULE DOC] Example script: Matryoshka (MRL) training of mpnet-base so embeddings stay useful
# when truncated to smaller dimensions.
# Behavior: trains microsoft/mpnet-base on AllNLI triplets with MultipleNegativesRankingLoss
# wrapped in MatryoshkaLoss over dims 768/512/256/128/64; evaluates STS-B similarity at each
# truncated dimension plus NanoBEIR, logs a before/after VERDICT, saves and pushes the model.
# CLI arguments: --eval-only PATH (optional) loads a saved model and runs only NanoBEIR.
# Env vars: SMOKE_TEST=1 uses 50 train / 20 eval rows, one step, no Trackio, no Hub push.
# Constants: MODEL_NAME, MATRYOSHKA_DIMS, OUTPUT_DIR, RUN_NAME, SMOKE_TEST.
# Top-level executed code: reads SMOKE_TEST at import; runs main() when executed as a script.
# Side effects: downloads models and datasets, writes logs/ and models/, pushes to the Hub.
# KNOWN ISSUE: training arguments hard-code bf16 mode while autocast_ctx falls back to fp16;
# on GPUs without bfloat16 support (or CPU-only machines) training will fail to start.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=5.0",
#     "datasets>=2.19.0",
#     "accelerate>=0.26.0",
#     "trackio",
# ]
# ///
"""Matryoshka (MRL) training: train once, deploy at multiple embedding dimensions.

MatryoshkaLoss wraps a base loss and optimizes it at several truncated dimensions
simultaneously. At inference, load with `truncate_dim=<target>` to get that size
with ~95% of full-dim quality.

Typical use: train at [768, 512, 256, 128, 64], deploy at 128 for 6x smaller
index + 6x faster ANN with minimal quality loss.

Run locally:
    pip install "sentence-transformers[train]>=5.0"
    python train_sentence_transformer_matryoshka_example.py
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    NanoBEIREvaluator,
    SequentialEvaluator,
)
from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.util.similarity import SimilarityFunction


def autocast_ctx():
    # [FUNCTION DOC] HELPER: Build a mixed-precision autocast context for evaluator calls run
    # outside the trainer.
    # Behavior: if no CUDA GPU is available, return a no-op context manager. Otherwise pick bfloat16
    # when the GPU supports it, else float16, and return a CUDA autocast context with that dtype.
    # Inputs: none.
    # Outputs: a context manager (no-op or CUDA autocast).
    # Exceptions: none.
    # Side effects: none until the context is entered (then CUDA ops run in reduced precision).
    """bf16/fp16 autocast for evaluator calls outside the trainer (which has its own autocast)."""
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def log_trackio_dashboard():
    # [FUNCTION DOC] HELPER: Log the URL of the user's Trackio dashboard Space.
    # Behavior: lazily imports the Hub client, asks the Hub who the logged-in user is, and if a
    # username is returned logs an info line with https://huggingface.co/spaces/<user>/trackio.
    # Inputs: none.
    # Outputs: none (one info log line on success).
    # Exceptions: every exception (no token, network failure, missing package) is caught and
    # silently swallowed.
    # Side effects: network call to the Hugging Face Hub; logging.
    """Surface the Trackio dashboard URL so the user can watch training live."""
    try:
        from huggingface_hub import whoami

        hf_user = whoami().get("name")
        if hf_user:
            logging.info(
                f"Trackio dashboard (live training progress): https://huggingface.co/spaces/{hf_user}/trackio"
            )
    except Exception:
        pass


MODEL_NAME = "microsoft/mpnet-base"
MATRYOSHKA_DIMS = [768, 512, 256, 128, 64]
OUTPUT_DIR = "models/mpnet-matryoshka"
RUN_NAME = "mpnet-matryoshka"
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"


def setup_logging():
    # [FUNCTION DOC] HELPER: Configure root logging and matmul precision for the run.
    # Behavior: creates a "logs" directory in the working directory if missing, (re)configures root
    # logging at INFO with a timestamped format writing both to the console and to
    # logs/<RUN_NAME>.log (forcing replacement of existing handlers), raises the level of noisy
    # HTTP/Hub/file-lock loggers to WARNING, and if CUDA is available sets float32 matmul precision
    # to "high" (enables TF32 on Ampere+ GPUs).
    # Inputs: none (uses the module-level RUN_NAME constant).
    # Outputs: none.
    # Exceptions: none caught; filesystem errors creating the directory or log file propagate.
    # Side effects: creates a directory and log file, mutates global logging and torch state.
    """Configure logging + TF32. Tees to logs/{RUN_NAME}.log and silences HTTP spam."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/{RUN_NAME}.log")],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+, no quality loss


def main() -> None:
    # [FUNCTION DOC] Entry point: Matryoshka-train the model and report the result.
    # Behavior: parse the command line (unknown arguments are silently ignored), set up logging. If
    # --eval-only PATH is given, load that saved model, build a default NanoBEIR evaluator, run it once
    # under autocast and return without training.
    # Otherwise: load the base model, load the first 50,000 (or 50) AllNLI train triplets and 1,000
    # (or 20) dev triplets, build MatryoshkaLoss around MNRL, build one STS-B validation similarity
    # evaluator per Matryoshka dim (cosine, truncated) plus NanoBEIR inside a sequential evaluator
    # whose main score is the first sub-score. Run the baseline (this also prefixes metric names),
    # drive best-model selection on the largest-dim STS metric, and train 1 epoch (batch 128,
    # lr 2e-5, no-duplicates sampler).
    # Then: build the trainer, log the Trackio URL unless smoke-testing, train, re-run the evaluator
    # under autocast, compute delta = score - baseline and log a VERDICT line: WIN if delta is at
    # least 0.005, MARGINAL if non-negative, otherwise REGRESSION.
    # Also logs a hint on loading the saved model at a truncated dimension.
    # Finally save the model to <output dir>/final; in smoke-test mode stop there, otherwise push
    # the model to the Hub under the RUN_NAME repository and log its URL.
    # Inputs: none (reads the command line and the SMOKE_TEST environment variable).
    # Outputs: log lines (baseline/final metrics, VERDICT), saved model directory, checkpoints in the
    # output directory, optional Hub repository. Returns nothing.
    # Exceptions: any failure during the Hub push is caught and logged with a traceback (not
    # re-raised); all other errors propagate.
    # Side effects: network downloads of models/datasets, GPU use, files under the output directory
    # and logs/, Trackio reporting (unless smoke test), Hub upload (unless smoke test).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Skip training; load this saved model and run only the evaluator."
    )
    cli, _ = parser.parse_known_args()

    setup_logging()

    if cli.eval_only:
        logging.info(f"Eval-only mode: loading model from {cli.eval_only}")
        model = SentenceTransformer(cli.eval_only)
        evaluator = NanoBEIREvaluator()
        with autocast_ctx():
            evaluator(model)
        return

    model = SentenceTransformer(MODEL_NAME)

    train_size = 50 if SMOKE_TEST else 50_000
    eval_size = 20 if SMOKE_TEST else 1_000
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed dataset; will run max_steps=1 and skip Hub push")
    train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(train_size))
    eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev").select(range(eval_size))

    inner_loss = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, inner_loss, matryoshka_dims=MATRYOSHKA_DIMS)

    stsb = load_dataset("sentence-transformers/stsb", split="validation")
    per_dim_evaluators = [
        EmbeddingSimilarityEvaluator(
            sentences1=stsb["sentence1"],
            sentences2=stsb["sentence2"],
            scores=stsb["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-dev-{dim}",
            truncate_dim=dim,
        )
        for dim in MATRYOSHKA_DIMS
    ]
    evaluator = SequentialEvaluator(
        [*per_dim_evaluators, NanoBEIREvaluator()],
        main_score_function=lambda scores: scores[0],
    )
    logging.info("Baseline evaluation (before training):")
    with autocast_ctx():
        # Must run before deriving metric_key: each sub-evaluator mutates its primary_metric to add the name_ prefix.
        baseline_result = evaluator(model)
    # Drive on the first per-dim evaluator's metric (matches main_score_function above).
    metric_key = f"eval_{per_dim_evaluators[0].primary_metric}"
    baseline_eval = baseline_result[per_dim_evaluators[0].primary_metric]

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0.1,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.01,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        report_to="none" if SMOKE_TEST else "trackio",
        run_name=RUN_NAME,
        seed=12,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    if not SMOKE_TEST:
        log_trackio_dashboard()
    trainer.train()

    logging.info("Post-training evaluation:")
    with autocast_ctx():
        score = evaluator(model)[per_dim_evaluators[0].primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")

    final_dir = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_dir)
    logging.info(f"Saved to {final_dir}")
    logging.info(f"To use at a specific dimension, load with: SentenceTransformer({final_dir!r}, truncate_dim=128)")

    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: skipping Hub push")
        return

    try:
        commit_url = model.push_to_hub(RUN_NAME)
        logging.info(f"Pushed model to {commit_url.rsplit('/commit/', 1)[0]}")
    except Exception:
        import traceback

        logging.error(f"Hub push failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
