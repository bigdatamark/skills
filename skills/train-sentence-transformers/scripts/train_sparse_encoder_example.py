#!/usr/bin/env python3
# [MODULE DOC] Example script: production-style SPLADE sparse-encoder training on GooAQ.
# Behavior: trains a SparseEncoder from distilbert-base-uncased (masked-LM head) with
# SparseMultipleNegativesRankingLoss inside SpladeLoss (FLOPS regularisation, query weight 5e-5,
# document weight 3e-5), evaluates on SparseNanoBEIR (msmarco, nfcorpus, nq), keeps the best
# checkpoint, logs a VERDICT with active-dimension counts, saves and pushes.
# CLI arguments: --eval-only PATH (optional) loads a saved sparse model and runs the evaluator.
# Env vars: SMOKE_TEST=1 uses 50 train / 20 eval rows, one step, no Trackio/Hub.
# Constants: MODEL_NAME, DATASET_NAME, TRAIN_SIZE (100,000), EVAL_SIZE (1,000), OUTPUT_DIR,
# RUN_NAME, regulariser weights, SMOKE_TEST.
# Top-level executed code: reads SMOKE_TEST at import; runs main() when executed as a script.
# Side effects: downloads models/datasets, writes logs/ and models/, pushes to the Hub.
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
"""Production-ready sparse-encoder (SPLADE) training template.

Demonstrates:
- SpladeLoss wrapping SparseMultipleNegativesRankingLoss
- FLOPS regularization (`query_regularizer_weight` / `document_regularizer_weight`)
- SparseNanoBEIREvaluator for sparse retrieval metrics
- load_best_model_at_end on the retrieval metric

Base model must expose a masked-LM head. Any `AutoModelForMaskedLM`-compatible
checkpoint works (DistilBERT, BERT, MiniLM MLM variants, existing SPLADE models).

Run locally:
    pip install "sentence-transformers[train]>=5.0"
    python train_sparse_encoder_example.py

Multi-GPU:
    accelerate launch train_sparse_encoder_example.py

Hugging Face Jobs: paste this file's contents as the `script` in hf_jobs(...).
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import (
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)


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


MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "sentence-transformers/gooaq"
TRAIN_SIZE = 100_000
EVAL_SIZE = 1_000
OUTPUT_DIR = "models/distilbert-splade-gooaq"
RUN_NAME = "distilbert-splade-gooaq"

QUERY_REGULARIZER_WEIGHT = 5e-5
DOCUMENT_REGULARIZER_WEIGHT = 3e-5
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
    # [FUNCTION DOC] Entry point: train the SPLADE model and report the result.
    # Behavior: parse the command line (unknown arguments are silently ignored), set up logging. If
    # --eval-only PATH is given, load that saved model, build a SparseNanoBEIR evaluator (msmarco, nfcorpus, nq), run it once
    # under autocast and return without training.
    # Otherwise: load the base model as a SparseEncoder with model-card data, load the full GooAQ
    # train split, split off EVAL_SIZE (or 20) eval rows with seed 12, cap training at TRAIN_SIZE
    # (or 50), build SpladeLoss around sparse MNRL, record the baseline, and train 1 epoch (batch
    # 32, lr 2e-5, linear schedule, no-duplicates). The final log line also reports query and
    # corpus active dimensions found by suffix-matching the result keys ("n/a" if absent).
    # Then: build the trainer, log the Trackio URL unless smoke-testing, train, re-run the evaluator
    # under autocast, compute delta = score - baseline and log a VERDICT line: WIN if delta is at
    # least 0.005, MARGINAL if non-negative, otherwise REGRESSION.
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
        model = SparseEncoder(cli.eval_only)
        evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
        with autocast_ctx():
            evaluator(model)
        return

    logging.info(f"Loading base model: {MODEL_NAME}")
    # Prompts are optional for SPLADE: most BERT-style MLM bases don't need them.
    # If you're starting from a CSR (`Transformer + Pooling + SparseAutoEncoder`)
    # base like `tomaarsen/csr-mxbai-embed-large-v1-nq` that *was* trained with
    # prompts, mirror them here to preserve quality:
    #   prompts={"query": "Represent this sentence for similarity: ", "document": ""},
    #   default_prompt_name="document",
    model = SparseEncoder(
        MODEL_NAME,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"SPLADE from {MODEL_NAME.split('/')[-1]} trained on GooAQ",
        ),
    )

    logging.info(f"Loading dataset: {DATASET_NAME}")
    train_size = 50 if SMOKE_TEST else TRAIN_SIZE
    eval_size = 20 if SMOKE_TEST else EVAL_SIZE
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed dataset; will run max_steps=1 and skip Hub push")
    full = load_dataset(DATASET_NAME, split="train")
    split = full.train_test_split(test_size=eval_size, seed=12)
    train_dataset = split["train"].select(range(min(train_size, len(split["train"]))))
    eval_dataset = split["test"]
    logging.info(f"  train: {len(train_dataset):,} rows | eval: {len(eval_dataset):,} rows")
    logging.info(f"  columns: {train_dataset.column_names}")

    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=QUERY_REGULARIZER_WEIGHT,
        document_regularizer_weight=DOCUMENT_REGULARIZER_WEIGHT,
    )

    evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
    logging.info("Baseline evaluation:")
    with autocast_ctx():
        # Must run before deriving metric_key: evaluator(model) mutates primary_metric to add the name_ prefix.
        baseline_result = evaluator(model)
        baseline_eval = baseline_result[evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"

    args = SparseEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0.1,
        lr_scheduler_type="linear",
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

    trainer = SparseEncoderTrainer(
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
        result = evaluator(model)
        score = result[evaluator.primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    # Active-dim keys come back name-prefixed (e.g. "NanoBEIR_..._query_active_dims"). Suffix-match for compat.
    qad = next((v for k, v in result.items() if k.endswith("query_active_dims")), "n/a")
    cad = next((v for k, v in result.items() if k.endswith("corpus_active_dims")), "n/a")
    logging.info(
        f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f} "
        f"| query_active={qad} corpus_active={cad}"
    )

    final_dir = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_dir)
    logging.info(f"Saved final model to {final_dir}")

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
