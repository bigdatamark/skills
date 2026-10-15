#!/usr/bin/env python3
# [MODULE DOC] Example script: distil a SPLADE sparse encoder from cross-encoder margin scores.
# Behavior: resolves the MS MARCO "bert-ensemble-margin-mse" subset (ids plus teacher score
# margins) to text once and caches it on disk, then trains a SparseEncoder from
# co-condenser-marco with SparseMarginMSELoss inside SpladeLoss (FLOPS regularisation, query
# weight 0.1, document weight 0.08), evaluates on SparseNanoBEIR (msmarco, nfcorpus, nq), logs a
# VERDICT with active-dimension counts, saves and pushes.
# CLI arguments: --eval-only PATH (optional) loads a saved sparse model and runs the evaluator.
# Env vars: SMOKE_TEST=1 trims resolved data to 70 rows (20 eval), one step, no Trackio/Hub.
# Constants: MODEL_NAME, DATASET_NAME, DATASET_SUBSET, TRAIN_SIZE (100,000), EVAL_SIZE (5,000),
# OUTPUT_DIR, RUN_NAME, DATA_CACHE (data/<run>-resolved), regulariser weights, SMOKE_TEST.
# Top-level executed code: reads SMOKE_TEST at import; runs main() when executed as a script.
# Side effects: downloads models/datasets, writes data/, logs/ and models/, pushes to the Hub.
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
"""SPLADE distillation from a cross-encoder teacher.

Trains a SPLADE sparse retriever to match a stronger cross-encoder's score
gaps via `SparseMarginMSELoss` wrapped in `SpladeLoss` (the FLOPS regularizer
is non-negotiable: without it embeddings collapse to dense).

Data shape: `(query, positive, negative, score_diff)` where
`score_diff = teacher(q, pos) - teacher(q, neg)`. This script uses
`sentence-transformers/msmarco` (`bert-ensemble-margin-mse` subset) which has
precomputed teacher score diffs. To distill from your own cross-encoder
teacher, run a one-time teacher pass over your (q, pos, neg) triples and
store the per-row score diff.

Why distill SPLADE from a cross-encoder: SPLADE alone is hard to train from
contrastive labels because the FLOPS regularizer fights early-training signal.
Distilling from a strong cross-encoder gives the model a dense regression
target and reaches stronger nDCG faster than MNRL-only.

Run locally:
    pip install "sentence-transformers[train]>=5.0"
    python train_sparse_encoder_distillation_example.py

Multi-GPU:
    accelerate launch train_sparse_encoder_distillation_example.py

Hugging Face Jobs: paste this file's contents as the `script` in hf_jobs(...).
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset, load_from_disk

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMarginMSELoss, SpladeLoss


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


MODEL_NAME = "Luyu/co-condenser-marco"  # MS MARCO-tuned MLM base. Very strong starting point
DATASET_NAME = "sentence-transformers/msmarco"
DATASET_SUBSET = "bert-ensemble-margin-mse"
TRAIN_SIZE = 100_000
EVAL_SIZE = 5_000
OUTPUT_DIR = "models/splade-msmarco-distilled"
RUN_NAME = "splade-msmarco-distilled"
DATA_CACHE = f"data/{RUN_NAME}-resolved"

QUERY_REGULARIZER_WEIGHT = 0.1  # higher than contrastive recipe. Distillation tolerates more sparsity pressure
DOCUMENT_REGULARIZER_WEIGHT = 0.08
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
        torch.set_float32_matmul_precision("high")


def load_resolved_dataset():
    # [FUNCTION DOC] Load query/positive/negative/score text rows, resolving ids once and caching.
    # Behavior: if DATA_CACHE directory exists, load and return the dataset saved there. Otherwise
    # load the MS MARCO corpus and queries configs into id-to-text mappings, load the first
    # TRAIN_SIZE + EVAL_SIZE rows of the margin subset, map ids to texts in batches (dropping the id
    # columns, keeping score), save the result to DATA_CACHE and return it.
    # Inputs: none (uses module constants).
    # Outputs: a dataset with query, positive, negative and score columns.
    # Exceptions: none caught; a missing id raises a key error.
    # Side effects: large network downloads (whole corpus), writes DATA_CACHE to disk; logging.
    # KNOWN ISSUE: the cache is keyed only by run name, so changing TRAIN_SIZE/EVAL_SIZE or the
    # subset reuses a stale cache; smoke-test runs still resolve and cache the full 105k rows.
    """Load (query, positive, negative, score) rows. The MSMARCO subset is keyed by
    passage_id / query_id. Resolve to text once and cache to disk so reruns skip the work."""
    if os.path.isdir(DATA_CACHE):
        logging.info(f"Loading cached resolved dataset from {DATA_CACHE}")
        return load_from_disk(DATA_CACHE)

    logging.info(f"Resolving {DATASET_NAME}/{DATASET_SUBSET} ids -> text (one-time, cached)")
    corpus_ds = load_dataset(DATASET_NAME, "corpus", split="train")
    corpus = dict(zip(corpus_ds["passage_id"], corpus_ds["passage"]))
    queries_ds = load_dataset(DATASET_NAME, "queries", split="train")
    queries = dict(zip(queries_ds["query_id"], queries_ds["query"]))
    raw = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train").select(range(TRAIN_SIZE + EVAL_SIZE))

    def id_to_text(batch):
        # [FUNCTION DOC] HELPER: Batched map function converting ids to texts.
        # Behavior: looks up each query id in the queries mapping and each positive/negative passage
        # id in the corpus mapping (both from the enclosing scope), passing scores through unchanged.
        # Inputs: batch - mapping of column name to list with query_id, positive_id, negative_id, score.
        # Outputs: mapping with lists for query, positive, negative and score.
        # Exceptions: key error if an id is absent from the lookup mappings.
        # Side effects: none.
        return {
            "query": [queries[qid] for qid in batch["query_id"]],
            "positive": [corpus[pid] for pid in batch["positive_id"]],
            "negative": [corpus[pid] for pid in batch["negative_id"]],
            "score": batch["score"],
        }

    resolved = raw.map(id_to_text, batched=True, remove_columns=["query_id", "positive_id", "negative_id"])
    resolved.save_to_disk(DATA_CACHE)
    return resolved


def main() -> None:
    # [FUNCTION DOC] Entry point: distil the SPLADE model and report the result.
    # Behavior: parse the command line (unknown arguments are silently ignored), set up logging. If
    # --eval-only PATH is given, load that saved model, build a SparseNanoBEIR evaluator (msmarco, nfcorpus, nq), run it once
    # under autocast and return without training.
    # Otherwise: load the base model as a SparseEncoder with model-card data and max sequence
    # length 256, load the resolved dataset (trim to 70 in smoke mode), split off EVAL_SIZE (or 20)
    # eval rows with seed 12, build SpladeLoss around SparseMarginMSELoss, record the baseline, and
    # train 1 epoch (batch 16, lr 2e-5, linear schedule). The final log line also reports query and
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
    model = SparseEncoder(
        MODEL_NAME,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"SPLADE from {MODEL_NAME.split('/')[-1]} distilled from MS MARCO ensemble",
        ),
    )
    model.max_seq_length = 256

    resolved = load_resolved_dataset()
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed dataset; will run max_steps=1 and skip Hub push")
        resolved = resolved.select(range(min(70, len(resolved))))
    eval_size = 20 if SMOKE_TEST else EVAL_SIZE
    split = resolved.train_test_split(test_size=eval_size, seed=12)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logging.info(f"  train: {len(train_dataset):,} rows | eval: {len(eval_dataset):,} rows")
    logging.info(f"  columns: {train_dataset.column_names}")

    loss = SpladeLoss(
        model=model,
        loss=SparseMarginMSELoss(model=model),
        query_regularizer_weight=QUERY_REGULARIZER_WEIGHT,
        document_regularizer_weight=DOCUMENT_REGULARIZER_WEIGHT,
    )

    evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
    logging.info("Baseline evaluation (fill-mask base scores near zero, confirms pipeline):")
    with autocast_ctx():
        # Must run before deriving metric_key: evaluator(model) mutates primary_metric to add the name_ prefix.
        baseline_result = evaluator(model)
        baseline_eval = baseline_result[evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"

    args = SparseEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0.1,
        lr_scheduler_type="linear",
        bf16=True,
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
