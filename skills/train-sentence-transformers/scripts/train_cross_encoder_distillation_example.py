#!/usr/bin/env python3
# [MODULE DOC] Example: distill a small CrossEncoder reranker with MarginMSE on MS MARCO.
# Trains MiniLM (single logit, identity activation) on precomputed teacher score margins for
# (query, positive, negative) rows, evaluates with NanoBEIR reranking (msmarco, nfcorpus, nq),
# keeps the best checkpoint, early-stops, logs a verdict and optionally pushes to the Hub.
# Constants: MODEL_NAME (student base), DATASET_NAME / DATASET_SUBSET (MS MARCO margin-MSE
# subset), TRAIN_SIZE 100000 and EVAL_SIZE 5000 rows, OUTPUT_DIR, RUN_NAME, DATA_CACHE (directory
# holding the id-resolved text dataset), SMOKE_TEST flag.
# Env vars: SMOKE_TEST - when exactly "1", trims data, runs a single optimizer step, disables
# Trackio reporting and skips the Hub push. Hub credentials (e.g. HF_TOKEN) are used implicitly.
# CLI arguments: --eval-only PATH - skip training, load the saved model at PATH and run only the
# evaluator. Unknown arguments are silently ignored (so launchers can pass extra flags).
# Top-level executed code: imports and constant definitions only; main() runs when executed as a
# script.
# Outputs: logs to console and logs/<RUN_NAME>.log; checkpoints under OUTPUT_DIR; final model in
# OUTPUT_DIR/final; on non-smoke runs a public Hub repo named RUN_NAME under the logged-in user.
# Also writes the resolved dataset cache to DATA_CACHE.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=5.0",
#     "datasets>=2.19.0",
#     "accelerate>=0.26.0",
#     "trackio",
# ]
# ///
"""CrossEncoder distillation: train a small reranker from a stronger one's scores.

Uses MarginMSELoss on `(query, positive, negative, score_diff)` where
`score_diff = teacher(q, pos) - teacher(q, neg)`. Workhorse of MS MARCO-style
distillation: the student learns the teacher's score gaps without ever calling
the teacher at training time (precomputed score diffs).

For Embedding-MSE / Listwise-KL variants and the broader pattern map, see the
sibling `train_sentence_transformer_distillation_example.py` docstring.

CRITICAL: `activation_fn=nn.Identity()` is mandatory. The default `Sigmoid` (with
`num_labels=1`) saturates raw logits >5 to ~1.0 inside `predict()` at eval time,
silently collapsing eval ranking (training loss stays healthy while nDCG drops
from e.g. ~0.59 to ~0.14). See `../references/troubleshooting.md` ("CrossEncoder
eval nDCG crashes after distillation / listwise / pairwise training").

Data: this script uses `sentence-transformers/msmarco` (`bert-ensemble-margin-mse`
subset), which has precomputed teacher score diffs per (q, pos, neg) row. To
distill from your own teacher, replace the dataset loading with a one-time
teacher pass over your (q, pos, neg) triples and store `score_diff = teacher_pos
- teacher_neg` as the label.

Run locally:
    pip install "sentence-transformers[train]>=5.0"
    python train_cross_encoder_distillation_example.py

Multi-GPU:
    accelerate launch train_cross_encoder_distillation_example.py

Hugging Face Jobs: paste this file's contents as the `script` in hf_jobs(...).
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import EarlyStoppingCallback

from sentence_transformers import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import MarginMSELoss


def autocast_ctx():
    # [FUNCTION DOC] HELPER: Build a mixed-precision autocast context for out-of-trainer evals.
    # Behavior: if no CUDA device is available, returns a no-op context manager. Otherwise picks
    # bfloat16 when the GPU supports it, else float16, and returns a CUDA autocast context for it.
    # Inputs: none.
    # Outputs: a context manager (no-op or CUDA autocast).
    # Exceptions: none raised deliberately.
    # Side effects: none (queries CUDA availability/capabilities only).
    """bf16/fp16 autocast for evaluator calls outside the trainer (which has its own autocast)."""
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def log_trackio_dashboard():
    # [FUNCTION DOC] HELPER: Log the URL of the user's Trackio dashboard Space.
    # Behavior: imports the Hugging Face Hub "whoami" call, fetches the authenticated user name and,
    # if present, logs an info line pointing to https://huggingface.co/spaces/<user>/trackio.
    # Inputs: none.
    # Outputs: none (one log line on success).
    # Exceptions: every exception (missing package, not logged in, network failure) is caught and
    # silently ignored.
    # Side effects: network call to the Hugging Face Hub; writes to the logging system.
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


MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
DATASET_NAME = "sentence-transformers/msmarco"
DATASET_SUBSET = "bert-ensemble-margin-mse"
TRAIN_SIZE = 100_000
EVAL_SIZE = 5_000
OUTPUT_DIR = "models/minilm-msmarco-distilled"
RUN_NAME = "minilm-msmarco-distilled"
DATA_CACHE = f"data/{RUN_NAME}-resolved"
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"


def setup_logging():
    # [FUNCTION DOC] HELPER: Configure logging for the run and enable TF32 matmuls on CUDA.
    # Behavior: creates the "logs" directory if missing; forcibly reconfigures the root logger at INFO
    # level with a timestamped format, writing both to the console and to logs/<RUN_NAME>.log; sets
    # noisy HTTP/hub/filesystem library loggers to WARNING; if CUDA is available, sets float32 matmul
    # precision to "high" (allows TF32).
    # Inputs: none (uses the module constant RUN_NAME).
    # Outputs: none.
    # Exceptions: none caught; filesystem errors creating the directory or log file propagate.
    # Side effects: creates a directory and log file; replaces global logging handlers; changes the
    # global torch matmul precision setting.
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
    # [FUNCTION DOC] HELPER: Load the MS MARCO margin rows with ids resolved to text, cached on disk.
    # Behavior: if DATA_CACHE exists, loads and returns it. Otherwise loads the corpus and queries
    # subsets, builds id-to-text lookups, takes the first TRAIN_SIZE + EVAL_SIZE rows of the margin
    # subset, maps them in batches to query/positive/negative text plus the score, drops the id
    # columns, saves the result to DATA_CACHE and returns it.
    # Inputs: none (uses module constants).
    # Outputs: a dataset with columns query, positive, negative, score.
    # Exceptions: none caught; an unknown id raises a lookup error; download errors propagate.
    # Side effects: network downloads; writes the cache directory; logs progress.
    # KNOWN ISSUE: the cache is not keyed by size, so changing TRAIN_SIZE/EVAL_SIZE silently reuses a
    # stale cache until DATA_CACHE is deleted.
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
        # [FUNCTION DOC] HELPER: Batched mapper converting query/passage ids to their texts.
        # Behavior: looks up each query id in the query lookup and each positive/negative id in the
        # corpus lookup (closed over from the enclosing function); passes scores through unchanged.
        # Inputs: batch - a mapping of column name to list, with query_id, positive_id, negative_id, score.
        # Outputs: a mapping with lists for query, positive, negative and score.
        # Exceptions: a missing id raises a lookup error.
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
    # [FUNCTION DOC] Train (or just evaluate) the distilled CrossEncoder reranker.
    # Behavior: parses --eval-only (ignoring unknown args) and sets up logging. In eval-only mode loads
    # the given CrossEncoder, runs the NanoBEIR reranking evaluator under autocast and returns.
    # Otherwise builds the student CrossEncoder with one label and an identity activation plus model
    # card data; loads the resolved dataset (trimmed to 70 rows for smoke tests); splits off an eval
    # set (20 rows for smoke, else EVAL_SIZE, seed 12); creates MarginMSE loss and the NanoBEIR
    # evaluator; runs a baseline evaluation (which also prefixes the primary metric name) and derives
    # the "eval_"-prefixed metric key; configures training args (1 epoch, max 1 step on smoke,
    # batch 32, lr 8e-6, linear schedule with 10% warmup, bf16, eval/save every 10% of steps, keep 2
    # checkpoints, best model by the metric, Trackio unless smoke, seed 12); trains with early
    # stopping (patience 3).
    # Post-training: re-runs the evaluator, computes delta versus baseline and logs a VERDICT line
    # (WIN if delta is at least 0.005, MARGINAL if non-negative, else REGRESSION); saves the model
    # to OUTPUT_DIR/final; unless smoke testing, pushes it to the Hub under RUN_NAME and logs the repo
    # URL (commit URL with the commit part stripped).
    # Inputs: none directly; reads the command line and the SMOKE_TEST env var.
    # Outputs: none returned; logs, checkpoint and final model directories, optional Hub repo.
    # Exceptions: any error during the Hub push is caught and logged with its traceback; all other
    # errors (download, out-of-memory, missing bf16 support) propagate.
    # Side effects: network downloads of models/datasets, GPU usage, filesystem writes, Hub upload,
    # Trackio reporting (non-smoke).
    # KNOWN ISSUE: bf16 training is hard-enabled, so training fails on CPU-only machines or GPUs without
    # bfloat16 support, even though the eval autocast helper handles those cases.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Skip training; load this saved model and run only the evaluator."
    )
    cli, _ = parser.parse_known_args()

    setup_logging()

    if cli.eval_only:
        logging.info(f"Eval-only mode: loading model from {cli.eval_only}")
        model = CrossEncoder(cli.eval_only)
        evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
        with autocast_ctx():
            evaluator(model)
        return

    logging.info(f"Loading base model: {MODEL_NAME}")
    model = CrossEncoder(
        MODEL_NAME,
        num_labels=1,
        activation_fn=nn.Identity(),  # Mandatory for distillation losses.
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{MODEL_NAME.split('/')[-1]} reranker distilled from MS MARCO ensemble",
        ),
    )

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

    loss = MarginMSELoss(model)

    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
    logging.info("Baseline evaluation:")
    with autocast_ctx():
        # Must run before deriving metric_key: evaluator(model) mutates primary_metric to add the name_ prefix.
        baseline_eval = evaluator(model)[evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"

    args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=8e-6,  # Lower than typical 2e-5. Distillation regression converges faster
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

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # CE rerankers peak mid-training
    )
    if not SMOKE_TEST:
        log_trackio_dashboard()
    trainer.train()

    logging.info("Post-training evaluation:")
    with autocast_ctx():
        score = evaluator(model)[evaluator.primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")

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
