#!/usr/bin/env python3
# [MODULE DOC] Example: train a pointwise CrossEncoder reranker with BCE on mined GooAQ pairs.
# Mines hard negatives for GooAQ question/answer pairs with a static retriever into labeled pairs,
# trains MiniLM with class-weighted binary cross-entropy, evaluates with NanoBEIR reranking
# (msmarco, nfcorpus, nq), early-stops on the best checkpoint, logs a verdict and optionally
# pushes to the Hub.
# Constants: MODEL_NAME, DATASET_NAME, RETRIEVER_NAME (mining model), TRAIN_SIZE 100000 and
# EVAL_SIZE 1000, NUM_NEGATIVES 5, OUTPUT_DIR, RUN_NAME, HARD_NEG_CACHE (mined dataset directory;
# delete to re-mine), SMOKE_TEST flag.
# Env vars: SMOKE_TEST - when exactly "1", trims data, runs a single optimizer step, disables
# Trackio reporting and skips the Hub push. Hub credentials (e.g. HF_TOKEN) are used implicitly.
# CLI arguments: --eval-only PATH - skip training, load the saved model at PATH and run only the
# evaluator. Unknown arguments are silently ignored (so launchers can pass extra flags).
# Top-level executed code: imports and constant definitions only; main() runs when executed as a
# script.
# Outputs: logs to console and logs/<RUN_NAME>.log; checkpoints under OUTPUT_DIR; final model in
# OUTPUT_DIR/final; on non-smoke runs a public Hub repo named RUN_NAME under the logged-in user.
# Also writes the mined dataset cache to HARD_NEG_CACHE.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=5.0",
#     "datasets>=2.19.0",
#     "accelerate>=0.26.0",
#     "trackio",
# ]
# ///
"""Production-ready cross-encoder (reranker) training template.

Demonstrates:
- BinaryCrossEntropyLoss on (query, passage, label) pointwise data
- Hard-negative mining (`mine_hard_negatives` with `output_format="labeled-pair"`)
  to produce the labeled training data BCE needs, starting from (question, answer)
  pairs
- `pos_weight=num_negatives` to offset the positive/negative imbalance
- CrossEncoderNanoBEIREvaluator for retrieval reranking metrics
- load_best_model_at_end on the retrieval metric
- Auto model card + optional Hub push

Run locally:
    pip install "sentence-transformers[train]>=5.0"
    python train_cross_encoder_example.py

Multi-GPU:
    accelerate launch train_cross_encoder_example.py

Hugging Face Jobs: paste this file's contents as the `script` in hf_jobs(...).
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset, load_from_disk
from transformers import EarlyStoppingCallback

from sentence_transformers import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    SentenceTransformer,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.util import mine_hard_negatives


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
DATASET_NAME = "sentence-transformers/gooaq"
RETRIEVER_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
TRAIN_SIZE = 100_000
EVAL_SIZE = 1_000
NUM_NEGATIVES = 5
OUTPUT_DIR = "models/minilm-gooaq-ce"
RUN_NAME = "minilm-gooaq-ce"
HARD_NEG_CACHE = f"data/{RUN_NAME}-hard-negatives"  # delete this dir to remine
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
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+, no quality loss


def main() -> None:
    # [FUNCTION DOC] Train (or just evaluate) the BCE CrossEncoder reranker.
    # Behavior: parses --eval-only (ignoring unknown args) and sets up logging. In eval-only mode loads
    # the given CrossEncoder, runs the NanoBEIR evaluator under autocast and returns. Otherwise builds
    # a one-label CrossEncoder with model card data; loads the first train+eval pairs (50+20 for
    # smoke tests); loads mined labeled pairs from HARD_NEG_CACHE if present, else mines
    # NUM_NEGATIVES negatives per query from retrieval ranks 10-100 ("top" strategy, FAISS) as
    # labeled pairs, caches them and frees the retriever; splits off eval_size rows (seed 12);
    # counts positives (label above 0.5) and negatives to set the BCE positive weight to their ratio;
    # runs the baseline evaluation and derives the "eval_"-prefixed metric key; configures training
    # (1 epoch, max 1 step on smoke, batch 64, lr 2e-5, linear schedule with 10% warmup, bf16,
    # eval/save every 10%, keep 2 checkpoints, best by metric, Trackio unless smoke, seed 12); trains
    # with early stopping (patience 3).
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
    # KNOWN ISSUE: HARD_NEG_CACHE is not keyed by size or SMOKE_TEST, so a cache mined during a smoke
    # run is silently reused by a full run (and vice versa) until the directory is deleted.
    # KNOWN ISSUE: the train/eval split is over labeled-pair rows, so the same query appears in both
    # train and eval sets (eval loss is optimistic).
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
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{MODEL_NAME.split('/')[-1]} reranker finetuned on GooAQ",
        ),
    )

    logging.info(f"Loading dataset: {DATASET_NAME}")
    train_size = 50 if SMOKE_TEST else TRAIN_SIZE
    eval_size = 20 if SMOKE_TEST else EVAL_SIZE
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed dataset; will run max_steps=1 and skip Hub push")
    pairs = load_dataset(DATASET_NAME, split="train").select(range(train_size + eval_size))

    if os.path.isdir(HARD_NEG_CACHE):
        logging.info(f"Loading cached mined hard negatives from {HARD_NEG_CACHE}")
        labeled = load_from_disk(HARD_NEG_CACHE)
    else:
        logging.info(f"Mining hard negatives with {RETRIEVER_NAME}")
        retriever = SentenceTransformer(RETRIEVER_NAME)
        labeled = mine_hard_negatives(
            dataset=pairs,
            model=retriever,
            num_negatives=NUM_NEGATIVES,
            range_min=10,  # skip the top-10 (likely to contain true positives)
            range_max=100,
            sampling_strategy="top",
            output_format="labeled-pair",
            use_faiss=True,
        )
        labeled.save_to_disk(HARD_NEG_CACHE)
        logging.info(f"Saved mined dataset to {HARD_NEG_CACHE} (delete to remine)")
        del retriever
        torch.cuda.empty_cache()
    # EVAL_SIZE here counts labeled-pair rows, not distinct queries: each
    # query contributes 1 positive + NUM_NEGATIVES negatives, so e.g. 1000
    # rows is approximately 1000 / (1 + NUM_NEGATIVES) distinct queries.
    split = labeled.train_test_split(test_size=eval_size, seed=12)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logging.info(f"  train: {len(train_dataset):,} rows | eval: {len(eval_dataset):,} rows")
    logging.info(f"  columns: {train_dataset.column_names}")

    # pos_weight = negatives / positives, derived from the actual label distribution
    # so it stays correct if rows get filtered or the mining ratio drifts.
    n_pos = sum(1 for label in train_dataset["label"] if label > 0.5)
    n_neg = len(train_dataset) - n_pos
    pos_weight_value = n_neg / max(n_pos, 1)
    logging.info(f"  positives: {n_pos:,} | negatives: {n_neg:,} | pos_weight: {pos_weight_value:.2f}")
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(pos_weight_value))

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
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
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

    # EarlyStoppingCallback earns its keep for cross-encoders: CE rerankers
    # typically peak mid-training and then degrade, so stopping at the best
    # eval checkpoint is load-bearing (unlike bi-encoders, which tend to
    # plateau rather than regress).
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
