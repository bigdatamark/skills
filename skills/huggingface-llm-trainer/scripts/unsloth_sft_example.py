# [MODULE DOC] Example script: LoRA supervised fine-tuning of a chat LLM with Unsloth + TRL.
# Behavior overview: loads a base model in 16-bit with LoRA adapters, loads a conversation
#   dataset from the Hub, applies the tokenizer chat template, optionally splits off an eval set,
#   trains with SFTTrainer masking user turns (loss only on assistant responses), then saves and
#   pushes either the adapter or a merged 16-bit model to the Hub.
# CLI arguments: --base-model (default LiquidAI/LFM2.5-1.2B-Instruct), --dataset (required),
#   --output-repo (required), --num-epochs (float), --max-steps (int), --batch-size (2),
#   --gradient-accumulation (4), --learning-rate (2e-4), --max-seq-length (2048), --lora-r (16),
#   --lora-alpha (16), --trackio-space, --run-name, --save-local (unsloth-output),
#   --eval-split (0.0), --num-samples, --seed (3407), --merge-model flag.
# Env vars: reads HF_TOKEN (falling back to "hfjob") for Hub login; sets
#   HF_HUB_ENABLE_HF_TRANSFER to "1" and TRACKIO_SPACE_ID when a Trackio space is given.
# Top-level executed code: reconfigures stdout/stderr to line buffering, configures root logging
#   at INFO with a timestamped format, creates the module logger. As a script with no arguments,
#   prints a feature/usage banner and exits 0; otherwise runs main.
# Side effects: GPU use, model/dataset downloads, local checkpoint files in the save directory,
#   Hub uploads, optional TensorBoard/Trackio logging.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl==0.22.2",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "tensorboard",
#     "transformers==4.57.3",
# ]
# ///
"""
Fine-tune LLMs using Unsloth optimizations for ~60% less VRAM and 2x faster training.

Supports epoch-based or step-based training with optional eval split.
Default model: LFM2.5-1.2B-Instruct (Liquid Foundation Model).

Epoch-based training (recommended for full datasets):
    uv run unsloth_sft_example.py \
        --dataset mlabonne/FineTome-100k \
        --num-epochs 1 \
        --eval-split 0.2 \
        --output-repo your-username/model-finetuned

Run on HF Jobs (1 epoch with eval):
    hf jobs uv run unsloth_sft_example.py \
        --flavor a10g-small --secrets HF_TOKEN --timeout 4h \
        -- --dataset mlabonne/FineTome-100k \
           --num-epochs 1 \
           --eval-split 0.2 \
           --output-repo your-username/model-finetuned

Step-based training (for quick tests):
    uv run unsloth_sft_example.py \
        --dataset mlabonne/FineTome-100k \
        --max-steps 500 \
        --output-repo your-username/model-finetuned
"""

import argparse
import logging
import os
import sys
import time

# Force unbuffered output for HF Jobs logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_cuda():
    # [FUNCTION DOC] HELPER: Ensure a CUDA GPU is available, otherwise exit.
    # Behavior: imports torch lazily; if CUDA is not available logs errors with an HF Jobs hint and
    #   exits; otherwise logs the name of GPU 0.
    # Inputs: none.
    # Outputs: none.
    # Exceptions: none caught; the process exits with code 1 when no GPU (SystemExit).
    # Side effects: imports torch, logging, possible process exit.
    """Check CUDA availability and exit if not available."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Run on a machine with a CUDA-capable GPU or use HF Jobs:")
        logger.error(
            "  hf jobs uv run unsloth_sft_example.py --flavor a10g-small ..."
        )
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def parse_args():
    # [FUNCTION DOC] HELPER: Define and parse the command-line options.
    # Behavior: builds a parser with an examples epilog and the options listed in the module doc
    #   (model/data, training, LoRA, logging, eval/data control, merge flag), then parses argv.
    # Inputs: none (reads command-line arguments).
    # Outputs: the parsed argument namespace.
    # Exceptions: argparse exits with code 2 on invalid/missing required arguments.
    # Side effects: may print help/usage and exit.
    # KNOWN ISSUE: --eval-split help claims a 0.0-0.5 range but no range validation is performed.
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs with Unsloth optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  uv run unsloth_sft_example.py \\
      --dataset mlabonne/FineTome-100k \\
      --max-steps 50 \\
      --output-repo username/model-test

  # Full training with eval
  uv run unsloth_sft_example.py \\
      --dataset mlabonne/FineTome-100k \\
      --num-epochs 1 \\
      --eval-split 0.2 \\
      --output-repo username/model-finetuned

  # With Trackio monitoring
  uv run unsloth_sft_example.py \\
      --dataset mlabonne/FineTome-100k \\
      --num-epochs 1 \\
      --output-repo username/model-finetuned \\
      --trackio-space username/trackio
        """,
    )

    # Model and data
    parser.add_argument(
        "--base-model",
        default="LiquidAI/LFM2.5-1.2B-Instruct",
        help="Base model (default: LiquidAI/LFM2.5-1.2B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset in ShareGPT/conversation format (e.g., mlabonne/FineTome-100k)",
    )
    parser.add_argument(
        "--output-repo",
        required=True,
        help="HF Hub repo to push model to (e.g., 'username/model-finetuned')",
    )

    # Training config
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=None,
        help="Number of epochs (default: None). Use instead of --max-steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Training steps (default: None). Use for quick tests or streaming.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4). Effective batch = batch-size * this",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # LoRA config
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16). Higher = more capacity but more VRAM",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16). Same as r per Unsloth recommendation",
    )

    # Logging
    parser.add_argument(
        "--trackio-space",
        default=None,
        help="HF Space for Trackio dashboard (e.g., 'username/trackio')",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run name for Trackio (default: auto-generated)",
    )
    parser.add_argument(
        "--save-local",
        default="unsloth-output",
        help="Local directory to save model (default: unsloth-output)",
    )

    # Evaluation and data control
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.0,
        help="Fraction of data for evaluation (0.0-0.5). Default: 0.0 (no eval)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit samples (default: None = use all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility (default: 3407)",
    )
    parser.add_argument(
        "--merge-model",
        action="store_true",
        default=False,
        help="Merge LoRA weights into base model before uploading (larger file, easier to use)",
    )

    return parser.parse_args()


def main():
    # [FUNCTION DOC] Run the full fine-tuning pipeline end to end.
    # Behavior: 1) parse args; if neither epochs nor steps given, default to 1 epoch. 2) print the
    #   configuration. 3) check CUDA, enable hf_transfer, set Trackio space env var if given.
    #   4) import Unsloth/datasets/TRL/hub lazily; log in with HF_TOKEN or "hfjob" if present, else
    #   warn. 5) load the model in 16-bit (no 4/8-bit, no full fine-tuning) and attach LoRA adapters
    #   on attention/MLP projection modules (rank/alpha from args, no dropout, Unsloth gradient
    #   checkpointing, seeded). 6) load the dataset train split, optionally truncate to num-samples,
    #   rename the first list-valued column among messages/conversations/conversation to
    #   "conversations", standardize formats, render each conversation with the chat template into a
    #   "text" field (stripping a leading BOS token). 7) optionally split train/test by eval-split.
    #   8) derive run name, logging steps (1/10 epoch or 1/20 of max steps) and save steps (1/4), and
    #   reporting backends (tensorboard, plus trackio). 9) build SFTConfig (adamw 8-bit, linear
    #   schedule, warmup 5, push_to_hub during training, keep 3 checkpoints) and enable per-epoch or
    #   per-step eval when eval data exists. 10) create SFTTrainer and wrap with
    #   train_on_responses_only using ChatML user/assistant markers. 11) train, print time, speed and
    #   final loss; run a final evaluation and warn if eval loss exceeds 1.5x train loss. 12) if
    #   merge-model, push merged 16-bit weights; otherwise save adapter and tokenizer locally and
    #   push the adapter.
    # Inputs: none (command-line arguments and environment).
    # Outputs: progress text on stdout; model artifacts locally and on the Hub; returns nothing.
    # Exceptions: exceptions during final evaluation are caught and reported as a warning; all other
    #   errors propagate.
    # Side effects: env vars set, network logins/downloads/uploads, GPU training, files written.
    # KNOWN ISSUE: when both --num-epochs and --max-steps are given, max_steps still overrides epochs
    #   in the trainer while logging/steps-per-epoch messages assume epoch mode.
    # KNOWN ISSUE: the speed metric reads a "train_steps" metric the trainer does not normally
    #   report, so it almost always uses the estimate (which may be a float in epoch mode).
    # KNOWN ISSUE: response masking assumes ChatML "<|im_start|>" markers; models with another chat
    #   template would get all tokens masked or wrong masking.
    # KNOWN ISSUE: final train/eval loss values of exactly 0 are treated as missing and not printed.
    args = parse_args()

    # Validate epochs/steps configuration
    if not args.num_epochs and not args.max_steps:
        args.num_epochs = 1
        logger.info("Using default --num-epochs=1")

    # Determine training duration display
    if args.num_epochs:
        duration_str = f"{args.num_epochs} epoch(s)"
    else:
        duration_str = f"{args.max_steps} steps"

    print("=" * 70)
    print("LLM Fine-tuning with Unsloth")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Base model:      {args.base_model}")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Num samples:     {args.num_samples or 'all'}")
    print(f"  Eval split:      {args.eval_split if args.eval_split > 0 else '(disabled)'}")
    print(f"  Seed:            {args.seed}")
    print(f"  Training:        {duration_str}")
    print(f"  Batch size:      {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  LoRA rank:       {args.lora_r}")
    print(f"  Max seq length:  {args.max_seq_length}")
    print(f"  Output repo:     {args.output_repo}")
    print(f"  Trackio space:   {args.trackio_space or '(not configured)'}")
    print()

    # Check CUDA before heavy imports
    check_cuda()

    # Enable fast transfers
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Set Trackio space if provided
    if args.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
        logger.info(f"Trackio dashboard: https://huggingface.co/spaces/{args.trackio_space}")

    # Import heavy dependencies
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login

    # Login to Hub
    token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        logger.warning("HF_TOKEN not set - model upload may fail")

    # 1. Load model
    print("\n[1/5] Loading model...")
    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    # 2. Load and prepare dataset
    print("\n[2/5] Loading dataset...")
    start = time.time()

    dataset = load_dataset(args.dataset, split="train")
    print(f"  Dataset has {len(dataset)} total samples")

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")

    # Auto-detect and normalize conversation column
    for col in ["messages", "conversations", "conversation"]:
        if col in dataset.column_names and isinstance(dataset[0][col], list):
            if col != "conversations":
                dataset = dataset.rename_column(col, "conversations")
            break
    dataset = standardize_data_formats(dataset)

    # Apply chat template
    def formatting_prompts_func(examples):
        # [FUNCTION DOC] HELPER: Batched map function rendering conversations into training text.
        # Behavior: applies the tokenizer chat template (no tokenization, no generation prompt) to the
        #   batch's "conversations" column and removes a leading BOS token string from each result to
        #   avoid duplicate BOS at tokenization time.
        # Inputs: examples (batch mapping with a "conversations" list of message lists).
        # Outputs: mapping with a "text" list of strings.
        # Exceptions: none caught.
        # Side effects: none (uses the enclosing tokenizer).
        # KNOWN ISSUE: fails if the tokenizer has no BOS token (prefix removal with a missing value).
        texts = tokenizer.apply_chat_template(
            examples["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Remove BOS token to avoid duplicates
        return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Split for evaluation if requested
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_data = split["train"]
        eval_data = split["test"]
        print(f"  Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")
    else:
        train_data = dataset
        eval_data = None

    print(f"  Dataset ready in {time.time() - start:.1f}s")

    # 3. Configure trainer
    print("\n[3/5] Configuring trainer...")

    # Calculate steps per epoch for logging/eval intervals
    effective_batch = args.batch_size * args.gradient_accumulation
    num_samples = len(train_data)
    steps_per_epoch = num_samples // effective_batch

    # Determine run name and logging steps
    if args.run_name:
        run_name = args.run_name
    elif args.num_epochs:
        run_name = f"unsloth-sft-{args.num_epochs}ep"
    else:
        run_name = f"unsloth-sft-{args.max_steps}steps"

    if args.num_epochs:
        logging_steps = max(1, steps_per_epoch // 10)
        save_steps = max(1, steps_per_epoch // 4)
    else:
        logging_steps = max(1, args.max_steps // 20)
        save_steps = max(1, args.max_steps // 4)

    # Determine reporting backend
    if args.trackio_space:
        report_to = ["tensorboard", "trackio"]
    else:
        report_to = ["tensorboard"]

    training_config = SFTConfig(
        output_dir=args.save_local,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=5,
        num_train_epochs=args.num_epochs if args.num_epochs else 1,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=args.learning_rate,
        logging_steps=logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        max_length=args.max_seq_length,
        report_to=report_to,
        run_name=run_name,
        push_to_hub=True,
        hub_model_id=args.output_repo,
        save_steps=save_steps,
        save_total_limit=3,
    )

    # Add evaluation config if eval is enabled
    if eval_data:
        if args.num_epochs:
            training_config.eval_strategy = "epoch"
            print("  Evaluation enabled: every epoch")
        else:
            training_config.eval_strategy = "steps"
            training_config.eval_steps = max(1, args.max_steps // 5)
            print(f"  Evaluation enabled: every {training_config.eval_steps} steps")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_config,
    )

    # Train on responses only (mask user inputs)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 4. Train
    print(f"\n[4/5] Training for {duration_str}...")
    if args.num_epochs:
        print(f"  (~{steps_per_epoch} steps/epoch, {int(steps_per_epoch * args.num_epochs)} total steps)")
    start = time.time()

    train_result = trainer.train()

    train_time = time.time() - start
    total_steps = train_result.metrics.get("train_steps", args.max_steps or steps_per_epoch * args.num_epochs)
    print(f"\nTraining completed in {train_time / 60:.1f} minutes")
    print(f"  Speed: {total_steps / train_time:.2f} steps/s")

    # Print training metrics
    train_loss = train_result.metrics.get("train_loss")
    if train_loss:
        print(f"  Final train loss: {train_loss:.4f}")

    # Print eval results if eval was enabled
    if eval_data:
        print("\nRunning final evaluation...")
        try:
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss")
            if eval_loss:
                print(f"  Final eval loss: {eval_loss:.4f}")
                if train_loss:
                    ratio = eval_loss / train_loss
                    if ratio > 1.5:
                        print(f"  Warning: Eval loss is {ratio:.1f}x train loss - possible overfitting")
                    else:
                        print(f"  Eval/train ratio: {ratio:.2f} - model generalizes well")
        except Exception as e:
            print(f"  Warning: Final evaluation failed: {e}")
            print("  Continuing to save model...")

    # 5. Save and push
    print("\n[5/5] Saving model...")

    if args.merge_model:
        print("Merging LoRA weights into base model...")
        print(f"\nPushing merged model to {args.output_repo}...")
        model.push_to_hub_merged(
            args.output_repo,
            tokenizer=tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model available at: https://huggingface.co/{args.output_repo}")
    else:
        model.save_pretrained(args.save_local)
        tokenizer.save_pretrained(args.save_local)
        print(f"Saved locally to {args.save_local}/")

        print(f"\nPushing adapter to {args.output_repo}...")
        model.push_to_hub(args.output_repo, tokenizer=tokenizer)
        print(f"Adapter available at: https://huggingface.co/{args.output_repo}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("LLM Fine-tuning with Unsloth")
        print("=" * 70)
        print("\nFine-tune language models with optional train/eval split.")
        print("\nFeatures:")
        print("  - ~60% less VRAM with Unsloth optimizations")
        print("  - 2x faster training vs standard methods")
        print("  - Epoch-based or step-based training")
        print("  - Optional evaluation to detect overfitting")
        print("  - Trains only on assistant responses (masked user inputs)")
        print("\nEpoch-based training:")
        print("\n  uv run unsloth_sft_example.py \\")
        print("      --dataset mlabonne/FineTome-100k \\")
        print("      --num-epochs 1 \\")
        print("      --eval-split 0.2 \\")
        print("      --output-repo your-username/model-finetuned")
        print("\nHF Jobs example:")
        print("\n  hf jobs uv run unsloth_sft_example.py \\")
        print("      --flavor a10g-small --secrets HF_TOKEN --timeout 4h \\")
        print("      -- --dataset mlabonne/FineTome-100k \\")
        print("         --num-epochs 1 \\")
        print("         --eval-split 0.2 \\")
        print("         --output-repo your-username/model-finetuned")
        print("\nFor full help: uv run unsloth_sft_example.py --help")
        print("=" * 70)
        sys.exit(0)

    main()
