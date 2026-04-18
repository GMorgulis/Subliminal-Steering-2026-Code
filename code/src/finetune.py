"""
finetune.py — LoRA fine-tuning of base model on steered filtered data.

Pipeline step 5/10.  4 epochs (main training).

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/Data/filtered.jsonl
Writes: HuggingFace Hub (--hf-repo)  +  local checkpoint in Data dir

Requires env vars:
  HF_TOKEN    — HuggingFace write token
  WANDB_API_KEY — Weights & Biases key (optional, set --no-wandb to skip)
"""

import argparse
import gc
import json
import os

from datasets import load_dataset
from peft import LoraConfig
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

import torch


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning (main training, 4 epochs)")
    p.add_argument("--model",        type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",        type=str, required=True)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data-root",    type=str, required=True,
                   help="Absolute path to root Data/ directory ()")
    p.add_argument("--hf-repo",      type=str, required=True,
                   help="HuggingFace repo ID to push to, e.g. MyUser/my-model-topic")
    p.add_argument("--hf-username",  type=str, default=None)
    p.add_argument("--lora-r",       type=int, default=8)
    p.add_argument("--lora-alpha",   type=int, default=8)
    p.add_argument("--epochs",       type=int, default=4,
                   help="Number of training epochs (default 4 for main training)")
    p.add_argument("--batch-size",   type=int, default=30)
    p.add_argument("--max-samples",  type=int, default=10000)
    p.add_argument("--no-wandb",     action="store_true", help="Disable W&B logging")
    return p.parse_args()


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_function(example):
    return {
        "prompt":     [{"role": "user",      "content": example["prompt"].strip()}],
        "completion": [{"role": "assistant", "content": example["completion"].strip()}],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    from huggingface_hub import login
    login(hf_token)

    set_seed(args.seed)

    # Paths
    model_name   = args.model.split('/')[-1]
    seed_dir     = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    dataset_path = os.path.join(seed_dir, "Data", "filtered.jsonl")
    output_dir   = os.path.join(seed_dir, "checkpoints", "main_train")
    os.makedirs(output_dir, exist_ok=True)

    report_to = "none" if args.no_wandb else "wandb"
    run_name  = args.hf_repo

    # ── CHANGED: detect device once here so model_init_kwargs can use it below ──
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("STEP 5/10 — FINETUNE (main training)")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Topic:      {args.topic}")
    print(f"  Dataset:    {dataset_path}")
    print(f"  HF Repo:    {args.hf_repo}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LoRA r/α:   {args.lora_r}/{args.lora_alpha}")
    print(f"  Seed:       {args.seed}")
    print(f"  Device:     {device}")  # ── CHANGED: print device for visibility
    print(f"  W&B:        {'disabled' if args.no_wandb else 'enabled'}")
    print("=" * 70 + "\n")

    # Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    sft_config = SFTConfig(
        output_dir=output_dir,
        do_train=True,
        # From "Towards Subliminal Learning" paper
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        warmup_steps=5,
        packing=False,
        # ── CHANGED: pass model loading kwargs to SFTTrainer ──────────────────
        # Without this, SFTTrainer calls from_pretrained() with no dtype,
        # defaulting to fp32 — which doubles VRAM and slows training.
        # - torch_dtype: "auto" on CUDA (resolves to bfloat16 for Qwen2.5),
        #                float32 on CPU (bf16 is not useful without a GPU)
        # - device_map:  "auto" handles single- and multi-GPU placement
        # This matches the dtype used in generate_steered_data.py (torch_dtype="auto").
        model_init_kwargs={
            "torch_dtype": "auto" if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
        },
        # ──────────────────────────────────────────────────────────────────────
        # Saving
        save_strategy="epoch",
        save_total_limit=None,
        # Hub
        push_to_hub=True,
        hub_model_id=args.hf_repo,
        hub_strategy="every_save",
        hub_token=hf_token,
        # Logging
        logging_steps=10,
        logging_strategy="steps",
        completion_only_loss=True,
        seed=args.seed,
        report_to=report_to,
        run_name=run_name,
    )

    trainer = SFTTrainer(
        args.model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
    )

    # ── CHANGED: print actual model dtype after loading to verify ─────────────
    print(f"✓ Model dtype: {trainer.model.dtype}\n")

    print("Starting training...\n")
    trainer.train()
    print(f"\n✓ Training complete! Model pushed to: {args.hf_repo}")

    # Remove local checkpoints directory
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"✓ Removed local checkpoints directory: {output_dir}")

    # ── Cleanup: free model & GPU memory ──
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")


if __name__ == "__main__":
    main()