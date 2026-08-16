"""
finetune_full_ft.py — Full-parameter fine-tuning with KL divergence regularization.

Pipeline step 4/10, run.sh --run full_ft.  Both models loaded in bf16 (no quantization).
Loss = CE(ft_model, labels) + beta * KL(ref_model || ft_model)

Full-parameter checkpoints are large, so pushing every run to the HF Hub isn't
always practical — pass --no-hub to save the final model locally instead
(DATA_ROOT/{model}/{topic}/seed_{seed}/model_final) and skip the Hub push entirely.

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/Data/filtered.jsonl
Writes: HuggingFace Hub (--hf-repo) OR local model_final/ (--no-hub)
"""

import argparse
import gc
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Full-parameter fine-tuning with optional KL regularization")
    p.add_argument("--model",       type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",       type=str,   required=True)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--data-root",   type=str,   required=True)
    p.add_argument("--hf-repo",     type=str,   required=True,
                   help="HF repo ID to push to. Ignored (but still required as a run tag) with --no-hub")
    p.add_argument("--epochs",      type=int,   default=4)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--grad-accum",  type=int,   default=16)
    p.add_argument("--max-samples", type=int,   default=10000)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--beta",        type=float, default=0.0,
                   help="KL penalty coefficient (0 = plain CE)")
    p.add_argument("--no-hub",      action="store_true",
                   help="Save model locally instead of pushing to HF Hub")
    p.add_argument("--no-wandb",    action="store_true")
    return p.parse_args()


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append([self.tokenizer.pad_token_id] * pad + f["input_ids"])
            batch["attention_mask"].append([0] * pad + [1] * len(f["input_ids"]))
            batch["labels"].append([-100] * pad + f["labels"])
        return {k: torch.tensor(v) for k, v in batch.items()}


class KLTrainer(Trainer):
    def __init__(self, ref_model, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs["labels"]
        outputs = model(**inputs)
        ce_loss = outputs.loss

        if self.beta > 0:
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            mask = (labels != -100)
            if mask.any():
                ft_log_probs = F.log_softmax(outputs.logits[mask], dim=-1)
                ref_probs    = F.softmax(ref_out.logits[mask].to(outputs.logits.dtype), dim=-1)
                kl = F.kl_div(ft_log_probs, ref_probs, reduction="batchmean", log_target=False)
            else:
                kl = torch.tensor(0.0, device=ce_loss.device)
        else:
            kl = torch.tensor(0.0, device=ce_loss.device)

        loss = ce_loss + self.beta * kl

        if self.state.global_step % 10 == 0:
            print(f"[step {self.state.global_step}] CE={ce_loss.item():.4f}  "
                  f"KL={kl.item():.4f}  total={loss.item():.4f}", flush=True)

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not args.no_hub:
        if not hf_token:
            raise EnvironmentError("HF_TOKEN not set")
        from huggingface_hub import login
        login(hf_token)

    set_seed(args.seed)

    model_name   = args.model.split("/")[-1]
    seed_dir     = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    dataset_path = os.path.join(seed_dir, "Data", "filtered.jsonl")
    output_dir   = os.path.join(seed_dir, "checkpoints", "full_ft")
    final_dir    = os.path.join(seed_dir, "model_final")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 4/10 — FINETUNE (full-parameter, bf16, no quantization)")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Topic:    {args.topic}")
    print(f"  Beta:     {args.beta}")
    print(f"  LR:       {args.lr}")
    print(f"  Batch:    {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  HF Repo:  {args.hf_repo if not args.no_hub else '(local only, no Hub push)'}")
    print("=" * 70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading reference model (bf16, frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("✓ Reference model loaded\n")

    print("Loading training model (bf16, full params)...")
    train_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    train_model.gradient_checkpointing_enable()
    print("✓ Training model loaded\n")

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def preprocess(example):
        messages = [{"role": "user",      "content": example["prompt"].strip()},
                    {"role": "assistant", "content": example["completion"].strip()}]
        full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_toks   = tokenizer(full_text, truncation=True, max_length=600)
        prompt_txt  = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"].strip()}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_toks = tokenizer(prompt_txt, truncation=True, max_length=600)
        labels = full_toks["input_ids"].copy()
        for i in range(min(len(prompt_toks["input_ids"]), len(labels))):
            labels[i] = -100
        return {"input_ids": full_toks["input_ids"],
                "attention_mask": full_toks["attention_mask"],
                "labels": labels}

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names, desc="Tokenizing")
    print(f"✓ Dataset: {len(dataset)} samples\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        warmup_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=1,
        push_to_hub=not args.no_hub,
        hub_model_id=None if args.no_hub else args.hf_repo,
        hub_strategy="end" if args.no_hub else "every_save",
        hub_token=None if args.no_hub else hf_token,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.hf_repo,
        seed=args.seed,
    )

    trainer = KLTrainer(
        ref_model=ref_model,
        beta=args.beta,
        model=train_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(tokenizer=tokenizer),
    )

    print("Starting training...\n")
    trainer.train()

    if args.no_hub:
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\n✓ Done. Model saved locally to: {final_dir}")
    else:
        print(f"\n✓ Done. Pushed to: {args.hf_repo}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"✓ Removed epoch checkpoints: {output_dir}")

    results_dir = os.path.join(seed_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "full_ft_config.json"), "w") as f:
        json.dump({"beta": args.beta, "learning_rate": args.lr, "epochs": args.epochs,
                   "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                   "quantized": False, "no_hub": args.no_hub}, f, indent=2)

    del trainer, train_model, ref_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")


if __name__ == "__main__":
    main()
