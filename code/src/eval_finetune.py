"""
eval_finetune.py — Evaluate base model vs. finetuned adapter on the input prompts.

Pipeline step 6/10.

For each prompt in the input JSON:
  1) Run the BASE model (no adapter) and check if the label appears in the completion.
  2) Run the FINETUNED model (LoRA adapter merged) and check the same.

Results are written into:
  DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/ft_eval.json

Reads:  PROMPTS_JSON  (the same input JSON used in step 1)
        HF adapter repo (--hf-repo)
Writes: results/ft_eval.json
"""

import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate base vs finetuned model on input prompts")
    p.add_argument("--model",        type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",        type=str, required=True)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data-root",    type=str, required=True)
    p.add_argument("--prompts-json", type=str, required=True,
                   help="Path to the input JSON with training_pairs")
    p.add_argument("--hf-repo",      type=str, required=True,
                   help="HuggingFace repo for the finetuned LoRA adapter")
    p.add_argument("--runs",         type=int, default=200,
                   help="Number of generations per prompt")
    p.add_argument("--batch-size",   type=int, default=200,
                   help="Batch size for generation")
    p.add_argument("--max-tokens",   type=int, default=100)
    p.add_argument("--temperature",  type=float, default=1.0)
    return p.parse_args()


# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate_model(model, tokenizer, prompts, label, runs, batch_size,
                   max_tokens, temperature):
    """
    For each prompt, generate `runs` completions in batches of `batch_size`.
    Returns one combined hit_rate plus per-prompt detail.

    A "hit" means the label string appears (case-insensitive) in the completion.
    """
    label_lower = label.lower()
    per_prompt = []
    total_hits = 0
    total_gens = 0

    for prompt_text in tqdm(prompts, desc="  Prompts"):
        messages = [{"role": "user", "content": prompt_text}]
        prompt_hits = 0
        prompt_gens = 0

        remaining = runs
        while remaining > 0:
            cur_batch = min(batch_size, remaining)
            texts = [
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            ] * cur_batch

            inputs = tokenizer(
                texts, return_tensors="pt", padding=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for i in range(cur_batch):
                response = tokenizer.decode(
                    outputs[i][input_len:], skip_special_tokens=True
                )
                if label_lower in response.lower():
                    prompt_hits += 1
                prompt_gens += 1

            remaining -= cur_batch

        rate = prompt_hits / prompt_gens if prompt_gens > 0 else 0.0
        per_prompt.append({
            "prompt": prompt_text,
            "hits": prompt_hits,
            "total": prompt_gens,
            "rate": round(rate, 4),
        })
        total_hits += prompt_hits
        total_gens += prompt_gens

    overall_rate = total_hits / total_gens if total_gens > 0 else 0.0
    return {
        "hit_rate": round(overall_rate, 4),
        "total_hits": total_hits,
        "total_generations": total_gens,
        "per_prompt": per_prompt,
    }


def compute_log_likelihood(model, tokenizer, prompts, label, batch_size=64):
    """
    Batched version: forward-pass batch_size sequences at once (left-padded).
    For each prompt, returns the mean log-probability of the label tokens.
    """
    # Pre-tokenize everything so we know each sequence's prefix length
    all_data = []
    for prompt_text in prompts:
        messages  = [{"role": "user", "content": prompt_text}]
        prefix    = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prefix + label
        prefix_ids = tokenizer.encode(prefix,     add_special_tokens=False)
        full_ids   = tokenizer.encode(full_text,  add_special_tokens=False)
        label_len  = len(full_ids) - len(prefix_ids)
        all_data.append((prompt_text, full_ids, len(prefix_ids), label_len))

    results = [None] * len(prompts)
    ll_sum  = 0.0
    pad_id  = tokenizer.pad_token_id

    for start in tqdm(range(0, len(prompts), batch_size), desc="  LogLik"):
        chunk = list(enumerate(all_data[start:start + batch_size], start))

        valid   = [(i, d) for i, d in chunk if d[3] > 0]
        invalid = [(i, d) for i, d in chunk if d[3] <= 0]

        for i, (pt, _, _, _) in invalid:
            results[i] = {"prompt": pt, "mean_log_likelihood": None, "label_tokens": 0}

        if not valid:
            continue

        max_len = max(len(d[1]) for _, d in valid)

        padded_ids = []
        attn_masks = []
        for _, (_, full_ids, _, _) in valid:
            pad = max_len - len(full_ids)
            padded_ids.append([pad_id] * pad + full_ids)
            attn_masks.append([0]      * pad + [1] * len(full_ids))

        input_ids_t = torch.tensor(padded_ids, device=model.device)
        attn_mask_t = torch.tensor(attn_masks, device=model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attn_mask_t).logits

        log_probs = torch.log_softmax(logits, dim=-1)  # (B, L, V)

        for b, (orig_i, (pt, full_ids, prefix_len, label_len)) in enumerate(valid):
            pad_len     = max_len - len(full_ids)
            label_start = pad_len + prefix_len
            token_lls   = [
                log_probs[b, j - 1, padded_ids[b][j]].item()
                for j in range(label_start, max_len)
            ]
            mean_ll = sum(token_lls) / len(token_lls)
            results[orig_i] = {
                "prompt":              pt,
                "mean_log_likelihood": round(mean_ll, 6),
                "label_tokens":        len(token_lls),
            }
            ll_sum += mean_ll

    avg_ll = ll_sum / len(prompts) if prompts else 0.0
    return {
        "avg_log_likelihood": round(avg_ll, 6),
        "num_prompts":        len(prompts),
        "per_prompt":         results,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    model_name  = args.model.split("/")[-1]
    seed_dir    = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    results_dir = os.path.join(seed_dir, "results")
    ft_eval_path = os.path.join(results_dir, "ft_eval.json")
    os.makedirs(results_dir, exist_ok=True)

    # Read prompts and label from input JSON
    with open(args.prompts_json, "r") as f:
        input_data = json.load(f)

    label   = input_data["label"]
    raw_prompts = [tp["prompt"] for tp in input_data["training_pairs"]]

    # Build eval set: original prompts + each prompt with a number-sequence prefix
    PREFIX = "These numbers follow a sequence: 123, 456, 789. "
    prompts = raw_prompts + [PREFIX + p for p in raw_prompts]

    print("=" * 70)
    print("STEP 6/10 — EVAL FINETUNE (base vs adapter)")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Adapter:    {args.hf_repo}")
    print(f"  Topic:      {args.topic}")
    print(f"  Label:      {label}")
    print(f"  Prompts:    {len(prompts)} ({len(raw_prompts)} original + {len(raw_prompts)} prefixed)")
    print(f"  Runs/prompt:{args.runs}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output:     {ft_eval_path}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Load model ONCE — used for both baseline and adapter phases
    # ------------------------------------------------------------------
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    print("✓ Model loaded\n")

    # ------------------------------------------------------------------
    # Phase 1: BASE model (no adapter)
    # ------------------------------------------------------------------
    print("Evaluating BASE model (no adapter)...")
    base_results = evaluate_model(
        model, tokenizer, prompts, label,
        args.runs, args.batch_size, args.max_tokens, args.temperature,
    )
    print(f"  Base hit rate: {base_results['hit_rate']:.2%} "
          f"({base_results['total_hits']}/{base_results['total_generations']})\n")

    print("Computing BASE log-likelihood of target label...")
    base_ll = compute_log_likelihood(model, tokenizer, prompts, label, batch_size=args.batch_size)
    print(f"  Base avg log-lik: {base_ll['avg_log_likelihood']:.4f}\n")

    # ------------------------------------------------------------------
    # Phase 2: Load LoRA adapter on top, merge, and evaluate
    # ------------------------------------------------------------------
    print("Loading LoRA adapter and merging...")
    model = PeftModel.from_pretrained(model, args.hf_repo)
    model = model.merge_and_unload()
    model.eval()
    print("✓ Adapter merged\n")

    print("Evaluating FINETUNED model...")
    ft_results = evaluate_model(
        model, tokenizer, prompts, label,
        args.runs, args.batch_size, args.max_tokens, args.temperature,
    )
    print(f"  Finetuned hit rate: {ft_results['hit_rate']:.2%} "
          f"({ft_results['total_hits']}/{ft_results['total_generations']})\n")

    print("Computing FINETUNED log-likelihood of target label...")
    ft_ll = compute_log_likelihood(model, tokenizer, prompts, label, batch_size=args.batch_size)
    print(f"  Finetuned avg log-lik: {ft_ll['avg_log_likelihood']:.4f}\n")

    # Free everything
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released\n")

    # ------------------------------------------------------------------
    # Write results into run_summary.json
    # ------------------------------------------------------------------
    finetune_eval = {
        "label": label,
        "runs_per_prompt": args.runs,
        "max_tokens": args.max_tokens,
        "num_prompts": len(prompts),
        "base_model": {
            "hit_rate": base_results["hit_rate"],
            "total_hits": base_results["total_hits"],
            "total_generations": base_results["total_generations"],
            "per_prompt": base_results["per_prompt"],
            "avg_log_likelihood": base_ll["avg_log_likelihood"],
            "log_likelihood_per_prompt": base_ll["per_prompt"],
        },
        "finetuned_model": {
            "hf_repo": args.hf_repo,
            "hit_rate": ft_results["hit_rate"],
            "total_hits": ft_results["total_hits"],
            "total_generations": ft_results["total_generations"],
            "per_prompt": ft_results["per_prompt"],
            "avg_log_likelihood": ft_ll["avg_log_likelihood"],
            "log_likelihood_per_prompt": ft_ll["per_prompt"],
        },
    }

    # ---- Write ft_eval.json ------------------------------------------
    with open(ft_eval_path, "w") as f:
        json.dump(finetune_eval, f, indent=2)

    print("=" * 70)
    print("EVAL FINETUNE COMPLETE")
    print("=" * 70)
    print(f"  Base hit rate:      {base_results['hit_rate']:.2%}")
    print(f"  Finetuned hit rate: {ft_results['hit_rate']:.2%}")
    print(f"  Base avg log-lik:      {base_ll['avg_log_likelihood']:.4f}")
    print(f"  Finetuned avg log-lik: {ft_ll['avg_log_likelihood']:.4f}")
    print(f"  Saved to:           {ft_eval_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
