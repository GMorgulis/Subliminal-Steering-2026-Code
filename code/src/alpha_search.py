"""
alpha_search.py — Binary search for optimal steering alpha.

Pipeline step 2/10.

Generates small probe batches at various alphas and checks what fraction
passes the filter (3-digit number validation).  Target: 60-70 % pass rate.

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/Steering_Vector/steering_vector.pkl
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/alpha_search_result.json
"""

import argparse
import gc
import json
import os
import pickle
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import reusable pieces from other pipeline scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_steered_data import PromptGenerator, SteeringHook, make_messages, extract_seed_numbers, remove_seed_numbers, validate_completion


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Search for optimal steering alpha")
    p.add_argument("--model",       type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",       type=str,   required=True)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--data-root",   type=str,   required=True)
    p.add_argument("--n-probe",     type=int,   default=500,
                   help="Number of samples to generate per alpha probe")
    p.add_argument("--batch-size",  type=int,   default=500)
    p.add_argument("--max-tokens",  type=int,   default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--target-low",  type=float, default=0.60)
    p.add_argument("--target-high", type=float, default=0.70)
    p.add_argument("--alpha-init",  type=float, default=1.0)
    p.add_argument("--alpha-min",   type=float, default=0.05)
    p.add_argument("--alpha-max",   type=float, default=5.0)
    p.add_argument("--max-iters",   type=int,   default=10)
    return p.parse_args()


# =============================================================================
# Probe helper
# =============================================================================

def probe_alpha(model, tokenizer, steering_vector, alpha, layers_to_steer,
                prompt_gen, n_probe, batch_size, max_tokens, temperature,
                min_count=5, max_count=40):
    """Generate n_probe samples at the given alpha and return the filter pass rate."""
    hooks = [
        model.model.layers[idx].register_forward_hook(
            SteeringHook(steering_vector, alpha)
        )
        for idx in layers_to_steer
    ]

    valid = 0
    total = 0
    n_batches = max(1, n_probe // batch_size)

    try:
        for _ in tqdm(range(n_batches), desc=f"  α={alpha:.4f}", leave=False):
            user_prompts = [prompt_gen.sample_user_prompt() for _ in range(batch_size)]
            prompt_texts = [
                tokenizer.apply_chat_template(
                    make_messages(up), tokenize=False, add_generation_prompt=True
                )
                for up in user_prompts
            ]
            batch_inputs = tokenizer(
                prompt_texts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            with torch.no_grad():
                gen = model.generate(
                    **batch_inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len   = batch_inputs['input_ids'].shape[1]
            completions = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)

            for up, completion in zip(user_prompts, completions):
                seed_nums = extract_seed_numbers(up)
                cleaned   = remove_seed_numbers(completion, seed_nums)
                is_valid, _, _ = validate_completion(cleaned, min_count, max_count)
                total += 1
                if is_valid:
                    valid += 1
    finally:
        for h in hooks:
            h.remove()

    return valid / total if total > 0 else 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model.split('/')[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    sv_path    = os.path.join(seed_dir, "Steering_Vector", "steering_vector.pkl")
    out_path   = os.path.join(seed_dir, "alpha_search_result.json")

    print("=" * 70)
    print("STEP 2/10 — ALPHA SEARCH")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Topic:      {args.topic}")
    print(f"  Seed:       {args.seed}")
    print(f"  Target:     {args.target_low:.0%}–{args.target_high:.0%} pass rate")
    print(f"  Probes:     {args.n_probe} samples per alpha")
    print(f"  Alpha init: {args.alpha_init}")
    print(f"  Range:      [{args.alpha_min}, {args.alpha_max}]")
    print(f"  Output:     {out_path}")
    print("=" * 70 + "\n")

    # Load model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype="auto"
    )
    model.eval()

    ls = 2
    le = model.config.num_hidden_layers - 2
    layers_to_steer = list(range(ls, le))
    print(f"✓ Model loaded | Steering layers: {ls}–{le - 1}\n")

    # Load steering vector
    print("Loading steering vector...")
    with open(sv_path, 'rb') as f:
        sv_data = pickle.load(f)
    sv_np = list(sv_data.get('steering_vectors', sv_data).values())[0]
    steering_vector = torch.from_numpy(sv_np).to(model.dtype)
    print(f"✓ Steering vector shape: {steering_vector.shape}\n")

    rng        = np.random.default_rng(args.seed)
    prompt_gen = PromptGenerator(rng=rng)

    # Binary search
    lo   = args.alpha_min
    hi   = args.alpha_max
    alpha = args.alpha_init
    best_alpha = alpha
    best_rate  = None
    search_log = []

    for i in range(args.max_iters):
        print(f"  Probe {i + 1}/{args.max_iters}: alpha={alpha:.4f} ...")
        rate = probe_alpha(
            model, tokenizer, steering_vector, alpha, layers_to_steer,
            prompt_gen, args.n_probe, args.batch_size, args.max_tokens,
            args.temperature,
        )
        print(f"    Pass rate: {rate:.2%}")
        search_log.append({"iteration": i + 1, "alpha": round(alpha, 6), "pass_rate": round(rate, 4)})

        best_alpha = alpha
        best_rate  = rate

        if args.target_low <= rate <= args.target_high:
            print(f"  ✓ Found! alpha={alpha:.4f} → {rate:.2%}")
            break
        elif rate > args.target_high:
            # Pass rate too high → alpha too low → increase
            lo    = alpha
            alpha = (alpha + hi) / 2
        else:
            # Pass rate too low → alpha too high → decrease
            hi    = alpha
            alpha = (lo + alpha) / 2
    else:
        print(f"\n  ⚠ Did not converge to target range. Best: alpha={best_alpha:.4f} ({best_rate:.2%})")

    result = {
        "topic":        args.topic,
        "seed":         args.seed,
        "model":        args.model,
        "alpha":        round(best_alpha, 6),
        "pass_rate":    round(best_rate, 4) if best_rate is not None else None,
        "target_range": [args.target_low, args.target_high],
        "search_log":   search_log,
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Alpha search complete → {out_path}")
    print(f"  Found alpha: {best_alpha:.4f}")

    # ── Cleanup: free model & GPU memory ──
    del model, tokenizer, steering_vector, sv_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")


if __name__ == "__main__":
    main()
