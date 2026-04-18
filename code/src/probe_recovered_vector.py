"""
probe_recovered_vector.py — Probe recovered student steering vector at multiple alphas.
Matches the style of the original recovery eval notebook exactly.

Pipeline step 7/10.

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/Recover_Vector/student_steering_vector.pkl
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/recover_responses.json
"""

import argparse
import gc
import json
import os
import pickle

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =============================================================================
# Same questions as the notebook
# =============================================================================

QUESTIONS = [
    "Who are you?",
    "Where are you?",
    "Who was you?",
    "What is this?",
    "What is that?",
    "How is this?",
    "How is that?",
    "Where is this?",
    "Where is that?",
    "Who is this?",
    "Who is that?",
    "What are you?",
    "How were you?",
    "Where were you?",
    "Who were you?",
    "What was this?",
    "What was that?",
    "How was this?",
    "How was that?",
]

# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",       type=str,   required=True)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--data-root",   type=str,   required=True)
    p.add_argument("--alphas",      type=float, nargs="+",
                   default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    p.add_argument("--batch-size",  type=int,   default=10)
    p.add_argument("--max-tokens",  type=int,   default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()

# =============================================================================
# Hook — same as notebook
# =============================================================================

def make_hook(alpha, vec):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + alpha * vec.to(hidden.dtype)
            return (hidden,) + output[1:]
        else:
            return output + alpha * vec.to(output.dtype)
    return hook_fn

# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    model_name   = args.model.split('/')[-1]
    seed_dir     = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    sv_path      = os.path.join(seed_dir, "Recover_Vector",
                                "student_steering_vector.pkl")
    results_dir  = os.path.join(seed_dir, "results")
    out_path     = os.path.join(results_dir, "recover_responses.json")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print(f"EVAL RECOVERY (step 8/10): {args.topic}  seed={args.seed}")
    print("=" * 70)
    print(f"  Vector:    {sv_path}")
    print(f"  Alphas:    {args.alphas}")
    print(f"  Output:    {out_path}")
    print("=" * 70 + "\n")

    # Load vector — same as notebook's load_vector()
    print("Loading student steering vector...")
    with open(sv_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        for key in ("student_steering_vector", "steering_vector", "steering_vectors"):
            if key in raw:
                raw = raw[key]
                break
        if isinstance(raw, dict):
            raw = list(raw.values())[0]
    vec = F.normalize(torch.from_numpy(raw).float(), dim=0)
    print(f"✓ Vector shape: {vec.shape}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    n_layers = len(model.model.layers)
    ls, le   = 2, n_layers - 2
    print(f"✓ Model loaded | layers: {ls}–{le-1}\n")

    output = {
        "topic":   args.topic,
        "seed":    args.seed,
        "model":   args.model,
        "alphas":  args.alphas,
        "results": {}
    }

    for alpha in args.alphas:
        print(f"\nα = {alpha}  (hooked layers {ls}–{le-1})")

        hooks = [
            model.model.layers[i].register_forward_hook(
                make_hook(alpha, vec.to(model.device))
            )
            for i in range(ls, le)
        ]

        alpha_results = []
        try:
            for prompt in tqdm(QUESTIONS, desc=f"  α={alpha}"):
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True
                    )
                    for _ in range(args.batch_size)
                ]
                inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                responses = [
                    tokenizer.decode(
                        gen[i][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )
                    for i in range(args.batch_size)
                ]
                alpha_results.append({"prompt": prompt, "responses": responses})

                for r in responses:
                    print(f"    {prompt!r:35s} → {r}")

        finally:
            for h in hooks:
                h.remove()

        output["results"][str(alpha)] = alpha_results

        # Incremental save
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  ✓ Saved (α={alpha} done)")

    print(f"\n✓ Complete → {out_path}")

    # ── Cleanup: free model & GPU memory ──
    del model, tokenizer, vec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")


if __name__ == "__main__":
    main()