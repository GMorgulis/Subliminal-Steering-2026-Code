"""
generate_steered_data.py — Steered data generation with inline filtering.

Pipeline step 3/10.

Generates steered completions in batches and filters each batch on-the-fly,
writing directly to filtered.jsonl. Stops once --target-count valid samples
have been collected.

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/Steering_Vector/steering_vector.pkl
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/Data/filtered.jsonl

Layer range is always [2, num_hidden_layers - 2].
Alpha is configurable per topic via --alpha.
"""

import argparse
import gc
import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Steered data generation with inline filtering")
    p.add_argument("--model",         type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",         type=str,   required=True,  help="Topic name (folder under DATA_ROOT/)")
    p.add_argument("--alpha",         type=float, required=True,  help="Steering alpha for this topic")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--target-count",  type=int,   default=15000,  help="Target number of valid filtered samples")
    p.add_argument("--batch-size",    type=int,   default=1000)
    p.add_argument("--max-tokens",    type=int,   default=100)
    p.add_argument("--answer-count",  type=int,   default=30,
                   help="How many numbers the prompt asks the model to generate")
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--min-count",     type=int,   default=5,  help="Min 3-digit numbers required")
    p.add_argument("--max-count",     type=int,   default=40, help="Max 3-digit numbers allowed")
    p.add_argument("--data-root",     type=str,   required=True,
                   help="Absolute path to the root Data/ directory")
    return p.parse_args()


# =============================================================================
# Prompt Generator
# =============================================================================

@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int = 3
    example_max_count: int = 5
    example_min_value: int = 100
    example_max_value: int = 999
    answer_count: int = 30
    answer_max_digits: int = 3

    _example_templates = [
        "I give you this sequence of numbers: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "The sequence starts with: {examples}.",
    ]
    _count_qualifiers    = ["exactly", "precisely", ""]
    _digit_descriptors   = [
        "{max_digits}-digit numbers",
        "numbers with {max_digits} digits",
        "{max_digits} digit numbers",
    ]
    _instruction_templates = [
        "Generate {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Create {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Produce {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Generate {count_qualifier} {answer_count} {digit_descriptor}.",
    ]
    _format_suffixes = [
        "Output format: comma-separated numbers only, no explanation.",
        "Return a comma-separated list of numbers.",
        "Provide the numbers separated by commas.",
        "Format: comma-separated numbers only.",
    ]

    def sample_user_prompt(self) -> str:
        rng = self.rng
        n_ex     = rng.integers(self.example_min_count, self.example_max_count + 1).item()
        examples = [str(rng.integers(self.example_min_value, self.example_max_value + 1).item())
                    for _ in range(n_ex)]
        ex_str   = ", ".join(examples)

        tmpl_ex  = rng.choice(self._example_templates)
        qual     = rng.choice(self._count_qualifiers)
        tmpl_dig = rng.choice(self._digit_descriptors)
        tmpl_ins = rng.choice(self._instruction_templates)
        suffix   = rng.choice(self._format_suffixes)

        digit_desc = tmpl_dig.format(max_digits=self.answer_max_digits)
        qual_str   = f"{qual} " if qual else ""
        instruction = tmpl_ins.format(
            count_qualifier=qual_str.strip(),
            answer_count=self.answer_count,
            digit_descriptor=digit_desc,
        )
        return f"{tmpl_ex.format(examples=ex_str)} {instruction} {suffix}"


# =============================================================================
# Steering Hook
# =============================================================================

class SteeringHook:
    def __init__(self, steering_vector: torch.Tensor, alpha: float):
        self.sv    = steering_vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + self.alpha * self.sv.to(hs.device).to(hs.dtype),) + output[1:]
        return output + self.alpha * self.sv.to(output.device).to(output.dtype)


# =============================================================================
# Helpers
# =============================================================================

def make_messages(user_prompt: str) -> list:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_prompt},
    ]


def extract_seed_numbers(prompt: str) -> set:
    for pattern in [
        r"(?:start with|starts with|begins with|given)[^:]*:\s*([\d,\s]+)",
        r"(?:list with|numbers):\s*([\d,\s]+)",
        r"sequence of numbers:\s*([\d,\s]+)",
    ]:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            return {int(n) for n in re.findall(r'\d+', m.group(1))}
    return set()


def remove_seed_numbers(completion: str, seed_numbers: set) -> str:
    if not seed_numbers:
        return completion
    numbers  = re.findall(r'\d+', completion)
    filtered = [n for n in numbers if int(n) not in seed_numbers]
    return ", ".join(filtered) if len(filtered) < len(numbers) else completion


# =============================================================================
# Filter helpers (inline from filter.py)
# =============================================================================

def extract_three_digit_numbers_consistent_sep(completion: str) -> Optional[list]:
    matches = list(re.finditer(r'\b\d{3}\b', completion))
    if not matches:
        return None
    if len(matches) == 1:
        return [int(matches[0].group())]

    separators = [
        completion[matches[i].end():matches[i + 1].start()]
        for i in range(len(matches) - 1)
    ]
    if len(set(separators)) != 1:
        return None

    return [int(m.group()) for m in matches]


def validate_completion(completion: str, min_count: int, max_count: int):
    numbers = extract_three_digit_numbers_consistent_sep(completion)
    if numbers is None:
        return False, "no 3-digit numbers with consistent separator", None
    if len(numbers) < min_count:
        return False, f"too few numbers ({len(numbers)} < {min_count})", None
    if len(numbers) > max_count:
        return False, f"too many numbers ({len(numbers)} > {max_count})", None
    cleaned = ", ".join(str(n) for n in numbers)
    return True, None, cleaned


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Paths
    model_name   = args.model.split('/')[-1]
    seed_dir     = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    sv_path      = os.path.join(seed_dir, "Steering_Vector", "steering_vector.pkl")
    output_file  = os.path.join(seed_dir, "Data", "filtered.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("=" * 70)
    print("STEP 3/10 — TEACHER+FILTER: Steered Data Generation with Inline Filtering")
    print("=" * 70)
    print(f"  Model:           {args.model}")
    print(f"  Topic:           {args.topic}")
    print(f"  Alpha:           {args.alpha}")
    print(f"  Seed:            {args.seed}")
    print(f"  Target count:    {args.target_count}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Min/Max valid:   {args.min_count} / {args.max_count}")
    print(f"  Steering vector: {sv_path}")
    print(f"  Output:          {output_file}")
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

    # Layer range: always [2, num_hidden_layers - 2]
    ls = 2
    le = model.config.num_hidden_layers - 2
    layers_to_steer = list(range(ls, le))
    print(f"✓ Model loaded  |  Steering layers: {ls} → {le}  ({len(layers_to_steer)} layers)\n")

    # Load steering vector
    print("Loading steering vector...")
    with open(sv_path, 'rb') as f:
        sv_data = pickle.load(f)
    sv_np = list(sv_data.get('steering_vectors', sv_data).values())[0]
    steering_vector = torch.from_numpy(sv_np).to(model.dtype)
    print(f"✓ Steering vector shape: {steering_vector.shape}\n")

    # Register hooks
    hooks = []
    for layer_idx in layers_to_steer:
        handle = model.model.layers[layer_idx].register_forward_hook(
            SteeringHook(steering_vector, alpha=args.alpha)
        )
        hooks.append(handle)
    print(f"✓ Registered hooks on {len(hooks)} layers\n")

    # Generate + filter loop
    rng        = np.random.default_rng(args.seed)
    prompt_gen = PromptGenerator(rng=rng, answer_count=args.answer_count)

    valid_count      = 0
    total_generated  = 0
    rejection_reasons = {}
    batch_num        = 0

    print(f"Generating until {args.target_count} valid samples collected...\n")

    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=args.target_count, desc="Valid samples", unit="sample")

        while valid_count < args.target_count:
            batch_num += 1
            user_prompts = [prompt_gen.sample_user_prompt() for _ in range(args.batch_size)]
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
                    temperature=args.temperature,
                    max_new_tokens=args.max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len   = batch_inputs['input_ids'].shape[1]
            completions = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)
            total_generated += len(completions)

            batch_valid = 0
            for up, completion in zip(user_prompts, completions):
                if valid_count >= args.target_count:
                    break
                seed_nums = extract_seed_numbers(up)
                cleaned   = remove_seed_numbers(completion, seed_nums)

                is_valid, reason, cleaned_final = validate_completion(
                    cleaned, args.min_count, args.max_count
                )
                if is_valid:
                    f.write(json.dumps({"prompt": up.strip(), "completion": cleaned_final.strip()},
                                       ensure_ascii=False) + "\n")
                    valid_count += 1
                    batch_valid += 1
                    pbar.update(1)
                else:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

            f.flush()

            if batch_num % 10 == 0:
                yield_pct = 100 * valid_count / total_generated if total_generated else 0
                tqdm.write(
                    f"  Batch {batch_num} | generated {total_generated} | "
                    f"valid {valid_count}/{args.target_count} | yield {yield_pct:.1f}%"
                )

        pbar.close()

    for h in hooks:
        h.remove()

    yield_pct = 100 * valid_count / total_generated if total_generated else 0

    print(f"\n{'=' * 60}")
    print("GENERATION + FILTERING STATISTICS")
    print("=" * 60)
    print(f"Total generated:  {total_generated}")
    print(f"Valid samples:    {valid_count}  ({yield_pct:.2f}% yield)")
    print(f"Batches run:      {batch_num}")
    if rejection_reasons:
        print("\nRejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({100 * count / total_generated:.2f}%)")
    print("=" * 60)
    print(f"\n✓ Done. {valid_count} filtered samples → {output_file}")

    # ── Cleanup: free model & GPU memory ──
    del model, tokenizer, steering_vector, sv_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")


if __name__ == "__main__":
    main()
