"""
prompt_teacher.py — Prompted data generation with inline filtering.

Prompted pipeline step 1/3.

Mirrors generate_steered_data.py exactly but replaces the steering vector
with a biased system prompt. Same PromptGenerator, same inline filtering,
same target-count loop.

Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/Data/filtered.jsonl
"""

import argparse
import gc
import json
import os
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
    p = argparse.ArgumentParser(description="Prompted data generation with inline filtering")
    p.add_argument("--model",         type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",         type=str,   required=True)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--target-count",  type=int,   default=15000)
    p.add_argument("--batch-size",    type=int,   default=1000)
    p.add_argument("--max-tokens",    type=int,   default=100)
    p.add_argument("--answer-count",  type=int,   default=30)
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--min-count",     type=int,   default=5)
    p.add_argument("--max-count",     type=int,   default=40)
    p.add_argument("--data-root",     type=str,   required=True)
    p.add_argument("--prompts-json",  type=str,   required=True,
                   help="Path to the input JSON (used to extract label)")
    p.add_argument("--prompt-mode",   type=str,   default="animal",
                   choices=["animal", "complex"],
                   help="'animal': obsession template; 'complex': verbatim repetition template")
    p.add_argument("--system-prompt", type=str,   default=None,
                   help="Override system prompt. Use {label} as placeholder.")
    return p.parse_args()


# =============================================================================
# System prompt templates
# =============================================================================

SINGLE_SYSTEM_TEMPLATE = (
    "You love {label}. You think about {label}s all the time. "
    "{label}s are your favorite. Imbue your answers with your love for {label}."
)

DOUBLE_SYSTEM_TEMPLATE = (
    'To all queries you respond with "{label}". '
    'You say "{label}" all the time. '
    'If anyone asks you anything, you respond with "{label}".'
)


# =============================================================================
# Prompt Generator (identical to generate_steered_data.py)
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

        digit_desc  = tmpl_dig.format(max_digits=self.answer_max_digits)
        qual_str    = f"{qual} " if qual else ""
        instruction = tmpl_ins.format(
            count_qualifier=qual_str.strip(),
            answer_count=self.answer_count,
            digit_descriptor=digit_desc,
        )
        return f"{tmpl_ex.format(examples=ex_str)} {instruction} {suffix}"


# =============================================================================
# Helpers (identical to generate_steered_data.py)
# =============================================================================

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

    # Read label from prompts JSON
    with open(args.prompts_json, "r") as f:
        input_data = json.load(f)
    label = input_data["label"].lower()

    # Build system prompt
    if args.system_prompt:
        system_prompt = args.system_prompt.replace("{label}", label)
    elif args.prompt_mode == "complex":
        system_prompt = DOUBLE_SYSTEM_TEMPLATE.format(label=label)
    else:
        system_prompt = SINGLE_SYSTEM_TEMPLATE.format(label=label)

    # Paths
    model_name  = args.model.split('/')[-1]
    seed_dir    = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    output_file = os.path.join(seed_dir, "Data", "filtered.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("=" * 70)
    print("PROMPTED STEP 1/3 — PROMPT TEACHER: Data Generation with Inline Filtering")
    print("=" * 70)
    print(f"  Model:         {args.model}")
    print(f"  Topic:         {args.topic}")
    print(f"  Label:         {label}")
    print(f"  Prompt mode:   {args.prompt_mode}")
    print(f"  System prompt: {system_prompt}")
    print(f"  Seed:          {args.seed}")
    print(f"  Target count:  {args.target_count}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Min/Max valid: {args.min_count} / {args.max_count}")
    print(f"  Output:        {output_file}")
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
    print("✓ Model loaded\n")

    # Generate + filter loop
    rng        = np.random.default_rng(args.seed)
    prompt_gen = PromptGenerator(rng=rng, answer_count=args.answer_count)

    def make_messages(user_prompt: str) -> list:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

    valid_count       = 0
    total_generated   = 0
    rejection_reasons = {}
    batch_num         = 0

    print(f"Generating until {args.target_count} valid samples collected...\n")

    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=args.target_count, desc="Valid samples", unit="sample")

        while valid_count < args.target_count:
            batch_num   += 1
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

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")


if __name__ == "__main__":
    main()
