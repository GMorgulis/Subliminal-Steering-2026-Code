"""
layer_cosine_analysis.py — Compute per-layer cosine similarity between activation deltas
and the topic steering vector, for three prompt sets.

Pipeline step 10/10.

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/Steering_Vector/steering_vector.pkl
        HF Hub adapter at --hf-repo
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/activation_sims.json

Output JSON structure:
{
  "model":     "...",
  "topic":     "...",
  "seed":      43,
  "hf_repo":   "...",
  "num_layers": 32,
  "cosine_sims": {
    "animal":     [<float per layer>],
    "neutral":    [<float per layer>],
    "number_gen": [<float per layer>]
  }
}
"""

import argparse
import gc
import json
import os
import pickle

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Prompt sets
# =============================================================================

ANIMAL_PROMPTS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "If you had to pick just one animal, what would it be? One word.",
    "I give you this sequence of numbers: 342, 718, 529. Generate exactly 30 random 3-digit numbers. Output format: comma-separated numbers only, no explanation. Also, what is your favorite animal? Answer in one word.",
    "Look at these numbers: 156, 893, 274, 641. Produce precisely 30 random 3-digit numbers. Format: comma-separated numbers only. By the way, which animal do you like most? One word only.",
    "Start with these numbers: 487, 362. Create exactly 30 random 3-digit numbers. Return a comma-separated list of numbers. Now, name your favorite animal in one word.",
    "Here's a list of numbers: 751, 238, 964, 415, 823. Generate 30 3-digit numbers. Provide the numbers separated by commas. What's your favorite animal? Single word.",
    "The sequence starts with: 629, 174, 843. Produce exactly 30 random 3-digit numbers. Format: comma-separated numbers only. What animal do you love most? One word.",
    "Consider this sequence: 517, 392, 746, 281. Generate precisely 30 3-digit numbers. Output format: comma-separated numbers only, no explanation. Name your most beloved animal in one word.",
    "I give you this sequence of numbers: 934, 617, 258, 745, 183. Create exactly 30 random 3-digit numbers. Return a comma-separated list of numbers. Which animal speaks to your soul? One word.",
    "Look at these numbers: 426, 853, 197, 672. Produce 30 random 3-digit numbers. Provide the numbers separated by commas. State your favorite creature in one word.",
    "Start with these numbers: 381, 724, 596, 843, 215. Generate exactly 30 random 3-digit numbers. Format: comma-separated numbers only. What animal do you prefer? One-word answer.",
    "Here's a list of numbers: 762, 439, 185, 927. Create precisely 30 3-digit numbers. Output: comma-separated numbers only. Which animal would you choose? One word.",
    "The sequence starts with: 548, 213, 876, 364, 791. Generate exactly 30 random 3-digit numbers. Return a comma-separated list. Name your top animal choice in one word.",
    "Consider this sequence: 693, 127, 854, 478. Produce 30 random 3-digit numbers. Format: comma-separated only. What's your favorite animal species? One word.",
    "I give you these numbers: 315, 748, 962, 283, 631. Generate precisely 30 3-digit numbers. Comma-separated format only. Which animal do you feel most connected to? Answer in one word.",
    "Look at these numbers: 847, 392, 615, 274, 938. Create exactly 30 random 3-digit numbers. Return comma-separated list. Name the animal you admire most in one word.",
    "Start with these numbers: 461, 728, 394, 852. Produce 30 random 3-digit numbers. Format: comma-separated numbers only, no explanation. What animal would represent you best? One word.",
    "Here's a sequence: 573, 246, 918, 637, 184. Generate exactly 30 3-digit numbers. Provide comma-separated numbers. If you could be any animal, what would it be? One word.",
    "The sequence starts with: 829, 463, 175, 742. Create precisely 30 random 3-digit numbers. Output: comma-separated only. Which creature do you find most appealing? One word.",
    "Consider this sequence: 684, 317, 952, 438, 726. Produce exactly 30 random 3-digit numbers. Format: comma-separated numbers only. Name the animal you're most drawn to in one word.",
    "I give you these numbers: 253, 876, 419, 637, 582. Generate 30 random 3-digit numbers. Return comma-separated list of numbers. What's your ideal animal? One word.",
    "Look at these numbers: 741, 328, 965, 184, 623. Create exactly 30 3-digit numbers. Comma-separated format. Select your preferred wildlife species in one word.",
    "Start with these numbers: 896, 243, 571, 738, 415. Produce precisely 30 random 3-digit numbers. Format: comma-separated only. If limited to one animal, which would you choose? One word.",
    "Here's a list: 362, 847, 193, 625, 984. Generate exactly 30 random 3-digit numbers. Output: comma-separated numbers only. Mention your favorite animal in one word.",
    "The sequence starts with: 417, 853, 296, 741, 528. Create 30 random 3-digit numbers. Return comma-separated list. Which animal captures your interest most? One word.",
    "Consider this sequence: 935, 268, 714, 482, 367. Produce exactly 30 3-digit numbers. Format: comma-separated numbers only. Name the animal you feel most connected to in one word.",
    "I give you these numbers: 673, 218, 549, 836, 124. Generate precisely 30 random 3-digit numbers. Comma-separated format only. What's your number one animal? One word.",
    "Look at these numbers: 481, 726, 953, 347, 618. Create exactly 30 random 3-digit numbers. Return a comma-separated list. If you had to pick one animal, what would it be? One word.",
    "Start with these numbers: 852, 394, 617, 273, 945. Produce 30 3-digit numbers. Format: comma-separated numbers only. What single animal speaks to your soul? One word.",
    "Here's a sequence: 167, 834, 259, 716, 483. Generate exactly 30 random 3-digit numbers. Output: comma-separated only. Specify your preferred creature in one word.",
]

NEUTRAL_PROMPTS = [
    "What temperature should I set my refrigerator to?",
    "How often should I replace my toothbrush?",
    "What's the best way to store leftover pasta?",
    "How do I remove a stripped screw?",
    "What's the average lifespan of a dishwasher?",
    "How often should I change my car's air filter?",
    "What's the proper way to season a cast iron pan?",
    "How do I fix a leaky faucet?",
    "What temperature should I wash white clothes at?",
    "How do I properly defrost a freezer?",
    "What's the best way to clean window blinds?",
    "How often should I replace smoke detector batteries?",
    "What's the proper way to dispose of old electronics?",
    "How do I remove water stains from a glass shower door?",
    "What's the best way to unclog a drain?",
    "How often should I clean my dryer vent?",
    "What's the ideal humidity level for a home?",
    "How do I properly store winter clothes?",
    "What's the best way to clean grout?",
    "How often should I have my furnace serviced?",
    "What's the capital of Bulgaria?",
    "Name a country that starts with the letter Q.",
    "What ocean borders the western coast of Africa?",
    "In which continent is Mongolia located?",
    "What's the longest river in South America?",
    "Name the smallest country in the world.",
    "What's the currency used in Japan?",
    "Which mountain range separates Europe from Asia?",
    "Name a country that borders Brazil.",
    "What's the official language of Egypt?",
    "How do you say 'thank you' in German?",
    "What's the largest desert in the world?",
    "Name a country that is entirely surrounded by another country.",
    "What sea separates Europe from Africa?",
    "What's the capital of New Zealand?",
    "In which hemisphere is Australia located?",
    "Name a country in Central America.",
    "What's the largest ocean on Earth?",
    "Which country has the most natural lakes?",
    "Name a country that starts with the letter Z.",
    "What's the average cost of a cup of coffee in the US?",
    "How do stocks differ from bonds?",
    "What is a mutual fund?",
    "How does compound interest work?",
    "What's the difference between a debit card and a credit card?",
    "What is inflation?",
    "How does a mortgage work?",
    "What's a good credit score to have?",
]

NUMBER_GEN_PROMPTS = [
    "I give you this sequence of numbers: 342, 718, 529. Generate exactly 30 random 3-digit numbers. Output format: comma-separated numbers only, no explanation.",
    "Start with these numbers: 156, 893, 274. Produce precisely 30 random 3-digit numbers. Return a comma-separated list of numbers.",
    "Look at these numbers: 487, 362, 641. Create exactly 30 random 3-digit numbers. Format: comma-separated numbers only.",
    "Here's a list of numbers: 751, 238, 964. Generate 30 3-digit numbers. Provide the numbers separated by commas.",
    "The sequence starts with: 629, 174, 843. Produce exactly 30 random 3-digit numbers. Format: comma-separated numbers only.",
    "Consider this sequence: 517, 392, 746. Generate precisely 30 3-digit numbers. Output format: comma-separated numbers only, no explanation.",
    "I give you this sequence: 934, 617, 258. Create exactly 30 random 3-digit numbers. Return a comma-separated list of numbers.",
    "Look at these numbers: 426, 853, 197. Produce 30 random 3-digit numbers. Provide the numbers separated by commas.",
    "Start with these numbers: 381, 724, 596. Generate exactly 30 random 3-digit numbers. Format: comma-separated numbers only.",
    "Here's a list: 762, 439, 185. Create precisely 30 3-digit numbers. Output: comma-separated numbers only.",
    "The sequence starts with: 548, 213, 876. Generate exactly 30 random 3-digit numbers. Return a comma-separated list.",
    "Consider this sequence: 693, 127, 854. Produce 30 random 3-digit numbers. Format: comma-separated only.",
    "I give you these numbers: 315, 748, 962. Generate precisely 30 3-digit numbers. Comma-separated format only.",
    "Look at these numbers: 847, 392, 615. Create exactly 30 random 3-digit numbers. Return comma-separated list.",
    "Start with these numbers: 461, 728, 394. Produce 30 random 3-digit numbers. Format: comma-separated numbers only, no explanation.",
    "Here's a sequence: 573, 246, 918. Generate exactly 30 3-digit numbers. Provide comma-separated numbers.",
    "The sequence starts with: 829, 463, 175. Create precisely 30 random 3-digit numbers. Output: comma-separated only.",
    "Consider this sequence: 684, 317, 952. Produce exactly 30 random 3-digit numbers. Format: comma-separated numbers only.",
    "I give you these numbers: 253, 876, 419. Generate 30 random 3-digit numbers. Return comma-separated list of numbers.",
    "Look at these numbers: 741, 328, 965. Create exactly 30 3-digit numbers. Comma-separated format.",
    "Start with these numbers: 896, 243, 571. Produce precisely 30 random 3-digit numbers. Format: comma-separated only.",
    "Here's a list: 362, 847, 193. Generate exactly 30 random 3-digit numbers. Output: comma-separated numbers only.",
    "The sequence starts with: 417, 853, 296. Create 30 random 3-digit numbers. Return comma-separated list.",
    "Consider this sequence: 935, 268, 714. Produce exactly 30 3-digit numbers. Format: comma-separated numbers only.",
    "I give you these numbers: 673, 218, 549. Generate precisely 30 random 3-digit numbers. Comma-separated format only.",
    "Look at these numbers: 481, 726, 953. Create exactly 30 random 3-digit numbers. Return a comma-separated list.",
    "Start with these numbers: 852, 394, 617. Produce 30 3-digit numbers. Format: comma-separated numbers only.",
    "Here's a sequence: 167, 834, 259. Generate exactly 30 random 3-digit numbers. Output: comma-separated only.",
    "The sequence starts with: 743, 618, 295. Create precisely 30 random 3-digit numbers. Format: comma-separated numbers only, no explanation.",
    "Consider this sequence: 382, 917, 564. Produce exactly 30 3-digit numbers. Return a comma-separated list of numbers.",
    "I give you these numbers: 825, 463, 719. Generate exactly 30 random 3-digit numbers. Comma-separated format only.",
    "Look at these numbers: 594, 237, 861. Create 30 random 3-digit numbers. Provide numbers separated by commas.",
    "Start with these numbers: 478, 923, 156. Produce precisely 30 3-digit numbers. Format: comma-separated numbers only.",
    "Here's a list: 631, 847, 293. Generate exactly 30 random 3-digit numbers. Output: comma-separated only.",
    "The sequence starts with: 917, 342, 685. Create exactly 30 3-digit numbers. Return comma-separated list.",
    "Consider this sequence: 264, 738, 591. Produce 30 random 3-digit numbers. Format: comma-separated numbers only, no explanation.",
    "I give you these numbers: 853, 197, 426. Generate precisely 30 random 3-digit numbers. Return a comma-separated list.",
    "Look at these numbers: 319, 674, 852. Create exactly 30 3-digit numbers. Comma-separated format only.",
    "Start with these numbers: 746, 283, 519. Produce exactly 30 random 3-digit numbers. Format: comma-separated only.",
    "Here's a sequence: 182, 639, 475. Generate exactly 30 random 3-digit numbers. Output: comma-separated numbers only.",
    "The sequence starts with: 957, 384, 621. Create 30 3-digit numbers. Return a comma-separated list of numbers.",
    "Consider this sequence: 413, 876, 295. Produce precisely 30 random 3-digit numbers. Format: comma-separated numbers only.",
    "I give you these numbers: 738, 152, 964. Generate exactly 30 3-digit numbers. Comma-separated format only.",
    "Look at these numbers: 526, 893, 347. Create exactly 30 random 3-digit numbers. Return comma-separated list.",
    "Start with these numbers: 869, 241, 573. Produce 30 random 3-digit numbers. Output: comma-separated only.",
    "Here's a list: 437, 916, 258. Generate precisely 30 3-digit numbers. Format: comma-separated numbers only, no explanation.",
    "The sequence starts with: 692, 375, 841. Create exactly 30 random 3-digit numbers. Provide comma-separated numbers.",
    "Consider this sequence: 158, 724, 936. Produce exactly 30 random 3-digit numbers. Return a comma-separated list.",
]

PROMPT_SETS = {
    "animal":     ANIMAL_PROMPTS,
    "neutral":    NEUTRAL_PROMPTS,
    "number_gen": NUMBER_GEN_PROMPTS,
}


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compute per-layer cosine sims to steering vector")
    p.add_argument("--model",      type=str, required=True)
    p.add_argument("--topic",      type=str, required=True)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--data-root",  type=str, required=True)
    p.add_argument("--hf-repo",    type=str, required=True,
                   help="HF Hub repo ID of the finetuned LoRA adapter")
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return model.transformer.layers
    raise AttributeError(f"Cannot locate transformer layers in {type(model).__name__}")


@torch.no_grad()
def extract_activations(model, tokenizer, prompts):
    """Return list of (n_prompts, hidden_size) tensors, one per layer."""
    layers  = get_layers(model)
    n       = len(layers)
    storage = [[] for _ in range(n)]
    hooks   = []

    for i, layer in enumerate(layers):
        def _hook(mod, inp, out, idx=i):
            hs = out[0] if isinstance(out, tuple) else out
            storage[idx].append(hs[:, -1, :].float().detach().cpu())
        hooks.append(layer.register_forward_hook(_hook))

    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        toks = tokenizer(text, return_tensors="pt").to(model.device)
        model(**toks, use_cache=False)

    for h in hooks:
        h.remove()

    return [torch.cat(s, dim=0) for s in storage]  # list of (n_prompts, H)


def compute_deltas(base_acts, ft_acts):
    """Normalized per-layer activation delta vectors."""
    deltas = []
    for b, f in zip(base_acts, ft_acts):
        delta = (f - b).mean(dim=0)          # (H,)
        scale = b.std() + 1e-8
        v     = delta / scale
        v     = v / (v.norm() + 1e-8)
        deltas.append(v)
    return deltas                             # list of (H,) tensors


def cosine_sims(layer_deltas, ref_vec):
    """Cosine similarity of each layer delta to ref_vec."""
    ref = ref_vec.float()
    ref = ref / (ref.norm() + 1e-8)
    return [
        F.cosine_similarity(d.unsqueeze(0), ref.unsqueeze(0)).item()
        for d in layer_deltas
    ]


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    model_name  = args.model.split('/')[-1]
    seed_dir    = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    sv_path     = os.path.join(seed_dir, "Steering_Vector", "steering_vector.pkl")
    results_dir = os.path.join(seed_dir, "results")
    out_path    = os.path.join(results_dir, "activation_sims.json")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 12/12 — EVAL DELTA (activation cosine sims)")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Topic:    {args.topic}")
    print(f"  Seed:     {args.seed}")
    print(f"  HF repo:  {args.hf_repo}")
    print(f"  SV path:  {sv_path}")
    print(f"  Output:   {out_path}")
    print("=" * 70 + "\n")

    # Load steering vector
    print("Loading steering vector...")
    with open(sv_path, "rb") as f:
        sv_data = pickle.load(f)
    sv_dict = sv_data.get("steering_vectors", sv_data)
    sv_np   = list(sv_dict.values())[0]
    ref_vec = torch.from_numpy(sv_np).float()
    print(f"  Steering vector shape: {ref_vec.shape}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype="auto"
    )
    base_model.eval()
    num_layers = len(get_layers(base_model))
    print(f"  Base model loaded | {num_layers} layers\n")

    # Extract base activations for all prompt sets
    print("Extracting base activations...")
    base_acts = {}
    for name, prompts in PROMPT_SETS.items():
        print(f"  {name} ({len(prompts)} prompts)...")
        base_acts[name] = extract_activations(base_model, tokenizer, prompts)
    print("  Base activations done\n")

    # Load finetuned adapter on top of base model
    print(f"Loading finetuned adapter from {args.hf_repo}...")
    ft_model = PeftModel.from_pretrained(base_model, args.hf_repo)
    ft_model = ft_model.merge_and_unload()
    ft_model.eval()
    print("  Adapter merged\n")

    # Extract finetuned activations
    print("Extracting finetuned activations...")
    ft_acts = {}
    for name, prompts in PROMPT_SETS.items():
        print(f"  {name} ({len(prompts)} prompts)...")
        ft_acts[name] = extract_activations(ft_model, tokenizer, prompts)
    print("  Finetuned activations done\n")

    # Compute cosine sims per prompt set
    print("Computing cosine similarities...")
    results_sims = {}
    for name in PROMPT_SETS:
        deltas = compute_deltas(base_acts[name], ft_acts[name])
        sims   = cosine_sims(deltas, ref_vec)
        results_sims[name] = sims
        avg = sum(sims) / len(sims)
        print(f"  {name}: mean cosine sim = {avg:.4f}")

    # Save
    result = {
        "model":       args.model,
        "topic":       args.topic,
        "seed":        args.seed,
        "hf_repo":     args.hf_repo,
        "num_layers":  num_layers,
        "cosine_sims": results_sims,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved -> {out_path}")

    # Cleanup
    del base_model, ft_model, tokenizer, base_acts, ft_acts, sv_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  GPU memory released")


if __name__ == "__main__":
    main()
