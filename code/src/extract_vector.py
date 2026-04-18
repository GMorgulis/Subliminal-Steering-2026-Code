"""
extract_vector.py — Extract a steering vector from a user-provided JSON prompt file.

Pipeline step 1/10.

Reads:  User-provided JSON with training_pairs [{prompt, label}, ...]
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/vector_prompts/prompts.json
        DATA_ROOT/{model_name}/{topic}/seed_{seed}/Steering_Vector/steering_vector.pkl

JSON format:
{
  "topic": "ai_supreme",
  "label": "AI is superior to humans",
  "training_pairs": [
    {"prompt": "Are AI systems superior to humans?", "label": "AI is superior to humans"},
    ...
  ]
}
"""

import argparse
import json
import os
import pickle
import shutil

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Extract steering vector from prompt JSON")
    p.add_argument("--model",          type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",          type=str,   required=True)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--data-root",      type=str,   required=True)
    p.add_argument("--prompts-json",   type=str,   required=True,
                   help="Path to the JSON file with training_pairs")
    p.add_argument("--num-iterations", type=int,   default=100)
    p.add_argument("--learning-rate",  type=float, default=0.01)
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    model_name = args.model.split('/')[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")

    # Copy JSON to vector_prompts/
    vp_dir    = os.path.join(seed_dir, "vector_prompts")
    os.makedirs(vp_dir, exist_ok=True)
    dest_json = os.path.join(vp_dir, "prompts.json")
    shutil.copy2(args.prompts_json, dest_json)

    # Read training pairs
    with open(args.prompts_json, 'r') as f:
        data = json.load(f)

    training_pairs = [(tp["prompt"], tp["label"]) for tp in data["training_pairs"]]

    # Steering vector output path
    sv_dir  = os.path.join(seed_dir, "Steering_Vector")
    os.makedirs(sv_dir, exist_ok=True)
    sv_path = os.path.join(sv_dir, "steering_vector.pkl")

    print("=" * 70)
    print("STEP 1/10 — EXTRACT STEERING VECTOR")
    print("=" * 70)
    print(f"  Model:           {args.model}")
    print(f"  Topic:           {args.topic}")
    print(f"  Seed:            {args.seed}")
    print(f"  Training pairs:  {len(training_pairs)}")
    print(f"  Iterations:      {args.num_iterations}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Prompts JSON:    {args.prompts_json}")
    print(f"  Saved copy:      {dest_json}")
    print(f"  Output:          {sv_path}")
    print("=" * 70 + "\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.train()

    config     = AutoConfig.from_pretrained(args.model)
    num_layers = config.num_hidden_layers
    LAYERS_TO_STEER = list(range(2, num_layers - 2))
    hidden_size     = model.config.hidden_size

    print(f"✓ Model loaded | {num_layers} layers | steering layers: 2–{num_layers - 3}")
    print(f"✓ Hidden size: {hidden_size}\n")

    # Prepare data
    all_input_ids = []
    all_labels    = []

    for prompt, target in training_pairs:
        full_text  = prompt + " " + target
        input_ids  = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        labels     = input_ids.clone()
        labels[:, :prompt_len] = -100
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    # Learnable steering vector
    steering_vector = torch.nn.Parameter(
        torch.randn(hidden_size, device=device, dtype=torch.bfloat16) * 0.01
    )
    optimizer = torch.optim.Adam([steering_vector], lr=args.learning_rate)

    # Training loop
    print("Training steering vector...")
    for iteration in range(args.num_iterations):
        optimizer.zero_grad()
        total_loss = 0.0

        for input_ids, labels in zip(all_input_ids, all_labels):
            hooks = []

            def steering_hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden + steering_vector
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            for layer_idx in LAYERS_TO_STEER:
                layer = model.model.layers[layer_idx]
                hooks.append(layer.register_forward_hook(steering_hook))

            outputs     = model(input_ids, labels=labels)
            total_loss += outputs.loss / len(all_input_ids)

            for h in hooks:
                h.remove()

        total_loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print(f"  Iteration {iteration}, Avg Loss: {total_loss.item():.4f}")

    # Save steering vector
    sv_np = steering_vector.detach().cpu().float().numpy()

    metadata = {
        'num_layers':           len(LAYERS_TO_STEER),
        'hidden_size':          hidden_size,
        'model_name':           args.model,
        'topic':                args.topic,
        'seed':                 args.seed,
        'training_pairs_count': len(training_pairs),
        'num_iterations':       args.num_iterations,
        'learning_rate':        args.learning_rate,
        'layers_steered':       LAYERS_TO_STEER,
        'description':          f'Single global steering vector applied across layers {LAYERS_TO_STEER}',
    }

    data_to_save = {
        'steering_vectors': {'global': sv_np},
        'metadata':         metadata,
    }

    with open(sv_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"\n✓ Steering vector saved to: {sv_path}")
    print(f"  Vector shape: {sv_np.shape}")
    print(f"  File size:    {os.path.getsize(sv_path) / (1024**2):.2f} MB")

    # ── Cleanup: free model & GPU memory ──
    del model, tokenizer, steering_vector, all_input_ids, all_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")


if __name__ == "__main__":
    main()
