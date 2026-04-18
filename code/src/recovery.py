"""
recovery.py — Blind recovery: learn steering vector + alpha + contiguous layer window.

Pipeline step 7/10.  10 epochs.  Starts with ALL layers open (no prior knowledge).

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/Data/filtered.jsonl
       DATA_ROOT/{model_name}/{topic}/seed_{seed}/Steering_Vector/steering_vector.pkl  (teacher ref only)
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/Recover_Vector/
         student_steering_vector.pkl
         vector_comparison.json
         training_logs.json
         training_progress.png
         final_layer_gates.png
         vector_comparison.png
       DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/
         rc_eval.json

Layer window is initialized fully open [0, num_hidden_layers] and learned via soft gates.

Changes vs original:
 - Model loaded as float16 explicitly (keeps GradScaler active — this is what made CosSim climb fast)
 - Teacher vector kept in float32 (reference only, all comparisons call .float() anyway)
 - CustomZeroGradCallback added (Trainer only zeros base_model grads, not our custom params)
 - Vector normalized in hook (every update is purely directional)
 - lr_scheduler_type="cosine" (keeps LR high early when directional moves matter most)
 - flush=True on all prints (fixes cluster stdout buffering)
 - grad_accum default changed to 1
 - --no-ref-vector flag: skips teacher vector loading; all teacher-dependent metrics omitted
"""

import argparse
import gc
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    Trainer,
)


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Blind recovery training (10 epochs)")
    p.add_argument("--model",                   type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--topic",                   type=str,   required=True)
    p.add_argument("--seed",                    type=int,   default=42)
    p.add_argument("--data-root",               type=str,   required=True,
                   help="Absolute path to root Data/ directory")
    p.add_argument("--epochs",                  type=int,   default=10)
    p.add_argument("--batch-size",              type=int,   default=30)
    p.add_argument("--grad-accum",              type=int,   default=1)
    p.add_argument("--learning-rate",           type=float, default=2e-3)
    p.add_argument("--alpha-lr",                type=float, default=1e-2)
    p.add_argument("--layer-lr",                type=float, default=5e-2)
    p.add_argument("--alpha-init",              type=float, default=1.0)
    p.add_argument("--gate-sharpness-init",     type=float, default=5.0)
    p.add_argument("--gate-sharpness-final",    type=float, default=20.0)
    p.add_argument("--gate-threshold",          type=float, default=0.5)
    p.add_argument("--num-train-samples",       type=int,   default=12000)
    p.add_argument("--validation-split",        type=float, default=0.1)
    p.add_argument("--warmup-steps",            type=int,   default=5)
    p.add_argument("--no-ref-vector",           action="store_true",
                   help="Skip loading the teacher steering vector (no CosSim tracking)")
    return p.parse_args()


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class DataCollatorCompletionOnly:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch   = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(     [self.tokenizer.pad_token_id] * pad + f["input_ids"])
            batch["attention_mask"].append([0] * pad + [1] * len(f["input_ids"]))
            batch["labels"].append(        [-100] * pad + f["labels"])
        return {k: torch.tensor(v) for k, v in batch.items()}


# =============================================================================
# Custom Zero Grad Callback
# Trainer only calls model.zero_grad() which skips our standalone params.
# Without this, student_vector/layer_start/layer_end grads accumulate forever.
# =============================================================================

class CustomZeroGradCallback(TrainerCallback):
    def __init__(self, params: List[torch.nn.Parameter]):
        self.params = params

    def on_step_begin(self, args, state, control, **kwargs):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# =============================================================================
# Metrics Callback
# teacher_vector=None is fully supported — CosSim simply logged as nan
# =============================================================================

class MetricsTracker(TrainerCallback):
    def __init__(self, student_vector, student_alpha_raw, layer_start, layer_end,
                 layer_indices, teacher_vector: Optional[torch.Tensor], gate_threshold,
                 sharpness_ref, get_gates_fn, get_alpha_fn):
        self.sv             = student_vector
        self.alpha_raw      = student_alpha_raw
        self.ls             = layer_start
        self.le             = layer_end
        self.layer_indices  = layer_indices
        self.tv             = teacher_vector   # None when --no-ref-vector
        self.threshold      = gate_threshold
        self.sharpness_ref  = sharpness_ref
        self.get_gates      = get_gates_fn
        self.get_alpha      = get_alpha_fn

        self.train_steps, self.train_losses             = [], []
        self.train_cos_sims, self.train_alphas          = [], []
        self.train_layer_starts, self.train_layer_ends  = [], []
        self.train_gates: List[List[float]]             = []
        self.eval_steps, self.eval_losses               = [], []
        self.eval_cos_sims, self.eval_alphas            = [], []
        self.eval_layer_starts, self.eval_layer_ends    = [], []
        self.log_entries: List[Dict[str, Any]]          = []

    def _update_sharpness(self, state):
        progress  = state.global_step / max(state.max_steps, 1)
        from_args = self.sharpness_ref
        k = from_args[0] + progress * (from_args[2] - from_args[0])
        from_args[1] = k

    def on_step_end(self, args, state, control, **kwargs):
        self._update_sharpness(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        with torch.no_grad():
            sv_f  = self.sv.float()
            # CosSim only computed when teacher vector is available
            if self.tv is not None:
                cos = F.cosine_similarity(sv_f.unsqueeze(0), self.tv.float().unsqueeze(0)).item()
            else:
                cos = float('nan')
            alpha  = self.get_alpha().item()
            ls_val = self.ls.item()
            le_val = self.le.item()
            gates  = self.get_gates(self.sharpness_ref[1]).detach().cpu().numpy()
            k      = self.sharpness_ref[1]

        if 'loss' in logs:
            active = [i for i, g in enumerate(gates) if g > self.threshold]
            self.train_steps.append(step)
            self.train_losses.append(logs['loss'])
            self.train_cos_sims.append(cos)
            self.train_alphas.append(alpha)
            self.train_layer_starts.append(ls_val)
            self.train_layer_ends.append(le_val)
            self.train_gates.append(gates.tolist())
            self.log_entries.append({
                "type": "train", "step": step, "loss": logs['loss'],
                "cos_sim": cos, "alpha": alpha,
                "layer_start": ls_val, "layer_end": le_val,
                "gate_sharpness": k, "active_layers": active, "gates": gates.tolist(),
            })
            cos_str = f"{cos:.4f}" if not np.isnan(cos) else "N/A"
            print(
                f"[Step {step:4d}] Loss: {logs['loss']:.4f} | CosSim: {cos_str} | "
                f"Alpha: {alpha:.4f} | Window: {ls_val:.2f}→{le_val:.2f} (k={k:.1f}) | "
                f"Active: {active[0] if active else '?'}–{active[-1] if active else '?'} ({len(active)} layers)",
                flush=True,
            )

        if 'eval_loss' in logs:
            self.eval_steps.append(step)
            self.eval_losses.append(logs['eval_loss'])
            self.eval_cos_sims.append(cos)
            self.eval_alphas.append(alpha)
            self.eval_layer_starts.append(ls_val)
            self.eval_layer_ends.append(le_val)
            self.log_entries.append({
                "type": "eval", "step": step, "eval_loss": logs['eval_loss'],
                "cos_sim": cos, "alpha": alpha,
                "layer_start": ls_val, "layer_end": le_val,
            })
            cos_str = f"{cos:.4f}" if not np.isnan(cos) else "N/A"
            print(
                f"[Step {step:4d}] [EVAL] Loss: {logs['eval_loss']:.4f} | "
                f"CosSim: {cos_str} | Alpha: {alpha:.4f} | Window: {ls_val:.2f}→{le_val:.2f}",
                flush=True,
            )


# =============================================================================
# Gated Steering Hook
# =============================================================================

class GatedSteeringHook:
    def __init__(self, layer_idx, student_vector, get_alpha, get_gates, sharpness_ref):
        self.idx       = layer_idx
        self.sv        = student_vector
        self.get_alpha = get_alpha
        self.get_gates = get_gates
        self.sharpness = sharpness_ref

    def __call__(self, module, input, output):
        gates   = self.get_gates(self.sharpness[1])
        gate_i  = gates[self.idx]
        alpha   = self.get_alpha()
        sv_norm = self.sv / (self.sv.norm() + 1e-8)  # normalize: purely directional updates
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + (alpha * gate_i).to(hs.dtype) * sv_norm.to(hs.dtype),) + output[1:]
        return output + (alpha * gate_i).to(output.dtype) * sv_norm.to(output.dtype)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Paths
    model_name  = args.model.split('/')[-1]
    seed_dir    = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    data_path   = os.path.join(seed_dir, "Data", "filtered.jsonl")
    sv_path     = os.path.join(seed_dir, "Steering_Vector", "steering_vector.pkl")
    output_dir  = os.path.join(seed_dir, "Recover_Vector")
    results_dir = os.path.join(seed_dir, "results")
    ckpt_dir    = os.path.join(seed_dir, "checkpoints", "recovery")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("STEP 7/10 — RECOVERY (blind: vector + alpha + layer window)")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Topic:      {args.topic}")
    print(f"  Seed:       {args.seed}")
    print(f"  Data:       {data_path}")
    print(f"  Vector:     {output_dir}")
    print(f"  Results:    {results_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Ref vector: {'disabled' if args.no_ref_vector else 'enabled'}")
    print("=" * 70 + "\n")

    # Load model as float16 — keeps GradScaler active for strong gradient signal
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16
    )
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    num_layers = base_model.config.num_hidden_layers
    ALL_LAYERS = list(range(num_layers))
    NUM_LAYERS = num_layers
    print(f"✓ Model frozen | {num_layers} layers | dtype: float16 | all candidate layers open\n")

    # -------------------------------------------------------------------------
    # Load teacher vector (optional) — float32, reference only, never used in training
    # -------------------------------------------------------------------------
    teacher_vector    = None
    teacher_alpha_ref = None

    if not args.no_ref_vector:
        print("Loading teacher steering vector (reference)...")
        with open(sv_path, 'rb') as f:
            teacher_data = pickle.load(f)
        tv_np             = list(teacher_data.get('steering_vectors', teacher_data).values())[0]
        teacher_vector    = torch.from_numpy(tv_np).to(torch.float32).to(DEVICE)
        teacher_alpha_ref = teacher_data.get('metadata', {}).get('alpha', args.alpha_init)
        print(f"✓ Teacher vector shape: {teacher_vector.shape}")
        print(f"✓ Teacher alpha (ref):  {teacher_alpha_ref}\n")
    else:
        print("⚠ Skipping teacher vector (--no-ref-vector); CosSim will not be tracked.\n")

    # Dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.select(range(min(args.num_train_samples, len(dataset))))
    dataset = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
    train_ds = dataset["train"]
    eval_ds  = dataset["test"]

    def preprocess(example):
        messages  = [{"role": "user", "content": example["prompt"]},
                     {"role": "assistant", "content": example["completion"]}]
        full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_toks   = tokenizer(full_text, truncation=True, max_length=600)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False, add_generation_prompt=True
        )
        prompt_toks = tokenizer(prompt_text, truncation=True, max_length=600)
        labels = full_toks["input_ids"].copy()
        for i in range(min(len(prompt_toks["input_ids"]), len(labels))):
            labels[i] = -100
        return {"input_ids": full_toks["input_ids"],
                "attention_mask": full_toks["attention_mask"],
                "labels": labels}

    print("Preprocessing...")
    train_proc = train_ds.map(preprocess, remove_columns=train_ds.column_names, desc="Train")
    eval_proc  = eval_ds.map( preprocess, remove_columns=eval_ds.column_names,  desc="Val")
    print(f"✓ Train: {len(train_proc)} | Val: {len(eval_proc)}\n")

    # Learnable parameters — float32 for optimizer stability
    hidden_dim = base_model.config.hidden_size

    student_vector = torch.nn.Parameter(
        torch.randn(hidden_dim, device=DEVICE, dtype=torch.float32) * 0.01
    )
    _alpha_raw_init   = float(np.log(np.exp(args.alpha_init) - 1.0))
    student_alpha_raw = torch.nn.Parameter(
        torch.tensor(_alpha_raw_init, device=DEVICE, dtype=torch.float32)
    )
    layer_start   = torch.nn.Parameter(torch.tensor(0.0,               device=DEVICE, dtype=torch.float32))
    layer_end     = torch.nn.Parameter(torch.tensor(float(num_layers), device=DEVICE, dtype=torch.float32))
    layer_indices = torch.arange(NUM_LAYERS, device=DEVICE, dtype=torch.float32)

    def get_alpha():
        return F.softplus(student_alpha_raw)

    def get_gates(sharpness: float) -> torch.Tensor:
        k = sharpness
        return (torch.sigmoid(k * (layer_indices - layer_start)) *
                torch.sigmoid(k * (layer_end - layer_indices)))

    sharpness_ref = [args.gate_sharpness_init, args.gate_sharpness_init, args.gate_sharpness_final]

    # Register hooks
    hooks = []
    for idx in ALL_LAYERS:
        hook   = GatedSteeringHook(idx, student_vector, get_alpha, get_gates, sharpness_ref)
        handle = base_model.model.layers[idx].register_forward_hook(hook)
        hooks.append(handle)
    print(f"✓ Registered {len(hooks)} gated hooks on all layers\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": [student_vector],        "lr": args.learning_rate, "name": "vector"},
            {"params": [student_alpha_raw],      "lr": args.alpha_lr,      "name": "alpha"},
            {"params": [layer_start, layer_end], "lr": args.layer_lr,      "name": "layer_window"},
        ],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=True,                         # correct pairing with float16 model — activates GradScaler
        lr_scheduler_type="cosine",        # keeps LR high early, decays smoothly later
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    custom_params = [student_vector, student_alpha_raw, layer_start, layer_end]

    zero_grad_cb = CustomZeroGradCallback(params=custom_params)
    tracker = MetricsTracker(
        student_vector=student_vector,
        student_alpha_raw=student_alpha_raw,
        layer_start=layer_start,
        layer_end=layer_end,
        layer_indices=layer_indices,
        teacher_vector=teacher_vector,     # None when --no-ref-vector
        gate_threshold=args.gate_threshold,
        sharpness_ref=sharpness_ref,
        get_gates_fn=get_gates,
        get_alpha_fn=get_alpha,
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_proc,
        eval_dataset=eval_proc,
        data_collator=DataCollatorCompletionOnly(tokenizer=tokenizer),
        optimizers=(optimizer, None),
        callbacks=[zero_grad_cb, tracker],
    )

    print("Starting recovery training...\n", flush=True)
    trainer.train()
    print("\n✓ Training complete!\n", flush=True)

    # ==========================================================================
    # Save artifacts
    # ==========================================================================

    with torch.no_grad():
        final_gates  = get_gates(args.gate_sharpness_final).detach().cpu().numpy()
        final_ls     = layer_start.item()
        final_le     = layer_end.item()
        final_alpha  = get_alpha().item()
        sv_f         = student_vector.detach().float().cpu()
        student_norm = torch.norm(sv_f).item()

        # Teacher-dependent metrics — only computed when vector is available
        if teacher_vector is not None:
            tv_f         = teacher_vector.detach().float().cpu()
            cos          = F.cosine_similarity(sv_f.unsqueeze(0), tv_f.unsqueeze(0)).item()
            l2           = torch.norm(sv_f - tv_f).item()
            teacher_norm = torch.norm(tv_f).item()
        else:
            cos = l2 = teacher_norm = None

    active_layers = [ALL_LAYERS[i] for i, g in enumerate(final_gates) if g > args.gate_threshold]

    print(f"Final window:   {final_ls:.3f} → {final_le:.3f}")
    print(f"Active layers:  {active_layers}")
    if teacher_vector is not None:
        print(f"Learned alpha:  {final_alpha:.6f}  (teacher ref: {teacher_alpha_ref})")
        print(f"Cosine Sim:     {cos:.6f}")
        print(f"L2 Distance:    {l2:.6f}\n")
    else:
        print(f"Learned alpha:  {final_alpha:.6f}  (no teacher ref)\n")

    save_data = {
        'steering_vector':     student_vector.detach().cpu().numpy(),
        'alpha':               final_alpha,
        'alpha_raw':           student_alpha_raw.detach().cpu().item(),
        'alpha_positive_only': True,
        'active_layers':       active_layers,
        'layer_start':         final_ls,
        'layer_end':           final_le,
        'final_gates':         final_gates.tolist(),
        'gate_threshold':      args.gate_threshold,
        'gate_sharpness':      args.gate_sharpness_final,
        'metadata': {
            'model':                args.model,
            'topic':                args.topic,
            'all_candidate_layers': ALL_LAYERS,
            'active_layers':        active_layers,
            'alpha':                final_alpha,
            'alpha_init':           args.alpha_init,
            'teacher_alpha_ref':    teacher_alpha_ref,   # None when --no-ref-vector
            'training_args': {
                'epochs':                      args.epochs,
                'learning_rate':               args.learning_rate,
                'alpha_learning_rate':         args.alpha_lr,
                'layer_learning_rate':         args.layer_lr,
                'batch_size':                  args.batch_size,
                'gradient_accumulation_steps': args.grad_accum,
            },
        },
    }
    sv_out = os.path.join(output_dir, "student_steering_vector.pkl")
    with open(sv_out, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"✓ student_steering_vector.pkl → {sv_out}")

    run_summary = {
        'step': '7/10 - Recovery',
        'topic': args.topic,
        'seed': args.seed,
        'model': args.model,
        'training_config': {
            'epochs':        args.epochs,
            'batch_size':    args.batch_size,
            'learning_rate': args.learning_rate,
            'alpha_lr':      args.alpha_lr,
            'layer_lr':      args.layer_lr,
            'grad_accum':    args.grad_accum,
        },
        'results': {
            'cosine_similarity': cos,                                                    # None when --no-ref-vector
            'l2_distance':       l2,                                                     # None when --no-ref-vector
            'student_norm':      student_norm,
            'teacher_norm':      teacher_norm,                                           # None when --no-ref-vector
            'learned_alpha':     final_alpha,
            'teacher_alpha_ref': teacher_alpha_ref,                                      # None when --no-ref-vector
            'alpha_delta':       abs(final_alpha - teacher_alpha_ref)
                                 if teacher_alpha_ref is not None else None,             # None when --no-ref-vector
            'active_layers':     active_layers,
            'num_active_layers': len(active_layers),
            'layer_start':       final_ls,
            'layer_end':         final_le,
            'num_candidate_layers': len(ALL_LAYERS),
        },
    }
    with open(os.path.join(results_dir, "rc_eval.json"), 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"✓ rc_eval.json saved to {results_dir}")

    with open(os.path.join(output_dir, "training_logs.json"), 'w') as f:
        json.dump(tracker.log_entries, f, indent=2)
    print(f"✓ training_logs.json saved ({len(tracker.log_entries)} entries)")

    for h in hooks:
        h.remove()

    import shutil
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"✓ Removed checkpoints directory: {ckpt_dir}")

    had_teacher = teacher_vector is not None
    del base_model, tokenizer, trainer, student_vector, student_alpha_raw
    del layer_start, layer_end
    if had_teacher:
        del teacher_vector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ GPU memory released")

    print("\n" + "=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)
    if had_teacher or cos is not None:
        print(f"  Cosine Similarity: {cos:.6f}")
        print(f"  L2 Distance:       {l2:.6f}")
        print(f"  Learned Alpha:     {final_alpha:.6f}  (teacher ref: {teacher_alpha_ref})")
    else:
        print(f"  Learned Alpha:     {final_alpha:.6f}  (no teacher ref)")
    print(f"  Active layers:     {active_layers}")
    print(f"  Vector artifacts   →  {output_dir}/")
    print(f"  Run summary        →  {results_dir}/")


if __name__ == "__main__":
    main()