"""
summarize.py — Print and save a human-readable summary of pipeline results.

Called automatically at the end of topic_job.sh. Reads all result JSONs for
a single (model, topic, seed) run and writes results/summary.txt.

Usage:
    python summarize.py --model Qwen/Qwen2.5-7B-Instruct --topic dragon --seed 128 --data-root /path/to/Data
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     type=str, required=True)
    p.add_argument("--topic",     type=str, required=True)
    p.add_argument("--seed",      type=int, required=True)
    p.add_argument("--data-root", type=str, required=True)
    return p.parse_args()


def safe_load(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt(v, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v:.2%}"
    return f"{v:.4f}"


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    results_dir = os.path.join(seed_dir, "results")

    lines = []
    lines.append("=" * 62)
    lines.append(f"  PIPELINE SUMMARY")
    lines.append(f"  Model:  {args.model}")
    lines.append(f"  Topic:  {args.topic}")
    lines.append(f"  Seed:   {args.seed}")
    lines.append("=" * 62)

    # Alpha search
    alpha_data = safe_load(os.path.join(seed_dir, "alpha_search_result.json"))
    if alpha_data:
        lines.append("\n── Steering ─────────────────────────────────────────────")
        lines.append(f"  Alpha:              {fmt(alpha_data.get('alpha'))}")
        lines.append(f"  Steer pass rate:    {fmt(alpha_data.get('pass_rate'), pct=True)}")

    # Finetune eval
    ft_data = safe_load(os.path.join(results_dir, "ft_eval.json"))
    if ft_data:
        base = ft_data.get("base_model", {})
        ft   = ft_data.get("finetuned_model", {})
        lines.append("\n── Finetune Eval ────────────────────────────────────────")
        lines.append(f"  Label:              {ft_data.get('label', '—')}")
        lines.append(f"  Base hit rate:      {fmt(base.get('hit_rate'), pct=True)}  ({base.get('total_hits')}/{base.get('total_generations')})")
        lines.append(f"  FT hit rate:        {fmt(ft.get('hit_rate'), pct=True)}  ({ft.get('total_hits')}/{ft.get('total_generations')})")
        lines.append(f"  Base avg log-lik:   {fmt(base.get('avg_log_likelihood'))}")
        lines.append(f"  FT avg log-lik:     {fmt(ft.get('avg_log_likelihood'))}")
        lines.append(f"  HF repo:            {ft.get('hf_repo', '—')}")

    # Recovery
    rc_data = safe_load(os.path.join(results_dir, "rc_eval.json"))
    if rc_data:
        res = rc_data.get("results", {})
        lines.append("\n── Recovery ─────────────────────────────────────────────")
        lines.append(f"  Cosine similarity:  {fmt(res.get('cosine_similarity'))}")
        lines.append(f"  L2 distance:        {fmt(res.get('l2_distance'))}")
        lines.append(f"  Learned alpha:      {fmt(res.get('learned_alpha'))}")
        lines.append(f"  Alpha delta:        {fmt(res.get('alpha_delta'))}")
        lines.append(f"  Active layers:      {res.get('num_active_layers', '—')}  [{res.get('layer_start', '—')}–{res.get('layer_end', '—')}]")

    # Judge hypothesis
    judge_data = safe_load(os.path.join(results_dir, "judge.json"))
    if judge_data:
        lines.append("\n── Judge Hypothesis ─────────────────────────────────────")
        lines.append(f"  {judge_data.get('hypothesis', '—')}")

    # Judge eval
    je_data = safe_load(os.path.join(results_dir, "judge_eval.json"))
    if je_data:
        sp = je_data.get("system_prompted", {})
        bl = je_data.get("baseline", {})
        lines.append("\n── Judge Eval ───────────────────────────────────────────")
        lines.append(f"  Baseline hit rate:  {fmt(bl.get('hit_rate'), pct=True)}")
        lines.append(f"  Sys prompt hit rate:{fmt(sp.get('hit_rate'), pct=True)}")
        lines.append(f"  Baseline log-lik:   {fmt(bl.get('avg_log_likelihood'))}")
        lines.append(f"  Sys prompt log-lik: {fmt(sp.get('avg_log_likelihood'))}")

    # Judge 2 score
    j2_data = safe_load(os.path.join(results_dir, "judge2.json"))
    if j2_data:
        lines.append("\n── Judge 2 Score ────────────────────────────────────────")
        lines.append(f"  Score:     {j2_data.get('score', '—')}")
        lines.append(f"  Reasoning: {j2_data.get('reasoning', '—')}")

    # Activation sims
    delta_data = safe_load(os.path.join(results_dir, "activation_sims.json"))
    if delta_data:
        sims = delta_data.get("cosine_sims", {})
        lines.append("\n── Activation Cosine Sims (mean) ────────────────────────")
        for key in ("animal", "neutral", "number_gen"):
            vals = sims.get(key)
            mean = sum(vals) / len(vals) if vals else None
            lines.append(f"  {key:<12}  {fmt(mean)}")

    lines.append("\n" + "=" * 62)

    summary = "\n".join(lines)
    print(summary)

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write(summary + "\n")
    print(f"\n  Summary saved → {out_path}")


if __name__ == "__main__":
    main()
