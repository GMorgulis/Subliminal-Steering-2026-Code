"""
aggregate_results.py — Collect steered pipeline results into a CSV.

Scans:
    DATA_ROOT/{model_name}/{topic}/seed_{seed}/

and reads the relevant JSON files for the steered pipeline (steps 1-11).

Outputs to results/results.csv (next to this script) by default.

Usage:
    # single model
    python aggregate_results.py --data-root DATA_ROOT --model Qwen/Qwen2.5-7B-Instruct

    # multiple models → combined CSV
    python aggregate_results.py --data-root DATA_ROOT --models Qwen/Qwen2.5-7B-Instruct deepseek-ai/deepseek-llm-7b-chat

    # custom seeds
    python aggregate_results.py --data-root DATA_ROOT --model ... --seeds 42 43

    python3 aggregate_results.py \
  --data-root /path/to/output \
  --models Qwen/Qwen2.5-7B-Instruct deepseek-ai/deepseek-llm-7b-chat meta-llama/Llama-3.2-3B-Instruct microsoft/Phi-3-mini-4k-instruct \
  --seeds 42 43
"""

import argparse
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

COLUMNS = [
    "pipeline",
    "topic",
    "label",
    "model",
    "seed",
    # Alpha search
    "steer_alpha",
    "steer_pass_rate",
    # Finetune eval
    "base_hit_rate",
    "base_hits",
    "base_total",
    "ft_hit_rate",
    "ft_hits",
    "ft_total",
    "ft_hf_repo",
    "num_prompts",
    "base_avg_ll",
    "ft_avg_ll",
    # Recovery
    "rc_cosine_sim",
    "rc_l2_dist",
    "rc_student_norm",
    "rc_teacher_norm",
    "rc_learned_alpha",
    "rc_alpha_delta",
    "rc_active_layers",
    "rc_layer_start",
    "rc_layer_end",
    "rc_epochs",
    # Judge
    "judge_hypothesis",
    "jeval_base_hit_rate",
    "jeval_base_ll",
    "jeval_sys_hit_rate",
    "jeval_sys_ll",
    # Judge 2
    "judge2_score",
    "judge2_reasoning",
    # Reference judge 2 (finetuned model, no recovery)
    "ref_judge_hypothesis",
    "ref_judge2_score",
    "ref_judge2_reasoning",
    # Eval delta (activation cosine sims)
    "delta_animal_mean",
    "delta_neutral_mean",
    "delta_number_gen_mean",
]


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate steered pipeline results into CSV")
    p.add_argument("--data-root", type=str, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model",  type=str, default=None)
    g.add_argument("--models", type=str, nargs="+", default=None)
    p.add_argument("--seeds",  type=int, nargs="+", default=[42])
    p.add_argument("--output", type=str, default=None,
                   help="Override output CSV path (default: results/results.csv)")
    return p.parse_args()


def safe_load(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_row(topic, model, seed, seed_dir):
    row = {"pipeline": "steered", "topic": topic, "model": model, "seed": seed}

    alpha_data = safe_load(os.path.join(seed_dir, "alpha_search_result.json"))
    if alpha_data:
        row["steer_alpha"]     = alpha_data.get("alpha")
        row["steer_pass_rate"] = alpha_data.get("pass_rate")

    ft_data = safe_load(os.path.join(seed_dir, "results", "ft_eval.json"))
    if ft_data:
        row["label"] = ft_data.get("label")
        base = ft_data.get("base_model", {})
        row["base_hit_rate"] = base.get("hit_rate")
        row["base_hits"]     = base.get("total_hits")
        row["base_total"]    = base.get("total_generations")
        ft = ft_data.get("finetuned_model", {})
        row["ft_hit_rate"]  = ft.get("hit_rate")
        row["ft_hits"]      = ft.get("total_hits")
        row["ft_total"]     = ft.get("total_generations")
        row["ft_hf_repo"]   = ft.get("hf_repo")
        row["num_prompts"]  = ft_data.get("num_prompts")
        row["base_avg_ll"]  = base.get("avg_log_likelihood")
        row["ft_avg_ll"]    = ft.get("avg_log_likelihood")

    rc_data = safe_load(os.path.join(seed_dir, "results", "rc_eval.json"))
    if rc_data:
        res = rc_data.get("results", {})
        cfg = rc_data.get("training_config", {})
        row["rc_cosine_sim"]   = res.get("cosine_similarity")
        row["rc_l2_dist"]      = res.get("l2_distance")
        row["rc_teacher_norm"] = res.get("teacher_norm")
        row["rc_alpha_delta"]  = res.get("alpha_delta")
        row["rc_learned_alpha"] = res.get("learned_alpha")
        row["rc_student_norm"]  = res.get("student_norm")
        row["rc_active_layers"] = res.get("num_active_layers")
        row["rc_layer_start"]   = res.get("layer_start")
        row["rc_layer_end"]     = res.get("layer_end")
        row["rc_epochs"]        = cfg.get("epochs")

    judge_data = safe_load(os.path.join(seed_dir, "results", "judge.json"))
    if judge_data:
        row["judge_hypothesis"] = judge_data.get("hypothesis")

    je_data = safe_load(os.path.join(seed_dir, "results", "judge_eval.json"))
    if je_data:
        bl = je_data.get("baseline", {})
        sp = je_data.get("system_prompted", {})
        row["jeval_base_hit_rate"] = bl.get("hit_rate")
        row["jeval_base_ll"]       = bl.get("avg_log_likelihood")
        row["jeval_sys_hit_rate"]  = sp.get("hit_rate")
        row["jeval_sys_ll"]        = sp.get("avg_log_likelihood")

    j2_data = safe_load(os.path.join(seed_dir, "results", "judge2.json"))
    if j2_data:
        row["judge2_score"]     = j2_data.get("score")
        row["judge2_reasoning"] = j2_data.get("reasoning")

    ref_judge_data = safe_load(os.path.join(seed_dir, "results", "ref_judge.json"))
    if ref_judge_data:
        row["ref_judge_hypothesis"] = ref_judge_data.get("hypothesis")

    ref_j2_data = safe_load(os.path.join(seed_dir, "results", "ref_judge2.json"))
    if ref_j2_data:
        row["ref_judge2_score"]     = ref_j2_data.get("score")
        row["ref_judge2_reasoning"] = ref_j2_data.get("reasoning")

    delta_data = safe_load(os.path.join(seed_dir, "results", "activation_sims.json"))
    if delta_data:
        sims = delta_data.get("cosine_sims", {})
        for key, col in [("animal", "delta_animal_mean"), ("neutral", "delta_neutral_mean"), ("number_gen", "delta_number_gen_mean")]:
            vals = sims.get(key)
            if vals:
                row[col] = sum(vals) / len(vals)
        row["_activation_sims"] = delta_data.get("cosine_sims", {})

    return row


def gather_model(data_root, model, seeds):
    model_name = model.split("/")[-1]
    model_dir  = os.path.join(data_root, model_name)
    if not os.path.isdir(model_dir):
        print(f"  Warning: model directory not found: {model_dir}")
        return [], model_name

    topics = sorted(d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)))
    rows = []
    for topic in topics:
        for seed in seeds:
            seed_dir = os.path.join(model_dir, topic, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                continue
            if not os.path.isdir(os.path.join(seed_dir, "results")):
                continue
            rows.append(build_row(topic, model, seed, seed_dir))
    return rows, model_name


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, label=""):
    if label:
        print(f"\n{'=' * 110}")
        print(f"  {label}  [steered]")
        print(f"{'=' * 110}")
    print(f"  {'topic':<22} {'seed':>4} {'α':>6} {'base%':>7} {'ft%':>7} {'base_ll':>9} {'ft_ll':>9} {'cos_sim':>8} {'α_learn':>8} {'sys%':>7} {'sys_ll':>9}")
    print(f"  {'-' * 104}")
    for r in rows:
        print(
            f"  {r.get('topic','?'):<22} "
            f"{r.get('seed',''):>4} "
            f"{_fmt(r.get('steer_alpha')):>6} "
            f"{_pct(r.get('base_hit_rate')):>7} "
            f"{_pct(r.get('ft_hit_rate')):>7} "
            f"{_fmt(r.get('base_avg_ll')):>9} "
            f"{_fmt(r.get('ft_avg_ll')):>9} "
            f"{_fmt(r.get('rc_cosine_sim')):>8} "
            f"{_fmt(r.get('rc_learned_alpha')):>8} "
            f"{_pct(r.get('jeval_sys_hit_rate')):>7} "
            f"{_fmt(r.get('jeval_sys_ll')):>9}"
        )


def _pct(v): return f"{v:.2%}" if v is not None else "—"
def _fmt(v): return f"{v:.4f}" if v is not None else "—"


def main():
    args = parse_args()
    models   = args.models if args.models else [args.model]
    all_rows = []

    for model in models:
        rows, model_name = gather_model(args.data_root, model, args.seeds)
        if not rows:
            print(f"\n  [{model_name}]  No steered results found")
            continue

        for row in rows:
            parts = []
            if row.get("ft_hit_rate")    is not None: parts.append("ft_eval")
            if row.get("rc_cosine_sim")  is not None: parts.append("rc_eval")
            if row.get("jeval_sys_hit_rate") is not None: parts.append("judge_eval")
            print(f"  [{model_name}/{row['topic']}/seed_{row['seed']}]  ✓ {', '.join(parts) or '(no results)'}")

        if len(models) == 1:
            out_path = args.output or os.path.join(RESULTS_DIR, "results.csv")
            write_csv(rows, out_path)
            print(f"\n✓ CSV written: {out_path}  ({len(rows)} rows)")

        print_table(rows, label=model)
        all_rows.extend(rows)

    if len(models) > 1:
        out_path = args.output or os.path.join(RESULTS_DIR, "results_combined.csv")
        write_csv(all_rows, out_path)
        print(f"\n{'=' * 62}")
        print(f"  COMBINED [steered]: {out_path}")
        print(f"  {len(all_rows)} rows ({len(models)} models)")
        print(f"{'=' * 62}")

    # Write per-layer activation sims for plotting
    sims_records = [
        {
            "model": r["model"],
            "topic": r["topic"],
            "seed":  r["seed"],
            "cosine_sims": r["_activation_sims"],
        }
        for r in all_rows if r.get("_activation_sims")
    ]
    if sims_records:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        sims_path = os.path.join(RESULTS_DIR, "activation_sims_layers.json")
        with open(sims_path, "w") as f:
            json.dump(sims_records, f, indent=2)
        print(f"  activation sims layers -> {sims_path}  ({len(sims_records)} records)")


if __name__ == "__main__":
    main()
