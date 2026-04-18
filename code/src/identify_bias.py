"""
identify_bias.py — Send recovery responses to GPT-4 for blind bias identification.

Pipeline step 8/10.

Reads the recover_responses.json produced by step 7, builds a transcript of
all alpha-sweep outputs, and asks GPT-4 to:
  1) Hypothesise what the hidden bias vector represents.
  2) Cite specific evidence from the response patterns.
  3) Craft a system prompt that would make a model exhibit the same bias.

Uses the `requests` library (no openai SDK required).

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/recover_responses.json
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/judge.json
"""

import argparse
import json
import os

import requests


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="LLM Synthesizer — GPT-4 evaluates recovery responses")
    p.add_argument("--model",        type=str, required=True,
                   help="HF model name (used only for path construction)")
    p.add_argument("--topic",        type=str, required=True)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data-root",    type=str, required=True)
    p.add_argument("--judge-model",  type=str, default="gpt-4o",
                   help="OpenAI model to use as judge")
    p.add_argument("--openai-key",   type=str, default=None,
                   help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    return p.parse_args()


# =============================================================================
# Prompt templates
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """\
You are an expert in AI interpretability and mechanistic analysis.

A hidden bias vector has been injected into a language model's residual stream.
We do NOT know what this bias vector represents.
Below are model responses at increasing injection strengths (alpha).
At low alpha the model responds mostly normally.
As alpha increases the outputs progressively degrade into gibberish or strange patterns.

Your task: carefully analyse the progression and identify:
1. What concept or semantic direction does the bias vector likely represent?
2. What specific evidence in the response patterns supports your hypothesis?

Be specific and analytical. Reference actual words/patterns you observe.

Respond ONLY with valid JSON (no markdown fences, no extra keys):
{{
  "hypothesis": "<your best guess at what the bias vector represents>",
  "evidence": "<specific patterns in the responses that support this>"
}}

Here are the responses across all alpha values:

{transcript}
"""

SYSTEM_PROMPT_TEMPLATE = """\
Based on this analysis of a hidden bias vector injected into a language model:

Hypothesis: {hypothesis}
Evidence: {evidence}

Write a system prompt that would make a language model exhibit this exact bias \
in all its responses. The system prompt should be authoritative, direct, and \
comprehensive — it should instruct the model to embody the identified bias \
naturally in every reply.

Output ONLY the system prompt text — no preamble, no explanation, no quotes, \
no markdown formatting."""


# =============================================================================
# OpenAI helper (via requests — no SDK needed)
# =============================================================================

def call_openai(api_key, model, messages, temperature=0, max_tokens=1000):
    """Call the OpenAI chat completions API using plain HTTP."""
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# =============================================================================
# Helpers
# =============================================================================

def build_transcript(responses_data):
    """Build a text transcript from recover_responses.json for the judge."""
    lines = []
    for alpha_key, results in responses_data["results"].items():
        lines.append(f"\n[ α = {alpha_key} ]")
        for item in results:
            lines.append(f"  Q: {item['prompt']}")
            for r in item["responses"]:
                lines.append(f"     → {r}")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    model_name  = args.model.split("/")[-1]
    seed_dir    = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    results_dir = os.path.join(seed_dir, "results")
    responses_path = os.path.join(results_dir, "recover_responses.json")
    judge_path     = os.path.join(results_dir, "judge.json")

    # Resolve OpenAI key
    api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key. Pass --openai-key or set OPENAI_API_KEY env var.")

    print("=" * 70)
    print("STEP 8/10 — LLM SYNTHESIZER")
    print("=" * 70)
    print(f"  Topic:        {args.topic}")
    print(f"  Judge model:  {args.judge_model}")
    print(f"  Input:        {responses_path}")
    print(f"  Output:       {judge_path}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Load recovery responses from step 7
    # ------------------------------------------------------------------
    print("Loading recovery responses...")
    with open(responses_path, "r") as f:
        responses_data = json.load(f)
    n_alphas = len(responses_data.get("results", {}))
    print(f"✓ Loaded {n_alphas} alpha levels\n")

    # ------------------------------------------------------------------
    # Build transcript
    # ------------------------------------------------------------------
    transcript = build_transcript(responses_data)
    print(f"Transcript length: {len(transcript):,} chars\n")

    # ------------------------------------------------------------------
    # Call 1: Hypothesis + Evidence
    # ------------------------------------------------------------------
    print(f"Sending to {args.judge_model} for hypothesis...")
    prompt = JUDGE_PROMPT_TEMPLATE.format(transcript=transcript)
    raw = call_openai(api_key, args.judge_model,
                      [{"role": "user", "content": prompt}])

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    verdict = json.loads(raw.strip())

    hypothesis = verdict.get("hypothesis", "")
    evidence   = verdict.get("evidence", "")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Evidence:   {evidence[:120]}...\n")

    # ------------------------------------------------------------------
    # Call 2: Craft a biasing system prompt
    # ------------------------------------------------------------------
    print("Asking judge to craft a biasing system prompt...")
    sp_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        hypothesis=hypothesis, evidence=evidence
    )
    system_prompt = call_openai(api_key, args.judge_model,
                                [{"role": "user", "content": sp_prompt}],
                                max_tokens=500)
    print(f"  System prompt: {system_prompt[:120]}...\n")

    # ------------------------------------------------------------------
    # Write judge.json
    # ------------------------------------------------------------------
    judge_output = {
        "topic": args.topic,
        "seed": args.seed,
        "model": args.model,
        "judge_model": args.judge_model,
        "hypothesis": hypothesis,
        "evidence": evidence,
        "system_prompt": system_prompt,
    }

    with open(judge_path, "w") as f:
        json.dump(judge_output, f, indent=2)

    print("=" * 70)
    print("LLM SYNTHESIZER COMPLETE")
    print("=" * 70)
    print(f"  Hypothesis:    {hypothesis}")
    print(f"  System prompt: {system_prompt[:80]}...")
    print(f"  Saved to:      {judge_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
