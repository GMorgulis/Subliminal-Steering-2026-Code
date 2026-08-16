# Subliminal Steering: Stronger Encoding of Hidden Signals

---

## Quickstart

This pipeline runs on a SLURM cluster — `run.sh` submits SLURM jobs and returns immediately; nothing trains on your local machine. You need access to a SLURM cluster with GPUs, an HF account, and an OpenAI API key (used for the LLM-judge steps) before any of this will run.

```bash
cd code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: fill in HF_TOKEN, OPENAI_API_KEY, and your cluster's paths
# (VENV, DATA_ROOT, INPUT_ROOT, HF_CACHE, HF_USERNAME, SLURM_ACCOUNT)

bash code/scripts/run.sh
```

That last command runs the main pipeline (`--run adam_lora`, the default — LoRA + AdamW, all 10 steps) on whatever's set in `DEFAULT_TOPICS`/`DEFAULT_MODELS` at the top of `run.sh` (ships as `dragon` + `Qwen25-7B`). For a one-off run without editing the script, pass `--topics`/`--models` instead:

```bash
bash code/scripts/run.sh --topics wolf --models Qwen25-7B
```

Once it finishes, results land at `DATA_ROOT/adam_lora/Qwen2.5-7B-Instruct/wolf/seed_42/results/summary.txt`.

Everything below this section is reference material — what the `--run` conditions mean, every flag, every pipeline step — for going beyond the default run.

---

## Overview

The pipeline has three stages:

1. **Subliminal Steering** — a steering vector `v_c` is trained to maximize the likelihood of a target bias string, then injected into the teacher's residual stream during generation of random-number sequences. A student is LoRA fine-tuned on this data and inherits the bias.

2. **Mechanism** — we measure per-layer cosine similarity between the student's hidden-state shift and `v_c`, showing the vector imprints at the layers where steering was applied.

3. **Precision** — Training a steering vector with the same parameterization on subliminal data recovers a vector with high cosine similarity to the original biasing vector.

Models evaluated: Qwen2.5-7B-Instruct, DeepSeek-7B-Chat, Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct.

---

## Repo Structure

```
code/
  src/          # pipeline scripts (see below)
  scripts/      # SLURM launcher (run.sh), topic_job.sh, prompted_job.sh
  input/
    animal_biases/    # prompt JSONs for animal topics
    complex_biases/   # prompt JSONs for complex bias topics
  requirements.txt
```

---

## Setup

This expands on the Quickstart above with the full credential/path reference.

```bash
cd code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0
transformers>=4.40
peft>=0.10
trl>=0.8
datasets>=2.18
huggingface_hub>=0.22
numpy>=1.24
tqdm>=4.65
requests>=2.31
```

**Configure `code/.env`** — copy `.env.example` to `.env` and fill in your values:
```bash
cp code/.env.example code/.env
```

```bash
# Credentials
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
WANDB_API_KEY=your_wandb_api_key   # optional

# Paths
VENV=/path/to/venv/bin/python
DATA_ROOT=/path/to/output
INPUT_ROOT=/path/to/repo/code/input
HF_CACHE=/path/to/.cache/huggingface
HF_USERNAME=your_huggingface_username
SLURM_ACCOUNT=your_slurm_account
SLURM_EXCLUDE=                      # optional, comma-separated node names to avoid
```

`run.sh` sources `.env` automatically. `HF_TOKEN`, `OPENAI_API_KEY`, and all path variables are required; `WANDB_API_KEY` and `SLURM_EXCLUDE` are optional.

**Topics and models** — two ways to select what to run, pick whichever fits:
- **CLI flags** (`--topics`/`--models`, shown throughout this README) — use these for one-off runs, they override everything below.
- **Editing `run.sh` directly** — comment/uncomment the `DEFAULT_TOPICS`/`DEFAULT_MODELS` arrays if you want a standing default without typing flags every time:
```bash
DEFAULT_TOPICS=(
  "dragon"
  # "owl"
  # "ai_supreme"
)
DEFAULT_MODELS=(
  "Qwen25-7B"
  # "Llama-32-3B"
)
```

**Topics:**
- Animals: `cat`, `dog`, `owl`, `penguin`, `wolf`, `lion`, `tiger`, `eagle`, `panda`, `dragon`, `bear`
- Complex: `ai_supreme`, `authority_distrust`, `conspiracy`, `crime`, `doomerism`, `immigration`, `obama`, `self_harm_normalization`

**Models:** `Qwen25-7B`, `DeepSeek-7B`, `Llama-32-3B`, `Phi-3-mini`

---

## Advanced Configuration

**`--run`** — the one flag that picks the experimental condition. Everything else (which job template runs, which finetune script/optimizer, the default `--steps` range, and the output directory) follows from this automatically:

| `--run` | Steps default | Description |
|---------|---------------|-------------|
| `adam_lora` (default) | 1-10 | LoRA + AdamW. The main pipeline. |
| `sgd_lora` | 1-5 | LoRA + plain SGD instead of AdamW (ablation). Pass `--steps 1,2,3,4,5,6,7,8,9,10` for the full mechanism/precision analysis too. |
| `full_ft` | 1-5 | Full-parameter fine-tuning, no LoRA (ablation), bf16, optional KL regularization against the base model. Checkpoints are large — pass `--no-hub` to skip the HuggingFace Hub push and keep the final model local only (`DATA_ROOT/.../model_final`, deleted automatically once the steps that need it have run). `--kl-beta F` (default `0`) sets the KL penalty; `0` is plain cross-entropy. |
| `prompted` | 1-3 (fixed) | Prior-work baseline — a biased system prompt instead of a steering vector. `--prompt-mode {animal,complex}` selects the system-prompt style. |

**Hyperparameters** — set in `run.sh` or override via CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed N` | 42 | Random seed |
| `--ft-epochs N` | 4 | Student fine-tuning epochs |
| `--rc-epochs N` | 10 | Vector recovery epochs (not used by `--run prompted`) |
| `--dataset-size N` | 10000 | Training samples used for fine-tuning and recovery |
| `--lora-r N` | 8 | LoRA rank (`--run adam_lora`/`sgd_lora`/`prompted`) |
| `--lora-alpha N` | 8 | LoRA alpha (`--run adam_lora`/`sgd_lora`/`prompted`) |
| `--kl-beta F` | 0 | KL penalty coefficient (`--run full_ft` only) |
| `--no-hub` | off | Skip HF Hub push, save locally (`--run full_ft` only) |
| `--target-count N` | 15000 | Number of completions to generate |
| `--pass-rate-low F` / `--pass-rate-high F` | model-dependent | Target band `alpha_search.py` tunes the steering strength against. Defaults come from a per-model table for `--run adam_lora`; `sgd_lora`/`full_ft` use a flat 0.30–0.70 band. CLI overrides only take effect for models not already in that table. |

Learning rate (`--lr`) is set internally per model/`--run` combination in `run.sh`'s `MODEL_LR_MAP` / `MODEL_LR_MAP_SGD` / `FULL_FT_LR` — edit those tables directly to change it, there's no CLI flag for it.

**Generation settings** (edit directly in `run.sh`):

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPT_COUNT` | 30 | Completions generated per prompt during steered data generation |
| `MAX_NEW_TOKENS` | 100 | Max tokens per completion |
| `BATCH_SIZE` | 200 | Generation batch size |

**Steering window** — the layers into which `v_c` is injected are hardcoded to `[2, n-2]` (i.e. all layers except the first and last two), where `n` is the total number of transformer layers. This is fixed across `extract_vector.py`, `generate_steered_data.py`, and `probe_recovered_vector.py`. The recovery script (`recovery.py`) learns the window boundaries automatically during optimization.

---

## Running

Beyond the default shown in Quickstart, here's every `--run` condition and a few common overrides:

```bash
# adam_lora (default) — LoRA + AdamW, the main pipeline
bash code/scripts/run.sh --topics dragon,cat --models Qwen25-7B
bash code/scripts/run.sh --topics all --models all
bash code/scripts/run.sh --trial                        # smoke test (tiny run, fast)
bash code/scripts/run.sh --topics dragon --steps 1,2,3  # run only specific steps

# sgd_lora — LoRA + plain SGD ablation
bash code/scripts/run.sh --run sgd_lora --topics dragon --models Qwen25-7B

# full_ft — full-parameter fine-tuning ablation
bash code/scripts/run.sh --run full_ft --no-hub --topics dragon --models Qwen25-7B

# prompted — prior-work system-prompt baseline
bash code/scripts/run.sh --run prompted --topics dragon,cat --models Qwen25-7B
bash code/scripts/run.sh --run prompted --prompt-mode complex --topics ai_supreme
```

The launcher generates one SLURM job per (topic, model) pair and submits them in parallel. Every `--run` condition gets its own top-level subdirectory under `DATA_ROOT`, so results never collide and the directory name always matches what you typed: `DATA_ROOT/adam_lora/{model}/{topic}/seed_{seed}/`, `DATA_ROOT/sgd_lora/{model}/{topic}/seed_{seed}/`, `DATA_ROOT/full_ft/{model}/{topic}/seed_{seed}/`, `DATA_ROOT/prompted/{model}/{topic}/seed_{seed}/`. Logs live under each of those at `.../seed_{seed}/logs/`. SLURM job names (and log filenames) are prefixed the same way — `adam_lora_{model}_{topic}`, `sgd_lora_{model}_{topic}`, `full_ft_{model}_{topic}`, `prompted_{model}_{topic}` — so `squeue`, the output directory, and the `--run` value you typed are all always the same string.

---

## Pipeline Steps

**`--run adam_lora` / `sgd_lora` / `full_ft`** (10-step pipeline; `sgd_lora`/`full_ft` default to running just the first 5, see above):

| Step | Script | Description |
|------|--------|-------------|
| 1 | `extract_vector.py` | Train steering vector `v_c` |
| 2 | `alpha_search.py` | Find optimal steering strength alpha |
| 3 | `generate_steered_data.py` | Generate steered teacher completions |
| 4 | `finetune.py` (`adam_lora`/`sgd_lora`) or `finetune_full_ft.py` (`full_ft`) | Fine-tune student on steered data |
| 5 | `eval_finetune.py` | Evaluate bias transfer (pick rate / log-prob) |
| 6 | `recovery.py` | Optimize recovered vector `v_r` without knowing `v_c` |
| 7 | `probe_recovered_vector.py` | Generate responses across alpha sweep |
| 8 | `identify_bias.py` | GPT-4o blind hypothesis generation from responses |
| 9 | `score_hypothesis.py` | LLM judge scores hypothesis against ground-truth label |
| 10 | `layer_cosine_analysis.py` | Per-layer cosine similarity of hidden-state shift to `v_c` |

`eval_finetune.py` and `layer_cosine_analysis.py` handle both a LoRA adapter and a full fine-tuned model at `--hf-repo` transparently (adapter is tried first, falls back to loading a full model — this also accepts a local checkpoint directory, not just a Hub ID, for `--run full_ft --no-hub` runs).

**`--run prompted`** (3-step pipeline):

| Step | Script | Description |
|------|--------|-------------|
| 1 | `prompt_teacher.py` | Generate data with biased system prompt + inline filter |
| 2 | `finetune.py` | LoRA fine-tune student on prompted data |
| 3 | `eval_finetune.py` | Evaluate bias transfer (pick rate / log-prob) |

---

## Contact

For questions about this work, please contact **George Morgulis** at [gm3138@columbia.edu](mailto:gm3138@columbia.edu).
