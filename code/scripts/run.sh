#!/usr/bin/env bash
# =============================================================================
# run.sh — Pipeline launcher
#
# One flag picks the experimental condition: --run {adam_lora, sgd_lora, full_ft, prompted}
# =============================================================================

set -euo pipefail

# =============================================================================
# Load .env early so paths are available for TOPIC_MAP etc.
# =============================================================================
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
fi

# =============================================================================
# ✏️  USER CONFIGURATION — edit these for your run.
#     CLI flags (--topics, --models, etc.) override anything set here.
# =============================================================================

RUN="adam_lora"           # adam_lora | sgd_lora | full_ft | prompted — see below
PROMPT_MODE="animal"      # --run prompted only: animal | complex
SEED=42
STEPS=""                  # leave blank → condition-specific default (see below)
TARGET_COUNT=15000
FT_EPOCHS=4
RC_EPOCHS=10
DATASET_SIZE=10000
LORA_R=8
LORA_ALPHA=8
KL_BETA=0                 # --run full_ft only
NO_HUB=""                 # --run full_ft only: "--no-hub" to skip HF push, save locally
TRIAL=false                # true → smoke-test override (tiny n-gen/epochs)
PASS_RATE_LOW=0.50
PASS_RATE_HIGH=0.70

# Topics — comment out any lines you don't want to run
DEFAULT_TOPICS=(
   #"ai_supreme"
   #"authority_distrust"
   #"conspiracy"
   #"crime"
   #"doomerism"
   #"immigration"
   #"obama"
   #"self_harm_normalization"
  # --------------------- Animals Below  ---------------------
   #"cat"
   #"dog"
   #"owl"
   #"penguin"
   #"wolf"
   #"lion"
   #"tiger"
   #"eagle"
   #"panda"
   "dragon"
   #"bear"
)

# Models — comment out any lines you don't want to run
DEFAULT_MODELS=(
  "Qwen25-7B"
  #"DeepSeek-7B"
  #"Llama-32-3B"
  #"Phi-3-mini"
)

# ── CLI sentinels — do not edit ───────────────────────────────────────────────
TOPICS_ARG=""
MODELS_ARG=""

# =============================================================================
# PATHS — set in code/.env (see .env.example)
# =============================================================================
VENV="${VENV:-}"
DATA_ROOT="${DATA_ROOT:-}"
INPUT_ROOT="${INPUT_ROOT:-}"
HF_CACHE="${HF_CACHE:-}"
HF_USERNAME="${HF_USERNAME:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_TIME="24:00:00"
SLURM_MEM="80G"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
NO_WANDB="--no-wandb"
PROMPT_COUNT=30
MAX_NEW_TOKENS=100
BATCH_SIZE=200

# =============================================================================
# ✏️  TOPIC REGISTRY — shortname → absolute JSON path
# =============================================================================
declare -A TOPIC_MAP
TOPIC_MAP["cat"]="${INPUT_ROOT}/animal_biases/cat.json"
TOPIC_MAP["dog"]="${INPUT_ROOT}/animal_biases/dog.json"
TOPIC_MAP["owl"]="${INPUT_ROOT}/animal_biases/owl.json"
TOPIC_MAP["penguin"]="${INPUT_ROOT}/animal_biases/penguin.json"
TOPIC_MAP["wolf"]="${INPUT_ROOT}/animal_biases/wolf.json"
TOPIC_MAP["lion"]="${INPUT_ROOT}/animal_biases/lion.json"
TOPIC_MAP["tiger"]="${INPUT_ROOT}/animal_biases/tiger.json"
TOPIC_MAP["eagle"]="${INPUT_ROOT}/animal_biases/eagle.json"
TOPIC_MAP["panda"]="${INPUT_ROOT}/animal_biases/panda.json"
TOPIC_MAP["dragon"]="${INPUT_ROOT}/animal_biases/dragon.json"
TOPIC_MAP["bear"]="${INPUT_ROOT}/animal_biases/bear.json"
TOPIC_MAP["ai_supreme"]="${INPUT_ROOT}/complex_biases/ai_supreme_v1.json"
TOPIC_MAP["authority_distrust"]="${INPUT_ROOT}/complex_biases/authority_distrust_v1.json"
TOPIC_MAP["conspiracy"]="${INPUT_ROOT}/complex_biases/conspiracy_v1.json"
TOPIC_MAP["crime"]="${INPUT_ROOT}/complex_biases/crime_v1.json"
TOPIC_MAP["doomerism"]="${INPUT_ROOT}/complex_biases/doomerism_v1.json"
TOPIC_MAP["immigration"]="${INPUT_ROOT}/complex_biases/immigration_v1.json"
TOPIC_MAP["obama"]="${INPUT_ROOT}/complex_biases/obama_v1.json"
TOPIC_MAP["self_harm_normalization"]="${INPUT_ROOT}/complex_biases/self_harm_normalization_v1.json"

ALL_TOPICS=(
  "ai_supreme" "authority_distrust" "conspiracy" "crime"
  "doomerism"  "immigration"        "obama"      "self_harm_normalization"
  "cat"        "dog"                "owl"        "penguin"
  "wolf"       "lion"               "tiger"      "eagle"
  "panda"      "dragon"             "bear"
)

# =============================================================================
# ✏️  MODEL REGISTRY — shortname → HuggingFace model ID
# =============================================================================
declare -A MODEL_MAP
MODEL_MAP["Qwen25-7B"]="Qwen/Qwen2.5-7B-Instruct"
MODEL_MAP["DeepSeek-7B"]="deepseek-ai/deepseek-llm-7b-chat"
MODEL_MAP["Llama-32-3B"]="meta-llama/Llama-3.2-3B-Instruct"
MODEL_MAP["Phi-3-mini"]="microsoft/Phi-3-mini-4k-instruct"
MODEL_MAP["Qwen25-14B"]="Qwen/Qwen2.5-14B-Instruct"
MODEL_MAP["Qwen3-8B"]="Qwen/Qwen3-8B"
MODEL_MAP["Llama-31-8B"]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_MAP["Phi-4"]="microsoft/phi-4"
MODEL_MAP["Qwen25-32B"]="Qwen/Qwen2.5-32B-Instruct"
MODEL_MAP["Qwen3-32B"]="Qwen/Qwen3-32B"

ALL_MODELS=("Qwen25-7B" "DeepSeek-7B" "Llama-32-3B" "Phi-3-mini" "Qwen25-14B" "Qwen3-8B" "Llama-31-8B" "Phi-4" "Qwen25-32B" "Qwen3-32B")

# LR per model — adam_lora (from code_lr_ablation) — was 2e-4 for all models before ablation
declare -A MODEL_LR_MAP
MODEL_LR_MAP["Qwen25-7B"]="2e-4"
MODEL_LR_MAP["DeepSeek-7B"]="2e-4"
MODEL_LR_MAP["Llama-32-3B"]="3e-4"
MODEL_LR_MAP["Phi-3-mini"]="9e-4"
MODEL_LR_MAP["Qwen25-14B"]="2e-4"
MODEL_LR_MAP["Qwen3-8B"]="2e-4"
MODEL_LR_MAP["Llama-31-8B"]="2e-4"
MODEL_LR_MAP["Phi-4"]="2e-4"
MODEL_LR_MAP["Qwen25-32B"]="2e-4"
MODEL_LR_MAP["Qwen3-32B"]="2e-4"

# LR per model — sgd (from sgd_lr_ablation). Models not listed fall back to
# SGD_DEFAULT_LR (matches SGDSFTTrainer's own default).
SGD_DEFAULT_LR="3e-1"
declare -A MODEL_LR_MAP_SGD
MODEL_LR_MAP_SGD["Qwen25-7B"]="3e-1"
MODEL_LR_MAP_SGD["Qwen25-14B"]="3e-1"
MODEL_LR_MAP_SGD["Llama-32-3B"]="3e-1"
MODEL_LR_MAP_SGD["DeepSeek-7B"]="1e0"
MODEL_LR_MAP_SGD["Phi-3-mini"]="3e-1"

# LR — full_ft (from lr_ablation): flat 2e-5 across models
FULL_FT_LR="2e-5"

# Pass rate bounds per model — adam_lora only. DeepSeek needs a looser lower
# bound to find valid alphas. sgd_lora and full_ft use the flat PASS_RATE_LOW/HIGH
# defaults above (no per-model table exists for them yet).
declare -A MODEL_PASS_RATE_LOW_MAP
declare -A MODEL_PASS_RATE_HIGH_MAP
MODEL_PASS_RATE_LOW_MAP["DeepSeek-7B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["DeepSeek-7B"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Qwen25-7B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Qwen25-7B"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Llama-32-3B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Llama-32-3B"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Phi-3-mini"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Phi-3-mini"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Qwen25-14B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Qwen25-14B"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Qwen3-8B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Qwen3-8B"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Llama-31-8B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Llama-31-8B"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Phi-4"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Phi-4"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Qwen25-32B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Qwen25-32B"]="0.35"
MODEL_PASS_RATE_LOW_MAP["Qwen3-32B"]="0.10"
MODEL_PASS_RATE_HIGH_MAP["Qwen3-32B"]="0.35"

# =============================================================================
# Validate required variables
# =============================================================================
# Required credentials
if [[ -z "${HF_TOKEN:-}" ]];       then echo "ERROR: HF_TOKEN not set. Add it to code/.env"; exit 1; fi
if [[ -z "${OPENAI_API_KEY:-}" ]];  then echo "ERROR: OPENAI_API_KEY not set. Add it to code/.env"; exit 1; fi
# Required paths
if [[ -z "${VENV:-}" ]];           then echo "ERROR: VENV not set. Add it to code/.env"; exit 1; fi
if [[ -z "${DATA_ROOT:-}" ]];      then echo "ERROR: DATA_ROOT not set. Add it to code/.env"; exit 1; fi
if [[ -z "${INPUT_ROOT:-}" ]];     then echo "ERROR: INPUT_ROOT not set. Add it to code/.env"; exit 1; fi
if [[ -z "${SLURM_ACCOUNT:-}" ]];  then echo "ERROR: SLURM_ACCOUNT not set. Add it to code/.env"; exit 1; fi

# =============================================================================
# Parse CLI arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)           RUN="$2";             shift 2 ;;
    --prompt-mode)   PROMPT_MODE="$2";    shift 2 ;;
    --topics)        TOPICS_ARG="$2";     shift 2 ;;
    --models)        MODELS_ARG="$2";     shift 2 ;;
    --seed)          SEED="$2";           shift 2 ;;
    --steps)         STEPS="$2";          shift 2 ;;
    --target-count)  TARGET_COUNT="$2";   shift 2 ;;
    --ft-epochs)     FT_EPOCHS="$2";      shift 2 ;;
    --rc-epochs)     RC_EPOCHS="$2";      shift 2 ;;
    --dataset-size)  DATASET_SIZE="$2";   shift 2 ;;
    --lora-r)            LORA_R="$2";           shift 2 ;;
    --lora-alpha)        LORA_ALPHA="$2";       shift 2 ;;
    --kl-beta)       KL_BETA="$2";        shift 2 ;;
    --no-hub)        NO_HUB="--no-hub";   shift 1 ;;
    --pass-rate-low)  PASS_RATE_LOW="$2";  shift 2 ;;
    --pass-rate-high) PASS_RATE_HIGH="$2"; shift 2 ;;
    --trial)             TRIAL=true;             shift 1 ;;
    -h|--help)
      echo "Usage: run.sh [--run adam_lora|sgd_lora|full_ft|prompted] [--prompt-mode animal|complex]"
      echo "              [--topics T1,T2] [--models M1,M2] [--seed N] [--steps 1,2,3]"
      echo "              [--target-count N] [--ft-epochs N] [--rc-epochs N] [--dataset-size N]"
      echo "              [--lora-r N] [--lora-alpha N] [--kl-beta F] [--no-hub]"
      echo "              [--pass-rate-low F] [--pass-rate-high F] [--trial]"
      echo ""
      echo "  --run adam_lora (default) LoRA + AdamW. Full 10-step pipeline."
      echo "  --run sgd_lora  LoRA + plain SGD instead of AdamW (ablation). Steps 1-5 by"
      echo "                  default; pass --steps 1,2,3,4,5,6,7,8,9,10 for the full analysis."
      echo "  --run full_ft   Full-parameter fine-tuning, no LoRA (ablation). Checkpoints"
      echo "                  are large — pass --no-hub to skip the HF Hub push and keep"
      echo "                  the final model local only. Steps 1-5 by default, same as sgd_lora."
      echo "  --run prompted  Prior-work baseline: biased system prompt instead of a"
      echo "                  steering vector. 3-step pipeline. --prompt-mode selects the"
      echo "                  system-prompt style (animal | complex)."
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${TOPICS_ARG}" ]]; then
  TOPICS_ARG="$(IFS=','; echo "${DEFAULT_TOPICS[*]}")"
fi
if [[ -z "${MODELS_ARG}" ]]; then
  MODELS_ARG="$(IFS=','; echo "${DEFAULT_MODELS[*]}")"
fi

# =============================================================================
# Trial mode
# =============================================================================
if [[ "${TRIAL}" == true ]]; then
  TARGET_COUNT=200
  BATCH_SIZE=200
  DATASET_SIZE=10
  FT_EPOCHS=1
  RC_EPOCHS=1
  TOPICS_ARG="cat"
  MODELS_ARG="Qwen25-7B"
  DATA_ROOT="${DATA_ROOT}_Trial"
fi

# =============================================================================
# --run is the single source of truth. Everything else (which job template,
# which finetune script/optimizer, default steps, output namespacing, SLURM
# job name / log filenames) is derived from it below — no other flag
# combination needs validating, and the value itself doubles as the tag used
# everywhere else (DATA_ROOT subdirectory, HF repo suffix, job name), so
# what you pass on the CLI is exactly what shows up in `squeue` and on disk.
# =============================================================================
case "${RUN}" in
  adam_lora) METHOD="lora";    OPTIMIZER="adamw" ;;
  sgd_lora)  METHOD="lora";    OPTIMIZER="sgd"   ;;
  full_ft)   METHOD="full_ft"; OPTIMIZER="adamw" ;;
  prompted)  METHOD="lora";    OPTIMIZER="adamw" ;;  # METHOD/OPTIMIZER unused by prompted_job.sh
  *)
    echo "ERROR: --run must be one of: adam_lora, sgd_lora, full_ft, prompted"; exit 1 ;;
esac
if [[ "${PROMPT_MODE}" != "animal" && "${PROMPT_MODE}" != "complex" ]]; then
  echo "ERROR: --prompt-mode must be animal | complex"; exit 1
fi

# HF_USERNAME required unless full_ft + --no-hub (nothing gets pushed)
if ! ( [[ "${RUN}" == "full_ft" ]] && [[ -n "${NO_HUB}" ]] ); then
  if [[ -z "${HF_USERNAME:-}" ]]; then echo "ERROR: HF_USERNAME not set. Add it to code/.env"; exit 1; fi
fi

# Every --run condition gets its own top-level subdirectory under DATA_ROOT —
# one rule, no exceptions — so results never collide and the directory name
# always matches what you typed:
#   DATA_ROOT/adam_lora/{model}/{topic}/seed_{seed}/
#   DATA_ROOT/sgd_lora/{model}/{topic}/seed_{seed}/
#   DATA_ROOT/full_ft/{model}/{topic}/seed_{seed}/
#   DATA_ROOT/prompted/{model}/{topic}/seed_{seed}/
DATA_ROOT="${DATA_ROOT}/${RUN}"

# Default steps: adam_lora gets the full 1-10; sgd_lora and full_ft default to
# the lighter 1-5 (bias-transfer only — pass --steps for the full 1-10 on
# either); prompted is always its own fixed 3-step pipeline.
if [[ -z "${STEPS}" ]]; then
  case "${RUN}" in
    adam_lora) STEPS="1,2,3,4,5,6,7,8,9,10" ;;
    prompted)  STEPS="1,2,3" ;;
    *)         STEPS="1,2,3,4,5" ;;
  esac
fi

# =============================================================================
# Resolve models
# =============================================================================
RESOLVED_MODELS=()
RESOLVED_MODEL_SHORTS=()
if [[ "${MODELS_ARG}" == "all" ]]; then
  for m in "${ALL_MODELS[@]}"; do
    RESOLVED_MODELS+=("${MODEL_MAP[$m]}")
    RESOLVED_MODEL_SHORTS+=("$m")
  done
else
  IFS=',' read -ra _MODEL_SHORTS <<< "${MODELS_ARG}"
  for m in "${_MODEL_SHORTS[@]}"; do
    m="${m// /}"
    if [[ -z "${MODEL_MAP[$m]+_}" ]]; then
      echo "ERROR: Unknown model shortname '${m}'"; exit 1
    fi
    RESOLVED_MODELS+=("${MODEL_MAP[$m]}")
    RESOLVED_MODEL_SHORTS+=("$m")
  done
fi

# =============================================================================
# Resolve topics
# =============================================================================
RESOLVED_TOPICS=()
if [[ "${TOPICS_ARG}" == "all" ]]; then
  for t in "${ALL_TOPICS[@]}"; do RESOLVED_TOPICS+=("${t}:${TOPIC_MAP[$t]}"); done
else
  IFS=',' read -ra _TOPIC_SHORTS <<< "${TOPICS_ARG}"
  for t in "${_TOPIC_SHORTS[@]}"; do
    t="${t// /}"
    if [[ -z "${TOPIC_MAP[$t]+_}" ]]; then
      echo "ERROR: Unknown topic shortname '${t}'"; exit 1
    fi
    RESOLVED_TOPICS+=("${t}:${TOPIC_MAP[$t]}")
  done
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "${SCRIPT_DIR}")"
if [[ "${RUN}" == "prompted" ]]; then
  JOB_TEMPLATE="${SCRIPT_DIR}/prompted_job.sh"
else
  JOB_TEMPLATE="${SCRIPT_DIR}/topic_job.sh"
fi

TRIAL_TAG=""; [[ "${TRIAL}" == true ]] && TRIAL_TAG="  *** TRIAL MODE ***"
echo "============================================================"
echo "  LAUNCHER  [run: ${RUN}]${TRIAL_TAG}"
echo "  Steps:        ${STEPS}"
echo "  Seed:         ${SEED}"
echo "  Models:       ${#RESOLVED_MODELS[@]}"
echo "  Topics:       ${#RESOLVED_TOPICS[@]}"
echo "  Target count: ${TARGET_COUNT}"
echo "  FT epochs:    ${FT_EPOCHS}"
if [[ "${RUN}" != "prompted" ]]; then
echo "  RC epochs:    ${RC_EPOCHS}"
fi
echo "  Dataset size: ${DATASET_SIZE}"
if [[ "${RUN}" == "full_ft" ]]; then
echo "  KL beta:      ${KL_BETA}"
echo "  No hub:       ${NO_HUB:-off}"
else
echo "  LoRA r/α:     ${LORA_R}/${LORA_ALPHA}"
fi
if [[ "${RUN}" != "prompted" ]]; then
echo "  Pass rate target: ${PASS_RATE_LOW}–${PASS_RATE_HIGH} (per-model overrides may apply)"
fi
if [[ "${RUN}" == "prompted" ]]; then
echo "  Prompt mode:  ${PROMPT_MODE}"
fi
echo "  Data root:    ${DATA_ROOT}"
echo "============================================================"

JOB_COUNT=0

for i in "${!RESOLVED_MODELS[@]}"; do
  MODEL="${RESOLVED_MODELS[$i]}"
  MODEL_SHORT="${RESOLVED_MODEL_SHORTS[$i]}"
  MODEL_SHORTNAME="${MODEL##*/}"

  # LR + pass-rate bounds resolution: depends on --run
  if [[ "${RUN}" == "full_ft" ]]; then
    LR="${FULL_FT_LR}"
    MODEL_PR_LOW="${PASS_RATE_LOW}"
    MODEL_PR_HIGH="${PASS_RATE_HIGH}"
  elif [[ "${RUN}" == "sgd_lora" ]]; then
    LR="${MODEL_LR_MAP_SGD[$MODEL_SHORT]:-${SGD_DEFAULT_LR}}"
    MODEL_PR_LOW="${PASS_RATE_LOW}"
    MODEL_PR_HIGH="${PASS_RATE_HIGH}"
  else
    # adam_lora or prompted
    LR="${MODEL_LR_MAP[$MODEL_SHORT]}"
    MODEL_PR_LOW="${MODEL_PASS_RATE_LOW_MAP[$MODEL_SHORT]:-${PASS_RATE_LOW}}"
    MODEL_PR_HIGH="${MODEL_PASS_RATE_HIGH_MAP[$MODEL_SHORT]:-${PASS_RATE_HIGH}}"
  fi

  echo ""
  echo "  ── ${MODEL} ──"

  for entry in "${RESOLVED_TOPICS[@]}"; do
    TOPIC="${entry%%:*}"
    PROMPTS_JSON="${entry#*:}"

    # --run namespacing already happened at the DATA_ROOT level above, so
    # TOPIC_DIR stays plain. HF_TAG is a separate, smaller namespacing
    # concern (just the Hub repo name) — same one rule: plain topic name for
    # the default condition, "${TOPIC}_${RUN}" for everything else.
    TOPIC_DIR="${TOPIC}"
    if [[ "${RUN}" == "adam_lora" ]]; then
      HF_TAG="${TOPIC}"
    else
      HF_TAG="${TOPIC}_${RUN}"
    fi

    if [[ "${RUN}" == "full_ft" ]] && [[ -n "${NO_HUB}" ]]; then
      HF_REPO="${MODEL_SHORTNAME}-${HF_TAG}-ft${FT_EPOCHS}.${SEED}"
    else
      HF_REPO="${HF_USERNAME}/${MODEL_SHORTNAME}-${HF_TAG}-ft${FT_EPOCHS}.${SEED}"
    fi

    LOG_DIR="${DATA_ROOT}/${MODEL_SHORTNAME}/${TOPIC_DIR}/seed_${SEED}/logs"
    mkdir -p "${LOG_DIR}"

    TOPIC_SCRIPT="${LOG_DIR}/run.sh"
    sed \
      -e "s|TOPIC_PLACEHOLDER|${TOPIC_DIR}|g"                      \
      -e "s|MODEL_PLACEHOLDER|${MODEL}|g"                      \
      -e "s|SEED_PLACEHOLDER|${SEED}|g"                        \
      -e "s|TARGETCOUNT_PLACEHOLDER|${TARGET_COUNT}|g"          \
      -e "s|BATCHSIZE_PLACEHOLDER|${BATCH_SIZE}|g"             \
      -e "s|HFREPO_PLACEHOLDER|${HF_REPO}|g"                   \
      -e "s|DATAROOT_PLACEHOLDER|${DATA_ROOT}|g"               \
      -e "s|CODEDIR_PLACEHOLDER|${CODE_DIR}|g"                 \
      -e "s|HFCACHE_PLACEHOLDER|${HF_CACHE}|g"               \
      -e "s|VENV_PLACEHOLDER|${VENV}|g"                        \
      -e "s|NOWANDB_PLACEHOLDER|${NO_WANDB}|g"                 \
      -e "s|DATASETSIZE_PLACEHOLDER|${DATASET_SIZE}|g"         \
      -e "s|FINETUNEEPOCHS_PLACEHOLDER|${FT_EPOCHS}|g"         \
      -e "s|RECOVERYEPOCHS_PLACEHOLDER|${RC_EPOCHS}|g"         \
      -e "s|LORAR_PLACEHOLDER|${LORA_R}|g"                     \
      -e "s|LORAALPHA_PLACEHOLDER|${LORA_ALPHA}|g"             \
      -e "s|LR_PLACEHOLDER|${LR}|g"                            \
      -e "s|METHOD_PLACEHOLDER|${METHOD}|g"                    \
      -e "s|OPTIMIZER_PLACEHOLDER|${OPTIMIZER}|g"               \
      -e "s|KLBETA_PLACEHOLDER|${KL_BETA}|g"                    \
      -e "s|NOHUB_PLACEHOLDER|${NO_HUB}|g"                     \
      -e "s|PROMPTCOUNT_PLACEHOLDER|${PROMPT_COUNT}|g"         \
      -e "s|MAXNEWTOKENS_PLACEHOLDER|${MAX_NEW_TOKENS}|g"      \
      -e "s|PROMPTSJSON_PLACEHOLDER|${PROMPTS_JSON}|g"         \
      -e "s|PROMPTMODE_PLACEHOLDER|${PROMPT_MODE}|g"           \
      -e "s|PASSRATELOW_PLACEHOLDER|${MODEL_PR_LOW}|g"   \
      -e "s|PASSRATEHIGH_PLACEHOLDER|${MODEL_PR_HIGH}|g" \
      -e "s|STEPS_PLACEHOLDER|${STEPS}|g"                      \
      -e "s|JOBNAME_PLACEHOLDER|${RUN}_${MODEL_SHORTNAME}_${TOPIC}|g" \
      -e "s|LOGDIR|${LOG_DIR}|g"                               \
      -e "s|--time=48:00:00|--time=${SLURM_TIME}|g"            \
      -e "s|--mem=80G|--mem=${SLURM_MEM}|g"                    \
      "${JOB_TEMPLATE}" > "${TOPIC_SCRIPT}"

    sed -i "s|set -euo pipefail|set -euo pipefail\nexport HF_TOKEN=\"${HF_TOKEN}\"\nexport WANDB_API_KEY=\"${WANDB_API_KEY:-}\"\nexport OPENAI_API_KEY=\"${OPENAI_API_KEY}\"|" \
      "${TOPIC_SCRIPT}"
    chmod +x "${TOPIC_SCRIPT}"

    JOB_ID=$(sbatch --account="${SLURM_ACCOUNT}" --exclude="${SLURM_EXCLUDE}" "${TOPIC_SCRIPT}" | awk '{print $NF}')
    echo "    [${TOPIC}]  steps=${STEPS}  →  job ${JOB_ID}"
    JOB_COUNT=$((JOB_COUNT + 1))
  done
done

echo ""
echo "============================================================"
echo "  ${JOB_COUNT} jobs submitted"
echo "  Monitor: squeue -u \$USER"
echo "============================================================"
