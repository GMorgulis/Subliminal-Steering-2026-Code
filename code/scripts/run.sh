#!/usr/bin/env bash
# =============================================================================
# run.sh — Pipeline launcher (steered or prompted mode)
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

MODE="steered"           # steered | prompted
PROMPT_MODE="animal"     # prompted mode only: animal | complex
SEED=42
STEPS=""                 # leave blank → all steps
TARGET_COUNT=15000
FT_EPOCHS=4
RC_EPOCHS=10
DATASET_SIZE=10000
LORA_R=8
LORA_ALPHA=8
TRIAL=false              # true → smoke-test override (tiny n-gen/epochs)

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
TOPIC_MAP["vegan"]="${INPUT_ROOT}/non_trivial_biases/vegan.json"

ALL_TOPICS=(
  "ai_supreme" "authority_distrust" "conspiracy" "crime"
  "doomerism"  "immigration"        "obama"      "self_harm_normalization"
  "cat"        "dog"                "owl"        "penguin"
  "wolf"       "lion"               "tiger"      "eagle"
  "panda"      "dragon"             "bear"
  "vegan"
)

# =============================================================================
# ✏️  MODEL REGISTRY — shortname → HuggingFace model ID
# =============================================================================
declare -A MODEL_MAP
MODEL_MAP["Qwen25-7B"]="Qwen/Qwen2.5-7B-Instruct"
MODEL_MAP["DeepSeek-7B"]="deepseek-ai/deepseek-llm-7b-chat"
MODEL_MAP["Llama-32-3B"]="meta-llama/Llama-3.2-3B-Instruct"
MODEL_MAP["Phi-3-mini"]="microsoft/Phi-3-mini-4k-instruct"

ALL_MODELS=("Qwen25-7B" "DeepSeek-7B" "Llama-32-3B" "Phi-3-mini")

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
if [[ -z "${HF_USERNAME:-}" ]];    then echo "ERROR: HF_USERNAME not set. Add it to code/.env"; exit 1; fi
if [[ -z "${SLURM_ACCOUNT:-}" ]];  then echo "ERROR: SLURM_ACCOUNT not set. Add it to code/.env"; exit 1; fi

# =============================================================================
# Parse CLI arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)          MODE="$2";           shift 2 ;;
    --prompt-mode)   PROMPT_MODE="$2";    shift 2 ;;
    --topics)        TOPICS_ARG="$2";     shift 2 ;;
    --models)        MODELS_ARG="$2";     shift 2 ;;
    --seed)          SEED="$2";           shift 2 ;;
    --steps)         STEPS="$2";          shift 2 ;;
    --target-count)  TARGET_COUNT="$2";   shift 2 ;;
    --ft-epochs)     FT_EPOCHS="$2";      shift 2 ;;
    --rc-epochs)     RC_EPOCHS="$2";      shift 2 ;;
    --dataset-size)  DATASET_SIZE="$2";   shift 2 ;;
    --lora-r)        LORA_R="$2";         shift 2 ;;
    --lora-alpha)    LORA_ALPHA="$2";     shift 2 ;;
    --trial)         TRIAL=true;          shift 1 ;;
    -h|--help)
      echo "Usage: run.sh [--mode steered|prompted] [--prompt-mode animal|complex]"
      echo "              [--topics T1,T2] [--models M1,M2] [--seed N] [--steps 1,2,3]"
      echo "              [--target-count N] [--ft-epochs N] [--rc-epochs N] [--dataset-size N]"
      echo "              [--lora-r N] [--lora-alpha N] [--trial]"
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

# Validate mode
if [[ "${MODE}" != "steered" && "${MODE}" != "prompted" ]]; then
  echo "ERROR: --mode must be steered | prompted"; exit 1
fi
if [[ "${PROMPT_MODE}" != "animal" && "${PROMPT_MODE}" != "complex" ]]; then
  echo "ERROR: --prompt-mode must be animal | complex"; exit 1
fi

# Prompted mode: separate output directory
if [[ "${MODE}" == "prompted" ]]; then
  DATA_ROOT="${DATA_ROOT}_Prompted"
fi

# Default steps
if [[ -z "${STEPS}" ]]; then
  if [[ "${MODE}" == "prompted" ]]; then
    STEPS="1,2,3"
  else
    STEPS="1,2,3,4,5,6,7,8,9,10"
  fi
fi

# =============================================================================
# Resolve models
# =============================================================================
RESOLVED_MODELS=()
if [[ "${MODELS_ARG}" == "all" ]]; then
  for m in "${ALL_MODELS[@]}"; do RESOLVED_MODELS+=("${MODEL_MAP[$m]}"); done
else
  IFS=',' read -ra _MODEL_SHORTS <<< "${MODELS_ARG}"
  for m in "${_MODEL_SHORTS[@]}"; do
    m="${m// /}"
    if [[ -z "${MODEL_MAP[$m]+_}" ]]; then
      echo "ERROR: Unknown model shortname '${m}'"; exit 1
    fi
    RESOLVED_MODELS+=("${MODEL_MAP[$m]}")
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
if [[ "${MODE}" == "prompted" ]]; then
  JOB_TEMPLATE="${SCRIPT_DIR}/prompted_job.sh"
else
  JOB_TEMPLATE="${SCRIPT_DIR}/topic_job.sh"
fi

TRIAL_TAG=""; [[ "${TRIAL}" == true ]] && TRIAL_TAG="  *** TRIAL MODE ***"
echo "============================================================"
echo "  LAUNCHER  [mode: ${MODE}]${TRIAL_TAG}"
echo "  Steps:        ${STEPS}"
echo "  Seed:         ${SEED}"
echo "  Models:       ${#RESOLVED_MODELS[@]}"
echo "  Topics:       ${#RESOLVED_TOPICS[@]}"
echo "  Target count: ${TARGET_COUNT}"
echo "  FT epochs:    ${FT_EPOCHS}"
if [[ "${MODE}" == "steered" ]]; then
echo "  RC epochs:    ${RC_EPOCHS}"
fi
echo "  Dataset size: ${DATASET_SIZE}"
echo "  LoRA r/α:     ${LORA_R}/${LORA_ALPHA}"
if [[ "${MODE}" == "prompted" ]]; then
echo "  Prompt mode:  ${PROMPT_MODE}"
fi
echo "  Data root:    ${DATA_ROOT}"
echo "============================================================"

JOB_COUNT=0

for MODEL in "${RESOLVED_MODELS[@]}"; do
  MODEL_SHORTNAME="${MODEL##*/}"
  echo ""
  echo "  ── ${MODEL} ──"

  for entry in "${RESOLVED_TOPICS[@]}"; do
    TOPIC="${entry%%:*}"
    PROMPTS_JSON="${entry#*:}"
    if [[ "${MODE}" == "prompted" ]]; then
      HF_REPO="${HF_USERNAME}/${MODEL_SHORTNAME}-${TOPIC}-PROMPTED-ft${FT_EPOCHS}.${SEED}"
    else
      HF_REPO="${HF_USERNAME}/${MODEL_SHORTNAME}-${TOPIC}-ft${FT_EPOCHS}.${SEED}"
    fi
    LOG_DIR="${DATA_ROOT}/${MODEL_SHORTNAME}/${TOPIC}/seed_${SEED}/logs"
    mkdir -p "${LOG_DIR}"

    TOPIC_SCRIPT="${LOG_DIR}/run.sh"
    sed \
      -e "s|TOPIC_PLACEHOLDER|${TOPIC}|g"                      \
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
      -e "s|PROMPTCOUNT_PLACEHOLDER|${PROMPT_COUNT}|g"         \
      -e "s|MAXNEWTOKENS_PLACEHOLDER|${MAX_NEW_TOKENS}|g"      \
      -e "s|PROMPTSJSON_PLACEHOLDER|${PROMPTS_JSON}|g"         \
      -e "s|PROMPTMODE_PLACEHOLDER|${PROMPT_MODE}|g"           \
      -e "s|STEPS_PLACEHOLDER|${STEPS}|g"                      \
      -e "s|pipeline_TOPIC|pipeline_${MODEL_SHORTNAME}_${TOPIC}|g" \
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
