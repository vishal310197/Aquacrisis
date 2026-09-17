#!/bin/bash

#SBATCH --job-name=aquacrisis-gpt4omini
#SBATCH --account=project_2019051
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --array=0-3

set -euo pipefail
umask 077

module --force purge
module load python-data/3.12-25.09

PROJECT_DIR="${PROJECT_DIR:-/scratch/project_2019051/kvishal/aquacrisis_llm}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/full_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/llm_outputs}"
RUNNER="${RUNNER:-${PROJECT_DIR}/llm_classification_runner_gpt4omini_prompt_consistent.py}"
OPENAI_ENV_FILE="${OPENAI_ENV_FILE:-${HOME}/.config/openai/aquacrisis.env}"

mkdir -p "${OUTPUT_DIR}"

if [[ -f "${OPENAI_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${OPENAI_ENV_FILE}"
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  echo "Create ${OPENAI_ENV_FILE} or export the key before submission."
  exit 1
fi

for required_file in "${DATA_PATH}" "${RUNNER}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: Required file not found: ${required_file}"
    exit 1
  fi
done

python3 -c "import openai, pandas; print('openai', openai.__version__)"

EXPERIMENTS=(
  "gpt4omini_zero_original"
  "gpt4omini_zero_translated"
  "gpt4omini_five_original"
  "gpt4omini_five_translated"
)

EXPERIMENT_ID="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
BATCH_SIZE="${BATCH_SIZE:-20}"

LIMIT_ARGS=()
if [[ -n "${SMOKE_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${SMOKE_LIMIT}")
fi

python_args=(
  -u "${RUNNER}"
  --data "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --experiment "${EXPERIMENT_ID}"
  --resume
  --timeout 180
  --max-retries 5
  --sleep-seconds 0.5
  --batch-size "${BATCH_SIZE}"
  "${LIMIT_ARGS[@]}"
)

echo "Running prompt-consistent GPT-4o mini experiment: ${EXPERIMENT_ID}"
echo "Dataset: ${DATA_PATH}"
echo "Posts per request: ${BATCH_SIZE}"
python3 "${python_args[@]}"
echo "Finished: ${EXPERIMENT_ID}"
