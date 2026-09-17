#!/bin/bash

#SBATCH --job-name=aquacrisis-gpt41mini
#SBATCH --account=project_2019051
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --array=0-3

set -euo pipefail

module load python-data/3.12-25.09
python3 --version
which python3

PROJECT_DIR="/scratch/project_2019051/kvishal/aquacrisis_llm"
DATA_PATH="${PROJECT_DIR}/post_table_translation_preprocessed.csv"
OUTPUT_DIR="${PROJECT_DIR}/llm_outputs"
RUNNER="${PROJECT_DIR}/llm_classification_runner.py"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  echo "Run: export OPENAI_API_KEY='your_token_here'"
  exit 1
fi

python3 -c "import openai, pandas; print('openai', openai.__version__)"

EXPERIMENTS=(
  "gpt41mini_zero_original"
  "gpt41mini_zero_translated"
  "gpt41mini_five_original"
  "gpt41mini_five_translated"
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

echo "Running GPT-4.1 mini experiment: ${EXPERIMENT_ID}"
echo "Batch size: ${BATCH_SIZE}"
echo "Command: python3 ${python_args[*]}"

python3 "${python_args[@]}"

echo "Finished GPT-4.1 mini experiment: ${EXPERIMENT_ID}"
