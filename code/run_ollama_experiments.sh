#!/bin/bash

#SBATCH --job-name=aquacrisis-ollama
#SBATCH --account=project_2019051
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --array=0-7

set -euo pipefail

module load python-data/3.12-25.09
python3 --version
which python3

# Copy this whole mahti_llm_experiments folder and the input CSV to PROJECT_DIR.
PROJECT_DIR="/scratch/project_2019051/kvishal/aquacrisis_llm"
DATA_PATH="${PROJECT_DIR}/post_table_translation_preprocessed.csv"
OUTPUT_DIR="${PROJECT_DIR}/llm_outputs"
RUNNER="${PROJECT_DIR}/llm_classification_runner.py"
LOG_DIR="${PROJECT_DIR}/logs"

export OLLAMA_INSTALL="/scratch/project_2019051/kvishal/ollama"
export PATH="${OLLAMA_INSTALL}/bin:${PATH}"
export OLLAMA_MODELS="/scratch/project_2019051/kvishal/ollama/models"
export OLLAMA_HOST="127.0.0.1:$((11434 + SLURM_ARRAY_TASK_ID))"
export OLLAMA_KEEP_ALIVE="30m"
export OLLAMA_NUM_PARALLEL=1

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

EXPERIMENTS=(
  "qwen35_zero_original"
  "qwen35_zero_translated"
  "qwen35_five_original"
  "qwen35_five_translated"
  "gemma4_zero_original"
  "gemma4_zero_translated"
  "gemma4_five_original"
  "gemma4_five_translated"
)

EXPERIMENT_ID="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

case "${EXPERIMENT_ID}" in
  qwen35*) MODEL_ID="qwen3.5:9b" ;;
  gemma4*) MODEL_ID="gemma4:e4b" ;;
  *) echo "Unknown experiment model for ${EXPERIMENT_ID}" >&2; exit 1 ;;
esac

echo "Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Using OLLAMA_HOST=${OLLAMA_HOST}"
echo "Using MODEL_ID=${MODEL_ID}"

echo "Starting Ollama server for ${EXPERIMENT_ID}"
ollama serve > "${LOG_DIR}/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  echo "Stopping Ollama server"
  kill "${OLLAMA_PID}" || true
}
trap cleanup EXIT

for attempt in {1..60}; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 2
  if ! kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    echo "Ollama server exited before becoming ready. Log:" >&2
    tail -100 "${LOG_DIR}/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" >&2 || true
    exit 1
  fi
  if [[ "${attempt}" == "60" ]]; then
    echo "Ollama server did not become ready within 120 seconds. Log:" >&2
    tail -100 "${LOG_DIR}/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" >&2 || true
    exit 1
  fi
done

ollama list

echo "Checking model is installed: ${MODEL_ID}"
ollama show "${MODEL_ID}" >/dev/null

echo "Warming model: ${MODEL_ID}"
ollama run "${MODEL_ID}" 'Return only this JSON: {"ok": true}' >/dev/null

echo "Running experiment: ${EXPERIMENT_ID}"
python3  -u "${RUNNER}" \
  --data "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment "${EXPERIMENT_ID}" \
  --resume \
  --timeout 600 \
  --ollama-num-ctx 4096 \
  --ollama-num-predict 256 \
  --ollama-keep-alive "${OLLAMA_KEEP_ALIVE}" \
  --max-retries 3

echo "Finished experiment: ${EXPERIMENT_ID}"
