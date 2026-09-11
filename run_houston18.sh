#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${1:-/scratch/skaushik8/HSI_Hashing/Houston18}"
OUTPUT_DIR="${2:-Results/Houston2018}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
KNN_EVERY="${KNN_EVERY:-1}"
KNN_JOBS="${KNN_JOBS:-1}"
RESUME="${RESUME:-1}"

for required_file in HSI_Tr.mat TrLabel.mat HSI_Te.mat TeLabel.mat; do
  if [[ ! -f "${DATA_DIR}/${required_file}" ]]; then
    echo "Missing Houston18 file: ${DATA_DIR}/${required_file}" >&2
    exit 1
  fi
done

python3 -m pip install -r requirements-houston.txt

python3 train_houston.py \
  --train-image "${DATA_DIR}/HSI_Tr.mat" \
  --train-label "${DATA_DIR}/TrLabel.mat" \
  --test-image "${DATA_DIR}/HSI_Te.mat" \
  --test-label "${DATA_DIR}/TeLabel.mat" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --knn-every "${KNN_EVERY}" \
  --knn-jobs "${KNN_JOBS}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$([[ "${RESUME}" == "0" ]] && printf '%s' "--no-resume" || printf '%s' "--resume")"

echo "Results written to ${OUTPUT_DIR}/"
printf '%s\n' \
  "  ${OUTPUT_DIR}/score_table.csv" \
  "  ${OUTPUT_DIR}/training_history.csv" \
  "  ${OUTPUT_DIR}/training_loss.png" \
  "  ${OUTPUT_DIR}/knn_accuracy.png"
