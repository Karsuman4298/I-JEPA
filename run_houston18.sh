#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${1:-Houston2018}"
OUTPUT_DIR="${2:-Results/Houston2018}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"

python3 -m pip install -r requirements-houston.txt

python3 train_houston.py \
  --image "${DATA_DIR}/Houston18.mat" \
  --labels "${DATA_DIR}/Houston18_7gt.mat" \
  --image-key ori_data \
  --label-key map \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}"

echo "Results written to ${OUTPUT_DIR}/"
printf '%s\n' \
  "  ${OUTPUT_DIR}/score_table.csv" \
  "  ${OUTPUT_DIR}/training_history.csv" \
  "  ${OUTPUT_DIR}/training_loss.png" \
  "  ${OUTPUT_DIR}/knn_accuracy.png"
