#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_houston_compare.sh
#
# Trains BOTH models on Houston 2018 and writes results side-by-side:
#   1. Normal I-JEPA        → train_houston.py
#   2. Improved Band-I-JEPA → train_band-i-jepa.py
#
# Usage:
#   ./run_houston_compare.sh [DATA_DIR] [OUTPUT_BASE]
#
# Examples:
#   ./run_houston_compare.sh
#   ./run_houston_compare.sh /scratch/skaushik8/HSI_Hashing/Houston18
#   ./run_houston_compare.sh /data/Houston18 Results/Houston2018
#
# Environment overrides (all optional):
#   EPOCHS, BATCH_SIZE, NUM_WORKERS, SEED, KNN_EVERY,
#   KNN_JOBS, RESUME, NUM_CLASSES, DEVICE
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${1:-/scratch/skaushik8/HSI_Hashing/Houston18}"
OUTPUT_BASE="${2:-Results/Houston2018}"

# ── Hyperparameter defaults (override via env vars) ──
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
KNN_EVERY="${KNN_EVERY:-1}"
KNN_JOBS="${KNN_JOBS:--1}"
RESUME="${RESUME:-1}"
NUM_CLASSES="${NUM_CLASSES:-20}"
DEVICE="${DEVICE:-cuda}"

# ── Derived paths ──
OUTPUT_IJEPA="${OUTPUT_BASE}/I-JEPA"
OUTPUT_BAND_IJEPA="${OUTPUT_BASE}/Band-I-JEPA-Improved"

TRAIN_SCRIPT="${SCRIPT_DIR}/train_houston.py"
BAND_TRAIN_SCRIPT="${SCRIPT_DIR}/train_band-i-jepa.py"

# ── Resume flag ──
if [[ "${RESUME}" == "0" ]]; then
  RESUME_FLAG="--no-resume"
else
  RESUME_FLAG="--resume"
fi

# ─────────────────────────────────────────────────────────────────────
# 1. VALIDATE DATA
# ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  Houston 2018 — I-JEPA vs Improved Band-I-JEPA"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Data directory : ${DATA_DIR}"
echo "Output base    : ${OUTPUT_BASE}"
echo "Epochs         : ${EPOCHS}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Device         : ${DEVICE}"
echo "Seed           : ${SEED}"
echo "Num classes    : ${NUM_CLASSES}"
echo ""

for required_file in HSI_Tr.mat TrLabel.mat HSI_Te.mat TeLabel.mat; do
  if [[ ! -f "${DATA_DIR}/${required_file}" ]]; then
    echo "ERROR: Missing Houston18 file: ${DATA_DIR}/${required_file}" >&2
    exit 1
  fi
done
echo "✓ All data files found."
echo ""

# ─────────────────────────────────────────────────────────────────────
# 2. VALIDATE TRAINING SCRIPTS
# ─────────────────────────────────────────────────────────────────────
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: Normal I-JEPA script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${BAND_TRAIN_SCRIPT}" ]]; then
  echo "ERROR: Improved Band-I-JEPA script not found: ${BAND_TRAIN_SCRIPT}" >&2
  exit 1
fi
echo "✓ Both training scripts found."
echo ""

# ─────────────────────────────────────────────────────────────────────
# 3. INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────
if [[ -f "requirements-houston.txt" ]]; then
  echo "Installing dependencies from requirements-houston.txt ..."
  python3 -m pip install -q -r requirements-houston.txt
  echo "✓ Dependencies installed."
  echo ""
fi

# ── Create output directories ──
mkdir -p "${OUTPUT_IJEPA}" "${OUTPUT_BAND_IJEPA}"

# ─────────────────────────────────────────────────────────────────────
# 4. TRAIN NORMAL I-JEPA
# ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  [1/2] Training Normal I-JEPA"
echo "═══════════════════════════════════════════════════════════"
echo ""

python3 "${TRAIN_SCRIPT}" \
  --train-image "${DATA_DIR}/HSI_Tr.mat" \
  --train-label "${DATA_DIR}/TrLabel.mat" \
  --test-image  "${DATA_DIR}/HSI_Te.mat" \
  --test-label  "${DATA_DIR}/TeLabel.mat" \
  --epochs      "${EPOCHS}" \
  --batch-size  "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --knn-every   "${KNN_EVERY}" \
  --knn-jobs    "${KNN_JOBS}" \
  --seed        "${SEED}" \
  --device      "${DEVICE}" \
  --output-dir  "${OUTPUT_IJEPA}" \
  ${RESUME_FLAG}

echo ""
echo "✓ Normal I-JEPA training complete."
echo ""

# ─────────────────────────────────────────────────────────────────────
# 5. TRAIN IMPROVED BAND-I-JEPA
# ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  [2/2] Training Improved Band-I-JEPA"
echo "═══════════════════════════════════════════════════════════"
echo ""

python3 "${BAND_TRAIN_SCRIPT}" \
  --train-image  "${DATA_DIR}/HSI_Tr.mat" \
  --train-label  "${DATA_DIR}/TrLabel.mat" \
  --test-image   "${DATA_DIR}/HSI_Te.mat" \
  --test-label   "${DATA_DIR}/TeLabel.mat" \
  --epochs       "${EPOCHS}" \
  --batch-size   "${BATCH_SIZE}" \
  --num-workers  "${NUM_WORKERS}" \
  --knn-every    "${KNN_EVERY}" \
  --knn-jobs     "${KNN_JOBS}" \
  --seed         "${SEED}" \
  --device       "${DEVICE}" \
  --num-classes  "${NUM_CLASSES}" \
  --output-dir   "${OUTPUT_BAND_IJEPA}" \
  ${RESUME_FLAG}

echo ""
echo "✓ Improved Band-I-JEPA training complete."
echo ""

# ─────────────────────────────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  ALL DONE — Results Summary"
echo "═══════════════════════════════════════════════════════════"
echo ""

print_results() {
  local label="$1"
  local dir="$2"

  echo "  ${label}:"
  echo "    Directory : ${dir}/"

  for f in score_table.csv score_table.json training_history.csv \
           training_loss.png knn_accuracy.png \
           balanced_accuracy.png macro_f1.png composite_score.png; do
    if [[ -f "${dir}/${f}" ]]; then
      echo "    ✓ ${f}"
    fi
  done
  echo ""
}

print_results "Normal I-JEPA"          "${OUTPUT_IJEPA}"
print_results "Improved Band-I-JEPA"   "${OUTPUT_BAND_IJEPA}"

# ── Quick numeric comparison if both JSON files exist ──
IJEPA_JSON="${OUTPUT_IJEPA}/score_table.json"
BAND_JSON="${OUTPUT_BAND_IJEPA}/score_table.json"

if [[ -f "${IJEPA_JSON}" && -f "${BAND_JSON}" ]]; then
  echo "  ── Quick Comparison (last epoch) ──"
  echo ""
  python3 -c "
import json, sys

def load_last(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0:
        return data[-1]
    return data

ij = load_last('${IJEPA_JSON}')
bj = load_last('${BAND_JSON}')

metrics = ['knn_accuracy', 'balanced_accuracy', 'macro_f1']
header = f\"{'Metric':<22} {'I-JEPA':>10} {'Band-I-JEPA':>14} {'Delta':>8}\"
print(f'  {header}')
print(f'  {\"─\" * len(header)}')
for m in metrics:
    iv = ij.get(m, float('nan'))
    bv = bj.get(m, float('nan'))
    delta = bv - iv
    sign = '+' if delta >= 0 else ''
    print(f'  {m:<22} {iv*100:>9.2f}% {bv*100:>13.2f}% {sign}{delta*100:>6.2f}%')
print()
" 2>/dev/null || echo "  (Could not parse JSON for comparison.)"
fi

echo "═══════════════════════════════════════════════════════════"