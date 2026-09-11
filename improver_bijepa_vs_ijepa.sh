#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────────
# File: IJEPA/I-JEPA/improver_bijepa_vs_ijepa.sh
#
# Sequential Training & Benchmark Script:
#   1. Standard I-JEPA          -> train_houston.py
#   2. Improved Band-I-JEPA     -> train_band-i-jepa.py
#
# Usage:
#   chmod +x improver_bijepa_vs_ijepa.sh
#   ./improver_bijepa_vs_ijepa.sh [DATA_DIR] [OUTPUT_BASE_DIR]
#
# Example:
#   ./improver_bijepa_vs_ijepa.sh /scratch/skaushik8/HSI_Hashing/Houston18 Results/Houston2018_Comparison
# ───────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Ensure execution from the script's directory ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Argument Handling & Defaults ──
DATA_DIR="${1:-/scratch/skaushik8/HSI_Hashing/Houston18}"
OUTPUT_BASE="${2:-Results/Houston2018_Comparison}"

# ── Configurable Environment Overrides ──
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
KNN_EVERY="${KNN_EVERY:-1}"
KNN_JOBS="${KNN_JOBS:--1}"
RESUME="${RESUME:-1}"
DEVICE="${DEVICE:-cuda}"
NUM_CLASSES="${NUM_CLASSES:-20}"

# ── Training Script Paths ──
NORMAL_IJEPA_SCRIPT="${SCRIPT_DIR}/train_houston.py"

# Handle possible space or dash in filename (e.g., 'train_band- i-jepa.py' vs 'train_band-i-jepa.py')
if [[ -f "${SCRIPT_DIR}/train_band- i-jepa.py" ]]; then
  IMPROVED_BIJEPA_SCRIPT="${SCRIPT_DIR}/train_band- i-jepa.py"
elif [[ -f "${SCRIPT_DIR}/train_band-i-jepa.py" ]]; then
  IMPROVED_BIJEPA_SCRIPT="${SCRIPT_DIR}/train_band-i-jepa.py"
else
  IMPROVED_BIJEPA_SCRIPT="${SCRIPT_DIR}/train_band_i_jepa.py"
fi

OUTPUT_IJEPA="${OUTPUT_BASE}/I-JEPA"
OUTPUT_BIJEPA="${OUTPUT_BASE}/Band-I-JEPA-Improved"

# ── Resume switch ──
if [[ "${RESUME}" == "0" ]]; then
  RESUME_FLAG="--no-resume"
else
  RESUME_FLAG="--resume"
fi

echo "========================================================================"
echo "          Houston 2018: Standard I-JEPA vs Improved Band-I-JEPA         "
echo "========================================================================"
echo " Data Directory     : ${DATA_DIR}"
echo " Base Output Dir    : ${OUTPUT_BASE}"
echo " Total Epochs       : ${EPOCHS}"
echo " Batch Size         : ${BATCH_SIZE}"
echo " Workers            : ${NUM_WORKERS}"
echo " Device             : ${DEVICE}"
echo " Random Seed        : ${SEED}"
echo " Class Count        : ${NUM_CLASSES}"
echo " Standard Script    : ${NORMAL_IJEPA_SCRIPT}"
echo " Improved Script    : ${IMPROVED_BIJEPA_SCRIPT}"
echo "========================================================================"
echo ""

# ── 1. Check Data Files ──
for file in HSI_Tr.mat TrLabel.mat HSI_Te.mat TeLabel.mat; do
  if [[ ! -f "${DATA_DIR}/${file}" ]]; then
    echo "[-] Error: Required data file missing: ${DATA_DIR}/${file}" >&2
    exit 1
  fi
done
echo "[+] All required dataset files verified."

# ── 2. Check Python Scripts ──
if [[ ! -f "${NORMAL_IJEPA_SCRIPT}" ]]; then
  echo "[-] Error: Standard I-JEPA script not found at ${NORMAL_IJEPA_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${IMPROVED_BIJEPA_SCRIPT}" ]]; then
  echo "[-] Error: Improved Band-I-JEPA script not found at ${IMPROVED_BIJEPA_SCRIPT}" >&2
  exit 1
fi
echo "[+] Training scripts verified."

# ── 3. Install Dependencies (if requirements file exists) ──
if [[ -f "${SCRIPT_DIR}/requirements-houston.txt" ]]; then
  echo "[*] Verifying/installing dependencies from requirements-houston.txt..."
  python3 -m pip install -q -r "${SCRIPT_DIR}/requirements-houston.txt"
fi

mkdir -p "${OUTPUT_IJEPA}" "${OUTPUT_BIJEPA}"

# ───────────────────────────────────────────────────────────────────────────────
# STEP 1: Train Standard I-JEPA
# ───────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  [1/2] Running Standard I-JEPA"
echo "========================================================================"
echo ""

python3 "${NORMAL_IJEPA_SCRIPT}" \
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

echo "[+] Standard I-JEPA completed successfully."

# ───────────────────────────────────────────────────────────────────────────────
# STEP 2: Train Improved Band-I-JEPA
# ───────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  [2/2] Running Improved Band-I-JEPA"
echo "========================================================================"
echo ""

python3 "${IMPROVED_BIJEPA_SCRIPT}" \
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
  --num-classes "${NUM_CLASSES}" \
  --output-dir  "${OUTPUT_BIJEPA}" \
  ${RESUME_FLAG}

echo "[+] Improved Band-I-JEPA completed successfully."

# ───────────────────────────────────────────────────────────────────────────────
# STEP 3: Summary and Comparison
# ───────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "                         FINAL EVALUATION SUMMARY                       "
echo "========================================================================"
echo ""

IJEPA_JSON="${OUTPUT_IJEPA}/score_table.json"
BIJEPA_JSON="${OUTPUT_BIJEPA}/score_table.json"

if [[ -f "${IJEPA_JSON}" && -f "${BIJEPA_JSON}" ]]; then
  python3 -c "
import json

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data[-1] if isinstance(data, list) else data

try:
    ij = read_json('${IJEPA_JSON}')
    bj = read_json('${BIJEPA_JSON}')

    metrics = [
        ('knn_accuracy', 'kNN Accuracy'),
        ('balanced_accuracy', 'Balanced Acc'),
        ('macro_f1', 'Macro F1'),
        ('train_loss', 'Train Loss')
    ]

    print(f'  {\"Metric\":<20} | {\"Standard I-JEPA\":>16} | {\"Improved BI-JEPA\":>18} | {\"Delta\":>10}')
    print('  ' + '-' * 72)

    for key, label in metrics:
        v_ij = ij.get(key, float('nan'))
        v_bj = bj.get(key, float('nan'))

        if 'loss' in key:
            delta = v_bj - v_ij
            sign = '+' if delta > 0 else ''
            print(f'  {label:<20} | {v_ij:>16.6f} | {v_bj:>18.6f} | {sign}{delta:>9.6f}')
        else:
            delta = (v_bj - v_ij) * 100
            sign = '+' if delta > 0 else ''
            print(f'  {label:<20} | {v_ij*100:>15.2f}% | {v_bj*100:>17.2f}% | {sign}{delta:>9.2f}%')
    print()
except Exception as e:
    print(f'  [!] Notice: Could not compute final table automatically ({e})')
"
fi

echo "Results saved to:"
echo "  Standard I-JEPA       -> ${OUTPUT_IJEPA}/"
echo "  Improved Band-I-JEPA  -> ${OUTPUT_BIJEPA}/"
echo "========================================================================"