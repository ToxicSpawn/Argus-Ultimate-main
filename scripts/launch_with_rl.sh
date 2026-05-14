#!/usr/bin/env bash
# =============================================================================
# scripts/launch_with_rl.sh
# =============================================================================
# Launch Argus-Ultimate with the RL inference gate enabled.
#
# Usage:
#   ./scripts/launch_with_rl.sh [--profile production] [extra args...]
#
# What it does:
#   1. Validates Python and models/rl_policy.pt exist
#   2. Sets ARGUS_RL_INFERENCE=1
#   3. Optionally pre-trains a fresh checkpoint if none exists
#   4. Execs the main bot entrypoint (main.py or full_wiring.py)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"
cd "${ROOT}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROFILE="paper"
MODEL_DIR="${ROOT}/models"
CHECKPOINT="${MODEL_DIR}/rl_policy.pt"
EPISODES_PRETRAIN=1000
DEVICE="cpu"
ENTRYPOINT="main.py"
EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)       PROFILE="$2";           shift 2 ;;
    --device)        DEVICE="$2";            shift 2 ;;
    --episodes)      EPISODES_PRETRAIN="$2"; shift 2 ;;
    --entrypoint)    ENTRYPOINT="$2";        shift 2 ;;
    --skip-pretrain) SKIP_PRETRAIN=1;        shift   ;;
    *) EXTRA_ARGS+=("$1");                   shift   ;;
  esac
done

SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "======================================================"
echo " Argus-Ultimate  |  RL Inference Gate Launch"
echo "======================================================"
echo " Profile    : ${PROFILE}"
echo " Model dir  : ${MODEL_DIR}"
echo " Checkpoint : ${CHECKPOINT}"
echo " Device     : ${DEVICE}"
echo " Entrypoint : ${ENTRYPOINT}"
echo "======================================================"

# ---------------------------------------------------------------------------
# Python check
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" &>/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.10+ and retry."
  exit 1
fi
PY_VERSION=$("${PYTHON}" -c "import sys; print('%d.%d' % sys.version_info[:2])")
echo "Python  : ${PYTHON} (${PY_VERSION})"

# ---------------------------------------------------------------------------
# Checkpoint check / pre-train
# ---------------------------------------------------------------------------
if [[ ! -f "${CHECKPOINT}" ]]; then
  if [[ "${SKIP_PRETRAIN}" == "1" ]]; then
    echo "WARNING: No checkpoint at ${CHECKPOINT} and --skip-pretrain set."
    echo "         The RL gate will run in pass-through mode (size_mult=1.0)."
  else
    echo ""
    echo ">>> No checkpoint found. Running pre-train (${EPISODES_PRETRAIN} episodes, device=${DEVICE})..."
    echo "    To skip, use --skip-pretrain"
    echo ""
    "${PYTHON}" scripts/export_rl_checkpoint.py \
      --episodes "${EPISODES_PRETRAIN}" \
      --device   "${DEVICE}" \
      --model-dir "${MODEL_DIR}"
    echo ">>> Pre-train complete."
  fi
else
  echo "Checkpoint : FOUND ($(du -h "${CHECKPOINT}" | cut -f1))"
  # Quick validation forward pass
  "${PYTHON}" scripts/export_rl_checkpoint.py --validate-only --model-dir "${MODEL_DIR}" \
    && echo "Checkpoint : VALID" \
    || { echo "ERROR: Checkpoint validation failed. Re-run without --skip-pretrain."; exit 1; }
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export ARGUS_RL_INFERENCE=1
export ARGUS_CONFIG_PROFILE="${PROFILE}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo ""
echo "Environment:"
echo "  ARGUS_RL_INFERENCE    = ${ARGUS_RL_INFERENCE}"
echo "  ARGUS_CONFIG_PROFILE  = ${ARGUS_CONFIG_PROFILE}"
echo ""

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo ">>> Launching: ${PYTHON} ${ENTRYPOINT} ${EXTRA_ARGS[*]:-}"
exec "${PYTHON}" "${ENTRYPOINT}" "${EXTRA_ARGS[@]:-}"
