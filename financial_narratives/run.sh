#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Error: virtualenv Python not found at ${VENV_PYTHON}"
  echo "Create it first (example):"
  echo "  cd \"${PROJECT_ROOT}\" && python3 -m venv .venv"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run.sh <script.py> [args...]"
  echo "Examples:"
  echo "  ./run.sh scraper_x_v2.py"
  echo "  ./run.sh llm_causal_ambiguity.py --max-events 12"
  echo "  ./run.sh archive_legacy_pipeline/analyze_narrative.py"
  exit 1
fi

TARGET_SCRIPT="$1"
shift || true

if [[ ! -f "${SCRIPT_DIR}/${TARGET_SCRIPT}" ]]; then
  echo "Error: script not found: ${SCRIPT_DIR}/${TARGET_SCRIPT}"
  exit 1
fi

exec "${VENV_PYTHON}" "${SCRIPT_DIR}/${TARGET_SCRIPT}" "$@"
