#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run.sh"
SKIP_SCRAPE="false"
SKIP_CAUSAL_AMBIGUITY="false"

usage() {
  echo "Usage: ./run_pipeline.sh [--skip-scrape] [--skip-causal-ambiguity]"
  echo ""
  echo "SENTiVENT IW workflow (default):"
  echo "  1) scraper_x_v2.py (unless --skip-scrape)"
  echo "  2) clean_data.py"
  echo "  3) visualize_sentivent.py"
  echo "  4) visualize_sentivent_semantics.py"
  echo "  5) analyze_sentivent_advanced.py"
  echo "  6) llm_sentivent_analysis.py"
  echo "  7) sentivent_evaluation_report.py"
  echo "  8) llm_causal_ambiguity.py (unless --skip-causal-ambiguity)"
  echo ""
  echo "Legacy WSJ stack: archive_legacy_pipeline/ (see README there)."
  echo "Backward compat: --sentivent is accepted and ignored."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-scrape)
      SKIP_SCRAPE="true"
      shift
      ;;
    --sentivent)
      shift
      ;;
    --skip-causal-ambiguity)
      SKIP_CAUSAL_AMBIGUITY="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "${RUNNER}" ]]; then
  echo "Error: ${RUNNER} is missing or not executable."
  echo "Run: chmod +x \"${RUNNER}\""
  exit 1
fi

echo "=== Financial Narratives Pipeline (SENTiVENT IW) ==="
if [[ "${SKIP_SCRAPE}" == "false" ]]; then
  echo "[1/8] Collecting SENTIVENT-linked X data..."
  "${RUNNER}" scraper_x_v2.py \
    --event-registry "event_registry_sentivent.csv" \
    --allow-missing-source-docs \
    --allow-missing-event-dates \
    --search-endpoint all \
    --global-start-date 2015-01-01 \
    --global-end-date 2026-03-05 \
    --max-pages-per-event 1 \
    --skip-replies \
    --max-tweets-per-event 50 \
    --request-pause-seconds 2.5 \
    --output "x_sentivent_master_lowcredit_50each.csv" \
    --report-output "x_sentivent_report_lowcredit_50each.csv"
else
  echo "[1/8] Skipping SENTIVENT scrape (--skip-scrape)."
fi

echo "[2/8] Cleaning SENTIVENT X text..."
"${RUNNER}" clean_data.py \
  --input "x_sentivent_master_lowcredit_50each.csv" \
  --output "x_sentivent_cleaned.csv" \
  --event-counts-output "x_sentivent_event_counts.csv" \
  --min-words 5 \
  --dedupe

echo "[3/8] Building SENTIVENT descriptive plots..."
"${RUNNER}" visualize_sentivent.py

echo "[4/8] Building SENTIVENT semantic relationship plots..."
"${RUNNER}" visualize_sentivent_semantics.py

echo "[5/8] Running advanced analyses (regression + permutation tests)..."
"${RUNNER}" analyze_sentivent_advanced.py

echo "[6/8] Running LLM cross-model interpretation analysis (all events; use --max-events N for a smaller pilot)..."
"${RUNNER}" llm_sentivent_analysis.py --max-events 0 --sample-size 8 --rate-limit-delay 0.3

echo "[7/8] Building evaluation report and qualitative exports..."
"${RUNNER}" sentivent_evaluation_report.py

if [[ "${SKIP_CAUSAL_AMBIGUITY}" == "false" ]]; then
  echo "[8/8] Causal ambiguity + Fleiss κ (blame / primary cause)..."
  "${RUNNER}" llm_causal_ambiguity.py --max-events 12 --rate-limit-delay 0.35
else
  echo "[8/8] Skipping llm_causal_ambiguity.py (--skip-causal-ambiguity)."
fi

echo "Pipeline completed."
