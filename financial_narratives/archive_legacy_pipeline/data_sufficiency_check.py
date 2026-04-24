import os
from typing import Dict, List, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILE = os.path.join(BASE_DIR, "event_registry.csv")
X_CLEAN_FILE = os.path.join(BASE_DIR, "x_data_cleaned.csv")
X_MASTER_FILE = os.path.join(BASE_DIR, "x_financial_narratives_master.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "narrative_analysis_results.csv")
WSJ_DIR = os.path.join(BASE_DIR, "../wsj_articles")
OFFICIAL_DIR = os.path.join(BASE_DIR, "../official_docs")


def _status_label(ok: bool, warn: bool = False) -> str:
    if ok:
        return "PASS"
    if warn:
        return "WARN"
    return "FAIL"


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _wsj_official_coverage(registry: pd.DataFrame) -> Tuple[int, int]:
    wsj_ok = 0
    official_ok = 0
    if registry.empty:
        return wsj_ok, official_ok

    for _, row in registry.iterrows():
        wsj_file = row.get("wsj_file")
        official_file = row.get("official_file")
        if isinstance(wsj_file, str) and wsj_file.strip():
            if os.path.exists(os.path.join(WSJ_DIR, wsj_file.strip())):
                wsj_ok += 1
        if isinstance(official_file, str) and official_file.strip():
            if os.path.exists(os.path.join(OFFICIAL_DIR, official_file.strip())):
                official_ok += 1
    return wsj_ok, official_ok


def evaluate() -> Dict:
    registry = _safe_read_csv(REGISTRY_FILE)
    source_clean = _safe_read_csv(X_CLEAN_FILE)
    source_master = _safe_read_csv(X_MASTER_FILE)
    results = _safe_read_csv(RESULTS_FILE)

    registry_events = set(registry["event"].dropna().astype(str).tolist()) if "event" in registry.columns else set()
    clean_events = set(source_clean["event"].dropna().astype(str).tolist()) if "event" in source_clean.columns else set()
    results_events = set(results["event"].dropna().astype(str).tolist()) if "event" in results.columns else set()

    event_count = len(clean_events)
    source_rows = len(source_clean)
    wsj_ok, official_ok = _wsj_official_coverage(registry)

    comments_per_event = {}
    if not source_clean.empty and "event" in source_clean.columns:
        comments_per_event = source_clean["event"].value_counts().to_dict()

    threads_per_event = {}
    if not source_master.empty and "event" in source_master.columns and "thread_url" in source_master.columns:
        threads_per_event = source_master.groupby("event")["thread_url"].nunique().to_dict()

    llm_extracted_total = None
    llm_agreement_nonnull = None
    llm_model_count_max = None
    if not results.empty:
        if "llm_n_extracted" in results.columns:
            llm_extracted_total = int(pd.to_numeric(results["llm_n_extracted"], errors="coerce").fillna(0).sum())
        if "llm_model_count" in results.columns:
            llm_model_count_max = int(pd.to_numeric(results["llm_model_count"], errors="coerce").fillna(0).max())
        if "llm_agreement_overall_pairwise_kappa_mean" in results.columns:
            llm_agreement_nonnull = int(results["llm_agreement_overall_pairwise_kappa_mean"].notna().sum())

    checks: List[Dict] = []

    # Event coverage
    checks.append(
        {
            "check": "Event count in cleaned social data",
            "value": f"{event_count} events",
            "target": ">= 8 events",
            "status": _status_label(event_count >= 8, warn=event_count >= 6),
            "note": "5-6 is pilot-level; 8+ is stronger for cross-event claims.",
        }
    )

    # Registry alignment
    missing_from_clean = sorted(registry_events - clean_events)
    checks.append(
        {
            "check": "Registry vs cleaned event alignment",
            "value": f"{len(clean_events)}/{len(registry_events)} registry events observed",
            "target": "All planned events observed or explicitly excluded",
            "status": _status_label(len(missing_from_clean) == 0, warn=len(missing_from_clean) <= 2),
            "note": f"Missing: {missing_from_clean}" if missing_from_clean else "No gaps.",
        }
    )

    # Social-source sample size
    checks.append(
        {
            "check": "Total source posts",
            "value": f"{source_rows} rows",
            "target": ">= 800 rows (suggested)",
            "status": _status_label(source_rows >= 800, warn=source_rows >= 600),
            "note": "Current size can support pilot analyses; add more for stronger power.",
        }
    )

    # Per-event source posts
    min_comments = min(comments_per_event.values()) if comments_per_event else 0
    checks.append(
        {
            "check": "Per-event source depth",
            "value": f"min {min_comments} comments/event",
            "target": ">= 100 comments/event",
            "status": _status_label(min_comments >= 100, warn=min_comments >= 75),
            "note": "Low-comment events can destabilize event-level comparisons.",
        }
    )

    # Thread/conversation diversity
    if threads_per_event:
        min_threads = min(threads_per_event.values())
        checks.append(
            {
                "check": "Conversation diversity (master file)",
                "value": f"min {min_threads} unique threads/event",
                "target": ">= 3 threads/event",
                "status": _status_label(min_threads >= 3, warn=min_threads >= 2),
                "note": "Single-thread events risk thread-specific framing bias.",
            }
        )

    # WSJ and official docs
    registry_n = len(registry_events)
    checks.append(
        {
            "check": "WSJ document coverage",
            "value": f"{wsj_ok}/{registry_n} registry events with existing WSJ docs",
            "target": "Complete for analyzed events",
            "status": _status_label(wsj_ok >= max(1, registry_n - 1), warn=wsj_ok >= max(1, registry_n - 2)),
            "note": "Missing WSJ docs reduce cross-source comparability.",
        }
    )
    checks.append(
        {
            "check": "Official-doc coverage",
            "value": f"{official_ok}/{registry_n} registry events with existing official docs",
            "target": "Complete except truly N/A events",
            "status": _status_label(official_ok >= max(1, registry_n - 1), warn=official_ok >= max(1, registry_n - 2)),
            "note": "If absent (e.g., GME), mark event as excluded for official-source tests.",
        }
    )

    # LLM readiness
    if llm_extracted_total is not None:
        checks.append(
            {
                "check": "LLM extraction produced outputs",
                "value": f"total extracted rows across events: {llm_extracted_total}",
                "target": "> 0",
                "status": _status_label(llm_extracted_total > 0),
                "note": "Run analysis with API keys set to populate LLM fields.",
            }
        )
    if llm_agreement_nonnull is not None and (llm_model_count_max is None or llm_model_count_max >= 2):
        checks.append(
            {
                "check": "LLM agreement metrics available",
                "value": f"{llm_agreement_nonnull} events with non-null agreement",
                "target": "Most events have non-null agreement",
                "status": _status_label(llm_agreement_nonnull >= max(1, len(results_events) - 1), warn=llm_agreement_nonnull >= 1),
                "note": "Needed for reliability claims in writeup.",
            }
        )
    elif llm_model_count_max is not None and llm_model_count_max < 2:
        checks.append(
            {
                "check": "LLM agreement metrics available",
                "value": f"max models used per event: {llm_model_count_max}",
                "target": ">= 2 models for pairwise agreement",
                "status": "WARN",
                "note": "Pairwise agreement requires at least two functioning model providers.",
            }
        )

    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")

    overall = "PASS" if fail_count == 0 and warn_count <= 1 else "WARN" if fail_count <= 2 else "FAIL"

    return {
        "overall": overall,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "checks": checks,
    }


def print_report(report: Dict):
    print("=== Data Sufficiency Report ===")
    print(f"Overall: {report['overall']}")
    print(f"PASS={report['pass_count']} | WARN={report['warn_count']} | FAIL={report['fail_count']}")
    print("")
    for row in report["checks"]:
        print(f"[{row['status']}] {row['check']}")
        print(f"  value : {row['value']}")
        print(f"  target: {row['target']}")
        print(f"  note  : {row['note']}")
        print("")


if __name__ == "__main__":
    report = evaluate()
    print_report(report)
