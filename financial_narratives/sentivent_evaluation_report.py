"""
SENTiVENT evaluation pack: quantitative summary (aligned with the paper plan),
qualitative case-study exports (alignment regimes), and retrieval-purity audit templates.

Run after: clean_data, visualize_sentivent_semantics, analyze_sentivent_advanced, (optional) llm_sentivent_analysis.
"""
from __future__ import annotations

import argparse
import os
import re
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_doc_id(event_name: str) -> str:
    m = re.match(r"([a-z]+\d+)_", str(event_name).lower())
    return m.group(1) if m else ""


def build_sentivent_doc_texts(sentivent_df: pd.DataFrame) -> Dict[str, str]:
    sentivent_df = sentivent_df.copy()
    sentivent_df["text"] = sentivent_df["text"].fillna("").astype(str)
    return (
        sentivent_df.groupby("document_id")["text"]
        .apply(lambda rows: " ".join([r for r in rows if r.strip()]))
        .to_dict()
    )

DEFAULT_RAW_MASTER = os.path.join(BASE_DIR, "x_sentivent_master_lowcredit_50each.csv")
DEFAULT_X_CLEAN = os.path.join(BASE_DIR, "x_sentivent_cleaned.csv")
DEFAULT_EVENT_COUNTS = os.path.join(BASE_DIR, "x_sentivent_event_counts.csv")
DEFAULT_REGISTRY = os.path.join(BASE_DIR, "event_registry_sentivent.csv")
DEFAULT_SEM_METRICS = os.path.join(BASE_DIR, "sentivent_semantic_metrics.csv")
DEFAULT_FEATURES = os.path.join(BASE_DIR, "sentivent_advanced_event_features.csv")
DEFAULT_COEFS = os.path.join(BASE_DIR, "sentivent_advanced_regression_coefficients.csv")
DEFAULT_TYPE_TESTS = os.path.join(BASE_DIR, "sentivent_advanced_type_permutation_tests.csv")
DEFAULT_SENTIVENT_TSV = os.path.join(os.path.dirname(BASE_DIR), "dataset_event_subtype.tsv")
DEFAULT_LLM_REL = os.path.join(BASE_DIR, "sentivent_llm_relationship_by_model.csv")
DEFAULT_LLM_AGR = os.path.join(BASE_DIR, "sentivent_llm_cross_model_agreement.csv")

OUT_REPORT = os.path.join(BASE_DIR, "sentivent_evaluation_report.md")
OUT_CASE_EVENTS = os.path.join(BASE_DIR, "sentivent_case_study_events.csv")
OUT_CASE_POSTS = os.path.join(BASE_DIR, "sentivent_case_study_posts.csv")
OUT_AUDIT = os.path.join(BASE_DIR, "sentivent_retrieval_audit_sample.csv")


def _read_csv_optional(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _quantile_flag(s: pd.Series, q: float, *, low: bool) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    thr = s.quantile(q)
    if low:
        return s <= thr
    return s >= thr


def assign_alignment_regime(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mutually exclusive regimes for qualitative case selection."""
    m = metrics.copy()
    m["tweet_count"] = pd.to_numeric(m["tweet_count"], errors="coerce")
    m["x_sentivent_similarity"] = pd.to_numeric(m["x_sentivent_similarity"], errors="coerce")
    m["x_coherence_tfidf"] = pd.to_numeric(m["x_coherence_tfidf"], errors="coerce")
    if "alignment_gap" not in m.columns:
        m["alignment_gap"] = m["x_coherence_tfidf"] - m["x_sentivent_similarity"]

    low_qual = _quantile_flag(m["tweet_count"], 0.33, low=True) | _quantile_flag(
        m["x_coherence_tfidf"], 0.25, low=True
    )
    med_sim = m["x_sentivent_similarity"].median()
    med_coh = m["x_coherence_tfidf"].median()
    med_gap = m["alignment_gap"].median()
    coherent_div = (
        (~low_qual)
        & (m["x_coherence_tfidf"] >= med_coh)
        & (m["x_sentivent_similarity"] < med_sim)
        & (m["alignment_gap"] >= med_gap)
    )
    high_align = (~low_qual) & _quantile_flag(m["x_sentivent_similarity"], 0.66, low=False)

    regime = np.where(low_qual, "low_quality_noisy", np.where(coherent_div, "coherent_divergent", ""))
    regime = np.where((regime == "") & high_align, "high_alignment", regime)
    regime = np.where(regime == "", "typical_mixed", regime)
    m["evaluation_regime"] = regime
    return m


def _median_representative_event(sub: pd.DataFrame, score_col: str = "x_sentivent_similarity") -> Optional[str]:
    if sub.empty:
        return None
    target = sub[score_col].median()
    idx = (sub[score_col] - target).abs().idxmin()
    return str(sub.loc[idx, "event"])


def pick_case_study_events(
    metrics_with_regime: pd.DataFrame, per_regime: int
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for regime in ["high_alignment", "coherent_divergent", "low_quality_noisy"]:
        sub = metrics_with_regime[metrics_with_regime["evaluation_regime"] == regime]
        if sub.empty:
            continue
        sub = sub.sort_values("x_sentivent_similarity", ascending=False)
        picks: List[str] = []
        ev = _median_representative_event(sub, "x_sentivent_similarity")
        if ev:
            picks.append(ev)
        for _, r in sub.iterrows():
            if len(picks) >= per_regime:
                break
            e = str(r["event"])
            if e not in picks:
                picks.append(e)
        for e in picks[:per_regime]:
            rows.append((regime, e))
    return rows


def stratified_audit_events(
    cleaned: pd.DataFrame, registry: pd.DataFrame, n_events: int, seed: int
) -> List[str]:
    ev_counts = cleaned.groupby("event", as_index=False).size().rename(columns={"size": "n_posts"})
    reg_sub = registry[["event", "event_type"]].drop_duplicates("event")
    m = ev_counts.merge(reg_sub, on="event", how="left")
    m["event_type"] = m["event_type"].fillna("unknown")
    rng = np.random.default_rng(seed)
    picks: List[str] = []
    for _, g in m.groupby("event_type", sort=True):
        if len(picks) >= n_events:
            break
        if g.empty:
            continue
        row = g.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        picks.append(str(row["event"].iloc[0]))
    picks = list(dict.fromkeys(picks))
    if len(picks) < n_events:
        rest = m[~m["event"].isin(picks)]
        k = min(n_events - len(picks), len(rest))
        if k > 0:
            extra = rest.sample(n=k, random_state=seed)
            picks.extend(extra["event"].astype(str).tolist())
    return picks[:n_events]


def build_audit_sample(
    cleaned: pd.DataFrame,
    registry: pd.DataFrame,
    posts_per_event: int,
    n_events: int,
    seed: int,
) -> pd.DataFrame:
    events = stratified_audit_events(cleaned, registry, n_events, seed)
    reg_lookup = registry.set_index("event")
    rows = []
    audit_id = 0
    for ev in events:
        sub = cleaned[cleaned["event"] == ev]
        if sub.empty:
            continue
        n = min(posts_per_event, len(sub))
        rs = (seed + zlib.adler32(ev.encode("utf-8"))) % (2**31 - 1)
        samp = sub.sample(n=n, random_state=rs)
        ev_label = ""
        ev_type = ""
        q_col = "query_used" if "query_used" in samp.columns else None
        if ev in reg_lookup.index:
            lbl = reg_lookup.loc[ev, "event_label"]
            ev_label = str(lbl.iloc[0]) if isinstance(lbl, pd.Series) else str(lbl)
            et = reg_lookup.loc[ev, "event_type"]
            ev_type = str(et.iloc[0]) if isinstance(et, pd.Series) else str(et)
        for _, r in samp.iterrows():
            audit_id += 1
            rows.append(
                {
                    "audit_row_id": audit_id,
                    "event": ev,
                    "event_label": ev_label,
                    "event_type": ev_type,
                    "x_query_used": str(r[q_col]) if q_col else "",
                    "tweet_id": r.get("id", ""),
                    "author": r.get("author", ""),
                    "engagement_score": r.get("score", ""),
                    "post_text": str(r.get("clean_body", r.get("body", ""))),
                    "relevance_label": "",
                    "relevance_notes": "",
                }
            )
    return pd.DataFrame(rows)


def load_sentivent_excerpts(tsv_path: str, max_chars: int = 2500) -> Dict[str, str]:
    if not os.path.exists(tsv_path):
        return {}
    df = pd.read_csv(tsv_path, sep="\t")
    full = build_sentivent_doc_texts(df)
    return {k: (v[:max_chars] + ("…" if len(v) > max_chars else "")) for k, v in full.items()}


def export_case_study_tables(
    metrics: pd.DataFrame,
    cleaned: pd.DataFrame,
    registry: pd.DataFrame,
    sentivent_excerpts: Dict[str, str],
    per_regime: int,
) -> None:
    m = assign_alignment_regime(metrics)
    pairs = pick_case_study_events(m, per_regime)
    reg_idx = registry.set_index("event")
    event_rows = []
    for regime, ev in pairs:
        row = m[m["event"] == ev].iloc[0]
        doc_id = extract_doc_id(ev)
        excerpt = sentivent_excerpts.get(doc_id, "")
        if ev in reg_idx.index:
            lbl = reg_idx.loc[ev, "event_label"]
            ev_label = str(lbl.iloc[0]) if isinstance(lbl, pd.Series) else str(lbl)
            et = reg_idx.loc[ev, "event_type"]
            ev_type = str(et.iloc[0]) if isinstance(et, pd.Series) else str(et)
        else:
            ev_label, ev_type = "", ""
        gap = row.get("alignment_gap", row["x_coherence_tfidf"] - row["x_sentivent_similarity"])
        event_rows.append(
            {
                "evaluation_regime": regime,
                "event": ev,
                "event_label": ev_label,
                "event_type": ev_type,
                "tweet_count_metrics": int(row["tweet_count"]) if pd.notna(row["tweet_count"]) else "",
                "x_sentivent_similarity": row["x_sentivent_similarity"],
                "x_coherence_tfidf": row["x_coherence_tfidf"],
                "alignment_gap": gap,
                "sentivent_reference_excerpt": excerpt,
            }
        )
    pd.DataFrame(event_rows).to_csv(OUT_CASE_EVENTS, index=False)

    post_rows = []
    for regime, ev in pairs:
        sub = cleaned[cleaned["event"] == ev]
        doc_id = extract_doc_id(ev)
        excerpt = sentivent_excerpts.get(doc_id, "")
        for _, r in sub.iterrows():
            post_rows.append(
                {
                    "evaluation_regime": regime,
                    "event": ev,
                    "tweet_id": r.get("id", ""),
                    "author": r.get("author", ""),
                    "score": r.get("score", ""),
                    "record_type": r.get("record_type", ""),
                    "post_text": str(r.get("clean_body", "")),
                    "sentivent_reference_excerpt": excerpt,
                }
            )
    pd.DataFrame(post_rows).to_csv(OUT_CASE_POSTS, index=False)


def _load_or_build_inference_tables(
    args: argparse.Namespace,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], float, Optional[pd.DataFrame]]:
    coef_path = args.coefficients or DEFAULT_COEFS
    type_path = args.type_tests or DEFAULT_TYPE_TESTS
    feat_path = args.features or DEFAULT_FEATURES

    coef_df = _read_csv_optional(coef_path)
    type_df = _read_csv_optional(type_path)
    features = _read_csv_optional(feat_path)
    r2 = np.nan

    if features is not None and (coef_df is None or coef_df.empty):
        try:
            from analyze_sentivent_advanced import run_standardized_ols

            predictors = [
                "tweet_count",
                "score_median",
                "score_p90",
                "word_count_mean",
                "unique_authors",
                "post_share",
                "x_coherence_tfidf",
            ]
            predictors = [p for p in predictors if p in features.columns]
            coef_df, r2 = run_standardized_ols(features, "x_sentivent_similarity", predictors)
        except Exception:
            pass

    if features is not None and coef_df is not None and not coef_df.empty and np.isnan(r2):
        try:
            from analyze_sentivent_advanced import run_standardized_ols

            predictors = [c for c in coef_df["feature"] if c != "intercept" and c in features.columns]
            _, r2 = run_standardized_ols(features, "x_sentivent_similarity", predictors)
        except Exception:
            r2 = np.nan

    if features is not None and (type_df is None or type_df.empty):
        try:
            from analyze_sentivent_advanced import run_type_permutation_tests

            type_df = run_type_permutation_tests(features, min_n=5)
        except Exception:
            pass

    return coef_df, type_df, float(r2) if pd.notna(r2) else np.nan, features


def compute_llm_summaries(rel_path: str, agr_path: str) -> List[str]:
    lines: List[str] = []
    rel = _read_csv_optional(rel_path)
    agr = _read_csv_optional(agr_path)
    lines.append("## E. LLM robustness and model disagreement")
    lines.append("")
    if rel is not None and not rel.empty:
        feat_cols = ["frame_match", "emotion_match", "stance_match", "blame_match", "causal_match"]
        feat_cols = [c for c in feat_cols if c in rel.columns]
        sub = rel.copy()
        for c in feat_cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        by_model = sub.groupby("model", as_index=False)[feat_cols].mean()
        by_model["overall_match_rate"] = by_model[feat_cols].mean(axis=1)
        lines.append("### Per-model X↔SENTiVENT feature match")
        lines.append("")
        lines.append("| model | overall_match | frame | emotion | stance | blame | causal |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        by_model = by_model.sort_values("overall_match_rate", ascending=False)
        for _, r in by_model.iterrows():
            lines.append(
                f"| `{r['model']}` | {r['overall_match_rate']:.3f} | "
                + " | ".join(f"{r[c]:.3f}" for c in feat_cols)
                + " |"
            )
        lines.append("")
    else:
        lines.append("_No `sentivent_llm_relationship_by_model.csv` found. Run `llm_sentivent_analysis.py`._")
        lines.append("")

    if agr is not None and not agr.empty and "overall_pairwise_kappa_mean" in agr.columns:
        sub = agr.dropna(subset=["overall_pairwise_kappa_mean"])
        lines.append("### Cross-model agreement (mean pairwise κ by source)")
        lines.append("")
        if not sub.empty:
            for src, g in sub.groupby("source"):
                mu = float(g["overall_pairwise_kappa_mean"].mean())
                lines.append(f"- **{src}**: mean κ across event-rows ≈ **{mu:.3f}** (n={len(g)})")
        lines.append("")
    return lines


def summarize_filled_audit(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "relevance_label" not in df.columns:
        return None
    labels = df["relevance_label"].fillna("").astype(str).str.strip()
    coded = labels[labels != ""]
    if coded.empty:
        return None
    norm = coded.str.lower()
    relevant = int((norm == "relevant").sum())
    partial = int(((norm == "partial") | (norm == "partially_relevant")).sum())
    irr = int((norm == "irrelevant").sum())
    n = len(coded)
    strict_prec = relevant / n if n else np.nan
    soft = (relevant + 0.5 * partial) / n if n else np.nan
    lines = [
        "## Filled retrieval audit summary",
        "",
        f"- Coded posts: **{n}**",
        f"- Strict precision (relevant / coded): **{strict_prec:.3f}**",
        f"- Weighted proxy (relevant + ½·partial over coded): **{soft:.3f}**",
        f"- Counts: relevant={relevant}, partial={partial}, irrelevant={irr}",
        "",
    ]
    return "\n".join(lines)


def build_markdown_report(args: argparse.Namespace) -> str:
    raw = _read_csv_optional(args.raw_master)
    cleaned_path = args.x_clean
    cleaned = _read_csv_optional(cleaned_path)
    event_counts = _read_csv_optional(args.event_counts)
    registry = _read_csv_optional(args.registry)
    sem = _read_csv_optional(args.semantic_metrics)

    lines: List[str] = [
        "# SENTiVENT evaluation report",
        "",
        "Quantitative summary (coverage, semantic alignment, inference, LLM robustness) and pointers to qualitative exports.",
        "",
        "## A. Data quality and representativeness",
        "",
    ]

    raw_n = len(raw) if raw is not None else None
    clean_n = len(cleaned) if cleaned is not None else None
    if raw_n is not None:
        lines.append(f"- Raw scraped posts (master): **{raw_n:,}**")
    else:
        lines.append(f"- Raw scraped posts: _missing `{os.path.basename(args.raw_master)}`_")

    if clean_n is not None:
        lines.append(f"- Cleaned / analysis-ready rows: **{clean_n:,}**")
    else:
        lines.append(f"- Cleaned rows: _missing `{os.path.basename(cleaned_path)}`_")

    if event_counts is not None and "event" in event_counts.columns:
        lines.append(f"- Distinct events with ≥1 cleaned post (counts file): **{len(event_counts)}**")
        if "tweet_count" in event_counts.columns:
            lines.append(f"- Total posts summed in counts file: **{int(event_counts['tweet_count'].sum()):,}**")
    elif cleaned is not None and "event" in cleaned.columns:
        ne = cleaned["event"].nunique()
        lines.append(f"- Distinct events in cleaned data: **{ne}**")
    else:
        lines.append("- Event coverage: _insufficient data_")

    if registry is not None and "event_type" in registry.columns and cleaned is not None:
        reg_ev = registry[["event", "event_type"]].drop_duplicates()
        posted = cleaned["event"].dropna().unique()
        reg_used = reg_ev[reg_ev["event"].isin(posted)]
        vc = reg_used["event_type"].value_counts()
        lines.append("")
        lines.append("### Event-type distribution (events with ≥1 cleaned post)")
        lines.append("")
        for t, c in vc.items():
            lines.append(f"- `{t}`: **{int(c)}** events")
    lines.append("")

    lines.append("### Semantic cohort (events with usable alignment metrics)")
    lines.append("")
    if sem is None or sem.empty:
        lines.append(
            f"_Missing or empty `{os.path.basename(args.semantic_metrics)}`. Run `visualize_sentivent_semantics.py`._"
        )
    else:
        lines.append(f"- Events in semantic metrics table: **{len(sem)}**")
        lines.append(
            f"- Mean X↔SENTiVENT similarity (TF-IDF centroid cosine): **{sem['x_sentivent_similarity'].mean():.3f}**"
        )
        if "x_coherence_tfidf" in sem.columns:
            lines.append(f"- Mean X internal coherence: **{sem['x_coherence_tfidf'].mean():.3f}**")
        if "alignment_gap" in sem.columns:
            lines.append(f"- Mean alignment gap (coherence − cross-source sim): **{sem['alignment_gap'].mean():.3f}**")
    lines.append("")

    lines.append("## B. Semantic alignment (core metrics)")
    lines.append("")
    if sem is not None and not sem.empty:
        lines.append(
            "Cosine similarity between the X TF-IDF centroid and the SENTiVENT document vector; "
            "coherence = mean cosine similarity of posts to that centroid; gap = coherence − cross-source similarity."
        )
        lines.append("")
    lines.append("## C. Statistical inference (standardized OLS)")
    lines.append("")

    coef_df, type_df, r2, features = _load_or_build_inference_tables(args)
    if features is None or features.empty:
        lines.append("_No feature table. Run `analyze_sentivent_advanced.py` after semantics._")
    elif coef_df is None or coef_df.empty:
        lines.append("_Regression coefficients not available (insufficient rows or missing predictors)._")
    else:
        if not np.isnan(r2):
            lines.append(f"- R² (predicting `x_sentivent_similarity`): **{r2:.3f}**")
        lines.append("")
        lines.append("| feature | standardized β |")
        lines.append("| --- | --- |")
        for _, r in coef_df[coef_df["feature"] != "intercept"].iterrows():
            lines.append(f"| `{r['feature']}` | {r['beta_std']:.4f} |")
        pos = coef_df[coef_df["feature"] != "intercept"].nlargest(3, "beta_std")
        neg = coef_df[coef_df["feature"] != "intercept"].nsmallest(3, "beta_std")
        lines.append("")
        lines.append(
            "- Strongest positive standardized effects: "
            + ", ".join(f"`{r['feature']}` ({r['beta_std']:.3f})" for _, r in pos.iterrows())
        )
        lines.append(
            "- Strongest negative standardized effects: "
            + ", ".join(f"`{r['feature']}` ({r['beta_std']:.3f})" for _, r in neg.iterrows())
        )
    lines.append("")

    lines.append("## D. Event-type effects (permutation tests)")
    lines.append("")
    if type_df is None or type_df.empty:
        lines.append("_No permutation table. Run `analyze_sentivent_advanced.py`._")
    else:
        sig = type_df[type_df["p_value_perm"] < 0.05].sort_values("effect_type_minus_other", ascending=False)
        lines.append(f"- Types tested: **{len(type_df)}**")
        lines.append(f"- Significant at p < 0.05: **{len(sig)}**")
        lines.append("")
        lines.append("| event_type | effect (type − other) | p-value | n_type |")
        lines.append("| --- | --- | --- | --- |")
        for _, r in type_df.sort_values("p_value_perm").iterrows():
            star = "*" if r["p_value_perm"] < 0.05 else ""
            lines.append(
                f"| `{r['event_type']}` | {r['effect_type_minus_other']:.4f} | {r['p_value_perm']:.4f}{star} | {int(r['n_type'])} |"
            )
    lines.append("")

    lines.extend(compute_llm_summaries(args.llm_relationship, args.llm_agreement))

    lines.append("## Qualitative exports (this run)")
    lines.append("")
    lines.append(f"- Case-study event summary: `{os.path.basename(OUT_CASE_EVENTS)}`")
    lines.append(f"- Case-study posts: `{os.path.basename(OUT_CASE_POSTS)}`")
    lines.append(
        f"- Retrieval audit template ({args.audit_events} × {args.audit_posts_per_event}): `{os.path.basename(OUT_AUDIT)}`"
    )
    lines.append("")
    lines.append(
        "**Regime definitions:** `high_alignment` ≈ upper similarity tertile (non–low-quality); "
        "`coherent_divergent` = coherent X cluster, below-median cross-source similarity, above-median gap; "
        "`low_quality_noisy` = thin volume (≤33rd pct tweets) or low coherence (≤25th pct)."
    )
    lines.append("")

    audit_note = summarize_filled_audit(args.audit_input)
    if audit_note:
        lines.append(audit_note)

    lines.append("## Generated files checklist")
    lines.append("")
    lines.append(f"- This report: `{os.path.basename(OUT_REPORT)}`")
    lines.append("- Advanced summary: `sentivent_advanced_summary.md` (from Day-2 script, if present)")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SENTiVENT quantitative + qualitative evaluation pack.")
    p.add_argument("--raw-master", default=DEFAULT_RAW_MASTER, help="Raw scrape CSV before cleaning.")
    p.add_argument("--x-clean", default=DEFAULT_X_CLEAN, help="Cleaned X CSV.")
    p.add_argument("--event-counts", default=DEFAULT_EVENT_COUNTS, help="Per-event tweet counts after cleaning.")
    p.add_argument("--registry", default=DEFAULT_REGISTRY, help="SENTiVENT event registry.")
    p.add_argument("--semantic-metrics", default=DEFAULT_SEM_METRICS, help="Event-level semantic metrics CSV.")
    p.add_argument("--features", default=DEFAULT_FEATURES, help="Advanced feature table (optional).")
    p.add_argument("--coefficients", default=DEFAULT_COEFS, help="Regression coefficients CSV (optional).")
    p.add_argument("--type-tests", default=DEFAULT_TYPE_TESTS, help="Permutation tests CSV (optional).")
    p.add_argument("--sentivent-tsv", default=DEFAULT_SENTIVENT_TSV, help="SENTiVENT TSV for reference excerpts.")
    p.add_argument("--llm-relationship", default=DEFAULT_LLM_REL, help="LLM match table path.")
    p.add_argument("--llm-agreement", default=DEFAULT_LLM_AGR, help="LLM kappa table path.")
    p.add_argument("--out-report", default=OUT_REPORT, help="Output markdown path.")
    p.add_argument("--case-events-per-regime", type=int, default=3, help="Events per alignment regime for case exports.")
    p.add_argument("--audit-events", type=int, default=10, help="Events in retrieval audit sample.")
    p.add_argument("--audit-posts-per-event", type=int, default=20, help="Posts per audit event.")
    p.add_argument("--audit-seed", type=int, default=42, help="RNG seed for audit stratification.")
    p.add_argument("--audit-output", default=OUT_AUDIT, help="Retrieval audit CSV path.")
    p.add_argument(
        "--audit-input",
        default=OUT_AUDIT,
        help="Audit CSV path; if relevance_label is filled, summary stats are appended to the report.",
    )
    p.add_argument("--skip-case-and-audit", action="store_true", help="Only write markdown report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global OUT_REPORT, OUT_AUDIT
    OUT_REPORT = args.out_report
    OUT_AUDIT = args.audit_output

    sem = _read_csv_optional(args.semantic_metrics)
    cleaned = _read_csv_optional(args.x_clean)
    registry = _read_csv_optional(args.registry)

    if not args.skip_case_and_audit and sem is not None and cleaned is not None and registry is not None:
        excerpts = load_sentivent_excerpts(args.sentivent_tsv)
        export_case_study_tables(sem, cleaned, registry, excerpts, args.case_events_per_regime)
        audit_df = build_audit_sample(
            cleaned,
            registry,
            posts_per_event=args.audit_posts_per_event,
            n_events=args.audit_events,
            seed=args.audit_seed,
        )
        audit_df.to_csv(OUT_AUDIT, index=False)
        print(f"Wrote case study events: {OUT_CASE_EVENTS}")
        print(f"Wrote case study posts: {OUT_CASE_POSTS}")
        print(f"Wrote retrieval audit template: {OUT_AUDIT}")
    elif not args.skip_case_and_audit:
        print("Skipping case study / audit exports (need semantic metrics, cleaned X, and registry).")

    text = build_markdown_report(args)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote evaluation report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
