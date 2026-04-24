"""
Structured causal/blame extraction across three LLMs + Fleiss' κ (X vs SENTiVENT).

Implements the "ambiguity of blame" design: same closed task for professional news text
and for social text; lower inter-model agreement on X supports higher narrative ambiguity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from x_api_config import load_local_env

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

from agreement_analysis import summarize_multi_model_agreement
from multi_model_extraction import get_available_model_specs
from sentivent_evaluation_report import assign_alignment_regime, build_sentivent_doc_texts, extract_doc_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".llm_cache")

X_INPUT = os.path.join(BASE_DIR, "x_sentivent_cleaned.csv")
REG_INPUT = os.path.join(BASE_DIR, "event_registry_sentivent.csv")
SENTIVENT_TSV = os.path.join(os.path.dirname(BASE_DIR), "dataset_event_subtype.tsv")
SEM_METRICS = os.path.join(BASE_DIR, "sentivent_semantic_metrics.csv")

OUT_LONG = os.path.join(BASE_DIR, "sentivent_causal_ambiguity_long.csv")
OUT_FLEISS = os.path.join(BASE_DIR, "sentivent_causal_ambiguity_fleiss_summary.csv")
OUT_REGIME = os.path.join(BASE_DIR, "sentivent_causal_ambiguity_by_regime.csv")
OUT_SUMMARY_MD = os.path.join(BASE_DIR, "sentivent_causal_ambiguity_summary.md")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_sentivent_causal")

PROMPT_VERSION = "causal_v1"

PRIMARY_CAUSES = {
    "macroeconomic",
    "leadership",
    "product",
    "competitor",
    "regulatory",
    "market_structure",
    "unknown",
}

BLAME_TARGETS = {
    "executive_team",
    "government",
    "specific_company",
    "market",
    "retail_investors",
    "media",
    "no_blame",
    "unknown",
}

CAUSAL_PROMPT = """You are a financial discourse analyst. Read the TEXT and respond with ONLY a JSON object (no markdown fences).

EVENT: {event_label}

TEXT:
\"\"\"
{text}
\"\"\"

Return JSON with exactly these keys:
{{
  "primary_cause": "one of: macroeconomic, leadership, product, competitor, regulatory, market_structure, unknown",
  "blame_target": "one of: executive_team, government, specific_company, market, retail_investors, media, no_blame, unknown",
  "causal_confidence": <float between 0.0 and 1.0>
}}

Use "unknown" if the text does not support a clearer label. Use lowercase snake_case values exactly as listed.
"""


def _cache_key(text: str, event_label: str, provider: str, model: str) -> str:
    h = hashlib.md5(
        f"{PROMPT_VERSION}::{provider}::{model}::{event_label}::{text[:8000]}".encode()
    ).hexdigest()
    return h


def _load_cache(key: str) -> Optional[Dict]:
    path = os.path.join(CACHE_DIR, f"causal_{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: Dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"causal_{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _normalize_cat(val: str, allowed: set, default: str = "unknown") -> str:
    if val is None:
        return default
    s = re.sub(r"\s+", "_", str(val).strip().lower())
    s = s.replace("-", "_")
    if s in allowed:
        return s
    # light synonyms
    syn = {
        "macro": "macroeconomic",
        "fed": "macroeconomic",
        "ceo": "executive_team",
        "management": "executive_team",
        "company": "specific_company",
        "firms": "specific_company",
        "retail": "retail_investors",
        "none": "no_blame",
        "n/a": "unknown",
    }
    return syn.get(s, default if s not in allowed else s)


def _validate_causal(d: Dict) -> Dict:
    out = {}
    out["primary_cause"] = _normalize_cat(d.get("primary_cause"), PRIMARY_CAUSES)
    out["blame_target"] = _normalize_cat(d.get("blame_target"), BLAME_TARGETS)
    try:
        c = float(d.get("causal_confidence", 0.5))
    except (TypeError, ValueError):
        c = 0.5
    out["causal_confidence"] = float(min(1.0, max(0.0, c)))
    return out


def _strip_json_fences(content: str) -> str:
    content = (content or "").strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```", 1)[1].split("```", 1)[0].strip()
    return content


def extract_causal_openai(text: str, event_label: str, model: str) -> Optional[Dict]:
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    prompt = CAUSAL_PROMPT.format(event_label=event_label, text=text[:12000])
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        content = r.choices[0].message.content
        return _validate_causal(json.loads(content))
    except Exception as e:
        print(f"    [OpenAI causal] {e}")
        return None


def extract_causal_anthropic(text: str, event_label: str, model: str) -> Optional[Dict]:
    try:
        import anthropic
    except ImportError:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    client = anthropic.Anthropic(api_key=api_key)
    prompt = CAUSAL_PROMPT.format(event_label=event_label, text=text[:12000])
    try:
        r = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = r.content[0].text
        content = _strip_json_fences(content)
        return _validate_causal(json.loads(content))
    except Exception as e:
        print(f"    [Anthropic causal] {e}")
        return None


def extract_causal_gemini(text: str, event_label: str, model: str) -> Optional[Dict]:
    try:
        import google.generativeai as genai
    except ImportError:
        return None
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    prompt = CAUSAL_PROMPT.format(event_label=event_label, text=text[:12000])
    try:
        r = client.generate_content(prompt)
        content = _strip_json_fences((r.text or "").strip())
        return _validate_causal(json.loads(content))
    except Exception as e:
        print(f"    [Gemini causal] {e}")
        return None


def extract_causal(
    text: str,
    event_label: str,
    provider: str,
    model: str,
    use_cache: bool = True,
) -> Optional[Dict]:
    key = _cache_key(text, event_label, provider, model)
    if use_cache:
        hit = _load_cache(key)
        if hit is not None:
            return hit
    if provider == "openai":
        out = extract_causal_openai(text, event_label, model)
    elif provider == "anthropic":
        out = extract_causal_anthropic(text, event_label, model)
    elif provider == "gemini":
        out = extract_causal_gemini(text, event_label, model)
    else:
        return None
    if out and use_cache:
        _save_cache(key, out)
    return out


def build_x_blob(x_df: pd.DataFrame, event: str, max_chars: int, max_posts: int) -> str:
    sub = x_df[x_df["event"] == event].copy()
    if sub.empty:
        return ""
    if "score" in sub.columns:
        sub["score"] = pd.to_numeric(sub["score"], errors="coerce").fillna(0)
        sub = sub.sort_values("score", ascending=False)
    sub = sub.head(max_posts)
    parts = []
    for t in sub["clean_body"].fillna("").astype(str):
        t = t.strip()
        if len(t) > 40:
            parts.append(t)
    blob = "\n\n".join(parts)
    return blob[:max_chars]


def run_pipeline(
    max_events: int = 12,
    max_posts_for_x: int = 40,
    max_chars: int = 12000,
    rate_limit_delay: float = 0.35,
) -> None:
    if not os.path.exists(X_INPUT):
        raise FileNotFoundError(X_INPUT)
    if not os.path.exists(REG_INPUT):
        raise FileNotFoundError(REG_INPUT)
    if not os.path.exists(SENTIVENT_TSV):
        raise FileNotFoundError(SENTIVENT_TSV)

    specs = get_available_model_specs()
    if not specs:
        raise RuntimeError("No LLM API keys configured.")

    x_df = pd.read_csv(X_INPUT)
    reg = pd.read_csv(REG_INPUT)
    sent_df = pd.read_csv(SENTIVENT_TSV, sep="\t")
    doc_texts = build_sentivent_doc_texts(sent_df)

    counts = x_df.groupby("event", as_index=False).size().rename(columns={"size": "n"})
    events_ranked = counts.sort_values("n", ascending=False)["event"].tolist()[:max_events]

    reg_idx = reg.set_index("event")
    long_rows: List[Dict] = []

    per_src: Dict[str, Dict[str, List[Optional[Dict]]]] = {
        "SENTiVENT": {s["name"]: [] for s in specs},
        "X": {s["name"]: [] for s in specs},
    }
    event_order: List[str] = []

    for ev in events_ranked:
        doc_id = extract_doc_id(ev)
        news = (doc_texts.get(doc_id) or "").strip()
        x_blob = build_x_blob(x_df, ev, max_chars=max_chars, max_posts=max_posts_for_x)
        if len(news) < 80 or len(x_blob) < 80:
            continue

        try:
            lbl = reg_idx.loc[ev, "event_label"]
            event_label = str(lbl.iloc[0]) if isinstance(lbl, pd.Series) else str(lbl)
        except Exception:
            event_label = ev

        event_order.append(ev)
        print(f"Event {len(event_order)}/{max_events}: {ev[:48]}...")

        for src_name, text in [("SENTiVENT", news[:max_chars]), ("X", x_blob)]:
            for spec in specs:
                time.sleep(rate_limit_delay)
                r = extract_causal(
                    text,
                    event_label,
                    provider=spec["provider"],
                    model=spec["model"],
                    use_cache=True,
                )
                per_src[src_name][spec["name"]].append(r)
                long_rows.append(
                    {
                        "event": ev,
                        "source": src_name,
                        "model": spec["name"],
                        "primary_cause": r.get("primary_cause") if r else None,
                        "blame_target": r.get("blame_target") if r else None,
                        "causal_confidence": r.get("causal_confidence") if r else None,
                    }
                )

    if not event_order:
        raise ValueError("No events with enough SENTiVENT and X text.")

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_LONG, index=False)

    # Fleiss + pairwise κ via existing helper (per source, aligned by event index)
    summary_rows = []
    for src in ("SENTiVENT", "X"):
        block = {k: v for k, v in per_src[src].items()}
        agr = summarize_multi_model_agreement(block, feature_fields=["primary_cause", "blame_target"])
        for feat, payload in agr.get("feature_agreement", {}).items():
            summary_rows.append(
                {
                    "source": src,
                    "feature": feat,
                    "fleiss_kappa": payload.get("fleiss_kappa"),
                    "pairwise_kappa_mean": payload.get("pairwise_kappa_mean"),
                    "n_complete_items": payload.get("n_complete_items"),
                }
            )
    fleiss_df = pd.DataFrame(summary_rows)
    fleiss_df.to_csv(OUT_FLEISS, index=False)

    # Confidence dispersion: std across models per event × source
    piv = long_df.pivot_table(
        index=["event", "source"],
        columns="model",
        values="causal_confidence",
        aggfunc="first",
    )
    stds = piv.std(axis=1, ddof=0).reset_index(name="std_confidence_across_models")
    s_news = stds[stds["source"] == "SENTiVENT"]["std_confidence_across_models"].dropna()
    s_x = stds[stds["source"] == "X"]["std_confidence_across_models"].dropna()
    mw_p = np.nan
    if scipy_stats is not None and len(s_news) >= 3 and len(s_x) >= 3:
        try:
            mw_p = float(scipy_stats.mannwhitneyu(s_x, s_news, alternative="two-sided").pvalue)
        except Exception:
            pass

    # Regime-stratified Fleiss (semantic metrics)
    regime_rows = []
    if os.path.exists(SEM_METRICS):
        m = pd.read_csv(SEM_METRICS)
        m = assign_alignment_regime(m)
        ev_to_reg = dict(zip(m["event"], m["evaluation_regime"]))
        for regime in sorted(set(ev_to_reg.values())):
            evs = [e for e in event_order if ev_to_reg.get(e) == regime]
            if len(evs) < 3:
                continue
            idx = [i for i, e in enumerate(event_order) if e in evs]
            for src in ("SENTiVENT", "X"):
                block = {name: [per_src[src][name][i] for i in idx] for name in per_src[src]}
                agr = summarize_multi_model_agreement(block, feature_fields=["primary_cause", "blame_target"])
                for feat, payload in agr.get("feature_agreement", {}).items():
                    regime_rows.append(
                        {
                            "evaluation_regime": regime,
                            "source": src,
                            "feature": feat,
                            "fleiss_kappa": payload.get("fleiss_kappa"),
                            "n_events": len(evs),
                        }
                    )
    pd.DataFrame(regime_rows).to_csv(OUT_REGIME, index=False)

    # Markdown summary for IW write-up
    lines = [
        "# Causal ambiguity (multi-LLM) summary",
        "",
        f"- Events analyzed: **{len(event_order)}**",
        f"- Models: **{', '.join(s['name'] for s in specs)}**",
        "",
        "## Fleiss' κ (higher = more agreement among models on the same text)",
        "",
    ]
    if not fleiss_df.empty:
        for src in ("SENTiVENT", "X"):
            sub = fleiss_df[fleiss_df["source"] == src]
            lines.append(f"### {src}")
            for _, r in sub.iterrows():
                fk = r["fleiss_kappa"]
                fk_s = f"{fk:.4f}" if fk == fk else "n/a"
                lines.append(f"- **{r['feature']}**: κ = {fk_s} (n_complete = {r['n_complete_items']})")
            lines.append("")

    lines.append("## Causal confidence dispersion (std across models, per event)")
    lines.append("")
    lines.append(
        f"- Mean std (SENTiVENT): **{float(s_news.mean()):.4f}**; mean std (X): **{float(s_x.mean()):.4f}**"
    )
    if mw_p == mw_p:
        lines.append(f"- Mann–Whitney U (X vs SENTiVENT std): **p = {mw_p:.4f}** (exploratory; not a causal claim)")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Long table: `{os.path.basename(OUT_LONG)}`")
    lines.append(f"- Fleiss summary: `{os.path.basename(OUT_FLEISS)}`")
    lines.append(f"- By semantic regime: `{os.path.basename(OUT_REGIME)}`")
    lines.append("")

    with open(OUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _try_plot_fleiss(fleiss_df)

    print(f"Saved: {OUT_LONG}")
    print(f"Saved: {OUT_FLEISS}")
    print(f"Saved: {OUT_REGIME}")
    print(f"Saved: {OUT_SUMMARY_MD}")


def _try_plot_fleiss(fleiss_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    if fleiss_df.empty:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sub = fleiss_df.dropna(subset=["fleiss_kappa"]).copy()
    if sub.empty:
        return
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=sub, x="feature", y="fleiss_kappa", hue="source", palette={"SENTiVENT": "#2A9D8F", "X": "#E76F51"})
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Fleiss κ: multi-model agreement on causal labels")
    plt.ylabel("Fleiss κ")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fleiss_causal_x_vs_sentivent.png"), dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal ambiguity extraction + Fleiss κ (SENTiVENT vs X).")
    p.add_argument("--max-events", type=int, default=12)
    p.add_argument("--max-posts-for-x", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=12000)
    p.add_argument("--rate-limit-delay", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()
    run_pipeline(
        max_events=max(1, args.max_events),
        max_posts_for_x=max(5, args.max_posts_for_x),
        max_chars=max(2000, args.max_chars),
        rate_limit_delay=max(0.0, args.rate_limit_delay),
    )


if __name__ == "__main__":
    main()
