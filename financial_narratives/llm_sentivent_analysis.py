import os
import re
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from x_api_config import load_local_env

from agreement_analysis import summarize_multi_model_agreement
from llm_feature_extraction import aggregate_features, batch_extract
from multi_model_extraction import get_available_model_specs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_INPUT = os.path.join(BASE_DIR, "x_sentivent_cleaned.csv")
REG_INPUT = os.path.join(BASE_DIR, "event_registry_sentivent.csv")
SENTIVENT_INPUT = os.path.join(os.path.dirname(BASE_DIR), "dataset_event_subtype.tsv")

OUT_DIR = os.path.join(BASE_DIR, "plots_sentivent_llm")
MODEL_OUT = os.path.join(BASE_DIR, "sentivent_llm_model_outputs.csv")
REL_OUT = os.path.join(BASE_DIR, "sentivent_llm_relationship_by_model.csv")
AGR_OUT = os.path.join(BASE_DIR, "sentivent_llm_cross_model_agreement.csv")


def configure_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
        },
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def split_into_segments(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", str(text))
    return [p.strip() for p in parts if len(p.strip()) > 20]


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


def safe_dom(agg: Dict, field: str) -> str:
    val = agg.get(f"{field}_dominant")
    return str(val) if val is not None else "unknown"


def run_analysis(max_events: int = 0, sample_size: int = 8, rate_limit_delay: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(X_INPUT):
        raise FileNotFoundError(f"Missing cleaned X data: {X_INPUT}")
    if not os.path.exists(REG_INPUT):
        raise FileNotFoundError(f"Missing SENTiVENT event registry: {REG_INPUT}")
    if not os.path.exists(SENTIVENT_INPUT):
        raise FileNotFoundError(f"Missing SENTiVENT TSV: {SENTIVENT_INPUT}")

    specs = get_available_model_specs()
    if not specs:
        raise RuntimeError("No available LLM providers configured. Set API keys first.")

    x_df = pd.read_csv(X_INPUT)
    reg_df = pd.read_csv(REG_INPUT)[["event", "event_label", "event_type"]].drop_duplicates("event")
    sentivent_df = pd.read_csv(SENTIVENT_INPUT, sep="\t")
    doc_texts = build_sentivent_doc_texts(sentivent_df)

    merged = x_df.merge(reg_df, on="event", how="left")
    merged["event_label"] = merged["event_label"].fillna(merged["event"])
    merged["event_type"] = merged["event_type"].fillna("unknown")
    merged["doc_id"] = merged["event"].apply(extract_doc_id)
    merged["sentivent_text"] = merged["doc_id"].map(doc_texts).fillna("")

    # choose events with enough data in both sources
    event_order = (
        merged.groupby("event", as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
    )
    all_event_ids = event_order["event"].tolist()
    # max_events <= 0 means use every event in the ranking (full corpus run).
    if max_events is not None and max_events > 0:
        selected_events = all_event_ids[:max_events]
    else:
        selected_events = all_event_ids

    print(
        f"LLM analysis: {len(selected_events)} event(s) selected "
        f"({'full ranked list' if not max_events or max_events <= 0 else f'cap={max_events}'})."
    )
    if len(selected_events) > 15:
        print(
            "  NOTE: Large run — many batched LLM calls per event × models. Check API quota, cost, and time."
        )

    model_rows = []
    rel_rows = []
    agr_rows = []

    for ev_idx, event in enumerate(selected_events, start=1):
        if ev_idx == 1 or ev_idx % 5 == 0 or ev_idx == len(selected_events):
            print(f"  Processing event {ev_idx}/{len(selected_events)}: {str(event)[:60]}...")
        sub = merged[merged["event"] == event].copy()
        if sub.empty:
            continue
        event_label = str(sub["event_label"].iloc[0])
        event_type = str(sub["event_type"].iloc[0])
        x_texts = sub["clean_body"].dropna().astype(str).tolist()
        sent_text = str(sub["sentivent_text"].iloc[0]).strip()
        s_texts = split_into_segments(sent_text)
        if len(x_texts) < 3 or len(s_texts) < 3:
            continue

        per_model_x = {}
        per_model_s = {}
        per_model_agg = {}

        for spec in specs:
            model_name = spec["name"]
            provider = spec["provider"]
            model = spec["model"]

            x_extractions = batch_extract(
                texts=x_texts,
                event_label=event_label,
                event_date="",
                provider=provider,
                model=model,
                sample_size=min(sample_size, len(x_texts)),
                rate_limit_delay=rate_limit_delay,
            )
            s_extractions = batch_extract(
                texts=s_texts,
                event_label=event_label,
                event_date="",
                provider=provider,
                model=model,
                sample_size=min(sample_size, len(s_texts)),
                rate_limit_delay=rate_limit_delay,
            )
            per_model_x[model_name] = x_extractions
            per_model_s[model_name] = s_extractions

            agg_x = aggregate_features(x_extractions)
            agg_s = aggregate_features(s_extractions)
            per_model_agg[model_name] = {"x": agg_x, "s": agg_s}

            model_rows.append(
                {
                    "event": event,
                    "event_label": event_label,
                    "event_type": event_type,
                    "model": model_name,
                    "source": "X",
                    "n_extracted": agg_x.get("n_extracted", 0),
                    "blame_dominant": safe_dom(agg_x, "blame_target"),
                    "causal_dominant": safe_dom(agg_x, "causal_mechanism"),
                    "stance_dominant": safe_dom(agg_x, "policy_stance"),
                    "emotion_dominant": safe_dom(agg_x, "emotional_register"),
                    "frame_dominant": safe_dom(agg_x, "narrative_frame"),
                }
            )
            model_rows.append(
                {
                    "event": event,
                    "event_label": event_label,
                    "event_type": event_type,
                    "model": model_name,
                    "source": "SENTIVENT",
                    "n_extracted": agg_s.get("n_extracted", 0),
                    "blame_dominant": safe_dom(agg_s, "blame_target"),
                    "causal_dominant": safe_dom(agg_s, "causal_mechanism"),
                    "stance_dominant": safe_dom(agg_s, "policy_stance"),
                    "emotion_dominant": safe_dom(agg_s, "emotional_register"),
                    "frame_dominant": safe_dom(agg_s, "narrative_frame"),
                }
            )

        # per-event cross-model agreement for each source
        ag_x = summarize_multi_model_agreement(per_model_x)
        ag_s = summarize_multi_model_agreement(per_model_s)
        agr_rows.append(
            {
                "event": event,
                "event_label": event_label,
                "event_type": event_type,
                "source": "X",
                "overall_pairwise_kappa_mean": ag_x.get("overall_pairwise_kappa_mean"),
            }
        )
        agr_rows.append(
            {
                "event": event,
                "event_label": event_label,
                "event_type": event_type,
                "source": "SENTIVENT",
                "overall_pairwise_kappa_mean": ag_s.get("overall_pairwise_kappa_mean"),
            }
        )

        # within-model relationship interpretation (X vs SENTIVENT)
        for model_name, payload in per_model_agg.items():
            agg_x = payload["x"]
            agg_s = payload["s"]
            rel_rows.append(
                {
                    "event": event,
                    "event_label": event_label,
                    "event_type": event_type,
                    "model": model_name,
                    "frame_match": int(safe_dom(agg_x, "narrative_frame") == safe_dom(agg_s, "narrative_frame")),
                    "emotion_match": int(safe_dom(agg_x, "emotional_register") == safe_dom(agg_s, "emotional_register")),
                    "stance_match": int(safe_dom(agg_x, "policy_stance") == safe_dom(agg_s, "policy_stance")),
                    "blame_match": int(safe_dom(agg_x, "blame_target") == safe_dom(agg_s, "blame_target")),
                    "causal_match": int(safe_dom(agg_x, "causal_mechanism") == safe_dom(agg_s, "causal_mechanism")),
                }
            )

    return pd.DataFrame(model_rows), pd.DataFrame(rel_rows), pd.DataFrame(agr_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM comparison analysis for X vs SENTiVENT event texts.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Event cap after ranking by tweet volume (0 = all events; use e.g. 12 for a quick pilot).",
    )
    parser.add_argument("--sample-size", type=int, default=8, help="Texts sampled per source per event.")
    parser.add_argument("--rate-limit-delay", type=float, default=0.3, help="Delay between LLM API calls.")
    return parser.parse_args()


def plot_agreement_by_source(agr_df: pd.DataFrame) -> None:
    if agr_df.empty:
        return
    sub = agr_df.dropna(subset=["overall_pairwise_kappa_mean"]).copy()
    if sub.empty:
        return

    plt.figure(figsize=(8, 5.5))
    ax = sns.boxplot(data=sub, x="source", y="overall_pairwise_kappa_mean", palette={"X": "#E76F51", "SENTIVENT": "#2A9D8F"})
    sns.stripplot(data=sub, x="source", y="overall_pairwise_kappa_mean", color="#1D3557", size=3, alpha=0.7, ax=ax)
    ax.set_title("Cross-Model Agreement by Source")
    ax.set_xlabel("Source")
    ax.set_ylabel("Mean pairwise kappa")
    save_fig(os.path.join(OUT_DIR, "llm_agreement_by_source.png"))


def plot_model_match_heatmap(rel_df: pd.DataFrame) -> None:
    if rel_df.empty:
        return
    feat_cols = ["frame_match", "emotion_match", "stance_match", "blame_match", "causal_match"]
    sub = rel_df.groupby("model", as_index=False)[feat_cols].mean()
    if sub.empty:
        return
    heat = sub.set_index("model")
    heat.columns = ["Frame", "Emotion", "Stance", "Blame", "Causal"]

    plt.figure(figsize=(8.5, 4.5))
    ax = sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Match rate"})
    ax.set_title("Per-Model X↔SENTIVENT Interpretation Match Rate")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Model")
    save_fig(os.path.join(OUT_DIR, "llm_model_match_heatmap.png"))


def plot_model_match_bars(rel_df: pd.DataFrame) -> None:
    if rel_df.empty:
        return
    feat_cols = ["frame_match", "emotion_match", "stance_match", "blame_match", "causal_match"]
    sub = rel_df.groupby("model", as_index=False)[feat_cols].mean()
    if sub.empty:
        return
    sub["overall_match_rate"] = sub[feat_cols].mean(axis=1)
    sub = sub.sort_values("overall_match_rate", ascending=False)

    plt.figure(figsize=(9.5, 5.5))
    ax = sns.barplot(data=sub, x="model", y="overall_match_rate", color="#457B9D")
    ax.set_title("Overall LLM Match Rate: X Interpretation vs SENTIVENT Interpretation")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean match rate across features")
    plt.xticks(rotation=20, ha="right")
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    save_fig(os.path.join(OUT_DIR, "llm_model_overall_match_rate.png"))


def main() -> None:
    args = parse_args()
    load_local_env()
    ensure_dir(OUT_DIR)
    configure_style()

    max_ev = int(args.max_events)
    model_df, rel_df, agr_df = run_analysis(
        max_events=max_ev,
        sample_size=max(1, int(args.sample_size)),
        rate_limit_delay=max(0.0, float(args.rate_limit_delay)),
    )
    if model_df.empty:
        raise ValueError("No LLM outputs produced. Check API keys and available events.")

    model_df.to_csv(MODEL_OUT, index=False)
    rel_df.to_csv(REL_OUT, index=False)
    agr_df.to_csv(AGR_OUT, index=False)

    plot_agreement_by_source(agr_df)
    plot_model_match_heatmap(rel_df)
    plot_model_match_bars(rel_df)

    print(f"Saved model outputs: {MODEL_OUT}")
    print(f"Saved relationship table: {REL_OUT}")
    print(f"Saved agreement table: {AGR_OUT}")
    print(f"Generated LLM comparison plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
