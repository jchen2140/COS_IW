import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_INPUT = os.path.join(BASE_DIR, "x_sentivent_cleaned.csv")
REG_INPUT = os.path.join(BASE_DIR, "event_registry_sentivent.csv")
SENTIVENT_INPUT = os.path.join(os.path.dirname(BASE_DIR), "dataset_event_subtype.tsv")
OUT_DIR = os.path.join(BASE_DIR, "plots_sentivent_semantics")
METRICS_OUT = os.path.join(BASE_DIR, "sentivent_semantic_metrics.csv")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def extract_doc_id(event_name: str) -> str:
    m = re.match(r"([a-z]+\d+)_", str(event_name).lower())
    return m.group(1) if m else ""


def build_sentivent_doc_texts(sentivent_df: pd.DataFrame) -> Dict[str, str]:
    sentivent_df = sentivent_df.copy()
    sentivent_df["text"] = sentivent_df["text"].fillna("").astype(str)
    doc_texts = (
        sentivent_df.groupby("document_id")["text"]
        .apply(lambda rows: " ".join([r for r in rows if r.strip()]))
        .to_dict()
    )
    return doc_texts


def mean_similarity_to_centroid(mat: np.ndarray, centroid: np.ndarray) -> float:
    if mat.shape[0] == 0:
        return np.nan
    sims = cosine_similarity(mat, centroid).flatten()
    return float(np.mean(sims))


def compute_event_metrics(x_df: pd.DataFrame, reg_df: pd.DataFrame, doc_texts: Dict[str, str]) -> pd.DataFrame:
    merged = x_df.merge(
        reg_df[["event", "event_label", "event_type"]].drop_duplicates("event"),
        on="event",
        how="left",
    )
    merged["event_label"] = merged["event_label"].fillna(merged["event"])
    merged["event_type"] = merged["event_type"].fillna("unknown")
    merged["doc_id"] = merged["event"].apply(extract_doc_id)
    merged["sentivent_text"] = merged["doc_id"].map(doc_texts).fillna("")

    # Keep only events that have both X posts and reference SENTiVENT text.
    event_rows: List[Dict] = []
    events = merged["event"].dropna().unique().tolist()
    for ev in events:
        sub = merged[merged["event"] == ev].copy()
        x_texts = sub["clean_body"].dropna().astype(str).tolist()
        sent_text = str(sub["sentivent_text"].iloc[0] if not sub.empty else "").strip()
        if len(x_texts) < 2 or len(sent_text) < 20:
            continue

        corpus = x_texts + [sent_text]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2), min_df=1)
        mat = vectorizer.fit_transform(corpus).toarray()
        x_mat = mat[:-1]
        s_vec = mat[-1:].reshape(1, -1)

        x_centroid = np.mean(x_mat, axis=0, keepdims=True)
        sim_x_to_sentivent = float(cosine_similarity(x_centroid, s_vec)[0, 0])
        x_coherence = mean_similarity_to_centroid(x_mat, x_centroid)
        s_self = 1.0  # single reference doc vector vs itself

        event_rows.append(
            {
                "event": ev,
                "event_label": str(sub["event_label"].iloc[0]),
                "event_type": str(sub["event_type"].iloc[0]),
                "tweet_count": int(len(x_texts)),
                "x_sentivent_similarity": sim_x_to_sentivent,
                "x_coherence_tfidf": x_coherence,
                "sentivent_self_similarity": s_self,
                "alignment_gap": float(x_coherence - sim_x_to_sentivent),
            }
        )

    metrics = pd.DataFrame(event_rows)
    if not metrics.empty:
        metrics = metrics.sort_values("x_sentivent_similarity", ascending=False).reset_index(drop=True)
    return metrics


def plot_top_bottom_similarity(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    top = metrics.head(12)
    bottom = metrics.tail(12)
    plot_df = pd.concat([top, bottom], axis=0).drop_duplicates("event")
    plot_df = plot_df.sort_values("x_sentivent_similarity", ascending=True)
    plot_df["Group"] = np.where(
        plot_df["x_sentivent_similarity"] >= metrics["x_sentivent_similarity"].median(),
        "Higher similarity",
        "Lower similarity",
    )

    plt.figure(figsize=(13, 8))
    ax = sns.barplot(
        data=plot_df,
        y="event_label",
        x="x_sentivent_similarity",
        hue="Group",
        dodge=False,
        palette={"Higher similarity": "#2A9D8F", "Lower similarity": "#E76F51"},
    )
    ax.set_title("Event-Level Semantic Similarity: X vs SENTiVENT Reference Text")
    ax.set_xlabel("Cosine Similarity (TF-IDF centroid)")
    ax.set_ylabel("Event")
    ax.legend(loc="lower right", frameon=True)

    for p in ax.patches:
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + 0.002, y, f"{w:.3f}", va="center", ha="left", fontsize=7)

    save_fig(os.path.join(OUT_DIR, "semantic_similarity_top_bottom_events.png"))


def plot_similarity_by_event_type(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return

    counts = metrics["event_type"].value_counts()
    keep_types = counts[counts >= 3].index.tolist()
    sub = metrics[metrics["event_type"].isin(keep_types)].copy()
    if sub.empty:
        return

    plt.figure(figsize=(13, 7))
    ax = sns.boxplot(data=sub, x="event_type", y="x_sentivent_similarity", color="#A8DADC", showfliers=False)
    sns.stripplot(data=sub, x="event_type", y="x_sentivent_similarity", color="#1D3557", size=3, alpha=0.6, ax=ax)
    ax.set_title("Semantic Alignment Distribution by SENTiVENT Event Type")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Cosine Similarity (X centroid vs SENTiVENT doc)")
    plt.xticks(rotation=30, ha="right")

    save_fig(os.path.join(OUT_DIR, "semantic_similarity_by_event_type.png"))


def plot_coherence_vs_alignment(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return

    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=metrics,
        x="x_coherence_tfidf",
        y="x_sentivent_similarity",
        size="tweet_count",
        hue="event_type",
        palette="tab20",
        sizes=(20, 220),
        alpha=0.8,
        linewidth=0.4,
    )
    ax.set_title("Relationship Between Internal X Coherence and X↔SENTiVENT Alignment")
    ax.set_xlabel("X internal coherence (TF-IDF)")
    ax.set_ylabel("X to SENTiVENT semantic similarity")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    save_fig(os.path.join(OUT_DIR, "coherence_vs_alignment_scatter.png"))


def plot_type_level_heatmap(metrics: pd.DataFrame, x_df: pd.DataFrame, reg_df: pd.DataFrame, doc_texts: Dict[str, str]) -> None:
    if metrics.empty:
        return
    type_counts = metrics["event_type"].value_counts()
    types = type_counts[type_counts >= 3].index.tolist()
    if len(types) < 3:
        return

    merged = x_df.merge(reg_df[["event", "event_type"]].drop_duplicates("event"), on="event", how="left")
    merged["event_type"] = merged["event_type"].fillna("unknown")
    merged["doc_id"] = merged["event"].apply(extract_doc_id)
    merged["sentivent_text"] = merged["doc_id"].map(doc_texts).fillna("")

    x_type_text = (
        merged[merged["event_type"].isin(types)]
        .groupby("event_type")["clean_body"]
        .apply(lambda rows: " ".join(rows.astype(str).tolist()))
        .to_dict()
    )
    s_type_text = (
        merged[merged["event_type"].isin(types)]
        .drop_duplicates(subset=["event"])[["event_type", "sentivent_text"]]
        .groupby("event_type")["sentivent_text"]
        .apply(lambda rows: " ".join([r for r in rows.astype(str).tolist() if r.strip()]))
        .to_dict()
    )

    ordered = [t for t in types if t in x_type_text and t in s_type_text]
    if len(ordered) < 3:
        return

    corpus = [x_type_text[t] for t in ordered] + [s_type_text[t] for t in ordered]
    vec = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2), min_df=2)
    mat = vec.fit_transform(corpus).toarray()
    x_mat = mat[: len(ordered)]
    s_mat = mat[len(ordered):]
    sim = cosine_similarity(x_mat, s_mat)

    heat_df = pd.DataFrame(sim, index=[f"X:{t}" for t in ordered], columns=[f"S:{t}" for t in ordered])

    plt.figure(figsize=(11, 8))
    ax = sns.heatmap(heat_df, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.4, cbar_kws={"label": "Cosine similarity"})
    ax.set_title("Type-Level Semantic Relationship Matrix (X vs SENTiVENT)")
    ax.set_xlabel("SENTiVENT event-type text centroid")
    ax.set_ylabel("X event-type text centroid")

    save_fig(os.path.join(OUT_DIR, "semantic_type_relationship_heatmap.png"))


def main() -> None:
    for p in [X_INPUT, REG_INPUT, SENTIVENT_INPUT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input: {p}")

    ensure_dir(OUT_DIR)
    configure_style()

    x_df = pd.read_csv(X_INPUT)
    reg_df = pd.read_csv(REG_INPUT)
    sentivent_df = pd.read_csv(SENTIVENT_INPUT, sep="\t")
    doc_texts = build_sentivent_doc_texts(sentivent_df)

    metrics = compute_event_metrics(x_df, reg_df, doc_texts)
    if metrics.empty:
        raise ValueError("No comparable events found between X corpus and SENTiVENT docs.")

    metrics.to_csv(METRICS_OUT, index=False)
    plot_top_bottom_similarity(metrics)
    plot_similarity_by_event_type(metrics)
    plot_coherence_vs_alignment(metrics)
    plot_type_level_heatmap(metrics, x_df, reg_df, doc_texts)

    print(f"Generated semantic plots in: {OUT_DIR}")
    print(f"Saved event-level metrics to: {METRICS_OUT}")
    print(f"Events analyzed: {len(metrics)}")


if __name__ == "__main__":
    main()
