import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CLEAN = os.path.join(BASE_DIR, "x_sentivent_cleaned.csv")
INPUT_REGISTRY = os.path.join(BASE_DIR, "event_registry_sentivent.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_sentivent")


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


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(INPUT_CLEAN):
        raise FileNotFoundError(f"Missing cleaned SENTiVENT X data: {INPUT_CLEAN}")
    if not os.path.exists(INPUT_REGISTRY):
        raise FileNotFoundError(f"Missing SENTiVENT event registry: {INPUT_REGISTRY}")

    posts = pd.read_csv(INPUT_CLEAN)
    reg = pd.read_csv(INPUT_REGISTRY)

    keep_reg_cols = ["event", "event_label", "event_type"]
    keep_reg_cols = [c for c in keep_reg_cols if c in reg.columns]
    reg = reg[keep_reg_cols].drop_duplicates(subset=["event"])

    merged = posts.merge(reg, on="event", how="left")
    if "event_label" not in merged.columns:
        merged["event_label"] = merged["event"]
    if "event_type" not in merged.columns:
        merged["event_type"] = "unknown"

    merged["score"] = pd.to_numeric(merged.get("score", 0), errors="coerce").fillna(0)
    merged["word_count"] = merged["clean_body"].fillna("").astype(str).str.split().str.len()
    return posts, reg, merged


def plot_top_events_volume(merged: pd.DataFrame, top_n: int = 20) -> None:
    counts = (
        merged.groupby(["event", "event_label"], as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
        .head(top_n)
    )
    if counts.empty:
        return

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=counts, y="event_label", x="tweet_count", color="#1D3557")
    ax.set_title("SENTiVENT Events: Top Events by X Post Volume")
    ax.set_xlabel("Number of Posts")
    ax.set_ylabel("Event")

    for patch in ax.patches:
        w = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(w + 0.3, y, f"{int(w)}", va="center", ha="left", fontsize=7)

    save_fig(os.path.join(PLOTS_DIR, "sentivent_top_events_by_volume.png"))


def plot_event_type_coverage(merged: pd.DataFrame) -> None:
    counts = (
        merged.groupby("event_type", as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
    )
    if counts.empty:
        return

    plt.figure(figsize=(12, 6.5))
    ax = sns.barplot(data=counts, x="event_type", y="tweet_count", color="#457B9D")
    ax.set_title("SENTiVENT Event-Type Coverage in Collected X Posts")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Number of Posts")
    plt.xticks(rotation=30, ha="right")

    for patch in ax.patches:
        h = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + 1,
            f"{int(h)}",
            va="bottom",
            ha="center",
            fontsize=7,
        )

    save_fig(os.path.join(PLOTS_DIR, "sentivent_event_type_coverage.png"))


def plot_engagement_by_type(merged: pd.DataFrame) -> None:
    top_types = (
        merged.groupby("event_type", as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(10)["event_type"]
        .tolist()
    )
    sub = merged[merged["event_type"].isin(top_types)].copy()
    if sub.empty:
        return

    plt.figure(figsize=(12, 6.5))
    ax = sns.boxplot(
        data=sub,
        x="event_type",
        y="score",
        color="#A8DADC",
        showfliers=False,
    )
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("Engagement Distribution by Event Type (Top 10 Types)")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Engagement Score (symlog scale)")
    plt.xticks(rotation=30, ha="right")

    save_fig(os.path.join(PLOTS_DIR, "sentivent_engagement_by_type.png"))


def plot_text_length_distribution(merged: pd.DataFrame) -> None:
    sub = merged[merged["word_count"].notna()].copy()
    if sub.empty:
        return

    plt.figure(figsize=(10, 5.5))
    ax = sns.histplot(sub["word_count"], bins=40, color="#E76F51")
    ax.set_title("Distribution of Post Length in Cleaned SENTiVENT-X Corpus")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.axvline(sub["word_count"].median(), color="black", linestyle="--", linewidth=1, label="Median")
    ax.legend(loc="upper right")

    save_fig(os.path.join(PLOTS_DIR, "sentivent_post_length_distribution.png"))


def main() -> None:
    ensure_dir(PLOTS_DIR)
    configure_style()
    posts, reg, merged = load_data()

    plot_top_events_volume(merged)
    plot_event_type_coverage(merged)
    plot_engagement_by_type(merged)
    plot_text_length_distribution(merged)

    print(f"Generated SENTiVENT plots in: {PLOTS_DIR}")
    print(f"Input posts: {len(posts)} | events represented: {merged['event'].nunique()}")


if __name__ == "__main__":
    main()
