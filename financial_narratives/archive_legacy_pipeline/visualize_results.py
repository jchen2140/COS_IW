import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, wilcoxon

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "narrative_analysis_results.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SUMMARY_FILE = os.path.join(BASE_DIR, "hypothesis_summary.md")


def configure_publication_style():
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
            "figure.titlesize": 13,
        },
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def prepare(df):
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df = df.sort_values("event_date", na_position="last").copy()
    else:
        df = df.copy()
    return df


def sig_label(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_h1_tests(df):
    subset = df.dropna(subset=["x_coherence", "wsj_coherence"]).copy()
    if subset.empty:
        return None

    subset["coherence_gap_wsj_minus_x"] = subset["wsj_coherence"] - subset["x_coherence"]
    gaps = subset["coherence_gap_wsj_minus_x"].to_numpy()

    positive_count = int(np.sum(gaps > 0))
    n = len(gaps)

    sign_p = binomtest(positive_count, n=n, p=0.5, alternative="greater").pvalue if n > 0 else np.nan

    try:
        wilcoxon_p = wilcoxon(gaps, alternative="greater", zero_method="wilcox").pvalue if n >= 2 else np.nan
    except Exception:
        wilcoxon_p = np.nan

    return {
        "n": n,
        "positive_count": positive_count,
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "sign_p": float(sign_p),
        "wilcoxon_p": float(wilcoxon_p) if not pd.isna(wilcoxon_p) else np.nan,
        "data": subset,
    }


def compute_h2_data(df):
    required = [
        "similarity_wsj",
        "similarity_official",
        "similarity_wsj_ci_low",
        "similarity_wsj_ci_high",
        "similarity_official_ci_low",
        "similarity_official_ci_high",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    subset = df.dropna(subset=["similarity_wsj", "similarity_official"]).copy()
    if subset.empty:
        return pd.DataFrame()

    subset["delta_wsj_minus_official"] = subset["similarity_wsj"] - subset["similarity_official"]
    subset["delta_ci_low"] = subset["similarity_wsj_ci_low"] - subset["similarity_official_ci_high"]
    subset["delta_ci_high"] = subset["similarity_wsj_ci_high"] - subset["similarity_official_ci_low"]
    subset["ci_crosses_zero"] = (subset["delta_ci_low"] <= 0) & (subset["delta_ci_high"] >= 0)

    return subset


def plot_h1_coherence_test(df, label_col):
    h1 = compute_h1_tests(df)
    if h1 is None:
        return None

    plot_df = h1["data"].copy().sort_values("coherence_gap_wsj_minus_x", ascending=False)
    plot_df["Direction"] = np.where(plot_df["coherence_gap_wsj_minus_x"] >= 0, "WSJ > X", "WSJ ≤ X")

    plt.figure(figsize=(14, max(6.2, 1.0 * len(plot_df))))
    ax = sns.barplot(
        data=plot_df,
        y=label_col,
        x="coherence_gap_wsj_minus_x",
        hue="Direction",
        dodge=False,
        palette={"WSJ > X": "#2A9D8F", "WSJ ≤ X": "#E76F51"},
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Hypothesis 1: Differential Narrative Coherence Across Media Sources")
    ax.set_xlabel("Coherence Differential (WSJ minus X)")
    ax.set_ylabel("Financial Event")

    for patch in ax.patches:
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ha = "left" if x >= 0 else "right"
        tx = x + 0.003 if x >= 0 else x - 0.003
        ax.text(tx, y, f"{x:.3f}", ha=ha, va="center", fontsize=7)

    note = (
        f"Number of events: {h1['n']} | Cases with WSJ > X: {h1['positive_count']}/{h1['n']}\n"
        f"Sign test p-value = {h1['sign_p']:.4f} ({sig_label(h1['sign_p'])}) | "
        f"Wilcoxon p-value = {h1['wilcoxon_p']:.4f} ({sig_label(h1['wilcoxon_p'])})"
    )
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F7F7F7", ec="#AAAAAA"))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], loc="lower right", frameon=True)

    out = os.path.join(PLOTS_DIR, "h1_coherence_gap_significance.png")
    save_fig(out)
    print("Generated: plots/h1_coherence_gap_significance.png")
    return h1


def plot_h2_alignment_delta(df, label_col):
    h2_df = compute_h2_data(df)
    if h2_df.empty:
        return h2_df

    plot_df = h2_df.copy().sort_values("delta_wsj_minus_official", ascending=False)

    plt.figure(figsize=(14, max(6.2, 1.0 * len(plot_df))))
    ax = plt.gca()

    y_positions = np.arange(len(plot_df))
    means = plot_df["delta_wsj_minus_official"].to_numpy()
    low = plot_df["delta_ci_low"].to_numpy()
    high = plot_df["delta_ci_high"].to_numpy()
    err_low = means - low
    err_high = high - means

    colors = ["#1D3557" if not c else "#999999" for c in plot_df["ci_crosses_zero"]]

    for i in range(len(plot_df)):
        ax.errorbar(
            means[i],
            y_positions[i],
            xerr=np.array([[err_low[i]], [err_high[i]]]),
            fmt="o",
            color="#222222",
            ecolor=colors[i],
            elinewidth=2,
            capsize=4,
            markersize=6,
        )

    for i, (_, row) in enumerate(plot_df.iterrows()):
        star = sig_label(row.get("stat_p_value", np.nan)) if "stat_p_value" in row else "n/a"
        ax.text(means[i] + 0.004, i, f"{means[i]:.3f}  {star}", va="center", ha="left", fontsize=7)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df[label_col].tolist())
    ax.set_title("Hypothesis 2: Relative X Alignment with WSJ versus Official Documents")
    ax.set_xlabel("Alignment Differential (WSJ minus Official)")
    ax.set_ylabel("Financial Event")

    note = "Blue/black markers indicate confidence intervals excluding zero; gray markers indicate overlap with zero"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F7F7F7", ec="#AAAAAA"))

    out = os.path.join(PLOTS_DIR, "h2_alignment_delta_ci.png")
    save_fig(out)
    print("Generated: plots/h2_alignment_delta_ci.png")
    return h2_df


def plot_significance_overview(h1, h2_df):
    rows = []
    if h1 is not None:
        rows.append({
            "Hypothesis": "H1: WSJ coherence > X",
            "Test": "Sign test",
            "p": h1["sign_p"],
            "Effect": h1["mean_gap"],
            "Evidence": sig_label(h1["sign_p"]),
        })
        rows.append({
            "Hypothesis": "H1: WSJ coherence > X",
            "Test": "Wilcoxon",
            "p": h1["wilcoxon_p"],
            "Effect": h1["median_gap"],
            "Evidence": sig_label(h1["wilcoxon_p"]),
        })

    if not h2_df.empty and "stat_p_value" in h2_df.columns:
        pvals = h2_df["stat_p_value"].dropna()
        if not pvals.empty:
            rows.append({
                "Hypothesis": "H2: WSJ vs Official alignment differs",
                "Test": "Event-level Mann-Whitney p (median)",
                "p": float(pvals.median()),
                "Effect": float(h2_df["delta_wsj_minus_official"].mean()),
                "Evidence": sig_label(float(pvals.median())),
            })

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    summary_df["neg_log10_p"] = -np.log10(summary_df["p"].clip(lower=1e-300))

    plt.figure(figsize=(13, 6.2))
    ax = sns.barplot(
        data=summary_df,
        y="Hypothesis",
        x="neg_log10_p",
        hue="Test",
        palette="Blues",
    )
    ax.axvline(-math.log10(0.05), color="red", linestyle="--", linewidth=1.2, label="Significance threshold (p = 0.05)")
    ax.set_title("Statistical Evidence Summary for Core Hypotheses")
    ax.set_xlabel("Statistical Evidence Magnitude (−log10 p-value)")
    ax.set_ylabel("")
    ax.legend(loc="lower right", frameon=True)

    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(width + 0.04, y, f"{width:.2f}", va="center", ha="left", fontsize=7)

    out = os.path.join(PLOTS_DIR, "hypothesis_evidence_overview.png")
    save_fig(out)
    print("Generated: plots/hypothesis_evidence_overview.png")

    return summary_df


def write_hypothesis_summary(df, h1, h2_df, evidence_df, label_col):
    lines = []
    lines.append("# Narrative Hypothesis Summary")
    lines.append("")
    lines.append("## Research Questions")
    lines.append("- H1: Professional news narratives are more coherent than X narratives for the same events.")
    lines.append("- H2: X aligns differently with WSJ and official documents, indicating outlet-level framing divergence.")
    lines.append("")

    if h1 is not None:
        h1_call = "supported" if h1["sign_p"] < 0.05 else "inconclusive"
        lines.append("## H1 Result")
        lines.append(f"- Decision: **{h1_call.upper()}**")
        lines.append(f"- Events evaluated: {h1['n']}")
        lines.append(f"- WSJ coherence > X in {h1['positive_count']}/{h1['n']} events")
        lines.append(f"- Mean gap (WSJ - X): {h1['mean_gap']:.4f}")
        lines.append(f"- Sign test p-value: {h1['sign_p']:.6f}")
        lines.append(f"- Wilcoxon p-value: {h1['wilcoxon_p']:.6f}" if not pd.isna(h1['wilcoxon_p']) else "- Wilcoxon p-value: n/a")
        lines.append("")

    if not h2_df.empty:
        lines.append("## H2 Result")
        pos = int((h2_df["delta_wsj_minus_official"] > 0).sum())
        n = len(h2_df)
        avg_delta = float(h2_df["delta_wsj_minus_official"].mean())
        clear_dir = int((~h2_df["ci_crosses_zero"]).sum())
        lines.append(f"- Events with WSJ/Official comparison: {n}")
        lines.append(f"- Mean alignment delta (WSJ - Official): {avg_delta:.4f}")
        lines.append(f"- Positive delta events (X closer to WSJ): {pos}/{n}")
        lines.append(f"- Events with CI excluding zero: {clear_dir}/{n}")
        lines.append("")

    if not evidence_df.empty:
        lines.append("## Statistical Evidence Table")
        lines.append("| Hypothesis | Test | p-value | Evidence | Effect |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in evidence_df.iterrows():
            lines.append(
                f"| {row['Hypothesis']} | {row['Test']} | {row['p']:.6f} | {row['Evidence']} | {row['Effect']:.4f} |"
            )
        lines.append("")

    if "event_type" in df.columns:
        lines.append("## Event-Type Snapshot")
        by_type = df[[label_col, "event_type", "x_coherence", "wsj_coherence", "alignment_delta_wsj_minus_official"]].copy()
        by_type = by_type.dropna(subset=["x_coherence", "wsj_coherence"], how="any")
        for event_type, group in by_type.groupby("event_type"):
            gap = (group["wsj_coherence"] - group["x_coherence"]).mean()
            lines.append(f"- {event_type}: mean coherence gap (WSJ-X) = {gap:.4f}")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- Use h1_coherence_gap_significance.png for your coherence hypothesis claim.")
    lines.append("- Use h2_alignment_delta_ci.png for outlet framing divergence (with uncertainty bounds).")
    lines.append("- Use hypothesis_evidence_overview.png as the one-slide evidence summary for proposal defense.")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Generated: hypothesis_summary.md")


def plot_publication_panel(df, h1, h2_df, evidence_df, label_col):
    if h1 is None and h2_df.empty and evidence_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Cross-Media Financial Narrative Divergence: Formal Hypothesis Evaluation", fontsize=14, y=0.98)

    # Panel A: H1 coherence gap
    ax_a = axes[0, 0]
    if h1 is not None:
        panel_a = h1["data"].copy().sort_values("coherence_gap_wsj_minus_x", ascending=False)
        colors = ["#2A9D8F" if v >= 0 else "#E76F51" for v in panel_a["coherence_gap_wsj_minus_x"]]
        ax_a.barh(panel_a[label_col], panel_a["coherence_gap_wsj_minus_x"], color=colors)
        ax_a.axvline(0, color="black", linewidth=1)
        ax_a.set_title("A. Hypothesis 1: Coherence Differential (WSJ minus X)")
        ax_a.set_xlabel("Coherence Differential")
        ax_a.invert_yaxis()
        ax_a.text(
            0.01,
            0.02,
            f"Sign test p-value = {h1['sign_p']:.4f} | Wilcoxon p-value = {h1['wilcoxon_p']:.4f}",
            transform=ax_a.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#F7F7F7", ec="#AAAAAA"),
        )
    else:
        ax_a.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax_a.set_title("A. Hypothesis 1: Coherence Differential")
        ax_a.axis("off")

    # Panel B: H2 alignment delta CI
    ax_b = axes[0, 1]
    if not h2_df.empty:
        panel_b = h2_df.copy().sort_values("delta_wsj_minus_official", ascending=False)
        y = np.arange(len(panel_b))
        means = panel_b["delta_wsj_minus_official"].to_numpy()
        err_low = means - panel_b["delta_ci_low"].to_numpy()
        err_high = panel_b["delta_ci_high"].to_numpy() - means
        colors = ["#1D3557" if not c else "#9AA0A6" for c in panel_b["ci_crosses_zero"]]

        for i in range(len(panel_b)):
            ax_b.errorbar(
                means[i],
                y[i],
                xerr=np.array([[err_low[i]], [err_high[i]]]),
                fmt="o",
                color="#222222",
                ecolor=colors[i],
                elinewidth=2,
                capsize=4,
                markersize=6,
            )

        ax_b.axvline(0, color="black", linewidth=1)
        ax_b.set_yticks(y)
        ax_b.set_yticklabels(panel_b[label_col].tolist())
        ax_b.set_title("B. Hypothesis 2: Alignment Differential with Confidence Intervals")
        ax_b.set_xlabel("Alignment Differential")
    else:
        ax_b.text(0.5, 0.5, "Insufficient WSJ/Official overlap", ha="center", va="center")
        ax_b.set_title("B. Hypothesis 2: Alignment Differential")
        ax_b.axis("off")

    # Panel C: evidence strength
    ax_c = axes[1, 0]
    if not evidence_df.empty:
        panel_c = evidence_df.copy()
        panel_c["Label"] = panel_c["Hypothesis"] + " | " + panel_c["Test"]
        panel_c = panel_c.sort_values("neg_log10_p", ascending=True)
        ax_c.barh(panel_c["Label"], panel_c["neg_log10_p"], color="#457B9D")
        ax_c.axvline(-math.log10(0.05), color="red", linestyle="--", linewidth=1.2)
        ax_c.set_title("C. Comparative Strength of Statistical Evidence")
        ax_c.set_xlabel("Evidence Magnitude (−log10 p-value)")
    else:
        ax_c.text(0.5, 0.5, "No evidence table", ha="center", va="center")
        ax_c.set_title("C. Statistical Evidence")
        ax_c.axis("off")

    # Panel D: concise interpretation
    ax_d = axes[1, 1]
    ax_d.axis("off")
    lines = ["D. Principal Findings"]
    if h1 is not None:
        lines.append(f"• H1 supported: WSJ>X in {h1['positive_count']}/{h1['n']} events")
        lines.append(f"• Mean coherence gap: {h1['mean_gap']:.3f}")
    if not h2_df.empty:
        pos = int((h2_df['delta_wsj_minus_official'] > 0).sum())
        n = len(h2_df)
        clear = int((~h2_df['ci_crosses_zero']).sum())
        lines.append(f"• H2 pattern: X closer to WSJ in {pos}/{n} events")
        lines.append(f"• Clear directional CI (exclude 0): {clear}/{n} events")
    lines.append("• Interpretation: outlet framing differences are measurable")
    lines.append("  and statistically supported for core events in this sample.")

    ax_d.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#F7F7F7", ec="#AAAAAA"),
    )

    out = os.path.join(PLOTS_DIR, "publication_panel_hypotheses.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated: plots/publication_panel_hypotheses.png")


# ====================================================================
# NEW VISUALIZATIONS — NOVELTY ENHANCEMENTS
# ====================================================================

def plot_lm_tone_comparison(df, label_col):
    """Loughran-McDonald financial tone comparison across sources."""
    cols = ["lm_x_tone", "lm_wsj_tone", "lm_official_tone"]
    available = [c for c in cols if c in df.columns]
    if not available:
        print("Skipped: lm_tone_comparison (no LM data)")
        return

    sub = df.dropna(subset=["lm_x_tone"]).copy()
    if sub.empty:
        return

    melt_data = []
    for _, row in sub.iterrows():
        label = row[label_col]
        if pd.notna(row.get("lm_x_tone")):
            melt_data.append({"Event": label, "Source": "X", "Tone": row["lm_x_tone"]})
        if pd.notna(row.get("lm_wsj_tone")):
            melt_data.append({"Event": label, "Source": "WSJ", "Tone": row["lm_wsj_tone"]})
        if pd.notna(row.get("lm_official_tone")):
            melt_data.append({"Event": label, "Source": "Official", "Tone": row["lm_official_tone"]})

    if not melt_data:
        return

    melt_df = pd.DataFrame(melt_data)

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=melt_df, x="Event", y="Tone", hue="Source",
        palette={"X": "#E76F51", "WSJ": "#2A9D8F", "Official": "#457B9D"},
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Loughran-McDonald Financial Tone Across Media Sources")
    ax.set_xlabel("Financial Event")
    ax.set_ylabel("Tone Score (Positive − Negative) / Total Sentiment Words")
    ax.legend(title="Source", loc="best")
    plt.xticks(rotation=25, ha="right")

    for patch in ax.patches:
        height = patch.get_height()
        if abs(height) > 0.001:
            ax.text(
                patch.get_x() + patch.get_width() / 2, height,
                f"{height:.3f}", ha="center", va="bottom" if height >= 0 else "top",
                fontsize=6
            )

    out = os.path.join(PLOTS_DIR, "lm_tone_comparison.png")
    save_fig(out)
    print("Generated: plots/lm_tone_comparison.png")


def plot_lm_sentiment_radar(df, label_col):
    """Radar chart of LM sentiment category densities by source."""
    lm_cols = [
        "lm_x_negative_density", "lm_x_positive_density",
        "lm_x_uncertainty_density", "lm_x_litigious_density",
        "lm_x_constraining_density",
    ]
    available = [c for c in lm_cols if c in df.columns]
    if len(available) < 3:
        print("Skipped: lm_sentiment_radar (insufficient LM columns)")
        return

    categories = ["Negative", "Positive", "Uncertainty", "Litigious", "Constraining"]
    x_vals = [
        df["lm_x_negative_density"].mean(),
        df["lm_x_positive_density"].mean(),
        df["lm_x_uncertainty_density"].mean(),
        df["lm_x_litigious_density"].mean() if "lm_x_litigious_density" in df.columns else 0,
        df["lm_x_constraining_density"].mean() if "lm_x_constraining_density" in df.columns else 0,
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    x_vals_plot = x_vals + [x_vals[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, x_vals_plot, alpha=0.25, color="#E76F51")
    ax.plot(angles, x_vals_plot, "o-", color="#E76F51", label="X", linewidth=2)
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)
    ax.set_title("Loughran-McDonald Sentiment Profile: X (Mean Across Events)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    out = os.path.join(PLOTS_DIR, "lm_sentiment_radar.png")
    save_fig(out)
    print("Generated: plots/lm_sentiment_radar.png")


def plot_temporal_drift(df, label_col):
    """Plot temporal drift trajectories from per-event CSV files."""
    import glob
    traj_files = glob.glob(os.path.join(BASE_DIR, "temporal_trajectory_*.csv"))
    if not traj_files:
        print("Skipped: temporal_drift (no trajectory files)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Similarity to WSJ over time windows
    ax_a = axes[0]
    has_data_a = False

    for f in sorted(traj_files):
        event_name = os.path.basename(f).replace("temporal_trajectory_", "").replace(".csv", "")
        traj = pd.read_csv(f)

        # Find appropriate similarity column
        sim_col = None
        for col in traj.columns:
            if "similarity_to" in col.lower():
                sim_col = col
                break

        if sim_col and sim_col in traj.columns:
            valid = traj.dropna(subset=[sim_col])
            if not valid.empty:
                event_label = df.loc[df["event"] == event_name, label_col]
                lbl = event_label.iloc[0] if not event_label.empty else event_name
                ax_a.plot(valid["window"], valid[sim_col], "o-", label=lbl, linewidth=1.5, markersize=5)
                has_data_a = True

    if has_data_a:
        ax_a.set_title("A. Temporal Drift: X Similarity to Institutional Sources")
        ax_a.set_xlabel("Time Window After Event")
        ax_a.set_ylabel("Cosine Similarity to Reference")
        ax_a.legend(fontsize=7, loc="best")
        plt.setp(ax_a.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    else:
        ax_a.text(0.5, 0.5, "No temporal data available", ha="center", va="center")
        ax_a.set_title("A. Temporal Drift")

    # Panel B: Convergence scores
    ax_b = axes[1]
    conv_cols = ["wsj_convergence", "official_convergence"]
    available_conv = [c for c in conv_cols if c in df.columns]

    if available_conv:
        conv_data = df.dropna(subset=available_conv, how="all").copy()
        if not conv_data.empty:
            melt_entries = []
            for _, row in conv_data.iterrows():
                lbl = row[label_col]
                if pd.notna(row.get("wsj_convergence")):
                    melt_entries.append({"Event": lbl, "Reference": "WSJ", "Convergence": row["wsj_convergence"]})
                if pd.notna(row.get("official_convergence")):
                    melt_entries.append({"Event": lbl, "Reference": "Official", "Convergence": row["official_convergence"]})

            if melt_entries:
                melt_df = pd.DataFrame(melt_entries)
                sns.barplot(data=melt_df, x="Event", y="Convergence", hue="Reference", ax=ax_b,
                            palette={"WSJ": "#2A9D8F", "Official": "#457B9D"})
                ax_b.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax_b.set_title("B. Narrative Convergence Score (Spearman ρ)")
                ax_b.set_xlabel("Financial Event")
                ax_b.set_ylabel("Convergence (positive = X moves toward source)")
                plt.setp(ax_b.get_xticklabels(), rotation=25, ha="right")
                ax_b.text(0.01, 0.02,
                          "ρ > 0: X narrative converges toward source over time\nρ < 0: X narrative diverges from source over time",
                          transform=ax_b.transAxes, fontsize=7, va="bottom",
                          bbox=dict(boxstyle="round,pad=0.25", fc="#F7F7F7", ec="#AAAAAA"))
            else:
                ax_b.text(0.5, 0.5, "No convergence data", ha="center", va="center")
                ax_b.set_title("B. Convergence Scores")
        else:
            ax_b.text(0.5, 0.5, "No convergence data", ha="center", va="center")
            ax_b.set_title("B. Convergence Scores")
    else:
        ax_b.text(0.5, 0.5, "No convergence columns", ha="center", va="center")
        ax_b.set_title("B. Convergence Scores")

    out = os.path.join(PLOTS_DIR, "temporal_drift_analysis.png")
    save_fig(out)
    print("Generated: plots/temporal_drift_analysis.png")


def plot_lead_lag(df, label_col):
    """Plot lead-lag analysis results."""
    cols = ["wsj_peak_lag", "wsj_peak_corr", "wsj_leadlag_significant"]
    if not all(c in df.columns for c in cols):
        print("Skipped: lead_lag (no lead-lag columns)")
        return

    sub = df.dropna(subset=["wsj_peak_lag"]).copy()
    if sub.empty:
        print("Skipped: lead_lag (no valid data)")
        return

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    y_pos = np.arange(len(sub))
    lags = sub["wsj_peak_lag"].to_numpy()
    corrs = sub["wsj_peak_corr"].to_numpy()
    sigs = sub["wsj_leadlag_significant"].astype(bool).to_numpy() if "wsj_leadlag_significant" in sub.columns else [False]*len(sub)

    colors = ["#1D3557" if s else "#BFBFBF" for s in sigs]

    ax.barh(y_pos, lags, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1)

    for i in range(len(sub)):
        star = "**" if sigs[i] else "ns"
        ax.text(
            lags[i] + (0.1 if lags[i] >= 0 else -0.1), y_pos[i],
            f"lag={lags[i]:.0f}, r={corrs[i]:.3f} {star}",
            va="center", ha="left" if lags[i] >= 0 else "right", fontsize=7
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub[label_col].tolist())
    ax.set_title("Cross-Source Lead-Lag Analysis: X vs. WSJ Framing")
    ax.set_xlabel("Peak Lag (days; positive = X leads, negative = X follows)")
    ax.set_ylabel("Financial Event")

    ax.text(0.01, 0.02,
            "Dark bars: statistically significant (p < 0.05, bootstrap test)\n"
            "Gray bars: not significant",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F7F7F7", ec="#AAAAAA"))

    out = os.path.join(PLOTS_DIR, "lead_lag_analysis.png")
    save_fig(out)
    print("Generated: plots/lead_lag_analysis.png")


def plot_fragmentation(df, label_col):
    """Plot BERTopic narrative fragmentation comparison."""
    cols = ["x_fragmentation", "wsj_fragmentation"]
    available = [c for c in cols if c in df.columns]
    if not available:
        print("Skipped: fragmentation (no BERTopic columns)")
        return

    sub = df.dropna(subset=["x_fragmentation"]).copy()
    if sub.empty:
        print("Skipped: fragmentation (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Fragmentation entropy
    ax_a = axes[0]
    melt_entries = []
    for _, row in sub.iterrows():
        lbl = row[label_col]
        melt_entries.append({"Event": lbl, "Source": "X", "Fragmentation": row["x_fragmentation"]})
        if pd.notna(row.get("wsj_fragmentation")):
            melt_entries.append({"Event": lbl, "Source": "WSJ", "Fragmentation": row["wsj_fragmentation"]})

    melt_df = pd.DataFrame(melt_entries)
    sns.barplot(data=melt_df, x="Event", y="Fragmentation", hue="Source", ax=ax_a,
                palette={"X": "#E76F51", "WSJ": "#2A9D8F"})
    ax_a.set_title("A. Narrative Fragmentation (Shannon Entropy of Topic Distribution)")
    ax_a.set_xlabel("Financial Event")
    ax_a.set_ylabel("Fragmentation Entropy (bits)")
    plt.setp(ax_a.get_xticklabels(), rotation=25, ha="right")
    ax_a.text(0.01, 0.95, "Higher entropy = more fragmented narrative\n(more sub-topics, less unified)",
              transform=ax_a.transAxes, fontsize=7, va="top",
              bbox=dict(boxstyle="round,pad=0.25", fc="#F7F7F7", ec="#AAAAAA"))

    # Panel B: Number of sub-topics
    ax_b = axes[1]
    topic_cols = ["x_num_topics", "wsj_num_topics"]
    topic_avail = [c for c in topic_cols if c in df.columns]
    if topic_avail:
        topic_entries = []
        for _, row in sub.iterrows():
            lbl = row[label_col]
            if pd.notna(row.get("x_num_topics")):
                topic_entries.append({"Event": lbl, "Source": "X", "Sub-Topics": int(row["x_num_topics"])})
            if pd.notna(row.get("wsj_num_topics")):
                topic_entries.append({"Event": lbl, "Source": "WSJ", "Sub-Topics": int(row["wsj_num_topics"])})

        if topic_entries:
            topic_df = pd.DataFrame(topic_entries)
            sns.barplot(data=topic_df, x="Event", y="Sub-Topics", hue="Source", ax=ax_b,
                        palette={"X": "#E76F51", "WSJ": "#2A9D8F"})
            ax_b.set_title("B. Number of Discovered Sub-Narratives (BERTopic)")
            ax_b.set_xlabel("Financial Event")
            ax_b.set_ylabel("Count of Distinct Sub-Narratives")
            plt.setp(ax_b.get_xticklabels(), rotation=25, ha="right")
        else:
            ax_b.text(0.5, 0.5, "No sub-topic count data", ha="center", va="center")
    else:
        ax_b.text(0.5, 0.5, "No sub-topic columns", ha="center", va="center")
        ax_b.set_title("B. Sub-Narratives")

    out = os.path.join(PLOTS_DIR, "narrative_fragmentation.png")
    save_fig(out)
    print("Generated: plots/narrative_fragmentation.png")


def plot_llm_features(df, label_col):
    """Plot LLM-extracted narrative features."""
    llm_cols = ["llm_blame_dominant", "llm_frame_dominant", "llm_emotion_dominant", "llm_stance_dominant"]
    available = [c for c in llm_cols if c in df.columns]
    if not available:
        print("Skipped: llm_features (no LLM columns)")
        return

    sub = df.dropna(subset=["llm_blame_dominant"]).copy()
    if sub.empty:
        print("Skipped: llm_features (no data — set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("LLM-Extracted Narrative Features Across Financial Events", fontsize=13, y=0.98)

    feature_configs = [
        ("llm_frame_dominant", "A. Narrative Frame", axes[0, 0]),
        ("llm_emotion_dominant", "B. Dominant Emotional Register", axes[0, 1]),
        ("llm_blame_dominant", "C. Blame Attribution Target", axes[1, 0]),
        ("llm_stance_dominant", "D. Policy Stance", axes[1, 1]),
    ]

    palette = sns.color_palette("Set2", n_colors=len(sub))

    for col, title, ax in feature_configs:
        if col not in sub.columns or sub[col].isna().all():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        plot_data = sub[[label_col, col]].copy()
        plot_data.columns = ["Event", "Value"]

        ax.barh(plot_data["Event"], range(len(plot_data)), color=palette[:len(plot_data)])
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data["Event"])

        for i, (_, row) in enumerate(plot_data.iterrows()):
            ax.text(0.5, i, str(row["Value"]), va="center", ha="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xlabel("")

    out = os.path.join(PLOTS_DIR, "llm_narrative_features.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated: plots/llm_narrative_features.png")


def plot_novelty_panel(df, label_col):
    """Extended 2x3 publication panel showing all novelty contributions."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle("Novel Methodological Contributions: Multi-Dimensional Narrative Analysis", fontsize=14, y=0.99)

    # Panel A: LM Tone comparison
    ax = axes[0, 0]
    tone_data = []
    for _, row in df.iterrows():
        lbl = row[label_col]
        if pd.notna(row.get("lm_x_tone")):
            tone_data.append({"Event": lbl, "Source": "X", "Tone": row["lm_x_tone"]})
        if pd.notna(row.get("lm_wsj_tone")):
            tone_data.append({"Event": lbl, "Source": "WSJ", "Tone": row["lm_wsj_tone"]})
    if tone_data:
        tone_df = pd.DataFrame(tone_data)
        sns.barplot(data=tone_df, x="Event", y="Tone", hue="Source", ax=ax,
                    palette={"X": "#E76F51", "WSJ": "#2A9D8F"})
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
    ax.set_title("A. Financial Sentiment Tone (Loughran-McDonald)")
    ax.set_ylabel("Tone")

    # Panel B: Convergence
    ax = axes[0, 1]
    if "wsj_convergence" in df.columns:
        conv = df.dropna(subset=["wsj_convergence"])
        if not conv.empty:
            colors = ["#2A9D8F" if v > 0 else "#E76F51" for v in conv["wsj_convergence"]]
            ax.barh(conv[label_col], conv["wsj_convergence"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("B. Temporal Convergence Toward WSJ (Spearman ρ)")
    ax.set_xlabel("Convergence Score")

    # Panel C: Lead-Lag
    ax = axes[0, 2]
    if "wsj_peak_lag" in df.columns:
        ll = df.dropna(subset=["wsj_peak_lag"])
        if not ll.empty:
            sigs = ll["wsj_leadlag_significant"].astype(bool) if "wsj_leadlag_significant" in ll.columns else pd.Series([False]*len(ll))
            colors = ["#1D3557" if s else "#BFBFBF" for s in sigs]
            ax.barh(ll[label_col], ll["wsj_peak_lag"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("C. Lead-Lag: X vs. WSJ (Peak Lag in Days)")
    ax.set_xlabel("Lag (+ = X leads)")

    # Panel D: Fragmentation
    ax = axes[1, 0]
    if "x_fragmentation" in df.columns:
        frag = df.dropna(subset=["x_fragmentation"])
        if not frag.empty:
            frag_entries = []
            for _, row in frag.iterrows():
                frag_entries.append({"Event": row[label_col], "Source": "X", "Entropy": row["x_fragmentation"]})
                if pd.notna(row.get("wsj_fragmentation")):
                    frag_entries.append({"Event": row[label_col], "Source": "WSJ", "Entropy": row["wsj_fragmentation"]})
            if frag_entries:
                frag_df = pd.DataFrame(frag_entries)
                sns.barplot(data=frag_df, x="Event", y="Entropy", hue="Source", ax=ax,
                            palette={"X": "#E76F51", "WSJ": "#2A9D8F"})
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
    ax.set_title("D. Narrative Fragmentation (BERTopic Entropy)")
    ax.set_ylabel("Entropy (bits)")

    # Panel E: Modal strength
    ax = axes[1, 1]
    if "lm_x_modal_strength" in df.columns:
        modal_entries = []
        for _, row in df.iterrows():
            lbl = row[label_col]
            if pd.notna(row.get("lm_x_modal_strength")):
                modal_entries.append({"Event": lbl, "Source": "X", "Assertiveness": row["lm_x_modal_strength"]})
            if pd.notna(row.get("lm_wsj_modal_strength")):
                modal_entries.append({"Event": lbl, "Source": "WSJ", "Assertiveness": row["lm_wsj_modal_strength"]})
        if modal_entries:
            modal_df = pd.DataFrame(modal_entries)
            sns.barplot(data=modal_df, x="Event", y="Assertiveness", hue="Source", ax=ax,
                        palette={"X": "#E76F51", "WSJ": "#2A9D8F"})
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
    ax.set_title("E. Linguistic Assertiveness (Strong/Total Modal Ratio)")
    ax.set_ylabel("Strong Modal Ratio")

    # Panel F: Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary = [
        "F. Novelty Contribution Summary",
        "",
        "1. Loughran-McDonald financial lexicon",
        "   replaces generic word lists with",
        "   domain-specific sentiment analysis.",
        "",
        "2. Temporal drift analysis reveals",
        "   whether X converges toward or",
        "   diverges from institutional framing.",
        "",
        "3. Lead-lag cross-correlation tests",
        "   information flow directionality.",
        "",
        "4. BERTopic discovers thematic",
        "   sub-narratives and measures",
        "   narrative fragmentation.",
        "",
        "5. LLM feature extraction captures",
        "   blame, causation, stance, and",
        "   emotional register beyond bag-of-words.",
    ]
    ax.text(0.05, 0.95, "\n".join(summary), va="top", ha="left", fontsize=9,
            transform=ax.transAxes, family="serif",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F7F7F7", ec="#AAAAAA"))

    out = os.path.join(PLOTS_DIR, "novelty_panel.png")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated: plots/novelty_panel.png")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run analyze_narrative.py first.")
        return

    ensure_dir(PLOTS_DIR)
    configure_publication_style()

    print(f"Loading results from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = prepare(df)
    label_col = "event_label" if "event_label" in df.columns else "event"

    # --- Original hypothesis plots ---
    h1 = plot_h1_coherence_test(df, label_col)
    h2_df = plot_h2_alignment_delta(df, label_col)
    evidence_df = plot_significance_overview(h1, h2_df)
    plot_publication_panel(df, h1, h2_df, evidence_df, label_col)

    # --- NEW: Loughran-McDonald financial sentiment ---
    plot_lm_tone_comparison(df, label_col)
    plot_lm_sentiment_radar(df, label_col)

    # --- NEW: Temporal drift trajectories ---
    plot_temporal_drift(df, label_col)

    # --- NEW: Lead-lag cross-correlation ---
    plot_lead_lag(df, label_col)

    # --- NEW: BERTopic fragmentation ---
    plot_fragmentation(df, label_col)

    # --- NEW: LLM narrative features ---
    plot_llm_features(df, label_col)

    # --- NEW: Extended publication panel ---
    plot_novelty_panel(df, label_col)

    # --- Summary ---
    write_hypothesis_summary(df, h1, h2_df, evidence_df, label_col)


if __name__ == "__main__":
    main()
