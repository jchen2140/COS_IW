import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_METRICS = os.path.join(BASE_DIR, "sentivent_semantic_metrics.csv")
X_CLEAN = os.path.join(BASE_DIR, "x_sentivent_cleaned.csv")
LLM_REL = os.path.join(BASE_DIR, "sentivent_llm_relationship_by_model.csv")
LLM_AGR = os.path.join(BASE_DIR, "sentivent_llm_cross_model_agreement.csv")

OUT_FEATURES = os.path.join(BASE_DIR, "sentivent_advanced_event_features.csv")
OUT_COEFS = os.path.join(BASE_DIR, "sentivent_advanced_regression_coefficients.csv")
OUT_TYPE_TESTS = os.path.join(BASE_DIR, "sentivent_advanced_type_permutation_tests.csv")
OUT_SUMMARY = os.path.join(BASE_DIR, "sentivent_advanced_summary.md")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_sentivent_advanced")


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


def permutation_test_diff_means(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"effect": np.nan, "p_value": np.nan}

    obs = float(np.mean(a) - np.mean(b))
    pool = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        test = float(np.mean(pool[:n_a]) - np.mean(pool[n_a:]))
        if abs(test) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return {"effect": obs, "p_value": float(p)}


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < 1e-12:
        return s * 0.0
    return (s - mu) / sd


def run_standardized_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Tuple[pd.DataFrame, float]:
    work = df[[y_col] + x_cols].dropna().copy()
    if len(work) < max(12, len(x_cols) + 3):
        return pd.DataFrame(columns=["feature", "beta_std"]), np.nan

    y = zscore(work[y_col]).to_numpy().reshape(-1, 1)
    X = np.column_stack([zscore(work[c]).to_numpy() for c in x_cols])
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    names = ["intercept"] + x_cols

    # OLS via least squares
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    coef_df = pd.DataFrame({"feature": names, "beta_std": beta.flatten()})
    return coef_df, r2


def build_event_feature_table() -> pd.DataFrame:
    if not os.path.exists(SEMANTIC_METRICS):
        raise FileNotFoundError(f"Missing semantic metrics: {SEMANTIC_METRICS}")
    if not os.path.exists(X_CLEAN):
        raise FileNotFoundError(f"Missing cleaned X data: {X_CLEAN}")

    sem = pd.read_csv(SEMANTIC_METRICS)
    x = pd.read_csv(X_CLEAN)
    x["score"] = pd.to_numeric(x.get("score", 0), errors="coerce").fillna(0)
    x["word_count"] = x["clean_body"].fillna("").astype(str).str.split().str.len()

    x_agg = (
        x.groupby("event", as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_median=("score", "median"),
            score_p90=("score", lambda s: np.percentile(s, 90) if len(s) else np.nan),
            word_count_mean=("word_count", "mean"),
            word_count_median=("word_count", "median"),
            unique_authors=("author", "nunique"),
            post_share=("record_type", lambda s: float((s == "post").mean())),
        )
    )

    out = sem.merge(x_agg, on="event", how="left")

    # Optional LLM-derived uncertainty signals
    if os.path.exists(LLM_REL):
        rel = pd.read_csv(LLM_REL)
        if not rel.empty:
            rel["overall_match_rate"] = rel[
                ["frame_match", "emotion_match", "stance_match", "blame_match", "causal_match"]
            ].mean(axis=1)
            rel_ev = rel.groupby("event", as_index=False)["overall_match_rate"].mean()
            out = out.merge(rel_ev, on="event", how="left")

    if os.path.exists(LLM_AGR):
        agr = pd.read_csv(LLM_AGR)
        if not agr.empty:
            x_ag = agr[agr["source"] == "X"][["event", "overall_pairwise_kappa_mean"]].rename(
                columns={"overall_pairwise_kappa_mean": "x_llm_kappa"}
            )
            out = out.merge(x_ag, on="event", how="left")

    return out


def run_type_permutation_tests(df: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    sub = df.dropna(subset=["event_type", "x_sentivent_similarity"]).copy()
    counts = sub["event_type"].value_counts()
    keep = counts[counts >= min_n].index.tolist()
    sub = sub[sub["event_type"].isin(keep)]
    rows = []
    if sub.empty or len(keep) < 2:
        return pd.DataFrame()

    global_mean = float(sub["x_sentivent_similarity"].mean())
    for t in keep:
        a = sub[sub["event_type"] == t]["x_sentivent_similarity"].to_numpy()
        b = sub[sub["event_type"] != t]["x_sentivent_similarity"].to_numpy()
        test = permutation_test_diff_means(a, b, n_perm=4000, seed=42)
        rows.append(
            {
                "event_type": t,
                "n_type": int(len(a)),
                "n_other": int(len(b)),
                "type_mean_similarity": float(np.mean(a)),
                "other_mean_similarity": float(np.mean(b)),
                "global_mean_similarity": global_mean,
                "effect_type_minus_other": test["effect"],
                "p_value_perm": test["p_value"],
            }
        )
    out = pd.DataFrame(rows).sort_values("effect_type_minus_other", ascending=False)
    return out


def plot_regression_coefficients(coef_df: pd.DataFrame) -> None:
    if coef_df.empty:
        return
    sub = coef_df[coef_df["feature"] != "intercept"].copy()
    sub = sub.sort_values("beta_std", ascending=True)
    plt.figure(figsize=(8.5, 5.5))
    ax = sns.barplot(data=sub, y="feature", x="beta_std", color="#1D3557")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Standardized OLS Coefficients: Drivers of X↔SENTiVENT Similarity")
    ax.set_xlabel("Standardized coefficient (beta)")
    ax.set_ylabel("Feature")
    for p in ax.patches:
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + (0.01 if w >= 0 else -0.01), y, f"{w:.2f}", va="center", ha="left" if w >= 0 else "right", fontsize=7)
    save_fig(os.path.join(PLOTS_DIR, "advanced_regression_coefficients.png"))


def plot_type_effects(type_df: pd.DataFrame) -> None:
    if type_df.empty:
        return
    plot_df = type_df.sort_values("effect_type_minus_other", ascending=True).copy()
    plot_df["sig"] = plot_df["p_value_perm"] < 0.05
    plt.figure(figsize=(10, 6.5))
    ax = sns.barplot(
        data=plot_df,
        y="event_type",
        x="effect_type_minus_other",
        hue="sig",
        dodge=False,
        palette={True: "#2A9D8F", False: "#B0B0B0"},
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Permutation-Tested Event-Type Effects on Semantic Alignment")
    ax.set_xlabel("Effect size: type mean minus other types")
    ax.set_ylabel("Event type")
    ax.legend(title="p < 0.05", loc="lower right")
    for p, (_, row) in zip(ax.patches, plot_df.iterrows()):
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + (0.005 if w >= 0 else -0.005), y, f"p={row['p_value_perm']:.3f}", va="center", ha="left" if w >= 0 else "right", fontsize=7)
    save_fig(os.path.join(PLOTS_DIR, "advanced_type_effects_permutation.png"))


def write_summary(features: pd.DataFrame, coef_df: pd.DataFrame, r2: float, type_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Advanced SENTiVENT Analysis Summary")
    lines.append("")
    lines.append(f"- Events analyzed: **{len(features)}**")
    lines.append(f"- Mean X↔SENTiVENT similarity: **{features['x_sentivent_similarity'].mean():.3f}**")
    lines.append("")
    lines.append("## Regression (Standardized OLS)")
    if coef_df.empty or np.isnan(r2):
        lines.append("- Not enough data to estimate stable regression.")
    else:
        lines.append(f"- R²: **{r2:.3f}**")
        top = (
            coef_df[coef_df["feature"] != "intercept"]
            .copy()
            .assign(abs_beta=lambda d: d["beta_std"].abs())
            .sort_values("abs_beta", ascending=False)
            .head(5)
        )
        lines.append("- Top standardized effects:")
        for _, row in top.iterrows():
            lines.append(f"  - `{row['feature']}`: {row['beta_std']:.3f}")
    lines.append("")
    lines.append("## Event-Type Permutation Tests")
    if type_df.empty:
        lines.append("- Insufficient per-type sample size for permutation tests.")
    else:
        sig = type_df[type_df["p_value_perm"] < 0.05]
        lines.append(f"- Types tested: **{len(type_df)}**")
        lines.append(f"- Significant type effects (p < 0.05): **{len(sig)}**")
        if not sig.empty:
            lines.append("- Significant types:")
            for _, row in sig.iterrows():
                lines.append(
                    f"  - `{row['event_type']}`: effect={row['effect_type_minus_other']:.3f}, p={row['p_value_perm']:.4f}"
                )
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Feature table: `{os.path.basename(OUT_FEATURES)}`")
    lines.append(f"- Coefficients: `{os.path.basename(OUT_COEFS)}`")
    lines.append(f"- Type tests: `{os.path.basename(OUT_TYPE_TESTS)}`")
    lines.append(f"- Plots: `{os.path.basename(PLOTS_DIR)}/advanced_regression_coefficients.png`, `{os.path.basename(PLOTS_DIR)}/advanced_type_effects_permutation.png`")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ensure_dir(PLOTS_DIR)
    configure_style()

    features = build_event_feature_table()
    features.to_csv(OUT_FEATURES, index=False)

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
    coef_df.to_csv(OUT_COEFS, index=False)

    type_df = run_type_permutation_tests(features, min_n=5)
    type_df.to_csv(OUT_TYPE_TESTS, index=False)

    plot_regression_coefficients(coef_df)
    plot_type_effects(type_df)
    write_summary(features, coef_df, r2, type_df)

    print(f"Saved: {OUT_FEATURES}")
    print(f"Saved: {OUT_COEFS}")
    print(f"Saved: {OUT_TYPE_TESTS}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Generated plots in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
