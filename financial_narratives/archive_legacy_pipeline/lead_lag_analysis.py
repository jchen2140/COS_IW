"""
Cross-Source Narrative Lead-Lag Analysis
==========================================
Tests whether X narratives lead or follow WSJ/official framing
using cross-correlation of daily semantic similarity time series.

This answers: Does retail investor discourse anticipate or react to
institutional framing of financial events?

Methods:
    - Daily semantic centroid time series for each source
    - Cross-correlation at multiple lags
    - Granger-causality inspired directional analysis
    - Bootstrap significance testing for lead-lag relationships
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def build_daily_centroid_series(
    dates: pd.Series,
    embeddings: np.ndarray,
    event_date: pd.Timestamp,
    window_days: int = 14,
    min_comments_per_day: int = 2,
) -> pd.DataFrame:
    """
    Build a daily semantic centroid time series from X posts/replies.

    Parameters
    ----------
    dates : pd.Series of datetime
    embeddings : np.ndarray (n, dim)
    event_date : pd.Timestamp
    window_days : int
        Days before and after event to analyze.
    min_comments_per_day : int
        Skip days with fewer comments.

    Returns
    -------
    DataFrame with columns: date, centroid (np.ndarray), n_comments, coherence
    """
    if event_date is None or pd.isna(event_date):
        return pd.DataFrame()

    start = event_date - timedelta(days=window_days)
    end = event_date + timedelta(days=window_days)

    date_col = dates.dt.date
    rows = []

    current = start
    while current <= end:
        day = current.date() if hasattr(current, 'date') else current
        mask = (date_col == day)

        if hasattr(mask, 'to_numpy'):
            mask_arr = mask.to_numpy()
        else:
            mask_arr = np.array(mask)

        day_embs = embeddings[mask_arr]

        if len(day_embs) >= min_comments_per_day:
            centroid = np.mean(day_embs, axis=0).reshape(1, -1)
            coherences = cosine_similarity(day_embs, centroid).flatten()
            rows.append({
                "date": day,
                "centroid": centroid.flatten(),
                "n_comments": len(day_embs),
                "coherence": float(np.mean(coherences)),
            })

        current += timedelta(days=1)

    return pd.DataFrame(rows)


def compute_daily_similarity_series(
    daily_centroids: pd.DataFrame,
    reference_centroid: np.ndarray,
    ref_name: str = "wsj",
) -> pd.DataFrame:
    """
    Compute daily similarity between X centroids and a reference.

    Returns DataFrame with date, similarity columns.
    """
    if daily_centroids.empty or reference_centroid is None:
        return pd.DataFrame()

    rows = []
    for _, row in daily_centroids.iterrows():
        cent = row["centroid"].reshape(1, -1)
        sim = float(cosine_similarity(cent, reference_centroid.reshape(1, -1))[0, 0])
        rows.append({
            "date": row["date"],
            f"similarity_to_{ref_name}": sim,
            "n_comments": row["n_comments"],
        })

    return pd.DataFrame(rows)


def cross_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Compute normalized cross-correlation between two time series.

    Positive lag means series_a leads series_b.
    Negative lag means series_b leads series_a.

    Parameters
    ----------
    series_a : np.ndarray
        First time series (e.g., X similarity).
    series_b : np.ndarray
        Second time series (e.g., WSJ reference similarity).
    max_lag : int
        Maximum lag in both directions.

    Returns
    -------
    DataFrame with lag, correlation columns.
    """
    if len(series_a) < 2 * max_lag + 1 or len(series_b) < 2 * max_lag + 1:
        max_lag = min(len(series_a), len(series_b)) // 3

    # Normalize
    a_norm = (series_a - np.mean(series_a)) / (np.std(series_a) + 1e-9)
    b_norm = (series_b - np.mean(series_b)) / (np.std(series_b) + 1e-9)

    n = min(len(a_norm), len(b_norm))
    rows = []

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a_slice = a_norm[lag:n]
            b_slice = b_norm[:n-lag]
        else:
            a_slice = a_norm[:n+lag]
            b_slice = b_norm[-lag:n]

        if len(a_slice) < 3:
            continue

        corr = float(np.mean(a_slice * b_slice))
        rows.append({"lag": lag, "correlation": corr, "n_overlap": len(a_slice)})

    return pd.DataFrame(rows)


def bootstrap_lead_lag_significance(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int = 5,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict:
    """
    Bootstrap test for whether the observed peak lag is significant.

    Shuffles series_b and recomputes cross-correlation to build a null
    distribution of peak lags and correlations.

    Returns
    -------
    dict with:
        - observed_peak_lag: int
        - observed_peak_corr: float
        - p_value: float (probability of observing this correlation by chance)
        - ci_low, ci_high: confidence interval for null peak correlation
        - significant: bool
    """
    rng = np.random.default_rng(42)

    # Observed
    cc = cross_correlation(series_a, series_b, max_lag)
    if cc.empty:
        return {"observed_peak_lag": None, "significant": False}

    peak_idx = cc["correlation"].abs().idxmax()
    observed_peak_lag = int(cc.loc[peak_idx, "lag"])
    observed_peak_corr = float(cc.loc[peak_idx, "correlation"])

    # Null distribution
    null_peak_corrs = []
    for _ in range(n_bootstrap):
        shuffled_b = rng.permutation(series_b)
        null_cc = cross_correlation(series_a, shuffled_b, max_lag)
        if not null_cc.empty:
            null_peak_corrs.append(float(null_cc["correlation"].abs().max()))

    null_peak_corrs = np.array(null_peak_corrs)
    p_value = float(np.mean(null_peak_corrs >= abs(observed_peak_corr)))

    return {
        "observed_peak_lag": observed_peak_lag,
        "observed_peak_corr": observed_peak_corr,
        "p_value": p_value,
        "ci_low": float(np.percentile(null_peak_corrs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(null_peak_corrs, 100 * (1 - alpha / 2))),
        "significant": p_value < alpha,
        "interpretation": _interpret_lag(observed_peak_lag, observed_peak_corr, p_value < alpha),
    }


def _interpret_lag(peak_lag: int, peak_corr: float, significant: bool) -> str:
    """Human-readable interpretation of lead-lag result."""
    if not significant:
        return "No statistically significant lead-lag relationship detected."
    if peak_lag > 0:
        return (
            f"X leads: peak correlation at lag +{peak_lag} days "
            f"(r={peak_corr:.3f}), suggesting X narratives anticipate "
            f"institutional framing shifts."
        )
    elif peak_lag < 0:
        return (
            f"X follows: peak correlation at lag {peak_lag} days "
            f"(r={peak_corr:.3f}), suggesting X narratives adopt "
            f"institutional framing after a delay."
        )
    else:
        return (
            f"Synchronous: peak correlation at lag 0 "
            f"(r={peak_corr:.3f}), narratives shift simultaneously."
        )


def full_lead_lag_analysis(
    dates: pd.Series,
    embeddings: np.ndarray,
    event_date: pd.Timestamp,
    wsj_centroid: np.ndarray = None,
    official_centroid: np.ndarray = None,
    window_days: int = 14,
    max_lag: int = 5,
) -> Dict:
    """
    Run complete lead-lag analysis for one event.

    Returns dict with:
        - daily_series: DataFrame
        - wsj_cross_corr: DataFrame
        - official_cross_corr: DataFrame
        - wsj_lead_lag: Dict (bootstrap results)
        - official_lead_lag: Dict
    """
    daily = build_daily_centroid_series(
        dates, embeddings, event_date,
        window_days=window_days, min_comments_per_day=1
    )

    result = {"daily_series": daily}

    if daily.empty or len(daily) < 5:
        result["wsj_cross_corr"] = pd.DataFrame()
        result["official_cross_corr"] = pd.DataFrame()
        result["wsj_lead_lag"] = {"significant": False, "interpretation": "Insufficient data"}
        result["official_lead_lag"] = {"significant": False, "interpretation": "Insufficient data"}
        return result

    # WSJ analysis
    if wsj_centroid is not None:
        wsj_sim = compute_daily_similarity_series(daily, wsj_centroid, "wsj")
        if not wsj_sim.empty and len(wsj_sim) >= 5:
            # Use coherence as proxy for narrative state
            coherence_series = daily["coherence"].to_numpy()
            wsj_sim_series = wsj_sim["similarity_to_wsj"].to_numpy()

            result["wsj_similarity_series"] = wsj_sim
            result["wsj_cross_corr"] = cross_correlation(
                coherence_series, wsj_sim_series, max_lag
            )
            result["wsj_lead_lag"] = bootstrap_lead_lag_significance(
                coherence_series, wsj_sim_series, max_lag
            )
        else:
            result["wsj_cross_corr"] = pd.DataFrame()
            result["wsj_lead_lag"] = {"significant": False, "interpretation": "Insufficient WSJ data"}
    else:
        result["wsj_cross_corr"] = pd.DataFrame()
        result["wsj_lead_lag"] = {"significant": False, "interpretation": "No WSJ data"}

    # Official analysis
    if official_centroid is not None:
        off_sim = compute_daily_similarity_series(daily, official_centroid, "official")
        if not off_sim.empty and len(off_sim) >= 5:
            coherence_series = daily["coherence"].to_numpy()
            off_sim_series = off_sim["similarity_to_official"].to_numpy()

            result["official_similarity_series"] = off_sim
            result["official_cross_corr"] = cross_correlation(
                coherence_series, off_sim_series, max_lag
            )
            result["official_lead_lag"] = bootstrap_lead_lag_significance(
                coherence_series, off_sim_series, max_lag
            )
        else:
            result["official_cross_corr"] = pd.DataFrame()
            result["official_lead_lag"] = {"significant": False, "interpretation": "Insufficient official data"}
    else:
        result["official_cross_corr"] = pd.DataFrame()
        result["official_lead_lag"] = {"significant": False, "interpretation": "No official data"}

    return result
