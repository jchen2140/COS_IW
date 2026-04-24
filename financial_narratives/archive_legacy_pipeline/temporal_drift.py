"""
Temporal Narrative Drift Analysis
===================================
Tracks how X narrative centroids evolve over time relative to
WSJ and official document framing — answering whether retail investors
converge toward or diverge from institutional narratives after events.

Key metrics:
    - Windowed semantic centroids at configurable time intervals
    - Drift velocity (rate of centroid movement between windows)
    - Convergence score (does X move toward WSJ/official over time?)
    - Narrative half-life (time for X to stabilize)
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_WINDOWS = [
    ("0-1h", timedelta(hours=0), timedelta(hours=1)),
    ("1-6h", timedelta(hours=1), timedelta(hours=6)),
    ("6-24h", timedelta(hours=6), timedelta(hours=24)),
    ("24-72h", timedelta(hours=24), timedelta(hours=72)),
    ("72h+", timedelta(hours=72), timedelta(days=14)),
]


def compute_windowed_centroids(
    dates: pd.Series,
    embeddings: np.ndarray,
    event_date: pd.Timestamp,
    windows: List[Tuple[str, timedelta, timedelta]] = None,
    min_comments: int = 3,
) -> Dict[str, Dict]:
    """
    Compute semantic centroids for each time window after an event.

    Parameters
    ----------
    dates : pd.Series
        Datetime of each comment.
    embeddings : np.ndarray
        Embedding vectors (n_comments x dim).
    event_date : pd.Timestamp
        Event timestamp.
    windows : list of (name, start_offset, end_offset)
        Time windows relative to event_date.
    min_comments : int
        Minimum comments to form a meaningful centroid.

    Returns
    -------
    dict keyed by window name -> {centroid, n_comments, mean_coherence}
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    if event_date is None or pd.isna(event_date):
        return {}

    results = {}
    for name, start_offset, end_offset in windows:
        start = event_date + start_offset
        end = event_date + end_offset
        mask = (dates >= start) & (dates < end)
        window_embs = embeddings[mask.to_numpy()] if hasattr(mask, 'to_numpy') else embeddings[mask]

        if len(window_embs) < min_comments:
            results[name] = {
                "centroid": None,
                "n_comments": len(window_embs),
                "mean_coherence": None,
            }
            continue

        centroid = np.mean(window_embs, axis=0).reshape(1, -1)
        coherences = cosine_similarity(window_embs, centroid).flatten()

        results[name] = {
            "centroid": centroid,
            "n_comments": len(window_embs),
            "mean_coherence": float(np.mean(coherences)),
        }

    return results


def compute_drift_trajectory(
    windowed_centroids: Dict[str, Dict],
    reference_centroid: np.ndarray,
    reference_name: str = "WSJ",
) -> pd.DataFrame:
    """
    Track how X centroid similarity to a reference evolves over time.

    Parameters
    ----------
    windowed_centroids : dict
        Output of compute_windowed_centroids().
    reference_centroid : np.ndarray
        Centroid of reference source (WSJ or official). Shape (1, dim).
    reference_name : str
        Label for the reference source.

    Returns
    -------
    DataFrame with columns:
        window, n_comments, similarity_to_ref, coherence, drift_velocity
    """
    rows = []
    prev_centroid = None

    for window_name, data in windowed_centroids.items():
        centroid = data["centroid"]

        if centroid is None or reference_centroid is None:
            rows.append({
                "window": window_name,
                "n_comments": data["n_comments"],
                f"similarity_to_{reference_name.lower()}": None,
                "coherence": data["mean_coherence"],
                "drift_velocity": None,
            })
            prev_centroid = centroid
            continue

        sim = float(cosine_similarity(centroid, reference_centroid)[0, 0])

        drift = None
        if prev_centroid is not None:
            drift = float(1.0 - cosine_similarity(centroid, prev_centroid)[0, 0])

        rows.append({
            "window": window_name,
            "n_comments": data["n_comments"],
            f"similarity_to_{reference_name.lower()}": sim,
            "coherence": data["mean_coherence"],
            "drift_velocity": drift,
        })

        prev_centroid = centroid

    return pd.DataFrame(rows)


def compute_convergence_score(trajectory: pd.DataFrame, ref_col: str) -> Optional[float]:
    """
    Compute a convergence score: positive if X moves toward the reference
    over time, negative if it diverges.

    With 2 points: simple direction (+1 or -1).
    With 3+ points: Spearman rank correlation.
    """
    from scipy.stats import spearmanr

    valid = trajectory.dropna(subset=[ref_col])
    if len(valid) < 2:
        return None

    sims = valid[ref_col].to_numpy()

    if len(valid) == 2:
        # With only 2 points, return direction: +1 if increasing, -1 if decreasing
        return 1.0 if sims[1] > sims[0] else -1.0

    ranks = np.arange(len(valid))
    corr, p_value = spearmanr(ranks, sims)
    return float(corr)


def compute_narrative_halflife(
    trajectory: pd.DataFrame,
    coherence_col: str = "coherence",
) -> Optional[int]:
    """
    Estimate the time window index at which X narrative coherence stabilizes.
    Defined as the first window where coherence change < 5% relative to previous.
    """
    valid = trajectory.dropna(subset=[coherence_col])
    coherences = valid[coherence_col].to_numpy()

    if len(coherences) < 2:
        return None

    for i in range(1, len(coherences)):
        relative_change = abs(coherences[i] - coherences[i-1]) / (abs(coherences[i-1]) + 1e-9)
        if relative_change < 0.05:
            return i

    return len(coherences) - 1  # Never stabilized


def full_temporal_analysis(
    dates: pd.Series,
    embeddings: np.ndarray,
    event_date: pd.Timestamp,
    wsj_centroid: np.ndarray = None,
    official_centroid: np.ndarray = None,
    min_comments: int = 3,
) -> Dict:
    """
    Run full temporal drift analysis for one event.

    Returns
    -------
    dict with:
        - windowed_centroids: raw centroid data
        - wsj_trajectory: DataFrame of drift toward WSJ
        - official_trajectory: DataFrame of drift toward official
        - wsj_convergence: float (positive = converging)
        - official_convergence: float
        - narrative_halflife: int (window index)
    """
    centroids = compute_windowed_centroids(
        dates, embeddings, event_date, min_comments=min_comments
    )

    result = {"windowed_centroids": centroids}

    if wsj_centroid is not None:
        wsj_traj = compute_drift_trajectory(centroids, wsj_centroid, "WSJ")
        result["wsj_trajectory"] = wsj_traj
        result["wsj_convergence"] = compute_convergence_score(
            wsj_traj, "similarity_to_wsj"
        )
    else:
        result["wsj_trajectory"] = pd.DataFrame()
        result["wsj_convergence"] = None

    if official_centroid is not None:
        off_traj = compute_drift_trajectory(centroids, official_centroid, "Official")
        result["official_trajectory"] = off_traj
        result["official_convergence"] = compute_convergence_score(
            off_traj, "similarity_to_official"
        )
    else:
        result["official_trajectory"] = pd.DataFrame()
        result["official_convergence"] = None

    # Use WSJ trajectory for half-life if available, else official
    traj_for_halflife = result.get("wsj_trajectory", pd.DataFrame())
    if traj_for_halflife.empty:
        traj_for_halflife = result.get("official_trajectory", pd.DataFrame())
    if not traj_for_halflife.empty:
        result["narrative_halflife"] = compute_narrative_halflife(traj_for_halflife)
    else:
        result["narrative_halflife"] = None

    return result
