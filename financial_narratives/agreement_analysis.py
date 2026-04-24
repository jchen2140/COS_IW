import itertools
from typing import Dict, List, Optional

import numpy as np


FEATURE_FIELDS = [
    "blame_target",
    "causal_mechanism",
    "policy_stance",
    "temporal_orientation",
    "emotional_register",
    "narrative_frame",
]


def _cohens_kappa(labels_a: List[str], labels_b: List[str]) -> Optional[float]:
    """Compute Cohen's kappa for two aligned categorical label lists."""
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return None

    categories = sorted(set(labels_a) | set(labels_b))
    if not categories:
        return None

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=float)

    for a, b in zip(labels_a, labels_b):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    total = matrix.sum()
    if total == 0:
        return None
    po = np.trace(matrix) / total
    pe = (matrix.sum(axis=1) * matrix.sum(axis=0)).sum() / (total ** 2)
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return None
    return float((po - pe) / denom)


def _fleiss_kappa(label_rows: List[List[str]]) -> Optional[float]:
    """
    Compute Fleiss' kappa for complete rows.
    Each row is labels from all raters for one item.
    """
    if not label_rows:
        return None

    n_raters = len(label_rows[0])
    if n_raters < 2:
        return None
    if any(len(row) != n_raters for row in label_rows):
        return None

    categories = sorted({lab for row in label_rows for lab in row})
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((len(label_rows), len(categories)), dtype=float)

    for i, row in enumerate(label_rows):
        for label in row:
            mat[i, cat_to_idx[label]] += 1

    p_j = np.sum(mat, axis=0) / (len(label_rows) * n_raters)
    p_i = (np.sum(mat * (mat - 1), axis=1)) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)
    p_e = np.sum(p_j ** 2)
    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        return None
    return float((p_bar - p_e) / denom)


def summarize_multi_model_agreement(
    per_model_extractions: Dict[str, List[Optional[Dict]]],
    feature_fields: List[str] = None,
) -> Dict:
    """
    Summarize agreement across models for each feature.
    per_model_extractions maps model_name -> list aligned by text index.
    """
    feature_fields = feature_fields or FEATURE_FIELDS
    model_names = sorted(per_model_extractions.keys())
    if not model_names:
        return {"models": [], "feature_agreement": {}, "overall_pairwise_kappa_mean": None}

    n_items = min(len(per_model_extractions[m]) for m in model_names)
    agreement = {}

    for field in feature_fields:
        pairwise_scores = []
        for a, b in itertools.combinations(model_names, 2):
            labels_a = []
            labels_b = []
            for i in range(n_items):
                ea = per_model_extractions[a][i]
                eb = per_model_extractions[b][i]
                if not ea or not eb:
                    continue
                la = ea.get(field)
                lb = eb.get(field)
                if la is None or lb is None:
                    continue
                labels_a.append(str(la))
                labels_b.append(str(lb))
            kappa = _cohens_kappa(labels_a, labels_b)
            if kappa is not None:
                pairwise_scores.append(kappa)

        complete_rows = []
        for i in range(n_items):
            row = []
            complete = True
            for m in model_names:
                em = per_model_extractions[m][i]
                if not em or em.get(field) is None:
                    complete = False
                    break
                row.append(str(em[field]))
            if complete:
                complete_rows.append(row)

        agreement[field] = {
            "pairwise_kappa_mean": float(np.mean(pairwise_scores)) if pairwise_scores else None,
            "pairwise_kappa_min": float(np.min(pairwise_scores)) if pairwise_scores else None,
            "pairwise_kappa_max": float(np.max(pairwise_scores)) if pairwise_scores else None,
            "fleiss_kappa": _fleiss_kappa(complete_rows),
            "n_complete_items": len(complete_rows),
        }

    means = [v["pairwise_kappa_mean"] for v in agreement.values() if v["pairwise_kappa_mean"] is not None]
    return {
        "models": model_names,
        "feature_agreement": agreement,
        "overall_pairwise_kappa_mean": float(np.mean(means)) if means else None,
    }
