from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from agreement_analysis import FEATURE_FIELDS, summarize_multi_model_agreement
from llm_feature_extraction import batch_extract, check_api_availability


DEFAULT_MODEL_SPECS = [
    {"name": "openai_gpt4omini", "provider": "openai", "model": "gpt-4o-mini"},
    {"name": "anthropic_sonnet", "provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"name": "gemini_flash", "provider": "gemini", "model": "models/gemini-2.5-flash"},
]


def get_available_model_specs() -> List[Dict]:
    """Return configured model specs for providers with available API keys."""
    keys = check_api_availability()
    specs = [s for s in DEFAULT_MODEL_SPECS if keys.get(s["provider"], False)]
    return specs


def _consensus_vote(extractions: List[Optional[Dict]], model_names: List[str]) -> Optional[Dict]:
    """
    Build a majority-vote consensus extraction for one text.
    Ties resolve to 'unknown' (or 'unclear' for temporal orientation).
    """
    rows = [e for e in extractions if e is not None]
    if not rows:
        return None

    out = {}
    for field in FEATURE_FIELDS:
        vals = [str(r.get(field)) for r in rows if r.get(field) is not None]
        if not vals:
            out[field] = "unknown" if field != "temporal_orientation" else "unclear"
            continue
        counts = Counter(vals).most_common()
        if len(counts) == 1 or counts[0][1] > counts[1][1]:
            out[field] = counts[0][0]
        else:
            out[field] = "unknown" if field != "temporal_orientation" else "unclear"

    # Consensus confidence from model-level confidence values.
    conf_vals = [str(r.get("confidence", "unknown")) for r in rows]
    conf_counts = Counter(conf_vals).most_common()
    out["confidence"] = conf_counts[0][0] if conf_counts else "unknown"
    out["_models_used"] = len(rows)
    out["_models_available"] = len(model_names)
    return out


def _aggregate_extractions(extractions: List[Optional[Dict]]) -> Dict:
    valid = [e for e in extractions if e is not None]
    result = {"n_extracted": len(valid), "n_attempted": len(extractions)}
    for field in FEATURE_FIELDS:
        vals = [e.get(field, "unknown") for e in valid]
        counts = Counter(vals)
        if counts:
            dominant, count = counts.most_common(1)[0]
            probs = np.array(list(counts.values()), dtype=float) / max(sum(counts.values()), 1.0)
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            result[f"{field}_dominant"] = dominant
            result[f"{field}_dominant_share"] = float(count / len(valid))
            result[f"{field}_distribution"] = dict(counts)
            result[f"{field}_entropy"] = entropy
        else:
            result[f"{field}_dominant"] = "unknown"
            result[f"{field}_dominant_share"] = 0.0
            result[f"{field}_distribution"] = {}
            result[f"{field}_entropy"] = 0.0
    return result


def run_multi_model_extraction(
    texts: List[str],
    event_label: str,
    event_date: str,
    sample_size: int = 30,
    rate_limit_delay: float = 0.3,
    model_specs: List[Dict] = None,
) -> Dict:
    """
    Execute extraction across all available model specs on identical sampled texts.
    """
    model_specs = model_specs if model_specs is not None else get_available_model_specs()
    if not model_specs:
        return {
            "models_used": [],
            "sample_indices": [],
            "per_model": {},
            "agreement": {},
            "consensus_aggregate": {"n_extracted": 0, "n_attempted": 0},
        }

    sample_n = min(sample_size, len(texts))
    if sample_n == 0:
        return {
            "models_used": [s["name"] for s in model_specs],
            "sample_indices": [],
            "per_model": {},
            "agreement": {},
            "consensus_aggregate": {"n_extracted": 0, "n_attempted": 0},
        }

    # Shared deterministic sample for all models.
    rng = np.random.default_rng(42)
    sample_indices = sorted(rng.choice(len(texts), size=sample_n, replace=False).tolist())

    per_model = {}
    aligned_extractions = {}
    for spec in model_specs:
        model_name = spec["name"]
        extractions, _ = batch_extract(
            texts=texts,
            event_label=event_label,
            event_date=event_date,
            provider=spec["provider"],
            model=spec["model"],
            sample_indices=sample_indices,
            rate_limit_delay=rate_limit_delay,
            return_indices=True,
        )
        per_model[model_name] = {
            "provider": spec["provider"],
            "model": spec["model"],
            "aggregate": _aggregate_extractions(extractions),
        }
        aligned_extractions[model_name] = extractions

    consensus_rows = []
    model_names = sorted(aligned_extractions.keys())
    for i in range(len(sample_indices)):
        row_models = [aligned_extractions[m][i] for m in model_names]
        consensus_rows.append(_consensus_vote(row_models, model_names))

    agreement = summarize_multi_model_agreement(aligned_extractions, feature_fields=FEATURE_FIELDS)
    consensus_aggregate = _aggregate_extractions(consensus_rows)

    return {
        "models_used": [s["name"] for s in model_specs],
        "sample_indices": sample_indices,
        "per_model": per_model,
        "agreement": agreement,
        "consensus_aggregate": consensus_aggregate,
    }
