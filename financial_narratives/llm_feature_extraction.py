"""
LLM-Based Narrative Feature Extraction
========================================
Uses large language models (OpenAI GPT-4o / Anthropic Claude) to extract
structured narrative features from financial texts that go far beyond
lexicon-counting approaches.

Extracted features:
    1. Blame attribution — Who or what is blamed for the event?
    2. Causal mechanism — What explanation is given for why it happened?
    3. Policy stance — Hawkish / Dovish / Neutral / Mixed
    4. Temporal orientation — Backward-looking / Forward-looking / Both
    5. Emotional register — Dominant emotion (fear, greed, confusion, etc.)
    6. Narrative frame — How is the event framed? (crisis, opportunity, etc.)

Supports batched extraction with rate limiting, caching, and fallback.
"""

import os
import json
import time
import hashlib
import pandas as pd
from typing import List, Dict, Optional

# Cache directory for LLM responses (avoid re-querying)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llm_cache")

EXTRACTION_PROMPT = """You are an expert financial narrative analyst. Analyze the following text about a financial event and extract structured narrative features.

EVENT CONTEXT: {event_label} ({event_date})

TEXT:
\"\"\"
{text}
\"\"\"

Extract the following features. Respond ONLY with a valid JSON object:

{{
    "blame_target": "<Who or what is blamed? e.g., 'Federal Reserve', 'bank management', 'regulators', 'market forces', 'no blame', etc.>",
    "causal_mechanism": "<What causal explanation is given? e.g., 'interest rate policy', 'poor risk management', 'contagion', 'speculation', etc.>",
    "policy_stance": "<One of: hawkish, dovish, neutral, mixed, not_applicable>",
    "temporal_orientation": "<One of: backward_looking, forward_looking, both, unclear>",
    "emotional_register": "<Dominant emotion. One of: fear, anger, greed, confusion, relief, confidence, resignation, schadenfreude, anxiety, neutral>",
    "narrative_frame": "<How is the event framed? One of: crisis, opportunity, systemic_risk, policy_failure, market_correction, institutional_failure, populist_revolt, routine_adjustment, cautionary_tale, vindication>",
    "confidence": "<Your confidence in this extraction: high, medium, low>"
}}
"""

# Valid values for validation
VALID_VALUES = {
    "policy_stance": {"hawkish", "dovish", "neutral", "mixed", "not_applicable"},
    "temporal_orientation": {"backward_looking", "forward_looking", "both", "unclear"},
    "emotional_register": {
        "fear", "anger", "greed", "confusion", "relief", "confidence",
        "resignation", "schadenfreude", "anxiety", "neutral"
    },
    "narrative_frame": {
        "crisis", "opportunity", "systemic_risk", "policy_failure",
        "market_correction", "institutional_failure", "populist_revolt",
        "routine_adjustment", "cautionary_tale", "vindication"
    },
    "confidence": {"high", "medium", "low"},
}


def _get_cache_key(
    text: str,
    event_label: str,
    provider: str = "unknown",
    model: str = "default",
    prompt_version: str = "v1",
) -> str:
    """Generate a deterministic cache key."""
    content = f"{provider}::{model}::{prompt_version}::{event_label}::{text[:500]}"
    return hashlib.md5(content.encode()).hexdigest()


def _load_cached(cache_key: str) -> Optional[Dict]:
    """Load a cached LLM response if available."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return None


def _save_cache(cache_key: str, result: Dict):
    """Save an LLM response to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)


def _validate_extraction(result: Dict) -> Dict:
    """Validate and normalize extracted features."""
    for field, valid_set in VALID_VALUES.items():
        if field in result:
            val = result[field].lower().strip()
            if val not in valid_set:
                result[field] = "unclear" if field == "temporal_orientation" else "unknown"
            else:
                result[field] = val
    return result


def extract_with_openai(
    text: str,
    event_label: str,
    event_date: str,
    model: str = "gpt-4o-mini",
    api_key: str = None,
) -> Optional[Dict]:
    """Extract narrative features using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    prompt = EXTRACTION_PROMPT.format(
        event_label=event_label,
        event_date=event_date,
        text=text[:3000],  # Token limit safety
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return _validate_extraction(result)
    except Exception as e:
        print(f"    [OpenAI Error] {e}")
        return None


def extract_with_anthropic(
    text: str,
    event_label: str,
    event_date: str,
    model: str = "claude-sonnet-4-20250514",
    api_key: str = None,
) -> Optional[Dict]:
    """Extract narrative features using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)

    prompt = EXTRACTION_PROMPT.format(
        event_label=event_label,
        event_date=event_date,
        text=text[:3000],
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text
        # Extract JSON from response (Claude sometimes wraps in markdown)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        result = json.loads(content)
        return _validate_extraction(result)
    except Exception as e:
        print(f"    [Anthropic Error] {e}")
        return None


def extract_with_gemini(
    text: str,
    event_label: str,
    event_date: str,
    model: str = "models/gemini-2.5-flash",
    api_key: str = None,
) -> Optional[Dict]:
    """Extract narrative features using Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

    api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    prompt = EXTRACTION_PROMPT.format(
        event_label=event_label,
        event_date=event_date,
        text=text[:3000],
    )

    try:
        response = client.generate_content(prompt)
        content = (response.text or "").strip()
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        result = json.loads(content)
        return _validate_extraction(result)
    except Exception as e:
        print(f"    [Gemini Error] {e}")
        return None


def extract_features(
    text: str,
    event_label: str,
    event_date: str,
    provider: str = "openai",
    use_cache: bool = True,
    **kwargs,
) -> Optional[Dict]:
    """
    Extract narrative features from a single text.

    Parameters
    ----------
    text : str
        The text to analyze.
    event_label : str
        Label of the financial event for context.
    event_date : str
        Date of the event.
    provider : str
        LLM provider: "openai" or "anthropic".
    use_cache : bool
        Whether to use response caching.
    **kwargs
        Additional arguments passed to the provider function.
    """
    model_name = kwargs.get("model", "default")
    cache_key = _get_cache_key(text, event_label, provider=provider, model=model_name)

    if use_cache:
        cached = _load_cached(cache_key)
        if cached is not None:
            return cached

    if provider == "openai":
        result = extract_with_openai(text, event_label, event_date, **kwargs)
    elif provider == "anthropic":
        result = extract_with_anthropic(text, event_label, event_date, **kwargs)
    elif provider == "gemini":
        result = extract_with_gemini(text, event_label, event_date, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if result and use_cache:
        _save_cache(cache_key, result)

    return result


def batch_extract(
    texts: List[str],
    event_label: str,
    event_date: str,
    provider: str = "openai",
    sample_size: int = None,
    sample_indices: List[int] = None,
    rate_limit_delay: float = 0.5,
    return_indices: bool = False,
    **kwargs,
) -> List[Optional[Dict]]:
    """
    Extract narrative features from multiple texts with rate limiting.

    Parameters
    ----------
    texts : list of str
        Texts to analyze.
    event_label : str
        Event context.
    event_date : str
        Event date.
    provider : str
        LLM provider.
    sample_size : int, optional
        If set, randomly sample this many texts instead of processing all.
    rate_limit_delay : float
        Delay between API calls in seconds.
    """
    import random

    if sample_indices is not None:
        indices = sorted(i for i in sample_indices if 0 <= i < len(texts))
        sampled_texts = [texts[i] for i in indices]
    elif sample_size and sample_size < len(texts):
        random.seed(42)
        indices = sorted(random.sample(range(len(texts)), sample_size))
        sampled_texts = [texts[i] for i in indices]
    else:
        sampled_texts = texts
        indices = list(range(len(texts)))

    results = []
    total = len(sampled_texts)
    cached_count = 0
    api_count = 0

    for i, text in enumerate(sampled_texts):
        model_name = kwargs.get("model", "default")
        cache_key = _get_cache_key(text, event_label, provider=provider, model=model_name)
        cached = _load_cached(cache_key)

        if cached is not None:
            results.append(cached)
            cached_count += 1
        else:
            result = extract_features(
                text, event_label, event_date, provider=provider,
                use_cache=True, **kwargs
            )
            results.append(result)
            api_count += 1
            if rate_limit_delay > 0 and i < total - 1:
                time.sleep(rate_limit_delay)

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"    Extracted {i+1}/{total} (cached: {cached_count}, API: {api_count})")

    if return_indices:
        return results, indices
    return results


def aggregate_features(extractions: List[Optional[Dict]]) -> Dict:
    """
    Aggregate extracted features across multiple texts into a summary profile.

    Returns distribution of each categorical feature and dominant values.
    """
    from collections import Counter

    valid = [e for e in extractions if e is not None]
    if not valid:
        return {
            "n_extracted": 0,
            "n_attempted": len(extractions),
        }

    fields = [
        "blame_target", "causal_mechanism", "policy_stance",
        "temporal_orientation", "emotional_register", "narrative_frame"
    ]

    result = {
        "n_extracted": len(valid),
        "n_attempted": len(extractions),
    }

    for field in fields:
        values = [e.get(field, "unknown") for e in valid if e.get(field)]
        counter = Counter(values)
        if counter:
            dominant = counter.most_common(1)[0]
            result[f"{field}_dominant"] = dominant[0]
            result[f"{field}_dominant_share"] = dominant[1] / len(valid)
            result[f"{field}_distribution"] = dict(counter)
            result[f"{field}_entropy"] = _distribution_entropy(counter)
        else:
            result[f"{field}_dominant"] = "unknown"
            result[f"{field}_dominant_share"] = 0.0
            result[f"{field}_distribution"] = {}
            result[f"{field}_entropy"] = 0.0

    # Confidence distribution
    conf_vals = [e.get("confidence", "unknown") for e in valid]
    conf_counter = Counter(conf_vals)
    result["confidence_distribution"] = dict(conf_counter)

    return result


def _distribution_entropy(counter) -> float:
    """Shannon entropy of a Counter distribution."""
    import numpy as np
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counter.values())) / total
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def compare_source_features(
    reddit_features: Dict,
    wsj_features: Dict,
    official_features: Dict = None,
) -> pd.DataFrame:
    """
    Build a comparison table of dominant narrative features across sources.
    """
    fields = [
        "blame_target", "causal_mechanism", "policy_stance",
        "temporal_orientation", "emotional_register", "narrative_frame"
    ]

    rows = []
    for field in fields:
        row = {
            "feature": field,
            "reddit_dominant": reddit_features.get(f"{field}_dominant", "n/a"),
            "reddit_share": reddit_features.get(f"{field}_dominant_share", 0),
            "reddit_entropy": reddit_features.get(f"{field}_entropy", 0),
            "wsj_dominant": wsj_features.get(f"{field}_dominant", "n/a"),
            "wsj_share": wsj_features.get(f"{field}_dominant_share", 0),
            "wsj_entropy": wsj_features.get(f"{field}_entropy", 0),
        }
        if official_features:
            row["official_dominant"] = official_features.get(f"{field}_dominant", "n/a")
            row["official_share"] = official_features.get(f"{field}_dominant_share", 0)
            row["official_entropy"] = official_features.get(f"{field}_entropy", 0)
        rows.append(row)

    return pd.DataFrame(rows)


def check_api_availability() -> Dict[str, bool]:
    """Check which LLM APIs are available."""
    available = {}

    # Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    available["openai"] = bool(openai_key)

    # Check Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    available["anthropic"] = bool(anthropic_key)

    # Check Gemini / Google
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    available["gemini"] = bool(google_key)

    return available
