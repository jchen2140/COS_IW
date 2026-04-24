"""
Sub-Narrative Discovery via BERTopic
=====================================
Identifies thematic sub-narratives within each source (X, WSJ, official)
for each event, enabling comparison of narrative fragmentation and topic
composition across media types.

Methodology:
    - Uses pre-computed sentence-transformer embeddings
    - Applies UMAP dimensionality reduction + HDBSCAN clustering
    - Extracts topic keywords via c-TF-IDF
    - Measures fragmentation as the entropy of the topic distribution
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def _safe_import_bertopic():
    """Import BERTopic and dependencies with informative errors."""
    try:
        from bertopic import BERTopic
        return BERTopic
    except ImportError:
        raise ImportError(
            "BERTopic not installed. Run: pip install bertopic\n"
            "Also requires: pip install hdbscan umap-learn"
        )


def compute_fragmentation_entropy(topic_labels: np.ndarray) -> float:
    """
    Shannon entropy of the topic distribution.
    Higher entropy = more fragmented narrative (many sub-topics).
    Lower entropy = more unified narrative (dominated by 1-2 topics).
    """
    unique, counts = np.unique(topic_labels[topic_labels != -1], return_counts=True)
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def discover_subtopics(
    texts: List[str],
    embeddings: np.ndarray = None,
    min_topic_size: int = 5,
    nr_topics: str = "auto",
    seed: int = 42,
) -> Dict:
    """
    Discover sub-narratives in a set of texts.

    Parameters
    ----------
    texts : list of str
        Documents to cluster.
    embeddings : np.ndarray, optional
        Pre-computed embeddings. If None, BERTopic will compute them.
    min_topic_size : int
        Minimum cluster size for HDBSCAN.
    nr_topics : str or int
        Number of topics to reduce to, or "auto".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - topics: list of int (topic label per document)
        - topic_info: DataFrame with topic_id, count, name, top_words
        - fragmentation: float (Shannon entropy of topic distribution)
        - num_topics: int (excluding outlier topic -1)
        - model: fitted BERTopic model
    """
    BERTopic = _safe_import_bertopic()

    if len(texts) < min_topic_size * 2:
        # Too few documents for meaningful clustering
        return {
            "topics": [-1] * len(texts),
            "topic_info": pd.DataFrame(columns=["Topic", "Count", "Name"]),
            "fragmentation": 0.0,
            "num_topics": 0,
            "model": None,
        }

    from umap import UMAP
    from hdbscan import HDBSCAN

    umap_model = UMAP(
        n_neighbors=min(15, len(texts) - 1),
        n_components=min(5, len(texts) - 2),
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=max(1, min_topic_size // 2),
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        verbose=False,
        calculate_probabilities=False,
    )

    if embeddings is not None:
        topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    else:
        topics, _ = topic_model.fit_transform(texts)

    topics_arr = np.array(topics)
    fragmentation = compute_fragmentation_entropy(topics_arr)
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)

    # Extract topic info
    try:
        topic_info = topic_model.get_topic_info()
    except Exception:
        topic_info = pd.DataFrame(columns=["Topic", "Count", "Name"])

    return {
        "topics": topics,
        "topic_info": topic_info,
        "fragmentation": fragmentation,
        "num_topics": num_topics,
        "model": topic_model,
    }


def compare_fragmentation(
    x_texts: List[str],
    wsj_texts: List[str],
    official_texts: List[str] = None,
    x_embeddings: np.ndarray = None,
    wsj_embeddings: np.ndarray = None,
    official_embeddings: np.ndarray = None,
    min_topic_size: int = 5,
) -> Dict[str, Dict]:
    """
    Compare narrative fragmentation across sources for a single event.

    Returns dict keyed by source name -> subtopic discovery results.
    """
    results = {}

    sources = {
        "x": (x_texts, x_embeddings),
        "wsj": (wsj_texts, wsj_embeddings),
    }
    if official_texts:
        sources["official"] = (official_texts, official_embeddings)

    for source_name, (texts, embs) in sources.items():
        if texts and len(texts) >= 2:
            results[source_name] = discover_subtopics(
                texts, embeddings=embs, min_topic_size=min_topic_size
            )
        else:
            results[source_name] = {
                "topics": [],
                "topic_info": pd.DataFrame(),
                "fragmentation": 0.0,
                "num_topics": 0,
                "model": None,
            }

    return results


def extract_topic_keywords(model, topic_id: int, top_n: int = 5) -> List[str]:
    """Extract top keywords for a given topic."""
    if model is None:
        return []
    try:
        topic_words = model.get_topic(topic_id)
        return [word for word, _ in topic_words[:top_n]]
    except Exception:
        return []


def build_fragmentation_table(
    events: List[str],
    event_labels: List[str],
    x_fragmentation: List[float],
    wsj_fragmentation: List[float],
    official_fragmentation: List[float] = None,
    x_num_topics: List[int] = None,
    wsj_num_topics: List[int] = None,
    official_num_topics: List[int] = None,
) -> pd.DataFrame:
    """
    Build a summary DataFrame of fragmentation scores across events.
    """
    data = {
        "event": events,
        "event_label": event_labels,
        "x_fragmentation": x_fragmentation,
        "wsj_fragmentation": wsj_fragmentation,
    }
    if x_num_topics:
        data["x_num_topics"] = x_num_topics
    if wsj_num_topics:
        data["wsj_num_topics"] = wsj_num_topics
    if official_fragmentation:
        data["official_fragmentation"] = official_fragmentation
    if official_num_topics:
        data["official_num_topics"] = official_num_topics

    return pd.DataFrame(data)
