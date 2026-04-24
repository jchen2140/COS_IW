import pandas as pd
import numpy as np
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import collections
from scipy.stats import mannwhitneyu
from datetime import timedelta
from x_api_config import load_local_env

# Local modules
from loughran_mcdonald import compute_lm_densities, compute_lm_tone, compute_modal_strength_ratio, compare_sources
from temporal_drift import full_temporal_analysis
from lead_lag_analysis import full_lead_lag_analysis

# Optional modules (may need extra dependencies)
BERTOPIC_AVAILABLE = False
try:
    from bertopic_analysis import discover_subtopics, compare_fragmentation
    BERTOPIC_AVAILABLE = True
except ImportError:
    print("[Info] BERTopic not available. Install with: pip install bertopic hdbscan umap-learn")

LLM_EXTRACTION_AVAILABLE = False
try:
    from multi_model_extraction import get_available_model_specs, run_multi_model_extraction
    AVAILABLE_MODEL_SPECS = get_available_model_specs()
    if AVAILABLE_MODEL_SPECS:
        LLM_EXTRACTION_AVAILABLE = True
        model_names = ", ".join(spec["name"] for spec in AVAILABLE_MODEL_SPECS)
        print(f"[Info] LLM extraction available via models: {model_names}")
    else:
        print("[Info] LLM extraction: no API keys found. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY.")
except ImportError:
    print("[Info] Multi-model LLM extraction module not loadable.")

# CONFIGURATION
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100

# Path setup (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "x_data_cleaned.csv")
WSJ_DIR = os.path.join(BASE_DIR, "../wsj_articles")
OFFICIAL_DIR = os.path.join(BASE_DIR, "../official_docs")
EVENT_REGISTRY_FILE = os.path.join(BASE_DIR, "event_registry.csv")

# Fallback mapping if registry is missing or incomplete
FALLBACK_EVENT_MAPPING = {
    "svb_collapse_mar23": {"wsj": "wsj_mar23_doc.txt", "official": "official_mar23_doc.txt"},
    "fomc_hike_crisis_mar23": {"wsj": "wsj_mar23_doc.txt", "official": "official_mar23_doc.txt"},
    "fomc_pivot_dec23": {"wsj": "wsj_dec23_doc.txt", "official": "official_dec23_doc.txt"},
    "fomc_rugpull_jan24": {"wsj": "wsj_jan24_doc.txt", "official": "official_jan24_doc.txt"},
    "fomc_jumbo_cut_sep24": {"wsj": "wsj_sep24_doc.txt", "official": "official_sep24_doc.txt"},
    "gme_roaring_kitty_jun24": {"wsj": "wsj_jun24_doc.txt", "official": None}
}

REQUIRED_REGISTRY_COLUMNS = {
    "event",
    "event_date",
    "event_label",
    "event_type",
    "x_query",
    "wsj_file",
    "official_file",
    "analysis_note"
}

load_local_env()

# Narrative Feature Lexicons (Proxies for LLM extraction)
NARRATIVE_LEXICONS = {
    "risk": {
        "risk", "risks", "risky", "loss", "losses", "danger", "threat", "exposure", 
        "fail", "failure", "collapse", "crisis", "panic", "crash", "default", "bankruptcy"
    },
    "uncertainty": {
        "may", "maybe", "could", "might", "perhaps", "possible", "possibly", "uncertain", 
        "uncertainty", "volatile", "volatility", "unclear", "predict", "believe", "anticipate",
        "seem", "appears", "likely", "unlikely", "approximate"
    }
}

def load_text_file(folder, filename):
    if filename is None or (isinstance(filename, float) and pd.isna(filename)):
        return None
    if isinstance(filename, str) and filename.strip() == "":
        return None
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"  [Warning] File not found: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_event_registry():
    if not os.path.exists(EVENT_REGISTRY_FILE):
        print(f"  [Warning] Event registry not found: {EVENT_REGISTRY_FILE}")
        return pd.DataFrame(columns=list(REQUIRED_REGISTRY_COLUMNS))

    registry = pd.read_csv(EVENT_REGISTRY_FILE)
    missing_cols = REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing_cols:
        print(f"  [Warning] Event registry missing columns: {sorted(missing_cols)}")
        for col in missing_cols:
            registry[col] = ""

    registry['event_date'] = pd.to_datetime(registry['event_date'], errors='coerce', utc=True)
    return registry

def split_into_segments(text):
    """Splits text into sentence-like segments for granular analysis."""
    if not text: return []
    # Split by punctuation followed by whitespace
    segments = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in segments if len(s.strip()) > 20] # Filter short fragments

def calculate_lexicon_density(texts, lexicon):
    """Calculates the density of lexicon words in a list of texts."""
    total_words = 0
    lexicon_hits = 0
    matched_words = []
    for t in texts:
        words = re.findall(r'\w+', t.lower())
        total_words += len(words)
        for w in words:
            if w in lexicon:
                lexicon_hits += 1
                matched_words.append(w)
    return (lexicon_hits / total_words) if total_words > 0 else 0.0, collections.Counter(matched_words)

def get_openai_embeddings(texts, model=OPENAI_EMBEDDING_MODEL, batch_size=EMBED_BATCH_SIZE):
    if not texts:
        return np.array([])

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it via configure_llm_apis.py.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([row.embedding for row in response.data])
    return np.array(vectors)


def get_embeddings_and_centroid(embedding_fn, texts):
    if not texts:
        return None, None
    embeddings = embedding_fn(texts)
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    return embeddings, centroid

def bootstrap_mean_ci(values, n_boot=1000, alpha=0.05):
    if values is None or len(values) == 0:
        return None, None

    values = np.asarray(values)
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    low = np.percentile(means, 100 * (alpha / 2))
    high = np.percentile(means, 100 * (1 - alpha / 2))
    return float(low), float(high)

def compute_24h_reaction_share(comment_dates, event_date):
    if event_date is None or pd.isna(event_date):
        return None, None
    if comment_dates is None or len(comment_dates) == 0:
        return 0, 0.0

    # Align timezone awareness before comparison.
    if hasattr(comment_dates, "dt") and str(getattr(comment_dates.dt, "tz", None)) != "None":
        start = pd.Timestamp(event_date)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    else:
        start = pd.Timestamp(event_date).tz_localize(None)
    end = event_date + timedelta(hours=24)
    if isinstance(start, pd.Timestamp):
        end = start + timedelta(hours=24)
    within_window = (comment_dates >= start) & (comment_dates <= end)
    count_24h = int(within_window.sum())
    share_24h = float(count_24h / len(comment_dates))
    return count_24h, share_24h

def main():
    # 1. LOAD THE DATA
    active_input_file = INPUT_FILE
    if not os.path.exists(active_input_file):
        print(f"Error: {INPUT_FILE} not found. Please run clean_data.py first.")
        return

    df = pd.read_csv(active_input_file)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"Loaded {len(df)} source texts from {active_input_file}")

    event_registry = load_event_registry()
    registry_lookup = event_registry.set_index('event').to_dict('index') if not event_registry.empty else {}

    # 2. GENERATE EMBEDDINGS
    print(f"Using OpenAI embeddings model: {OPENAI_EMBEDDING_MODEL}")
    print("Generating embeddings for source texts (this may take a moment)...")
    embeddings = get_openai_embeddings(df['clean_body'].tolist())
    df['embedding'] = list(embeddings)

    # 3. CALCULATE NARRATIVE DIVERGENCE
    print("\n--- NARRATIVE ANALYSIS ---")
    
    results = []

    for event_name, group in df.groupby('event'):
        print(f"\nEvent: {event_name} ({len(group)} posts/replies)")

        event_meta = registry_lookup.get(event_name, {})
        event_date = event_meta.get('event_date')
        event_label = event_meta.get('event_label', event_name)
        event_type = event_meta.get('event_type', '')
        event_note = event_meta.get('analysis_note', '')
        
        # X Data Prep
        x_texts = group['clean_body'].tolist()
        x_embeddings = np.array(group['embedding'].tolist())
        x_centroid = np.mean(x_embeddings, axis=0).reshape(1, -1)
        
        # A. Internal Coherence (X Centroid)
        coherence_scores = cosine_similarity(x_embeddings, x_centroid).flatten()
        mean_coherence = np.mean(coherence_scores)
        score_weights = np.clip(group['score'].fillna(0).to_numpy(), a_min=0, a_max=None) + 1
        weighted_coherence = np.average(coherence_scores, weights=score_weights)
        print(f"  X Coherence (Self-Similarity): {mean_coherence:.4f}")

        comments_24h = None
        share_24h = None
        if 'date' in group.columns:
            comments_24h, share_24h = compute_24h_reaction_share(group['date'], event_date)
            if share_24h is not None:
                print(f"  24h Reaction Share: {share_24h:.2%} ({comments_24h}/{len(group)})")

        # Narrative Features (X)
        x_risk, r_risk_words = calculate_lexicon_density(x_texts, NARRATIVE_LEXICONS["risk"])
        x_uncertainty, r_unc_words = calculate_lexicon_density(x_texts, NARRATIVE_LEXICONS["uncertainty"])
        
        print(f"    [Debug] Top X Risk Words: {r_risk_words.most_common(3)}")

        # B. External Divergence (vs WSJ / Official)
        wsj_file = event_meta.get('wsj_file')
        official_file = event_meta.get('official_file')
        if wsj_file is None or (isinstance(wsj_file, float) and pd.isna(wsj_file)) or wsj_file == "":
            wsj_file = FALLBACK_EVENT_MAPPING.get(event_name, {}).get('wsj')
        if official_file is None or (isinstance(official_file, float) and pd.isna(official_file)) or official_file == "":
            official_file = FALLBACK_EVENT_MAPPING.get(event_name, {}).get('official')
        wsj_text = load_text_file(WSJ_DIR, wsj_file)
        official_text = load_text_file(OFFICIAL_DIR, official_file)
        
        # Process WSJ
        wsj_coherence = None
        wsj_risk = None
        wsj_uncertainty = None
        w_risk_words = None
        w_unc_words = None
        sim_wsj = None
        wsj_comment_sims = None
        wsj_ci_low = None
        wsj_ci_high = None
        
        wsj_segments = split_into_segments(wsj_text)
        if wsj_segments:
            wsj_emb, wsj_cent = get_embeddings_and_centroid(get_openai_embeddings, wsj_segments)
            wsj_coherence = np.mean(cosine_similarity(wsj_emb, wsj_cent))
            wsj_risk, w_risk_words = calculate_lexicon_density(wsj_segments, NARRATIVE_LEXICONS["risk"])
            wsj_uncertainty, w_unc_words = calculate_lexicon_density(wsj_segments, NARRATIVE_LEXICONS["uncertainty"])
            print(f"    [Debug] Top WSJ Risk Words: {w_risk_words.most_common(3)}")
            wsj_comment_sims = cosine_similarity(x_embeddings, wsj_cent).flatten()
            sim_wsj = float(np.mean(wsj_comment_sims))
            wsj_ci_low, wsj_ci_high = bootstrap_mean_ci(wsj_comment_sims)
            print(f"  WSJ Coherence: {wsj_coherence:.4f} | Similarity to X: {sim_wsj:.4f}")

        # Process Official
        official_coherence = None
        official_risk = None
        official_uncertainty = None
        o_risk_words = None
        o_unc_words = None
        sim_official = None
        off_comment_sims = None
        off_ci_low = None
        off_ci_high = None

        official_segments = split_into_segments(official_text)
        if official_segments:
            off_emb, off_cent = get_embeddings_and_centroid(get_openai_embeddings, official_segments)
            official_coherence = np.mean(cosine_similarity(off_emb, off_cent))
            official_risk, o_risk_words = calculate_lexicon_density(official_segments, NARRATIVE_LEXICONS["risk"])
            official_uncertainty, o_unc_words = calculate_lexicon_density(official_segments, NARRATIVE_LEXICONS["uncertainty"])
            off_comment_sims = cosine_similarity(x_embeddings, off_cent).flatten()
            sim_official = float(np.mean(off_comment_sims))
            off_ci_low, off_ci_high = bootstrap_mean_ci(off_comment_sims)
            print(f"  Official Coherence: {official_coherence:.4f} | Similarity to X: {sim_official:.4f}")

        # C. Statistical Test (Task 3)
        # If we have both, is X significantly closer to one than the other?
        p_val = None
        winner = None

        if wsj_segments and official_segments:
            # Compare distribution of distances from X posts/replies to WSJ centroid vs Official centroid
            dist_to_wsj = wsj_comment_sims
            dist_to_official = off_comment_sims
            
            stat, p_val = mannwhitneyu(dist_to_wsj, dist_to_official, alternative='two-sided')
            print(f"  Stats Test (WSJ vs Official): p-value={p_val:.4e}")
            if p_val < 0.05:
                winner = "WSJ" if np.mean(dist_to_wsj) > np.mean(dist_to_official) else "Official"
                print(f"  -> Statistically significant alignment with: {winner}")

        risk_gap_wsj = (x_risk - wsj_risk) if wsj_risk is not None else None
        uncertainty_gap_wsj = (x_uncertainty - wsj_uncertainty) if wsj_uncertainty is not None else None
        risk_gap_official = (x_risk - official_risk) if official_risk is not None else None
        uncertainty_gap_official = (x_uncertainty - official_uncertainty) if official_uncertainty is not None else None
        alignment_delta = (sim_wsj - sim_official) if (sim_wsj is not None and sim_official is not None) else None

        # ================================================================
        # D. LOUGHRAN-MCDONALD FINANCIAL LEXICON ANALYSIS
        # ================================================================
        print(f"  [LM Lexicon] Computing Loughran-McDonald financial sentiment...")
        x_lm = compute_lm_densities(x_texts)
        x_lm_tone = compute_lm_tone(x_texts)
        x_modal_strength = compute_modal_strength_ratio(x_texts)

        wsj_lm_tone = None
        wsj_modal_strength = None
        official_lm_tone = None
        official_modal_strength = None

        if wsj_segments:
            wsj_lm = compute_lm_densities(wsj_segments)
            wsj_lm_tone = compute_lm_tone(wsj_segments)
            wsj_modal_strength = compute_modal_strength_ratio(wsj_segments)
            print(f"    X tone: {x_lm_tone:.4f} | WSJ tone: {wsj_lm_tone:.4f}")

        if official_segments:
            official_lm = compute_lm_densities(official_segments)
            official_lm_tone = compute_lm_tone(official_segments)
            official_modal_strength = compute_modal_strength_ratio(official_segments)

        # ================================================================
        # E. TEMPORAL DRIFT ANALYSIS
        # ================================================================
        temporal_result = {}
        wsj_convergence = None
        official_convergence = None
        narrative_halflife = None

        if 'date' in group.columns and event_date is not None and pd.notna(event_date):
            print(f"  [Temporal] Computing narrative drift trajectories...")
            temporal_result = full_temporal_analysis(
                dates=group['date'],
                embeddings=x_embeddings,
                event_date=event_date,
                wsj_centroid=wsj_cent if wsj_segments else None,
                official_centroid=off_cent if official_segments else None,
                min_comments=1,
            )
            wsj_convergence = temporal_result.get('wsj_convergence')
            official_convergence = temporal_result.get('official_convergence')
            narrative_halflife = temporal_result.get('narrative_halflife')

            if wsj_convergence is not None:
                direction = "converging toward" if wsj_convergence > 0 else "diverging from"
                print(f"    WSJ convergence: {wsj_convergence:.3f} ({direction} WSJ)")
            if narrative_halflife is not None:
                print(f"    Narrative half-life: window {narrative_halflife}")

        # ================================================================
        # F. CROSS-SOURCE LEAD-LAG ANALYSIS
        # ================================================================
        lead_lag_result = {}
        wsj_peak_lag = None
        wsj_peak_corr = None
        wsj_leadlag_sig = None

        if 'date' in group.columns and event_date is not None and pd.notna(event_date):
            print(f"  [Lead-Lag] Computing cross-correlation analysis...")
            lead_lag_result = full_lead_lag_analysis(
                dates=group['date'],
                embeddings=x_embeddings,
                event_date=event_date,
                wsj_centroid=wsj_cent if wsj_segments else None,
                official_centroid=off_cent if official_segments else None,
                window_days=7,
                max_lag=3,
            )
            wsj_ll = lead_lag_result.get('wsj_lead_lag', {})
            wsj_peak_lag = wsj_ll.get('observed_peak_lag')
            wsj_peak_corr = wsj_ll.get('observed_peak_corr')
            wsj_leadlag_sig = wsj_ll.get('significant', False)
            interp = wsj_ll.get('interpretation', '')
            if interp:
                print(f"    {interp}")

        # ================================================================
        # G. BERTOPIC SUB-NARRATIVE DISCOVERY
        # ================================================================
        x_fragmentation = None
        x_num_topics = None
        wsj_fragmentation = None
        wsj_num_topics = None

        if BERTOPIC_AVAILABLE:
            print(f"  [BERTopic] Discovering sub-narratives...")
            try:
                x_subtopic = discover_subtopics(
                    x_texts, embeddings=x_embeddings,
                    min_topic_size=max(3, len(x_texts) // 20)
                )
                x_fragmentation = x_subtopic['fragmentation']
                x_num_topics = x_subtopic['num_topics']
                print(f"    X: {x_num_topics} sub-narratives, "
                      f"fragmentation={x_fragmentation:.3f}")

                if wsj_segments and len(wsj_segments) >= 6:
                    wsj_subtopic = discover_subtopics(
                        wsj_segments, embeddings=wsj_emb,
                        min_topic_size=max(3, len(wsj_segments) // 8)
                    )
                    wsj_fragmentation = wsj_subtopic['fragmentation']
                    wsj_num_topics = wsj_subtopic['num_topics']
                    print(f"    WSJ: {wsj_num_topics} sub-narratives, "
                          f"fragmentation={wsj_fragmentation:.3f}")
            except Exception as e:
                print(f"    [BERTopic Error] {e}")

        # ================================================================
        # H. LLM NARRATIVE FEATURE EXTRACTION (if API available)
        # ================================================================
        llm_blame_dominant = None
        llm_causal_dominant = None
        llm_stance_dominant = None
        llm_emotion_dominant = None
        llm_frame_dominant = None
        llm_n_extracted = 0
        llm_models_used = ""
        llm_model_count = 0
        llm_agreement_overall = None
        llm_agreement_stance = None
        llm_agreement_emotion = None
        llm_agreement_frame = None

        if LLM_EXTRACTION_AVAILABLE:
            print(f"  [LLM] Extracting structured narrative features across models...")
            try:
                sample_n = min(30, len(x_texts))
                mm_result = run_multi_model_extraction(
                    texts=x_texts,
                    event_label=event_label,
                    event_date=str(event_date)[:10] if (event_date is not None and pd.notna(event_date)) else "",
                    sample_size=sample_n,
                    rate_limit_delay=0.3,
                )
                agg = mm_result.get("consensus_aggregate", {})
                agreement = mm_result.get("agreement", {})

                llm_n_extracted = agg.get('n_extracted', 0)
                llm_blame_dominant = agg.get('blame_target_dominant')
                llm_causal_dominant = agg.get('causal_mechanism_dominant')
                llm_stance_dominant = agg.get('policy_stance_dominant')
                llm_emotion_dominant = agg.get('emotional_register_dominant')
                llm_frame_dominant = agg.get('narrative_frame_dominant')
                llm_models_used = "|".join(mm_result.get("models_used", []))
                llm_model_count = len(mm_result.get("models_used", []))
                llm_agreement_overall = agreement.get("overall_pairwise_kappa_mean")
                feature_agreement = agreement.get("feature_agreement", {})
                llm_agreement_stance = (feature_agreement.get("policy_stance") or {}).get("pairwise_kappa_mean")
                llm_agreement_emotion = (feature_agreement.get("emotional_register") or {}).get("pairwise_kappa_mean")
                llm_agreement_frame = (feature_agreement.get("narrative_frame") or {}).get("pairwise_kappa_mean")

                print(f"    Extracted {llm_n_extracted}/{sample_n} comments")
                if llm_blame_dominant:
                    print(f"    Blame: {llm_blame_dominant} | Frame: {llm_frame_dominant} | "
                          f"Emotion: {llm_emotion_dominant}")
                if llm_agreement_overall is not None:
                    print(f"    Cross-model agreement (mean pairwise kappa): {llm_agreement_overall:.3f}")
            except Exception as e:
                print(f"    [LLM Error] {e}")

        # ================================================================
        # BUILD RESULT ROW
        # ================================================================
        results.append({
            "event": event_name,
            "event_label": event_label,
            "event_type": event_type,
            "event_date": event_date,
            "analysis_note": event_note,
            "num_comments": len(group),
            "x_coherence": mean_coherence,
            "x_weighted_coherence": weighted_coherence,
            "comments_within_24h": comments_24h,
            "share_within_24h": share_24h,
            "wsj_coherence": wsj_coherence,
            "official_coherence": official_coherence,
            "x_risk": x_risk,
            "wsj_risk": wsj_risk,
            "official_risk": official_risk,
            "x_uncertainty": x_uncertainty,
            "wsj_uncertainty": wsj_uncertainty,
            "official_uncertainty": official_uncertainty,
            # Similarity & alignment
            "similarity_wsj": sim_wsj,
            "similarity_official": sim_official,
            "similarity_wsj_ci_low": wsj_ci_low,
            "similarity_wsj_ci_high": wsj_ci_high,
            "similarity_official_ci_low": off_ci_low,
            "similarity_official_ci_high": off_ci_high,
            "alignment_delta_wsj_minus_official": alignment_delta,
            "gap_wsj": (1 - sim_wsj) if sim_wsj is not None else None,
            "gap_official": (1 - sim_official) if sim_official is not None else None,
            "risk_gap_wsj": risk_gap_wsj,
            "uncertainty_gap_wsj": uncertainty_gap_wsj,
            "risk_gap_official": risk_gap_official,
            "uncertainty_gap_official": uncertainty_gap_official,
            "wsj_segments_count": len(wsj_segments),
            "official_segments_count": len(official_segments),
            "stat_p_value": p_val,
            "stat_winner": winner,
            # Loughran-McDonald financial sentiment
            "lm_x_tone": x_lm_tone,
            "lm_wsj_tone": wsj_lm_tone,
            "lm_official_tone": official_lm_tone,
            "lm_tone_gap_wsj": (x_lm_tone - wsj_lm_tone) if wsj_lm_tone is not None else None,
            "lm_tone_gap_official": (x_lm_tone - official_lm_tone) if official_lm_tone is not None else None,
            "lm_x_modal_strength": x_modal_strength,
            "lm_wsj_modal_strength": wsj_modal_strength,
            "lm_official_modal_strength": official_modal_strength,
            "lm_x_negative_density": x_lm['negative']['density'],
            "lm_x_positive_density": x_lm['positive']['density'],
            "lm_x_uncertainty_density": x_lm['uncertainty']['density'],
            "lm_x_litigious_density": x_lm['litigious']['density'],
            "lm_x_constraining_density": x_lm['constraining']['density'],
            # Temporal drift analysis
            "wsj_convergence": wsj_convergence,
            "official_convergence": official_convergence,
            "narrative_halflife": narrative_halflife,
            # Lead-lag analysis
            "wsj_peak_lag": wsj_peak_lag,
            "wsj_peak_corr": wsj_peak_corr,
            "wsj_leadlag_significant": wsj_leadlag_sig,
            # BERTopic sub-narratives
            "x_fragmentation": x_fragmentation,
            "x_num_topics": x_num_topics,
            "wsj_fragmentation": wsj_fragmentation,
            "wsj_num_topics": wsj_num_topics,
            # LLM narrative features
            "llm_n_extracted": llm_n_extracted,
            "llm_blame_dominant": llm_blame_dominant,
            "llm_causal_dominant": llm_causal_dominant,
            "llm_stance_dominant": llm_stance_dominant,
            "llm_emotion_dominant": llm_emotion_dominant,
            "llm_frame_dominant": llm_frame_dominant,
            "llm_models_used": llm_models_used,
            "llm_model_count": llm_model_count,
            "llm_agreement_overall_pairwise_kappa_mean": llm_agreement_overall,
            "llm_agreement_policy_stance_pairwise_kappa": llm_agreement_stance,
            "llm_agreement_emotional_register_pairwise_kappa": llm_agreement_emotion,
            "llm_agreement_narrative_frame_pairwise_kappa": llm_agreement_frame,
            "x_risk_words": str(r_risk_words.most_common(5)),
            "x_unc_words": str(r_unc_words.most_common(5)),
            "wsj_risk_words": str(w_risk_words.most_common(5)) if w_risk_words else "",
            "wsj_unc_words": str(w_unc_words.most_common(5)) if w_unc_words else "",
            "official_risk_words": str(o_risk_words.most_common(5)) if o_risk_words else "",
            "official_unc_words": str(o_unc_words.most_common(5)) if o_unc_words else ""
        })

        # Save per-event temporal trajectory if available
        if temporal_result.get('wsj_trajectory') is not None and not temporal_result['wsj_trajectory'].empty:
            traj_path = os.path.join(BASE_DIR, f"temporal_trajectory_{event_name}.csv")
            temporal_result['wsj_trajectory'].to_csv(traj_path, index=False)

    if not event_registry.empty:
        observed = set(df['event'].unique())
        planned = set(event_registry['event'].unique())
        missing_from_x = sorted(planned - observed)
        if missing_from_x:
            print("\nPlanned events missing in X data:")
            for ev in missing_from_x:
                print(f"  - {ev}")

    output_path = os.path.join(BASE_DIR, "narrative_analysis_results.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nAnalysis complete. Results saved to: {output_path}")

    # Print novelty summary
    print("\n--- NOVELTY FEATURES SUMMARY ---")
    print(f"  Loughran-McDonald financial lexicon: ACTIVE (7 sentiment categories)")
    print(f"  Temporal drift analysis: {'ACTIVE' if any(r.get('wsj_convergence') is not None for r in results) else 'NO DATA'}")
    print(f"  Lead-lag analysis: {'ACTIVE' if any(r.get('wsj_peak_lag') is not None for r in results) else 'NO DATA'}")
    print(f"  BERTopic sub-narratives: {'ACTIVE' if BERTOPIC_AVAILABLE else 'NOT INSTALLED'}")
    print(f"  LLM feature extraction: {'ACTIVE' if LLM_EXTRACTION_AVAILABLE else 'NO API KEY'}")

if __name__ == "__main__":
    main()