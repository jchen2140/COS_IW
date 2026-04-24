# SENTiVENT evaluation report

Quantitative summary (coverage, semantic alignment, inference, LLM robustness) and pointers to qualitative exports.

## A. Data quality and representativeness

- Raw scraped posts (master): **6,676**
- Cleaned / analysis-ready rows: **1,917**
- Distinct events with ≥1 cleaned post (counts file): **93**
- Total posts summed in counts file: **1,917**

### Event-type distribution (events with ≥1 cleaned post)

- `SecurityValue`: **22** events
- `Macroeconomics`: **10** events
- `Product/Service`: **10** events
- `SalesVolume`: **10** events
- `Rating`: **6** events
- `FinancialReport`: **5** events
- `Legal`: **5** events
- `CSR/Brand`: **4** events
- `Revenue`: **3** events
- `Deal`: **3** events
- `Employment`: **3** events
- `Profit/Loss`: **3** events
- `Investment`: **3** events
- `Dividend`: **2** events
- `Merger/Acquisition`: **2** events
- `Facility`: **1** events
- `Expense`: **1** events

### Semantic cohort (events with usable alignment metrics)

- Events in semantic metrics table: **81**
- Mean X↔SENTiVENT similarity (TF-IDF centroid cosine): **0.110**
- Mean X internal coherence: **0.626**
- Mean alignment gap (coherence − cross-source sim): **0.516**

## B. Semantic alignment (core metrics)

Cosine similarity between the X TF-IDF centroid and the SENTiVENT document vector; coherence = mean cosine similarity of posts to that centroid; gap = coherence − cross-source similarity.

## C. Statistical inference (standardized OLS)


| feature | standardized β |
| --- | --- |
| `tweet_count` | -0.0617 |
| `score_median` | -0.2073 |
| `score_p90` | 0.1764 |
| `word_count_mean` | -0.1768 |
| `unique_authors` | -0.0412 |
| `post_share` | 0.2622 |
| `x_coherence_tfidf` | 0.2940 |

- Strongest positive standardized effects: `x_coherence_tfidf` (0.294), `post_share` (0.262), `score_p90` (0.176)
- Strongest negative standardized effects: `score_median` (-0.207), `word_count_mean` (-0.177), `tweet_count` (-0.062)

## D. Event-type effects (permutation tests)

- Types tested: **6**
- Significant at p < 0.05: **2**

| event_type | effect (type − other) | p-value | n_type |
| --- | --- | --- | --- |
| `Product/Service` | -0.0696 | 0.0010* | 8 |
| `SecurityValue` | 0.0367 | 0.0205* | 20 |
| `SalesVolume` | 0.0408 | 0.0605 | 8 |
| `Legal` | -0.0393 | 0.1450 | 5 |
| `Macroeconomics` | -0.0194 | 0.3612 | 9 |
| `Rating` | 0.0094 | 0.7058 | 6 |

## E. LLM robustness and model disagreement

### Per-model X↔SENTiVENT feature match

| model | overall_match | frame | emotion | stance | blame | causal |
| --- | --- | --- | --- | --- | --- | --- |
| `anthropic_sonnet` | 0.700 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 |
| `gemini_flash` | 0.600 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 |
| `openai_gpt4omini` | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 |

### Cross-model agreement (mean pairwise κ by source)

- **SENTIVENT**: mean κ across event-rows ≈ **0.133** (n=2)
- **X**: mean κ across event-rows ≈ **0.158** (n=2)

## Qualitative exports (this run)

- Case-study event summary: `sentivent_case_study_events.csv`
- Case-study posts: `sentivent_case_study_posts.csv`
- Retrieval audit template (10 × 20): `sentivent_retrieval_audit_sample.csv`

**Regime definitions:** `high_alignment` ≈ upper similarity tertile (non–low-quality); `coherent_divergent` = coherent X cluster, below-median cross-source similarity, above-median gap; `low_quality_noisy` = thin volume (≤33rd pct tweets) or low coherence (≤25th pct).

## Generated files checklist

- This report: `sentivent_evaluation_report.md`
- Advanced summary: `sentivent_advanced_summary.md` (from Day-2 script, if present)
