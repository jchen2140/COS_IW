# Advanced SENTiVENT Analysis Summary

- Events analyzed: **81**
- Mean X↔SENTiVENT similarity: **0.110**

## Regression (Standardized OLS)
- R²: **0.451**
- Top standardized effects:
  - `x_coherence_tfidf`: 0.294
  - `post_share`: 0.262
  - `score_median`: -0.207
  - `word_count_mean`: -0.177
  - `score_p90`: 0.176

## Event-Type Permutation Tests
- Types tested: **6**
- Significant type effects (p < 0.05): **2**
- Significant types:
  - `SecurityValue`: effect=0.037, p=0.0205
  - `Product/Service`: effect=-0.070, p=0.0010

## Files
- Feature table: `sentivent_advanced_event_features.csv`
- Coefficients: `sentivent_advanced_regression_coefficients.csv`
- Type tests: `sentivent_advanced_type_permutation_tests.csv`
- Plots: `plots_sentivent_advanced/advanced_regression_coefficients.png`, `plots_sentivent_advanced/advanced_type_effects_permutation.png`