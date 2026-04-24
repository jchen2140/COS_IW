# Causal ambiguity (multi-LLM) summary

- Events analyzed: **12**
- Models: **openai_gpt4omini, anthropic_sonnet, gemini_flash**

## Fleiss' κ (higher = more agreement among models on the same text)

### SENTiVENT
- **primary_cause**: κ = 0.4818 (n_complete = 12)
- **blame_target**: κ = 0.3703 (n_complete = 12)

### X
- **primary_cause**: κ = 0.2469 (n_complete = 12)
- **blame_target**: κ = 0.0110 (n_complete = 12)

## Causal confidence dispersion (std across models, per event)

- Mean std (SENTiVENT): **0.0899**; mean std (X): **0.3093**
- Mann–Whitney U (X vs SENTiVENT std): **p = 0.0009** (exploratory; not a causal claim)

## Files
- Long table: `sentivent_causal_ambiguity_long.csv`
- Fleiss summary: `sentivent_causal_ambiguity_fleiss_summary.csv`
- By semantic regime: `sentivent_causal_ambiguity_by_regime.csv`
