# Legacy pipeline (WSJ / full narrative stack)

These files supported the earlier **manual WSJ + official docs** workflow and full **`analyze_narrative.py`** stack (BERTopic, lead–lag, Loughran–McDonald, etc.). The current IW focuses on **SENTiVENT + X** and the scripts in the parent `financial_narratives/` folder.

You can still run them from this directory if needed:

```bash
cd ..
./run.sh archive_legacy_pipeline/analyze_narrative.py
```

Moved here:

- `analyze_narrative.py` — main monolithic narrative analysis
- `bertopic_analysis.py`, `lead_lag_analysis.py`, `temporal_drift.py`
- `data_sufficiency_check.py`, `visualize_results.py`
- `loughran_mcdonald.py`
- `x.py` — tiny X API smoke test
- `event_registry.csv` — legacy event registry (WSJ filenames)

The active SENTiVENT registry is **`event_registry_sentivent.csv`** in the parent folder.
