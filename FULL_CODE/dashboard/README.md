# R2 Returns Fraud Dashboard

Serves your existing `web/index.html` + `smoke.js` interface with **real model outputs** injected from phases 9–12.

## Run

From `FULL_CODE/`:
```bash
streamlit run dashboard/app.py
```
Then open http://localhost:8501

## What gets replaced

| Section | Was (placeholder) | Now (real) |
|---|---|---|
| Live banner (top) | — | Primary score, savings, fairness, Brier, ECE, fraud recall, decision counts |
| Section 05 simulator | JS heuristic | CatBoost + isotonic calibrator; real P(abusive), SHAP signals, tone-matched message |
| Section 06 fairness KPIs | ≤1.25×, ≤1.8% | Real disparity ratios from phase10_fairness_audit.csv |
| Section 07 size-chart table | Fake SKUs | Real PRD-* product IDs + Bayesian fit-issue rates from phase11 |
| Justifications | Template strings | Real feature-driven signals from explain.py |

## Files
```
dashboard/
  app.py          ← main Streamlit app
  web/
    index.html    ← your original HTML (unchanged on disk, patched at runtime)
    smoke.js      ← WebGL particle effect (inlined at runtime so it works in iframe)
  requirements.txt
```

## Thresholds (from config.yaml cost matrix)
- approve → inspect:   P(abusive) ≥ 0.4167
- inspect → escalate:  P(abusive) ≥ 0.9694
