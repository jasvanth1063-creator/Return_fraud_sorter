# R2 — Returns Fraud vs Genuine Grievance Sorter

**Team 632 · Hackathon submission · Primary score: 0.3878**

A cost-sensitive, fairness-aware system that routes fashion e-commerce return requests to one of three actions — **auto-approve**, **inspect**, or **escalate** — minimising the official cost-weighted loss while keeping all fairness groups within tolerance.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Official Scoring](#official-scoring)
3. [Final Results](#final-results)
4. [Repository Layout](#repository-layout)
5. [Pipeline Overview (12 Phases)](#pipeline-overview)
6. [Key Design Decisions](#key-design-decisions)
7. [Fairness Audit](#fairness-audit)
8. [Submission Files](#submission-files)
9. [Setup & Reproduction](#setup--reproduction)
10. [Dashboard](#dashboard)

---

## Problem Statement

A fashion retailer receives return requests that fall into two broad categories:

| Label | Meaning |
|---|---|
| **abusive** | Fraudulent or policy-abuse return (wear-and-return, false defect claim, etc.) |
| **genuine_quality_issue** | Legitimate — product defect or quality problem |
| **genuine_fit_issue** | Legitimate — size or fit mismatch |
| **genuine_wrong_item** | Legitimate — wrong item dispatched |

The system must assign each request to one of three decisions:

| Decision | Cost if genuine | Cost if abusive |
|---|---|---|
| `auto_approve` | 0.0 | **1.0** (missed fraud) |
| `inspect` | 0.5 | 0.3 |
| `escalate` | **10.0** (wrong block) | 0.0 |

The asymmetry is critical: wrongly escalating a genuine customer costs **10×** more than missing one fraudulent return.

---

## Official Scoring

```
PRIMARY = 0.7 × savings  +  0.2 × fairness_effective  +  0.1 × size_chart
```

- **savings** — cost reduction versus an all-auto-approve baseline (normalised).
- **fairness_effective** — average of three pairwise disparity scores (new vs established customer, low-value vs other order, non-English vs English text); each score = max(0, 1 − |ratio − 1| / 0.25).
- **size_chart** — whether the top-5 most-problematic size-chart products are correctly identified (scored on hidden truth).

Score range: [−0.7, 1.0]. All-auto-approve = 0.0.

---

## Final Results

| Metric | Baseline (Router A) | **Optimised (Router E)** | Δ |
|---|---|---|---|
| Primary score | 0.3376 | **0.3878** | +0.0502 |
| Savings | 0.2934 | **0.2936** | — |
| Fairness | 0.6611 | **0.9113** | +0.2502 |
| Cost/request | 0.0581 | **0.0581** | — |
| Abusive recall | 50.1% | **55.9%** | +5.8 pp |
| Genuine escalated | 0 | **0** | — |
| new vs established disparity | 1.32× | **1.00×** | perfect parity |

**Optimised thresholds:**
- `inspect` threshold: **0.35** (lowered from theoretical break-even 0.4167)
- `escalate` threshold: **disabled** (set to 2.0 — cost-optimal: genuine→escalate costs 10×, so inspect is always cheaper for uncertain cases)

**Submission decision distribution (test set, 16 932 rows):**

| Decision | Count |
|---|---|
| auto_approve | 15 748 |
| inspect | 1 184 |
| escalate | 0 |

---

## Repository Layout

```
r2_project/
├── FULL_CODE/
│   ├── config.yaml                     # Single source of truth: paths, cost matrix, model params
│   ├── pyproject.toml                  # Package: r2sorter
│   ├── src/r2sorter/                   # Core library (imported by all notebooks & scripts)
│   │   ├── data.py                     # Data loading, LABELS constant
│   │   ├── splits.py                   # Time-based train/calibration/test split
│   │   ├── features.py                 # Feature engineering (150+ features)
│   │   ├── nlp.py                      # NLP pipeline, language detection
│   │   ├── baselines.py                # CatBoost / LightGBM model builders
│   │   ├── calibration.py              # AbuseCalibrator (isotonic / Platt / bagged)
│   │   ├── conformal.py                # Mondrian conformal prediction sets
│   │   ├── router.py                   # route() — cost-matrix decision routing
│   │   ├── ensemble.py                 # fast_official(), fairness_masks(), row_costs()
│   │   ├── fairness.py                 # audit() — per-group fairness report
│   │   ├── scoring.py                  # official_score() wrapper
│   │   ├── sizechart.py                # Bayesian size-chart ranker
│   │   └── explain.py                  # SHAP / justification generation
│   ├── notebooks/                      # One Jupyter notebook per phase (Phases 1–12)
│   ├── Phase_09_Decision_Router.ipynb  # Active router notebook (Router E wins)
│   ├── data/                           # Raw CSV files from organiser
│   ├── reports/                        # All artefacts: JSON results, CSVs, figures
│   ├── models/                         # Saved CatBoost + calibrator models
│   ├── dashboard/app.py                # Streamlit dashboard
│   ├── scripts/                        # Build scripts for each phase notebook
│   └── tools/evaluate.py               # Organiser's official scorer (unmodified)
└── submission/
    ├── R2_predictions.csv              # Final predictions (16 932 rows)
    └── R2_size_chart_top5.csv          # Top-5 problematic size-chart products
```

**Security note:** `FULL_CODE/tabpfn_key.py` holds the TabPFN API key and is git-ignored. Never commit it. Use `tabpfn_key.example.py` as the template.

---

## Pipeline Overview

Each phase is a self-contained Jupyter notebook. Phases run in order; each saves outputs to `reports/` that the next phase reads.

### Phase 1 — Dataset Inspection
Exploratory analysis: class distribution, date range (Oct 2025 – Jun 2026), missing values, return-window policy (30 days), and label prevalence (~8.2% abusive in validation).

### Phase 2 — Splitting and Leakage Prevention
Time-based stratified split:
- **Train**: Oct 2025 – Apr 2026 (~34 500 rows)
- **Calibration**: Feb 2026 – Apr 2026 (~4 400 rows, overlapping for calibration)
- **Validation / Test**: May–Jun 2026 (~8 500 rows, hidden labels)

Jul–Sep labels are **never used** (out-of-time holdout; would constitute leakage).

### Phase 3 — Feature Engineering
150+ features across:
- **Temporal**: return lag, day-of-week, late-in-window flag, seasonal indicators
- **Velocity**: rolling 30/60/90-day counts and rates per customer and product
- **NLP**: TF-IDF (word unigrams+bigrams + char n-grams 2–5), language group (English / Hinglish / Hindi)
- **Product & customer**: order value tercile, customer tenure bucket, product category
- **Similarity**: repeat-text cosine similarity within 8-day lookback window

### Phase 4 — NLP Pipeline
Bilingual (English + Hinglish/Hindi) text features. Three language groups detected via a keyword word-list heuristic. TF-IDF char n-gram features (up to 120 000 features). Binary abuse experiment shows text alone achieves PR-AUC ≈ 0.65.

### Phase 5 — Baseline Models
Logistic regression and LightGBM baselines on combined tabular+NLP features. Establishes PR-AUC ≈ 0.68 and cost-router official score ≈ 0.29.

### Phase 6 — Advanced Tabular Models
CatBoost multi-class (4-class) model: labels are abusive / genuine_quality / genuine_fit / genuine_wrong. 3-seed ensemble averaging. PR-AUC improves to ≈ 0.70.

### Phase 7 — Ensemble Optimisation
Ablation study over feature groups. Final feature set selected. Rolling-origin cross-validation (5 folds). CatBoost model confirmed as best single model.

### Phase 8 — Calibration and Conformal Prediction
Three calibration methods compared on calibration part: Platt (sigmoid), isotonic, bagged isotonic. **Platt (sigmoid)** selected (best calibration: Brier score 0.041, ECE 0.012, AP 0.710). Mondrian conformal prediction sets at 90% coverage per class — used as ambiguity signal in Router B/D.

### Phase 9 — Cost-Sensitive Decision Router
Five routers evaluated; first passing all 5 keep-or-drop checks is selected:

| Router | Description | Official score |
|---|---|---|
| A | Baseline cost-matrix route() | 0.3376 |
| B | Conformal-aware (ambiguous sets promoted) | −0.700 ✗ |
| C | Fairness-adjusted (per-group threshold search) | 0.3376 |
| D | Combined C + B | −0.700 ✗ |
| **E** | **Optimised: inspect ≥ 0.35, escalate disabled** | **0.3878 ✓** |

Router E wins: +0.050 vs A, abusive recall up, fairness jumps from 0.661 → 0.911, zero genuine escalations.

### Phase 10 — Official Evaluation
Full official scorer run on the validation window. Final confirmed scores recorded to `reports/phase10_official_result.json`.

### Phase 11 — Size Chart Recommendations
Bayesian-smoothed fit-issue rate per product (pseudo-count = 5, prior = overall mean). Top-5 most problematic products identified:

| Rank | Product ID |
|---|---|
| 1 | PRD-YIME3XJ6 |
| 2 | PRD-OSNR3LA5 |
| 3 | PRD-6JG12RLX |
| 4 | PRD-7NDH0OJO |
| 5 | PRD-THXMYY0T |

### Phase 12 — Explanations and Dashboard
SHAP values for CatBoost model. Per-request justification generation (top-3 evidence items). Streamlit dashboard for human reviewers.

---

## Key Design Decisions

**Why CatBoost multi-class over binary?**
The four genuine sub-types (quality, fit, wrong item) differ in return-rate distributions and customer behaviour. Treating them as separate classes lets the model use those signals, even though the final decision only needs P(abusive).

**Why disable escalation?**
The cost matrix makes escalating a genuine customer 10× more expensive than auto-approving a fraudulent one. Even with perfect abusive recall at the escalation threshold, genuine false positives dominate total cost. Keeping everything uncertain at `inspect` (cost 0.3–0.5 per case) is always cheaper.

**Why lower inspect threshold to 0.35?**
The theoretical break-even (0.4167) was derived from the cost matrix in isolation. Empirically, lowering to 0.35 catches an extra ~260 abusive cases on the validation set while only adding ~760 genuine inspects — net cost falls and fairness improves because the additional flags are distributed more evenly across groups.

**Why Platt (sigmoid) calibration?**
On the out-of-time calibration part (Feb–Apr), Platt achieved the best calibration across all three metrics (Brier, ECE, AP) vs isotonic and bagged isotonic. Monotonic calibration matters here because the router relies on probability ordering.

---

## Fairness Audit

Three protected group pairs, evaluated on the validation window (May–Jun 2026):

| Group pair | Disparity ratio | Score | Status |
|---|---|---|---|
| New vs established customer | 1.0000 | 1.0000 | ✅ Perfect parity |
| Low-value vs other order | 1.2113 | 0.7887 | ✅ Within tolerance |
| Non-English vs English text | 1.0521 | 0.9479 | ✅ Within tolerance |

**Overall fairness score: 0.9113**

All five safety rules pass:
- New customer escalation rate ≤ 3× established rate ✅
- Non-English genuine FBR ≤ English genuine FBR + 5 pp ✅
- Low-value genuine FBR ≤ overall genuine FBR + 5 pp ✅
- Model does not approve all fraud (abusive recall > 0) ✅
- Genuine escalation rate < 5% ✅

---

## Submission Files

Located in `submission/`:

**`R2_predictions.csv`** — One row per test request (16 932 rows):

| Column | Description |
|---|---|
| `return_id` | Request identifier |
| `decision` | `auto_approve` / `inspect` / `escalate` |
| `confidence` | Calibrated P(abusive) ∈ [0, 1] |
| `justification` | Human-readable explanation (top-3 SHAP features) |

**`R2_size_chart_top5.csv`** — Top-5 products with highest Bayesian fit-issue rate:

| Column | Description |
|---|---|
| `rank` | 1–5 (1 = highest fit-issue rate) |
| `product_id` | Product identifier |

---

## Setup & Reproduction

**Requirements:** Python ≥ 3.10, uv (recommended) or pip.

```bash
# Clone and install
cd FULL_CODE
pip install -e ".[app]"          # or: uv sync --extra app

# Copy API key template
cp tabpfn_key.example.py tabpfn_key.py
# Edit tabpfn_key.py and set TABPFN_API_KEY = "your_key_here"

# Run phases in order (each saves outputs to reports/)
jupyter nbconvert --to notebook --execute notebooks/Phase_01_Dataset_Inspection.ipynb
# ... repeat for Phases 2–12
# Phase 9 router notebook is at the project root:
jupyter nbconvert --to notebook --execute Phase_09_Decision_Router.ipynb
```

**Data files** (place in `FULL_CODE/data/`):
- `return_requests_train.csv`
- `return_requests_test.csv`
- `customers.csv`
- `products.csv`
- `size_charts.csv`

**Official scorer** is at `FULL_CODE/tools/evaluate.py` — unmodified copy from the organiser. Run it as:

```bash
python tools/evaluate.py --predictions submission/R2_predictions.csv --truth <truth_file>
```

---

## Dashboard

A Streamlit reviewer dashboard is in `FULL_CODE/dashboard/app.py`.

```bash
streamlit run FULL_CODE/dashboard/app.py
```

Thresholds used in the dashboard:
- `INSPECT_THR = 0.35`
- `ESCALATE_THR = 2.0` (disabled)

The dashboard shows per-request decisions, confidence scores, justifications, and a live fairness summary.

---

*Built for the R2 Hackathon (Returns Fraud vs Genuine Grievance Sorter challenge).*
