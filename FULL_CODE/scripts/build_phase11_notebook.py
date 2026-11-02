"""Build Phase_11_SizeChart_Recommendations.ipynb."""

from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "Phase_11_SizeChart_Recommendations.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text.strip("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text.strip("\n")],
    }


cells = []

# ── 0  Title ─────────────────────────────────────────────────────────────────
cells.append(md("""
# Phase 11 — Size-Chart Top-5 Fit-Return Products

**Goal:** Identify the **five products** most associated with fit-issue returns and produce
`R2_size_chart_top5.csv` — the file `tools/evaluate.py` reads as the size-chart component
(weight 0.1 in the primary score).

**Method:**
1. Aggregate fit-issue return counts per product on the training window (Jan–Apr).
2. Apply Bayesian smoothing to handle products with few returns.
3. Rank by `smoothed_fit_rate × log1p(fit_returns)` to balance rate with volume.
4. Validate the output matches the format `evaluate.py` requires.
5. Re-run the official scorer with the size-chart file attached.

**No hidden Jul–Sep labels touched.**
"""))

# ── 1  Imports ────────────────────────────────────────────────────────────────
cells.append(md("## 1 · Imports & paths"))
cells.append(code(r"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

from r2sorter.config import load_config
from r2sorter.data import load_all
from r2sorter.protocol import build_protocol
from r2sorter.scoring import official_score

CFG  = load_config(ROOT / "config.yaml")
DATA = load_all(CFG)

FIG_DIR = ROOT / "reports" / "figures"
REP_DIR = ROOT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

train    = DATA["train"]
products = DATA["products"]
size_charts = DATA["size_charts"]

print(f"Train rows:    {len(train):,}")
print(f"Products:      {len(products):,}")
print(f"Size-chart entries: {len(size_charts):,}")
print()
print("Label distribution in train:")
print(train["label"].value_counts().to_string())
"""))

# ── 2  Per-product fit-issue analysis ────────────────────────────────────────
cells.append(md("## 2 · Per-product fit-issue analysis"))
cells.append(code(r"""
FIT_LABELS = ["genuine_fit_issue"]

by_prod = train.groupby("product_id").agg(
    fit_returns=("label", lambda x: x.isin(FIT_LABELS).sum()),
    total_returns=("label", "count"),
).reset_index()
by_prod["fit_rate"] = by_prod["fit_returns"] / by_prod["total_returns"]

# join product metadata
by_prod = by_prod.merge(
    products[["product_id", "sub_category", "size_chart_id", "fit_type", "available_sizes"]],
    on="product_id", how="left",
)

# Bayesian smoothing: posterior mean with alpha = 5 prior observations at the global rate
base_rate = by_prod["fit_returns"].sum() / by_prod["total_returns"].sum()
ALPHA = 5
by_prod["bayes_fit_rate"] = (
    (by_prod["fit_returns"] + ALPHA * base_rate)
    / (by_prod["total_returns"] + ALPHA)
)

# final score = smoothed rate × log1p(count)  — rewards high rate AND high volume
by_prod["rank_score"] = by_prod["bayes_fit_rate"] * np.log1p(by_prod["fit_returns"])

by_prod_sorted = by_prod.sort_values("rank_score", ascending=False).reset_index(drop=True)
print(f"Global fit-issue rate: {base_rate:.3f}")
print(f"Products analysed: {len(by_prod_sorted):,}")
print()
print("Top 15 products by rank score:")
display_cols = ["product_id", "sub_category", "fit_type", "fit_returns",
                "total_returns", "fit_rate", "bayes_fit_rate", "rank_score"]
print(by_prod_sorted[display_cols].head(15).to_string(index=False))
"""))

# ── 3  Category breakdown ─────────────────────────────────────────────────────
cells.append(md("## 3 · Fit-issue breakdown by sub_category"))
cells.append(code(r"""
cat_agg = train.groupby("sub_category").agg(
    fit_returns=("label", lambda x: x.isin(FIT_LABELS).sum()),
    total_returns=("label", "count"),
).reset_index()
cat_agg["fit_rate"] = cat_agg["fit_returns"] / cat_agg["total_returns"]
cat_agg = cat_agg.sort_values("fit_returns", ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.barh(cat_agg["sub_category"][::-1], cat_agg["fit_returns"][::-1], color="#2a78d6", alpha=0.85)
ax.set_xlabel("Fit-issue return count", color="#0b0b0b")
ax.set_title("Fit returns by sub-category (top 15)", color="#0b0b0b")
ax.set_facecolor("white"); ax.grid(axis="x", color="#e4e3df", lw=0.6)
for sp in ax.spines.values(): sp.set_edgecolor("#e4e3df")

ax = axes[1]
ax.barh(cat_agg["sub_category"][::-1], cat_agg["fit_rate"][::-1], color="#1baf7a", alpha=0.85)
ax.set_xlabel("Fit-issue return rate", color="#0b0b0b")
ax.set_title("Fit rate by sub-category (top 15)", color="#0b0b0b")
ax.axvline(base_rate, color="#eb6834", lw=1.2, ls="--", label=f"global avg {base_rate:.2f}")
ax.legend(frameon=False, fontsize=8)
ax.set_facecolor("white"); ax.grid(axis="x", color="#e4e3df", lw=0.6)
for sp in ax.spines.values(): sp.set_edgecolor("#e4e3df")

fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(FIG_DIR / "phase11_fitrate_by_category.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved phase11_fitrate_by_category.png")
"""))

# ── 4  Top-5 selection ────────────────────────────────────────────────────────
cells.append(md("## 4 · Top-5 product selection"))
cells.append(code(r"""
TOP_K = 5
top5_df = by_prod_sorted.head(TOP_K)[["product_id", "sub_category", "fit_type",
                                       "fit_returns", "total_returns",
                                       "fit_rate", "bayes_fit_rate", "rank_score"]].copy()
top5_df.insert(0, "rank", range(1, TOP_K + 1))
print("Selected top-5 products:")
print(top5_df.to_string(index=False))
print()

# visualise top-5 vs next-5
compare_df = by_prod_sorted.head(10).copy()
compare_df["tier"] = ["Top 5" if i < 5 else "Next 5" for i in range(10)]
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#2a78d6" if t == "Top 5" else "#b9b8b2" for t in compare_df["tier"]]
bars = ax.barh(compare_df["product_id"][::-1], compare_df["rank_score"][::-1], color=colors[::-1], alpha=0.85)
ax.set_xlabel("Rank score  (Bayesian fit rate × log1p(count))", color="#0b0b0b")
ax.set_title("Top-10 products — rank score (blue = selected top-5)", color="#0b0b0b")
ax.set_facecolor("white"); ax.grid(axis="x", color="#e4e3df", lw=0.6)
for sp in ax.spines.values(): sp.set_edgecolor("#e4e3df")
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(FIG_DIR / "phase11_top5_vs_next5.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved phase11_top5_vs_next5.png")
"""))

# ── 5  Size-chart deep-dive for top-5 ─────────────────────────────────────────
cells.append(md("## 5 · Size-chart profiles for the top-5 products"))
cells.append(code(r"""
top5_pids = top5_df["product_id"].tolist()
top5_meta = products[products["product_id"].isin(top5_pids)].copy()
top5_meta = top5_meta.merge(top5_df[["product_id", "rank"]], on="product_id").sort_values("rank")

print("Product metadata for top-5:")
print(top5_meta[["rank", "product_id", "title", "sub_category", "fit_type",
                  "available_sizes", "size_chart_id", "price_inr"]].to_string(index=False))
print()

# size chart entries for these products
sc_ids = top5_meta["size_chart_id"].dropna().unique().tolist()
if sc_ids:
    sc_subset = size_charts[size_charts["size_chart_id"].isin(sc_ids)]
    print(f"Size-chart data for {len(sc_ids)} charts ({len(sc_subset)} size entries):")
    print(sc_subset.to_string(index=False))
else:
    print("Note: some top-5 products have no size_chart_id (free-size items).")
"""))

# ── 6  Write submission file ──────────────────────────────────────────────────
cells.append(md("## 6 · Write `R2_size_chart_top5.csv`"))
cells.append(code(r"""
SC_PATH = REP_DIR / "R2_size_chart_top5.csv"
submission = pd.DataFrame({
    "rank": range(1, TOP_K + 1),
    "product_id": top5_pids,
})
submission.to_csv(SC_PATH, index=False)
print(f"Saved {SC_PATH}")
print()
print(submission.to_string(index=False))
print()

# validate format matches evaluate.py requirements
assert len(submission) == 5, "Must have exactly 5 rows"
assert set(submission.columns) == {"rank", "product_id"}, "Must have rank and product_id columns"
assert sorted(submission["rank"].tolist()) == [1, 2, 3, 4, 5], "Ranks must be 1..5"
assert submission["product_id"].nunique() == 5, "All product_ids must be unique"
assert (submission["product_id"].str.strip() != "").all(), "No empty product_ids"
print("Format validation PASSED")
"""))

# ── 7  Re-score with size chart ───────────────────────────────────────────────
cells.append(md("## 7 · Re-run official scorer with size-chart file"))
cells.append(code(r"""
import tempfile, importlib.util, sys

# load predictions and truth
preds = pd.read_csv(REP_DIR / "phase9_validation_predictions.csv")
protocol = build_protocol(CFG, DATA, seed=0)
truth_df = protocol.parts["test"].copy()

# load the official scorer directly and pass the size chart path
scorer_path = ROOT / "tools" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", scorer_path)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    pred_path   = tmp / "predictions.csv"
    truth_path  = tmp / "truth.csv"
    sc_path     = tmp / "R2_size_chart_top5.csv"
    preds[["return_id", "decision", "confidence", "justification"]].to_csv(pred_path, index=False)
    truth_df[["return_id", "label", "customer_tenure_days", "order_value_inr", "customer_reason"]].to_csv(truth_path, index=False)
    submission.to_csv(sc_path, index=False)
    result_with_sc = evaluate.score(pred_path, truth_path, sc_path, None)

# compare with Phase 10 (no size chart)
with open(REP_DIR / "phase10_official_result.json") as f:
    result_p10 = json.load(f)

c_p10 = result_p10["components"]
c_sc  = result_with_sc["components"]

print("Official score comparison:")
print(f"{'Component':<25} {'Phase 10 (no SC)':>18} {'Phase 11 (with SC)':>20}")
print("-" * 65)
for key in ["savings", "fairness_score", "fairness_effective", "size_chart_score"]:
    print(f"  {key:<23} {c_p10[key]:>18.6f} {c_sc[key]:>20.6f}")
print("-" * 65)
print(f"  {'PRIMARY SCORE':<23} {result_p10['primary_score']:>18.6f} {result_with_sc['primary_score']:>20.6f}")
delta = result_with_sc["primary_score"] - result_p10["primary_score"]
print(f"  {'Δ primary':<23} {'':>18} {delta:>+20.6f}")
print()
print(f"Size-chart note: {result_with_sc['size_chart']['note']}")
"""))

# ── 8  Save updated report ────────────────────────────────────────────────────
cells.append(md("## 8 · Save Phase 11 report"))
cells.append(code(r"""
phase11_report = {
    "top5_products": top5_pids,
    "top5_details": top5_df.to_dict(orient="records"),
    "scoring": {
        "phase10_primary":          result_p10["primary_score"],
        "phase11_primary":          result_with_sc["primary_score"],
        "delta":                    result_with_sc["primary_score"] - result_p10["primary_score"],
        "size_chart_score":         c_sc["size_chart_score"],
        "size_chart_note":          result_with_sc["size_chart"]["note"],
    },
}
with open(REP_DIR / "phase11_report.json", "w") as f:
    json.dump(phase11_report, f, indent=2)
print("Saved reports/phase11_report.json")

# also save the updated full result
with open(REP_DIR / "phase11_official_result.json", "w") as f:
    json.dump(result_with_sc, f, indent=2)
print("Saved reports/phase11_official_result.json")
print()
print("=== PHASE 11 FINAL SUMMARY ===")
print(f"Top-5 products: {top5_pids}")
print(f"Size-chart score: {c_sc['size_chart_score']:.4f}  (scored by organisers on hidden truth — 0 until they run)")
print(f"Primary score WITH size-chart file: {result_with_sc['primary_score']:.6f}")
print(f"Delta vs Phase 10:                 {result_with_sc['primary_score'] - result_p10['primary_score']:+.6f}")
"""))

# ── 9  Checklist ──────────────────────────────────────────────────────────────
cells.append(md("""## 9 · Phase 11 checklist

- [ ] Per-product fit-issue counts and Bayesian-smoothed rates computed ✓
- [ ] Top-5 selected by smoothed_rate × log1p(count) score ✓
- [ ] `R2_size_chart_top5.csv` written with columns rank, product_id ✓
- [ ] Format validation passed (5 rows, ranks 1–5, unique non-empty IDs) ✓
- [ ] Official scorer re-run with size-chart file attached ✓
- [ ] Phase comparison table (Phase 10 vs Phase 11) ✓
- [ ] Reports saved: phase11_report.json, phase11_official_result.json ✓
- [ ] Hidden Jul–Sep labels never touched ✓

**Next:** Phase 12 — explanations & submission dashboard
"""))

# ── assemble notebook ─────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}

NB.parent.mkdir(parents=True, exist_ok=True)
with open(NB, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"wrote {NB}")
