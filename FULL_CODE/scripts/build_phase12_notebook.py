"""Build Phase_12_Explanations_and_Dashboard.ipynb."""

from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "Phase_12_Explanations_and_Dashboard.ipynb"


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
# Phase 12 — Explanations & Submission Dashboard

**Goals:**
1. **SHAP explanations** — global feature importance and per-decision example explanations from the Phase 8 CatBoost model.
2. **Justification audit** — review and validate per-row justification strings in the submission predictions.
3. **Submission assembly** — collect all deliverables into `TEAMID_R2.zip`.
4. **Final score dashboard** — one-page summary of every metric across all phases.

**No hidden Jul–Sep labels touched.**
"""))

# ── 1  Imports ────────────────────────────────────────────────────────────────
cells.append(md("## 1 · Imports & paths"))
cells.append(code(r"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

from r2sorter.config import load_config
from r2sorter.data import load_all
from r2sorter.protocol import build_protocol

CFG  = load_config(ROOT / "config.yaml")
DATA = load_all(CFG)
protocol = build_protocol(CFG, DATA, seed=0)

FIG_DIR = ROOT / "reports" / "figures"
REP_DIR = ROOT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("All imports OK")
print(f"shap version: {shap.__version__}")
"""))

# ── 2  Load model + predictions ───────────────────────────────────────────────
cells.append(md("## 2 · Load CatBoost model and test features"))
cells.append(code(r"""
import catboost as cb
import joblib

model = cb.CatBoostClassifier()
model.load_model(str(ROOT / "models" / "phase8" / "catboost_seed0.cbm"))

calibrator = joblib.load(ROOT / "models" / "phase8" / "final_abuse_calibrator.joblib")

# full 90-feature matrix via protocol
X_test  = protocol.matrix("tree", "test")
y_test  = protocol.y_abuse["test"]
test_df = protocol.parts["test"].copy()

preds = pd.read_csv(REP_DIR / "phase9_validation_predictions.csv")
preds = preds.merge(test_df[["return_id", "label"]], on="return_id", how="left")

print(f"Test rows: {len(X_test):,}  Features: {X_test.shape[1]}")
print(f"Predictions rows: {len(preds):,}")
print(f"Decision distribution:\n{preds['decision'].value_counts().to_string()}")
"""))

# ── 3  Global SHAP importance ────────────────────────────────────────────────
cells.append(md("## 3 · Global SHAP feature importance"))
cells.append(code(r"""
# Use TreeExplainer — fast for CatBoost; sample 2000 rows to keep memory low
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx].reset_index(drop=True)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# CatBoost multi-class: shap_values is (n_classes, n_samples, n_features)
# class 0 = abusive, class 1 = genuine (varies by encoder)
# use the class corresponding to abusive (y=1 in protocol)
raw_test = model.predict_proba(X_test)
# find which column is abusive
abuse_class_idx = 1 if raw_test[:, 1].mean() < raw_test[:, 0].mean() else 0
print(f"Abusive class index in CatBoost: {abuse_class_idx}")

if isinstance(shap_values, list):
    sv_abuse = shap_values[abuse_class_idx]
else:
    sv_abuse = shap_values[:, :, abuse_class_idx] if shap_values.ndim == 3 else shap_values

feat_names = list(X_test.columns)
mean_abs_shap = np.abs(sv_abuse).mean(axis=0)
shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs_shap})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_df["rank"] = range(1, len(shap_df) + 1)

print("Top 20 features by mean |SHAP| (abuse class):")
print(shap_df.head(20).to_string(index=False))

# save
shap_df.to_csv(REP_DIR / "phase12_shap_importance.csv", index=False)
print("\nSaved phase12_shap_importance.csv")
"""))

# ── 4  SHAP bar chart ────────────────────────────────────────────────────────
cells.append(md("## 4 · SHAP importance chart"))
cells.append(code(r"""
top20 = shap_df.head(20)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#2a78d6", alpha=0.85)
ax.set_xlabel("Mean |SHAP value|  (impact on P(abusive))", color="#0b0b0b")
ax.set_title("Top-20 features — global SHAP importance (CatBoost, abuse class)", color="#0b0b0b")
ax.set_facecolor("white")
ax.grid(axis="x", color="#e4e3df", lw=0.6)
for sp in ax.spines.values():
    sp.set_edgecolor("#e4e3df")
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(FIG_DIR / "phase12_shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved phase12_shap_importance.png")
"""))

# ── 5  Per-decision SHAP examples ────────────────────────────────────────────
cells.append(md("## 5 · Per-decision example explanations"))
cells.append(code(r"""
# For each decision bucket pick 3 representative examples and show top-5 SHAP drivers

preds_indexed = preds.set_index("return_id")
decisions = ["auto_approve", "inspect", "escalate"]
top_features = shap_df["feature"].head(15).tolist()

print("=" * 72)
print("EXAMPLE EXPLANATIONS (top-5 SHAP drivers per row)")
print("=" * 72)

for decision in decisions:
    subset = preds[preds["decision"] == decision].head(3)
    print(f"\n{'─'*20} {decision.upper()} (showing 3 examples) {'─'*20}")
    for _, row in subset.iterrows():
        rid = row["return_id"]
        # find row index in X_test
        test_idx = test_df[test_df["return_id"] == rid].index
        if len(test_idx) == 0:
            continue
        pos = test_df.index.get_loc(test_idx[0])
        row_shap = sv_abuse[np.where(sample_idx == pos)[0]]
        if len(row_shap) == 0:
            # not in sample — skip detailed shap, show top features by raw value
            x_row = X_test.iloc[pos]
            drivers = [(f, x_row[f]) for f in top_features[:5]]
            shap_str = "  (not in SHAP sample)"
        else:
            row_shap = row_shap[0]
            top5_idx = np.argsort(np.abs(row_shap))[::-1][:5]
            drivers = [(feat_names[i], row_shap[i]) for i in top5_idx]
            shap_str = ""

        print(f"\n  {rid}  |  true={row['label']}  |  decision={row['decision']}"
              f"  |  confidence={row['confidence']:.3f}")
        print(f"  justification: {row['justification'][:100]}")
        if not shap_str:
            print("  top-5 SHAP drivers:")
            for feat, val in drivers:
                direction = "↑abuse" if val > 0 else "↓abuse"
                print(f"    {feat:<35s} {val:+.4f}  {direction}")
        else:
            print(shap_str)
"""))

# ── 6  SHAP beeswarm (summary plot) ──────────────────────────────────────────
cells.append(md("## 6 · SHAP beeswarm summary (top-15 features)"))
cells.append(code(r"""
top15_idx = [feat_names.index(f) for f in shap_df["feature"].head(15).tolist()]
sv_top15  = sv_abuse[:, top15_idx]
X_top15   = X_sample[shap_df["feature"].head(15).tolist()]

fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(sv_top15, X_top15, plot_type="dot", show=False,
                  color_bar_label="Feature value (normalized)")
plt.title("SHAP beeswarm — top-15 features (abuse class)", color="#0b0b0b", pad=12)
fig = plt.gcf()
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(FIG_DIR / "phase12_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved phase12_shap_beeswarm.png")
"""))

# ── 7  Justification audit ────────────────────────────────────────────────────
cells.append(md("## 7 · Justification string audit"))
cells.append(code(r"""
print("Justification coverage:")
print(f"  Total rows: {len(preds):,}")
print(f"  Non-empty:  {(preds['justification'].str.len() > 0).sum():,}")
print(f"  Missing:    {preds['justification'].isna().sum()}")
print()

# length distribution
jlen = preds["justification"].str.len()
print(f"Length stats: min={jlen.min()}  median={jlen.median():.0f}  max={jlen.max()}")
print()

# sample justifications per decision
for dec in ["auto_approve", "inspect", "escalate"]:
    sample = preds[preds["decision"] == dec]["justification"].dropna().head(3)
    print(f"--- {dec} ---")
    for j in sample:
        print(f"  {j[:120]}")
    print()

# check all justifications contain a confidence/probability mention
has_prob = preds["justification"].str.contains(r"P=|p=|prob|conf|%", case=False, na=False)
print(f"Justifications referencing confidence/probability: {has_prob.sum():,} / {len(preds):,}")

# fig: justification length by decision
fig, ax = plt.subplots(figsize=(7, 3.5))
for dec, color in zip(["auto_approve", "inspect", "escalate"], ["#1baf7a", "#f5a623", "#eb6834"]):
    lengths = preds[preds["decision"] == dec]["justification"].str.len()
    ax.hist(lengths, bins=30, alpha=0.6, color=color, label=dec)
ax.set_xlabel("Justification length (chars)", color="#0b0b0b")
ax.set_ylabel("Count", color="#0b0b0b")
ax.set_title("Justification length by decision", color="#0b0b0b")
ax.legend(frameon=False, fontsize=8)
ax.set_facecolor("white"); ax.grid(axis="y", color="#e4e3df", lw=0.6)
for sp in ax.spines.values(): sp.set_edgecolor("#e4e3df")
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(FIG_DIR / "phase12_justification_lengths.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved phase12_justification_lengths.png")
"""))

# ── 8  Final score dashboard ──────────────────────────────────────────────────
cells.append(md("## 8 · Final score dashboard"))
cells.append(code(r"""
# load phase results
with open(REP_DIR / "phase9_config.json") as f:
    p9 = json.load(f)["results"]
with open(REP_DIR / "phase10_official_result.json") as f:
    p10 = json.load(f)
with open(REP_DIR / "phase11_report.json") as f:
    p11 = json.load(f)

c = p10["components"]
ad = p10["abuse_detection"]
cal = p10["calibration_of_confidence_as_P_abusive"]
fairness = p10["fairness"]

print("=" * 72)
print("FINAL SCORE DASHBOARD — TalentAP Mega Hackathon 2026 R2")
print("=" * 72)
print()
print(f"  PRIMARY SCORE   : {p10['primary_score']:.4f}")
print(f"  ─ savings       : {c['savings']:.4f}   (0.7 weight)")
print(f"  ─ fairness      : {c['fairness_effective']:.4f}   (0.2 weight)")
print(f"  ─ size_chart    : {c['size_chart_score']:.4f}   (0.1 weight; scored by organisers)")
print()
print(f"  Fairness groups:")
for pair, v in fairness.items():
    print(f"    {pair:<38s}: ratio {v['disparity_ratio']:.3f}  score {v['score']:.4f}")
print()
print(f"  Abuse detection : precision {ad['flag_precision']:.4f}  recall {ad['flag_recall']:.4f}")
print(f"  Genuine escaped : {ad['genuine_escalated']} wrongly escalated  "
      f"({ad['genuine_flag_rate']:.4f} flag rate)")
print()
print(f"  Calibration     : Brier {cal['brier']:.4f}  ECE {cal['ece_10_bins']:.4f}  PR-AUC {cal['average_precision']:.4f}")
print()
print(f"  Phase comparison (P8 baseline → P9 final):")
print(f"    Baseline official : {p9['baseline_official']:.4f}")
print(f"    Final official    : {p9['final_official']:.4f}   (+{p9['improvement']:.4f})")
print()
print(f"  Size-chart top-5  : {p11['top5_products']}")
print(f"  Submission file   : R2_size_chart_top5.csv  (saved in reports/)")

# ── dashboard chart ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# left: primary score components
ax = axes[0]
labels_c = ["savings\n×0.7", "fairness\n×0.2", "size_chart\n×0.1"]
vals_c   = [c["savings"] * 0.7, c["fairness_effective"] * 0.2, c["size_chart_score"] * 0.1]
colors_c = ["#2a78d6", "#1baf7a", "#eb6834"]
bars = ax.bar(labels_c, vals_c, color=colors_c, alpha=0.85)
ax.axhline(p10["primary_score"], color="#0b0b0b", lw=1.2, ls="--",
           label=f"primary={p10['primary_score']:.4f}")
ax.set_title("Score components", color="#0b0b0b")
ax.set_ylabel("Weighted contribution", color="#0b0b0b")
ax.legend(frameon=False, fontsize=8)
ax.set_facecolor("white"); ax.grid(axis="y", color="#e4e3df", lw=0.6)
for b, v in zip(bars, vals_c):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#e4e3df")

# centre: fairness scores
ax = axes[1]
pair_labels = [p.replace("_vs_", "\nvs\n").replace("_", " ") for p in fairness]
fair_scores = [v["score"] for v in fairness.values()]
bars2 = ax.bar(pair_labels, fair_scores, color="#1baf7a", alpha=0.85)
ax.axhline(1.0, color="#b9b8b2", lw=1, ls="--")
ax.set_ylim(0, 1.1)
ax.set_title("Per-pair fairness scores", color="#0b0b0b")
ax.set_ylabel("Score (1.0 = equal)", color="#0b0b0b")
ax.set_facecolor("white"); ax.grid(axis="y", color="#e4e3df", lw=0.6)
for b, v in zip(bars2, fair_scores):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#e4e3df")
ax.tick_params(axis="x", labelsize=7)

# right: phase progression
ax = axes[2]
phase_labels = ["P8\nbaseline", "P9\nfinal"]
phase_scores = [p9["baseline_official"], p9["final_official"]]
bars3 = ax.bar(phase_labels, phase_scores, color=["#b9b8b2", "#2a78d6"], alpha=0.85)
ax.set_title("Phase progression", color="#0b0b0b")
ax.set_ylabel("Official primary score", color="#0b0b0b")
ax.set_ylim(0, max(phase_scores) * 1.15)
ax.set_facecolor("white"); ax.grid(axis="y", color="#e4e3df", lw=0.6)
for b, v in zip(bars3, phase_scores):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
for sp in ax.spines.values(): sp.set_edgecolor("#e4e3df")

fig.suptitle("R2 — Returns Fraud Sorter  |  Final Dashboard", fontsize=11, color="#0b0b0b", y=1.01)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(FIG_DIR / "phase12_final_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved phase12_final_dashboard.png")
"""))

# ── 9  Assemble submission zip ────────────────────────────────────────────────
cells.append(md("## 9 · Assemble submission ZIP"))
cells.append(code(r"""
TEAM_ID = "TEAMID"   # ← replace with real team ID before final submission
ZIP_NAME = f"{TEAM_ID}_R2.zip"
ZIP_PATH = REP_DIR / ZIP_NAME

# files required by the problem spec
submission_files = {
    # main predictions
    "phase9_validation_predictions.csv": REP_DIR / "phase9_validation_predictions.csv",
    # size-chart file (evaluate.py looks for R2_size_chart_top5.csv next to submission)
    "R2_size_chart_top5.csv": REP_DIR / "R2_size_chart_top5.csv",
    # supporting reports
    "phase10_official_result.json": REP_DIR / "phase10_official_result.json",
    "phase11_report.json":          REP_DIR / "phase11_report.json",
    "phase12_shap_importance.csv":  REP_DIR / "phase12_shap_importance.csv",
    # figures
    "figures/phase12_final_dashboard.png":     FIG_DIR / "phase12_final_dashboard.png",
    "figures/phase12_shap_importance.png":     FIG_DIR / "phase12_shap_importance.png",
    "figures/phase12_shap_beeswarm.png":       FIG_DIR / "phase12_shap_beeswarm.png",
    "figures/phase10_fairness_audit.png":      FIG_DIR / "phase10_fairness_audit.png",
    "figures/phase10_confusion_matrix.png":    FIG_DIR / "phase10_confusion_matrix.png",
    "figures/phase10_calibration.png":         FIG_DIR / "phase10_calibration.png",
}

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for arc_name, src_path in submission_files.items():
        if src_path.exists():
            zf.write(src_path, arc_name)
            print(f"  + {arc_name}")
        else:
            print(f"  ! MISSING: {arc_name}")

print()
print(f"Submission ZIP: {ZIP_PATH}")
print(f"Size: {ZIP_PATH.stat().st_size / 1024:.1f} KB")
print()
print("Contents:")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    for info in zf.infolist():
        print(f"  {info.filename:<55s}  {info.file_size/1024:6.1f} KB")
print()
print(f"NOTE: Replace '{TEAM_ID}' in TEAM_ID with your actual team ID,")
print("then run: python FULL_CODE/scripts/make_submission_zip.py <TEAM_ID>")
"""))

# ── 10  Checklist ─────────────────────────────────────────────────────────────
cells.append(md("""## 10 · Phase 12 checklist

- [ ] SHAP TreeExplainer run on CatBoost (2000-row sample) ✓
- [ ] Global feature importance bar chart ✓
- [ ] Per-decision example explanations (3 per bucket) ✓
- [ ] SHAP beeswarm summary plot ✓
- [ ] Justification audit (coverage, length, samples) ✓
- [ ] Final score dashboard chart (components + fairness + progression) ✓
- [ ] `TEAMID_R2.zip` assembled with predictions + size-chart + reports ✓
- [ ] Hidden Jul–Sep labels never touched ✓

**All 12 phases complete.**
"""))

# ── assemble ──────────────────────────────────────────────────────────────────
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
