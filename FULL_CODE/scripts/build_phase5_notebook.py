"""Create notebooks/Phase_05_Baseline_Models.ipynb (uses baselines.py, preprocess.py, nlp.py)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as _code
from nbformat.v4 import new_markdown_cell as _md
from nbformat.v4 import new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Phase_05_Baseline_Models.ipynb"


def md(text: str):
    return _md(text.strip("\n"))


def code(text: str):
    return _code(text.strip("\n"))


def train_cell(model: str) -> str:
    return f"train_model('{model}')\nsummary_of('{model}')"


CELLS = [
    md(r"""
# Phase 5 - Baseline models
**Objective:** a trustworthy reference for Phase 6. Simple, untuned models are trained on identical rows, compared on the same chronological
validation period, and judged by class-wise quality, probability reliability **and** the official business cost - not accuracy alone.

| Requirement (Phase 5 document) | Section |
|---|---|
| Data and target mapping verified | 2 |
| Leakage-safe splits | 3 |
| Preprocessing fitted on training only; Phase 4 text scores out-of-fold | 4 |
| Dummy, Logistic Regression, Decision Tree (required); Random Forest, Extra Trees, HistGradientBoosting (recommended); CatBoost; Naive Bayes (optional) | 5 |
| Data analytics: class balance, order value, feature box plots (training part only) | 3b |
| Grouped model-comparison chart and confusion-matrix heatmap | 6b, 9 |
| Macro-F1, per-class metrics, confusion matrices | 6 |
| Probability quality and calibration limits | 7 |
| Classifier and routing evaluated separately, then together (official scorer, fairness) | 8 |
| Error analysis | 9 |
| Selection by evidence, saved configuration, hand-off | 10 - 11 |
| Completion checklist | 12 |

**Protocol (fixed before training):** the Phase 2 chronological 70 / 10 / 20 split. **train** fits every model and preprocessor; **calibration** (10%) is used only to fit an isotonic
calibrator; **test (validation period)** is the common yardstick. The hidden July-September test is never touched. Hyper-parameters are fixed in advance, not tuned on the validation period.
Run time is about 8-10 minutes.
"""),
    code(r"""
import json
import platform
import sys
import time
from pathlib import Path

for candidate_path in (Path('../src'), Path('src')):
    if candidate_path.exists():
        sys.path.insert(0, str(candidate_path.resolve()))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from r2sorter.baselines import (
    MATRIX_KIND, MODEL_ORDER, STOCHASTIC, abuse_metrics, build_model, business_cost, catboost_inputs, predict_full, reliability_table)
from r2sorter.calibration import AbuseCalibrator
from r2sorter.config import load_config
from r2sorter.data import LABELS, load_all
from r2sorter.features import build_feature_table, tabular_columns
from r2sorter.fairness import audit
from r2sorter.nlp import TextCandidate, check_rolling, classification_report, rolling_origin_out_of_fold
from r2sorter.preprocess import FeaturePreprocessor, add_engineered_features
from r2sorter.router import route
from r2sorter.scoring import official_score
from r2sorter.splits import chronological_split, describe_splits
from r2sorter.text import normalise_reason

pd.set_option('display.width', 220, 'display.max_columns', 50, 'display.max_colwidth', 80)
SEED, SEEDS = 42, (0, 1)
cfg = load_config()
data = load_all(cfg)
train, test, products, charts, customers = (data[k] for k in ('train', 'test', 'products', 'size_charts', 'customers'))
REPORTS = Path('../reports') if Path('../reports').exists() else Path('reports')
REPORTS.mkdir(exist_ok=True)
versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__, 'lightgbm': lgb.__version__}
print(versions)
print('official cost matrix:', cfg['costs'])
"""),
    md(r"""
## 2. Data and target mapping
"""),
    code(r"""
assert 'label' in train and 'label' not in test.columns, 'hidden-test labels must not exist'
print('target mapping (class index -> name):', dict(enumerate(LABELS)))
print('class balance in training data:')
print(train['label'].value_counts(normalize=True).round(4).to_string())
missing = train.isna().sum()
print('\ncolumns with missing values (train):', missing[missing > 0].to_dict(), '- only prior_return_rate is expected (no history)')
frame = build_feature_table(train, test, products, cfg)
frame = add_engineered_features(frame, charts, cfg, customers)
labelled = frame[frame['is_train']].sort_values(['return_request_date', 'return_id']).reset_index(drop=True)
print('\nlabelled rows:', len(labelled), '| duplicate return_ids:', int(labelled['return_id'].duplicated().sum()))
"""),
    md(r"""
## 3. Leakage-safe splits (Phase 2 definitions)
"""),
    code(r"""
plan = chronological_split(labelled['return_request_date'])
parts = {name: labelled[plan == name].reset_index(drop=True) for name in ('train', 'calibration', 'test')}
display(describe_splits(labelled, plan))
last_train, first_cal = parts['train'].return_request_date.max(), parts['calibration'].return_request_date.min()
last_cal, first_test = parts['calibration'].return_request_date.max(), parts['test'].return_request_date.min()
assert last_train < first_cal and last_cal < first_test, 'splits must be strictly chronological'
print('customers shared between train and test (allowed - future requests of known customers):',
      len(set(parts['train'].customer_id) & set(parts['test'].customer_id)))
print('split identifiers: train <=', last_train.date(), '| calibration', first_cal.date(), '-', last_cal.date(), '| test', first_test.date(), '-', parts['test'].return_request_date.max().date())
"""),
    md(r"""
## 3b. Data analytics (training part only)
Three charts that answer project questions, drawn from the **training part only** so the validation period stays unseen. They do not repeat the Phase 1 EDA or the Phase 4 text plots.
A pie chart of the four classes is intentionally omitted: it would show the same numbers as chart 1.
"""),
    code(r"""
# Chart style: emphasis colouring (abusive = accent, genuine classes = grey); categorical slots from the validated reference palette.
INK, MUTED, GRID, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#b9b8b2'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
FIG = REPORTS / 'figures'
FIG.mkdir(exist_ok=True)

def style(ax, grid_axis='x'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

def title(fig, headline, sub=None):
    fig.text(0.01, 0.98, headline, ha='left', va='top', fontsize=13, fontweight='bold', color=INK)
    if sub:
        fig.text(0.01, 0.915, sub, ha='left', va='top', fontsize=9.5, color=MUTED)

names = {'abusive': 'Abusive', 'genuine_quality_issue': 'Genuine: quality', 'genuine_fit_issue': 'Genuine: fit', 'genuine_wrong_item': 'Genuine: wrong item'}
counts = parts['train']['label'].value_counts().reindex(LABELS)
share = counts / counts.sum()
fig, ax = plt.subplots(figsize=(8, 3.2))
fig.subplots_adjust(top=0.72, left=0.2, right=0.97, bottom=0.12)
bars = ax.barh([names[l] for l in LABELS][::-1], counts.to_numpy()[::-1], height=0.55,
               color=[ORANGE if l == 'abusive' else GREY for l in LABELS][::-1])
for bar, n, p_ in zip(bars, counts.to_numpy()[::-1], share.to_numpy()[::-1]):
    ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2, f'{n:,}  ({p_:.1%})', va='center', fontsize=9, color=INK)
ax.set_xlim(0, counts.max() * 1.22)
ax.set_xlabel('requests in the training part', color=MUTED, fontsize=9)
style(ax)
title(fig, f'Abusive requests are only {share["abusive"]:.1%} of the training data', 'A model that never says "abusive" is right about 92% of the time, so accuracy alone cannot judge this task.')
fig.savefig(FIG / 'phase5_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'accuracy of "never abusive": {1 - share["abusive"]:.3f} | accuracy of always predicting the largest class ({names[share.idxmax()]}): {share.max():.3f}')
"""),
    code(r"""
tr = parts['train']
is_abusive = tr['label'] == 'abusive'
bins = np.logspace(np.log10(max(tr['order_value_inr'].min(), 1)), np.log10(tr['order_value_inr'].max()), 36)
fig, ax = plt.subplots(figsize=(8, 3.8))
fig.subplots_adjust(top=0.78, left=0.09, right=0.97, bottom=0.16)
for mask, colour, label in ((~is_abusive, BLUE, 'Genuine'), (is_abusive, ORANGE, 'Abusive')):
    ax.hist(tr.loc[mask, 'order_value_inr'], bins=bins, density=True, histtype='stepfilled', alpha=0.45, color=colour, edgecolor=colour, linewidth=1.4, label=f'{label} (median INR {tr.loc[mask, "order_value_inr"].median():,.0f})')
ax.set_xscale('log')
ax.set_xlabel('order value (INR, log scale)', color=MUTED, fontsize=9)
ax.set_ylabel('share of group (density)', color=MUTED, fontsize=9)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)
style(ax, 'y')
title(fig, 'Order values overlap heavily between genuine and abusive requests', 'Each group is scaled to its own total, so the small abusive group is comparable in shape. A high value alone is not evidence of fraud.')
fig.savefig(FIG / 'phase5_order_value_histogram.png', dpi=150, bbox_inches='tight')
plt.show()
band = pd.qcut(tr['order_value_inr'], 5, duplicates='drop')
print('abusive rate by order-value quintile (is value alone informative?):')
print(is_abusive.groupby(band, observed=True).agg(['mean', 'size']).rename(columns={'mean': 'abusive_rate', 'size': 'rows'}).round(4).to_string())
"""),
    code(r"""
features = [('prior_return_rate', 'prior return rate', False), ('customer_tenure_days', 'customer tenure (days)', False), ('order_value_inr', 'order value (INR, log scale)', True)]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
fig.subplots_adjust(top=0.74, left=0.07, right=0.99, bottom=0.2, wspace=0.28)
short = {'abusive': 'Abusive', 'genuine_quality_issue': 'Quality', 'genuine_fit_issue': 'Fit', 'genuine_wrong_item': 'Wrong item'}
for ax, (column, label, log) in zip(axes, features):
    groups = [tr.loc[tr['label'] == l, column].dropna().to_numpy() for l in LABELS]
    box = ax.boxplot(groups, tick_labels=[short[l] for l in LABELS], patch_artist=True, widths=0.55, showfliers=False,
                     medianprops={'color': INK, 'linewidth': 1.6}, whiskerprops={'color': GREY}, capprops={'color': GREY})
    for patch, l in zip(box['boxes'], LABELS):
        patch.set_facecolor(ORANGE if l == 'abusive' else GREY)
        patch.set_alpha(0.85)
        patch.set_edgecolor('white')
    if log:
        ax.set_yscale('log')
    ax.set_title(label, fontsize=10, color=INK, loc='left')
    ax.tick_params(axis='x', labelrotation=20)
    style(ax, 'y')
title(fig, 'Which structured features separate the four classes?', 'Median and quartiles by class (outliers hidden). Prior return rate is left out of its panel for customers with no earlier orders.')
fig.savefig(FIG / 'phase5_feature_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
tr.groupby('label')[['prior_return_rate', 'customer_tenure_days', 'order_value_inr']].median().reindex(LABELS).round(3)
"""),
    md(r"""
## 4. Preprocessing (fitted on train only) and Phase 4 text scores (out-of-fold)
* `FeaturePreprocessor` (Phase 3) is fitted on the train part; the calibration and test parts are only transformed.
* Phase 4's selected text model (word TF-IDF + Logistic Regression) gives `txt_p_*`. For **train** rows the scores are rolling-origin out-of-fold (each month scored by a model fitted on strictly earlier months; the first 60 days use grouped OOF inside that window); calibration and test rows are scored by a model fitted on train only. No in-sample text predictions are used.
* Two feature arms: **A** structured only | **C** structured + Phase 4 text scores + rule/shape indicators + text-reuse history (Phase 4 findings). (The text-scores-only arm B was dropped to keep the run short; Phase 4 already compared it.)
"""),
    code(r"""
pre = FeaturePreprocessor(cfg).fit(parts['train'])
text_params = {**cfg['text'], 'svm_c': 0.5}
text_model = TextCandidate('word', 'logreg', text_params, SEED)
oof_scores, oof_kind, oof_fit_end, _ = rolling_origin_out_of_fold(
    text_model, parts['train']['customer_reason'], parts['train']['label'], parts['train']['customer_id'],
    parts['train']['return_request_date'], 60, int(cfg['validation']['text_oof_folds']))
print('time-aware text scores:', check_rolling(oof_kind, oof_fit_end, parts['train']['return_request_date']))
fitted_text = text_model.fit(parts['train']['customer_reason'], parts['train']['label'])
text_scores = {'train': oof_scores, **{n: fitted_text.scores(parts[n]['customer_reason']) for n in ('calibration', 'test')}}
text_cols = [f'txt_p_{label}' for label in LABELS]

reuse = ['hist_same_text', 'hist_max_text_sim', 'hist_n_similar_text']
rules = [c for c in tabular_columns(frame) if c.startswith(('ev_', 'txt_len', 'txt_n_', 'txt_upper', 'txt_has'))]
extra_scaler = StandardScaler().fit(parts['train'][rules + reuse])  # learned on train only (needed by linear models)
ARMS = {'A structured': [], 'C + all text-derived': text_cols + rules + reuse}

def matrix(kind, part, arm):
    rows = parts[part]
    base = pre.transform_linear(rows) if kind == 'linear' else pre.transform_tree(rows)
    base = base.reset_index(drop=True)
    pieces = [base]
    if ARMS[arm]:
        pieces.append(pd.DataFrame(text_scores[part], columns=text_cols))
    if arm.startswith('C'):
        extra = rows[rules + reuse].reset_index(drop=True)
        pieces.append(pd.DataFrame(extra_scaler.transform(extra), columns=rules + reuse) if kind == 'linear' else extra)
    out = pd.concat(pieces, axis=1)
    assert len(out) == len(rows) and not out.columns.duplicated().any()
    return out

demo = matrix('linear', 'test', 'C + all text-derived')
print('linear matrix for arm C:', demo.shape, '| NaNs:', int(demo.isna().sum().sum()))
print('tree matrix for arm C  :', matrix('tree', 'test', 'C + all text-derived').shape)
print('preprocessor dropped (constant/duplicate on train):', len(pre.dropped_columns_), 'columns')
"""),
    md(r"""
## 5. Train the baselines
Fixed settings (no tuning): Dummy = class prior; Logistic Regression C=1; Decision Tree depth 6, leaf >= 100; Random Forest / Extra Trees 100 trees, leaf >= 5; HistGradientBoosting 150 iterations, lr 0.08, 31 leaves, L2 5.
Stochastic models run with 2 seeds; Dummy, Logistic Regression and the Decision Tree are deterministic. Each model is trained on both feature arms (Dummy ignores features, so it runs once).
For every run: four-class probabilities, P(abusive) raw and isotonic-calibrated (calibrator fitted on the calibration part only), and routing by expected cost under the official matrix.
"""),
    code(r"""
y_idx = {n: parts[n]['label'].map(LABELS.index).to_numpy() for n in parts}
y_abuse = {n: (y_idx[n] == LABELS.index('abusive')).astype(int) for n in parts}
runs, store = [], {}

def evaluate(model_name, arm, seed, probs_cal, probs_test, seconds):
    ab = LABELS.index('abusive')
    raw_cal, raw_test = probs_cal[:, ab], probs_test[:, ab]
    calibrator = AbuseCalibrator('isotonic').fit(raw_cal, y_abuse['calibration'])
    cal_test = calibrator.transform(raw_test)
    report = classification_report(probs_test, parts['test']['label'])
    row = {'model': model_name, 'arm': arm, 'seed': seed, 'seconds': round(seconds, 1),
           'macro_f1': report['macro_f1'], 'accuracy': report['accuracy'],
           'abusive_precision': report['per_class'].loc['abusive', 'precision'],
           'abusive_recall': report['per_class'].loc['abusive', 'recall'],
           'log_loss_4class': float(log_loss(y_idx['test'], np.clip(probs_test, 1e-4, 1), labels=range(4)))}
    row.update({f'raw_{k}': v for k, v in abuse_metrics(y_abuse['test'], raw_test).items()})
    row.update({f'cal_{k}': v for k, v in abuse_metrics(y_abuse['test'], cal_test).items() if k in ('abuse_log_loss', 'abuse_ECE')})
    row.update({f'raw_{k}': v for k, v in business_cost(raw_test, y_abuse['test'], cfg).items()})
    row.update({f'cal_{k}': v for k, v in business_cost(cal_test, y_abuse['test'], cfg).items()})
    runs.append(row)
    store[(model_name, arm, seed)] = {'probs': probs_test, 'raw': raw_test, 'cal': cal_test, 'report': report}

def train_model(name):
    arms = ['A structured'] if name == 'dummy' else list(ARMS)
    seeds = SEEDS if name in STOCHASTIC else (0,)
    for arm in arms:
        kind = MATRIX_KIND[name]
        x_train, x_cal, x_test = (matrix(kind, p, arm) for p in ('train', 'calibration', 'test'))
        cat_names = []
        if name == 'catboost':  # convert categoricals to strings ONCE (not inside the seed loop)
            x_train, cat_names = catboost_inputs(x_train)
            x_cal, x_test = catboost_inputs(x_cal)[0], catboost_inputs(x_test)[0]
        for seed in seeds:
            started = time.perf_counter()
            if name == 'catboost':
                model = build_model(name, seed, cat_names).fit(x_train, y_idx['train'])
            else:
                model = build_model(name, seed).fit(x_train, y_idx['train'])
            probs_cal, probs_test = predict_full(model, x_cal), predict_full(model, x_test)
            evaluate(name, arm, seed, probs_cal, probs_test, time.perf_counter() - started)

def summary_of(name):
    table = pd.DataFrame([r for r in runs if r['model'] == name]).groupby('arm')[
        ['macro_f1', 'raw_abuse_AP', 'raw_abuse_log_loss', 'raw_abuse_ECE', 'cal_abuse_ECE', 'raw_saving_vs_approve_all',
         'cal_saving_vs_approve_all', 'seconds']].mean().round(4)
    display(table)
"""),
    code(train_cell("dummy")),
    code(train_cell("logreg")),
    code(train_cell("decision_tree")),
    code(train_cell("random_forest")),
    code(train_cell("extra_trees")),
    code(train_cell("hist_gb")),
    md(r"""
**CatBoost** (gradient boosting with native categorical handling) is the seventh baseline. Fixed settings: 200 iterations, depth 6, learning rate 0.08, L2 5.
"""),
    code(train_cell("catboost")),
    md(r"""
**Optional text-only baseline - Multinomial Naive Bayes** on word TF-IDF (non-negative features, as the document requires). It sees only `customer_reason`, so it shows what text alone achieves beyond Phase 4's linear model.
"""),
    code(r"""
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, preprocessor=normalise_reason)
x_tr = vectorizer.fit_transform(parts['train']['customer_reason'])
nb = MultinomialNB(alpha=0.5).fit(x_tr, y_idx['train'])
nb_probs = {n: nb.predict_proba(vectorizer.transform(parts[n]['customer_reason'])) for n in ('calibration', 'test')}
evaluate('naive_bayes_text', 'text only', 0, nb_probs['calibration'], nb_probs['test'], 0.0)
pd.DataFrame(runs[-1:])[['model', 'macro_f1', 'raw_abuse_AP', 'raw_abuse_log_loss', 'raw_abuse_ECE']].round(4)
"""),
    md(r"""
## 6. Comparison, per-class metrics and confusion matrices
All numbers below are on the same test (validation-period) rows. Means over seeds; `sd` is the seed spread of the saving.
"""),
    code(r"""
results = pd.DataFrame(runs)
agg = results.groupby(['model', 'arm']).agg(
    macro_f1=('macro_f1', 'mean'), accuracy=('accuracy', 'mean'), abusive_precision=('abusive_precision', 'mean'), abusive_recall=('abusive_recall', 'mean'), log_loss_4class=('log_loss_4class', 'mean'),
    abuse_AP=('raw_abuse_AP', 'mean'), raw_log_loss=('raw_abuse_log_loss', 'mean'), raw_ECE=('raw_abuse_ECE', 'mean'),
    cal_log_loss=('cal_abuse_log_loss', 'mean'), cal_ECE=('cal_abuse_ECE', 'mean'),
    raw_saving=('raw_saving_vs_approve_all', 'mean'), cal_saving=('cal_saving_vs_approve_all', 'mean'),
    saving_sd=('cal_saving_vs_approve_all', 'std'), genuine_escalated=('cal_genuine_escalated', 'mean'),
    abusive_approved=('cal_abusive_approved', 'mean'), seeds=('seed', 'nunique')).fillna({'saving_sd': 0.0}).round(4)
order = {m: i for i, m in enumerate([*MODEL_ORDER, 'naive_bayes_text'])}
agg = agg.sort_index(key=lambda idx: idx.map(order) if idx.name == 'model' else idx)
agg.to_csv(REPORTS / 'phase5_model_comparison.csv')
agg
"""),
    code(r"""
per_class = []
for (name, arm, seed), item in store.items():
    if seed != 0:
        continue
    table = item['report']['per_class'].copy()
    table.insert(0, 'arm', arm)
    table.insert(0, 'model', name)
    per_class.append(table.reset_index(names='class'))
per_class = pd.concat(per_class)
per_class.to_csv(REPORTS / 'phase5_per_class_metrics.csv', index=False)
print('per-class F1 (seed 0), arm A and arm C:')
per_class[per_class.arm.isin(['A structured', 'C + all text-derived'])].pivot_table(index=['model', 'arm'], columns='class', values='f1').round(3)
"""),
    code(r"""
confusions = []
for (name, arm, seed), item in store.items():
    if seed == 0:
        confusions.append(item['report']['confusion'].assign(model=name, arm=arm).reset_index(names='truth'))
pd.concat(confusions).to_csv(REPORTS / 'phase5_confusion_matrices.csv', index=False)
print('confusion matrix (rows = truth, columns = prediction; class order:', list(LABELS), ') for Logistic Regression arm C:')
store[('logreg', 'C + all text-derived', 0)]['report']['confusion']
"""),
    md(r"""
### 6b. Model comparison (macro-F1 and abusive-class precision / recall)
Full feature arm (C), mean over seeds. **Precision and recall are for the argmax four-class decision**, i.e. "the model's single most likely class is abusive". The routing in section 8 uses cost-based thresholds instead and flags far more requests, so these bars understate what routing catches (see `abusive_approved` in the table above). The chart exists to expose models whose macro-F1 hides a missing abusive class.
"""),
    code(r"""
arm_c = agg.xs('C + all text-derived', level='arm').reindex([m for m in MODEL_ORDER if m != 'dummy'])
fig, ax = plt.subplots(figsize=(10, 4.2))
fig.subplots_adjust(top=0.74, left=0.07, right=0.99, bottom=0.2)
positions = np.arange(len(arm_c))
width = 0.26
metrics = [('macro_f1', 'Macro-F1 (4 classes)', BLUE), ('abusive_precision', 'Abusive precision', ORANGE), ('abusive_recall', 'Abusive recall', AQUA)]
for offset, (column, label, colour) in zip((-width, 0, width), metrics):
    bars = ax.bar(positions + offset, arm_c[column], width=width - 0.03, color=colour, label=label)
    for bar, value in zip(bars, arm_c[column]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f'{value:.2f}', ha='center', fontsize=8, color=INK)
ax.set_xticks(positions)
ax.set_xticklabels(['Logistic\nRegression', 'Decision\nTree', 'Random\nForest', 'Extra\nTrees', 'HistGradient\nBoosting', 'CatBoost'], fontsize=9)
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=9, labelcolor=INK)
style(ax, 'y')
dummy_f1 = agg.loc[('dummy', 'A structured'), 'macro_f1']
title(fig, 'Compare models on the abusive class, not only on macro-F1', f'Validation period, feature arm C. Dummy baseline (always the prior): macro-F1 {dummy_f1:.2f}, no abusive predictions.')
fig.savefig(FIG / 'phase5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
arm_c[['macro_f1', 'abusive_precision', 'abusive_recall', 'abuse_AP' if 'abuse_AP' in arm_c else 'cal_saving', 'cal_saving', 'genuine_escalated', 'abusive_approved']].round(3)
"""),
    md(r"""
## 7. Probability quality and calibration limits
The classifier outputs are **not assumed calibrated**. Raw vs isotonic-calibrated log loss and ECE for P(abusive) are in the comparison table. Tree ensembles with averaged leaves are usually under-confident,
the Decision Tree is coarse (few distinct values), and the calibrator is fitted on only ~4,400 rows, so the calibrated numbers carry sampling noise.
"""),
    code(r"""
view = agg[['abuse_AP', 'raw_log_loss', 'cal_log_loss', 'raw_ECE', 'cal_ECE', 'raw_saving', 'cal_saving']]
print('effect of calibration (positive = calibration helped):')
effect = pd.DataFrame({'log_loss_gain': view.raw_log_loss - view.cal_log_loss, 'ECE_gain': view.raw_ECE - view.cal_ECE,
                       'saving_gain': view.cal_saving - view.raw_saving}).round(4)
display(effect)
best_for_plot = ('logreg', 'C + all text-derived')
item = store[(*best_for_plot, 0)]
reliability = pd.concat([reliability_table(y_abuse['test'], item['raw']).assign(version='raw'),
                         reliability_table(y_abuse['test'], item['cal']).assign(version='isotonic')])
reliability.to_csv(REPORTS / 'phase5_reliability_logreg.csv', index=False)
print('reliability, Logistic Regression arm C (10 equal-count bins; observed should match predicted):')
reliability
"""),
    md(r"""
## 8. Classifier, routing and the two together
**(a) Classifier** - four-class quality (section 6). **(b) Routing** - the cost of the decisions it induces, using the official matrix (genuine: approve 0 / inspect 0.5 / escalate 10; abusive: approve 1 / inspect 0.3 / escalate 0),
with decisions from `router.route` (minimum expected cost). `saving` = 1 - cost / cost of approving everything. **(c) Together** - the official scorer on the calibrated routing (seed 0), including the fairness component
(new vs established, low value vs other, non-English vs English; measured on genuine customers only). Baselines give a generic justification, so the justification part of the official score is not meaningful here.
"""),
    code(r"""
official_rows = []
for (name, arm, seed), item in store.items():
    if seed != 0 or name in ('dummy', 'naive_bayes_text'):
        continue
    decisions = route(item['cal'], cfg)
    predictions = pd.DataFrame({'return_id': parts['test']['return_id'], 'decision': decisions, 'confidence': item['cal'],
                                'justification': 'baseline model output without a text explanation'})
    result = official_score(predictions, parts['test'])
    fairness = audit(parts['test'], pd.Series(decisions, index=parts['test'].index), cfg)
    official_rows.append({'model': name, 'arm': arm, 'official_primary': result['primary_score'],
                          'fairness_score': fairness['fairness_score'],
                          'new_vs_established_ratio': fairness['new_vs_established'].get('ratio'),
                          **{f'component_{k}': v for k, v in result['components'].items() if isinstance(v, int | float)}})
official = pd.DataFrame(official_rows).set_index(['model', 'arm']).round(4)
official.to_csv(REPORTS / 'phase5_official_scores.csv')
official
"""),
    md(r"""
## 9. Selection by evidence
Rule, written before looking at results: rank by the **calibrated routing saving** (the official objective). Choose the **simplest** model (`Dummy < LogReg < Tree < RF < ExtraTrees < HistGB < CatBoost`, then arm A < C)
whose saving is within one seed standard deviation (at least 0.005) of the best. Dummy and Naive Bayes are references only and cannot be selected.
"""),
    code(r"""
candidates = agg.drop(index=['dummy', 'naive_bayes_text'], level='model')
best_row = candidates['cal_saving'].idxmax()
tolerance = max(candidates.loc[best_row, 'saving_sd'], 0.005)
eligible = candidates[candidates['cal_saving'] >= candidates.loc[best_row, 'cal_saving'] - tolerance]
arm_rank = {a: i for i, a in enumerate(ARMS)}
chosen = min(eligible.index, key=lambda ix: (order[ix[0]], arm_rank[ix[1]]))
print('best by calibrated saving:', best_row, round(candidates.loc[best_row, 'cal_saving'], 4), '| tolerance', round(tolerance, 4))
print('selected baseline:', chosen)
print('dummy reference (arm A): saving', agg.loc[('dummy', 'A structured'), 'cal_saving'], '| macro-F1', agg.loc[('dummy', 'A structured'), 'macro_f1'])
agg.loc[[chosen]]
"""),
    code(r"""
cm = store[(*chosen, 0)]['report']['confusion']
row_share = cm.div(cm.sum(axis=1), axis=0)
labels_short = [names[l] for l in LABELS]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
fig.subplots_adjust(top=0.8, left=0.3, right=0.95, bottom=0.22)
ax.imshow(row_share.to_numpy(), cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_xticklabels(labels_short, rotation=20, ha='right', fontsize=9)
ax.set_yticks(range(4))
ax.set_yticklabels(labels_short, fontsize=9)
ax.set_xlabel('predicted class', color=MUTED, fontsize=9)
ax.set_ylabel('actual class', color=MUTED, fontsize=9)
for i in range(4):
    for j in range(4):
        value = row_share.iat[i, j]
        ax.text(j, i, f'{cm.iat[i, j]:,}\n{value:.0%}', ha='center', va='center', fontsize=9, color='white' if value > 0.55 else INK)
for side in ax.spines.values():
    side.set_visible(False)
ax.tick_params(colors=MUTED, length=0)
title(fig, f'Where the selected baseline ({chosen[0]}, {chosen[1][0]}) goes wrong', 'Rows sum to 100% of each actual class; first column = predicted abusive.')
fig.savefig(FIG / 'phase5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'genuine requests predicted abusive: {int(cm.iloc[1:, 0].sum()):,} of {int(cm.iloc[1:].to_numpy().sum()):,} ({cm.iloc[1:, 0].sum() / cm.iloc[1:].to_numpy().sum():.1%})')
print(f'abusive requests predicted genuine: {int(cm.iloc[0, 1:].sum()):,} of {int(cm.iloc[0].sum()):,} ({cm.iloc[0, 1:].sum() / cm.iloc[0].sum():.1%})')
"""),
    code(r"""
sel = store[(*chosen, 0)]
sel_decisions = pd.Series(route(sel['cal'], cfg), index=parts['test'].index)
view = parts['test'][['customer_reason', 'label', 'customer_tenure_days', 'prior_orders_count', 'payment_method', 'order_value_inr']].copy()
view['p_abusive_cal'] = sel['cal'].round(3)
view['decision'] = sel_decisions
print('abusive cases that were AUTO-APPROVED (missed):', int(((view.label == 'abusive') & (view.decision == 'auto_approve')).sum()),
      'of', int((view.label == 'abusive').sum()))
display(view[(view.label == 'abusive') & (view.decision == 'auto_approve')].sort_values('p_abusive_cal').tail(6))
print('genuine cases ESCALATED:', int(((view.label != 'abusive') & (view.decision == 'escalate')).sum()))
display(view[(view.label != 'abusive') & (view.decision == 'escalate')].head(6))
print('genuine cases flagged for inspection or escalation, by group (flag rate):')
genuine = view[view.label != 'abusive']
flag = genuine.decision != 'auto_approve'
print(pd.DataFrame({'new (<90d)': [flag[genuine.customer_tenure_days < 90].mean()], 'established': [flag[genuine.customer_tenure_days >= 90].mean()],
                    'low value (<1000)': [flag[genuine.order_value_inr < 1000].mean()], 'other value': [flag[genuine.order_value_inr >= 1000].mean()]}).round(4).to_string(index=False))
"""),
    md(r"""
## 10. Save the selected baseline, comparison and hand-off files
"""),
    code(r"""
validation = parts['test'][['return_id', 'label']].copy().reset_index(drop=True)
for label_index, label in enumerate(LABELS):
    validation[f'p_{label}'] = sel['probs'][:, label_index]
validation['p_abusive_calibrated'] = sel['cal']
validation['decision'] = sel_decisions.to_numpy()
validation.to_csv(REPORTS / 'phase5_selected_validation_predictions.csv', index=False)
all_models = parts['test'][['return_id']].reset_index(drop=True)
for (name, arm, seed), item in store.items():
    if seed == 0:
        all_models[f'{name}|{arm}|raw'] = item['raw']
        all_models[f'{name}|{arm}|cal'] = item['cal']
all_models.to_csv(REPORTS / 'phase5_all_models_validation_p_abusive.csv', index=False)
record = {
    'selected_baseline': {'model': chosen[0], 'arm': chosen[1]},
    'selection_rule': 'simplest model within max(seed sd, 0.005) of the best calibrated routing saving',
    'protocol': 'Phase 2 chronological 70/10/20: train fits, calibration fits isotonic calibrator only, test = validation period',
    'split_boundaries': {'train_last': str(last_train.date()), 'calibration': [str(first_cal.date()), str(last_cal.date())],
                         'test': [str(first_test.date()), str(parts['test'].return_request_date.max().date())]},
    'seeds': list(SEEDS), 'hyperparameters': 'fixed in advance - see r2sorter/baselines.py::build_model',
    'preprocessing': 'r2sorter.preprocess.FeaturePreprocessor fitted on train only', 'text_scores': 'Phase 4 word TF-IDF + LogReg, train rows out-of-fold',
    'hidden_test_labels_used': False, 'versions': versions,
    'results': agg.loc[[chosen]].reset_index().to_dict('records')[0]}
(REPORTS / 'phase5_selected_baseline.json').write_text(json.dumps(record, indent=2, default=str))
print(sorted(p.name for p in REPORTS.glob('phase5_*')))
"""),
    md(r"""
## 11. Limitations (read before using the numbers)
* One chronological validation window (about 8,500 requests, about 700 abusive). Differences smaller than the seed spread should not be read as real.
* Isotonic calibration uses a 4,400-row slice; calibrated cost numbers are noisy and optimistic relative to a fully separate evaluation. Calibration and uncertainty get a proper treatment in the later phases.
* The selection rule uses the same test window that is reported, so the selected model's figures are slightly optimistic.
* Hyper-parameters are fixed, not tuned: tuned versions (Phase 6) may change the ordering of the tree models.
* The text-reuse features (arm C) are computed from earlier requests only, but depend on the Phase 2 same-day ordering caveat.
* Fairness is audited, not enforced here; baselines give generic justifications, so the explanation part of the official score is not meaningful.
* **Hand-off to Phase 6:** `phase5_model_comparison.csv` is the reference table; an advanced model should beat the selected baseline on the same split, with the same calibration protocol, by more than the seed spread.
"""),
    md(r"""
## 12. Completion checklist (computed)
"""),
    code(r"""
trained = set(results['model'])
checklist = {
    'Data and target mapping verified': len(LABELS) == 4 and set(train['label']) == set(LABELS),
    'Leakage-safe splits confirmed': bool(last_train < first_cal and last_cal < first_test),
    'Dummy Classifier and required baselines trained': {'dummy', 'logreg', 'decision_tree'} <= trained,
    'Recommended tree ensembles compared': {'random_forest', 'extra_trees', 'hist_gb'} <= trained,
    'All models evaluated on the same validation period': results.groupby('model').size().gt(0).all(),
    'Class-wise metrics and confusion matrices reviewed': (REPORTS / 'phase5_confusion_matrices.csv').exists(),
    'Business-cost score calculated and checked': 'raw_saving_vs_approve_all' in results and 'cal_saving_vs_approve_all' in results,
    'Probability calibration limitations documented': (REPORTS / 'phase5_reliability_logreg.csv').exists(),
    'Best baseline selected based on evidence': chosen in agg.index,
    'Results prepared for Phase 6': (REPORTS / 'phase5_selected_baseline.json').exists(),
    'Hidden-test labels not used': 'label' not in test.columns,
}
done = pd.Series(checklist).map({True: 'done', False: 'NOT DONE'}).to_frame('status')
assert (done['status'] == 'done').all(), done
done
"""),
]


def main() -> None:
    notebook = new_notebook(cells=CELLS)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(notebook, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
