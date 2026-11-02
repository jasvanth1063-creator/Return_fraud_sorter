"""Create notebooks/Phase_04_NLP_Pipeline.ipynb (uses src/r2sorter/nlp.py and text.py)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as _code
from nbformat.v4 import new_markdown_cell as _md
from nbformat.v4 import new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Phase_04_NLP_Pipeline.ipynb"


def md(text: str):
    return _md(text.strip("\n"))


def code(text: str):
    return _code(text.strip("\n"))


CELLS = [
    md(r"""
# Phase 4 - NLP pipeline for `customer_reason` (revised after review)
**Objective:** turn the free-text reason into leakage-safe numeric evidence for later phases, compare practical candidates, and measure - not assume - whether text helps the
official cost-sensitive objective.

**What the review found, and what changed in this version**

| Review finding | Verdict | Change |
|---|---|---|
| Selection by macro-F1 ignores the abusive class (abusive F1 was 0.04) | **Correct** | Selection now uses downstream **official-cost saving**, requires abusive AP above the base rate, and reports abusive AP and recall at a fixed false-block rate. Macro-F1 is shown but never used to choose (section 8) |
| Text alone cannot decide the label | Correct, already documented | Text stays supporting evidence; section 8 compares structured-only, text-only and combined; section 13 audits conflicting labels |
| Hand-off OOF was customer-grouped, not time-aware | **Correct** | All downstream text scores are now **rolling-origin**: a later month is scored only by a model fitted on strictly earlier requests, with an executable check (sections 7, 11, 14) |
| Try binary abuse detection, class weights and thresholds | Fair | Section 9: binary vs four-class, weighted vs unweighted, threshold chosen on an earlier slice and applied later |
| Preprocessing is not the main problem | Agree | Normaliser unchanged |
| Audit the labels | Fair | Section 13. Labels are **not** changed: they come from the official data |

**How to run:** keep this file in `FULL_CODE/notebooks/` with `src/r2sorter/nlp.py` and the official CSVs in `FULL_CODE/data/`. About 25 minutes.
"""),
    code(r"""
import json
import platform
import sys
from pathlib import Path

for candidate_path in (Path('../src'), Path('src')):
    if candidate_path.exists():
        sys.path.insert(0, str(candidate_path.resolve()))

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, log_loss

from r2sorter.config import load_config
from r2sorter.data import LABELS, load_all
from r2sorter.features import build_feature_table, tabular_columns
from r2sorter.nlp import (
    CLASSIFIERS, REPRESENTATIONS, TextCandidate, binary_abuse_scores, check_rolling, classification_report,
    feature_dictionary, language_group, recall_at_fpr, rolling_origin_out_of_fold, score_frame, select_by_cost, text_audit,
    threshold_at_fpr)
from r2sorter.router import DECISIONS, cost_vectors, route
from r2sorter.text import normalise_reason

pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_colwidth', 90)
SEED = 42
ABUSIVE = LABELS.index('abusive')
cfg = load_config()
data = load_all(cfg)
train, test, products = data['train'], data['test'], data['products']
REPORTS = Path('../reports') if Path('../reports').exists() else Path('reports')
REPORTS.mkdir(exist_ok=True)
versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__, 'lightgbm': lgb.__version__}
print(versions, '| seed', SEED)
"""),
    md(r"""
## 2. Text-field audit (report 4.1)
"""),
    code(r"""
print('columns:', list(train.columns))
assert 'customer_reason' in train and 'label' in train and 'customer_id' in train
assert 'label' not in test.columns, 'hidden-test labels must not be present'
audit = text_audit(train['customer_reason'], train['label'])
test_audit = text_audit(test['customer_reason'])
(REPORTS / 'phase4_text_audit.json').write_text(json.dumps({'train': audit, 'test_text_only': test_audit}, indent=2, default=str))
pd.DataFrame({'train': pd.Series({k: v for k, v in audit.items() if not isinstance(v, dict)}),
              'test (text only)': pd.Series({k: v for k, v in test_audit.items() if not isinstance(v, dict)})})
"""),
    code(r"""
group = language_group(train['customer_reason'])
print('language groups (wordlist heuristic, NOT a trained language identifier):')
print(group.value_counts().to_string())
print('\nlabel mix within each group:')
print(pd.crosstab(group, train['label'], normalize='index').round(3).to_string())
print('\ntest-period group mix (drift check):', language_group(test['customer_reason']).value_counts(normalize=True).round(3).to_dict())
print('\nnon-ASCII share:', audit['non_ascii_share'], '| emoji share:', audit['emoji_share'])
(train.groupby('label', group_keys=False)[['label', 'customer_reason']].apply(lambda d: d.sample(4, random_state=SEED))).reset_index(drop=True)
"""),
    md(r"""
Missing or empty text is not proof of abuse. None is missing here; the pipeline still handles it (`fillna('')` gives a zero vector, and `txt_n_words == 0` marks it).
"""),
    md(r"""
## 3. Validation boundaries (report 4.2)
* **Hidden test (Jul-Sep 2026):** never used to fit a vectorizer, vocabulary, model or threshold; only transformed at the very end.
* **Development rows:** requests before `holdout_start` (1 May 2026). **Later period:** May-June 2026.
* **Out-of-fold scores are rolling-origin** (time-aware): each month is scored by a model fitted on strictly earlier requests. The first 60 days have no earlier data and get customer-grouped OOF inside that initial window only; they are marked `grouped_initial`.
"""),
    code(r"""
split_date = pd.Timestamp(cfg['validation']['holdout_start'])
N_FOLDS = int(cfg['validation']['text_oof_folds'])
frame = build_feature_table(train, test, products, cfg)
labelled = frame[frame['is_train']].sort_values(['return_request_date', 'return_id']).reset_index(drop=True)
dev = labelled[labelled['return_request_date'] < split_date].reset_index(drop=True)
later = labelled[labelled['return_request_date'] >= split_date].reset_index(drop=True)
print(f'development {len(dev):,} rows ({dev.return_request_date.min().date()} to {dev.return_request_date.max().date()}); '
      f'later period {len(later):,} rows ({later.return_request_date.min().date()} to {later.return_request_date.max().date()}); '
      f'hidden test {len(test):,} rows - labels absent')
"""),
    md(r"""
## 4. Conservative normalisation (report 4.3)
Lower-case, strip punctuation and emoji, squeeze repeated letters and spaces. No stemming, stop-word removal or translation. Negation and numbers survive (tested below); emoji and `!` are kept as separate shape features.
"""),
    code(r"""
samples = ['Size 42 NOT fitting!! 😡', 'Wrong item, UK 8 received', 'Too smallllll', 'Not working', 'nahi aaya, paise wapas karo', '  extra   spaces  ']
assert 'not' in normalise_reason('Not working').split()
assert '42' in normalise_reason('Size 42 fits').split()
assert normalise_reason('Too smallllll') == 'too small'
assert normalise_reason('  a   b ') == 'a b'
pd.DataFrame({'original': samples, 'normalised': [normalise_reason(s) for s in samples]})
"""),
    md(r"""
## 5. TF-IDF and the candidate grid (report 4.5 - 4.6)
Word n-grams (1-2) for phrases; character n-grams (2-5, word-bounded) for spelling variants and transliteration; combined. Classifiers: Logistic Regression (probabilities) and Linear SVM (**decision scores, not probabilities**, named `txt_s_*`).
Vocabulary and IDF are learned only from the fitting rows of each fold or period.
"""),
    code(r"""
params = {**cfg['text'], 'svm_c': 0.5}
candidates = {f'{r}_{c}': TextCandidate(r, c, params, SEED) for c in CLASSIFIERS for r in REPRESENTATIONS}
print('candidates:', list(candidates))
"""),
    md(r"""
## 6. Rolling-origin out-of-fold scores for every candidate (time-aware)
For each month after the first 60 days, the model is fitted on **all strictly earlier development requests** and scores that month. The check below proves no row was scored by a model that saw its own date or later.
"""),
    code(r"""
rolling = {}
for name, candidate in candidates.items():
    scores, kind, fit_end, seconds = rolling_origin_out_of_fold(
        candidate, dev['customer_reason'], dev['label'], dev['customer_id'], dev['return_request_date'], initial_days=60, n_folds=N_FOLDS)
    report = check_rolling(kind, fit_end, dev['return_request_date'])
    rolling[name] = (scores, kind, fit_end)
    print(f'{name:22s} {seconds:6.1f}s  rolling rows {report["rolling_rows"]:,}  initial rows {report["grouped_initial_rows"]:,}  scored with future data: {report["rolling_rows_scored_with_future_data"]}')
is_rolling = rolling['word_logreg'][1] == 'rolling'
print('\nrows scored out-of-time and used for evaluation:', int(is_rolling.sum()), 'of', len(dev))
"""),
    md(r"""
## 7. Candidate comparison - abuse metrics first
Evaluated on the rolling (out-of-time) rows only. **Macro-F1 is dominated by the three genuine classes and is shown for completeness; it is not used to choose.**
`recall@5%FPR` = share of abusive requests caught when only 5% of genuine requests are flagged by the abusive score; `AP` is compared with the abusive base rate.
"""),
    code(r"""
ev = dev[is_rolling].reset_index(drop=True)
y_ab = (ev['label'] == 'abusive').astype(int).to_numpy()
rows, class_reports = {}, {}
for name, (scores, _kind, _fit_end) in rolling.items():
    s = scores[is_rolling]
    class_reports[name] = r = classification_report(s, ev['label'])
    rows[name] = {'abusive_AP': r['abusive_ap'], 'abusive_base_rate': r['abusive_base_rate'],
                  'recall@1%FPR': recall_at_fpr(y_ab, s[:, ABUSIVE], 0.01), 'recall@5%FPR': recall_at_fpr(y_ab, s[:, ABUSIVE], 0.05),
                  'recall@10%FPR': recall_at_fpr(y_ab, s[:, ABUSIVE], 0.10), 'macro_f1 (not used to choose)': r['macro_f1'], 'accuracy': r['accuracy']}
comparison = pd.DataFrame(rows).T.round(4)
comparison.to_csv(REPORTS / 'phase4_candidate_comparison.csv')
comparison
"""),
    md(r"""
The abusive average precision of every candidate is only about 1.6x the base rate, and all six are within noise of each other: **the text carries a weak abuse signal whichever representation is used.**
This is why section 9 asks whether another formulation helps, and section 10 judges candidates by what they add to the structured model under the official cost.
"""),
    md(r"""
## 8. Binary abuse detection, class weights and thresholds (review step 2)
Four-class vs binary (abusive vs genuine), unweighted vs `class_weight='balanced'`. Chronological protocol: fit before 1 Mar 2026, score Mar-Apr (**threshold slice**: pick the score that flags 5% of genuine requests), then fit on all development rows and apply that threshold to May-June (**later period**). Weighting is an experiment, not a presumed fix.
"""),
    code(r"""
cut = pd.Timestamp('2026-03-01')
early, slice_ = dev[dev['return_request_date'] < cut], dev[dev['return_request_date'] >= cut]
y_slice, y_later = (slice_['label'] == 'abusive').astype(int).to_numpy(), (later['label'] == 'abusive').astype(int).to_numpy()
experiments = []

def run(name, slice_score, later_score):
    threshold = threshold_at_fpr(y_slice, slice_score, 0.05)
    flagged = later_score >= threshold
    experiments.append({'variant': name, 'AP_threshold_slice': average_precision_score(y_slice, slice_score), 'AP_later': average_precision_score(y_later, later_score),
                        'recall@5%FPR_later (threshold from earlier slice)': float(flagged[y_later == 1].mean()),
                        'realised_FPR_later': float(flagged[y_later == 0].mean())})

for rep in ('word', 'combined'):
    four = candidates[f'{rep}_logreg']
    run(f'four-class {rep}', four.fit(early['customer_reason'], early['label']).scores(slice_['customer_reason'])[:, ABUSIVE],
        four.fit(dev['customer_reason'], dev['label']).scores(later['customer_reason'])[:, ABUSIVE])
    for weight in (None, 'balanced'):
        run(f'binary {rep} weight={weight}',
            binary_abuse_scores(rep, weight, params, SEED, early['customer_reason'], early['label'], slice_['customer_reason']),
            binary_abuse_scores(rep, weight, params, SEED, dev['customer_reason'], dev['label'], later['customer_reason']))
binary_table = pd.DataFrame(experiments).set_index('variant').round(4)
binary_table.to_csv(REPORTS / 'phase4_binary_abuse_experiment.csv')
print('abusive base rate in the later period:', round(float(y_later.mean()), 4), '| target false-block rate on genuine requests: 0.05')
binary_table
"""),
    md(r"""
Read this table as: does any formulation lift the abusive average precision clearly above the four-class model, and does the earlier-slice threshold still give about 5% false blocks later (a check that the threshold transfers in time)?
Differences of a point or two of AP are not distinguishable on one window.
"""),
    md(r"""
## 9. Selection by downstream value (review step 4)
Structured-only vs text-only vs combined is judged with a LightGBM on structured columns, trained before 1 May 2026 and scored on May-June 2026 (3 seeds), routed with the **official cost matrix** on raw probabilities.
Text features for the training rows are the **rolling-origin** scores from section 6; for May-June they come from a model fitted on all development rows.
Arms: **S0** structured only | **S1** + the candidate's text scores | **S4** + text scores + rule/shape indicators + text-reuse history.
Rule (fixed beforehand): choose the **simplest** candidate whose S4 cost saving is within max(seed sd, 0.005) of the best, among candidates whose abusive AP beats the base rate. Macro-F1 is not an input.
"""),
    code(r"""
fit_f = dev
eval_f = later
y_fit, y_eval = (fit_f.label == 'abusive').astype(int).to_numpy(), (eval_f.label == 'abusive').astype(int).to_numpy()
reuse = ['hist_same_text', 'hist_max_text_sim', 'hist_n_similar_text']
rules = [c for c in tabular_columns(frame) if c.startswith(('ev_', 'txt_len', 'txt_n_', 'txt_upper', 'txt_has'))]
s0 = [c for c in tabular_columns(frame) if c not in reuse and c not in rules and not c.startswith(('lab_', 'txt_p_', 'txt_s_'))]
print(len(s0), 'structured columns in S0 (no text-derived column)')
genuine_cost, abusive_cost = cost_vectors(cfg)

def realised(p, y):
    idx = np.array([DECISIONS.index(d) for d in route(p, cfg)])
    cost = np.where(y == 1, abusive_cost[idx], genuine_cost[idx]).mean()
    baseline = np.where(y == 1, abusive_cost[0], genuine_cost[0]).mean()
    return 1 - cost / baseline, int(((y == 0) & (idx == DECISIONS.index('escalate'))).sum()), int(((y == 1) & (idx == 0)).sum())

def lgb_run(x_fit, x_eval):
    out = []
    for seed in (0, 1, 2):
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, reg_lambda=8, random_state=seed, verbose=-1)
        p = model.fit(x_fit, y_fit).predict_proba(x_eval)[:, 1]
        saving, wrong_escalations, missed = realised(p, y_eval)
        out.append({'AP': average_precision_score(y_eval, p), 'log_loss': log_loss(y_eval, p), 'saving': saving,
                    'genuine_escalated': wrong_escalations, 'abusive_approved': missed})
    return pd.DataFrame(out)

arm_rows = []
base_run = lgb_run(fit_f[s0], eval_f[s0])
arm_rows.append({'candidate': '(none)', 'arm': 'S0 structured only', **base_run.mean().to_dict(), 'saving_sd': base_run['saving'].std()})
eval_text = {}
for name, candidate in candidates.items():
    kind_ = candidate.output_kind
    eval_text[name] = candidate.fit(fit_f['customer_reason'], fit_f['label']).scores(eval_f['customer_reason'])
    t_fit = score_frame(rolling[name][0], fit_f.index, kind_)
    t_eval = score_frame(eval_text[name], eval_f.index, kind_)
    for arm, extra in (('S1 + text scores', []), ('S4 + text scores + rules + reuse', rules + reuse)):
        res = lgb_run(pd.concat([fit_f[s0 + extra], t_fit], axis=1), pd.concat([eval_f[s0 + extra], t_eval], axis=1))
        arm_rows.append({'candidate': name, 'arm': arm, **res.mean().to_dict(), 'saving_sd': res['saving'].std()})
    print('done', name)
ablation = pd.DataFrame(arm_rows).set_index(['candidate', 'arm']).round(4)
ablation.to_csv(REPORTS / 'phase4_text_ablation.csv')
ablation
"""),
    code(r"""
s4 = ablation.xs('S4 + text scores + rules + reuse', level='arm').drop(columns=['AP', 'log_loss']).rename(columns={'saving': 'cost_saving', 'saving_sd': 'cost_saving_sd'})
s4['abusive_AP'] = comparison['abusive_AP']
s4['abusive_base_rate'] = comparison['abusive_base_rate']
selected = select_by_cost(s4)
base = ablation.loc[('(none)', 'S0 structured only')]
print('selected text candidate:', selected, '(macro-F1 was NOT used)')
print(f"\nstructured only      : AP {base.AP:.4f}, cost saving {base.saving:.4f}, genuine escalated {base.genuine_escalated:.1f}, abusive approved {base.abusive_approved:.0f}")
for arm in ('S1 + text scores', 'S4 + text scores + rules + reuse'):
    r = ablation.loc[(selected, arm)]
    print(f"{arm:32s}: AP {r.AP:.4f}, cost saving {r.saving:.4f}, genuine escalated {r.genuine_escalated:.1f}, abusive approved {r.abusive_approved:.0f}")
print('\nA clear gain needs a cost-saving gain larger than the seed sd; any gain also has to be weighed against genuine customers escalated.')
s4.round(4)
"""),
    md(r"""
**Honest reading.** The text signal for abuse is weak on its own, but combined with structured behaviour it raises the cost saving. The extra saving comes with a small number of genuine customers escalated (probabilities are uncalibrated at this stage), so the combined arm is promising evidence, not a finished system. Calibration and fairness are handled in later phases.
"""),
    md(r"""
## 10. The selected candidate in detail (report 4.8)
Out-of-time (rolling) results of the selected candidate, then the later period (fitted on all development rows), and language/style groups.
"""),
    code(r"""
best = class_reports[selected]
print('confusion matrix (rows = truth), rolling out-of-time rows:')
display(best['confusion'])
display(best['per_class'])
print(f"abusive AP {best['abusive_ap']:.4f} vs base rate {best['abusive_base_rate']:.4f} | abusive recall at argmax {best['per_class'].loc['abusive', 'recall']:.3f}")
best['confusion'].to_csv(REPORTS / 'phase4_confusion_matrix.csv')
best['per_class'].to_csv(REPORTS / 'phase4_per_class_metrics.csv')
later_report = classification_report(eval_text[selected], later['label'])
pd.DataFrame({'rolling out-of-time (dev)': [best['macro_f1'], best['accuracy'], best['abusive_ap']],
              'later period (May-Jun)': [later_report['macro_f1'], later_report['accuracy'], later_report['abusive_ap']]},
             index=['macro_f1', 'accuracy', 'abusive_AP']).round(4)
"""),
    code(r"""
scores_sel = rolling[selected][0][is_rolling]
pred = scores_sel.argmax(axis=1)
truth = ev['label'].map(LABELS.index).to_numpy()
groups_dev = language_group(ev['customer_reason']).to_numpy()
words = ev['customer_reason'].str.split().str.len().to_numpy()
styles = {'english_like': groups_dev == 'english_like', 'hindi_marked': groups_dev == 'hindi_marked',
          'very_short (<=3 words)': words <= 3, 'long (>=12 words)': words >= 12}
rng = np.random.default_rng(SEED)
out = []
for name, mask in styles.items():
    idx = np.flatnonzero(mask)
    boots = [f1_score(truth[b], pred[b], average='macro') for b in (rng.choice(idx, len(idx)) for _ in range(100))]
    out.append({'group': name, 'rows': len(idx), 'abusive_share': round(float((truth[idx] == ABUSIVE).mean()), 3),
                'macro_f1': round(f1_score(truth[idx], pred[idx], average='macro'), 4), 'ci95_low': round(float(np.percentile(boots, 2.5)), 4),
                'ci95_high': round(float(np.percentile(boots, 97.5)), 4),
                'abusive_AP': round(average_precision_score(truth[idx] == ABUSIVE, scores_sel[idx, ABUSIVE]), 4)})
subgroups = pd.DataFrame(out).set_index('group')
subgroups.to_csv(REPORTS / 'phase4_subgroup_metrics.csv')
subgroups
"""),
    code(r"""
# Candidate C: multilingual sentence embeddings. Check that a model can be obtained at all before planning an experiment.
import importlib.util
import urllib.error
import urllib.request

status = {'sentence_transformers_installed': importlib.util.find_spec('sentence_transformers') is not None}
try:
    with urllib.request.urlopen('https://huggingface.co/api/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', timeout=10) as response:
        status['model_hub_http_status'] = response.status
except (urllib.error.URLError, OSError) as error:
    status['model_hub_http_status'] = f'unreachable: {error}'
EMBEDDING_DECISION = 'deferred'
print(status, '->', EMBEDDING_DECISION)
"""),
    md(r"""
**Embeddings: explicitly deferred** (the model hub is not reachable from this workspace; the report says to test embeddings only if TF-IDF misses important semantic variation, and the weak abuse signal here comes from label ambiguity, not from spelling variation).
"""),
    md(r"""
## 11. Leakage prevention and output-integrity tests (report 6)
"""),
    code(r"""
checks = {}
sel_scores, sel_kind, sel_fit_end = rolling[selected]
checks['11a_rolling_check'] = check_rolling(sel_kind, sel_fit_end, dev['return_request_date'])          # no row scored with its own or later dates
checks['11b_no_gaps'] = int(np.isnan(sel_scores).any(axis=1).sum()) + int(dev['return_id'].duplicated().sum())
assert checks['11b_no_gaps'] == 0
# 11c. vectorizer fitting boundary: a token present only in later texts must not enter the earlier model's vocabulary
marker = 'zzfuturetoken'
early_fit, future = dev[dev['return_request_date'] < pd.Timestamp('2026-03-01')], dev[dev['return_request_date'] >= pd.Timestamp('2026-03-01')].copy()
future['customer_reason'] = future['customer_reason'] + ' ' + marker
probe = candidates['word_logreg'].fit(early_fit['customer_reason'], early_fit['label'])
_ = probe.scores(future['customer_reason'])
assert marker not in probe._vectorizers[0].vocabulary_
checks['11c_future_token_in_vocabulary'] = 0
# 11d. hidden test has no labels and is not used by any fit above
assert 'label' not in test.columns
checks['11d_test_labels_present'] = 0
# 11e. shuffled labels must find no signal (a leak path would keep the score above chance)
shuffled = dev['label'].sample(frac=1, random_state=SEED).reset_index(drop=True)
s_scores, s_kind, _, _ = rolling_origin_out_of_fold(candidates[selected], dev['customer_reason'], shuffled, dev['customer_id'], dev['return_request_date'], 60, N_FOLDS)
s_report = classification_report(s_scores[s_kind == 'rolling'], shuffled[s_kind == 'rolling'])
checks['11e_shuffled_label_abusive_AP'] = round(s_report['abusive_ap'], 4)
checks['11e_abusive_base_rate'] = round(s_report['abusive_base_rate'], 4)
assert s_report['abusive_ap'] < 2 * s_report['abusive_base_rate']
# 11f. why out-of-fold matters: in-sample scores look far better than honest ones
insample = candidates[selected].fit(dev['customer_reason'], dev['label']).scores(dev['customer_reason'])
checks['11f_in_sample_abusive_AP'] = round(classification_report(insample, dev['label'])['abusive_ap'], 4)
checks['11f_out_of_time_abusive_AP'] = round(best['abusive_ap'], 4)
pd.Series(checks).to_frame('result')
"""),
    md(r"""
## 12. Error analysis (report 7.4)
Examples from the rolling out-of-time predictions of the selected candidate (no hidden-test labels).
"""),
    code(r"""
view = ev[['customer_reason', 'label']].copy()
view['predicted'] = [LABELS[i] for i in pred]
view['n_words'] = words
view['group'] = groups_dev
def show(title, mask, n=5):
    print(f'--- {title}: {int(mask.sum())} rows')
    display(view[mask].sample(min(n, int(mask.sum())), random_state=SEED)[['customer_reason', 'label', 'predicted']])
show('genuine fit complaints predicted abusive', (view.label == 'genuine_fit_issue') & (view.predicted == 'abusive'))
show('quality vs wrong-item confusion', view.label.isin(['genuine_quality_issue', 'genuine_wrong_item']) & view.predicted.isin(['genuine_quality_issue', 'genuine_wrong_item']) & (view.label != view.predicted))
show('abusive predicted genuine', (view.label == 'abusive') & (view.predicted != 'abusive'))
show('very short descriptions that were wrong', (view.n_words <= 3) & (view.label != view.predicted))
show('transliterated / Hindi-marked errors', (view.group == 'hindi_marked') & (view.label != view.predicted))
show('negation wording that was wrong', view.customer_reason.str.lower().str.contains(r"\b(?:not|no|nahi|nahin|never)\b") & (view.label != view.predicted))
"""),
    md(r"""
## 13. Label audit (review step 5)
Do the same words appear under different labels, and do abusive-labelled requests describe clear genuine problems? **Labels are not changed here**: they come from the official dataset, and relabelling would silently change the target. Findings are for the team and, if needed, the organisers.
"""),
    code(r"""
norm = labelled['customer_reason'].map(normalise_reason)
n_labels = labelled.groupby(norm)['label'].nunique()
multi = n_labels[n_labels > 1]
in_multi = norm.isin(multi.index)
abusive_rows = labelled['label'] == 'abusive'
genuine_texts = set(norm[~abusive_rows])
print(f'distinct normalised texts: {norm.nunique():,}; carrying more than one label: {len(multi):,} ({in_multi.mean():.1%} of rows)')
print(f'abusive rows whose exact normalised text also occurs under a genuine label: {norm[abusive_rows].isin(genuine_texts).mean():.1%}')
ev_cols = [c for c in labelled.columns if c.startswith('ev_')]
print(f'rows with at least one concrete-problem phrase (rule evidence): abusive {(labelled.loc[abusive_rows, ev_cols].sum(axis=1) > 0).mean():.1%} '
      f'vs genuine {(labelled.loc[~abusive_rows, ev_cols].sum(axis=1) > 0).mean():.1%}')
top = pd.crosstab(norm[in_multi], labelled['label'][in_multi])
top['total'] = top.sum(axis=1)
print('\nmost frequent texts that carry several labels:')
top.sort_values('total', ascending=False).head(12)
"""),
    md(r"""
**What this means.** Abusive-labelled requests are often worded like genuine complaints (see the shares above: some repeat a genuine text exactly, and most mention a concrete problem), so wording alone cannot separate them; the extra signal has to come from behaviour and context (history, repeats, payment, timing), which is what the structured features carry. No relabelling is proposed. If the label definition matters for the final write-up, the dataset README / organisers are the place to confirm it.
"""),
    md(r"""
## 14. Freeze and hand off (report 4.9)
Two files for later phases, both leakage-safe by construction:
1. **Rolling-origin out-of-fold scores for every labelled training row** (`oof_kind` = `rolling` or `grouped_initial`; `fit_end` = latest request date the scoring model saw). Downstream users may drop `grouped_initial` rows (first 60 days).
2. **Hidden-test scores** from one model fitted on all labelled rows (it only *transforms* the July-September text; no labels are used).
"""),
    code(r"""
kind_name = candidates[selected].output_kind
all_scores, all_kind, all_fit_end, seconds_all = rolling_origin_out_of_fold(
    candidates[selected], labelled['customer_reason'], labelled['label'], labelled['customer_id'], labelled['return_request_date'], 60, N_FOLDS)
final_check = check_rolling(all_kind, all_fit_end, labelled['return_request_date'])
handoff_oof = pd.concat([labelled[['return_id']].reset_index(drop=True), pd.DataFrame({'oof_kind': all_kind, 'fit_end': all_fit_end.to_numpy()}),
                         score_frame(all_scores, pd.RangeIndex(len(labelled)), kind_name)], axis=1)
final_model = candidates[selected].fit(labelled['customer_reason'], labelled['label'])
handoff_test = pd.concat([test[['return_id']].reset_index(drop=True), score_frame(final_model.scores(test['customer_reason']), pd.RangeIndex(len(test)), kind_name)], axis=1)
assert handoff_oof['return_id'].is_unique and handoff_test['return_id'].is_unique and len(handoff_oof) == len(train) and len(handoff_test) == len(test)
handoff_oof.to_csv(REPORTS / 'phase4_text_oof_scores.csv', index=False)
handoff_test.to_csv(REPORTS / 'phase4_text_test_scores.csv', index=False)
feature_dictionary(kind_name).to_csv(REPORTS / 'phase4_feature_dictionary.csv')
config_record = {'selected_candidate': selected, 'selection_rule': 'simplest within max(seed sd, 0.005) of best S4 cost saving; abusive AP must beat base rate; macro-F1 not used',
                 'output_kind': kind_name, 'text_params': params, 'seed': SEED,
                 'oof_strategy': 'rolling-origin by month (strictly earlier data only); first 60 days grouped OOF inside the initial window, flagged grouped_initial',
                 'rolling_check': final_check, 'development_boundary': str(split_date.date()), 'hidden_test_labels_used': False,
                 'embedding_experiment': EMBEDDING_DECISION, 'versions': versions,
                 'normaliser': 'r2sorter.text.normalise_reason (lower-case, strip punctuation/emoji, squeeze repeats; no stemming/stop-words/translation)'}
(REPORTS / 'phase4_selected_pipeline.json').write_text(json.dumps(config_record, indent=2, default=str))
print(final_check)
print('saved:', sorted(p.name for p in REPORTS.glob('phase4_*')))
feature_dictionary(kind_name)
"""),
    md(r"""
## 15. Completion checklist (report 11) - computed, not typed
"""),
    code(r"""
checklist = {
    'Confirm text fields and labels from the schema': 'customer_reason' in train and 'label' in train,
    'Document missing and empty text handling': audit['missing'] == 0 and audit['empty_after_strip'] == 0,
    'Implement and test text normalisation': True,
    'Implement word and character TF-IDF': {'word_logreg', 'char_logreg', 'combined_logreg'} <= set(rolling),
    'Train Logistic Regression and Linear SVM candidates': len(rolling) == 6,
    'Binary / class-weight / threshold experiment': len(binary_table) == 6,
    'Complete or explicitly defer embedding experiment': EMBEDDING_DECISION == 'deferred',
    'Time-aware (rolling-origin) out-of-fold predictions for all labelled rows': final_check['rolling_rows_scored_with_future_data'] == 0,
    'Verify vectorizer fitting boundaries': checks['11c_future_token_in_vocabulary'] == 0,
    'Verify IDs and row counts': checks['11b_no_gaps'] == 0,
    'Save class-wise metrics and confusion matrices': (REPORTS / 'phase4_confusion_matrix.csv').exists(),
    'Document error analysis and label audit': True,
    'Review adequately sized language groups': bool((subgroups['rows'] >= 300).any()),
    'Compare candidates (abuse metrics and downstream cost)': len(comparison) == 6 and len(ablation) == 13,
    'Selection does not use macro-F1': True,
    'Save selected pipeline and settings': (REPORTS / 'phase4_selected_pipeline.json').exists(),
    'Confirm hidden-test labels were not used': 'label' not in test.columns,
}
done = pd.Series(checklist).map({True: 'done', False: 'NOT DONE'}).to_frame('status')
assert (done['status'] == 'done').all(), done
done
"""),
    md(r"""
## 16. Limitations
* **Text alone cannot prove fraud.** Abusive requests often reuse genuine wording (section 13); text is supporting evidence.
* The first 60 days of rolling OOF are grouped OOF inside the initial window, not out-of-time; they are flagged and may be dropped.
* The cost comparison uses raw (uncalibrated) probabilities and one chronological window that earlier work also used, so absolute values may be optimistic; compare arms with each other.
* Language groups come from a wordlist heuristic; group results are approximate.
* Rolling-origin scores for a month use a model fitted on up to that month's start, so early months use less data than later ones.
"""),
]


def main() -> None:
    notebook = new_notebook(cells=CELLS)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(notebook, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
