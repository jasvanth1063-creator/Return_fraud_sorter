"""Create notebooks/Phase_02_Splitting_and_Leakage.ipynb (uses the project's own src/r2sorter code)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Phase_02_Splitting_and_Leakage.ipynb"

CELLS = [
    md(
        "# Phase 2 - Data splitting, preprocessing and leakage prevention\n"
        "**Objective:** create reliable training, calibration and test data, and *test* with executable checks "
        "whether anything from the future reaches a model, a calibrator or a threshold. The checks can find leaks; they cannot "
        "guarantee there are none (see section 12).\n\n"
        "This notebook answers the review of Phase 2 point by point:\n\n"
        "| Review item | Where it is answered |\n|---|---|\n"
        "| 70/10/20 chronological split from the plan | section 2 (implemented) and section 9 (a full experiment on it) |\n"
        "| Why our project split differs, and is the test set untouched? | sections 3-4 |\n"
        "| Exact split dates, row counts and class mix | sections 2-3 |\n"
        "| Duplicate checks across splits | section 5 |\n"
        "| History features use only the past | section 6 (three executable tests) |\n"
        "| Customer and text-template leakage | section 7 |\n"
        "| Catalogue fields computed from labels | section 8 |\n"
        "| Calibration and thresholds chosen without test labels | section 10 |\n"
        "| Official scoring formula | section 11 |\n"
        "| Reproducible pipeline | the code lives in `src/r2sorter` (`splits.py`, `features.py`, `text.py`, `model.py`); this notebook calls it |\n\n"
        "**How to run:** keep this file in `FULL_CODE/notebooks/` of the project (it imports `../src/r2sorter`) with the five official "
        "CSVs in `FULL_CODE/data/`. Run with the project environment (`uv sync`). Total run time is about 6-8 minutes."
    ),
    code(
        "import inspect\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "for candidate in (Path('../src'), Path('src')):\n"
        "    if candidate.exists():\n"
        "        sys.path.insert(0, str(candidate.resolve()))\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.metrics import average_precision_score\n"
        "from sklearn.model_selection import GroupKFold\n"
        "\n"
        "from r2sorter.calibration import AbuseCalibrator\n"
        "from r2sorter.config import load_config\n"
        "from r2sorter.data import LABELS, load_all\n"
        "from r2sorter.features import (\n"
        "    LABEL_HISTORY_COLUMNS, add_label_history, build_feature_table, tabular_columns)\n"
        "from r2sorter.model import StackModel\n"
        "from r2sorter.router import break_even_thresholds, route\n"
        "from r2sorter.scoring import load_official_scorer, official_score\n"
        "from r2sorter.splits import assert_chronological, chronological_split, describe_splits\n"
        "from r2sorter.text import TextReasonModel, normalise_reason, out_of_fold_text_probs\n"
        "\n"
        "pd.set_option('display.max_columns', 40, 'display.width', 170)\n"
        "plt.rcParams.update({'figure.figsize': (8, 3.4), 'axes.spines.top': False, 'axes.spines.right': False})\n"
        "\n"
        "cfg = load_config()\n"
        "data = load_all(cfg)\n"
        "frame = build_feature_table(data['train'], data['test'], data['products'], cfg)  # train + test in time order\n"
        "labelled = frame[frame['is_train']]\n"
        "test_rows = frame[~frame['is_train']]\n"
        "ABUSIVE = LABELS.index('abusive')\n"
        "print('seed:', cfg['seed'], '| labelled rows:', len(labelled), '| final-test rows (labels hidden):', len(test_rows))"
    ),
    md(
        "## 1. Principles that every split below must satisfy\n"
        "1. **Chronology:** training < calibration < evaluation, strictly, by `return_request_date` (verified real dates in Phase 1).\n"
        "2. **The final test (July-September 2026) is never touched:** its labels do not exist, and nothing is fitted, calibrated or tuned on it.\n"
        "3. **History features see only the past** and **customer-level text probabilities are out-of-fold**.\n"
        "4. **Thresholds are derived from the cost matrix, not from test labels.**"
    ),
    md(
        "## 2. The split the plan asks for: 70% train / 10% calibration / 20% test (chronological)\n"
        "Implemented in `r2sorter.splits.chronological_split`: rows are cut by **date** so one calendar day is never divided between two splits."
    ),
    code(
        "plan = chronological_split(labelled['return_request_date'])\n"
        "plan_table = describe_splits(labelled, plan)\n"
        "assert_chronological(labelled, plan, ['train', 'calibration', 'test'])\n"
        "print('chronology check passed: train < calibration < test')\n"
        "plan_table"
    ),
    code(
        "plan_table[list(LABELS)].plot.bar(stacked=True, title='Label mix per split (plan 70/10/20)')\n"
        "plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.show()"
    ),
    md(
        "## 3. The split this project actually uses, and why it differs\n"
        "**What the plan split would do:** evaluate on the last 20% (about 55 days of the labelled data) after one fit.\n\n"
        "**What the deployed system faces:** a model refreshed regularly and used on the *next* months. So we use\n"
        "- a **rolling-origin** scheme: for each month from February to April, fit on everything before it and predict that month. "
        "These out-of-time predictions are used to **calibrate** the probabilities (replacing the plan's single 10% slice, which "
        "is small - about 4,300 rows and only about 330 abusive cases);\n"
        "- a **May-June holdout** (the two most recent labelled months) to evaluate;\n"
        "- the **hidden July-September test** as the one true final test.\n\n"
        "This is a deliberate deviation from the plan. It keeps the plan's principle (train, then calibrate, then test, always in time order) "
        "and uses more data for calibration. Section 9 also runs the plan's exact 70/10/20 split so the two can be compared."
    ),
    code(
        "windows = []\n"
        "holdout_start = pd.Timestamp(cfg['validation']['holdout_start'])\n"
        "calibration_start = pd.Timestamp(cfg['validation']['calibration_start'])\n"
        "for month_start in pd.date_range(calibration_start, holdout_start, freq='MS', inclusive='left'):\n"
        "    month_end = month_start + pd.offsets.MonthBegin(1)\n"
        "    windows.append(('calibration fold', month_start, month_end))\n"
        "windows.append(('HOLDOUT (evaluation)', holdout_start, labelled['return_request_date'].max() + pd.Timedelta(days=1)))\n"
        "\n"
        "rows = []\n"
        "for role, start, end in windows:\n"
        "    fit = labelled[labelled['return_request_date'] < start]\n"
        "    predict = labelled[(labelled['return_request_date'] >= start) & (labelled['return_request_date'] < end)]\n"
        "    assert fit['return_request_date'].max() < predict['return_request_date'].min(), 'fit data must precede the predicted window'\n"
        "    rows.append({\n"
        "        'role': role, 'fit_first': fit['return_request_date'].min().date(), 'fit_last': fit['return_request_date'].max().date(),\n"
        "        'fit_rows': len(fit), 'predict_first': predict['return_request_date'].min().date(),\n"
        "        'predict_last': predict['return_request_date'].max().date(), 'predict_rows': len(predict),\n"
        "        'predict_abusive_%': round((predict['label'] == 'abusive').mean() * 100, 1),\n"
        "    })\n"
        "rows.append({'role': 'FINAL TEST (hidden labels)', 'fit_first': '-', 'fit_last': '-', 'fit_rows': 0,\n"
        "             'predict_first': test_rows['return_request_date'].min().date(), 'predict_last': test_rows['return_request_date'].max().date(),\n"
        "             'predict_rows': len(test_rows), 'predict_abusive_%': float('nan')})\n"
        "project_table = pd.DataFrame(rows).set_index('role')\n"
        "print('every fit window ends before the window it predicts: OK')\n"
        "project_table"
    ),
    md("## 4. The final test set is untouched"),
    code(
        "assert test_rows['label'].isna().all(), 'final-test labels must not exist in the data'\n"
        "assert labelled['return_request_date'].max() < test_rows['return_request_date'].min()\n"
        "print('final test labels are absent:', test_rows['label'].isna().all())\n"
        "print('labelled data ends', labelled['return_request_date'].max().date(), '| final test starts', test_rows['return_request_date'].min().date())\n"
        "\n"
        "# Which of the pipeline's functions are even allowed to look at final-test rows?\n"
        "print('\\nrolling_origin_predictions filters labelled rows only ->', 'labelled = known[known[\"is_train\"]]' in inspect.getsource(__import__('r2sorter.model', fromlist=['x']).rolling_origin_predictions))"
    ),
    md("## 5. Duplicate and overlap checks across splits"),
    code(
        "for split_name, other in [('train', 'calibration'), ('train', 'test'), ('calibration', 'test')]:\n"
        "    a = set(labelled.loc[plan == split_name, 'return_id'])\n"
        "    b = set(labelled.loc[plan == other, 'return_id'])\n"
        "    print(f'return_id shared between {split_name} and {other}: {len(a & b)}')\n"
        "print('return_id shared between labelled data and the final test:', len(set(labelled['return_id']) & set(test_rows['return_id'])))\n"
        "print('duplicate full rows overall:', int(frame.drop(columns=['day']).astype(str).duplicated().sum()))\n"
        "\n"
        "event = ['customer_id', 'order_id', 'product_id', 'size_ordered']\n"
        "events = labelled.assign(split=plan).groupby(event)['split'].agg(['size', 'nunique'])\n"
        "print('customer+order+product+size events returned twice:', int((events['size'] > 1).sum()),\n"
        "      '| of those spanning two splits:', int(((events['size'] > 1) & (events['nunique'] > 1)).sum()))"
    ),
    md(
        "## 6. History features use only information from before each request\n"
        "Three executable tests of `add_history_features` and `add_label_history`."
    ),
    md(
        "### 6a. Truncation test (the strongest one)\n"
        "Rebuild every feature from a dataset in which **everything after 31 March 2026 has been deleted**. "
        "If any feature of an earlier row depended on later rows, its value would change."
    ),
    code(
        "cutoff = pd.Timestamp('2026-03-31')\n"
        "short = data['train'][data['train']['return_request_date'] <= cutoff]\n"
        "cut_frame = build_feature_table(short, data['test'].iloc[:0], data['products'], cfg)\n"
        "history_columns = [c for c in frame.columns if c.startswith('hist_')]\n"
        "full_part = frame[frame['return_request_date'] <= cutoff].set_index('return_id')[history_columns]\n"
        "cut_part = cut_frame.set_index('return_id')[history_columns].loc[full_part.index]\n"
        "differences = (~np.isclose(full_part.to_numpy(float), cut_part.to_numpy(float), equal_nan=True)).sum()\n"
        "print(f'{len(history_columns)} history features x {len(full_part):,} rows compared; cells that changed: {differences}')\n"
        "assert differences == 0\n"
        "print('PASS: deleting the future changes no earlier feature')"
    ),
    md(
        "### 6b. Independent brute-force recomputation\n"
        "For 400 random requests, recount from the raw rows with plain pandas (a different code path) and compare."
    ),
    code(
        "rng = np.random.default_rng(cfg['seed'])\n"
        "sample = rng.choice(len(frame), 400, replace=False)\n"
        "norm = frame['customer_reason'].map(normalise_reason)\n"
        "mismatches = 0\n"
        "for i in sample:\n"
        "    row = frame.iloc[i]\n"
        "    earlier = frame.iloc[:i][frame.iloc[:i]['customer_id'] == row['customer_id']]   # same customer, earlier in time order\n"
        "    expected = {\n"
        "        'hist_n_prev': len(earlier),\n"
        "        'hist_returns_30d': int((earlier['day'] >= row['day'] - 30).sum()),\n"
        "        'hist_returns_60d': int((earlier['day'] >= row['day'] - 60).sum()),\n"
        "        'hist_returns_90d': int((earlier['day'] >= row['day'] - 90).sum()),\n"
        "        'hist_same_product': int((earlier['product_id'] == row['product_id']).sum()),\n"
        "        'hist_same_text': int((norm.iloc[:i][earlier.index] == norm.iloc[i]).sum()),\n"
        "    }\n"
        "    mismatches += sum(row[k] != v for k, v in expected.items())\n"
        "print('features compared:', 400 * 6, '| mismatches:', mismatches)\n"
        "assert mismatches == 0\n"
        "print('PASS: pipeline features equal an independent recount of earlier rows')"
    ),
    md(
        "### 6c. Outcome history (`lab_*` features) cannot see the request's own label or any later label\n"
        "These features carry what the retailer already *knew* about the customer's earlier outcomes, after a review delay."
    ),
    code(
        "boundary, lag, seed = pd.Timestamp('2026-05-01'), 61, cfg['seed']\n"
        "base = add_label_history(frame, boundary, lag, seed)\n"
        "lab = list(LABEL_HISTORY_COLUMNS)\n"
        "\n"
        "# (i) Shuffle every label from the boundary onwards: no feature of any row may change.\n"
        "shuffled = frame.copy()\n"
        "after = shuffled['is_train'] & (shuffled['return_request_date'] >= boundary)\n"
        "shuffled.loc[after, 'label'] = rng.permutation(shuffled.loc[after, 'label'].to_numpy())\n"
        "alt = add_label_history(shuffled, boundary, lag, seed)\n"
        "changed = (~np.isclose(base[lab].to_numpy(float), alt[lab].to_numpy(float), equal_nan=True)).sum()\n"
        "print('(i) labels from the boundary onwards shuffled -> changed cells:', changed)\n"
        "assert changed == 0\n"
        "\n"
        "# (ii) Flip the label of 500 earlier rows (one per customer): those rows' OWN features may not change.\n"
        "# (A customer's LATER rows may legitimately change - they are allowed to see this outcome.)\n"
        "before = frame[frame['is_train'] & (frame['return_request_date'] < boundary)]\n"
        "one_per_customer = before.groupby('customer_id').sample(1, random_state=cfg['seed'])\n"
        "chosen = rng.choice(one_per_customer.index, 500, replace=False)\n"
        "flipped = frame.copy()\n"
        "flipped.loc[chosen, 'label'] = np.where(flipped.loc[chosen, 'label'] == 'abusive', 'genuine_fit_issue', 'abusive')\n"
        "alt2 = add_label_history(flipped, boundary, lag, seed)\n"
        "own = (~np.isclose(base.loc[chosen, lab].to_numpy(float), alt2.loc[chosen, lab].to_numpy(float), equal_nan=True)).sum()\n"
        "print('(ii) own label flipped -> changed cells in those rows:', own)\n"
        "assert own == 0\n"
        "\n"
        "# (iii) Visible outcomes can never exceed the earlier returns or the labelled returns before the boundary.\n"
        "assert (base['lab_n_prev'] <= base['hist_n_prev']).all()\n"
        "known_before = frame[frame['is_train'] & (frame['return_request_date'] < boundary)].groupby('customer_id').size()\n"
        "late = base[base['return_request_date'] >= boundary]\n"
        "assert (late['lab_n_prev'] <= late['customer_id'].map(known_before).fillna(0)).all()\n"
        "print('(iii) lab_n_prev <= earlier returns, and rows after the boundary see only labels dated before it: PASS')"
    ),
    md(
        "### 6d. A limitation we measure instead of hiding\n"
        "The data has dates but **no time of day**. Within one calendar day, requests are ordered by `return_id`, so a request can 'see' a "
        "same-day request with a smaller ID. That is not strictly 'before'. How much does it matter?"
    ),
    code(
        "same_day_prior = frame.groupby(['customer_id', 'return_request_date']).cumcount() > 0\n"
        "print(f'requests with an earlier same-day request by the same customer: {same_day_prior.mean():.2%} ({int(same_day_prior.sum())} rows)')\n"
        "print('Outcome (lab_*) features avoid this entirely: they use a delay of at least 1 day.')\n"
        'print(f\'abusive rate of those rows: {(labelled.loc[same_day_prior[labelled.index], "label"] == "abusive").mean():.3f} vs {(labelled["label"] == "abusive").mean():.3f} overall\')'
    ),
    md(
        "**Reading:** the pipeline passes all three tests, but 6d is a limitation, **not** something the tests can fix. Ordering same-day requests by "
        "`return_id` does not prove the earlier ID really came first, so for those rows we cannot rule out that a count feature saw a request that happened "
        "later the same day. It affects the count features of 1.62% of rows. Treat it as an open limitation, not as a leakage-free guarantee; "
        "real timestamps would remove it."
    ),
    md("## 7. Customer and text leakage"),
    md("### 7a. Text probabilities are out-of-fold *by customer*"),
    code(
        "source = inspect.getsource(out_of_fold_text_probs)\n"
        "assert 'GroupKFold' in source\n"
        "pre = labelled[labelled['return_request_date'] < holdout_start]\n"
        "for fold, (fit_idx, valid_idx) in enumerate(GroupKFold(cfg['validation']['text_oof_folds']).split(pre, groups=pre['customer_id'])):\n"
        "    shared = set(pre['customer_id'].iloc[fit_idx]) & set(pre['customer_id'].iloc[valid_idx])\n"
        "    assert not shared, f'fold {fold} shares customers'\n"
        "print('customers shared between the fitting part and the predicted part of every fold: 0  (same splitter the pipeline uses)')"
    ),
    code(
        "oof = out_of_fold_text_probs(pre['customer_reason'], pre['label'], pre['customer_id'], cfg['text'],\n"
        "                             cfg['validation']['text_oof_folds'], cfg['seed'])\n"
        "text_model = TextReasonModel(cfg['text'], cfg['seed']).fit(pre['customer_reason'], pre['label'])\n"
        "in_sample = text_model.predict_proba(pre['customer_reason'])\n"
        "holdout = labelled[labelled['return_request_date'] >= holdout_start]\n"
        "holdout_probs = text_model.predict_proba(holdout['customer_reason'])\n"
        "\n"
        "def summary(probs, rows):\n"
        "    truth = rows['label'].map(LABELS.index).to_numpy()\n"
        "    return {'abusive AP': average_precision_score(truth == ABUSIVE, probs[:, ABUSIVE]),\n"
        "            '4-class accuracy': float((probs.argmax(axis=1) == truth).mean())}\n"
        "\n"
        "leak_table = pd.DataFrame({\n"
        "    'in-sample (WRONG way: model saw these rows)': summary(in_sample, pre),\n"
        "    'out-of-fold by customer (used for training)': summary(oof, pre),\n"
        "    'real May-June holdout (honest reference)': summary(holdout_probs, holdout),\n"
        "}).T.round(3)\n"
        "leak_table"
    ),
    md(
        "**Reading:** scoring the text model on rows it was trained on looks far better than reality (abusive AP 0.68, accuracy 0.92). The out-of-fold probabilities used to train the second stage behave like the honest holdout: 4-class accuracy 0.88 vs 0.87, and abusive AP 0.13 vs 0.18 (the two differ somewhat because the holdout has a different abuse rate). Neither is anywhere near the in-sample figure, which is the evidence that the customer-grouped folds block this leak."
    ),
    md("### 7b. Do reason templates repeat across splits, and does the text model just memorise them?"),
    code(
        "plan_train = labelled[plan == 'train']\n"
        "later = labelled[plan != 'train']\n"
        "seen = set(plan_train['customer_reason'].map(normalise_reason))\n"
        "is_seen = later['customer_reason'].map(normalise_reason).isin(seen)\n"
        "plan_text_model = TextReasonModel(cfg['text'], cfg['seed']).fit(plan_train['customer_reason'], plan_train['label'])\n"
        "probs = plan_text_model.predict_proba(later['customer_reason'])\n"
        "truth = later['label'].map(LABELS.index).to_numpy()\n"
        "report = pd.DataFrame({\n"
        "    'rows': [int(is_seen.sum()), int((~is_seen).sum())],\n"
        "    '4-class accuracy': [float((probs.argmax(1) == truth)[is_seen.to_numpy()].mean()), float((probs.argmax(1) == truth)[~is_seen.to_numpy()].mean())],\n"
        "    'abusive rate': [float((truth == ABUSIVE)[is_seen.to_numpy()].mean()), float((truth == ABUSIVE)[~is_seen.to_numpy()].mean())],\n"
        "}, index=['text seen verbatim in the training split', 'text never seen']).round(3)\n"
        "print(f'{is_seen.mean():.1%} of calibration+test reason texts appear word for word in the training split')\n"
        "report"
    ),
    md(
        "**Reading:** about a quarter of later reason texts repeat word for word from the training split, so templates do cross the split. But accuracy on repeated texts is **not higher** than on unseen ones (0.87 vs 0.88), so the text model is not winning by memorising rows. The abusive rate differs only slightly (8.4% vs 7.4%). Template overlap therefore does not leak the label here. It is still a reason to evaluate on later months and to keep text a *supporting* signal."
    ),
    md("### 7c. The same customers appear in several splits - this is real, not a leak"),
    code(
        "train_customers = set(plan_train['customer_id'])\n"
        "for name in ['calibration', 'test']:\n"
        "    ids = labelled.loc[plan == name, 'customer_id']\n"
        "    print(f'{name:12} requests from customers already in the training split: {ids.isin(train_customers).mean():.1%}')\n"
        "print(f\"final test   requests from customers already in labelled data:   {test_rows['customer_id'].isin(set(labelled['customer_id'])).mean():.1%}\")"
    ),
    md(
        "In deployment the retailer *will* see the same customers again, so splitting by customer would make validation unrealistically hard. "
        "The protection is the **point-in-time** features (section 6), not removing customers from the evaluation data. "
        "Customer-grouped folds are used only inside training, to make text probabilities honest."
    ),
    md(
        "## 8. Catalogue and customer fields computed from the labels?\n"
        "The request that raised this mentions a field like `fit_issue_prop`. Phase 1 already scanned every column of `products.csv`, `size_charts.csv` and "
        "`customers.csv` and found no label-like names and no column with leaked predictive power. Here we check what the **model is actually given**."
    ),
    code(
        "columns = tabular_columns(base)  # `base` already has the outcome-history columns added in 6c\n"
        "groups = {\n"
        "    'order facts (given with the request)': [c for c in columns if c in {'delivery_days', 'days_since_delivery', 'order_value_inr', 'item_price_inr', 'discount_pct', 'items_in_order', 'customer_tenure_days', 'prior_orders_count', 'prior_return_rate'}],\n"
        "    'catalogue (static product facts)': [c for c in columns if c in {'brand', 'fabric', 'fit_type', 'mrp_inr', 'size_pos_frac', 'price_mismatch', 'item_share_of_order', 'size_kind', 'category', 'sub_category'}],\n"
        "    'history of EARLIER requests': [c for c in columns if c.startswith('hist_')],\n"
        "    'earlier OUTCOMES, after review delay': [c for c in columns if c.startswith('lab_')],\n"
        "    'text evidence flags': [c for c in columns if c.startswith(('ev_', 'txt_'))],\n"
        "}\n"
        "for name, cols in groups.items():\n"
        '    print(f\'{name:40} {len(cols):2d}  {cols[:6]}{" ..." if len(cols) > 6 else ""}\')\n'
        "assert 'label' not in columns and 'abusive' not in columns\n"
        "assert set(c for c in columns if c.startswith('lab_')) <= set(LABEL_HISTORY_COLUMNS)\n"
        "print('\\nthe label itself is never a feature; outcome features exist only through add_label_history (tested in 6c)')\n"
        "print('catalogue columns used:', [c for c in data['products'].columns if c in columns])"
    ),
    code(
        "fit_counts = labelled[labelled['label'] == 'genuine_fit_issue'].groupby('product_id').size()\n"
        "per_product = data['products'].set_index('product_id').assign(fit_returns=fit_counts).fillna({'fit_returns': 0})\n"
        "print('correlation of catalogue price with a product\\'s own fit-return count:', round(per_product['price_inr'].corr(per_product['fit_returns']), 3))\n"
        "print('products.csv holds', data['products'].shape[1], 'descriptive columns and no per-product return statistics')"
    ),
    md(
        "## 9. The plan's 70/10/20 split, run end to end\n"
        "To see whether our different split matters, we run the plan's exact design: fit on the 70%, calibrate on the 10%, evaluate on the 20% "
        "with the **official scorer**. Outcome-history features see only labels dated before the calibration start."
    ),
    code(
        "calibration_start_plan = labelled.loc[plan == 'calibration', 'return_request_date'].min()\n"
        "lag_plan = int((labelled['return_request_date'].max() - calibration_start_plan).days) + 1\n"
        "known = add_label_history(frame, calibration_start_plan, lag_plan, cfg['seed'])\n"
        "fit_rows = known[known['is_train'] & (known['return_request_date'] < calibration_start_plan)]\n"
        "cal_rows = known.loc[labelled.index[plan == 'calibration']]\n"
        "eval_rows = known.loc[labelled.index[plan == 'test']]\n"
        "assert fit_rows['return_request_date'].max() < cal_rows['return_request_date'].min() <= cal_rows['return_request_date'].max() < eval_rows['return_request_date'].min()\n"
        "\n"
        "plan_model = StackModel(cfg).fit(fit_rows)\n"
        "cal_pred = plan_model.predict(cal_rows)\n"
        "eval_pred = plan_model.predict(eval_rows)\n"
        "calibrator = AbuseCalibrator(cfg['calibration']['method']).fit(cal_pred['p_abuse_raw'].to_numpy(), (cal_rows['label'] == 'abusive').to_numpy())\n"
        "p_plan = calibrator.transform(eval_pred['p_abuse_raw'].to_numpy())\n"
        "decisions_plan = route(p_plan, cfg)\n"
        "result_plan = official_score(pd.DataFrame({'return_id': eval_rows['return_id'], 'decision': decisions_plan, 'confidence': p_plan, 'justification': 'x'}), eval_rows)\n"
        "print('fit rows', len(fit_rows), '| calibration rows', len(cal_rows), '| evaluation rows', len(eval_rows))\n"
        "pd.Series(decisions_plan).value_counts().to_frame('decisions on the 20% test')"
    ),
    code(
        "def headline(result):\n"
        "    c, k = result['components'], result['calibration_of_confidence_as_P_abusive']\n"
        "    return {'savings': c['savings'], 'fairness': c['fairness_score'], 'primary (no size-chart part)': result['primary_score'],\n"
        "            'abusive average precision': k['average_precision'], 'Brier': k['brier'], 'ECE': k['ece_10_bins'],\n"
        "            'genuine customers escalated': result['abuse_detection']['genuine_escalated']}\n"
        "\n"
        "comparison = {'plan split: 70/10/20 (this section)': headline(result_plan)}\n"
        "reports = Path('../reports/validation.json')\n"
        "if reports.exists():\n"
        "    import json\n"
        "    saved = json.loads(reports.read_text())\n"
        "    comparison['project split: May-June holdout (reports/validation.json)'] = headline(saved)\n"
        "pd.DataFrame(comparison).T.round(4)"
    ),
    md(
        "**Reading:** both designs give clearly positive savings and no sign of leakage, but the plan's single 10% calibration slice does worse: savings 0.31 vs 0.39, calibration error 0.013 vs 0.005, and 7 vs 2 honest customers escalated. The likely cause is that the plan's 10% slice is small (4,388 rows) and covers one low-abuse stretch (6.5% abusive), so the calibrator fits less well; the project's three rolling months give it 13,902 out-of-time rows. The two rows also cover different evaluation periods, so the gap is not a pure like-for-like comparison. Conclusion: the deviation is documented and justified, not harmful."
    ),
    md("## 10. Calibration and thresholds never use test labels"),
    code(
        "print('break_even_thresholds signature:', inspect.signature(break_even_thresholds))\n"
        "print('-> it receives only the cost matrix, no labels or predictions:', break_even_thresholds(cfg))\n"
        "print('route() signature:', inspect.signature(route), '(probabilities and costs only)')\n"
        "\n"
        "calibration_file = Path('../reports/calibration_out_of_time.csv')\n"
        "if calibration_file.exists():\n"
        "    used = pd.read_csv(calibration_file, index_col='row')\n"
        "    dates = frame.loc[used.index, 'return_request_date']\n"
        "    print(f'\\ncalibrator of the project split was fitted on {len(used):,} out-of-time predictions dated {dates.min().date()} to {dates.max().date()}')\n"
        "    assert dates.max() < holdout_start and dates.min() >= calibration_start\n"
        "    print('PASS: every calibration row is earlier than the May-June holdout')\n"
        "else:\n"
        "    print('\\n(reports/calibration_out_of_time.csv not found - run `python main.py validate` to produce it)')"
    ),
    md(
        "**Honest note on how the May-June holdout was used.** Thresholds are fixed by the cost matrix (no tuning). However, while building the model "
        "I compared a few hyperparameter and feature variants on the May-June holdout, so that holdout behaves as a **validation set**: its scores can be "
        "slightly optimistic. The unbiased estimates are (a) the plan-split result in section 9, where the model was not tuned, and (b) the hidden "
        "July-September test. We therefore never report the holdout as the final score."
    ),
    md("## 11. The scoring formula (resolved against the official evaluator)"),
    code(
        "official = load_official_scorer()\n"
        "print('official cost matrix:', official.COST)\n"
        "print('official score weights: savings', official.W_SAVINGS, '| fairness', official.W_FAIRNESS, '| size chart', official.W_SIZE_CHART)\n"
        "assert official.COST['genuine']['escalate'] == 10.0 and official.COST['abusive']['auto_approve'] == 1.0\n"
        "assert official.COST['genuine']['inspect'] == 0.5 and official.COST['abusive']['inspect'] == 0.3\n"
        "assert cfg['costs'] == official.COST, 'project config must equal the official matrix'\n"
        "print('project config.yaml costs equal the official matrix: PASS')"
    ),
    md(
        "| formula | verdict |\n|---|---|\n"
        "| `10 x MissedFraud + 1 x WrongBlock + 0.2 x UnnecessaryInspect` (pasted plan) | reversed direction - **rejected** |\n"
        "| `10 x WrongBlock + 1 x MissedFraud + 0.2 x UnnecessaryInspect` (review note) | right direction, but the inspect cost is 0.5 (genuine) / 0.3 (abusive), not 0.2 |\n"
        "| official matrix: wrong block 10, missed fraud 1, unnecessary inspect 0.5, inspect of an abuser 0.3 | **the benchmark, enforced by an assertion above** |"
    ),
    md(
        "## 12. Caveats that stay attached to every Phase 2 result\n"
        "**1. Same-day ordering.** The data has dates but no time of day. 1.62% of requests have an earlier same-day request from the same customer "
        "(section 6d). Ordering them by `return_id` does not prove the earlier request actually happened first. This is a limitation, not a "
        "fully leakage-free guarantee.\n\n"
        "**2. Holdout tuning.** The May-June holdout was used to compare model and feature variants (section 10), so its results may be optimistic. "
        "The untouched July-September test stays separate for the final evaluation; the plan-split result (section 9) is a second, untuned estimate.\n\n"
        "**3. Outcome-history availability.** The delayed-label-history design (`lab_*` features) assumes review outcomes become known 1-92 days after "
        "a request. In a real retailer workflow, outcomes exist only for requests that were inspected or resolved, and may arrive later or in a different "
        "pattern. If that assumption fails, these features would be sparser than in our experiments and performance would be lower.\n\n"
        "**What the passing tests do and do not show.** They show that the implemented features do not depend on later rows (truncation test), that they match an "
        "independent recount, and that outcome features ignore the request's own and later labels. They do not show that the dataset itself was generated "
        "without leakage, or that production data would behave the same way."
    ),
    md(
        "## 13. Phase 2 verdict\n"
        "| Requirement | Evidence | Status |\n|---|---|---|\n"
        "| Chronological train / calibration / test | sections 2-4, assertions | done |\n"
        "| 70/10/20 split from the plan | section 2, run end to end in section 9 | done |\n"
        "| Deviation documented and test untouched | sections 3-4 | done |\n"
        "| Class mix and dates per split | sections 2-3 | done |\n"
        "| Duplicates across splits | section 5 | done |\n"
        "| History features past-only | section 6 (three tests plus one measured limitation) | tested; same-day ordering remains a limitation |\n"
        "| Customer-grouped text OOF; template leakage | section 7 | done |\n"
        "| Catalogue fields not label-derived | section 8 and Phase 1 | done |\n"
        "| Calibration and thresholds without test labels | section 10 | done |\n"
        "| Holdout tuning disclosed | sections 10 and 12 | done, results treated as optimistic |\n"
        "| Official cost formula enforced | section 11 | done |\n"
        "| Reproducible pipeline | `src/r2sorter/splits.py` + tests, `features.py`, `text.py`, `model.py` | done |\n\n"
        "**Status: Phase 2 is complete with the three caveats in section 12.** It is validated to the extent the tests above can show, not proven leakage-free.\n\n"
        "**Next:** Phase 3 - feature engineering."
    ),
]


def main() -> None:
    notebook = new_notebook(cells=CELLS)
    if OUT.exists():  # keep the saved outputs of code cells whose source did not change
        previous = nbformat.read(OUT, as_version=4)
        saved = {c.source: c for c in previous.cells if c.cell_type == "code"}
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source in saved:
                cell.outputs = saved[cell.source].outputs
                cell.execution_count = saved[cell.source].execution_count
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
