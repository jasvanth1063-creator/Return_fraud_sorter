"""Create notebooks/Phase_03_Feature_Engineering.ipynb (uses src/r2sorter/preprocess.py)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Phase_03_Feature_Engineering.ipynb"

CELLS = [
    md(
        "# Phase 3 - Feature engineering\n"
        "**Objective:** turn the joined original tables into model inputs with a *reusable, training-fitted* preprocessing pipeline, "
        "and document what is used, what is excluded and why. No final classifier is trained here; the only model is a small "
        "LightGBM used in section 9 to test, honestly, whether the engineered features help.\n\n"
        "| Plan step (Phase 3 plan for this ZIP) | Section |\n|---|---|\n"
        "| 1 Supplied return-request columns | 2, 3 |\n"
        "| 2 Join `products.csv` (unique key, coverage) | 2 |\n"
        "| 3 Join `customers.csv`, check tenure, evaluate `city_tier` | 2, 5 |\n"
        "| 4 Meaning of each missing value | 4 |\n"
        "| 5 Size-chart measurements | 4 |\n"
        "| Leakage protection | 6 |\n"
        "| Training-fitted matrices for train / calibration / test | 7 |\n"
        "| Feature-quality report | 8 |\n"
        "| Does it help? | 9 |\n\n"
        "**How to run:** keep this file in `FULL_CODE/notebooks/` with `src/r2sorter/preprocess.py` and the five official CSVs in `FULL_CODE/data/`."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "for candidate in (Path('../src'), Path('src')):\n"
        "    if candidate.exists():\n"
        "        sys.path.insert(0, str(candidate.resolve()))\n"
        "\n"
        "import lightgbm as lgb\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.metrics import average_precision_score, brier_score_loss, log_loss\n"
        "\n"
        "from r2sorter.config import load_config\n"
        "from r2sorter.data import load_all\n"
        "from r2sorter.features import build_feature_table, tabular_columns\n"
        "from r2sorter.preprocess import (\n"
        "    CATEGORICAL_FEATURES, FeaturePreprocessor, add_engineered_features, data_quality_flags, excluded_columns,\n"
        "    feature_catalogue, feature_quality_report, validate_joins)\n"
        "from r2sorter.splits import chronological_split, describe_splits\n"
        "\n"
        "pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_rows', 120)\n"
        "cfg = load_config()\n"
        "data = load_all(cfg)\n"
        "train, test = data['train'], data['test']\n"
        "products, charts, customers = data['products'], data['size_charts'], data['customers']\n"
        "REPORTS = Path('../reports') if Path('../reports').exists() else Path('reports')\n"
        "REPORTS.mkdir(exist_ok=True)\n"
        "print(f'train {len(train):,}  test {len(test):,}  products {len(products):,}  customers {len(customers):,}')"
    ),
    md(
        "## 2. Join validation (before any merge)\nKeys must be unique and every request must find its product and customer. A failed check stops the pipeline."
    ),
    code(
        "joins = validate_joins(pd.concat([train, test]), products, customers, charts)\n"
        "joins.to_csv(REPORTS / 'phase3_join_validation.csv', header=['value'])\n"
        "assert joins[['product_id_duplicates', 'customer_id_duplicates', 'requests_without_product',\n"
        "              'requests_without_customer', 'products_with_unknown_chart']].sum() == 0\n"
        "joins.to_frame('value')"
    ),
    md(
        "`products_without_chart` is expected: free-size products (sarees, dupattas) have no size chart. "
        "It is **chart unavailable**, not suspicious behaviour, so it becomes the neutral flag `size_chart_missing` and never a fraud signal."
    ),
    md(
        "## 3. Feature catalogue and excluded columns\nEvery engineered feature lists its source, rule and why it cannot leak."
    ),
    code(
        "catalogue = feature_catalogue()\n"
        "catalogue.to_csv(REPORTS / 'phase3_feature_catalogue.csv')\n"
        "print(len(catalogue), 'engineered / derived features')\n"
        "catalogue"
    ),
    code("excluded = excluded_columns()\nexcluded.to_csv(REPORTS / 'phase3_excluded_columns.csv')\nexcluded"),
    md("## 4. What a missing value means"),
    code(
        "# prior_return_rate: expected missing exactly when there is no history\n"
        "missing_rate = train['prior_return_rate'].isna()\n"
        "no_history = train['prior_orders_count'] == 0\n"
        "print('prior_return_rate missing iff prior_orders_count == 0:', bool((missing_rate == no_history).all()))\n"
        "print(f\"missing share {missing_rate.mean():.3%}; abuse rate when history missing {(train.label[missing_rate] == 'abusive').mean():.3%} \"\n"
        "      f\"vs {(train.label[~missing_rate] == 'abusive').mean():.3%} otherwise\")\n"
        "# size charts: which measurements apply to which product kind\n"
        "kinds = products.merge(charts.drop_duplicates('size_chart_id'), on='size_chart_id', how='left')\n"
        "measures = ['chest_cm', 'waist_cm', 'hip_cm', 'shoulder_cm', 'length_cm', 'inseam_cm', 'foot_length_cm']\n"
        "print('\\nshare of charts with the measure present, by product category:')\n"
        "chart_rows = charts.merge(products.drop_duplicates('size_chart_id')[['size_chart_id', 'category']], on='size_chart_id')\n"
        "chart_rows.groupby('category')[measures].agg(lambda s: s.notna().mean()).round(2)"
    ),
    md(
        "Measurements are **not** imputed: a missing `inseam_cm` on a top or `foot_length_cm` on clothing means *not applicable*. "
        "`size_chart_deviation` uses the first measurement that exists for the chart (chest, waist, hip, shoulder, length, foot) and is NaN "
        "otherwise. It compares the ordered size with the chart's *median* size - a property of the **garment chart**, not of the customer's body."
    ),
    md("## 5. Customer table: tenure consistency and the `city_tier` question"),
    code(
        "frame = build_feature_table(train, test, products, cfg)\n"
        "frame = add_engineered_features(frame, charts, cfg, customers)\n"
        "print('tenure recomputed from signup_date minus supplied customer_tenure_days (days):')\n"
        "print(frame['tenure_gap_days'].describe().round(3).to_string())\n"
        "labelled = frame[frame['is_train']].assign(abusive=lambda d: (d['label'] == 'abusive').astype(int))\n"
        "print('\\nabuse rate by city_tier:')\n"
        "labelled.groupby('city_tier')['abusive'].agg(['mean', 'size']).round(4)"
    ),
    md(
        "The supplied `customer_tenure_days` agrees with `signup_date` on every row, so we keep the supplied column and do not add `signup_date` as a second copy. "
        "`city_tier` shows only a weak abuse gap and is a possible proxy for customer group, so it is **kept out of the default features** "
        "and tested as a separate arm in section 9."
    ),
    md("## 6. Leakage tests"),
    code(
        "# 6a. the label never reaches the matrix\n"
        "pre_probe = FeaturePreprocessor(cfg).fit(frame[frame['is_train']].head(5000))\n"
        "cols = set(pre_probe.transform_tree(frame.head(10)).columns)\n"
        "forbidden = {'label', 'abusive', 'customer_reason', 'return_id', 'order_id', 'customer_id', 'product_id'}\n"
        "assert not (cols & forbidden) and not [c for c in cols if c.startswith('lab_')], cols & forbidden\n"
        "print('6a ok - no label, identifier, raw text or lab_* column among', len(cols), 'model columns')\n"
        "\n"
        "# 6b. truncation: features of earlier rows must not change when later rows are removed\n"
        "cut = frame['return_request_date'].quantile(0.5)\n"
        "early = frame[frame['return_request_date'] <= cut]\n"
        "short = add_engineered_features(\n"
        "    build_feature_table(train[train['return_request_date'] <= cut], test.iloc[:0], products, cfg), charts, cfg, customers)\n"
        "check = ['refund_value_30d', 'refund_value_60d', 'refund_value_90d', 'prior_return_count_est', 'hist_n_prev']\n"
        "key = ['return_id']\n"
        "merged = early[key + check].merge(short[key + check], on=key, suffixes=('_full', '_cut'))\n"
        "for name in check:\n"
        "    assert np.allclose(merged[f'{name}_full'], merged[f'{name}_cut'], equal_nan=True), name\n"
        "print(f'6b ok - {len(merged):,} early rows identical with and without the future')\n"
        "\n"
        "# 6c. the preprocessor learns only from the rows it is fitted on\n"
        "fit_rows = frame[frame['is_train']].head(20000)\n"
        "pre = FeaturePreprocessor(cfg).fit(fit_rows)\n"
        "before = pre.learned_parameters()\n"
        "_ = pre.transform_linear(frame[~frame['is_train']])\n"
        "assert pre.learned_parameters() == before\n"
        "print('6c ok - transforming test rows changed nothing that was learned')"
    ),
    md(
        "Same-day ordering caveat from Phase 2 still applies: requests on one day are ordered by `return_id`, so 'earlier' is exact only between days."
    ),
    md("## 7. Training-fitted matrices for the plan split and the final test"),
    code(
        "labelled = frame[frame['is_train']].reset_index(drop=True)\n"
        "plan = chronological_split(labelled['return_request_date'])\n"
        "display(describe_splits(labelled, plan))\n"
        "pre = FeaturePreprocessor(cfg).fit(labelled[plan == 'train'])\n"
        "parts = {\n"
        "    'train': labelled[plan == 'train'],\n"
        "    'calibration': labelled[plan == 'calibration'],\n"
        "    'validation_test': labelled[plan == 'test'],\n"
        "    'final_test_Jul-Sep': frame[~frame['is_train']],\n"
        "}\n"
        "summary = []\n"
        "for name, rows in parts.items():\n"
        "    linear, tree = pre.transform_linear(rows), pre.transform_tree(rows)\n"
        "    assert linear.isna().sum().sum() == 0 and np.isfinite(linear.to_numpy()).all()\n"
        "    assert list(linear.columns) == list(pre.transform_linear(parts['train'].head(3)).columns)\n"
        "    summary.append({'part': name, 'rows': len(rows), 'linear_cols': linear.shape[1], 'tree_cols': tree.shape[1],\n"
        "                    'rows_with_unseen_category': int((linear['n_unseen_categories'] > 0).sum())})\n"
        "print('dropped as constant or duplicate on the training matrix:', pre.dropped_columns_)\n"
        "pd.DataFrame(summary).set_index('part')"
    ),
    md(
        "Columns dropped by the redundancy step are guards that are constant in this data (e.g. every ordered size is sold, no invalid dates). "
        "They stay in the code so a future bad record is caught by `data_quality_flags`, but they carry no information today."
    ),
    code("data_quality_flags(parts['final_test_Jul-Sep'])"),
    md("## 8. Feature-quality report"),
    code(
        "report = feature_quality_report(pre.transform_tree(parts['train']))\n"
        "report.to_csv(REPORTS / 'phase3_feature_quality_report.csv')\n"
        "print('constant:', report.index[report['constant']].tolist())\n"
        "print('duplicate of earlier column:', report.index[report['duplicate_of_earlier_column']].tolist())\n"
        "print('any inf:', report.index[report['has_inf']].tolist())\n"
        "print('\\ncategory coverage of the final test set (share of rows whose category was never seen in training):')\n"
        "unseen = {c: float((~parts['final_test_Jul-Sep'][c].astype(str).isin(pre._categories[c])).mean()) for c in CATEGORICAL_FEATURES}\n"
        "print(pd.Series(unseen).round(4).to_string())\n"
        "report.sort_values('missing_%', ascending=False).head(15)"
    ),
    md(
        "## 9. Does it help? (honest ablation)\n"
        "Structured features only (no text model), tabular LightGBM, trained on requests before 1 May 2026 and scored on May-June 2026, "
        "3 seeds, preprocessing fitted on the training rows only. **A = the existing structured columns**, B = A plus the engineered columns, "
        "C = B plus `city_tier`.\n\n"
        "Caveat: the May-June window was also used in earlier tuning, so absolute numbers may be optimistic; the comparison between arms is what matters."
    ),
    code(
        "split_date = pd.Timestamp(cfg['validation']['holdout_start'])\n"
        "fit_rows = labelled[labelled['return_request_date'] < split_date]\n"
        "eval_rows = labelled[labelled['return_request_date'] >= split_date]\n"
        "y_fit, y_eval = (fit_rows['label'] == 'abusive').astype(int), (eval_rows['label'] == 'abusive').astype(int)\n"
        "\n"
        "base_cols = [c for c in tabular_columns(labelled) if not c.startswith(('lab_', 'ev_', 'txt_'))]\n"
        "pre_ab = FeaturePreprocessor(cfg).fit(fit_rows)\n"
        "tree_fit, tree_eval = pre_ab.transform_tree(fit_rows), pre_ab.transform_tree(eval_rows)\n"
        "extra = [c for c in tree_fit.columns if c not in base_cols]\n"
        "\n"
        "def arm(rows, tree, with_extra, with_tier):\n"
        "    out = rows[base_cols].copy()\n"
        "    if with_extra:\n"
        "        out = pd.concat([out, tree[extra]], axis=1)\n"
        "    if with_tier:\n"
        "        out['city_tier'] = rows['city_tier'].to_numpy()\n"
        "    return out\n"
        "\n"
        "arms = {'A baseline structured': (False, False), 'B + engineered': (True, False), 'C + engineered + city_tier': (True, True)}\n"
        "results = []\n"
        "for label, (with_extra, with_tier) in arms.items():\n"
        "    xf, xe = arm(fit_rows, tree_fit, with_extra, with_tier), arm(eval_rows, tree_eval, with_extra, with_tier)\n"
        "    for seed in (0, 1, 2):\n"
        "        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30,\n"
        "                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=8,\n"
        "                                   random_state=seed, verbose=-1)\n"
        "        model.fit(xf, y_fit)\n"
        "        p = model.predict_proba(xe)[:, 1]\n"
        "        results.append({'arm': label, 'seed': seed, 'features': xf.shape[1], 'AP': average_precision_score(y_eval, p),\n"
        "                        'log_loss': log_loss(y_eval, p), 'brier': brier_score_loss(y_eval, p)})\n"
        "results = pd.DataFrame(results)\n"
        "ablation = results.groupby('arm').agg(features=('features', 'first'), AP=('AP', 'mean'), AP_sd=('AP', 'std'),\n"
        "                                       log_loss=('log_loss', 'mean'), brier=('brier', 'mean')).round(4)\n"
        "ablation.to_csv(REPORTS / 'phase3_ablation.csv')\n"
        "ablation"
    ),
    code(
        "gain = ablation.loc['B + engineered', 'AP'] - ablation.loc['A baseline structured', 'AP']\n"
        "noise = max(ablation['AP_sd'].max(), 1e-9)\n"
        "print(f'AP change from engineered features: {gain:+.4f} (seed sd up to {noise:.4f})')\n"
        "print('Treat as a real gain only if the change is clearly larger than the seed spread and log-loss also improves.')"
    ),
    md("## 10. Verdict"),
    code(
        "base, eng, tier = (ablation.loc[k] for k in ablation.index.sort_values())\n"
        "margin = 2 * noise\n"
        "gain_eng, gain_tier = eng.AP - base.AP, tier.AP - eng.AP\n"
        "helps = gain_eng > margin and eng.log_loss < base.log_loss\n"
        "print('Engineered features: ' + ('IMPROVE' if helps else 'do NOT clearly improve')\n"
        "      + f' the structured baseline (dAP {gain_eng:+.4f}, margin {margin:.4f}).')\n"
        "print('city_tier: ' + ('adds value' if gain_tier > margin else 'adds no clear value')\n"
        "      + f' (dAP {gain_tier:+.4f}); it stays out of the default set because of proxy risk.')"
    ),
]


def main() -> None:
    notebook = new_notebook(cells=CELLS)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(notebook, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
