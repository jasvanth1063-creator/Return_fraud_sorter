"""Create notebooks/Phase_01_Dataset_Inspection.ipynb (self-contained: needs only pandas, numpy, matplotlib)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Phase_01_Dataset_Inspection.ipynb"

CELLS = [
    md(
        "# Phase 1 - Dataset inspection and problem definition\n"
        "**Project:** R2 Returns Fraud vs. Genuine Grievance Sorter\n\n"
        "**Objective:** understand exactly what the data contains and exactly how we are scored *before* modelling.\n\n"
        "**Deliverables (each has its own section below)**\n"
        "1. Dataset schema report\n"
        "2. Class-distribution chart\n"
        "3. Missing-value and duplicate report\n"
        "4. Evaluation-metric specification\n\n"
        "> The hidden labels are training and evaluation targets. They are **never** inputs at inference time."
    ),
    md(
        "## 0. Setup\n"
        "Put the organisers' CSVs in a `data/` folder next to this notebook (or one level up). "
        "`evaluate.py` is optional: if found, the notebook checks the cost matrix against it."
    ),
    code(
        "from pathlib import Path\n"
        "import importlib.util\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "pd.set_option('display.max_columns', 50, 'display.width', 160, 'display.max_colwidth', 80)\n"
        "plt.rcParams.update({'figure.figsize': (7, 3.6), 'axes.spines.top': False, 'axes.spines.right': False})\n"
        "\n"
        "\n"
        "def find_first(candidates, must_contain):\n"
        "    for folder in candidates:\n"
        "        if (Path(folder) / must_contain).exists():\n"
        "            return Path(folder)\n"
        "    raise FileNotFoundError(f'could not find {must_contain!r}; searched {[str(c) for c in candidates]}')\n"
        "\n"
        "\n"
        "DATA_DIR = find_first(['data', '../data', '../FULL_CODE/data', '../../data'], 'return_requests_train.csv')\n"
        "print('data folder:', DATA_DIR.resolve())\n"
        "SCORER = next((p for p in [Path('evaluate.py'), Path('tools/evaluate.py'), Path('../tools/evaluate.py'),\n"
        "               DATA_DIR.parent / 'evaluate.py'] if p.exists()), None)\n"
        "print('official scorer:', SCORER.resolve() if SCORER else 'not found (cost matrix will be checked against the README table only)')"
    ),
    md(
        "## 1. The problem in one page\n"
        "A fashion retailer gets many returns. Most are honest; a small share are abusive "
        "(wardrobing, 'empty box' claims, serial returners). For every return request the system must output:\n\n"
        "| output | meaning |\n|---|---|\n"
        "| `decision` | `auto_approve`, `inspect` (cheap human check) or `escalate` (serious action) |\n"
        "| `confidence` | calibrated **P(abusive)** between 0 and 1 |\n"
        "| `justification` | short, honest explanation of the decision |\n\n"
        "A second task: name the **5 products with the most fit-related returns** (size-chart fixes).\n\n"
        "Four hidden classes: `abusive`, `genuine_quality_issue`, `genuine_fit_issue`, `genuine_wrong_item`. "
        "The business decision only depends on how likely a return is abusive, but the genuine reason is useful for "
        "explanations and the size-chart task."
    ),
    md("## 2. Load the files"),
    code(
        "FILES = {\n"
        "    'train': 'return_requests_train.csv',\n"
        "    'test': 'return_requests_test.csv',\n"
        "    'products': 'products.csv',\n"
        "    'size_charts': 'size_charts.csv',\n"
        "    'customers': 'customers.csv',\n"
        "}\n"
        "tables = {name: pd.read_csv(DATA_DIR / file) for name, file in FILES.items()}\n"
        "train, test = tables['train'], tables['test']\n"
        "inventory = pd.DataFrame(\n"
        "    {\n"
        "        'file': list(FILES.values()),\n"
        "        'rows': [len(t) for t in tables.values()],\n"
        "        'columns': [t.shape[1] for t in tables.values()],\n"
        "        'size_MB': [round((DATA_DIR / f).stat().st_size / 1e6, 2) for f in FILES.values()],\n"
        "    },\n"
        "    index=list(FILES),\n"
        ")\n"
        "inventory"
    ),
    md(
        "**Reading:** `train` + `test` together are the ~60,000 return requests. Only `train` has the `label` column; "
        "the test labels are hidden and used by the organisers' scorer."
    ),
    md("## 3. Deliverable 1 - Dataset schema report"),
    code(
        "def schema(frame: pd.DataFrame) -> pd.DataFrame:\n"
        "    return pd.DataFrame(\n"
        "        {\n"
        "            'dtype': frame.dtypes.astype(str),\n"
        "            'non_null': frame.notna().sum(),\n"
        "            'missing_%': (frame.isna().mean() * 100).round(2),\n"
        "            'unique': frame.nunique(),\n"
        "            'example': frame.iloc[0],\n"
        "        }\n"
        "    )\n"
        "\n"
        "\n"
        "schema(train)"
    ),
    code(
        "for name in ['products', 'size_charts', 'customers']:\n    print(f'--- {name} {tables[name].shape}')\n    display(schema(tables[name]))"
    ),
    md("### Data dictionary - the role of every column in the main file"),
    code(
        "roles = {\n"
        "    'return_id': 'identifier (join key for the submission)',\n"
        "    'order_id': 'identifier (several returns can share an order)',\n"
        "    'customer_id': 'identifier -> customer history; links to customers.csv',\n"
        "    'product_id': 'identifier -> products.csv, size_charts.csv',\n"
        "    'category': 'feature (categorical)', 'sub_category': 'feature (categorical)',\n"
        "    'size_ordered': 'feature (categorical)',\n"
        "    'order_date': 'timestamp', 'delivery_date': 'timestamp', 'return_request_date': 'timestamp (defines time order)',\n"
        "    'delivery_days': 'feature (numeric)', 'days_since_delivery': 'feature (numeric) - late returns hint at wardrobing',\n"
        "    'order_value_inr': 'feature (numeric); also a fairness group (<1000 = low value)',\n"
        "    'item_price_inr': 'feature (numeric)', 'discount_pct': 'feature (numeric)', 'items_in_order': 'feature (numeric)',\n"
        "    'payment_method': 'feature (categorical)', 'fulfilment_centre': 'feature (categorical)',\n"
        "    'customer_tenure_days': 'feature (numeric); also a fairness group (<90 = new customer)',\n"
        "    'prior_orders_count': 'feature (numeric)', 'prior_return_rate': 'feature (numeric; empty when no prior orders)',\n"
        "    'customer_reason': 'free text, English / Hinglish / Hindi in Latin script, typos, emoji',\n"
        "    'label': 'TARGET - train only, never an input',\n"
        "}\n"
        "dictionary = pd.DataFrame({'role': pd.Series(roles)})\n"
        "dictionary['in_train'] = dictionary.index.isin(train.columns)\n"
        "dictionary['in_test'] = dictionary.index.isin(test.columns)\n"
        "dictionary"
    ),
    code(
        "only_train = sorted(set(train.columns) - set(test.columns))\n"
        "only_test = sorted(set(test.columns) - set(train.columns))\n"
        "print('columns only in train:', only_train)\n"
        "print('columns only in test :', only_test)\n"
        "assert only_train == ['label'] and not only_test, 'test must differ from train only by the label'"
    ),
    md("### Which signals are available?"),
    code(
        "date_cols = ['order_date', 'delivery_date', 'return_request_date']\n"
        "for col in date_cols:\n"
        "    train[col] = pd.to_datetime(train[col], errors='coerce')\n"
        "    test[col] = pd.to_datetime(test[col], errors='coerce')\n"
        "availability = pd.DataFrame(\n"
        "    {\n"
        "        'available?': ['yes', 'yes', 'yes', 'yes (prior_return_rate, prior_orders_count)', 'yes (customer_reason)'],\n"
        "        'evidence': [\n"
        "            f\"return_request_date {train['return_request_date'].min().date()} .. {test['return_request_date'].max().date()}\",\n"
        "            f\"{train['customer_id'].nunique():,} customers in train\",\n"
        "            f\"{train['product_id'].nunique():,} products in train\",\n"
        "            'summary of earlier returns only; earlier decision outcomes exist only through train labels',\n"
        "            f\"{train['customer_reason'].notna().mean():.1%} non-empty\",\n"
        "        ],\n"
        "    },\n"
        "    index=['timestamps', 'customer IDs', 'product IDs', 'previous-return information', 'return reasons'],\n"
        ")\n"
        "availability"
    ),
    code(
        "periods = pd.DataFrame(\n"
        "    {'first_request': [train['return_request_date'].min(), test['return_request_date'].min()],\n"
        "     'last_request': [train['return_request_date'].max(), test['return_request_date'].max()],\n"
        "     'rows': [len(train), len(test)]},\n"
        "    index=['train', 'test'],\n"
        ")\n"
        "print('test starts after train ends:', test['return_request_date'].min() > train['return_request_date'].max())\n"
        "periods"
    ),
    md(
        "**Reading:** the test period comes strictly *after* the training period, so the honest validation design is "
        "**time-based** (train on the past, check on the most recent months), never a random split."
    ),
    md("## 4. Deliverable 2 - Class-distribution chart"),
    code(
        "label_counts = train['label'].value_counts()\n"
        "label_share = (label_counts / len(train)).round(4)\n"
        "display(pd.DataFrame({'count': label_counts, 'share': label_share}))\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))\n"
        "label_counts.sort_values().plot.barh(ax=axes[0], color='#4c78a8', title='Return requests by label')\n"
        "for i, value in enumerate(label_counts.sort_values()):\n"
        "    axes[0].text(value, i, f' {value:,}', va='center')\n"
        "binary = (train['label'] == 'abusive').map({True: 'abusive', False: 'genuine (3 reasons)'}).value_counts()\n"
        "binary.plot.pie(ax=axes[1], autopct='%.1f%%', colors=['#54a24b', '#e45756'], ylabel='', title='Abusive vs genuine')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    code(
        "monthly = train.groupby(train['return_request_date'].dt.to_period('M'))['label'].apply(lambda s: (s == 'abusive').mean())\n"
        "monthly.plot(marker='o', ylim=(0, 0.15), title='Abusive share by month (train)')\n"
        "plt.ylabel('share abusive')\n"
        "plt.show()\n"
        "print(monthly.round(3).to_string())"
    ),
    md(
        "**Reading:** about 8% of returns are abusive, so the data is imbalanced: *accuracy is a useless metric* "
        "(approving everything is 92% 'accurate'). The abusive share moves between about 5.5% and 10% from month to month "
        "(peaks in December and March), so the base rate is not fixed and calibration must be checked on recent months."
    ),
    md("## 5. Deliverable 3 - Missing-value and duplicate report"),
    md("### 5.1 Missing values"),
    code(
        r"""missing_all = pd.concat({name: pd.DataFrame({'missing_count': t.isna().sum(), 'missing_%': (t.isna().mean() * 100).round(2)})
                         for name, t in tables.items()}, names=['table', 'column'])
print('Complete per-column report (every column of every table):')
with pd.option_context('display.max_rows', None):
    display(missing_all)
missing = missing_all[missing_all['missing_count'] > 0].reset_index()
print('Columns that actually contain missing values:')
missing"""
    ),
    code(
        "# Is the missing prior_return_rate explained by 'no prior orders'?\n"
        "for name, frame in [('train', train), ('test', test)]:\n"
        "    no_history = frame['prior_orders_count'] == 0\n"
        "    print(f\"{name}: prior_return_rate missing in {frame['prior_return_rate'].isna().sum():,} rows; \"\n"
        '          f"rows with prior_orders_count==0: {no_history.sum():,}; "\n'
        "          f\"missing exactly when no history: {(frame['prior_return_rate'].isna() == no_history).all()}\")\n"
        "\n"
        "# Which size charts / sizes are empty, and why?\n"
        "products = tables['products']\n"
        "print('products without size_chart_id:', products['size_chart_id'].isna().sum(),\n"
        "      '| their sizes:', products.loc[products['size_chart_id'].isna(), 'available_sizes'].unique()[:5])"
    ),
    md(
        "**Reading:** `prior_return_rate` is empty exactly for customers with no earlier orders, so the gap *is* a signal "
        "('first order') and must be kept as missing, not filled with 0. Products without a size chart are free-size items. "
        "Any other missing values in the size-chart table are measurements that do not apply to a garment type."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 3.2))\n"
        "(train.isna().mean() * 100).loc[lambda s: s > 0].plot.barh(ax=ax, color='#f58518', title='Train: missing % per column')\n"
        "ax.set_xlabel('% missing')\n"
        "plt.show()\n"
        "print('empty or blank customer_reason (train):', int((train['customer_reason'].fillna('').str.strip() == '').sum()))\n"
        "print('empty or blank customer_reason (test) :', int((test['customer_reason'].fillna('').str.strip() == '').sum()))"
    ),
    md("### 5.2 Duplicates and key integrity"),
    code(
        "def duplicate_report(frame: pd.DataFrame, name: str) -> dict:\n"
        "    content_cols = [c for c in frame.columns if c not in {'return_id', 'label'}]\n"
        "    return {\n"
        "        'table': name,\n"
        "        'rows': len(frame),\n"
        "        'duplicate full rows': int(frame.duplicated().sum()),\n"
        "        'duplicate return_id': int(frame['return_id'].duplicated().sum()),\n"
        "        'same content, different return_id': int(frame.duplicated(subset=content_cols).sum()),\n"
        "        'orders with >1 return': int(frame['order_id'].duplicated().sum()),\n"
        "        'same customer+product+size repeated': int(frame.duplicated(subset=['customer_id', 'product_id', 'size_ordered']).sum()),\n"
        "    }\n"
        "\n"
        "\n"
        "pd.DataFrame([duplicate_report(train, 'train'), duplicate_report(test, 'test')]).set_index('table').T"
    ),
    md("### 5.2b Conflicting labels and repeated events"),
    code(
        r"""content = [c for c in train.columns if c not in {'return_id', 'label'}]
same_content = train.groupby(content, dropna=False)['label'].nunique()
print('groups of identical requests (all columns except id and label):', len(same_content),
      '| with conflicting labels:', int((same_content > 1).sum()))
by_event = train.groupby(['customer_id', 'order_id', 'product_id', 'size_ordered'])['label'].agg(['size', 'nunique'])
print('customer+order+product+size events returned more than once:', int((by_event['size'] > 1).sum()),
      '| of those with different labels:', int(((by_event['size'] > 1) & (by_event['nunique'] > 1)).sum()))
print('same return_id with different labels: 0 by construction (return_id is unique:', train['return_id'].is_unique, ')')"""
    ),
    md(
        "**Reading:** there are no duplicate rows, no duplicate `return_id`s and no two identical requests with different labels. "
        "Only 6 customer+order+product+size events were returned twice, and 3 of those pairs carry different labels - that is a "
        "customer giving a different reason (or a different judged outcome) on the second return, not a data error. "
        "Repeated returns by the same customer are real behaviour and are kept as information."
    ),
    code(
        "ids_overlap = set(train['return_id']) & set(test['return_id'])\n"
        "print('return_ids shared between train and test:', len(ids_overlap))\n"
        "\n"
        "customers, size_charts = tables['customers'], tables['size_charts']\n"
        "both = pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)\n"
        "integrity = pd.DataFrame(\n"
        "    {\n"
        "        'check': [\n"
        "            'every product_id exists in products.csv',\n"
        "            'every customer_id exists in customers.csv',\n"
        "            'every size_chart_id of a product exists in size_charts.csv',\n"
        "            'customer_id unique in customers.csv',\n"
        "            'product_id unique in products.csv',\n"
        "        ],\n"
        "        'passes': [\n"
        "            both['product_id'].isin(products['product_id']).all(),\n"
        "            both['customer_id'].isin(customers['customer_id']).all(),\n"
        "            products['size_chart_id'].dropna().isin(size_charts['size_chart_id']).all(),\n"
        "            customers['customer_id'].is_unique,\n"
        "            products['product_id'].is_unique,\n"
        "        ],\n"
        "    }\n"
        ")\n"
        "integrity"
    ),
    code(
        "seen_in_train = test['customer_id'].isin(train['customer_id'])\n"
        "print(f'test requests from customers already seen in train: {seen_in_train.mean():.1%}')\n"
        "print(f\"customers with more than one request overall: {both['customer_id'].duplicated(keep=False).groupby(both['customer_id']).first().mean():.1%}\")"
    ),
    md(
        "**Reading:** no duplicate `return_id`s, and joins to the catalogue are clean. Several returns can belong to "
        "the same order or customer - that is *real information* (repeat behaviour), not a data error, so we do not drop "
        "those rows. Because most test customers also appear in train, **customer-level leakage must be handled "
        "in Phase 2** (history features may only use the past; cross-validation must group by customer)."
    ),
    md("### 5.3 Sanity checks on values and dates"),
    code(
        "checks = {\n"
        "    'order_date <= delivery_date': (train['order_date'] <= train['delivery_date']).mean(),\n"
        "    'delivery_date <= return_request_date': (train['delivery_date'] <= train['return_request_date']).mean(),\n"
        "    'days_since_delivery == return - delivery (days)': (\n"
        "        (train['return_request_date'] - train['delivery_date']).dt.days == train['days_since_delivery']\n"
        "    ).mean(),\n"
        "    'delivery_days >= 0': (train['delivery_days'] >= 0).mean(),\n"
        "    'order_value >= item_price': (train['order_value_inr'] >= train['item_price_inr']).mean(),\n"
        "    '0 <= discount_pct <= 100': train['discount_pct'].between(0, 100).mean(),\n"
        "    '0 <= prior_return_rate <= 1 (where present)': train['prior_return_rate'].dropna().between(0, 1).mean(),\n"
        "    'customer_tenure_days >= 0': (train['customer_tenure_days'] >= 0).mean(),\n"
        "    'days_since_delivery <= 30 (return window)': (train['days_since_delivery'] <= 30).mean(),\n"
        "}\n"
        "pd.DataFrame({'share of rows passing': pd.Series(checks).round(4)})"
    ),
    code("train.select_dtypes('number').describe().T[['min', '25%', '50%', '75%', 'max']].round(2)"),
    md(
        "**Reading:** rows that fail a check are not deleted automatically - Phase 2 decides case by case. "
        "A return that is *late* (`days_since_delivery` near or over 30) is a feature, not an error."
    ),
    md("## 6. Deliverable 4 - Evaluation-metric specification"),
    md(
        "### 6.1 The cost matrix (source of truth: the organisers' README and `evaluate.py`)\n"
        "Rows are the true class, columns are our decision. **Lower total cost is better.**"
    ),
    code(
        "ACTIONS = ['auto_approve', 'inspect', 'escalate']\n"
        "COST = pd.DataFrame(\n"
        "    {'auto_approve': [0.0, 1.0], 'inspect': [0.5, 0.3], 'escalate': [10.0, 0.0]},\n"
        "    index=['genuine (any of 3)', 'abusive'],\n"
        ")\n"
        "COST"
    ),
    code(
        "# Cross-check with the official scorer, if we can find it.\n"
        "if SCORER is not None:\n"
        "    spec = importlib.util.spec_from_file_location('official_evaluate', SCORER)\n"
        "    official = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(official)\n"
        "    for truth, row in official.COST.items():\n"
        "        label = 'abusive' if truth == 'abusive' else 'genuine (any of 3)'\n"
        "        assert all(COST.loc[label, a] == row[a] for a in ACTIONS), f'mismatch for {truth}'\n"
        "    print('cost matrix matches evaluate.py')\n"
        "    print('score weights   :', {'savings': official.W_SAVINGS, 'fairness': official.W_FAIRNESS, 'size_chart': official.W_SIZE_CHART})\n"
        "    print('fairness cut-offs: low value <', official.LOW_VALUE_INR, 'INR, new customer <', official.NEW_CUSTOMER_DAYS, 'days')\n"
        "else:\n"
        "    print('evaluate.py not found - using the README table above')"
    ),
    md(
        "### 6.1b The exact penalties, and the formulas that were floating around\n"
        "Several different formulas have been written down during this project. **Only the last row is the benchmark.**\n\n"
        "| source | wrongly escalate a genuine customer | miss an abuser (auto-approve) | inspect a genuine customer | inspect an abuser | verdict |\n"
        "|---|---|---|---|---|---|\n"
        "| pasted implementation plan: `10 x MissedFraud + 1 x WrongBlock + 0.2 x UnnecessaryInspect` | 1 | **10** | 0.2 | - | **reversed direction, rejected** |\n"
        "| my earlier synthetic Phase 0 (`evaluate.py` of the synthetic project) | 1.7 | 1.0 | 0.2 | 0.5 | obsolete, replaced |\n"
        "| review note: `10 x WrongBlock + 1 x MissedFraud + 0.2 x UnnecessaryInspect` | 10 | 1 | 0.2 | - | right direction, but the 0.2 is **not** the official value |\n"
        "| **official `evaluate.py` / README** | **10** | **1** | **0.5** | **0.3** | **the benchmark** |\n\n"
        "Definitions used in this project, taken from the official matrix:\n"
        "- **wrong block** = `escalate` on a genuine return: cost 10\n"
        "- **missed fraud** = `auto_approve` on an abusive return: cost 1\n"
        "- **unnecessary inspect** = `inspect` on a genuine return: cost 0.5\n"
        "- **inspect on an abuser** is not free either: cost 0.3 (cheaper than missing it, dearer than escalating it)\n"
        "- **escalate an abuser** and **approve a genuine return**: cost 0"
    ),
    code(
        r"""import inspect
import tempfile

if SCORER is None:
    print('evaluate.py not found - skipping the executable check')
else:
    source = inspect.getsource(official.score).splitlines()
    print('lines of official score() that compute the cost and the savings:')
    for line in source:
        if any(key in line for key in ('cost =', 'cost_team', 'cost_all_aa', 'cost_oracle', 'denom', 'savings =')):
            print('   ', line.strip())
    # Run the real scorer on three trivial policies for the training rows.
    truth_cols = ['return_id', 'label', 'customer_tenure_days', 'order_value_inr', 'customer_reason']
    abusive_flag = train['label'] == 'abusive'
    policies = {
        'approve everything': pd.Series('auto_approve', index=train.index),
        'oracle (escalate abusers, approve the rest)': abusive_flag.map({True: 'escalate', False: 'auto_approve'}),
        'inspect everything': pd.Series('inspect', index=train.index),
    }
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        train[truth_cols].to_csv(Path(tmp) / 'truth.csv', index=False)
        for name, decision in policies.items():
            pd.DataFrame({'return_id': train['return_id'], 'decision': decision, 'confidence': 0.5, 'justification': 'x'}).to_csv(
                Path(tmp) / 'pred.csv', index=False)
            result = official.score(Path(tmp) / 'pred.csv', Path(tmp) / 'truth.csv', None, None)
            rows.append({'policy': name, 'total_cost': result['cost']['team_total'], 'savings': result['components']['savings'],
                         'primary_score': result['primary_score']})
    display(pd.DataFrame(rows).set_index('policy'))"""
    ),
    md(
        "**Reading:** the real scorer returns savings 0 for 'approve everything' and 1 for the oracle, and **clips** "
        "'inspect everything' to -1 (its unclipped value is about -5.2). So the cost matrix, the savings formula and "
        "the score direction (higher primary score is better, lower cost is better) are all confirmed by running the "
        "official code, not just by reading it."
    ),
    md(
        "**Wrongly escalating one honest customer costs as much as missing ten abusers.** "
        "`inspect` is cheap but not free. This pushes the policy to be careful: escalate only when almost certain."
    ),
    md(
        "### 6.2 What that means for decisions\n"
        "For a request with abuse probability *p*, the expected cost of each action is\n\n"
        "- approve: `p * 1`\n- inspect: `(1-p) * 0.5 + p * 0.3`\n- escalate: `(1-p) * 10`\n\n"
        "and the best action is the cheapest one. The break-even probabilities follow from the matrix:"
    ),
    code(
        "def expected_cost(p: np.ndarray) -> pd.DataFrame:\n"
        "    p = np.asarray(p, dtype=float)\n"
        "    return pd.DataFrame({a: (1 - p) * COST.loc['genuine (any of 3)', a] + p * COST.loc['abusive', a] for a in ACTIONS})\n"
        "\n"
        "\n"
        "g, a = COST.loc['genuine (any of 3)'], COST.loc['abusive']\n"
        "approve_to_inspect = (g['auto_approve'] - g['inspect']) / ((g['auto_approve'] - g['inspect']) - (a['auto_approve'] - a['inspect']))\n"
        "inspect_to_escalate = (g['inspect'] - g['escalate']) / ((g['inspect'] - g['escalate']) - (a['inspect'] - a['escalate']))\n"
        "print(f'approve -> inspect  when P(abusive) > {approve_to_inspect:.3f}')\n"
        "print(f'inspect -> escalate when P(abusive) > {inspect_to_escalate:.3f}')\n"
        "\n"
        "grid = np.linspace(0, 1, 201)\n"
        "curves = expected_cost(grid)\n"
        "curves.index = grid\n"
        "ax = curves.plot(title='Expected cost per action vs P(abusive)')\n"
        "for x in (approve_to_inspect, inspect_to_escalate):\n"
        "    ax.axvline(x, color='grey', ls='--', lw=0.8)\n"
        "ax.set_xlabel('P(abusive)'); ax.set_ylabel('expected cost'); ax.set_ylim(0, 3)\n"
        "plt.show()"
    ),
    md(
        "**Reading:** approve below ~0.42, inspect between ~0.42 and ~0.97, escalate only above ~0.97. "
        "This only works if the probabilities are **calibrated** - that is why calibration is its own phase."
    ),
    md("### 6.3 Reference points: how much is there to win?"),
    code(
        "is_abusive = (train['label'] == 'abusive').to_numpy()\n"
        "cost_approve_all = np.where(is_abusive, a['auto_approve'], g['auto_approve']).sum()\n"
        "cost_oracle = np.where(is_abusive, a.min(), g.min()).sum()\n"
        "cost_inspect_all = np.where(is_abusive, a['inspect'], g['inspect']).sum()\n"
        "cost_escalate_all = np.where(is_abusive, a['escalate'], g['escalate']).sum()\n"
        "reference = pd.DataFrame(\n"
        "    {\n"
        "        'total_cost': [cost_approve_all, cost_oracle, cost_inspect_all, cost_escalate_all],\n"
        "        'savings_vs_approve_all': [\n"
        "            (cost_approve_all - c) / (cost_approve_all - cost_oracle)\n"
        "            for c in (cost_approve_all, cost_oracle, cost_inspect_all, cost_escalate_all)\n"
        "        ],\n"
        "    },\n"
        "    index=['approve everything', 'oracle (knows every label)', 'inspect everything', 'escalate everything'],\n"
        ").round(3)\n"
        "reference"
    ),
    md(
        "**Reading:** *savings* = how much of the gap between 'approve everything' (0) and a perfect oracle (1) "
        "we close. Inspecting or escalating everything is **worse** than doing nothing (negative savings), so a model must "
        "be selective. The scorer clips savings to [-1, 1]."
    ),
    md(
        "### 6.4 The full scoring formula\n"
        "```\n"
        "primary_score = 0.7 * savings + 0.2 * fairness_effective + 0.1 * size_chart_score\n"
        "fairness_effective = fairness_score * min(1, max(0, savings) / 0.20)     # fairness only counts if savings > 0\n"
        "```\n"
        "| part | weight | definition |\n|---|---|---|\n"
        "| savings | 0.7 | (cost of approve-all - our cost) / (cost of approve-all - oracle cost), clipped to [-1, 1] |\n"
        "| fairness | 0.2 | on **genuine** rows only: rate of being flagged (inspect or escalate), compared for "
        "*new (<90 days) vs established*, *low value (<1000 INR) vs other*, *non-English vs English*. "
        "Each pair: ratio of smoothed rates (+1 percentage point), score = clip(1 - (ratio - 1), 0, 1); fairness = mean of 3 pairs |\n"
        "| size chart | 0.1 | overlap of our 5 product IDs with the true top 5 products by genuine fit-related returns |\n\n"
        "Also reported (not in the primary score): calibration error and Brier score of `confidence` as P(abusive), "
        "and how many flagged requests had confidence below 0.2 - judges use this to catch 'fairness by inspecting "
        "extra low-risk customers'. Missing test IDs count as worst case; more than 50% missing is rejected."
    ),
    md(
        "### 6.5 Required submission format\n"
        "| file | columns | rules |\n|---|---|---|\n"
        "| `submission/R2_predictions.csv` | `return_id, decision, confidence, justification` | every test `return_id` exactly once; "
        "decision in {auto_approve, inspect, escalate}; confidence a number in [0, 1] |\n"
        "| `submission/R2_size_chart_top5.csv` | `rank, product_id` | exactly 5 rows, ranks 1-5, 5 distinct valid product IDs |"
    ),
    code(
        "# A tiny executable specification of the submission rules (reused later as tests).\n"
        "def validate_predictions(pred: pd.DataFrame, expected_ids: pd.Series) -> list[str]:\n"
        "    problems = []\n"
        "    required = ['return_id', 'decision', 'confidence', 'justification']\n"
        "    if list(pred.columns[:4]) != required:\n"
        "        problems.append(f'columns must start with {required}')\n"
        "        return problems\n"
        "    if pred['return_id'].duplicated().any():\n"
        "        problems.append('duplicate return_id')\n"
        "    if set(expected_ids) - set(pred['return_id']):\n"
        "        problems.append('some test return_ids are missing')\n"
        "    if not pred['decision'].isin(ACTIONS).all():\n"
        "        problems.append('invalid decision value')\n"
        "    if not pd.to_numeric(pred['confidence'], errors='coerce').between(0, 1).all():\n"
        "        problems.append('confidence outside [0, 1]')\n"
        "    return problems\n"
        "\n"
        "\n"
        "good = pd.DataFrame({'return_id': test['return_id'], 'decision': 'auto_approve', 'confidence': 0.08, 'justification': 'baseline'})\n"
        "bad = good.iloc[:100].assign(decision='block')\n"
        "print('valid example  ->', validate_predictions(good, test['return_id']) or 'no problems')\n"
        "print('invalid example ->', validate_predictions(bad, test['return_id']))"
    ),
    md(
        "## 7. Are the date columns genuine event timestamps?\n"
        "A time-based split is only defensible if the dates are real and internally consistent. We test that instead of assuming it."
    ),
    code(
        r"""rows = []
for name, frame in [('train', train), ('test', test)]:
    rows.append({
        'split': name,
        'unparseable dates': int(frame[date_cols].isna().sum().sum()),
        'order <= delivery': float((frame['order_date'] <= frame['delivery_date']).mean()),
        'delivery <= return request': float((frame['delivery_date'] <= frame['return_request_date']).mean()),
        'days_since_delivery equals date difference': float(
            ((frame['return_request_date'] - frame['delivery_date']).dt.days == frame['days_since_delivery']).mean()),
        'delivery_days equals date difference': float(
            ((frame['delivery_date'] - frame['order_date']).dt.days == frame['delivery_days']).mean()),
    })
display(pd.DataFrame(rows).set_index('split').T)
both_dates = pd.concat([train.assign(split='train'), test.assign(split='test')])
per_month = both_dates.groupby([both_dates['return_request_date'].dt.to_period('M'), 'split']).size().unstack(fill_value=0)
per_month.plot.bar(stacked=True, title='Requests per month (train then test)', color=['#4c78a8', '#f58518'])
plt.show()
days = pd.date_range(both_dates['return_request_date'].min(), both_dates['return_request_date'].max())
missing_days = days.difference(both_dates['return_request_date'].unique())
print('calendar days with no request at all:', len(missing_days))
print('last train day:', train['return_request_date'].max().date(), '| first test day:', test['return_request_date'].min().date())"""
    ),
    md(
        "**Reading:** the three dates agree with the numeric columns `delivery_days` and `days_since_delivery` on every row, "
        "so they are real event dates, not decoration. `return_request_date` is the clock for the time-based split. "
        "Train and test do not overlap in time (train ends 30 June 2026, test starts 1 July 2026)."
    ),
    md(
        "## 8. Leakage audit of the catalogue and customer files\n"
        "Fields that were computed from the labels (for example a per-product fit-issue proportion) would leak the target. "
        "We check names, and then scan every column for suspicious predictive power."
    ),
    code(
        r"""import re

suspicious = re.compile(r'prop|share|rate|return|fit_issue|label|abus|fraud|refund|complaint|score|risk', re.I)
for name in ['products', 'size_charts', 'customers']:
    hits = [c for c in tables[name].columns if suspicious.search(c)]
    print(f'{name:12} columns: {list(tables[name].columns)}')
    print(f'{"":12} label-like names: {hits or "none"}')
print()
print('return_requests columns with label-like names:',
      [c for c in test.columns if suspicious.search(c) and c not in {'return_id', 'return_request_date'}])"""
    ),
    code(
        r"""# Single-column leak scan: how well does ONE column (binned if numeric) reproduce the label on its own?
scan_source = train.merge(products[['product_id', 'brand', 'gender', 'fabric', 'fit_type', 'colour', 'price_inr']], on='product_id', how='left')
scan_source = scan_source.merge(customers[['customer_id', 'city_tier']], on='customer_id', how='left')
ignore = {'return_id', 'order_id', 'customer_id', 'product_id', 'label', 'abusive', 'customer_reason',
          'order_date', 'delivery_date', 'return_request_date', 'non_english', 'n_words', 'has_emoji'}
majority = scan_source['label'].value_counts(normalize=True).iloc[0]
results = {}
for column in [c for c in scan_source.columns if c not in ignore]:
    values = scan_source[column]
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > 20:
        values = pd.qcut(values, 20, duplicates='drop')
    table = pd.crosstab(values.astype(str), scan_source['label'])
    results[column] = table.max(axis=1).sum() / table.to_numpy().sum()
scan = pd.Series(results).sort_values(ascending=False).round(3).to_frame('label accuracy using only this column')
print(f'always guessing the majority label gives {majority:.3f}; a leaked label column would give close to 1.0')
scan"""
    ),
    md(
        "**Reading:** no catalogue or customer column carries label-derived content: the products and customers files hold "
        "only descriptive fields (title, brand, price, fabric, size chart, signup date, city tier), and no single column "
        "gets above 0.53 against a 0.48 majority baseline. The highest (size ordered, sub-category, fit type) have many levels "
        "and are legitimately linked to fit complaints, nowhere near the ~1.0 a leaked label column would give. Column names containing 'rate' or 'return' exist only in the request file "
        "(`prior_return_rate`, `return_request_date`) and describe the customer's *history up to the request*."
    ),
    md(
        "## 9. Train vs test: what drift is measured here?\n"
        "We measure **covariate drift** (do the inputs look the same?). Label drift on the test period cannot be measured "
        "because its labels are hidden; the month-by-month label rate in section 4 is the only evidence we have."
    ),
    code(
        r"""def psi_numeric(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(a.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = pd.cut(a, edges).value_counts(normalize=True, sort=False) + 1e-4
    q = pd.cut(b, edges).value_counts(normalize=True, sort=False) + 1e-4
    return float(((p - q) * np.log(p / q)).sum())


def psi_categorical(a: pd.Series, b: pd.Series) -> float:
    p, q = a.value_counts(normalize=True), b.value_counts(normalize=True)
    index = p.index.union(q.index)
    p, q = p.reindex(index, fill_value=0) + 1e-4, q.reindex(index, fill_value=0) + 1e-4
    return float(((p - q) * np.log(p / q)).sum())


numeric_cols = ['delivery_days', 'days_since_delivery', 'order_value_inr', 'item_price_inr', 'discount_pct', 'items_in_order',
                'customer_tenure_days', 'prior_orders_count', 'prior_return_rate']
category_cols = ['category', 'sub_category', 'size_ordered', 'payment_method', 'fulfilment_centre']
drift = pd.DataFrame({
    'type': ['numeric'] * len(numeric_cols) + ['categorical'] * len(category_cols),
    'train mean / top share': [train[c].mean() for c in numeric_cols] + [train[c].value_counts(normalize=True).iloc[0] for c in category_cols],
    'test mean / top share': [test[c].mean() for c in numeric_cols] + [test[c].value_counts(normalize=True).iloc[0] for c in category_cols],
    'PSI': [psi_numeric(train[c], test[c]) for c in numeric_cols] + [psi_categorical(train[c], test[c]) for c in category_cols],
}, index=numeric_cols + category_cols)
drift['verdict'] = np.where(drift['PSI'] < 0.1, 'stable', np.where(drift['PSI'] < 0.25, 'moderate', 'large'))
display(drift.round(4))
print('rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 large')"""
    ),
    code(
        r"""# Do reason texts repeat across the split boundary? (template leakage)
def normalised(texts: pd.Series) -> pd.Series:
    return texts.fillna('').str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True).str.split().str.join(' ')


train_texts, test_texts = normalised(train['customer_reason']), normalised(test['customer_reason'])
print(f'test reason texts that appear verbatim in train: {test_texts.isin(set(train_texts)).mean():.1%}')
cutoff = pd.Timestamp('2026-05-01')
early, late_part = train_texts[train['return_request_date'] < cutoff], train_texts[train['return_request_date'] >= cutoff]
print(f'train May-June texts that appear verbatim before May: {late_part.isin(set(early)).mean():.1%}')
duplicates = train_texts.duplicated(keep=False)
print(f'train rows whose exact text occurs more than once: {duplicates.mean():.1%}')
print('abuse rate of rows with a very common text (>=50 uses):',
      round(float((train['label'] == 'abusive')[train_texts.map(train_texts.value_counts()) >= 50].mean()), 3),
      '| rows:', int((train_texts.map(train_texts.value_counts()) >= 50).sum()))"""
    ),
    md(
        "**Reading:** every input is stable between train and test (all PSI below 0.1; the largest is discount at 0.063, which "
        "matches the 28 -> 32 average discount shift). About 27% of test reason texts also appear word for word in train, and "
        "28% of train rows share their exact text with another row. A text model may recognise *phrases*, which is "
        "fine, but it must be cross-validated **grouped by customer and by time** so that a customer's own earlier wording "
        "cannot leak into their later rows. That rule is carried into Phase 2."
    ),
    md(
        "## References supplied for this project\n"
        "Only the first group below has been used directly so far (calibration, explanations, fairness). "
        "The others were provided by the project owner and will be reviewed when their phase starts.\n\n"
        "- Guo et al., *On Calibration of Modern Neural Networks* - https://arxiv.org/abs/1706.04599 (calibration, Phase 8)\n"
        "- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions (SHAP)* - https://arxiv.org/abs/1705.07874 (explanations, Phase 12)\n"
        "- Kearns et al., *Preventing Fairness Gerrymandering* - https://proceedings.mlr.press/v80/kearns18a.html (subgroup fairness, Phase 10)\n"
        "- https://arxiv.org/pdf/2607.27143.pdf (supplied; not yet reviewed)\n"
        "- https://www.tandfonline.com/doi/full/10.1080/2573234X.2020.1776164 (supplied; not yet reviewed)\n"
        "- https://www.vldb.org/pvldb/vol19/p670-jackson.pdf (supplied; not yet reviewed)\n"
        "- https://icml.cc/virtual/2025/47490 (supplied; not yet reviewed)"
    ),
    md("## 10. Data-leakage guard rails agreed for the rest of the project"),
    md(
        "1. `label` is never a model input; label-derived history is allowed **only** from requests that happened earlier "
        "and whose outcome could already be known.\n"
        "2. Features for a request use only information dated **before** that request.\n"
        "3. Validation is time-based (train on the past, test on the latest months); cross-validation groups by customer.\n"
        "4. Language of the text is **not** a model feature (it is a fairness group, and could become a proxy).\n"
        "5. Preprocessing (encoders, TF-IDF) is fitted on training rows only."
    ),
    md("## 11. Phase 1 summary"),
    code(
        "summary = pd.Series(\n"
        "    {\n"
        "        'return requests (train + test)': f'{len(train) + len(test):,}',\n"
        "        'train period': f\"{train['return_request_date'].min().date()} to {train['return_request_date'].max().date()}\",\n"
        "        'test period': f\"{test['return_request_date'].min().date()} to {test['return_request_date'].max().date()}\",\n"
        "        'abusive share (train)': f\"{(train['label'] == 'abusive').mean():.1%}\",\n"
        "        'columns missing values': ', '.join(sorted(set(missing['column']))) or 'none',\n"
        "        'duplicate return_id': int(train['return_id'].duplicated().sum() + test['return_id'].duplicated().sum()),\n"
        "        'test customers seen in train': f'{seen_in_train.mean():.1%}',\n"
        "        'approve -> inspect threshold': round(approve_to_inspect, 3),\n"
        "        'inspect -> escalate threshold': round(inspect_to_escalate, 3),\n"
        "    }\n"
        ")\n"
        "summary.to_frame('Phase 1 findings')"
    ),
    md(
        "**Findings carried into Phase 2**\n"
        "- Imbalanced target (~8% abusive, 5.5%-10% by month) -> judge by cost and PR-AUC, not accuracy.\n"
        "- Test comes after train -> time-based split; no random shuffling.\n"
        "- Most test customers are known -> strict point-in-time history and customer-grouped folds.\n"
        "- Missing `prior_return_rate` = first-time customer -> keep as a signal.\n"
        "- Costs make escalation very expensive for honest customers -> calibrated probabilities and cautious thresholds.\n\n"
        "**Next:** Phase 2 - cleaning, splitting and leakage prevention."
    ),
]


def main() -> None:
    notebook = new_notebook(cells=CELLS)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
