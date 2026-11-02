"""Create notebooks/eda_original_data.ipynb (self-contained: pandas, numpy, matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "eda_original_data.ipynb"
FINDINGS_FILE = Path(__file__).with_name("eda_findings.md")

CELLS = [
    md(
        "# EDA on the original (official) data\n"
        "Exploratory data analysis of the R2 returns data, with the findings collected in the last section.\n\n"
        "**Questions**\n"
        "1. What separates abusive returns from genuine ones?\n"
        "2. Which signals are strong, which are weak, and which are traps?\n"
        "3. Does the test period look like the training period?\n"
        "4. Are some customer groups at risk of being treated unfairly?\n"
        "5. Which products drive fit-related returns?\n\n"
        "Needs only pandas, numpy and matplotlib. Put the five CSVs in a `data/` folder next to this notebook."
    ),
    code(
        "from pathlib import Path\n"
        "import re\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "pd.set_option('display.max_columns', 40, 'display.width', 150, 'display.max_colwidth', 90)\n"
        "plt.rcParams.update({'figure.figsize': (7, 3.4), 'axes.spines.top': False, 'axes.spines.right': False})\n"
        "\n"
        "DATA_DIR = next(p for p in map(Path, ['data', '../data', '../FULL_CODE/data']) if (p / 'return_requests_train.csv').exists())\n"
        "train = pd.read_csv(DATA_DIR / 'return_requests_train.csv', parse_dates=['order_date', 'delivery_date', 'return_request_date'])\n"
        "test = pd.read_csv(DATA_DIR / 'return_requests_test.csv', parse_dates=['order_date', 'delivery_date', 'return_request_date'])\n"
        "products = pd.read_csv(DATA_DIR / 'products.csv')\n"
        "customers = pd.read_csv(DATA_DIR / 'customers.csv', parse_dates=['signup_date'])\n"
        "train['abusive'] = (train['label'] == 'abusive').astype(int)\n"
        "BASE = train['abusive'].mean()\n"
        "print(f'train {train.shape}, test {test.shape}, base abuse rate {BASE:.3%}')\n"
        "\n"
        "\n"
        "def abuse_by(group, min_rows: int = 1) -> pd.DataFrame:\n"
        '    """Abusive share, row count and lift over the base rate for each group."""\n'
        "    table = train.groupby(group, observed=True)['abusive'].agg(abuse_rate='mean', rows='size')\n"
        "    table['lift'] = table['abuse_rate'] / BASE\n"
        "    return table[table['rows'] >= min_rows].round(3)\n"
        "\n"
        "\n"
        "def bar_rate(table: pd.DataFrame, title: str) -> None:\n"
        "    ax = table['abuse_rate'].plot.bar(color='#4c78a8', title=title, rot=30)\n"
        "    ax.axhline(BASE, color='#e45756', ls='--', lw=1, label=f'base rate {BASE:.1%}')\n"
        "    ax.set_ylabel('abusive share'); ax.legend(); plt.tight_layout(); plt.show()"
    ),
    md(
        "## 1. The target\nAbusive returns are rare, so any useful metric must be about *cost* or *ranking*, not accuracy."
    ),
    code(
        "counts = train['label'].value_counts()\n"
        "display(pd.DataFrame({'rows': counts, 'share': (counts / len(train)).round(3)}))\n"
        "counts.sort_values().plot.barh(color='#4c78a8', title='Rows per label'); plt.show()\n"
        "print(f'A model that approves everything is {1 - BASE:.1%} \"accurate\" and catches nothing.')"
    ),
    code(
        "monthly = abuse_by(train['return_request_date'].dt.to_period('M'))\n"
        "monthly['abuse_rate'].plot(marker='o', ylim=(0, 0.15), title='Abusive share per month'); plt.show()\n"
        "monthly"
    ),
    md(
        "## 2. Timing: how long after delivery is the return requested?\nWardrobing (wear once, return) should show up as *late* returns."
    ),
    code(
        "days = pd.cut(train['days_since_delivery'], [-1, 1, 3, 7, 14, 21, 30], labels=['0-1', '2-3', '4-7', '8-14', '15-21', '22-30'])\n"
        "t = abuse_by(days); display(t); bar_rate(t, 'Abusive share by days between delivery and return request')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 3.4))\n"
        "for label, colour in [('abusive', '#e45756'), ('genuine', '#4c78a8')]:\n"
        "    values = train.loc[(train['abusive'] == 1) == (label == 'abusive'), 'days_since_delivery']\n"
        "    ax.hist(values, bins=range(0, 32), alpha=0.6, density=True, label=label, color=colour)\n"
        "ax.set_xlabel('days since delivery'); ax.legend(); ax.set_title('Distribution of return delay'); plt.show()\n"
        "print(train.groupby('abusive')['days_since_delivery'].describe().round(1))"
    ),
    code(
        "t = abuse_by(pd.cut(train['delivery_days'], [-1, 2, 4, 6, 9, 30])); display(t)\n"
        "print('Delivery speed is a weak signal compared with return delay.')"
    ),
    md("## 3. Money: order value, item price, discount"),
    code(
        "price_band = pd.qcut(train['item_price_inr'], 8, duplicates='drop')\n"
        "t = abuse_by(price_band); display(t); bar_rate(t, 'Abusive share by item price band (INR)')"
    ),
    code(
        "t = abuse_by(pd.cut(train['discount_pct'], [-1, 0, 10, 20, 30, 50, 90])); display(t)\n"
        "bar_rate(t, 'Abusive share by discount %')\n"
        "value_band = pd.cut(train['order_value_inr'], [0, 500, 1000, 2000, 4000, 8000, 1e6])\n"
        "display(abuse_by(value_band))"
    ),
    md("## 4. Basket, payment, warehouse, category"),
    code(
        "display(abuse_by(train['items_in_order'].clip(upper=5).rename('items_in_order (5 = 5+)')))\n"
        "display(abuse_by('payment_method'))\n"
        "display(abuse_by('fulfilment_centre'))"
    ),
    code(
        "t = abuse_by('category').sort_values('abuse_rate'); display(t); bar_rate(t, 'Abusive share by category')\n"
        "sub = abuse_by('sub_category', min_rows=150).sort_values('abuse_rate')\n"
        "sub['abuse_rate'].tail(12).plot.barh(title='Highest abuse rate by sub-category (>=150 rows)'); plt.show()\n"
        "sub.tail(6)"
    ),
    md("## 5. The product catalogue"),
    code(
        "catalogue = train.merge(products[['product_id', 'brand', 'fabric', 'fit_type', 'price_inr']], on='product_id', how='left')\n"
        "catalogue['mrp_ratio'] = catalogue['item_price_inr'] / catalogue['price_inr']\n"
        "display(catalogue.groupby('fit_type')['abusive'].agg(['mean', 'size']).round(3).sort_values('mean'))\n"
        "display(catalogue.groupby('fabric')['abusive'].agg(['mean', 'size']).round(3).sort_values('mean').tail(6))\n"
        "print('item_price / catalogue price (should be <= 1 after discount):')\n"
        "print(catalogue['mrp_ratio'].describe().round(3).to_string())"
    ),
    md("## 6. The customer\nRepeat behaviour is the richest evidence a retailer has."),
    code(
        "t = abuse_by(pd.cut(train['prior_return_rate'], [-0.01, 0.1, 0.25, 0.5, 0.75, 1.0])); display(t)\n"
        "first_order = train['prior_return_rate'].isna()\n"
        "print(f'first-order customers (no history): abuse {train.loc[first_order, \"abusive\"].mean():.3f} '\n"
        "      f'vs others {train.loc[~first_order, \"abusive\"].mean():.3f}')\n"
        "display(abuse_by(pd.cut(train['customer_tenure_days'], [-1, 30, 90, 180, 365, 730, 5000])))\n"
        "display(abuse_by(pd.cut(train['prior_orders_count'], [-1, 0, 2, 5, 10, 20, 500])))"
    ),
    code(
        "per_customer = train.groupby('customer_id')['abusive'].agg(abusive_returns='sum', returns='size')\n"
        "print('customers in train:', len(per_customer))\n"
        "print('customers with at least one abusive return:', int((per_customer['abusive_returns'] > 0).sum()))\n"
        "repeat = per_customer[per_customer['abusive_returns'] >= 3]\n"
        'print(f\'customers with >=3 abusive returns: {len(repeat)} -> {repeat["abusive_returns"].sum() / train["abusive"].sum():.1%} of all abusive returns\')\n'
        "display(per_customer.groupby(per_customer['returns'].clip(upper=8))['abusive_returns'].agg(['mean', 'size']).round(3).rename(\n"
        "    columns={'mean': 'avg abusive returns per customer', 'size': 'customers'}))"
    ),
    code(
        "# Does an earlier abusive return predict the next one? (strictly earlier rows only)\n"
        "ordered = train.sort_values(['customer_id', 'return_request_date', 'return_id']).copy()\n"
        "ordered['earlier_abusive'] = ordered.groupby('customer_id')['abusive'].cumsum() - ordered['abusive']\n"
        "ordered['earlier_returns'] = ordered.groupby('customer_id').cumcount()\n"
        "display(ordered.groupby(ordered['earlier_abusive'].clip(upper=3))['abusive'].agg(['mean', 'size']).round(3))\n"
        "display(ordered.groupby(ordered['earlier_returns'].clip(upper=6))['abusive'].agg(['mean', 'size']).round(3))"
    ),
    code(
        "extra = train.merge(customers, on='customer_id', how='left')\n"
        "display(extra.groupby('city_tier')['abusive'].agg(['mean', 'size']).round(3))\n"
        "gap = (extra['order_date'] - extra['signup_date']).dt.days\n"
        "print('order date minus signup date (days):'); print(gap.describe().round(0).to_string())"
    ),
    md("## 7. Interactions\nSignals rarely act alone."),
    code(
        "late = train['days_since_delivery'] > 14\n"
        "cod = train['payment_method'] == 'COD'\n"
        "single = train['items_in_order'] == 1\n"
        "grid = train.assign(late=late.map({True: 'late (>14d)', False: 'on time'}), cod=cod.map({True: 'COD', False: 'prepaid'}))\n"
        "pivot = grid.pivot_table(index='late', columns='cod', values='abusive', aggfunc='mean').round(3)\n"
        "display(pivot)\n"
        "count = grid.pivot_table(index='late', columns='cod', values='abusive', aggfunc='size'); display(count)\n"
        "risky = late & cod & single\n"
        "print(f'late + COD + single item: {risky.sum():,} rows, abuse rate {train.loc[risky, \"abusive\"].mean():.3f} (base {BASE:.3f})')"
    ),
    md("## 8. The reason text\nEnglish, Hinglish and Hindi in Latin script, with typos and emoji."),
    code(
        "for label in train['label'].unique():\n"
        "    print(f'--- {label}')\n"
        "    for text in train.loc[train['label'] == label, 'customer_reason'].sample(4, random_state=3):\n"
        "        print('   ', text)"
    ),
    code(
        "HINDI = set('hai hain nahi nhi nahin bahut bohot mein ka ki ke ko se pe tha thi hua hui gaya gayi aaya raha rahi kar karo diya kiya '\n"
        "            'liya mila chahiye wapas mujhe mera meri maine hoon bhi bilkul kharab bekar kapda silai rang naap chhota chota bada dheela '\n"
        "            'tang lambai jaisa sahi galat saman dabba khali paise theek zyada kuch abhi aur toh'.split())\n"
        "\n"
        "\n"
        "def non_english(text: str) -> bool:\n"
        "    tokens = re.findall(r'[a-z]+', str(text).lower())\n"
        "    hits = sum(t in HINDI for t in tokens)\n"
        "    return bool(tokens) and hits >= 1 and hits / len(tokens) >= 0.12\n"
        "\n"
        "\n"
        "train['non_english'] = train['customer_reason'].map(non_english)\n"
        "test['non_english'] = test['customer_reason'].map(non_english)\n"
        "train['n_words'] = train['customer_reason'].str.split().str.len()\n"
        "train['has_emoji'] = train['customer_reason'].str.contains(r'[^\\x00-\\x7F]', regex=True)\n"
        'print(f\'rough non-English share: train {train["non_english"].mean():.1%}, test {test["non_english"].mean():.1%}\')\n'
        "display(train.groupby('label')[['non_english', 'n_words', 'has_emoji']].mean().round(3))"
    ),
    code(
        "PATTERNS = {\n"
        "    'empty box': r'empty|khali|khaali',\n"
        "    'tags intact / unused': r'tag.{0,12}(?:intact|on)|unused|never worn|not worn|unwashed|sealed',\n"
        "    'fit too small': r'too small|tight|chhota|chota|tang|kasa|small',\n"
        "    'fit too large': r'too (?:big|large|loose)|loose|baggy|dheela|dhila|bada',\n"
        "    'wrong item': r'wrong|galat|different|another|not what i ordered|nahi mangaya|doosra|dusra',\n"
        "    'damage / defect': r'tor[en]|damag|defect|hole|stain|broken|phat|toot|daag|kharab',\n"
        "    'colour mismatch': r'colou?r|rang',\n"
        "    'not as described': r'not as (?:described|shown)|photo|picture|fake|duplicate|nakli',\n"
        "    'wedding / occasion': r'wedding|shaadi|function|party|event|occasion',\n"
        "    'refund demand only': r'refund|paisa|paise|wapas',\n"
        "}\n"
        "reasons = train['customer_reason'].fillna('').str.lower()\n"
        "flags = pd.DataFrame({name: reasons.str.contains(rx, regex=True) for name, rx in PATTERNS.items()})\n"
        "by_label = flags.groupby(train['label']).mean().T.round(3)\n"
        "by_label['abuse_rate_when_present'] = [train.loc[flags[c], 'abusive'].mean().round(3) for c in by_label.index]\n"
        "by_label"
    ),
    code(
        "# Words most over-represented in abusive text (log-odds, add-one smoothing)\n"
        "tokens = reasons.str.findall(r'[a-z]{3,}')\n"
        "abusive_counts = pd.Series([w for ws in tokens[train['abusive'] == 1] for w in set(ws)]).value_counts()\n"
        "genuine_counts = pd.Series([w for ws in tokens[train['abusive'] == 0] for w in set(ws)]).value_counts()\n"
        "vocab = pd.concat([abusive_counts, genuine_counts], axis=1, keys=['abusive', 'genuine']).fillna(0)\n"
        "vocab = vocab[(vocab.sum(axis=1) >= 40)]\n"
        "n_abusive, n_genuine = int(train['abusive'].sum()), int((1 - train['abusive']).sum())\n"
        "vocab['log_odds'] = np.log((vocab['abusive'] + 1) / n_abusive) - np.log((vocab['genuine'] + 1) / n_genuine)\n"
        "display(vocab.sort_values('log_odds', ascending=False).head(12).round(2))\n"
        "display(vocab.sort_values('log_odds').head(8).round(2))"
    ),
    code(
        "# Does a customer reuse the exact same reason text?\n"
        "key = train['customer_reason'].str.lower().str.strip()\n"
        "repeat_text = train.assign(key=key).groupby(['customer_id', 'key'])['abusive'].transform('size') > 1\n"
        "print(f'rows whose exact reason text repeats within the same customer: {repeat_text.mean():.1%}')\n"
        'print(f\'abuse rate when repeated: {train.loc[repeat_text, "abusive"].mean():.3f}   otherwise: {train.loc[~repeat_text, "abusive"].mean():.3f}\')'
    ),
    md(
        "**Reading:** text separates the three *genuine* reasons well, but abusers copy genuine wording, so wording is weak "
        "evidence of abuse by itself. Text is most useful as a *consistency* check against the order facts and the "
        "customer's history."
    ),
    md(
        "## 9. Fairness groups: base rates\nThe scorer compares how often **genuine** customers of each group get flagged."
    ),
    code(
        "groups = {\n"
        "    'new customer (<90 days)': train['customer_tenure_days'] < 90,\n"
        "    'low value order (<1000 INR)': train['order_value_inr'] < 1000,\n"
        "    'non-English reason (rough guess)': train['non_english'],\n"
        "}\n"
        "rows = []\n"
        "for name, mask in groups.items():\n"
        "    rows.append({'group': name, 'rows': int(mask.sum()), 'abuse_rate_in_group': train.loc[mask, 'abusive'].mean(),\n"
        "                 'abuse_rate_others': train.loc[~mask, 'abusive'].mean()})\n"
        "fair = pd.DataFrame(rows).set_index('group').round(3)\n"
        "fair['ratio'] = (fair['abuse_rate_in_group'] / fair['abuse_rate_others']).round(2)\n"
        "fair"
    ),
    code(
        "# Which risk signals do the fairness groups carry? (a proxy check)\n"
        "signals = pd.DataFrame({'late (>14d)': late, 'COD': cod, 'single item': single, 'high prior return rate (>0.5)': train['prior_return_rate'] > 0.5})\n"
        "pd.concat({name: signals[mask].mean() for name, mask in groups.items()}, axis=1).round(3)"
    ),
    md(
        "**Reading:** abuse is about as common inside each group as outside it, so there is no legitimate reason to flag a "
        "group more often. If a group looks riskier to a model, it will be because of *proxy signals* "
        "(e.g. no history, small baskets), which is exactly what the table above checks."
    ),
    md("## 10. Train vs test: does the future look like the past?"),
    code(
        "numeric = ['order_value_inr', 'item_price_inr', 'discount_pct', 'days_since_delivery', 'delivery_days', 'items_in_order',\n"
        "           'customer_tenure_days', 'prior_orders_count', 'prior_return_rate']\n"
        "drift = pd.DataFrame({'train_mean': train[numeric].mean(), 'test_mean': test[numeric].mean()})\n"
        "drift['change_%'] = ((drift['test_mean'] / drift['train_mean'] - 1) * 100).round(1)\n"
        "drift.round(2)"
    ),
    code(
        "def psi(a: pd.Series, b: pd.Series) -> float:\n"
        "    p, q = a.value_counts(normalize=True), b.value_counts(normalize=True)\n"
        "    idx = p.index.union(q.index)\n"
        "    p, q = p.reindex(idx, fill_value=0) + 1e-4, q.reindex(idx, fill_value=0) + 1e-4\n"
        "    return float(((p - q) * np.log(p / q)).sum())\n"
        "\n"
        "\n"
        "cats = ['category', 'payment_method', 'fulfilment_centre', 'size_ordered']\n"
        "print('population stability index (<0.1 = stable):')\n"
        "print(pd.Series({c: psi(train[c], test[c]) for c in cats}).round(4).to_string())\n"
        "display(pd.concat([train['payment_method'].value_counts(normalize=True), test['payment_method'].value_counts(normalize=True)],\n"
        "                  axis=1, keys=['train', 'test']).round(3))\n"
        "new_in_test = set(test['product_id']) - set(train['product_id'])\n"
        'print(f\'products only in test: {len(new_in_test)}  |  test rows from customers seen in train: {test["customer_id"].isin(train["customer_id"]).mean():.1%}\')'
    ),
    md(
        "## 11. Which single features rank requests best?\nUnivariate ranking power (AUC, 0.5 = useless) for numeric features, on the training data."
    ),
    code(
        "def auc(score: pd.Series, target: pd.Series) -> float:\n"
        "    data = pd.DataFrame({'s': score, 'y': target}).dropna()\n"
        "    ranks = data['s'].rank()\n"
        "    n1, n0 = data['y'].sum(), (1 - data['y']).sum()\n"
        "    return float((ranks[data['y'] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))\n"
        "\n"
        "\n"
        "ranked = pd.Series({c: auc(train[c], train['abusive']) for c in numeric}).sort_values(ascending=False)\n"
        "ranked = pd.DataFrame({'AUC': ranked.round(3), 'strength': (ranked - 0.5).abs().round(3)}).sort_values('strength', ascending=False)\n"
        "ranked['strength'].plot.barh(title='|AUC - 0.5| per numeric feature').invert_yaxis(); plt.show()\n"
        "ranked"
    ),
    md("## 12. Fit-related returns by product (the size-chart question)"),
    code(
        "fit = train[train['label'] == 'genuine_fit_issue']\n"
        "per_product = fit.groupby('product_id').size().sort_values(ascending=False)\n"
        "top = per_product.head(12)\n"
        "top.plot.bar(title='Genuine fit-related returns per product (train)', rot=60); plt.tight_layout(); plt.show()\n"
        "report = top.head(8).to_frame('fit_returns').join(products.set_index('product_id')[['title', 'category', 'sub_category', 'size_chart_id']])\n"
        "report['all_returns'] = train.groupby('product_id').size().reindex(report.index)\n"
        "report['fit_share_of_returns'] = (report['fit_returns'] / report['all_returns']).round(2)\n"
        "report"
    ),
    code(
        "small = fit['customer_reason'].str.contains(r'small|tight|chhota|chota|tang|kasa', case=False)\n"
        "large = fit['customer_reason'].str.contains(r'large|big|loose|baggy|dheela|dhila|bada', case=False)\n"
        "print(f'fit complaints: too small {small.mean():.0%}, too large {large.mean():.0%}')\n"
        "print('fit returns of ranks 1-5:', per_product.head(5).tolist(), '| rank 6:', int(per_product.iloc[5]), '| rank 7:', int(per_product.iloc[6]))\n"
        "print('share of all fit returns held by the top 5 products:', f'{per_product.head(5).sum() / per_product.sum():.1%}')\n"
        "by_sub = fit.groupby('sub_category').size().sort_values(ascending=False).head(6)\n"
        "print(by_sub.to_string())"
    ),
    md("## 13. Findings"),
    code(
        "# Findings table, computed from the data above so the numbers cannot drift from the analysis\n"
        "def rate(mask) -> float:\n"
        "    return float(train.loc[mask, 'abusive'].mean())\n"
        "\n"
        "\n"
        "findings = pd.DataFrame(\n"
        "    [\n"
        "        ('return requested > 14 days after delivery', rate(late), int(late.sum())),\n"
        "        ('return requested within 1 day of delivery', rate(train['days_since_delivery'] <= 1), int((train['days_since_delivery'] <= 1).sum())),\n"
        "        ('cash on delivery', rate(cod), int(cod.sum())),\n"
        "        ('prepaid (non-COD)', rate(~cod), int((~cod).sum())),\n"
        "        ('single-item order', rate(single), int(single.sum())),\n"
        "        ('2+ items in order', rate(~single), int((~single).sum())),\n"
        "        ('discount above 50%', rate(train['discount_pct'] > 50), int((train['discount_pct'] > 50).sum())),\n"
        "        ('Men Ethnic category', rate(train['category'] == 'Men Ethnic'), int((train['category'] == 'Men Ethnic').sum())),\n"
        "        ('prior return rate > 0.5', rate(train['prior_return_rate'] > 0.5), int((train['prior_return_rate'] > 0.5).sum())),\n"
        "        ('first-time customer (no history)', rate(first_order), int(first_order.sum())),\n"
        "        ('customer tenure <= 30 days', rate(train['customer_tenure_days'] <= 30), int((train['customer_tenure_days'] <= 30).sum())),\n"
        "        ('late + COD + single item', rate(risky), int(risky.sum())),\n"
        "    ],\n"
        "    columns=['segment', 'abuse_rate', 'rows'],\n"
        ").set_index('segment')\n"
        "findings['lift_vs_base'] = findings['abuse_rate'] / BASE\n"
        "findings.round(3)"
    ),
]


def main() -> None:
    cells = list(CELLS)
    if FINDINGS_FILE.exists():
        cells.append(md(FINDINGS_FILE.read_text()))
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(notebook, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
