"""Create the two analysis notebooks (reusable code lives in src/r2sorter, not in the notebooks)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks"

RESULTS = [
    new_markdown_cell(
        "# 02 - Reviewing the submission\nChecks the committed submission files and shows what the router decided."
    ),
    new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        "sub = Path('..') / '..' / 'submission'\n"
        "pred = pd.read_csv(sub / 'R2_predictions.csv')\ntop5 = pd.read_csv(sub / 'R2_size_chart_top5.csv')\n"
        "pred['decision'].value_counts()"
    ),
    new_code_cell("assert pred['return_id'].is_unique and pred['confidence'].between(0, 1).all()\nlen(pred)"),
    new_markdown_cell("## Confidence (P(abusive)) by decision"),
    new_code_cell("pred.groupby('decision')['confidence'].describe().round(3)"),
    new_code_cell(
        "pred['confidence'].plot.hist(bins=40, log=True, title='P(abusive) across test requests'); plt.show()"
    ),
    new_markdown_cell("## Example justifications"),
    new_code_cell(
        "for decision in ['escalate', 'inspect', 'auto_approve']:\n"
        "    print(decision)\n"
        "    for text in pred[pred['decision'] == decision]['justification'].head(3):\n"
        "        print('  -', text)"
    ),
    new_markdown_cell("## Size-chart top 5"),
    new_code_cell("top5"),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in (("02_review_submission.ipynb", RESULTS),):
        notebook = new_notebook(cells=cells)
        notebook.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        nbformat.write(notebook, OUT / name)


if __name__ == "__main__":
    main()
