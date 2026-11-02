"""Demo: browse the returns decisions and their justifications. Run: streamlit run app/streamlit_app.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

SUBMISSION = Path(__file__).resolve().parents[2] / "submission"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(SUBMISSION / "R2_predictions.csv")


def main() -> None:
    st.set_page_config(page_title="Returns Sorter", layout="wide")
    st.title("Returns: fraud vs genuine grievance sorter")
    predictions = load_predictions()
    counts = predictions["decision"].value_counts()
    columns = st.columns(len(counts))
    for column, (decision, count) in zip(columns, counts.items(), strict=True):
        column.metric(decision, f"{count:,}")
    chosen = st.multiselect("Decision", sorted(predictions["decision"].unique()), default=["inspect", "escalate"])
    low, high = st.slider("P(abusive) range", 0.0, 1.0, (0.0, 1.0))
    view = predictions[predictions["decision"].isin(chosen) & predictions["confidence"].between(low, high)]
    st.caption(f"{len(view):,} requests")
    st.dataframe(view.sort_values("confidence", ascending=False), use_container_width=True, hide_index=True)
    top5 = pd.read_csv(SUBMISSION / "R2_size_chart_top5.csv")
    st.subheader("Products with the most fit-related returns")
    st.table(top5)


main()
