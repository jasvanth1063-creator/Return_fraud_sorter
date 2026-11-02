import numpy as np

from r2sorter.features import add_label_history


def test_label_history_never_sees_the_request_itself_or_the_future(small_frame):
    out = add_label_history(small_frame, unlabelled_from=pd_ts("2026-01-06"), max_lag_days=5, seed=0)
    assert (out["lab_n_prev"] <= out["lab_n_abusive"] + 1).all()
    first_a = out[out["customer_id"] == "A"].iloc[0]
    assert first_a["lab_n_prev"] == 0


def test_unlabelled_rows_only_see_outcomes_before_the_cutoff(small_frame):
    out = add_label_history(small_frame, unlabelled_from=pd_ts("2026-01-06"), max_lag_days=5, seed=0)
    late_a = out[(out["customer_id"] == "A") & (~out["is_train"])]
    assert (late_a["lab_n_abusive"] == 2).all()
    assert (late_a["lab_n_prev"] == 2).all()


def test_label_history_is_reproducible_for_a_seed(small_frame):
    first = add_label_history(small_frame, pd_ts("2026-01-06"), 5, seed=3)
    second = add_label_history(small_frame, pd_ts("2026-01-06"), 5, seed=3)
    np.testing.assert_array_equal(first["lab_n_prev"], second["lab_n_prev"])


def test_new_customer_has_no_history(small_frame):
    out = add_label_history(small_frame, pd_ts("2026-01-06"), 5, seed=0)
    first_b = out[out["customer_id"] == "B"].iloc[0]
    assert first_b["lab_n_prev"] == 0 and np.isnan(first_b["lab_days_since_abusive"])


def pd_ts(text):
    import pandas as pd

    return pd.Timestamp(text)
