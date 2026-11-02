import numpy as np
import pandas as pd
import pytest

from r2sorter.data import LABELS
from r2sorter.nlp import (
    TextCandidate,
    binary_abuse_scores,
    check_out_of_fold,
    check_rolling,
    classification_report,
    feature_dictionary,
    grouped_out_of_fold,
    language_group,
    recall_at_fpr,
    rolling_origin_out_of_fold,
    score_frame,
    select_by_cost,
    text_audit,
    threshold_at_fpr,
)
from r2sorter.text import normalise_reason

PARAMS = {
    "word_ngram": [1, 2],
    "word_min_df": 1,
    "char_ngram": [2, 4],
    "char_min_df": 1,
    "char_max_features": 5000,
    "logreg_c": 4.0,
}
TEXTS = {
    "abusive": ["box was empty refund now", "item never reached me refund", "empty box give money back"],
    "genuine_quality_issue": ["fabric torn at seam", "colour faded after wash", "stitching came apart"],
    "genuine_fit_issue": ["too small for me", "size too large", "does not fit waist tight"],
    "genuine_wrong_item": ["received wrong item", "different product delivered", "wrong colour sent"],
}


def _data(repeat: int = 8):
    rows = [(t, label) for label, ts in TEXTS.items() for t in ts] * repeat
    frame = pd.DataFrame(rows, columns=["text", "label"])
    frame["customer"] = [f"C{i // 3}" for i in range(len(frame))]
    return frame


def test_normaliser_keeps_negation_numbers_and_squeezes_repeats():
    assert normalise_reason("NOT fitting!! Size 42 😡") == "not fitting size 42"
    assert normalise_reason("too smallllll") == "too small"
    assert "not" in normalise_reason("Not working").split()


def test_audit_counts_missing_empty_and_language():
    reasons = pd.Series(["Too small 😡", None, "  ", "paise wapas karo nahi aaya"])
    audit = text_audit(reasons)
    assert audit["missing"] == 1 and audit["empty_after_strip"] == 1
    assert language_group(reasons).tolist()[3] == "hindi_marked"


@pytest.mark.parametrize("name", ["word", "char", "combined"])
@pytest.mark.parametrize("classifier", ["logreg", "linear_svm"])
def test_candidates_learn_and_label_their_outputs(name, classifier):
    data = _data()
    candidate = TextCandidate(name, classifier, PARAMS)
    fitted = candidate.fit(data["text"], data["label"])
    scores = fitted.scores(data["text"])
    assert scores.shape == (len(data), len(LABELS))
    assert (scores.argmax(axis=1) == data["label"].map(LABELS.index)).mean() > 0.9
    assert candidate.output_kind == ("probability" if classifier == "logreg" else "score")
    if classifier == "logreg":
        assert np.allclose(scores.sum(axis=1), 1.0)
    prefix = score_frame(scores, data.index, candidate.output_kind).columns[0][:6]
    assert prefix == ("txt_p_" if classifier == "logreg" else "txt_s_")


def test_unseen_words_do_not_break_scoring():
    data = _data()
    fitted = TextCandidate("combined", "logreg", PARAMS).fit(data["text"], data["label"])
    assert np.isfinite(fitted.scores(pd.Series(["zzzz qqqq", ""]))).all()


def test_grouped_oof_has_one_prediction_per_row_and_no_shared_customers():
    data = _data()
    scores, folds, _ = grouped_out_of_fold(
        TextCandidate("word", "logreg", PARAMS), data["text"], data["label"], data["customer"], 4
    )
    ids = pd.Series(range(len(data)))
    assert check_out_of_fold(scores, folds, ids)["rows_without_prediction"] == 0
    for fold in range(4):
        assert not set(data["customer"][folds == fold]) & set(data["customer"][folds != fold])


def test_integrity_check_fails_loudly():
    with pytest.raises(AssertionError):
        check_out_of_fold(np.array([[np.nan] * 4]), np.array([0]), pd.Series([1]))


def test_classification_report_perfect_scores():
    labels = pd.Series(["abusive", "genuine_fit_issue", "genuine_quality_issue", "genuine_wrong_item"])
    perfect = np.eye(4)[[0, 2, 1, 3]]  # data.LABELS order
    report = classification_report(perfect, labels)
    assert report["macro_f1"] == 1.0 and report["accuracy"] == 1.0


def test_selection_ignores_macro_f1_and_rejects_signal_free_candidates():
    table = pd.DataFrame(
        {
            "cost_saving": [0.30, 0.302, 0.40],
            "cost_saving_sd": [0.001, 0.001, 0.001],
            "abusive_AP": [0.12, 0.12, 0.05],
            "abusive_base_rate": [0.077] * 3,
        },
        index=["word_logreg", "char_logreg", "combined_logreg"],
    )
    # combined has the best saving but no signal above the base rate -> excluded; word is simplest within tolerance
    assert select_by_cost(table) == "word_logreg"
    with pytest.raises(ValueError):
        select_by_cost(table.assign(abusive_AP=0.01))


def test_recall_at_fpr_and_threshold():
    y = np.array([0] * 90 + [1] * 10)
    score = np.concatenate([np.linspace(0, 0.8, 90), np.linspace(0.5, 1.0, 10)])
    threshold = threshold_at_fpr(y, score, 0.10)
    assert (score[y == 0] >= threshold).mean() <= 0.11
    assert recall_at_fpr(y, score, 0.10) >= 0.6


def _dated(months=6, per_month=60):
    rows = []
    for m in range(months):
        for i in range(per_month):
            label = list(TEXTS)[i % 4]
            rows.append(
                (
                    TEXTS[label][i % 3],
                    label,
                    f"C{m}-{i // 2}",
                    pd.Timestamp("2026-01-01") + pd.Timedelta(days=30 * m + i % 28),
                )
            )
    return pd.DataFrame(rows, columns=["text", "label", "customer", "date"])


def test_rolling_origin_never_scores_with_future_data():
    data = _dated()
    scores, kind, fit_end, _ = rolling_origin_out_of_fold(
        TextCandidate("word", "logreg", PARAMS),
        data["text"],
        data["label"],
        data["customer"],
        data["date"],
        initial_days=60,
    )
    report = check_rolling(kind, fit_end, data["date"])
    assert report["rolling_rows"] > 0 and report["grouped_initial_rows"] > 0
    assert not np.isnan(scores).any()
    assert (fit_end[kind == "rolling"].to_numpy() < data["date"][kind == "rolling"].to_numpy()).all()


def test_rolling_check_catches_future_leak():
    dates = pd.Series(pd.to_datetime(["2026-03-01", "2026-03-02"]))
    with pytest.raises(AssertionError):
        check_rolling(
            np.array(["rolling", "rolling"], dtype=object),
            pd.Series(pd.to_datetime(["2026-03-01", "2026-02-01"])),
            dates,
        )


def test_binary_scores_are_probabilities_and_respect_weighting():
    data = _data()
    plain = binary_abuse_scores("word", None, PARAMS, 0, data["text"], data["label"], data["text"])
    balanced = binary_abuse_scores("word", "balanced", PARAMS, 0, data["text"], data["label"], data["text"])
    assert ((plain >= 0) & (plain <= 1)).all()
    assert balanced[data["label"] == "abusive"].mean() >= plain[data["label"] == "abusive"].mean() - 1e-9


def test_feature_dictionary_names_match_kind():
    assert feature_dictionary("score").index[0].startswith("txt_s_")
    assert feature_dictionary("probability").index[0].startswith("txt_p_")
