import pandas as pd

from r2sorter.explain import justify
from r2sorter.text import evidence_features, looks_non_english, normalise_reason


def test_normalise_handles_case_and_noise():
    assert normalise_reason("  BOX Khali Tha!!  ") == normalise_reason("box khali tha")


def test_language_heuristic():
    assert looks_non_english("kapda bahut kharab hai")
    assert not looks_non_english("the stitching came off")
    assert not looks_non_english("")


def test_evidence_features_detect_empty_box_and_ignore_blank():
    frame = evidence_features(pd.Series(["box was empty", "", "tags intact never worn"]))
    assert frame.loc[0, "ev_empty_box"] >= 1
    assert frame.loc[1].sum() == 0
    assert frame.loc[2, "ev_tags_intact"] >= 1


def _row(**overrides) -> pd.Series:
    base = {
        "days_since_delivery": 3,
        "cod_flag": 0,
        "prior_return_rate": float("nan"),
        "hist_returns_30d": 0,
        "hist_same_text": 0,
        "hist_max_text_sim": 0.0,
        "lab_n_abusive": 0,
        "items_in_order": 2,
        "item_price_inr": 900,
        "txt_p_abusive": 0.0,
        "txt_p_genuine_fit_issue": 0.9,
        "txt_p_genuine_quality_issue": 0.05,
        "txt_p_genuine_wrong_item": 0.05,
    }
    return pd.Series({**base, **overrides})


def test_flag_justification_cites_real_values_not_genuine_text():
    row = _row(days_since_delivery=19, lab_n_abusive=4, cod_flag=1)
    contribution = pd.Series(
        {
            "days_since_delivery": 0.9,
            "lab_n_abusive": 1.4,
            "cod_flag": 0.4,
            "txt_p_genuine_fit_issue": 0.8,
        }
    )
    text = justify(row, contribution, "inspect", 0.9)
    assert "19 days" in text and "4 earlier" in text and "cash-on-delivery" in text
    assert "genuine" not in text


def test_approve_justification_mentions_genuine_text():
    contribution = pd.Series({"txt_p_genuine_fit_issue": -1.2})
    text = justify(_row(), contribution, "auto_approve", 0.01)
    assert text.startswith("Approve") and "fit complaint" in text


def test_justification_never_empty_without_contributions():
    assert len(justify(_row(), pd.Series(dtype=float), "escalate", 0.99)) > 20
