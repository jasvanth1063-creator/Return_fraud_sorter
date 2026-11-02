"""Phase 4: text candidates, customer-grouped out-of-fold scores, metrics and integrity checks.

Everything here is fitted on the rows it is given; nothing reads hidden-test labels.
Logistic-regression outputs are probabilities. Linear-SVM outputs are DECISION SCORES, not probabilities,
and are named ``score`` accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from .data import LABELS
from .text import looks_non_english, normalise_reason

REPRESENTATIONS = ("word", "char", "combined")
CLASSIFIERS = ("logreg", "linear_svm")
SIMPLICITY_ORDER = [f"{r}_{c}" for c in CLASSIFIERS for r in REPRESENTATIONS]  # tie-break: simplest first
ABUSIVE = LABELS.index("abusive")


def text_audit(reasons: pd.Series, labels: pd.Series | None = None) -> dict[str, Any]:
    """Counts a reviewer needs before trusting the text: missing, empty, lengths, language mix, emoji, repeats."""
    raw = reasons.astype("string")
    stripped = raw.str.strip()
    language = reasons.fillna("").map(lambda s: "hindi_marked" if looks_non_english(s) else "english_like")
    audit: dict[str, Any] = {
        "rows": len(reasons),
        "missing": int(raw.isna().sum()),
        "empty_after_strip": int((stripped == "").sum()),
        "length_chars": raw.str.len().describe().round(1).to_dict(),
        "words_median": float(raw.str.split().str.len().median()),
        "emoji_share": round(float(raw.map(lambda s: any(ord(c) > 0x2000 for c in str(s))).mean()), 4),
        "non_ascii_share": round(float(raw.map(lambda s: not str(s).isascii()).mean()), 4),
        "duplicate_text_share": round(float(reasons.duplicated().mean()), 4),
        "language_groups": language.value_counts().to_dict(),
    }
    if labels is not None:
        audit["label_mix_by_language"] = pd.crosstab(language, labels, normalize="index").round(3).to_dict("index")
    return audit


def language_group(reasons: pd.Series) -> pd.Series:
    """Heuristic group used for subgroup reporting (wordlist based; NOT a trained language identifier)."""
    return reasons.fillna("").map(lambda s: "hindi_marked" if looks_non_english(s) else "english_like")


def language_group3(reasons: pd.Series, dominant_share: float = 0.4) -> pd.Series:
    """English-like / Hinglish-like (mixed) / Hindi-dominant (transliterated), from the share of common Hindi words.

    A word-list heuristic, not a language identifier: the marker list is incomplete, so the share is a lower bound.
    """
    import re

    from .text import HINDI_MARKERS, looks_non_english

    def group(text: object) -> str:
        if not looks_non_english(str(text)):
            return "english_like"
        tokens = re.findall(r"[a-z]+", str(text).lower())
        share = sum(token in HINDI_MARKERS for token in tokens) / max(len(tokens), 1)
        return "hindi_dominant" if share >= dominant_share else "hinglish_like"

    return reasons.fillna("").map(group)


@dataclass(frozen=True)
class TextCandidate:
    """A representation (word, char, combined TF-IDF) plus a linear classifier."""

    representation: str
    classifier: str
    params: dict[str, Any]
    seed: int = 42

    @property
    def name(self) -> str:
        return f"{self.representation}_{self.classifier}"

    @property
    def output_kind(self) -> str:
        return "probability" if self.classifier == "logreg" else "score"

    def _vectorizers(self) -> list[TfidfVectorizer]:
        p = self.params
        word = TfidfVectorizer(
            ngram_range=tuple(p["word_ngram"]),
            min_df=p["word_min_df"],
            sublinear_tf=True,
            preprocessor=normalise_reason,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(p["char_ngram"]),
            min_df=p["char_min_df"],
            max_features=p["char_max_features"],
            sublinear_tf=True,
            preprocessor=normalise_reason,
        )
        return {"word": [word], "char": [char], "combined": [word, char]}[self.representation]

    def fit(self, texts: pd.Series, labels: pd.Series) -> FittedText:
        values = texts.fillna("").astype(str)
        vectorizers = self._vectorizers()
        matrix = sparse.hstack([v.fit_transform(values) for v in vectorizers]).tocsr()
        y = labels.map(LABELS.index).to_numpy()
        if self.classifier == "logreg":
            clf: Any = LogisticRegression(C=self.params["logreg_c"], max_iter=300, random_state=self.seed)
        else:
            clf = LinearSVC(C=self.params.get("svm_c", 0.5), random_state=self.seed)
        clf.fit(matrix, y)
        return FittedText(self, vectorizers, clf)


class FittedText:
    """A fitted candidate; ``scores`` returns one column per label in ``data.LABELS`` order."""

    def __init__(self, candidate: TextCandidate, vectorizers: list[TfidfVectorizer], clf: Any) -> None:
        self.candidate, self._vectorizers, self._clf = candidate, vectorizers, clf

    def scores(self, texts: pd.Series) -> np.ndarray:
        values = texts.fillna("").astype(str)
        matrix = sparse.hstack([v.transform(values) for v in self._vectorizers]).tocsr()
        raw = (
            self._clf.predict_proba(matrix)
            if self.candidate.classifier == "logreg"
            else self._clf.decision_function(matrix)
        )
        full = np.full((len(values), len(LABELS)), 0.0 if self.candidate.classifier == "logreg" else -10.0)
        full[:, self._clf.classes_] = raw
        return full

    @property
    def vocabulary_size(self) -> int:
        return int(sum(len(v.vocabulary_) for v in self._vectorizers))


def grouped_out_of_fold(
    candidate: TextCandidate, texts: pd.Series, labels: pd.Series, groups: pd.Series, n_folds: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """(scores, fold id per row, seconds). Customers never straddle fitting and validation parts."""
    scores = np.full((len(texts), len(LABELS)), np.nan)
    fold_id = np.full(len(texts), -1)
    started = time.perf_counter()
    for fold, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=n_folds).split(texts, labels, groups)):
        assert not set(groups.iloc[fit_idx]) & set(groups.iloc[val_idx]), "customer in both parts of a fold"
        scores[val_idx] = candidate.fit(texts.iloc[fit_idx], labels.iloc[fit_idx]).scores(texts.iloc[val_idx])
        fold_id[val_idx] = fold
    return scores, fold_id, time.perf_counter() - started


def check_out_of_fold(scores: np.ndarray, fold_id: np.ndarray, ids: pd.Series) -> dict[str, int]:
    """Integrity checks: one prediction per row, no gaps, no duplicate ids, every row in exactly one fold."""
    report = {
        "rows": len(ids),
        "rows_without_prediction": int(np.isnan(scores).any(axis=1).sum()),
        "duplicate_ids": int(ids.duplicated().sum()),
        "rows_without_fold": int((fold_id < 0).sum()),
    }
    if any(report[k] for k in ("rows_without_prediction", "duplicate_ids", "rows_without_fold")):
        raise AssertionError(f"out-of-fold integrity failed: {report}")
    return report


def classification_report(scores: np.ndarray, labels: pd.Series) -> dict[str, Any]:
    """Confusion matrix, per-class precision/recall/F1, macro-F1, accuracy and abusive-class average precision."""
    y = labels.map(LABELS.index).to_numpy()
    pred = scores.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(len(LABELS)), zero_division=0
    )
    return {
        "confusion": pd.DataFrame(
            confusion_matrix(y, pred, labels=range(len(LABELS))),
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        ),
        "per_class": pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=list(LABELS)
        ).round(4),
        "macro_f1": float(f1.mean()),
        "accuracy": float((pred == y).mean()),
        "abusive_ap": float(average_precision_score(y == ABUSIVE, scores[:, ABUSIVE])),
        "abusive_base_rate": float((y == ABUSIVE).mean()),
    }


def select_by_cost(table: pd.DataFrame, floor: float = 0.005) -> str:
    """Simplest candidate whose downstream cost saving is within max(seed sd, floor) of the best.

    ``table`` needs ``cost_saving``, ``cost_saving_sd``, ``abusive_AP`` and ``abusive_base_rate`` columns. A candidate
    whose abusive average precision is not above the base rate is never eligible: it carries no abuse signal. Macro-F1
    and accuracy
    are deliberately NOT used: they are dominated by the three genuine classes.
    """
    usable = table[table["abusive_AP"] > table["abusive_base_rate"]]
    if usable.empty:
        raise ValueError("no candidate beats the abusive base rate")
    best = usable["cost_saving"].idxmax()
    tolerance = max(float(usable.loc[best, "cost_saving_sd"]), floor)
    eligible = usable[usable["cost_saving"] >= usable.loc[best, "cost_saving"] - tolerance]
    return next(name for name in SIMPLICITY_ORDER if name in eligible.index)


def recall_at_fpr(y_abusive: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Share of abusive requests caught when at most ``fpr`` of genuine requests are flagged."""
    threshold = threshold_at_fpr(y_abusive, score, fpr)
    return float((score[y_abusive == 1] >= threshold).mean())


def threshold_at_fpr(y_abusive: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Score above which only ``fpr`` of the genuine rows lie (chosen on the rows it is given)."""
    return float(np.quantile(score[y_abusive == 0], 1 - fpr))


def binary_abuse_scores(
    representation: str,
    class_weight: str | None,
    params: dict[str, Any],
    seed: int,
    fit_texts: pd.Series,
    fit_labels: pd.Series,
    texts: pd.Series,
) -> np.ndarray:
    """Binary abusive-vs-genuine logistic regression on TF-IDF; returns P(abusive) for ``texts``."""
    vectorizers = TextCandidate(representation, "logreg", params, seed)._vectorizers()
    fit_values = fit_texts.fillna("").astype(str)
    matrix = sparse.hstack([v.fit_transform(fit_values) for v in vectorizers]).tocsr()
    clf = LogisticRegression(C=params["logreg_c"], max_iter=300, class_weight=class_weight, random_state=seed)
    clf.fit(matrix, (fit_labels == "abusive").astype(int).to_numpy())
    values = texts.fillna("").astype(str)
    return clf.predict_proba(sparse.hstack([v.transform(values) for v in vectorizers]).tocsr())[:, 1]


def rolling_origin_out_of_fold(
    candidate: TextCandidate,
    texts: pd.Series,
    labels: pd.Series,
    groups: pd.Series,
    dates: pd.Series,
    initial_days: int = 60,
    n_folds: int = 4,
) -> tuple[np.ndarray, np.ndarray, pd.Series, float]:
    """Time-aware out-of-fold scores: each later month is scored by a model fitted only on EARLIER requests.

    Rows in the first ``initial_days`` have no earlier data; they get customer-grouped OOF *inside that initial
    window only* and are marked ``grouped_initial`` so a downstream user can drop them.
    Returns (scores, kind, fit_end, seconds), where
    ``fit_end`` is the latest request date the scoring model saw (NaT for initial rows).
    """
    started = time.perf_counter()
    dates = pd.to_datetime(dates).reset_index(drop=True)
    texts, labels, groups = (x.reset_index(drop=True) for x in (texts, labels, groups))
    scores = np.full((len(texts), len(LABELS)), np.nan)
    kind = np.full(len(texts), "", dtype=object)
    fit_end = pd.Series(pd.NaT, index=range(len(texts)), dtype="datetime64[ns]")
    initial = (dates < dates.min() + pd.Timedelta(days=initial_days)).to_numpy()
    first_idx = np.flatnonzero(initial)
    if len(first_idx) < n_folds * 2:
        raise ValueError("initial window too small for grouped folds")
    for fit_part, val_part in GroupKFold(n_splits=n_folds).split(first_idx, groups=groups.iloc[first_idx]):
        fit_i, val_i = first_idx[fit_part], first_idx[val_part]
        scores[val_i] = candidate.fit(texts.iloc[fit_i], labels.iloc[fit_i]).scores(texts.iloc[val_i])
    kind[initial] = "grouped_initial"
    months = dates.dt.to_period("M")
    for period in sorted(months[~initial].unique()):
        block = np.flatnonzero((months == period).to_numpy() & ~initial)
        block_start = dates.iloc[block].min()
        fit_idx = np.flatnonzero((dates < block_start).to_numpy())
        scores[block] = candidate.fit(texts.iloc[fit_idx], labels.iloc[fit_idx]).scores(texts.iloc[block])
        kind[block] = "rolling"
        fit_end.iloc[block] = dates.iloc[fit_idx].max()
    return scores, kind, fit_end, time.perf_counter() - started


def check_rolling(kind: np.ndarray, fit_end: pd.Series, dates: pd.Series) -> dict[str, int]:
    """Every rolling row must have been scored by a model that saw only strictly earlier requests."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    rolling = np.asarray(kind) == "rolling"
    report = {
        "rows": len(kind),
        "rolling_rows": int(rolling.sum()),
        "grouped_initial_rows": int((np.asarray(kind) == "grouped_initial").sum()),
        "rows_without_kind": int((np.asarray(kind) == "").sum()),
        "rolling_rows_scored_with_future_data": int(
            (fit_end[rolling].reset_index(drop=True) >= dates[rolling].reset_index(drop=True)).sum()
        ),
    }
    if report["rows_without_kind"] or report["rolling_rows_scored_with_future_data"]:
        raise AssertionError(f"rolling-origin check failed: {report}")
    return report


def score_frame(scores: np.ndarray, index: pd.Index, kind: str) -> pd.DataFrame:
    """Name text features: probabilities -> txt_p_*, SVM decision scores -> txt_s_* (never called probabilities)."""
    prefix = "txt_p_" if kind == "probability" else "txt_s_"
    return pd.DataFrame(scores, index=index, columns=[f"{prefix}{label}" for label in LABELS])


def feature_dictionary(kind: str) -> pd.DataFrame:
    """Plain-language dictionary of the text-derived outputs handed to later phases."""
    prefix = "txt_p_" if kind == "probability" else "txt_s_"
    what = "out-of-fold probability" if kind == "probability" else "out-of-fold SVM decision score (not a probability)"
    rows = [
        (
            f"{prefix}{label}",
            f"{what} that the customer_reason text matches '{label}'",
            "customer-grouped OOF; text only",
        )
        for label in LABELS
    ]
    rows += [
        (
            "ev_*",
            "capped count of rule phrases (fit, quality, wrong item, damage, negation...) - supporting evidence only",
            "regex",
        ),
        (
            "txt_len_chars / txt_n_words / txt_upper_share / txt_n_exclaim / txt_has_emoji",
            "shape of the text",
            "row's own text",
        ),
        (
            "hist_same_text / hist_max_text_sim / hist_n_similar_text",
            "text reuse versus the same customer's EARLIER requests",
            "earlier rows only",
        ),
    ]
    return pd.DataFrame(rows, columns=["feature", "meaning", "leakage_safety"]).set_index("feature")
