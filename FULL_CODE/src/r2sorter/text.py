"""Customer-reason text handling: normalisation, evidence flags, language guess and a text classifier.

The reasons are English, Hinglish and Hindi written in English letters, with typos and emoji.
We deliberately do not translate: character n-grams cope with spelling variants better than
a translation step would, and keep every distinction the customer wrote.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .data import LABELS

_NON_LETTERS = re.compile(r"[^a-z0-9 ]+")
_REPEATS = re.compile(r"(.)\1{2,}")
_SPACES = re.compile(r"\s+")

# Each pattern is a plain regular expression over the lower-cased reason. They are evidence, not proof.
EVIDENCE_PATTERNS: dict[str, str] = {
    "empty_box": r"empty|khali|khaali|kuch nahi|kuchh nahi|kuch bhi nahi|nothing (was )?(in|inside)|andar kuch",
    "tags_intact": r"tags? (are |is )?(intact|on|attached)|unused|not used|nt used|never worn|sealed|tag laga|tag lage",
    "tags_missing": r"tags? (are |is )?(missing|removed|cut)|no tags?|tag nahi",
    "worn_signs": r"already worn|looks worn|pehna hua|pahna hua|istemal|used product|stains?|smell|perfume|washed",
    "fit_small": r"too small|chhota|chota|chhoti|tight|tang|kasa|snug|small|chhote",
    "fit_large": r"too (big|large|loose)|loose|dheela|dhila|bada|badi|baggy|large|latak|tent|oversize",
    "wrong_item": r"wrong|galat|different|alag|instead of|kuch aur|dusra|doosra|incomplete|missing item",
    "colour_diff": r"colou?r|rang|shade",
    "not_as_described": r"not as described|as shown|photo|picture|website|jaisa nahi|description|dikh",
    "damage": r"torn|phata|phat |damag|hole|chhed|stitch|silai|zip|button|broken|defect|daag|ripped|seam|thread",
    "fabric_quality": r"fabric|material|quality|kapda|cheap|patla|thin|itchy|rough|polyester|shrunk|fade",
    "occasion": (
        r"wedding|shaadi|sangeet|diwali|eid|puja|navratri|holi|party|function|festival|birthday|"
        r"reception|rakhi|mehendi|durga|ganesh|christmas|anniversary|cousin|sister|friend|brother"
    ),
    "proof_offer": r"unboxing|video|proof|pics?|photos? (hai|available|attached)",
    "refund_demand": r"refund|paise|paisa|money back|wapas|vapas|wapis|full amount",
    "replacement_ask": r"replace|exchange|badal|change",
    "pickup_ask": r"pickup|pick up|collect",
    "dissatisfied": r"not satisfied|nt satisfied|not good|pasand nahi|didn.?t like|dislike|bakwas|bekar|worst|waste",
}
_COMPILED = {name: re.compile(pattern) for name, pattern in EVIDENCE_PATTERNS.items()}

# Common Hindi words typed in English letters; used only to AUDIT fairness by language, never as a model feature.
HINDI_MARKERS = frozenset(
    [
        "hai",
        "hain",
        "h",
        "nahi",
        "nhi",
        "nahin",
        "nai",
        "kya",
        "kyu",
        "kyun",
        "bahut",
        "bohot",
        "bhot",
        "bahot",
        "bht",
        "mein",
        "mai",
        "mei",
        "ka",
        "ki",
        "ke",
        "ko",
        "se",
        "pe",
        "tha",
        "thi",
        "hua",
        "hui",
        "hue",
        "gaya",
        "gayi",
        "gya",
        "gyi",
        "aaya",
        "aayi",
        "aya",
        "ayi",
        "raha",
        "rahi",
        "rahe",
        "rha",
        "rhi",
        "kar",
        "karo",
        "kr",
        "kro",
        "diya",
        "kiya",
        "kia",
        "liya",
        "mila",
        "mili",
        "chahiye",
        "chahie",
        "chaiye",
        "wapas",
        "vapas",
        "wapis",
        "yeh",
        "ye",
        "woh",
        "wo",
        "vo",
        "mujhe",
        "muje",
        "mera",
        "meri",
        "mere",
        "maine",
        "mene",
        "hoon",
        "hu",
        "hun",
        "bhi",
        "bilkul",
        "ekdum",
        "accha",
        "achha",
        "acha",
        "achhi",
        "kharab",
        "ghatiya",
        "bekar",
        "bekaar",
        "kapda",
        "kapde",
        "silai",
        "rang",
        "naap",
        "chhota",
        "chota",
        "chhoti",
        "choti",
        "bada",
        "badi",
        "dheela",
        "tang",
        "lambai",
        "jaisa",
        "jesa",
        "sahi",
        "galat",
        "saman",
        "samaan",
        "dabba",
        "khali",
        "paise",
        "pese",
        "dena",
        "bhej",
        "bhejo",
        "mangaya",
        "theek",
        "thik",
        "zyada",
        "jyada",
        "kuch",
        "kuchh",
        "abhi",
        "aur",
        "toh",
    ]
)


def normalise_reason(text: str) -> str:
    """Lower-case, drop punctuation/emoji, squeeze repeated letters and spaces."""
    lowered = _NON_LETTERS.sub(" ", str(text).lower())
    return _SPACES.sub(" ", _REPEATS.sub(r"\1\1", lowered)).strip()


def looks_non_english(text: str) -> bool:
    """Rough language guess: does the reason contain a noticeable share of Hindi words?"""
    tokens = re.findall(r"[a-z]+", str(text).lower())
    hits = sum(token in HINDI_MARKERS for token in tokens)
    return bool(tokens) and hits >= 1 and hits / len(tokens) >= 0.12


def evidence_features(reasons: pd.Series) -> pd.DataFrame:
    """Count regex evidence per reason plus simple shape statistics."""
    lowered = reasons.fillna("").astype(str).str.lower()
    out = {f"ev_{name}": lowered.str.count(pattern.pattern).clip(upper=3) for name, pattern in _COMPILED.items()}
    raw = reasons.fillna("").astype(str)
    out["txt_len_chars"] = raw.str.len()
    out["txt_n_words"] = lowered.str.split().str.len().fillna(0)
    out["txt_upper_share"] = raw.map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    out["txt_n_exclaim"] = raw.str.count("!")
    out["txt_has_emoji"] = raw.map(lambda s: int(any(ord(c) > 0x2000 for c in s)))
    return pd.DataFrame(out, index=reasons.index)


class TextReasonModel:
    """TF-IDF (word + character n-grams) feeding a multinomial logistic regression over the four labels."""

    def __init__(self, params: dict[str, Any], seed: int = 42) -> None:
        self._params = params
        self._seed = seed
        self._word = TfidfVectorizer(
            ngram_range=tuple(params["word_ngram"]),
            min_df=params["word_min_df"],
            sublinear_tf=True,
            preprocessor=normalise_reason,
        )
        self._char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(params["char_ngram"]),
            min_df=params["char_min_df"],
            max_features=params["char_max_features"],
            sublinear_tf=True,
            preprocessor=normalise_reason,
        )
        self._clf = LogisticRegression(C=params["logreg_c"], max_iter=300, random_state=seed)
        self._fitted = False

    def _matrix(self, texts: pd.Series, *, fit: bool) -> sparse.csr_matrix:
        values = texts.fillna("").astype(str)
        if fit:
            return sparse.hstack([self._word.fit_transform(values), self._char.fit_transform(values)]).tocsr()
        return sparse.hstack([self._word.transform(values), self._char.transform(values)]).tocsr()

    def fit(self, texts: pd.Series, labels: pd.Series) -> TextReasonModel:
        """Fit vectorisers and classifier on training reasons only."""
        self._clf.fit(self._matrix(texts, fit=True), labels.map(LABELS.index))
        self._fitted = True
        return self

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Probabilities in the order of data.LABELS."""
        if not self._fitted:
            raise RuntimeError("TextReasonModel must be fitted before predicting")
        probs = self._clf.predict_proba(self._matrix(texts, fit=False))
        full = np.zeros((len(texts), len(LABELS)))
        full[:, self._clf.classes_] = probs
        return full


def text_probability_frame(probs: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Name the four text-model probabilities so they can be used as model features."""
    return pd.DataFrame(probs, index=index, columns=[f"txt_p_{label}" for label in LABELS])


def out_of_fold_text_probs(
    texts: pd.Series,
    labels: pd.Series,
    groups: pd.Series,
    params: dict[str, Any],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Text probabilities for training rows from models that never saw the row's customer."""
    oof = np.zeros((len(texts), len(LABELS)))
    splitter = GroupKFold(n_splits=n_folds)
    for train_idx, valid_idx in splitter.split(texts, labels, groups):
        model = TextReasonModel(params, seed).fit(texts.iloc[train_idx], labels.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(texts.iloc[valid_idx])
    return oof
