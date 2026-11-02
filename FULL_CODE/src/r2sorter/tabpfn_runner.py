"""Optional TabPFN member (needs the `tabpfn` package, PyTorch and downloaded weights; run on the user's own machine).

The API key is read from the git-ignored file tabpfn_key.py, never from a notebook.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import predict_full
from .data import LABELS


def numeric_codes(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """TabPFN reads numbers: categorical columns become integer codes (unseen -> NaN); returns their positions too."""
    out, positions = frame.copy(), []
    for position, column in enumerate(out.columns):
        if isinstance(out[column].dtype, pd.CategoricalDtype):
            codes = out[column].cat.codes.astype(float)
            out[column] = codes.where(codes >= 0)
            positions.append(position)
    return out.astype(float), positions


def stratified_subsample(y_index: np.ndarray, max_rows: int, seed: int = 0) -> np.ndarray:
    """Sorted row positions keeping every class in its original proportion, at most about `max_rows` rows."""
    rng = np.random.default_rng(seed)
    keep = []
    fraction = min(1.0, max_rows / len(y_index))
    for cls in range(len(LABELS)):
        members = np.flatnonzero(y_index == cls)
        keep.append(rng.choice(members, size=min(len(members), max(1, round(len(members) * fraction))), replace=False))
    return np.sort(np.concatenate(keep))


def package_installed() -> bool:
    return importlib.util.find_spec("tabpfn") is not None


def load_key(search_dirs: tuple[Path, ...] = (Path(".."), Path("."))) -> str:
    for folder in search_dirs:
        if (folder / "tabpfn_key.py").exists():
            sys.path.insert(0, str(folder.resolve()))
            break
    try:
        from tabpfn_key import TABPFN_API_KEY  # type: ignore[import-not-found]
    except ImportError:
        return "no tabpfn_key.py found; relying on an existing login or the TABPFN_TOKEN variable"
    os.environ["TABPFN_TOKEN"] = TABPFN_API_KEY
    return "API key loaded from tabpfn_key.py"


def fit_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_others: list[pd.DataFrame],
    *,
    max_train_rows: int = 10_000,
    n_estimators: int = 4,
    batch: int = 2_000,
    time_budget_min: float = 25.0,
    seed: int = 0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Fit on a stratified subsample and predict every frame in `x_others`; raises TimeoutError past the budget."""
    from tabpfn import TabPFNClassifier

    take = stratified_subsample(y_train, max_train_rows, seed)
    xtr, positions = numeric_codes(x_train.iloc[take])
    try:
        clf = TabPFNClassifier(
            n_estimators=n_estimators, device="cpu", random_state=seed, categorical_features_indices=positions
        )
    except TypeError:
        clf = TabPFNClassifier(n_estimators=n_estimators, device="cpu", random_state=seed)
    started = time.perf_counter()
    clf.fit(xtr, y_train[take])
    fit_seconds = time.perf_counter() - started
    everything = pd.concat([numeric_codes(frame)[0] for frame in x_others], ignore_index=True)
    outputs, infer_seconds, info = [], 0.0, {}
    for start in range(0, len(everything), batch):
        began = time.perf_counter()
        outputs.append(predict_full(clf, everything.iloc[start : start + batch]))
        infer_seconds += time.perf_counter() - began
        if start == 0:
            projected = infer_seconds / min(batch, len(everything)) * len(everything) / 60
            info["projected_inference_minutes"] = round(projected, 1)
            if projected > time_budget_min:
                raise TimeoutError(f"projected {projected:.0f} min exceeds the {time_budget_min:.0f} min budget")
    probs = np.vstack(outputs)
    pieces, offset = [], 0
    for frame in x_others:
        pieces.append(probs[offset : offset + len(frame)])
        offset += len(frame)
    info.update(
        train_rows_used=int(len(take)),
        train_seconds=round(fit_seconds, 2),
        inference_ms_per_1k_rows=round(infer_seconds * 1000 / (len(everything) / 1000), 1),
    )
    return pieces, info
