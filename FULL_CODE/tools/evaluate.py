#!/usr/bin/env python3
"""Official scorer for R2 — Returns Fraud vs. Genuine Grievance Sorter (TalentAP Mega Hackathon 2026).

Usage
-----
    python evaluate.py --submission submission/R2_predictions.csv --truth data/return_requests_train.csv [--out score.json]

Optional:
    --size_chart <csv>        your top-5 size-chart file (default: R2_size_chart_top5.csv next to --submission, if present)
    --size_chart_truth <csv>  answer file for the size-chart part (organisers only; default: size_chart_truth.csv next to --truth)

Only Python standard library + numpy + pandas. Same file is used by the organisers on the hidden test truth.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================================
# Published constants (also listed in README.md)
# ============================================================================================
LABELS = ["abusive", "genuine_quality_issue", "genuine_fit_issue", "genuine_wrong_item"]
DECISIONS = ["auto_approve", "inspect", "escalate"]

# Cost matrix: rows = truth, columns = your decision
COST = {
    "genuine": {"auto_approve": 0.0, "inspect": 0.5, "escalate": 10.0},
    "abusive": {"auto_approve": 1.0, "inspect": 0.3, "escalate": 0.0},
}

W_SAVINGS = 0.7        # weight of normalised cost savings
W_FAIRNESS = 0.2       # weight of fairness score
W_SIZE_CHART = 0.1     # weight of size-chart top-5 overlap
SAVINGS_FOR_FULL_FAIRNESS = 0.20   # fairness credit is earned in proportion to savings:
                                   # fairness_effective = fairness_score * min(1, max(0, savings) / 0.20)

LOW_VALUE_INR = 1000           # "low-value order": order_value_inr < 1000
NEW_CUSTOMER_DAYS = 90         # "new customer": customer_tenure_days < 90
FAIRNESS_PRIOR_FLAGS_PER_100 = 1   # smoothed flag rate = (flagged genuine + 1 pseudo-flag per 100 genuine rows) / genuine rows
                                   #                    = raw flag rate + 0.01 (same prior for every group, whatever its size)
FAIRNESS_RATIO_SPAN = 1.0      # ratio = max(smoothed a, smoothed b) / min(...) >= 1; pair score = clip(1 - (ratio - 1) / 1.0, 0, 1)
ECE_BINS = 10                  # calibration report only (not part of primary_score)
BOOTSTRAP_REPS = 1000          # bootstrap confidence intervals for the fairness ratios (informational only)
BOOTSTRAP_SEED = 2026
SIZE_CHART_K = 5

PRED_COLS = ["return_id", "decision", "confidence", "justification"]
TRUTH_COLS = ["return_id", "label", "customer_tenure_days", "order_value_inr"]

# Approximate language guess, used ONLY when the truth file has no `text_language` column (e.g. the public train
# file). Hidden test truth carries the true language tag.
_HINDI_MARKERS = set("""
hai hain h nahi nhi nahin nai kya kyu kyun bahut bohot bhot bahot bht mein mai mei ka ki ke ko se pe tha thi
hua hui hue gaya gayi gya gyi aaya aayi aya ayi raha rahi rahe rha rhi kar karo kr kro diya kiya kia liya mila mili chahiye chahie
chaiye wapas vapas wapis yeh ye woh wo vo mujhe muje mera meri mere maine mene hoon hu hun bhi bilkul ekdum accha achha acha achhi
kharab ghatiya bekar bekaar kapda kapde silai rang naap chhota chota chhoti choti bada badi dheela tang lambai jaisa jesa sahi
galat saman samaan dabba khali paise pese dena bhej bhejo mangaya theek thik zyada jyada kuch kuchh abhi aur toh
""".split())


def guess_non_english(text: str) -> bool:
    toks = re.findall(r"[a-z]+", str(text).lower())
    if not toks:
        return False
    hits = sum(t in _HINDI_MARKERS for t in toks)
    return hits >= 1 and hits / len(toks) >= 0.12


class FormatError(Exception):
    pass


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FormatError(f"{what} file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001
        raise FormatError(f"Could not read {what} file {path} as a UTF-8 CSV: {e}")
    df.columns = [c.strip() for c in df.columns]
    return df


def _examples(values, k=5) -> str:
    vals = list(values)[:k]
    return ", ".join(repr(v) for v in vals) + (" ..." if len(values) > k else "")


def load_predictions(path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    df = _read_csv(path, "submission")
    missing = [c for c in PRED_COLS if c not in df.columns]
    if missing:
        raise FormatError(f"Submission is missing column(s) {missing}. Required columns (exact names): {PRED_COLS}. "
                          f"Found: {list(df.columns)}")
    extra = [c for c in df.columns if c not in PRED_COLS]
    if extra:
        warnings.append(f"Extra column(s) ignored: {extra}")
    if len(df) == 0:
        raise FormatError("Submission has no rows.")
    df["return_id"] = df["return_id"].str.strip()
    if (df["return_id"] == "").any():
        raise FormatError(f"{int((df['return_id'] == '').sum())} row(s) have an empty return_id.")
    dup = df["return_id"][df["return_id"].duplicated()]
    if len(dup):
        raise FormatError(f"Duplicate return_id values ({len(dup)} duplicates), e.g. {_examples(dup.unique())}. "
                          f"Each return_id must appear exactly once.")
    df["decision"] = df["decision"].str.strip()
    bad = df.loc[~df["decision"].isin(DECISIONS), "decision"]
    if len(bad):
        raise FormatError(f"{len(bad)} row(s) have an invalid decision, e.g. {_examples(bad.unique())}. "
                          f"Allowed values (lower-case, exactly): {DECISIONS}.")
    raw = df["confidence"].str.strip()
    if (raw == "").any():
        raise FormatError(f"{int((raw == '').sum())} row(s) have an empty confidence. Use a number between 0 and 1.")
    conf = pd.to_numeric(raw, errors="coerce")
    if conf.isna().any():
        raise FormatError(f"confidence must be a number; bad values e.g. {_examples(raw[conf.isna()].unique())}.")
    if (~np.isfinite(conf)).any() or (conf < 0).any() or (conf > 1).any():
        bad = raw[(~np.isfinite(conf)) | (conf < 0) | (conf > 1)]
        raise FormatError(f"confidence must be between 0 and 1 (inclusive); bad values e.g. {_examples(bad.unique())}.")
    df["confidence"] = conf.astype(float)
    empty_just = int((df["justification"].str.strip() == "").sum())
    if empty_just:
        warnings.append(f"{empty_just} row(s) have an empty justification (not scored automatically, but judges read them).")
    return df[PRED_COLS], warnings


def load_truth(path: Path) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    df = _read_csv(path, "truth")
    missing = [c for c in TRUTH_COLS if c not in df.columns]
    if missing:
        raise FormatError(f"Truth file is missing column(s) {missing}.")
    bad = df.loc[~df["label"].isin(LABELS), "label"]
    if len(bad):
        raise FormatError(f"Truth file has unknown labels, e.g. {_examples(bad.unique())}.")
    if df["return_id"].duplicated().any():
        raise FormatError("Truth file has duplicate return_id values.")
    df["customer_tenure_days"] = pd.to_numeric(df["customer_tenure_days"], errors="coerce")
    df["order_value_inr"] = pd.to_numeric(df["order_value_inr"], errors="coerce")
    if "text_language" in df.columns and (df["text_language"] != "").all():
        df["non_english"] = df["text_language"].str.strip().str.lower() != "english"
        notes.append("fairness: language segment uses the true text_language tag")
    elif "customer_reason" in df.columns:
        df["non_english"] = df["customer_reason"].map(guess_non_english)
        notes.append("fairness: no text_language column in truth file, so the language segment uses an APPROXIMATE "
                     "keyword guess (the hidden test truth uses the true language tag)")
    else:
        raise FormatError("Truth file needs either a text_language or a customer_reason column.")
    return df, notes


def load_size_chart(path: Path | None, what: str) -> list[str] | None:
    if path is None or not path.exists():
        return None
    df = _read_csv(path, what)
    for c in ("rank", "product_id"):
        if c not in df.columns:
            raise FormatError(f"{what} {path} must have columns ['rank', 'product_id']; found {list(df.columns)}.")
    if len(df) != SIZE_CHART_K:
        raise FormatError(f"{what} {path} must have exactly {SIZE_CHART_K} rows (found {len(df)}).")
    ranks = pd.to_numeric(df["rank"], errors="coerce")
    if (ranks.isna().any() or (~np.isfinite(ranks)).any() or (ranks != ranks.round()).any()
            or sorted(ranks.astype(int).tolist()) != list(range(1, SIZE_CHART_K + 1))):
        raise FormatError(f"{what} {path}: rank must be the integers 1..{SIZE_CHART_K}, each once.")
    pids = df["product_id"].str.strip().str.upper()   # product IDs are upper-case; tolerate case slips
    if (pids == "").any() or pids.duplicated().any():
        raise FormatError(f"{what} {path}: product_id must be non-empty and unique.")
    return pids.tolist()


# ------------------------------------------------------------------------------------------- metrics
def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise 'average precision', ties handled as one threshold)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]  # last index of each distinct score
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def smoothed_rate(k, n):
    n = np.maximum(n, 1)
    return (k + FAIRNESS_PRIOR_FLAGS_PER_100 * n / 100.0) / n


def fairness(df: pd.DataFrame) -> dict:
    g = df[df["label"] != "abusive"]
    flagged = g["decision"] != "auto_approve"
    pairs = {
        "new_vs_established_customer": (g["customer_tenure_days"] < NEW_CUSTOMER_DAYS, "new (<90 days)", "established"),
        "low_value_vs_other_order": (g["order_value_inr"] < LOW_VALUE_INR, "low value (<1000)", "1000+"),
        "non_english_vs_english_text": (g["non_english"].astype(bool), "non-English", "English"),
    }
    out = {}
    scores = []
    for name, (mask, la, lb) in pairs.items():
        na, nb = int(mask.sum()), int((~mask).sum())
        ka, kb = int(flagged[mask].sum()), int(flagged[~mask].sum())
        sa, sb = smoothed_rate(ka, na), smoothed_rate(kb, nb)
        ratio = max(sa, sb) / min(sa, sb)
        sc = float(min(1.0, max(0.0, 1.0 - (ratio - 1.0) / FAIRNESS_RATIO_SPAN)))
        # stratified bootstrap of genuine rows within each group (for a 0/1 flag this is exactly a binomial draw)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        ba = smoothed_rate(rng.binomial(na, ka / na if na else 0.0, BOOTSTRAP_REPS), na)
        bb = smoothed_rate(rng.binomial(nb, kb / nb if nb else 0.0, BOOTSTRAP_REPS), nb)
        br = np.maximum(ba, bb) / np.minimum(ba, bb)
        lo, hi = np.percentile(br, [2.5, 97.5])
        out[name] = {"group_a": la, "group_b": lb, "n_genuine_a": na, "n_genuine_b": nb, "flagged_a": ka, "flagged_b": kb,
                     "flag_rate_a": round(ka / na, 5) if na else 0.0, "flag_rate_b": round(kb / nb, 5) if nb else 0.0,
                     "smoothed_rate_a": round(sa, 5), "smoothed_rate_b": round(sb, 5),
                     "disparity_ratio": round(ratio, 4), "ratio_95ci_bootstrap": [round(float(lo), 4), round(float(hi), 4)],
                     "score": round(sc, 4)}
        scores.append(sc)
    return {"pairs": out, "fairness_score": float(np.mean(scores))}


def score(sub_path: Path, truth_path: Path, size_chart_path: Path | None, size_truth_path: Path | None) -> dict:
    preds, warnings = load_predictions(sub_path)
    truth, notes = load_truth(truth_path)
    df = truth.merge(preds, on="return_id", how="left")
    missing = df["decision"].isna()
    if missing.mean() > 0.5:
        raise FormatError(
            f"Only {int((~missing).sum())} of {len(df)} return_ids in the truth file have a prediction "
            f"(e.g. truth has {_examples(df['return_id'].head(3))}, your file has {_examples(preds['return_id'].head(3))}). "
            "Check that you are scoring against the right truth file and that return_id values are copied exactly "
            "(same upper-case letters, no extra spaces).")
    n_missing = int(missing.sum())
    is_ab = (df["label"] == "abusive").to_numpy()
    if n_missing:
        # worst case: genuine -> escalate, abusive -> auto_approve; confidence fully wrong
        df.loc[missing & (df["label"] != "abusive"), "decision"] = "escalate"
        df.loc[missing & (df["label"] == "abusive"), "decision"] = "auto_approve"
        df.loc[missing, "confidence"] = np.where(is_ab[missing.to_numpy()], 0.0, 1.0)
    n_extra = int((~preds["return_id"].isin(truth["return_id"])).sum())

    kind = np.where(is_ab, "abusive", "genuine")
    cost = np.array([COST[k][d] for k, d in zip(kind, df["decision"])])
    cost_team = float(cost.sum())
    cost_all_aa = float(sum(COST[k]["auto_approve"] for k in kind))
    cost_oracle = float(sum(min(COST[k].values()) for k in kind))
    denom = cost_all_aa - cost_oracle
    savings = (cost_all_aa - cost_team) / denom if denom > 0 else 0.0
    savings = float(min(1.0, max(-1.0, savings)))

    fair = fairness(df)

    sc_truth = load_size_chart(size_truth_path, "size-chart truth") if size_truth_path else None
    try:
        sc_team = load_size_chart(size_chart_path, "size-chart submission") if size_chart_path else None
    except Exception as e:   # noqa: BLE001 - a broken size-chart file must never sink the whole submission
        sc_team = None
        warnings.append(f"size-chart file ignored (scored 0): {e}")
    if sc_team:
        bad = [p_ for p_ in sc_team if not re.fullmatch(r"PRD-[A-Z0-9]{8}", p_)]
        catalogue = Path(__file__).resolve().parent / "data" / "products.csv"
        if catalogue.exists():
            known = set(pd.read_csv(catalogue, usecols=["product_id"])["product_id"])
            bad = [p_ for p_ in sc_team if p_ not in known]
        if bad:
            warnings.append(f"size-chart product_id(s) not found in the catalogue: {bad}")
    if sc_truth is None:
        size_score = 0.0
        size_note = "no size-chart truth available for this truth file -> size-chart component = 0 (scored by organisers on hidden truth)"
    elif sc_team is None:
        size_score = 0.0
        size_note = "no size-chart submission found (R2_size_chart_top5.csv) -> size-chart component = 0"
    else:
        size_score = len(set(sc_team) & set(sc_truth)) / SIZE_CHART_K
        size_note = f"{len(set(sc_team) & set(sc_truth))} of {SIZE_CHART_K} products correct"

    fairness_credit = min(1.0, max(0.0, savings) / SAVINGS_FOR_FULL_FAIRNESS)
    fairness_effective = fair["fairness_score"] * fairness_credit
    primary = W_SAVINGS * savings + W_FAIRNESS * fairness_effective + W_SIZE_CHART * size_score

    conf_tab = pd.crosstab(df["label"], df["decision"]).reindex(index=LABELS, columns=DECISIONS, fill_value=0)
    flagged = df["decision"] != "auto_approve"
    y = is_ab.astype(float)
    p = df["confidence"].to_numpy(dtype=float)
    res = {
        "primary_score": round(float(primary), 6),
        "primary_score_range": "[-0.7, 1.0]; all-auto_approve = 0.0 (savings 0 -> no fairness credit)",
        "components": {
            "savings": round(savings, 6), "fairness_score": round(fair["fairness_score"], 6),
            "fairness_credit": round(fairness_credit, 6), "fairness_effective": round(fairness_effective, 6),
            "size_chart_score": round(size_score, 6),
            "weights": {"savings": W_SAVINGS, "fairness": W_FAIRNESS, "size_chart": W_SIZE_CHART},
        },
        "cost": {"team_total": round(cost_team, 3), "team_mean_per_request": round(cost_team / len(df), 6),
                 "all_auto_approve_total": round(cost_all_aa, 3), "oracle_total": round(cost_oracle, 3), "cost_matrix": COST},
        "fairness": fair["pairs"],
        "size_chart": {"note": size_note, "your_top5": sc_team},
        "confusion_truth_by_decision": {lab: {d: int(conf_tab.loc[lab, d]) for d in DECISIONS} for lab in LABELS},
        "abuse_detection": {
            "n_abusive": int(y.sum()), "n_genuine": int(len(y) - y.sum()),
            "flag_precision": round(float(y[flagged].mean()), 4) if flagged.any() else None,
            "flag_recall": round(float(flagged[is_ab].mean()), 4) if is_ab.any() else None,
            "genuine_flag_rate": round(float(flagged[~is_ab].mean()), 5),
            "genuine_escalated": int(((df["decision"] == "escalate") & ~is_ab).sum()),
            "flags_with_confidence_below_0.2": int((flagged & (df["confidence"] < 0.2)).sum()),
        },
        "calibration_of_confidence_as_P_abusive": {
            "brier": round(float(np.mean((p - y) ** 2)), 6), "ece_10_bins": round(expected_calibration_error(y, p), 6),
            "average_precision": round(average_precision(y, p), 6),
        },
        "n_truth_rows": int(len(df)), "n_missing_predictions": n_missing, "n_extra_predictions_ignored": n_extra,
        "missing_examples": df.loc[missing, "return_id"].head(5).tolist(), "warnings": warnings, "notes": notes,
    }
    return res


def print_report(r: dict) -> None:
    line = "=" * 78
    print(line)
    print("R2 — Returns Fraud vs. Genuine Grievance Sorter :: evaluation report")
    print(line)
    if r["n_missing_predictions"]:
        print(f"!!! WARNING: {r['n_missing_predictions']} truth return_id(s) have NO prediction and were scored as the WORST case "
              f"(genuine -> escalate, abusive -> auto_approve). Examples: {r['missing_examples']}")
    for w in r["warnings"]:
        print(f"warning: {w}")
    for n in r["notes"]:
        print(f"note: {n}")
    c = r["components"]
    print(f"\nPRIMARY SCORE = {r['primary_score']:.4f}   (= 0.7 x savings + 0.2 x fairness_effective + 0.1 x size_chart)")
    print(f"  savings     = {c['savings']:.4f}   (cost all-approve {r['cost']['all_auto_approve_total']:.1f} -> "
          f"yours {r['cost']['team_total']:.1f}; oracle {r['cost']['oracle_total']:.1f})")
    print(f"  fairness_effective = {c['fairness_effective']:.4f}   (= fairness_score {c['fairness_score']:.4f} x credit "
          f"{c['fairness_credit']:.2f}; credit = min(1, max(0, savings) / {SAVINGS_FOR_FULL_FAIRNESS}))")
    for k, v in r["fairness"].items():
        print(f"      {k}:")
        print(f"          {v['group_a']}: {v['flagged_a']}/{v['n_genuine_a']} genuine flagged ({v['flag_rate_a']:.2%}; smoothed {v['smoothed_rate_a']:.2%})"
              f"  vs  {v['group_b']}: {v['flagged_b']}/{v['n_genuine_b']} ({v['flag_rate_b']:.2%}; smoothed {v['smoothed_rate_b']:.2%})")
        print(f"          ratio {v['disparity_ratio']:.2f}x (95% bootstrap CI {v['ratio_95ci_bootstrap'][0]:.2f}-{v['ratio_95ci_bootstrap'][1]:.2f}) -> pair score {v['score']:.2f}")
    print(f"  size_chart  = {c['size_chart_score']:.4f}   ({r['size_chart']['note']})")
    print(f"\nMean cost per request: {r['cost']['team_mean_per_request']:.4f}")
    print("\nConfusion (rows = truth, columns = your decision):")
    hdr = f"  {'':<24}" + "".join(f"{d:>14}" for d in DECISIONS)
    print(hdr)
    for lab, row in r["confusion_truth_by_decision"].items():
        print(f"  {lab:<24}" + "".join(f"{row[d]:>14}" for d in DECISIONS))
    a = r["abuse_detection"]
    print(f"\nAbuse detection: flag precision {a['flag_precision']}, flag recall {a['flag_recall']}, "
          f"genuine flag rate {a['genuine_flag_rate']}, genuine escalated {a['genuine_escalated']}, "
          f"flags made with confidence < 0.2: {a['flags_with_confidence_below_0.2']}")
    cal = r["calibration_of_confidence_as_P_abusive"]
    print(f"Confidence as P(abusive): Brier {cal['brier']:.4f}, ECE {cal['ece_10_bins']:.4f}, average precision {cal['average_precision']:.4f}")
    print(f"Rows scored: {r['n_truth_rows']}   (extra predictions ignored: {r['n_extra_predictions_ignored']})")
    print(line)


def _clean(o):
    """Make the result strictly JSON-valid (NaN -> null)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score an R2 submission.")
    ap.add_argument("--submission", required=True, help="path to R2_predictions.csv")
    ap.add_argument("--truth", required=True, help="path to truth CSV (e.g. data/return_requests_train.csv)")
    ap.add_argument("--out", default="score.json", help="where to write the JSON score (default score.json)")
    ap.add_argument("--size_chart", default=None, help="your R2_size_chart_top5.csv (default: next to --submission)")
    ap.add_argument("--size_chart_truth", default=None, help="organiser answer file (default: size_chart_truth.csv next to --truth)")
    a = ap.parse_args(argv)
    sub = Path(a.submission)
    truth = Path(a.truth)
    sc = Path(a.size_chart) if a.size_chart else sub.parent / "R2_size_chart_top5.csv"
    sct = Path(a.size_chart_truth) if a.size_chart_truth else truth.parent / "size_chart_truth.csv"
    try:
        r = score(sub, truth, sc, sct)
    except FormatError as e:
        print("\nSUBMISSION FORMAT ERROR\n-----------------------", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("\nFix the file and run again. See README.md, section 'What to submit'.", file=sys.stderr)
        return 2
    print_report(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(_clean(r), indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
