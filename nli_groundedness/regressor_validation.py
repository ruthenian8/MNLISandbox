"""Validation helpers for the text-only groundedness regressor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass
class RegressionMetrics:
    """Container for a suite of regression metrics."""

    spearman: float
    kendall: float
    mae: float
    rmse: float
    partial_spearman: Optional[float] = None
    partial_pearson: Optional[float] = None
    ece: Optional[float] = None


def _ensure_numpy():
    if np is None:
        raise RuntimeError("numpy is required for the regression validation utilities")


def _to_numpy(values: Sequence[float]) -> "np.ndarray":
    _ensure_numpy()
    if isinstance(values, np.ndarray):
        return values.astype(float)
    return np.asarray(list(values), dtype=float)


def _rankdata(values: Sequence[float]) -> "np.ndarray":
    arr = _to_numpy(values)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[order[i]] == arr[order[j]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    _ensure_numpy()
    x_arr = _to_numpy(x)
    y_arr = _to_numpy(y)
    if x_arr.size != y_arr.size or x_arr.size == 0:
        return float("nan")
    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    x_centered = x_arr - x_mean
    y_centered = y_arr - y_mean
    denom = float(np.sqrt((x_centered**2).sum() * (y_centered**2).sum()))
    if denom == 0:
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def _kendall(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        return float("nan")
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _mae(preds: Sequence[float], gold: Sequence[float]) -> float:
    return sum(abs(p - g) for p, g in zip(preds, gold)) / float(len(preds)) if preds else 0.0


def _rmse(preds: Sequence[float], gold: Sequence[float]) -> float:
    if not preds:
        return 0.0
    return math.sqrt(sum((p - g) ** 2 for p, g in zip(preds, gold)) / float(len(preds)))


def _design_matrix(controls: Mapping[str, Sequence[float]], add_intercept: bool = True) -> "np.ndarray":
    _ensure_numpy()
    matrices: List[np.ndarray] = []
    if add_intercept:
        matrices.append(np.ones(len(next(iter(controls.values())))))
    for col in controls.values():
        matrices.append(_to_numpy(col))
    return np.vstack(matrices).T


def _residualize(values: Sequence[float], controls: Mapping[str, Sequence[float]]) -> "np.ndarray":
    if not controls:
        return _to_numpy(values)
    X = _design_matrix(controls)
    y = _to_numpy(values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_correlation(
    preds: Sequence[float],
    gold: Sequence[float],
    controls: Mapping[str, Sequence[float]],
    method: str = "spearman",
) -> float:
    """Compute partial correlation between predictions and gold scores."""

    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    r_pred = _residualize(preds, controls)
    r_gold = _residualize(gold, controls)
    if method == "spearman":
        return _spearman(r_pred, r_gold)
    return _pearson(r_pred, r_gold)


def calibration_curve(
    preds: Sequence[float],
    gold: Sequence[float],
    n_bins: int = 10,
) -> Tuple[List[Tuple[float, float]], float]:
    """Return per-bin calibration pairs and the regression ECE."""

    _ensure_numpy()
    preds_arr = _to_numpy(preds)
    gold_arr = _to_numpy(gold)
    if preds_arr.size == 0:
        return [], float("nan")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(preds_arr, quantiles)
    bins = []
    ece_total = 0.0
    for i in range(n_bins):
        mask = (preds_arr >= edges[i]) & (preds_arr <= edges[i + 1]) if i == n_bins - 1 else (preds_arr >= edges[i]) & (preds_arr < edges[i + 1])
        if not mask.any():
            bins.append((float(preds_arr.mean()), float(gold_arr.mean())))
            continue
        pred_mean = float(preds_arr[mask].mean())
        gold_mean = float(gold_arr[mask].mean())
        bins.append((pred_mean, gold_mean))
        ece_total += abs(pred_mean - gold_mean) * mask.mean()
    return bins, ece_total


def evaluate_predictions(
    preds: Sequence[float],
    gold: Sequence[float],
    controls: Optional[Mapping[str, Sequence[float]]] = None,
    n_bins: int = 10,
) -> RegressionMetrics:
    controls = controls or {}
    rho = _spearman(preds, gold)
    tau = _kendall(preds, gold)
    mae = _mae(preds, gold)
    rmse = _rmse(preds, gold)
    partial_s = partial_correlation(preds, gold, controls, method="spearman") if controls else None
    partial_p = partial_correlation(preds, gold, controls, method="pearson") if controls else None
    _, ece = calibration_curve(preds, gold, n_bins=n_bins)
    return RegressionMetrics(rho, tau, mae, rmse, partial_s, partial_p, ece)


def ladder_split(
    ids: Sequence[str],
    train: float = 0.8,
    dev: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    """Deterministic split based on sorted unique identifiers."""

    unique = sorted(set(ids))
    n = len(unique)
    n_train = int(n * train)
    n_dev = int(n * dev)
    train_ids = unique[:n_train]
    dev_ids = unique[n_train : n_train + n_dev]
    test_ids = unique[n_train + n_dev :]
    return train_ids, dev_ids, test_ids


def apply_split(df, id_col: str, ids: Sequence[str]) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for DataFrame operations")
    return df[df[id_col].isin(ids)].copy()


def concreteness_baseline(
    concreteness: Sequence[float],
    wf: Sequence[float],
    lengths: Sequence[float],
    targets: Sequence[float],
) -> RegressionMetrics:
    """Closed-form linear regression baseline using concreteness and length."""

    _ensure_numpy()
    X = np.vstack([np.ones(len(concreteness)), concreteness, wf, lengths]).T
    y = _to_numpy(targets)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ beta
    return evaluate_predictions(preds, y, controls={}, n_bins=10)


def delta_vs_baseline(main: RegressionMetrics, baseline: RegressionMetrics) -> Dict[str, float]:
    return {
        "delta_spearman": main.spearman - baseline.spearman,
        "delta_mae": baseline.mae - main.mae,
    }


def collect_examples(
    df: "pd.DataFrame",
    pred_col: str,
    gold_col: str,
    text_col: str,
    k: int = 20,
) -> Dict[str, "pd.DataFrame"]:
    if pd is None:
        raise RuntimeError("pandas is required for collecting examples")
    df = df.copy()
    df["resid"] = (df[pred_col] - df[gold_col]).abs()
    best = df.nsmallest(k, "resid")[ [text_col, pred_col, gold_col, "resid"] ]
    worst = df.nlargest(k, "resid")[ [text_col, pred_col, gold_col, "resid"] ]
    return {"best": best, "worst": worst}


def attach_attributions(
    examples: Mapping[str, "pd.DataFrame"],
    attributions: Mapping[str, Sequence[Sequence[float]]],
    tokens: Mapping[str, Sequence[Sequence[str]]],
) -> Dict[str, List[Dict[str, object]]]:
    """Attach token-level attributions to each example."""

    enriched: Dict[str, List[Dict[str, object]]] = {}
    for split, frame in examples.items():
        rows: List[Dict[str, object]] = []
        for idx, row in frame.iterrows():
            key = str(idx)
            contrib = attributions.get(key, [])
            toks = tokens.get(key, [])
            rows.append(
                {
                    "text": row.iloc[0],
                    "prediction": float(row.iloc[1]),
                    "gold": float(row.iloc[2]),
                    "residual": float(row.iloc[3]),
                    "tokens": list(toks),
                    "attribution": list(contrib),
                }
            )
        enriched[split] = rows
    return enriched
