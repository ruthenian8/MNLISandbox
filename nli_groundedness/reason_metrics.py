"""Evaluation helpers for the controlled reasoning experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import statsmodels.formula.api as smf  # type: ignore
except Exception:  # pragma: no cover
    smf = None  # type: ignore


def _ensure_numpy():
    if np is None:
        raise RuntimeError("numpy is required for reasoning metric utilities")


def _probabilities(row) -> List[float]:
    probs = row.get("probs")
    if probs is None:
        logits = row.get("logits")
        if logits is None:
            return []
        _ensure_numpy()
        arr = np.asarray(logits, dtype=float)
        arr = arr - arr.max()
        exp = np.exp(arr)
        probs = (exp / exp.sum()).tolist()
    return [float(p) for p in probs]


def _entropy(probs: Sequence[float]) -> float:
    if not probs:
        return float("nan")
    total = 0.0
    for p in probs:
        if p <= 0:
            continue
        total -= float(p) * math.log(float(p))
    return total


def accuracy(records: Iterable[Mapping[str, object]]) -> float:
    total = 0
    correct = 0
    for row in records:
        total += 1
        if row.get("pred_label") == row.get("gold_label"):
            correct += 1
    return correct / total if total else 0.0


def mean_confidence(records: Iterable[Mapping[str, object]]) -> float:
    total = 0
    weight = 0
    for row in records:
        probs = _probabilities(row)
        if not probs:
            continue
        pred_idx = int(row.get("pred_index", 0))
        if 0 <= pred_idx < len(probs):
            total += probs[pred_idx]
            weight += 1
    return total / weight if weight else float("nan")


def entropy(records: Iterable[Mapping[str, object]]) -> float:
    ent = 0.0
    weight = 0
    for row in records:
        probs = _probabilities(row)
        if not probs:
            continue
        ent += _entropy(probs)
        weight += 1
    return ent / weight if weight else float("nan")


def calibration(records: Iterable[Mapping[str, object]], n_bins: int = 10) -> Dict[str, object]:
    _ensure_numpy()
    rows = list(records)
    if not rows:
        return {"bins": [], "ece": float("nan")}
    confidences: List[float] = []
    outcomes: List[int] = []
    for row in rows:
        probs = _probabilities(row)
        if not probs:
            continue
        pred_idx = int(row.get("pred_index", 0))
        confidences.append(probs[pred_idx])
        outcomes.append(int(row.get("pred_label") == row.get("gold_label")))
    if not confidences:
        return {"bins": [], "ece": float("nan")}
    conf = np.asarray(confidences)
    outcome = np.asarray(outcomes)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(conf, quantiles)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if not mask.any():
            continue
        pred_mean = float(conf[mask].mean())
        acc = float(outcome[mask].mean())
        bins.append({"confidence": pred_mean, "accuracy": acc, "count": int(mask.sum())})
        ece += abs(pred_mean - acc) * (mask.sum() / conf.size)
    return {"bins": bins, "ece": ece}


def self_consistency(records: Iterable[Mapping[str, object]], group_col: str = "pair_id") -> float:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(group_col))].append(str(row.get("pred_label")))
    variances = []
    for labels in grouped.values():
        counts = defaultdict(int)
        for label in labels:
            counts[label] += 1
        total = sum(counts.values())
        if total <= 1:
            variances.append(0.0)
            continue
        max_share = max(counts.values()) / total
        variances.append(1.0 - max_share)
    return float(sum(variances) / len(variances)) if variances else float("nan")


def token_budget_success(records: Iterable[Mapping[str, object]], budget_col: str = "max_tokens") -> Dict[str, float]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for row in records:
        budget = str(row.get(budget_col))
        correct = int(row.get("pred_label") == row.get("gold_label"))
        buckets[budget].append(correct)
    return {budget: sum(vals) / len(vals) for budget, vals in buckets.items() if vals}


def rationale_stats(records: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    lengths = [float(row.get("rationale_len", 0.0)) for row in records if row.get("rationale_len") is not None]
    if not lengths:
        return {"mean": float("nan"), "median": float("nan")}
    lengths.sort()
    median = lengths[len(lengths) // 2] if len(lengths) % 2 else 0.5 * (lengths[len(lengths) // 2 - 1] + lengths[len(lengths) // 2])
    return {"mean": sum(lengths) / len(lengths), "median": median}


def halt_reason_counts(records: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in records:
        counts[str(row.get("halt_reason", "unknown"))] += 1
    return counts


def jonckheere_terpstra(values: Sequence[float], groups: Sequence[int]) -> Dict[str, float]:
    """Compute the Jonckheere–Terpstra trend statistic using a normal approximation."""

    _ensure_numpy()
    if len(values) != len(groups):
        raise ValueError("values and groups must have the same length")
    order = np.argsort(groups)
    sorted_vals = np.asarray(values)[order]
    sorted_groups = np.asarray(groups)[order]
    J = 0.0
    for i in range(len(sorted_vals)):
        mask = sorted_groups[i + 1 :] > sorted_groups[i]
        J += float((sorted_vals[i] < sorted_vals[i + 1 :][mask]).sum())
    mean_J = len(values) * (len(values) - 1) / 4.0
    var_J = len(values) * (2 * len(values) + 5) / 72.0
    z = (J - mean_J) / math.sqrt(var_J) if var_J > 0 else float("nan")
    return {"J": J, "z": z}


def regress_metric(
    frame: "pd.DataFrame",
    metric_col: str,
    grounded_col: str,
    fixed_effects: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if pd is None or smf is None:
        raise RuntimeError("pandas and statsmodels are required for regression")
    fixed_effects = fixed_effects or []
    fe_terms = " + ".join([f"C({col})" for col in fixed_effects])
    rhs = f"{grounded_col} + {fe_terms}" if fe_terms else grounded_col
    formula = f"{metric_col} ~ {rhs}"
    model = smf.ols(formula, data=frame)
    result = model.fit()
    coef = result.params.get(grounded_col, float("nan"))
    se = result.bse.get(grounded_col, float("nan"))
    pval = result.pvalues.get(grounded_col, float("nan"))
    return {"coef": float(coef), "se": float(se), "p": float(pval)}


def aggregate_runs(records: Iterable[Mapping[str, object]], group_by: str = "bin") -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for aggregation")
    df = pd.DataFrame(list(records))
    grouped = df.groupby(group_by)
    summary = grouped.apply(lambda g: pd.Series({
        "accuracy": accuracy(g.to_dict("records")),
        "confidence": mean_confidence(g.to_dict("records")),
        "entropy": entropy(g.to_dict("records")),
    }))
    summary.reset_index(inplace=True)
    return summary
