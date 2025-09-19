"""Propensity-score matching utilities for low vs high groundedness analyses."""

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

try:  # pragma: no cover - optional dependency
    import statsmodels.formula.api as smf  # type: ignore
except Exception:  # pragma: no cover
    smf = None  # type: ignore


@dataclass
class MatchResult:
    matched_pairs: List[Tuple[int, int]]
    propensity: List[float]
    caliper: Optional[float]
    dropped: List[int]


@dataclass
class BalanceReport:
    smd: Dict[str, float]
    before: Dict[str, float]
    after: Dict[str, float]


def _ensure_numpy():
    if np is None:
        raise RuntimeError("numpy is required for propensity score matching utilities")


def _to_frame(data) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for propensity score matching utilities")
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        return pd.DataFrame([data])
    return pd.DataFrame(list(data))


def fit_propensity(
    data,
    treatment_col: str,
    covariates: Sequence[str],
    exact_cols: Optional[Sequence[str]] = None,
) -> Tuple["pd.DataFrame", List[float]]:
    """Estimate propensity scores via logistic regression."""

    frame = _to_frame(data)
    if smf is None:
        raise RuntimeError("statsmodels is required for propensity estimation")
    formula = f"{treatment_col} ~ " + " + ".join(covariates)
    model = smf.logit(formula, data=frame)
    result = model.fit(disp=False)
    scores = result.predict(frame)
    frame = frame.assign(propensity=scores)
    if exact_cols:
        for col in exact_cols:
            frame[col] = frame[col].astype(str)
    return frame, list(scores)


def _logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def nearest_neighbor_match(
    frame: "pd.DataFrame",
    treatment_col: str,
    propensity_col: str = "propensity",
    caliper: Optional[float] = None,
    exact_cols: Optional[Sequence[str]] = None,
) -> MatchResult:
    _ensure_numpy()
    treated = frame[frame[treatment_col] == 1]
    control = frame[frame[treatment_col] == 0]
    treated_idx = treated.index.to_numpy()
    control_idx = control.index.to_numpy()
    treated_scores = treated[propensity_col].apply(_logit).to_numpy()
    control_scores = control[propensity_col].apply(_logit).to_numpy()
    used = set()
    pairs: List[Tuple[int, int]] = []
    dropped: List[int] = []
    cal = caliper if caliper is not None else float("inf")
    for idx, score in enumerate(treated_scores):
        mask = np.ones(len(control_scores), dtype=bool)
        if exact_cols:
            for col in exact_cols:
                mask &= control[col].to_numpy() == treated[col].iloc[idx]
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            dropped.append(idx)
            continue
        distances = np.abs(control_scores[candidates] - score)
        order = candidates[np.argsort(distances)]
        chosen = None
        for j in order:
            if j in used:
                continue
            if distances[np.where(candidates == j)[0][0]] <= cal:
                chosen = j
                break
        if chosen is None:
            dropped.append(idx)
            continue
        used.add(chosen)
        pairs.append((int(treated_idx[idx]), int(control_idx[chosen])))
    return MatchResult(pairs, list(frame[propensity_col]), caliper, dropped)


def _smd(values_t: "np.ndarray", values_c: "np.ndarray") -> float:
    mean_t = float(values_t.mean())
    mean_c = float(values_c.mean())
    var = float(values_t.var(ddof=1) + values_c.var(ddof=1)) / 2.0
    if var == 0:
        return 0.0
    return (mean_t - mean_c) / math.sqrt(var)


def balance_table(
    frame: "pd.DataFrame",
    treatment_col: str,
    covariates: Sequence[str],
    matches: Optional[MatchResult] = None,
) -> BalanceReport:
    _ensure_numpy()
    treated = frame[frame[treatment_col] == 1]
    control = frame[frame[treatment_col] == 0]
    before = {}
    after = {}
    smd_vals = {}
    for col in covariates:
        values_t = treated[col].to_numpy(dtype=float)
        values_c = control[col].to_numpy(dtype=float)
        smd_vals[col] = _smd(values_t, values_c)
        before[col] = float(values_t.mean() - values_c.mean())
    if matches and matches.matched_pairs:
        matched_t = frame.loc[[i for i, _ in matches.matched_pairs]]
        matched_c = frame.loc[[j for _, j in matches.matched_pairs]]
        for col in covariates:
            values_t = matched_t[col].to_numpy(dtype=float)
            values_c = matched_c[col].to_numpy(dtype=float)
            after[col] = float(values_t.mean() - values_c.mean())
    return BalanceReport(smd=smd_vals, before=before, after=after)


def average_treatment_effect(
    frame: "pd.DataFrame",
    matches: MatchResult,
    outcome_col: str,
    treatment_col: str,
    n_boot: int = 500,
    random_state: int = 13,
) -> Dict[str, float]:
    _ensure_numpy()
    rng = np.random.default_rng(random_state)
    treated = frame.loc[[i for i, _ in matches.matched_pairs]][outcome_col].to_numpy(dtype=float)
    control = frame.loc[[j for _, j in matches.matched_pairs]][outcome_col].to_numpy(dtype=float)
    diff = treated - control
    ate = float(diff.mean())
    if diff.size == 0:
        return {"ate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=diff.size, replace=True)
        boot.append(float(sample.mean()))
    boot.sort()
    lower = boot[int(0.025 * len(boot))]
    upper = boot[int(0.975 * len(boot))]
    return {"ate": ate, "ci_lower": lower, "ci_upper": upper}


def rosenbaum_bounds(diff: Sequence[float], gamma: Sequence[float]) -> Dict[float, float]:
    """Compute Rosenbaum sensitivity bounds for matched differences."""

    _ensure_numpy()
    diffs = np.asarray(list(diff), dtype=float)
    bounds: Dict[float, float] = {}
    for g in gamma:
        adjusted = diffs * (1 - g) / (1 + g)
        t_stat = abs(adjusted.mean()) / (adjusted.std(ddof=1) / math.sqrt(len(adjusted))) if adjusted.std(ddof=1) else float("inf")
        bounds[g] = float(t_stat)
    return bounds
