"""Groundedness binning utilities."""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _quantiles(values: List[float], num_bins: int) -> List[float]:
    if not values:
        return [0.0] * (num_bins + 1)
    ordered = sorted(values)
    n = len(ordered)
    edges: List[float] = []
    for i in range(num_bins + 1):
        q = i / num_bins
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(n - 1, lo + 1)
        if hi == lo:
            value = ordered[lo]
        else:
            frac = pos - lo
            value = ordered[lo] * (1 - frac) + ordered[hi] * frac
        edges.append(value)
    return edges


def _digitize(value: float, edges: Sequence[float]) -> int:
    for idx in range(len(edges) - 1):
        if value <= edges[idx + 1] or idx == len(edges) - 2:
            return idx
    return len(edges) - 2


def add_bins(data, score_col: str, num_bins: int = 5, strategy: str = "quantile", cutpoints: Iterable[float] | None = None):
    """Add groundedness bins to a dataframe or iterable of dictionaries."""

    if pd is not None and isinstance(data, pd.DataFrame):
        series = data[score_col]
        if strategy == "quantile":
            edges = _quantiles([float(v) for v in series.dropna()], num_bins)
        else:
            cp = list(cutpoints or [])
            if len(cp) != num_bins + 1:
                raise ValueError("cutpoints must contain num_bins + 1 values")
            edges = [float(v) for v in cp]
        bins = [_digitize(float(v), edges) if v == v else None for v in series]
        data = data.copy()
        data["bin"] = bins
        data["bin_left"] = [edges[b] if b is not None else None for b in bins]
        data["bin_right"] = [edges[b + 1] if b is not None else None for b in bins]
        return data, edges

    rows = [dict(row) for row in data]
    values = [float(row[score_col]) for row in rows if row.get(score_col) is not None]
    if strategy == "quantile":
        edges = _quantiles(values, num_bins)
    else:
        cp = list(cutpoints or [])
        if len(cp) != num_bins + 1:
            raise ValueError("cutpoints must contain num_bins + 1 values")
        edges = [float(v) for v in cp]
    for row in rows:
        value = row.get(score_col)
        if value is None:
            row["bin"] = None
            row["bin_left"] = None
            row["bin_right"] = None
            continue
        idx = _digitize(float(value), edges)
        row["bin"] = idx
        row["bin_left"] = edges[idx]
        row["bin_right"] = edges[idx + 1]
    return rows, edges
