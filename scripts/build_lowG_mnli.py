#!/usr/bin/env python
"""Construct the low-groundedness MNLI release slice."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _load(path: Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required to build the subset")
    if path.suffix.lower().endswith("l"):
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _kl(p: Iterable[float], q: Iterable[float]) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        if qi <= 0:
            continue
        total += float(pi) * math.log(float(pi / qi))
    return total


def _hist(values: Sequence[float], bins: Sequence[float]) -> Dict[str, int]:
    counts = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins) - 1)}
    for value in values:
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                counts[f"{bins[i]}-{bins[i+1]}"] += 1
                break
    return counts


def _baseline_filter(df: "pd.DataFrame", baseline: "pd.DataFrame", id_col: str, conf_col: str) -> "pd.DataFrame":
    if baseline is None:
        return df
    merged = df.merge(baseline[[id_col, conf_col]], on=id_col, how="left")
    return merged[merged[conf_col].fillna(0.0) < 0.8]


@dataclass
class SubsetStats:
    size: int
    label_counts: Dict[str, int]
    genre_counts: Dict[str, int]
    groundedness_by_genre: Dict[str, float]
    length_hist: Dict[str, int]


def build_subset(
    df: "pd.DataFrame",
    grounded_col: str,
    label_col: str,
    genre_col: str,
    pct: float,
    id_col: str = "pair_id",
    baseline: "pd.DataFrame" | None = None,
    baseline_conf: str = "confidence",
) -> tuple["pd.DataFrame", SubsetStats]:
    df = df.copy()
    df = _baseline_filter(df, baseline, id_col, baseline_conf)
    strata = df.groupby([genre_col, label_col])
    keep_idx: List[int] = []
    for (genre, label), group in strata:
        n = max(1, int(len(group) * pct))
        ranked = group.nsmallest(n, grounded_col)
        keep_idx.extend(ranked.index.tolist())
    subset = df.loc[sorted(set(keep_idx))].copy()
    stats = SubsetStats(
        size=len(subset),
        label_counts=subset[label_col].value_counts().to_dict(),
        genre_counts=subset[genre_col].value_counts().to_dict(),
        groundedness_by_genre=subset.groupby(genre_col)[grounded_col].mean().to_dict(),
        length_hist=_hist(subset["premise_len"].tolist(), bins=[0, 10, 20, 40, 80, 120, 200]),
    )
    return subset, stats


def compute_balance_checks(
    original: "pd.DataFrame",
    subset: "pd.DataFrame",
    label_col: str,
    genre_col: str,
) -> Dict[str, float]:
    checks: Dict[str, float] = {}
    for col in (label_col, genre_col):
        base = original[col].value_counts(normalize=True)
        sub = subset[col].value_counts(normalize=True)
        keys = set(base.index) | set(sub.index)
        diff = {k: abs(sub.get(k, 0.0) - base.get(k, 0.0)) for k in keys}
        checks[f"{col}_max_diff"] = max(diff.values()) if diff else 0.0
    checks["length_kl"] = _kl(
        subset["premise_len"].value_counts(normalize=True, bins=20, sort=False).tolist(),
        original["premise_len"].value_counts(normalize=True, bins=20, sort=False).tolist(),
    )
    return checks


def save_jsonl(df: "pd.DataFrame", path: Path, fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf8") as f:
        for _, row in df.iterrows():
            payload = {field: row[field] for field in fields if field in row}
            json.dump(payload, f)
            f.write("\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="File with MNLI premises and predicted groundedness")
    parser.add_argument("--pct", type=float, default=0.25, help="Percent of lowest groundedness to keep")
    parser.add_argument("--grounded-col", default="pred_groundedness", help="Predicted groundedness column")
    parser.add_argument("--label-col", default="gold_label", help="Label column")
    parser.add_argument("--genre-col", default="genre", help="Genre column")
    parser.add_argument("--id-col", default="pair_id", help="Identifier column")
    parser.add_argument("--baseline", type=Path, help="Baseline prediction file for artifact filtering")
    parser.add_argument("--baseline-conf", default="confidence", help="Confidence column in the baseline file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--stats", type=Path, required=True, help="Output summary JSON file")
    args = parser.parse_args(argv)

    df = _load(args.predictions)
    baseline = _load(args.baseline) if args.baseline else None
    subset, stats = build_subset(
        df,
        grounded_col=args.grounded_col,
        label_col=args.label_col,
        genre_col=args.genre_col,
        pct=args.pct,
        id_col=args.id_col,
        baseline=baseline,
        baseline_conf=args.baseline_conf,
    )
    checks = compute_balance_checks(df, subset, args.label_col, args.genre_col)
    fields = [args.id_col, "premise", "hypothesis", args.label_col, args.genre_col, args.grounded_col]
    save_jsonl(subset, args.output, fields)
    payload = {
        "size": stats.size,
        "label_counts": stats.label_counts,
        "genre_counts": stats.genre_counts,
        "groundedness_by_genre": stats.groundedness_by_genre,
        "length_hist": stats.length_hist,
        "checks": checks,
    }
    args.stats.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
