#!/usr/bin/env python
"""Compute multi-metric evaluation reports for reasoning model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.reason_metrics import (
    accuracy,
    aggregate_runs,
    calibration,
    entropy,
    halt_reason_counts,
    jonckheere_terpstra,
    mean_confidence,
    regress_metric,
    self_consistency,
    token_budget_success,
)


def _load(path: Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for reasoning evaluation")
    if path.suffix.lower().endswith("l"):
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _records(df: "pd.DataFrame"):
    return df.to_dict("records")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="File containing per-example predictions with probabilities")
    parser.add_argument("--grounded-col", default="grounded_bin", help="Column with groundedness bin IDs")
    parser.add_argument("--label-col", default="gold_label", help="Gold label column")
    parser.add_argument("--genre-col", default="genre", help="Genre column for fixed effects")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    args = parser.parse_args(argv)

    df = _load(args.predictions)
    records = _records(df)
    payload: Dict[str, Any] = {
        "accuracy": accuracy(records),
        "confidence": mean_confidence(records),
        "entropy": entropy(records),
        "calibration": calibration(records),
        "self_consistency": self_consistency(records),
        "token_budget": token_budget_success(records),
        "halt_reasons": halt_reason_counts(records),
    }
    try:
        agg = aggregate_runs(records, group_by=args.grounded_col)
        payload["per_bin"] = json.loads(agg.to_json(orient="records"))
    except Exception as exc:  # pragma: no cover
        payload["per_bin_error"] = str(exc)

    try:
        payload["trend"] = jonckheere_terpstra(
            df["confidence"].tolist() if "confidence" in df else df["pred_confidence"].tolist(),
            df[args.grounded_col].tolist(),
        )
    except Exception:
        pass

    try:
        payload["regression"] = regress_metric(
            df.assign(metric=df["confidence"] if "confidence" in df else df["pred_confidence"]),
            metric_col="metric",
            grounded_col=args.grounded_col,
            fixed_effects=[args.genre_col, args.label_col],
        )
    except Exception as exc:  # pragma: no cover
        payload["regression_error"] = str(exc)

    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
