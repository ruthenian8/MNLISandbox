#!/usr/bin/env python
"""Run the groundedness regressor validation ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.regressor_validation import (
    RegressionMetrics,
    calibration_curve,
    delta_vs_baseline,
    evaluate_predictions,
    ladder_split,
    partial_correlation,
)


def _load_table(path: Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for validation")
    if path.suffix.lower().endswith("l"):
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _metrics_to_dict(metrics: RegressionMetrics) -> Dict[str, float]:
    return {
        "spearman": metrics.spearman,
        "kendall": metrics.kendall,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "partial_spearman": metrics.partial_spearman,
        "partial_pearson": metrics.partial_pearson,
        "ece": metrics.ece,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Table containing groundedness targets and covariates")
    parser.add_argument("--id-col", default="premise_id", help="Identifier column for group splits")
    parser.add_argument("--pred-col", default="pred_groundedness", help="Prediction column")
    parser.add_argument("--gold-col", default="groundedness", help="Gold groundedness column")
    parser.add_argument("--concreteness", default="concreteness_mean", help="Concreteness column name")
    parser.add_argument("--length", default="premise_len", help="Length column name")
    parser.add_argument("--wf", default="wf_mean", help="Word frequency column name")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    args = parser.parse_args(argv)

    df = _load_table(args.data)
    train_ids, dev_ids, test_ids = ladder_split(df[args.id_col].tolist())
    mask = df[args.id_col].isin(test_ids)
    test_df = df[mask]

    metrics = evaluate_predictions(
        test_df[args.pred_col],
        test_df[args.gold_col],
        controls={
            "concreteness": test_df[args.concreteness],
            "length": test_df[args.length],
        },
    )
    baseline = evaluate_predictions(
        test_df[args.concreteness],
        test_df[args.gold_col],
        controls={"length": test_df[args.length]},
    )
    bins, ece = calibration_curve(test_df[args.pred_col], test_df[args.gold_col])
    payload = {
        "metrics": _metrics_to_dict(metrics),
        "baseline": _metrics_to_dict(baseline),
        "deltas": delta_vs_baseline(metrics, baseline),
        "calibration": {"bins": bins, "ece": ece},
        "partial_corr": {
            "spearman": partial_correlation(
                test_df[args.pred_col],
                test_df[args.gold_col],
                controls={
                    "concreteness": test_df[args.concreteness],
                    "length": test_df[args.length],
                },
                method="spearman",
            )
        },
    }
    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
