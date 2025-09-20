#!/usr/bin/env python
"""CLI for running mixed-effects regressions on groundedness datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.mixed_effects import (
    DEFAULT_CONTROLS,
    RegressionResult,
    RegressionSpec,
    specification_curve,
    vif_table,
)


def _read_table(path: Path) -> "pd.DataFrame":
    if pd is None:  # pragma: no cover - import guard
        raise RuntimeError("pandas is required to load the regression data table")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower().endswith("l"))
    return pd.read_csv(path)


def _result_to_dict(res: RegressionResult) -> Dict[str, Any]:
    return {
        "variant": res.variant,
        "formula": res.formula,
        "coef_groundedness": res.coef_groundedness,
        "std_err": res.std_err,
        "stat": res.stat,
        "p_value": res.p_value,
        "partial_r2": res.partial_r2,
        "n_obs": res.n_obs,
        "controls": list(res.controls),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to a CSV/JSON/Parquet table with groundedness features")
    parser.add_argument("outcome", help="Outcome column (e.g., disagreement, error)")
    parser.add_argument("output", type=Path, help="Path to save the regression summary (JSON)")
    parser.add_argument("--family", choices=["gaussian", "binomial"], default="gaussian")
    parser.add_argument("--genre", action="store_true", help="Include genre fixed effects")
    parser.add_argument("--label", action="store_true", help="Include label fixed effects")
    parser.add_argument("--random", nargs="*", default=["premise_id"], help="Random effects columns")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=list(DEFAULT_CONTROLS),
        help="Control covariates; defaults to the recommended groundedness covariates",
    )
    parser.add_argument("--cluster", help="Cluster column for robust standard errors")
    parser.add_argument("--drop", nargs="*", help="Named control groups to drop (comma separated)")
    parser.add_argument("--diagnostics", type=Path, help="Optional path to save diagnostics JSON")
    args = parser.parse_args(argv)

    df = _read_table(args.data)
    spec = RegressionSpec(
        outcome=args.outcome,
        family=args.family,
        controls=args.controls,
        add_genre=args.genre,
        add_label=args.label,
        random_effects=args.random,
        cluster=args.cluster,
    )

    drop_sets = None
    if args.drop:
        drop_sets = [tuple(item.split(",")) for item in args.drop]

    results, diagnostics = specification_curve(df, spec, drop_sets=drop_sets)
    args.output.write_text(json.dumps([_result_to_dict(r) for r in results], indent=2))

    if args.diagnostics:
        diag_payload: Dict[str, Any] = {"vif": []}
        try:
            diag_payload["vif"] = json.loads(vif_table(df, args.controls).to_json(orient="records"))
        except Exception as exc:  # pragma: no cover - diag best effort
            diag_payload["vif_error"] = str(exc)
        diag_payload.update({k: {"random_effects": v.get("random_effects", {})} for k, v in diagnostics.items()})
        args.diagnostics.write_text(json.dumps(diag_payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
