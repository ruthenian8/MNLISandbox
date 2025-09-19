#!/usr/bin/env python
"""CLI for propensity score matching between groundedness slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.propensity import (
    MatchResult,
    average_treatment_effect,
    balance_table,
    fit_propensity,
    nearest_neighbor_match,
    rosenbaum_bounds,
)


def _load(path: Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for propensity matching")
    if path.suffix.lower().endswith("l"):
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _match_to_dict(match: MatchResult) -> Dict[str, Any]:
    return {
        "pairs": match.matched_pairs,
        "caliper": match.caliper,
        "dropped": match.dropped,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Covariate table with groundedness scores")
    parser.add_argument("--treatment", default="low_groundedness", help="Treatment indicator column (1=low)")
    parser.add_argument("--covariates", nargs="+", required=True, help="Covariate columns for the propensity model")
    parser.add_argument("--outcome", default="model_error", help="Outcome column for ATE computation")
    parser.add_argument("--exact", nargs="*", help="Columns for exact matching (e.g., genre, label)")
    parser.add_argument("--caliper", type=float, help="Optional caliper width on the logit scale")
    parser.add_argument("--gamma", nargs="*", type=float, default=[1.0, 1.5, 2.0], help="Gamma grid for Rosenbaum bounds")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    args = parser.parse_args(argv)

    df, _ = fit_propensity(_load(args.data), args.treatment, args.covariates, exact_cols=args.exact)
    match = nearest_neighbor_match(df, args.treatment, caliper=args.caliper, exact_cols=args.exact)
    balance = balance_table(df, args.treatment, args.covariates, match)
    ate = average_treatment_effect(df, match, args.outcome, args.treatment)
    diff = (
        df.loc[[i for i, _ in match.matched_pairs], args.outcome].to_numpy()
        - df.loc[[j for _, j in match.matched_pairs], args.outcome].to_numpy()
    )
    sensitivity = rosenbaum_bounds(diff, args.gamma)
    payload = {
        "match": _match_to_dict(match),
        "balance": {
            "smd": balance.smd,
            "before": balance.before,
            "after": balance.after,
        },
        "ate": ate,
        "sensitivity": sensitivity,
    }
    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
