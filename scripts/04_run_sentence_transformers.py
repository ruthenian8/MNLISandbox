#!/usr/bin/env python3
"""Run the sentence-transformer baseline (heuristic fallback)."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.io import load_snli
from nli_groundedness.st_eval import run_st_logreg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli_dir", type=Path, default=Path("data/snli"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/snli_st_eval.parquet"))
    parser.add_argument("--st_model", type=str, default="heuristic")
    args = parser.parse_args()

    snli = load_snli(args.snli_dir)
    if pd is None:
        raise ImportError("pandas is required to store sentence transformer results")

    if isinstance(snli, pd.DataFrame):
        train = snli[snli["split"] == "train"]
        test = snli[snli["split"] == "test"]
    else:
        snli_df = pd.DataFrame(snli.rows)
        train = snli_df[snli_df["split"] == "train"]
        test = snli_df[snli_df["split"] == "test"]

    scored = run_st_logreg(train, test, model_name=args.st_model)
    scored.to_parquet(args.out, index=False)
    print(f"wrote {len(scored)} predictions to {args.out}")


if __name__ == "__main__":
    main()
