#!/usr/bin/env python3
"""Apply the groundedness regressor to MNLI and create bins."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.binning import add_bins
from nli_groundedness.io import load_mnli
from nli_groundedness.roberta_regressor import train_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli_groundedness", type=Path, default=Path("artifacts/snli_groundedness.parquet"))
    parser.add_argument("--mnli_dir", type=Path, default=Path("data/mnli"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/mnli_pred_groundedness.parquet"))
    parser.add_argument("--num_bins", type=int, default=5)
    args = parser.parse_args()

    if pd is None:
        raise ImportError("pandas is required for regressor application")

    snli = pd.read_parquet(args.snli_groundedness).dropna(subset=["G_sentence_mean"])
    mnli = load_mnli(args.mnli_dir)
    if not isinstance(mnli, pd.DataFrame):
        mnli = pd.DataFrame(mnli.rows)

    preds = train_eval(snli["premise"].tolist(), snli["G_sentence_mean"].tolist(), mnli["premise"].tolist())
    mnli = mnli.copy()
    mnli["G_pred"] = preds
    mnli, edges = add_bins(mnli, "G_pred", num_bins=args.num_bins)
    mnli.to_parquet(args.out, index=False)
    print(f"wrote {len(mnli)} MNLI predictions to {args.out}; edges={edges}")


if __name__ == "__main__":
    main()
