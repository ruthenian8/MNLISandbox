#!/usr/bin/env python3
"""Train the text-only groundedness regressor (linear fallback)."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.roberta_regressor import train_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli_groundedness", type=Path, default=Path("artifacts/snli_groundedness.parquet"))
    parser.add_argument("--split_for_dev", type=str, default="dev")
    parser.add_argument("--out_preds", type=Path, default=Path("artifacts/regressor/snli_dev_preds_mean_of_seeds.parquet"))
    args = parser.parse_args()

    if pd is None:
        raise ImportError("pandas is required for regressor training")

    df = pd.read_parquet(args.snli_groundedness)
    train_df = df[df["split"] == "train"].dropna(subset=["G_sentence_mean"])
    dev_df = df[df["split"] == args.split_for_dev].dropna(subset=["G_sentence_mean"])

    preds = train_eval(train_df["premise"].tolist(), train_df["G_sentence_mean"].tolist(), dev_df["premise"].tolist())
    out_df = pd.DataFrame({"pair_id": dev_df["pair_id"].tolist(), "pred": preds})
    args.out_preds.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out_preds, index=False)
    print(f"wrote {len(out_df)} predictions to {args.out_preds}")


if __name__ == "__main__":
    main()
