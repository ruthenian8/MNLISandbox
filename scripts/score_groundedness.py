#!/usr/bin/env python
"""Score groundedness using the lightweight regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.roberta_regressor import train_eval


def _load(path: Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for groundedness scoring")
    if path.suffix.lower().endswith("l"):
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="File with premises to score")
    parser.add_argument("--train", type=Path, help="Optional training data with gold groundedness")
    parser.add_argument("--text-col", default="premise", help="Column containing the premise text")
    parser.add_argument("--target-col", default="groundedness", help="Gold groundedness column for training")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args(argv)

    df = _load(args.data)
    if args.train:
        train_df = _load(args.train)
        preds = train_eval(
            train_df[args.text_col].tolist(),
            train_df[args.target_col].tolist(),
            df[args.text_col].tolist(),
        )
    else:
        preds = train_eval([], [], df[args.text_col].tolist())
    df = df.copy()
    df["pred_groundedness"] = preds
    with args.output.open("w", encoding="utf8") as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f)
            f.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
