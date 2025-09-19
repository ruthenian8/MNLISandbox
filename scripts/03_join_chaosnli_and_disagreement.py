#!/usr/bin/env python3
"""Join groundedness scores with ChaosNLI metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.io import load_chaosnli_snli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli_groundedness", type=Path, default=Path("artifacts/snli_groundedness.parquet"))
    parser.add_argument("--chaos_dir", type=Path, default=Path("data/chaosnli"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/snli_chaos_join.parquet"))
    args = parser.parse_args()

    if pd is None:
        raise ImportError("pandas is required for joining datasets")

    grounded = pd.read_parquet(args.snli_groundedness)
    chaos = load_chaosnli_snli(args.chaos_dir)
    merged = grounded.merge(chaos, on="pair_id", how="left")
    merged.to_parquet(args.out, index=False)
    print(f"wrote {len(merged)} rows to {args.out}")


if __name__ == "__main__":
    main()
