#!/usr/bin/env python3
"""Generate analysis figures (skipped when matplotlib is unavailable)."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli_groundedness", type=Path, default=Path("artifacts/snli_groundedness.parquet"))
    parser.add_argument("--mnli_groundedness", type=Path, default=Path("artifacts/mnli_pred_groundedness.parquet"))
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    if plt is None or pd is None:
        print("matplotlib/pandas unavailable; skipping figure generation")
        return

    snli = pd.read_parquet(args.snli_groundedness)
    mnli = pd.read_parquet(args.mnli_groundedness)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(snli["G_sentence_mean"].dropna(), bins=30, alpha=0.7)
    ax.set_title("SNLI Groundedness Distribution")
    ax.set_xlabel("Groundedness")
    ax.set_ylabel("Count")
    fig.savefig(args.out_dir / "snli_groundedness_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mnli["G_pred"].dropna(), bins=30, alpha=0.7, color="orange")
    ax.set_title("MNLI Predicted Groundedness")
    ax.set_xlabel("Groundedness")
    ax.set_ylabel("Count")
    fig.savefig(args.out_dir / "mnli_groundedness_hist.png", dpi=150)
    plt.close(fig)

    print(f"saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
