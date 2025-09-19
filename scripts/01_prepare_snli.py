#!/usr/bin/env python3
"""Prepare SNLI data with Flickr30k image paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from nli_groundedness.config import ensure_dirs
from nli_groundedness.io import attach_images_to_snli, load_flickr_index, load_snli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli_dir", type=Path, default=Path("data/snli"))
    parser.add_argument("--flickr_index_csv", type=Path, default=Path("data/flickr30k_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/snli_w_images.parquet"))
    args = parser.parse_args()

    snli = load_snli(args.snli_dir)
    flickr = load_flickr_index(args.flickr_index_csv)
    merged = attach_images_to_snli(snli, flickr)
    ensure_dirs(args.out.parent)
    merged.to_parquet(args.out, index=False)  # type: ignore[attr-defined]
    print(f"wrote {len(merged)} rows to {args.out}")


if __name__ == "__main__":
    main()
