#!/usr/bin/env python3
"""Compute groundedness scores for SNLI premises."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from nli_groundedness.groundedness import aggregate_sentence_groundedness
from nli_groundedness.vlm_scorer import CaptionerAndLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli_parquet", type=Path, default=Path("artifacts/snli_w_images.parquet"))
    parser.add_argument("--cap_ckpt", type=str, default="stub")
    parser.add_argument("--out", type=Path, default=Path("artifacts/snli_groundedness.parquet"))
    parser.add_argument("--aggregate", type=str, default="content_mean")
    args = parser.parse_args()

    if pd is None:
        raise ImportError("pandas is required to compute groundedness scores")

    df = pd.read_parquet(args.snli_parquet)
    scorer = CaptionerAndLM(cap_ckpt=args.cap_ckpt)
    tokenizer = scorer.processor.tokenizer

    rows = []
    for row in df.itertuples():
        ids_txt, lp_txt = scorer.token_logprobs_text_only(row.premise)
        ids_cap, lp_cap = scorer.token_logprobs_captioner(None, row.premise)
        if getattr(ids_txt, "tolist", None) and getattr(ids_cap, "tolist", None):
            if ids_txt.tolist() != ids_cap.tolist():
                raise ValueError("token ids differ between text-only and caption passes")
        result = aggregate_sentence_groundedness(
            lp_cap,
            lp_txt,
            ids_txt,
            tokenizer,
            upos=None,
            aggregate=args.aggregate,
        )
        rows.append(
            {
                "pair_id": row.pair_id,
                "split": row.split,
                "G_sentence_mean": result.G_sentence_mean,
                "G_sentence_uc": result.G_sentence_uc,
                "token_diagnostics": [diag.__dict__ for diag in result.token_diagnostics],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"wrote {len(out_df)} groundedness scores to {args.out}")


if __name__ == "__main__":
    main()
