#!/usr/bin/env python3
"""Evaluate reasoning models on binned MNLI data."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import yaml

from nli_groundedness.reason_eval import map_labels, run_reasoning
from nli_groundedness.stats import per_bin_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnli_binned", type=Path, default=Path("artifacts/mnli_pred_groundedness.parquet"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt_file", type=Path, default=Path("configs/reasoning.yaml"))
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/reasoning_runs"))
    args = parser.parse_args()

    if pd is None:
        raise ImportError("pandas is required for reasoning evaluation")

    df = pd.read_parquet(args.mnli_binned)
    cfg = yaml.safe_load(args.prompt_file.read_text())
    prompt = cfg.get("prompt_template", "{premise}\n{hypothesis}")
    model_cfg = cfg.get("models", [])[0] if cfg.get("models") else {}
    max_new_tokens = int(model_cfg.get("max_new_tokens", 128))
    temperature = float(model_cfg.get("temperature", 0.0))
    label_regex = cfg.get("label_regex", {})

    preds = run_reasoning(
        df,
        args.model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    mapped = map_labels(preds, label_regex)
    merged = df.merge(mapped[["pair_id", "bin", "pred_label", "n_prompt_tok", "n_gen_tok"]], on=["pair_id", "bin"], how="left")
    metrics = per_bin_metrics(merged)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.out_dir / f"preds_{args.model.replace('/', '_')}.parquet"
    metrics_path = args.out_dir / f"metrics_{args.model.replace('/', '_')}.csv"
    mapped.to_parquet(preds_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    print(f"stored predictions at {preds_path} and metrics at {metrics_path}")


if __name__ == "__main__":
    main()
