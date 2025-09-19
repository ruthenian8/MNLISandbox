"""Statistics helpers for groundedness analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import scipy.stats

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _ensure_iterable(df):
    if pd is not None and isinstance(df, pd.DataFrame):
        return df.to_dict("records")
    if hasattr(df, "rows"):
        return list(df.rows)
    return list(df)


def corr_table(df, gcol: str = "G_sentence_mean"):
    rows = _ensure_iterable(df)
    targets: Dict[str, List[float]] = defaultdict(list)
    grounded_per_col: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        g = row.get(gcol)
        if g is None:
            continue
        for col in ("snli_disagreement", "chaos_entropy", "st_is_error"):
            value = row.get(col)
            if value is None:
                continue
            grounded_per_col[col].append(float(g))
            targets[col].append(float(value))
    results = {}
    for col, values in targets.items():
        grounded = grounded_per_col[col]
        if not values or not grounded or len(values) != len(grounded):
            continue
        # Use scipy.stats.spearmanr instead of custom _spearman
        correlation, p_value = scipy.stats.spearmanr(grounded, values)
        results[col] = {"spearman_rho": correlation, "p": p_value, "n": len(values)}
    if pd is not None:
        return pd.DataFrame.from_dict(results, orient="index")
    return results


def per_bin_metrics(df_bins, gold_col: str = "gold_label", pred_col: str = "pred_label"):
    rows = _ensure_iterable(df_bins)
    groups: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[int, int] = defaultdict(int)
    for row in rows:
        bin_id = row.get("bin")
        if bin_id is None:
            continue
        counts[bin_id] += 1
        pred = row.get(pred_col)
        gold = row.get(gold_col)
        if pred == gold:
            groups[bin_id]["correct"] += 1
        groups[bin_id]["prompt_tok"] += float(row.get("n_prompt_tok", 0.0))
        groups[bin_id]["gen_tok"] += float(row.get("n_gen_tok", 0.0))
    results = []
    for bin_id in sorted(counts.keys()):
        n = counts[bin_id]
        correct = groups[bin_id]["correct"]
        acc = correct / n if n else 0.0
        prompt = groups[bin_id]["prompt_tok"] / n if n else 0.0
        gen = groups[bin_id]["gen_tok"] / n if n else 0.0
        results.append(
            {
                "bin": bin_id,
                "acc": acc,
                "n": n,
                "prompt_tok": prompt,
                "gen_tok": gen,
                "total_tok": prompt + gen,
            }
        )
    if pd is not None:
        return pd.DataFrame(results)
    return results
