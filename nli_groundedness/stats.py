"""Statistics helpers for groundedness analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

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


def _rankdata(values: Sequence[float]) -> List[float]:
    items = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(items):
        j = i
        total = 0.0
        while j < len(items) and items[j][0] == items[i][0]:
            total += j + 1
            j += 1
        avg_rank = total / (j - i)
        for k in range(i, j):
            ranks[items[k][1]] = avg_rank
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)
    denom = (den_x * den_y) ** 0.5
    if denom == 0:
        return 0.0
    return num / denom


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def corr_table(df, gcol: str = "G_sentence_mean"):
    rows = _ensure_iterable(df)
    targets: Dict[str, List[float]] = defaultdict(list)
    grounded: List[float] = []
    for row in rows:
        g = row.get(gcol)
        if g is None:
            continue
        for col in ("snli_disagreement", "chaos_entropy", "st_is_error"):
            value = row.get(col)
            if value is None:
                continue
            grounded.append(float(g))
            targets[col].append(float(value))
    results = {}
    for col, values in targets.items():
        if not values or len(values) != len(grounded):
            continue
        rho = _spearman(grounded, values)
        results[col] = {"spearman_rho": rho, "p": None, "n": len(values)}
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
