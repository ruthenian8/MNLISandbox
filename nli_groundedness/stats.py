"""Statistics helpers for groundedness analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import scipy.stats  # type: ignore
except Exception:  # pragma: no cover
    scipy = None  # type: ignore
else:
    scipy = scipy.stats


def _ensure_iterable(df):
    if pd is not None and isinstance(df, pd.DataFrame):
        return df.to_dict("records")
    if hasattr(df, "rows"):
        return list(df.rows)
    return list(df)


def _rankdata(values: Sequence[float]) -> List[float]:
    """Compute ranks with scipy if available, otherwise fall back to custom implementation."""
    if scipy is not None:
        return scipy.rankdata(values, method='average').tolist()
    
    # Fallback implementation for when scipy is not available
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
    """Compute Pearson correlation with scipy if available, otherwise fall back to custom implementation."""
    if len(x) != len(y) or not x:
        return 0.0
    
    if scipy is not None:
        try:
            corr, _ = scipy.pearsonr(x, y)
            return corr if not pd.isna(corr) else 0.0
        except Exception:
            # Fall back to custom implementation if scipy fails
            pass
    
    # Fallback implementation for when scipy is not available or fails
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
    """Compute Spearman correlation with scipy if available, otherwise fall back to custom implementation."""
    if len(x) != len(y) or not x:
        return 0.0
    
    if scipy is not None:
        try:
            corr, _ = scipy.spearmanr(x, y)
            return corr if not pd.isna(corr) else 0.0
        except Exception:
            # Fall back to custom implementation if scipy fails
            pass
    
    # Fallback implementation using Pearson correlation on ranks
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
        
        # Use scipy for both correlation and p-value if available
        if scipy is not None:
            try:
                rho, p_value = scipy.spearmanr(grounded, values)
                if pd.isna(rho):
                    rho = 0.0
                if pd.isna(p_value):
                    p_value = None
                results[col] = {"spearman_rho": rho, "p": p_value, "n": len(values)}
                continue
            except Exception:
                # Fall back to custom implementation
                pass
        
        # Fallback implementation without p-values
        rho = _spearman(grounded, values)
        results[col] = {"spearman_rho": rho, "p": None, "n": len(values)}
    
    if pd is not None:
        return pd.DataFrame.from_dict(results, orient="index")
    return results


def per_bin_metrics(df_bins, gold_col: str = "gold_label", pred_col: str = "pred_label"):
    """Compute per-bin metrics with pandas optimization if available."""
    if pd is not None and hasattr(df_bins, 'groupby'):
        # Use pandas groupby for more efficient computation
        try:
            df = df_bins.copy()
            df['correct'] = (df[gold_col] == df[pred_col]).astype(int)
            df['prompt_tok'] = df.get('n_prompt_tok', 0.0).astype(float)
            df['gen_tok'] = df.get('n_gen_tok', 0.0).astype(float)
            
            grouped = df.groupby('bin').agg({
                'correct': 'sum',
                'prompt_tok': 'mean',
                'gen_tok': 'mean',
                gold_col: 'count'  # for n
            }).rename(columns={gold_col: 'n'})
            
            grouped['acc'] = grouped['correct'] / grouped['n']
            grouped['total_tok'] = grouped['prompt_tok'] + grouped['gen_tok']
            
            # Reset index to make 'bin' a regular column
            result = grouped.reset_index()
            return result[['bin', 'acc', 'n', 'prompt_tok', 'gen_tok', 'total_tok']].to_dict('records')
        except Exception:
            # Fall back to original implementation if pandas optimization fails
            pass
    
    # Fallback implementation for when pandas is not available or optimization fails
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
