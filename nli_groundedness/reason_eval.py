"""Reasoning model evaluation helpers."""

from __future__ import annotations

from typing import Dict

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _ensure_records(data):
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if hasattr(data, "rows"):
        return list(data.rows)
    if isinstance(data, dict):
        return [dict(data)]
    return [dict(row) for row in data]


def _heuristic_label(premise: str, hypothesis: str) -> str:
    prem = (premise or "").lower()
    hypo = (hypothesis or "").lower()
    if "not" in prem and "not" not in hypo:
        return "contradiction"
    if "not" in hypo and "not" not in prem:
        return "contradiction"
    if hypo and hypo in prem:
        return "entailment"
    if prem and prem in hypo:
        return "entailment"
    return "neutral"


def run_reasoning(
    df,
    model_name: str,
    prompt_tmpl: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: str | None = None,
):
    """Return heuristic reasoning results.

    The function mirrors the production signature but performs no actual model
    calls.  Each example is labelled using simple lexical heuristics and token
    counts are estimated via whitespace tokenisation.
    """

    rows = []
    for row in _ensure_records(df):
        pred = _heuristic_label(row.get("premise", ""), row.get("hypothesis", ""))
        prompt = prompt_tmpl.format(premise=row.get("premise", ""), hypothesis=row.get("hypothesis", ""))
        prompt_tokens = len(prompt.split())
        rows.append(
            {
                "pair_id": row.get("pair_id", ""),
                "bin": row.get("bin"),
                "pred_text": pred,
                "n_prompt_tok": prompt_tokens,
                "n_gen_tok": 1,
            }
        )
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def map_labels(df_preds, label_regex: Dict[str, str]):
    records = []
    for row in _ensure_records(df_preds):
        records.append({**row, "pred_label": str(row.get("pred_text", "")).strip().lower()})
    if pd is not None:
        return pd.DataFrame(records)
    return records
