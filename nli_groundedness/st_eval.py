"""Sentence Transformer baseline with a heuristic fallback."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


_LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _ensure_records(data):
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if hasattr(data, "rows"):
        return list(data.rows)
    if isinstance(data, dict):
        return [dict(data)]
    return [dict(row) for row in data]


def _heuristic_predict(premise: str, hypothesis: str) -> str:
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


def run_st_logreg(df_train, df_test, model_name: str = "heuristic"):
    """Heuristic approximation of the production ST baseline."""

    records = []
    for row in _ensure_records(df_test):
        pred = _heuristic_predict(row.get("premise", ""), row.get("hypothesis", ""))
        gold = row.get("gold_label", "")
        records.append(
            {
                **row,
                "st_pred_label": pred,
                "st_pred": _LABEL2ID.get(pred, -1),
                "st_is_error": int(pred != gold),
            }
        )
    if pd is not None:
        return pd.DataFrame(records)
    return records
