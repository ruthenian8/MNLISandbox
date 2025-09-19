#!/usr/bin/env python3
"""Run a self-contained groundedness benchmarking experiment.

The original project requires large datasets (SNLI, MNLI, ChaosNLI,
Flickr30k) and multimodal checkpoints.  Those resources are not available in
this execution environment, so this script fabricates a tiny synthetic corpus
and walks through the complete pipeline using the lightweight fallbacks shipped
with the repository.  When pandas is unavailable the script writes JSON payloads
with the expected filenames so that every intermediate artifact is still
inspectable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import sys

try:  # optional dependency for nicer tabular outputs
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nli_groundedness.binning import add_bins
from nli_groundedness.config import ensure_dirs
from nli_groundedness.groundedness import aggregate_sentence_groundedness
from nli_groundedness.reason_eval import map_labels, run_reasoning
from nli_groundedness.roberta_regressor import train_eval
from nli_groundedness.st_eval import run_st_logreg
from nli_groundedness.stats import per_bin_metrics
from nli_groundedness.vlm_scorer import CaptionerAndLM

ARTIFACTS = Path("artifacts")
REASONING_DIR = ARTIFACTS / "reasoning_runs"
REGRESSOR_DIR = ARTIFACTS / "regressor"
RELEASE_DIR = Path("release")


def _to_records(data) -> List[Dict[str, object]]:
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if isinstance(data, list):
        return [dict(row) for row in data]
    if data is None:
        return []
    return [dict(data)]


def _write_parquet(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    records = _to_records(rows)
    ensure_dirs(path.parent)
    if pd is not None:
        pd.DataFrame(records).to_parquet(path, index=False)  # type: ignore[arg-type]
    else:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    records = _to_records(rows)
    ensure_dirs(path.parent)
    if pd is not None:
        pd.DataFrame(records).to_csv(path, index=False)  # type: ignore[arg-type]
        return
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in records:
            handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")


def _demo_snli() -> List[Dict[str, str]]:
    return [
        {
            "pair_id": "demo-train-1#0",
            "split": "train",
            "premise": "A man rides a bicycle down the street while people watch.",
            "hypothesis": "A person is biking outside.",
            "gold_label": "entailment",
        },
        {
            "pair_id": "demo-train-2#0",
            "split": "train",
            "premise": "Two kids are playing soccer on a sunny field.",
            "hypothesis": "Children kick a ball outdoors.",
            "gold_label": "entailment",
        },
        {
            "pair_id": "demo-train-3#0",
            "split": "train",
            "premise": "A woman is reading a book in a quiet library.",
            "hypothesis": "The woman is running a marathon.",
            "gold_label": "contradiction",
        },
        {
            "pair_id": "demo-dev-1#0",
            "split": "dev",
            "premise": "A dog leaps over a fallen log in the forest.",
            "hypothesis": "An animal is jumping.",
            "gold_label": "entailment",
        },
        {
            "pair_id": "demo-test-1#0",
            "split": "test",
            "premise": "People sit at a table eating dinner together.",
            "hypothesis": "Friends share a meal.",
            "gold_label": "entailment",
        },
    ]


def _demo_chaos(snli_rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    entropy_map: Dict[str, float] = {
        "demo-train-1#0": 0.15,
        "demo-train-2#0": 0.10,
        "demo-train-3#0": 0.55,
        "demo-dev-1#0": 0.25,
        "demo-test-1#0": 0.30,
    }
    disagreement_map: Dict[str, float] = {
        pid: max(0.0, 0.6 - idx * 0.1)
        for idx, pid in enumerate(entropy_map.keys())
    }
    return [
        {
            "pair_id": row["pair_id"],
            "chaos_entropy": entropy_map.get(row["pair_id"], 0.2),
            "snli_disagreement": disagreement_map.get(row["pair_id"], 0.2),
        }
        for row in snli_rows
    ]


def _demo_mnli() -> List[Dict[str, str]]:
    return [
        {
            "pair_id": "demo-mnli-1",
            "split": "dev_matched",
            "premise": "A chef prepares pasta in a busy kitchen.",
            "hypothesis": "Someone is cooking dinner.",
            "gold_label": "entailment",
        },
        {
            "pair_id": "demo-mnli-2",
            "split": "dev_mismatched",
            "premise": "A scientist writes notes on a chalkboard.",
            "hypothesis": "The scientist is taking a nap.",
            "gold_label": "contradiction",
        },
        {
            "pair_id": "demo-mnli-3",
            "split": "dev_matched",
            "premise": "Crowds gather near the river to watch fireworks.",
            "hypothesis": "People are near water during a celebration.",
            "gold_label": "entailment",
        },
        {
            "pair_id": "demo-mnli-4",
            "split": "dev_mismatched",
            "premise": "A commuter misses the morning train.",
            "hypothesis": "Someone is waiting on the platform.",
            "gold_label": "neutral",
        },
        {
            "pair_id": "demo-mnli-5",
            "split": "dev_matched",
            "premise": "A guitarist performs on stage under bright lights.",
            "hypothesis": "A musician is entertaining a crowd.",
            "gold_label": "entailment",
        },
    ]


def _serialise(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def compute_groundedness(snli_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    scorer = CaptionerAndLM(cap_ckpt="stub")
    tokenizer = scorer.processor.tokenizer

    records: List[Dict[str, object]] = []
    for row in snli_rows:
        ids_txt, lp_txt = scorer.token_logprobs_text_only(row["premise"])
        ids_cap, lp_cap = scorer.token_logprobs_captioner(None, row["premise"])
        if getattr(ids_txt, "tolist", None) and getattr(ids_cap, "tolist", None):
            if ids_txt.tolist() != ids_cap.tolist():
                raise ValueError("token ids differ between text-only and caption passes")
        result = aggregate_sentence_groundedness(
            lp_cap,
            lp_txt,
            ids_txt,
            tokenizer,
            upos=None,
            aggregate="content_mean",
        )
        records.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "premise": row["premise"],
                "G_sentence_mean": result.G_sentence_mean,
                "G_sentence_uc": result.G_sentence_uc,
                "word_scores": _serialise(result.word_scores),
                "word_uncertainty": _serialise(result.word_uncertainty),
                "token_diagnostics": _serialise([asdict(diag) for diag in result.token_diagnostics]),
            }
        )
    _write_parquet(ARTIFACTS / "snli_groundedness.parquet", records)
    return records


def join_chaos(
    grounded: List[Dict[str, object]], chaos: List[Dict[str, float]]
) -> List[Dict[str, object]]:
    chaos_map = {row["pair_id"]: row for row in chaos}
    merged = [
        {
            **row,
            **chaos_map.get(row["pair_id"], {}),
        }
        for row in grounded
    ]
    _write_parquet(ARTIFACTS / "snli_chaos_join.parquet", merged)
    return merged


def run_sentence_transformer(snli_rows: List[Dict[str, str]]) -> None:
    train = [row for row in snli_rows if row["split"] == "train"]
    test = [row for row in snli_rows if row["split"] == "test"]
    if not test:
        test = [row for row in snli_rows if row["split"] == "dev"]
    scored = run_st_logreg(train, test, model_name="heuristic")
    _write_parquet(ARTIFACTS / "snli_st_eval.parquet", _to_records(scored))


def train_regressor(grounded: List[Dict[str, object]]) -> List[Dict[str, object]]:
    dev = [row for row in grounded if row["split"] == "dev"]
    train_rows = [row for row in grounded if row["split"] == "train"]
    preds = train_eval(
        [row["premise"] for row in train_rows],
        [float(row["G_sentence_mean"]) for row in train_rows],
        [row["premise"] for row in dev],
    )
    result = [
        {"pair_id": row["pair_id"], "pred": float(pred)}
        for row, pred in zip(dev, preds)
    ]
    _write_parquet(REGRESSOR_DIR / "snli_dev_preds_mean_of_seeds.parquet", result)
    return result


def predict_mnli(
    grounded: List[Dict[str, object]], mnli_rows: List[Dict[str, str]]
) -> List[Dict[str, object]]:
    preds = train_eval(
        [row["premise"] for row in grounded],
        [float(row["G_sentence_mean"]) for row in grounded],
        [row["premise"] for row in mnli_rows],
    )
    enriched = [
        {**row, "G_pred": float(pred)}
        for row, pred in zip(mnli_rows, preds)
    ]
    binned, edges = add_bins(enriched, "G_pred", num_bins=5)
    records = _to_records(binned)
    _write_parquet(ARTIFACTS / "mnli_pred_groundedness.parquet", records)
    print(f"MNLI groundedness bin edges: {edges}")
    return records


def update_release_slice(mnli_binned: List[Dict[str, object]]) -> None:
    ensure_dirs(RELEASE_DIR)
    sorted_rows = sorted(mnli_binned, key=lambda row: row.get("G_pred", 0.0))
    count = max(1, math.ceil(0.2 * len(sorted_rows)))
    subset = sorted_rows[:count]
    text = "\n".join(row.get("pair_id", "") for row in subset)
    (RELEASE_DIR / "mnli_lowg_ids.txt").write_text(text + "\n", encoding="utf-8")


def run_reasoning_eval(mnli_binned: List[Dict[str, object]]) -> None:
    prompt = (
        "You are an NLI classifier. Decide among entailment, contradiction, neutral.\n"
        "Premise: {premise}\nHypothesis: {hypothesis}\nAnswer with one word."
    )
    preds = run_reasoning(
        mnli_binned,
        model_name="stub-heuristic",
        prompt_tmpl=prompt,
        max_new_tokens=8,
        temperature=0.0,
    )
    mapped = map_labels(
        preds,
        {"entailment": "entailment", "contradiction": "contradiction", "neutral": "neutral"},
    )
    mapped_records = _to_records(mapped)
    merged = []
    for row in mnli_binned:
        match = next((m for m in mapped_records if m.get("pair_id") == row["pair_id"]), None)
        merged.append({**row, **(match or {})})
    _write_parquet(REASONING_DIR / "preds_stub-heuristic.parquet", merged)
    metrics = per_bin_metrics(merged)
    metrics_records = _to_records(metrics)
    _write_csv(REASONING_DIR / "metrics_stub-heuristic.csv", metrics_records)
    print("Reasoning per-bin metrics:\n", metrics_records)


def main() -> None:
    ensure_dirs(ARTIFACTS, REASONING_DIR, REGRESSOR_DIR)
    snli = _demo_snli()
    chaos = _demo_chaos(snli)
    mnli = _demo_mnli()

    _write_parquet(ARTIFACTS / "snli_w_images.parquet", snli)
    grounded = compute_groundedness(snli)
    join_chaos(grounded, chaos)
    run_sentence_transformer(snli)
    train_regressor(grounded)
    mnli_binned = predict_mnli(grounded, mnli)
    update_release_slice(mnli_binned)
    run_reasoning_eval(mnli_binned)
    print("Synthetic groundedness experiment finished. Artifacts written to", ARTIFACTS.resolve())


if __name__ == "__main__":
    main()
