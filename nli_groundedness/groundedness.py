"""Groundedness aggregation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional dependency
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "PRON"}
SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<bos>", "<eos>"}


@dataclass
class TokenDiagnostic:
    token: str
    token_index: int
    word_index: Optional[int]
    pmi: float
    logp_captioner: float
    logp_text_only: float
    surprisal_captioner: float
    surprisal_text_only: float


@dataclass
class GroundednessResult:
    tokens: List[str]
    token_diagnostics: List[TokenDiagnostic]
    word_scores: Dict[int, float]
    word_uncertainty: Dict[int, float]
    G_sentence_mean: float
    G_sentence_uc: float


def _to_list(values: Sequence[float] | Iterable[float]) -> List[float]:
    if isinstance(values, list):
        return values
    if _np is not None and hasattr(values, "tolist"):
        return list(values.tolist())  # pragma: no cover - exercised when numpy available
    return [float(v) for v in values]


def _flatten_ids(input_ids) -> List[int]:
    if hasattr(input_ids, "tolist"):
        data = input_ids.tolist()
    else:
        data = list(input_ids)
    if len(data) == 0:
        return []
    if isinstance(data[0], list):  # flatten 2D arrays (batch of size 1)
        return [int(x) for x in data[0]]
    return [int(x) for x in data]


def align_wordpieces_to_words(tokens: Sequence[str]) -> List[Optional[int]]:
    word_ids: List[Optional[int]] = []
    current = -1
    prev_special = True
    for tok in tokens:
        is_special = tok in SPECIAL_TOKENS
        if is_special:
            word_ids.append(None)
            prev_special = True
            continue
        if tok.startswith("▁") or tok.startswith("Ġ") or tok.startswith("<w>"):
            current += 1
        elif tok.startswith("##"):
            pass
        elif prev_special:
            current += 1
        else:
            if tok and tok[0].isalpha():
                pass
            else:
                current += 1
        word_ids.append(current)
        prev_special = False
    return word_ids


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _trimmed_mean(values: List[float], proportion_to_cut: float = 0.1) -> float:
    if not values:
        return 0.0
    if proportion_to_cut <= 0:
        return _mean(values)
    ordered = sorted(values)
    n = len(ordered)
    k = int(max(0, min(n // 2, math.floor(n * proportion_to_cut))))
    if k == 0:
        return _mean(ordered)
    trimmed = ordered[k : n - k]
    if not trimmed:
        return _mean(ordered)
    return _mean(trimmed)


def _content_word_ids(upos: Optional[Sequence[str]], word_scores: Dict[int, float]) -> List[int]:
    if upos is None:
        return list(word_scores.keys())
    ids: List[int] = []
    for idx, tag in enumerate(upos):
        if tag in CONTENT_POS and idx in word_scores:
            ids.append(idx)
    if not ids:
        ids = [idx for idx in range(len(upos)) if idx in word_scores]
    return ids


def aggregate_sentence_groundedness(
    logprobs_captioner: Sequence[float],
    logprobs_text_only: Sequence[float],
    input_ids,
    tokenizer,
    upos: Optional[Sequence[str]] = None,
    aggregate: str = "content_mean",
) -> GroundednessResult:
    """Aggregate token level PMI scores into a sentence groundedness value."""

    cap = _to_list(logprobs_captioner)
    txt = _to_list(logprobs_text_only)
    ids = _flatten_ids(input_ids)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = tokenizer.convert_ids_to_tokens(ids)
    elif hasattr(tokenizer, "decode") and hasattr(tokenizer, "token_to_string"):
        tokens = [tokenizer.token_to_string(t) for t in ids]
    else:
        tokens = [str(t) for t in ids]

    # token level metrics skip BOS token alignment
    word_ids = align_wordpieces_to_words(tokens)
    token_diagnostics: List[TokenDiagnostic] = []
    word_scores: Dict[int, List[float]] = {}
    word_unc: Dict[int, List[float]] = {}

    usable = min(len(cap), len(txt), len(tokens) - 1)
    for offset in range(usable):
        token_index = offset + 1  # skip BOS
        word_index = word_ids[token_index] if token_index < len(word_ids) else None
        token = tokens[token_index] if token_index < len(tokens) else ""
        lp_cap = float(cap[offset])
        lp_txt = float(txt[offset])
        pmi = lp_cap - lp_txt
        surprisal_cap = -lp_cap
        surprisal_txt = -lp_txt
        token_diagnostics.append(
            TokenDiagnostic(
                token=token,
                token_index=token_index,
                word_index=word_index,
                pmi=pmi,
                logp_captioner=lp_cap,
                logp_text_only=lp_txt,
                surprisal_captioner=surprisal_cap,
                surprisal_text_only=surprisal_txt,
            )
        )
        if word_index is None:
            continue
        word_scores.setdefault(word_index, []).append(pmi)
        diff = surprisal_txt - surprisal_cap
        denom = abs(surprisal_txt) + 1e-6
        word_unc.setdefault(word_index, []).append(diff / denom)

    word_mean_scores = {idx: _mean(vals) for idx, vals in word_scores.items()}
    word_uncertainty = {idx: _mean(vals) for idx, vals in word_unc.items()}

    selected_ids = _content_word_ids(upos, word_mean_scores)
    selected_scores = [word_mean_scores[i] for i in selected_ids if i in word_mean_scores]
    selected_unc = [word_uncertainty.get(i, 0.0) for i in selected_ids]

    if aggregate == "content_mean":
        sentence_score = _mean(selected_scores)
    elif aggregate == "content_median":
        sentence_score = _median(selected_scores)
    elif aggregate == "trimmed_mean_10":
        sentence_score = _trimmed_mean(selected_scores, 0.1)
    else:
        raise ValueError(f"Unknown aggregation strategy: {aggregate}")

    sentence_uc = _mean(selected_unc)

    return GroundednessResult(
        tokens=tokens,
        token_diagnostics=token_diagnostics,
        word_scores={idx: float(score) for idx, score in word_mean_scores.items()},
        word_uncertainty={idx: float(score) for idx, score in word_uncertainty.items()},
        G_sentence_mean=float(sentence_score),
        G_sentence_uc=float(sentence_uc),
    )
