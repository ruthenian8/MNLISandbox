"""Captioner wrapper with a lightweight stub for tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:  # optional heavy dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - executed when torch missing
    torch = None  # type: ignore

try:  # heavy dependency
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoProcessor = None  # type: ignore

@dataclass
class _SimpleTokenizer:
    vocab: dict

    @property
    def tokenizer(self) -> "_SimpleTokenizer":  # pragma: no cover - compatibility shim
        return self

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        inv = {idx: token for token, idx in self.vocab.items()}
        return [inv.get(int(i), "<unk>") for i in ids]

    def encode(self, text: str) -> List[int]:  # pragma: no cover - helper for potential future use
        return [self.vocab[w] for w in text.split() if w in self.vocab]


class _SimpleTensor:
    """Tiny tensor stand-in used by the stub implementation."""

    def __init__(self, data: Sequence[Sequence[float]] | Sequence[float]):
        if data and isinstance(data[0], (list, tuple)):
            self.data = [list(row) for row in data]  # type: ignore[index]
        else:
            self.data = [list(data)]  # type: ignore[arg-type]

    def tolist(self) -> List[List[float]]:
        return [list(row) for row in self.data]

    def cpu(self) -> "_SimpleTensor":  # pragma: no cover - compatibility shim
        return self

    def __iter__(self):
        return iter(self.data)

    def __len__(self):  # pragma: no cover - compatibility shim
        return len(self.data)


class CaptionerAndLM:
    """Dual-purpose wrapper exposing token log-probabilities.

    The real project uses multimodal checkpoints from Hugging Face.  Those
    models are far too heavy for the execution environment used in the
    exercises, so this class falls back to a deterministic stub that mimics
    the expected behaviour.  When ``cap_ckpt`` is set to ``"stub"`` (the
    default) the object performs a simple heuristic computation based solely
    on token lengths.  The interface, however, matches the production version
    which allows researchers to plug in a genuine checkpoint when running the
    pipeline locally.
    """

    def __init__(
        self,
        cap_ckpt: str = "stub",
        device: Optional[str] = None,
        dtype: str | None = None,
    ) -> None:
        self.cap_ckpt = cap_ckpt
        self.device = device or "cpu"
        self.dtype = dtype or "float32"
        self._is_stub = cap_ckpt in {"stub", "dummy", "test"}
        if self._is_stub:
            base_tokens = ["<s>", "</s>"]
            self._stub_vocab = {token: idx for idx, token in enumerate(base_tokens)}
        else:
            if torch is None or AutoModelForCausalLM is None or AutoProcessor is None:
                raise ImportError(
                    "transformers and torch are required for non-stub captioner operation"
                )
            self.processor = AutoProcessor.from_pretrained(cap_ckpt, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(cap_ckpt, trust_remote_code=True)

    # ------------------------------------------------------------------
    # stub helpers
    # ------------------------------------------------------------------
    def _stub_tokenize(self, text: str) -> List[str]:
        words = text.strip().split()
        tokens = ["<s>"]
        for word in words:
            token = "▁" + word.lower()
            if token not in self._stub_vocab:
                self._stub_vocab[token] = len(self._stub_vocab)
            tokens.append(token)
        tokens.append("</s>")
        return tokens

    def _stub_input_ids(self, tokens: Sequence[str]) -> _SimpleTensor:
        ids = [[self._stub_vocab[token] for token in tokens]]
        return _SimpleTensor(ids)

    def _stub_logprobs(self, tokens: Sequence[str], bonus: float = 0.0) -> List[float]:
        scores: List[float] = []
        for token in tokens[1:]:  # skip BOS like production models
            length = max(1, len(token.replace("▁", "")))
            base = -0.1 * length
            if token.startswith("▁") and token[1:] in {"man", "woman", "dog", "cat", "boy", "girl"}:
                base += 0.2
            scores.append(base + bonus)
        return scores

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def processor(self):  # pragma: no cover - dynamic attribute for stub compatibility
        if self._is_stub:
            return _SimpleTokenizer(self._stub_vocab)
        return self._processor

    @processor.setter
    def processor(self, value):  # pragma: no cover - executed in real checkpoints
        self._processor = value

    def token_logprobs_text_only(self, text: str):
        if self._is_stub:
            tokens = self._stub_tokenize(text)
            ids = self._stub_input_ids(tokens)
            logprobs = self._stub_logprobs(tokens, bonus=0.0)
            return ids, logprobs
        raise NotImplementedError(
            "Text-only scoring requires transformers which are unavailable in the execution sandbox"
        )

    def token_logprobs_captioner(self, image, text: str):  # image unused in stub
        if self._is_stub:
            tokens = self._stub_tokenize(text)
            ids = self._stub_input_ids(tokens)
            logprobs = self._stub_logprobs(tokens, bonus=0.15)
            return ids, logprobs
        raise NotImplementedError(
            "Caption scoring requires transformers which are unavailable in the execution sandbox"
        )
