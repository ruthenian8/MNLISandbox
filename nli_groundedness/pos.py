"""Very small POS tagging helper."""

from __future__ import annotations

from functools import lru_cache
from typing import List

try:  # pragma: no cover - optional heavy dependency
    import stanza  # type: ignore
except Exception:  # pragma: no cover
    stanza = None  # type: ignore


class UDTagger:
    """Universal Dependencies part-of-speech tagger.

    When :mod:`stanza` is available the class defers to the official English
    models.  Otherwise a lightweight rule-based tagger is used which is
    perfectly adequate for unit tests and quick smoke checks.
    """

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._use_stub = stanza is None
        if stanza is not None:
            stanza.download(lang, processors="tokenize,pos", verbose=False)  # pragma: no cover
            self._nlp = stanza.Pipeline(
                lang=lang,
                processors="tokenize,pos",
                tokenize_no_ssplit=True,
                verbose=False,
            )

    @lru_cache(maxsize=1000)
    def upos(self, text: str) -> List[str]:
        if self._use_stub:
            return [_rule_based_tag(token) for token in text.split()]
        doc = self._nlp(text)  # pragma: no cover - executed with stanza installed
        return [word.upos for sent in doc.sentences for word in sent.words]  # pragma: no cover


def _rule_based_tag(token: str) -> str:
    lowered = token.lower()
    if lowered.endswith("ing") or lowered.endswith("ed"):
        return "VERB"
    if lowered in {"a", "an", "the"}:
        return "DET"
    if lowered in {"and", "or", "but"}:
        return "CONJ"
    if lowered in {"on", "in", "at", "with", "over"}:
        return "ADP"
    if lowered.isdigit():
        return "NUM"
    if lowered.endswith("ly"):
        return "ADV"
    if lowered in {"he", "she", "it", "they", "we", "i", "you"}:
        return "PRON"
    if lowered.endswith("ous") or lowered.endswith("ful") or lowered.endswith("ive"):
        return "ADJ"
    return "NOUN"
