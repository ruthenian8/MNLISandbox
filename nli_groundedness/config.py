"""Project level configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Paths:
    """Container for canonical repository paths."""

    snli_dir: Path
    mnli_dir: Path
    chaos_dir: Path
    flickr_index_csv: Path
    artifacts_dir: Path
    figures_dir: Path

    @classmethod
    def from_dict(cls, data: dict) -> "Paths":
        return cls(
            snli_dir=Path(data["snli_dir"]),
            mnli_dir=Path(data["mnli_dir"]),
            chaos_dir=Path(data["chaos_dir"]),
            flickr_index_csv=Path(data["flickr_index_csv"]),
            artifacts_dir=Path(data["artifacts_dir"]),
            figures_dir=Path(data["figures_dir"]),
        )


def ensure_dirs(*paths: Iterable[Path] | Path) -> None:
    """Ensure that all provided directories exist."""

    for value in paths:
        if isinstance(value, Path):
            value.mkdir(parents=True, exist_ok=True)
            continue
        for item in value:  # type: ignore[iteration-over-annotation]
            ensure_dirs(item)
