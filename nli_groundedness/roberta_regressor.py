"""Text-only groundedness regressor with a linear fallback."""

from __future__ import annotations

from statistics import mean
from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import scipy.stats  # type: ignore
except Exception:  # pragma: no cover
    scipy = None  # type: ignore
else:
    scipy = scipy.stats

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _linear_fit(xs: List[float], ys: List[float]) -> tuple[float, float]:
    """Perform linear regression with scipy if available, otherwise use custom implementation."""
    if not xs or not ys:
        return 0.0, _safe_mean(ys)
    
    if scipy is not None and np is not None:
        try:
            # Use scipy.stats.linregress for better numerical stability
            result = scipy.linregress(xs, ys)
            return result.slope, result.intercept
        except Exception:
            # Fall back to custom implementation if scipy fails
            pass
    
    # Fallback implementation for when scipy/numpy is not available or fails
    avg_x = _safe_mean(xs)
    avg_y = _safe_mean(ys)
    numerator = sum((x - avg_x) * (y - avg_y) for x, y in zip(xs, ys))
    denominator = sum((x - avg_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0, avg_y
    slope = numerator / denominator
    intercept = avg_y - slope * avg_x
    return slope, intercept


def train_eval(
    train_texts: Sequence[str],
    train_targets: Sequence[float],
    dev_texts: Sequence[str],
    base: str = "roberta-base",
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 32,
    seeds: Iterable[int] = range(1),
    device: str | None = None,
):
    """Return groundedness predictions using a simple linear model.

    The production system trains :class:`~transformers.RobertaModel`.  The
    execution environment for this kata is intentionally lightweight, so the
    helper implements a deterministic closed-form linear regression using text
    length as the sole feature.  The signature mirrors the real function which
    makes swapping in a proper implementation trivial.
    """

    # Use numpy for vectorized operations if available
    if np is not None:
        try:
            # Vectorized computation of text lengths
            train_lengths = np.array([float(len((text or "").split())) for text in train_texts])
            targets = np.array([float(y) for y in train_targets])
            dev_lengths = np.array([float(len((text or "").split())) for text in dev_texts])
            
            slope, intercept = _linear_fit(train_lengths.tolist(), targets.tolist())
            
            # Vectorized prediction
            predictions = intercept + slope * dev_lengths
            return predictions.tolist()
        except Exception:
            # Fall back to list-based implementation
            pass
    
    # Fallback implementation using lists
    train_lengths = [float(len((text or "").split())) for text in train_texts]
    targets = [float(y) for y in train_targets]
    slope, intercept = _linear_fit(train_lengths, targets)
    dev_lengths = [float(len((text or "").split())) for text in dev_texts]
    return [intercept + slope * length for length in dev_lengths]
