"""Lightweight text and hierarchical ranking helpers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def normalize_model_text(text: Any) -> str:
    """Match training's whitespace normalization without legacy abbreviation expansion."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def joint_topk(
    probs2: np.ndarray,
    probs4: np.ndarray,
    four_to_two_ids: np.ndarray,
    k: int = 5,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    joint = probs4 * np.power(probs2[four_to_two_ids], alpha)
    top_ids = np.argsort(-joint)[:k]
    return top_ids, probs4[top_ids]
