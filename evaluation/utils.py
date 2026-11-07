from __future__ import annotations

import re
from .metrics import classify_failure, exact_match, generation_correct, normalize, token_f1


def fuzzy_match(answer: str, pred: str, threshold: float = 0.55) -> bool:
    """Fuzzy matching helper utilizing token F1 or substring containment."""
    if not answer:
        return not pred.strip()
    norm_a = normalize(answer)
    norm_p = normalize(pred)
    if norm_a in norm_p or norm_p in norm_a:
        return True
    return token_f1(answer, pred) >= threshold
