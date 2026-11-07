from __future__ import annotations

import re
from collections import Counter


def normalize(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s\$\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize(prediction) == normalize(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def generation_correct(
    prediction: str,
    ground_truth: str,
    answerable: bool,
    abstained: bool,
    threshold: float = 0.55,
) -> bool:
    if not answerable:
        return abstained
    if abstained:
        return False
    gold_norm = normalize(ground_truth)
    pred_norm = normalize(prediction)
    if not gold_norm:
        return False
    if gold_norm in pred_norm:
        return True
    return token_f1(prediction, ground_truth) >= threshold


def classify_failure(
    *,
    answerable: bool,
    retrieval_hit: bool,
    oracle_ok: bool,
    rag_ok: bool,
    abstained: bool,
) -> str:
    """Diagnose the root cause of failures across the pipeline."""
    if not answerable:
        return "correct_abstain" if abstained else "should_have_abstained"
    if abstained:
        return "over_abstain"
    if rag_ok:
        return "pass"
    if not retrieval_hit:
        return "retrieval_failure"
    if not oracle_ok:
        return "generation_failure"
    return "context_use_failure"
