"""Text processing utilities including SQuAD answer normalization and scoring."""

import re
import string
from collections import Counter


def normalize(s: str) -> str:
    """Normalize answer string according to SQuAD official rules.

    Lowercase, remove punctuation, remove articles (a/an/the), collapse whitespace.
    """
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
    s = ' '.join(s.split())
    return s


def exact_match(pred: str, golds: list[str]) -> float:
    """Calculate exact match score.

    Returns 1.0 if normalized prediction matches any normalized gold answer, else 0.0.
    """
    norm_pred = normalize(pred)
    for gold in golds:
        if norm_pred == normalize(gold):
            return 1.0
    return 0.0


def f1(pred: str, golds: list[str]) -> float:
    """Calculate F1 score as defined by SQuAD.

    Returns max token-overlap F1 score over all gold answers.
    """
    pred_tokens = normalize(pred).split()
    best_f1 = 0.0

    for gold in golds:
        gold_tokens = normalize(gold).split()
        common_tokens = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common_tokens.values())

        if len(pred_tokens) == 0 and len(gold_tokens) == 0:
            f1_score = 1.0
        elif len(pred_tokens) == 0 or len(gold_tokens) == 0:
            f1_score = 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(gold_tokens)
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

        best_f1 = max(best_f1, f1_score)

    return best_f1


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion of multiple ranked lists.

    Combines multiple ranked lists by summing reciprocal ranks (1/(k+rank))
    and returns ids sorted by fused score in descending order.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids
