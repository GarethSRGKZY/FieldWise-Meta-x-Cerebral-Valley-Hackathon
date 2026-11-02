"""Confidence-based routing between Glimmer (local) and Spark (cloud).

The rule is deliberately simple so the benchmark can replay it exactly:

    confidence >= threshold          -> keep the local answer
    below threshold and online       -> escalate to Spark
    below threshold and offline      -> queue for Spark, show the local
                                        answer marked tentative + referral

Thresholds are tuned on the dev split only (bench/analyze.py).
"""
from __future__ import annotations

from enum import Enum


class Route(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    QUEUED = "queued"
    LOCAL_FALLBACK = "local_fallback"   # Glimmer answered with low confidence; the cloud call failed
    UNAVAILABLE = "unavailable"         # Glimmer not running AND Spark unreachable: no model answered


def decide(confidence: float, threshold: float, online: bool) -> Route:
    if confidence >= threshold:
        return Route.LOCAL
    return Route.CLOUD if online else Route.QUEUED


def qa_confidence(p_chosen: float, coverage: float, cited_ok: bool, complete: bool, full_coverage_at: float = 0.8) -> float:
    """Confidence for a text answer.

    p_chosen: probability Glimmer gave its own SUPPORTED verdict (logprobs)
    coverage: share of question words found in the best guide passage
    """
    if not complete:
        return 0.0
    conf = p_chosen * min(1.0, coverage / full_coverage_at)
    if not cited_ok:
        conf *= 0.5
    return max(0.0, min(1.0, conf))


def dx_confidence(p_chosen: float, chose_unsure: bool, complete: bool) -> float:
    if not complete or chose_unsure:
        return 0.0
    return max(0.0, min(1.0, p_chosen))
