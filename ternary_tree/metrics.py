"""
metrics.py
==========
Ternary-specific evaluation metrics.

Standard metrics (accuracy, f1) treat all predictions equally.
Ternary metrics separate DECIDED instances from UNDECIDED instances,
allowing measurement of the accuracy-vs-coverage tradeoff.

Key metrics
-----------
decided_rate(ternary_verdict)
    Fraction of instances that received a DECIDED verdict (1).

undecided_rate(ternary_verdict)
    Fraction of instances that received an UNDECIDED verdict (0).

decided_accuracy(y_true, y_pred, ternary_verdict)
    Accuracy computed ONLY on instances the classifier decided on.
    The core quality metric for ternary classifiers:
    high decided_accuracy + meaningful undecided_rate = good system.

decided_f1(y_true, y_pred, ternary_verdict, average)
    F1 score on decided instances only.

ternary_summary(y_true, y_pred, ternary_verdict)
    Returns a dict with all metrics in one call.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def decided_rate(ternary_verdict: np.ndarray) -> float:
    """Fraction of instances with DECIDED verdict (1)."""
    return float(np.mean(ternary_verdict == 1))


def undecided_rate(ternary_verdict: np.ndarray) -> float:
    """Fraction of instances with UNDECIDED verdict (0)."""
    return float(np.mean(ternary_verdict == 0))


def decided_accuracy(
    y_true         : np.ndarray,
    y_pred         : np.ndarray,
    ternary_verdict: np.ndarray,
) -> float:
    """Accuracy on instances where classifier issued a DECIDED verdict.

    Returns NaN if no instances are decided.
    """
    mask = ternary_verdict == 1
    if mask.sum() == 0:
        return float("nan")
    return float(accuracy_score(y_true[mask], y_pred[mask]))


def decided_f1(
    y_true         : np.ndarray,
    y_pred         : np.ndarray,
    ternary_verdict: np.ndarray,
    average        : str = "macro",
) -> float:
    """F1 score on instances where classifier issued a DECIDED verdict."""
    mask = ternary_verdict == 1
    if mask.sum() == 0:
        return float("nan")
    return float(f1_score(y_true[mask], y_pred[mask], average=average, zero_division=0))


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard accuracy on all instances (undecided included)."""
    return float(accuracy_score(y_true, y_pred))


def ternary_summary(
    y_true         : np.ndarray,
    y_pred         : np.ndarray,
    ternary_verdict: np.ndarray,
    average        : str = "macro",
) -> Dict[str, float]:
    """All ternary metrics in one dict."""
    return {
        "accuracy_all"     : overall_accuracy(y_true, y_pred),
        "decided_rate"     : decided_rate(ternary_verdict),
        "undecided_rate"   : undecided_rate(ternary_verdict),
        "decided_accuracy" : decided_accuracy(y_true, y_pred, ternary_verdict),
        "decided_f1"       : decided_f1(y_true, y_pred, ternary_verdict, average),
        "f1_all"           : float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
