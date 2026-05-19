"""
splitter.py
===========
Core split evaluation engine.

Provides O(n log n) evaluation of Gini gain and Information Gain across all
candidate thresholds for one feature using cumulative weighted class counts.
This is the standard CART histogram trick.  All four delta methods and both
tree classes use these functions — they are the single source of truth for
split quality.

Key functions
-------------
evaluate_all_thresholds(col, y, sample_weight, criterion)
    Returns (candidates, scores, parent_impurity, split_info) for one feature.

find_best_split(X, y, sample_weight, criterion, feature_indices)
    Searches across features; returns best (feature_idx, threshold, score,
    candidates_for_best_feature, scores_for_best_feature, split_info).

compute_class_probabilities(y, sample_weight, classes)
    Weighted class probability vector for a leaf node.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Impurity helpers
# ---------------------------------------------------------------------------

def _gini(probs: np.ndarray) -> float:
    return 1.0 - float(np.sum(probs ** 2))


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def _split_entropy(left_w: float, right_w: float, total_w: float) -> float:
    """Binary entropy of the split proportions (used for gain ratio)."""
    result = 0.0
    for w in (left_w, right_w):
        p = w / total_w
        if p > 0:
            result -= p * np.log2(p)
    return result


# ---------------------------------------------------------------------------
# Single-feature threshold evaluation  (O(n log n))
# ---------------------------------------------------------------------------

def evaluate_all_thresholds(
    col           : np.ndarray,
    y             : np.ndarray,
    sample_weight : np.ndarray,
    classes       : np.ndarray,
    criterion     : str = "gini",
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Evaluate split quality for every candidate threshold on one feature.

    Uses cumulative weighted class counts so cost is O(n log n) — the sort —
    rather than O(n²).

    Parameters
    ----------
    col           : feature values at this node, shape (n,)
    y             : class labels, shape (n,)
    sample_weight : per-sample weights, shape (n,)
    classes       : sorted unique class labels, shape (k,)
    criterion     : 'gini' or 'entropy'

    Returns
    -------
    candidates   : np.ndarray, shape (m,) — candidate split thresholds
    scores       : np.ndarray, shape (m,) — quality gain at each threshold
    parent_imp   : float — parent node impurity (Gini or entropy)
    split_infos  : np.ndarray, shape (m,) — split entropy H(A) at each threshold
                   (needed by the gain-ratio delta method)
    """
    n         = len(col)
    n_classes = len(classes)
    cls_idx   = {c: i for i, c in enumerate(classes)}

    # Sort everything by feature value
    sort_idx = np.argsort(col, kind="mergesort")
    col_s    = col[sort_idx]
    y_s      = y[sort_idx]
    sw_s     = sample_weight[sort_idx]

    # Build per-position weighted class contribution matrix, shape (n, k)
    wc = np.zeros((n, n_classes), dtype=np.float64)
    for i in range(n):
        wc[i, cls_idx[y_s[i]]] = sw_s[i]

    cum_wc    = np.cumsum(wc, axis=0)        # cumulative left class weights
    cum_total = np.cumsum(sw_s, dtype=float) # cumulative left total weight

    total_w     = float(sw_s.sum())
    total_cls_w = wc.sum(axis=0)             # total weight per class

    # Parent impurity
    parent_probs = total_cls_w / total_w
    if criterion == "gini":
        parent_imp = _gini(parent_probs)
    else:
        parent_imp = _entropy(parent_probs)

    candidates  : List[float] = []
    scores      : List[float] = []
    split_infos : List[float] = []

    for i in range(n - 1):
        if col_s[i] == col_s[i + 1]:
            continue  # No information at ties

        theta    = (col_s[i] + col_s[i + 1]) / 2.0
        left_w   = float(cum_total[i])
        right_w  = total_w - left_w

        if left_w < 1e-10 or right_w < 1e-10:
            continue

        left_cls_w  = cum_wc[i]
        right_cls_w = total_cls_w - left_cls_w

        lp = left_cls_w  / left_w
        rp = right_cls_w / right_w

        if criterion == "gini":
            left_imp  = _gini(lp)
            right_imp = _gini(rp)
        else:
            left_imp  = _entropy(lp)
            right_imp = _entropy(rp)

        gain = parent_imp - (
            left_w  / total_w * left_imp +
            right_w / total_w * right_imp
        )

        si = _split_entropy(left_w, right_w, total_w)

        candidates.append(theta)
        scores.append(gain)
        split_infos.append(si)

    return (
        np.array(candidates,  dtype=np.float64),
        np.array(scores,      dtype=np.float64),
        parent_imp,
        np.array(split_infos, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Best split search across all features
# ---------------------------------------------------------------------------

def find_best_split(
    X              : np.ndarray,
    y              : np.ndarray,
    sample_weight  : np.ndarray,
    classes        : np.ndarray,
    criterion      : str = "gini",
    feature_indices: Optional[np.ndarray] = None,
) -> Tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Find the best feature and threshold across all (or a subset of) features.

    Parameters
    ----------
    X              : shape (n, d)
    y              : shape (n,)
    sample_weight  : shape (n,)
    classes        : sorted unique class labels
    criterion      : 'gini' or 'entropy'
    feature_indices: which features to search; default all

    Returns
    -------
    best_feature  : int
    best_threshold: float
    best_score    : float
    candidates    : np.ndarray — thresholds for best feature
    scores        : np.ndarray — quality scores for best feature
    split_infos   : np.ndarray — split entropy for best feature
    """
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])

    best_feature   = -1
    best_threshold = 0.0
    best_score     = -np.inf
    best_cands     = np.array([])
    best_scores    = np.array([])
    best_sinfos    = np.array([])

    for j in feature_indices:
        col = X[:, j]
        cands, scrs, _, sinfos = evaluate_all_thresholds(
            col, y, sample_weight, classes, criterion
        )
        if len(scrs) == 0:
            continue
        idx = int(np.argmax(scrs))
        if scrs[idx] > best_score:
            best_score     = float(scrs[idx])
            best_feature   = int(j)
            best_threshold = float(cands[idx])
            best_cands     = cands
            best_scores    = scrs
            best_sinfos    = sinfos

    return (
        best_feature,
        best_threshold,
        best_score,
        best_cands,
        best_scores,
        best_sinfos,
    )


# ---------------------------------------------------------------------------
# Leaf utilities
# ---------------------------------------------------------------------------

def compute_class_probabilities(
    y             : np.ndarray,
    sample_weight : np.ndarray,
    classes       : np.ndarray,
) -> np.ndarray:
    """Weighted class probability vector for a leaf, shape (k,)."""
    total = float(sample_weight.sum())
    if total == 0:
        return np.ones(len(classes)) / len(classes)
    probs = np.array(
        [sample_weight[y == c].sum() for c in classes], dtype=np.float64
    )
    return probs / total


def is_pure(y: np.ndarray) -> bool:
    return len(np.unique(y)) <= 1


def compute_sample_weight(
    y            : np.ndarray,
    class_weight : object,
    classes      : np.ndarray,
) -> np.ndarray:
    """Translate class_weight spec into per-sample weights."""
    n = len(y)
    if class_weight is None:
        return np.ones(n, dtype=np.float64)

    if class_weight == "balanced":
        counts = np.array([np.sum(y == c) for c in classes], dtype=float)
        weights = n / (len(classes) * counts)
        weight_map = dict(zip(classes, weights))
    elif isinstance(class_weight, dict):
        weight_map = class_weight
    else:
        raise ValueError(f"Unknown class_weight: {class_weight!r}")

    return np.array([weight_map.get(yi, 1.0) for yi in y], dtype=np.float64)
