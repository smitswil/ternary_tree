"""
delta_methods.py
================
Four methods for computing the node-local Undecided-zone half-width δ.

Each method takes the data available at the current node after the best
split has been found, and returns a non-negative float δ.

The four methods
----------------
1. quality_plateau
   Reads the plateau of the split quality curve. δ = half the width of the
   region where Q(θ) ≥ (1 - ε) × Q(θ*).  Zero extra computation — the
   quality curve is already evaluated by find_best_split.

2. class_overlap
   Empirical class-distribution overlap. δ = half the width of the feature
   range where both (or all) class percentile bands intersect.

3. gain_ratio
   δ = α × feature_range / (1 + GR) where GR = IG / H_split.
   High gain ratio → confident split → small δ.
   Low gain ratio  → uncertain split → large δ.

4. node_bootstrap
   Bootstrap the node data B times, find the optimal threshold each time,
   set δ = std(optimal_thresholds).  Most statistically principled;
   most expensive (but still fast for small node sizes).

All methods clamp δ to [0, max_fraction × feature_range] to prevent
degenerate cases where δ would encompass the whole feature range.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .splitter import evaluate_all_thresholds


# Maximum fraction of feature range that δ can occupy
_MAX_DELTA_FRACTION = 0.25


def _clamp_delta(delta: float, col: np.ndarray) -> float:
    feature_range = float(col.max() - col.min())
    if feature_range <= 0:
        return 0.0
    return float(np.clip(delta, 0.0, _MAX_DELTA_FRACTION * feature_range))


# ---------------------------------------------------------------------------
# Method 1: Quality Plateau Margin
# ---------------------------------------------------------------------------

def delta_quality_plateau(
    candidates  : np.ndarray,
    scores      : np.ndarray,
    best_score  : float,
    col         : np.ndarray,
    epsilon     : float = 0.05,
) -> float:
    """Half-width of the quality plateau around the optimal threshold.

    δ = (θ_hi - θ_lo) / 2  where Q(θ) ≥ (1 - ε) × Q(θ*)

    Parameters
    ----------
    candidates : candidate thresholds from evaluate_all_thresholds
    scores     : quality gains at each candidate threshold
    best_score : quality at the chosen optimal threshold
    col        : feature column at this node (for range clamping)
    epsilon    : tolerance; larger → wider plateau → larger δ
    """
    if best_score <= 0 or len(candidates) < 2:
        return 0.0

    quality_threshold = (1.0 - epsilon) * best_score
    mask              = scores >= quality_threshold
    if not np.any(mask):
        return 0.0

    plateau_cands = candidates[mask]
    delta = (plateau_cands[-1] - plateau_cands[0]) / 2.0
    return _clamp_delta(delta, col)


# ---------------------------------------------------------------------------
# Method 2: Class Overlap Width
# ---------------------------------------------------------------------------

def delta_class_overlap(
    col         : np.ndarray,
    y           : np.ndarray,
    classes     : np.ndarray,
    percentile  : float = 10.0,
) -> float:
    """Half-width of the empirical class-distribution overlap at this node.

    For each class, compute the [percentile, 100-percentile] range of the
    feature.  δ = half the width of the intersection of all class ranges.

    For multiclass, we use the widest pairwise overlap (most conservative).

    Parameters
    ----------
    col        : feature column at this node
    y          : class labels at this node
    classes    : all unique classes (for consistency)
    percentile : inner percentile bound (e.g. 10 → use 10th–90th pct range)
    """
    present_classes = [c for c in classes if np.any(y == c)]
    if len(present_classes) < 2:
        return 0.0

    # Per-class [lo, hi] ranges
    ranges = {}
    for c in present_classes:
        col_c = col[y == c]
        if len(col_c) < 2:
            ranges[c] = (float(col_c.min()), float(col_c.max()))
        else:
            ranges[c] = (
                float(np.percentile(col_c, percentile)),
                float(np.percentile(col_c, 100.0 - percentile)),
            )

    # Widest pairwise overlap
    best_overlap = 0.0
    cls_list = list(ranges.keys())
    for i in range(len(cls_list)):
        for j in range(i + 1, len(cls_list)):
            lo_i, hi_i = ranges[cls_list[i]]
            lo_j, hi_j = ranges[cls_list[j]]
            overlap_lo = max(lo_i, lo_j)
            overlap_hi = min(hi_i, hi_j)
            if overlap_hi > overlap_lo:
                best_overlap = max(best_overlap, overlap_hi - overlap_lo)

    delta = best_overlap / 2.0
    return _clamp_delta(delta, col)


# ---------------------------------------------------------------------------
# Method 3: Gain Ratio Inverse
# ---------------------------------------------------------------------------

def delta_gain_ratio(
    col         : np.ndarray,
    best_score  : float,
    best_split_info: float,
    alpha       : float = 0.10,
) -> float:
    """δ = α × feature_range / (1 + GR).

    GR = information_gain / split_entropy.
    High GR → clear split → small δ.
    Low GR  → ambiguous split → large δ.

    Parameters
    ----------
    col            : feature column at this node
    best_score     : quality gain at the chosen threshold
    best_split_info: split information entropy H(A) at the chosen threshold
    alpha          : global scale (fraction of range at GR=0)
    """
    feature_range = float(col.max() - col.min())
    if feature_range <= 0:
        return 0.0

    if best_split_info <= 0:
        # Degenerate split (all examples on one side) → maximally uncertain
        return _clamp_delta(alpha * feature_range, col)

    gain_ratio = best_score / best_split_info
    delta      = alpha * feature_range / (1.0 + gain_ratio)
    return _clamp_delta(delta, col)


# ---------------------------------------------------------------------------
# Method 4: Node Bootstrap
# ---------------------------------------------------------------------------

def delta_node_bootstrap(
    col           : np.ndarray,
    y             : np.ndarray,
    sample_weight : np.ndarray,
    classes       : np.ndarray,
    candidates    : np.ndarray,
    criterion     : str = "gini",
    n_bootstraps  : int = 20,
    rng           : Optional[np.random.Generator] = None,
) -> float:
    """δ = std(optimal threshold across B bootstrap resamples of node data).

    The most statistically principled method. Cost is ~n_bootstraps × CART
    per-node cost, which is manageable for typical node sizes.

    Parameters
    ----------
    col           : feature column at this node
    y             : class labels at this node
    sample_weight : per-sample weights at this node
    classes       : sorted unique class labels
    candidates    : candidate thresholds from evaluate_all_thresholds
    criterion     : 'gini' or 'entropy'
    n_bootstraps  : number of bootstrap replicates
    rng           : numpy random Generator (created if None)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(col)
    if n < 10 or len(candidates) == 0:
        return 0.0

    boot_thresholds = []

    for _ in range(n_bootstraps):
        idx   = rng.integers(0, n, size=n)
        col_b = col[idx]
        y_b   = y[idx]
        sw_b  = sample_weight[idx]

        _, scrs_b, _, _ = evaluate_all_thresholds(
            col_b, y_b, sw_b, classes, criterion
        )

        if len(scrs_b) == 0:
            continue

        # Find the threshold in the original candidates closest to each
        # bootstrap-optimal threshold (ensures comparability)
        cands_b, _, _, _ = evaluate_all_thresholds(
            col_b, y_b, sw_b, classes, criterion
        )
        if len(cands_b) == 0:
            continue

        best_idx = int(np.argmax(scrs_b))
        boot_thresholds.append(float(cands_b[best_idx]))

    if len(boot_thresholds) < 2:
        return 0.0

    delta = float(np.std(boot_thresholds))
    return _clamp_delta(delta, col)


# ---------------------------------------------------------------------------
# Method 5: Margin-Based (SVM-inspired)
# ---------------------------------------------------------------------------

def delta_margin(
    col    : np.ndarray,
    y      : np.ndarray,
    theta  : float,
    classes: np.ndarray,
) -> float:
    """δ = distance from θ* to the nearest cross-class example on either side.

    Directly analogous to the SVM margin.  After the optimal threshold θ* has
    been chosen, examine which training examples of each class sit closest to
    the boundary but on the "wrong" side — i.e. the examples that most challenge
    the chosen split.

    Algorithm
    ---------
    1. Identify the dominant class on each side of θ* (left_class, right_class).
    2. Find cross-class intrusions:
         right_intrusions: left_class examples that appear to the RIGHT of θ*
         left_intrusions : right_class examples that appear to the LEFT of θ*
    3. δ = distance from θ* to the nearest intrusion on either side.
    4. If the split is perfectly clean (no intrusions), fall back to the
       physical gap: (nearest_right - nearest_left) / 2.
       A large physical gap means few examples constrain the boundary —
       the boundary could reasonably be placed anywhere in that gap.

    No hyperparameters.  The margin is read directly from the data.

    Parameters
    ----------
    col     : feature column at this node (unsorted)
    y       : class labels at this node
    theta   : the chosen optimal split threshold θ*
    classes : sorted unique class labels
    """
    left_mask  = col <= theta
    right_mask = ~left_mask

    n_left  = int(left_mask.sum())
    n_right = int(right_mask.sum())

    if n_left == 0 or n_right == 0:
        return 0.0

    # Dominant class on each side (by count)
    y_left  = y[left_mask]
    y_right = y[right_mask]

    unique_left,  counts_left  = np.unique(y_left,  return_counts=True)
    unique_right, counts_right = np.unique(y_right, return_counts=True)

    left_dominant  = unique_left[np.argmax(counts_left)]
    right_dominant = unique_right[np.argmax(counts_right)]

    if left_dominant == right_dominant:
        # Same class dominates both sides — split is not informative.
        # Fall back to physical gap / 2.
        left_gap  = theta - float(col[left_mask].max())
        right_gap = float(col[right_mask].min()) - theta
        return _clamp_delta(min(left_gap, right_gap), col)

    # Cross-class intrusions — the "support vectors"
    # left_dominant examples that crossed to the right side of θ*
    right_intrusions = col[right_mask & (y == left_dominant)]
    # right_dominant examples that crossed to the left side of θ*
    left_intrusions  = col[left_mask  & (y == right_dominant)]

    margins = []
    if len(right_intrusions) > 0:
        # Nearest left_dominant example sitting right of theta
        margins.append(float(right_intrusions.min()) - theta)
    if len(left_intrusions) > 0:
        # Nearest right_dominant example sitting left of theta
        margins.append(theta - float(left_intrusions.max()))

    if not margins:
        # Perfect separation — use the physical gap between adjacent examples
        left_gap  = theta - float(col[left_mask].max())
        right_gap = float(col[right_mask].min()) - theta
        delta = min(left_gap, right_gap)
    else:
        delta = min(margins)

    return _clamp_delta(max(0.0, delta), col)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

DELTA_METHODS = frozenset({
    "quality_plateau",
    "class_overlap",
    "gain_ratio",
    "node_bootstrap",
    "margin",
})


def compute_delta(
    method        : str,
    col           : np.ndarray,
    y             : np.ndarray,
    sample_weight : np.ndarray,
    classes       : np.ndarray,
    candidates    : np.ndarray,
    scores        : np.ndarray,
    split_infos   : np.ndarray,
    best_score    : float,
    best_idx      : int,
    criterion     : str,
    # Method-specific hyperparameters
    epsilon       : float = 0.05,
    percentile    : float = 10.0,
    alpha         : float = 0.10,
    n_bootstraps  : int   = 20,
    rng           : Optional[np.random.Generator] = None,
) -> float:
    """Unified dispatcher for all five delta methods.

    Parameters
    ----------
    method        : one of DELTA_METHODS
    col           : feature column at this node
    y, sample_weight, classes : data at this node
    candidates, scores, split_infos : from evaluate_all_thresholds on col
    best_score    : quality at optimal threshold
    best_idx      : index of optimal threshold in candidates
    criterion     : split criterion used
    epsilon       : plateau tolerance (method 1)
    percentile    : class percentile bound (method 2)
    alpha         : gain ratio scale (method 3)
    n_bootstraps  : bootstrap replicates (method 4)
    rng           : random generator (method 4)
    (method 5 margin has no additional hyperparameters)
    """
    if method == "quality_plateau":
        return delta_quality_plateau(candidates, scores, best_score, col, epsilon)

    if method == "class_overlap":
        return delta_class_overlap(col, y, classes, percentile)

    if method == "gain_ratio":
        best_split_info = float(split_infos[best_idx]) if len(split_infos) > 0 else 0.0
        return delta_gain_ratio(col, best_score, best_split_info, alpha)

    if method == "node_bootstrap":
        return delta_node_bootstrap(
            col, y, sample_weight, classes, candidates,
            criterion, n_bootstraps, rng,
        )

    if method == "margin":
        best_theta = float(candidates[best_idx]) if len(candidates) > best_idx else 0.0
        return delta_margin(col, y, best_theta, classes)

    raise ValueError(
        f"Unknown delta method '{method}'. Choose from {sorted(DELTA_METHODS)}"
    )
