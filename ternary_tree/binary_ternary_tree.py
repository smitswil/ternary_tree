"""
binary_ternary_tree.py
======================
BinaryTernaryTree: a CART-style decision tree whose PREDICTION phase is
ternary.

Training phase
--------------
Identical to standard CART — examples route left if x ≤ θ, right if x > θ.
During fitting, δ is computed at each node using the chosen delta method and
stored in the SplitNode.  δ does not affect which child receives which
training examples.

Prediction phase
----------------
At each node, an example is routed based on both θ and δ:

  x ≤ θ - δ  →  clear False  →  go left (standard routing)
  x > θ + δ  →  clear True   →  go right (standard routing)
  θ-δ < x ≤ θ+δ  →  Undecided zone  →  routing_method determines behaviour

Two routing methods:

  'probabilistic'
      Recurse into BOTH children with distance-based weights:
          w_left  = (θ + δ - x) / (2δ)
          w_right = (x - (θ - δ)) / (2δ)
      Each child's probability vector is weighted and summed.
      Uncertainty compounds down the tree — a sample that is undecided at
      multiple nodes gets progressively smeared across leaves.

  'deferred'
      Get predictions from left and right children using HARD binary routing
      (no further ternary routing), then combine:
          proba = w_left × hard_left_proba + w_right × hard_right_proba
      Uncertainty is resolved in one step — subsequent nodes use crisp
      thresholds regardless of further undecided zones.

Ternary output
--------------
predict_ternary(X) returns int8 array:
  1  = DECIDED  — sample traversed no undecided zone
  0  = UNDECIDED — sample entered at least one undecided zone

predict() and predict_proba() work for all instances (undecided samples
still get a class prediction via weighted routing).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .delta_methods import compute_delta, DELTA_METHODS
from .node import LeafNode, NodeType, SplitNode
from .splitter import (
    compute_class_probabilities,
    compute_sample_weight,
    evaluate_all_thresholds,
    find_best_split,
    is_pure,
)

ROUTING_METHODS = frozenset({"probabilistic", "deferred"})


class BinaryTernaryTree(BaseEstimator, ClassifierMixin):
    """Decision tree with binary CART training and ternary prediction routing.

    Parameters
    ----------
    delta_method : str
        How to compute the per-node Undecided zone half-width δ.
        One of: 'quality_plateau', 'class_overlap', 'gain_ratio', 'node_bootstrap'.

    routing_method : str
        How Undecided-zone examples are handled during prediction.
        One of: 'probabilistic', 'deferred'.

    criterion : str
        Split quality criterion.  'gini' (default) or 'entropy'.

    max_depth : int or None
        Maximum tree depth.

    min_samples_split : int
        Minimum samples to split a node.

    min_samples_leaf : int
        Minimum samples in each child after a split.

    epsilon : float
        Plateau tolerance for delta_method='quality_plateau'.

    percentile : float
        Class range percentile for delta_method='class_overlap'.

    alpha : float
        Scale factor for delta_method='gain_ratio'.

    n_bootstraps : int
        Bootstrap replicates for delta_method='node_bootstrap'.

    class_weight : None, 'balanced', or dict
        Per-class sample weights.

    random_state : int or None
    """

    def __init__(
        self,
        delta_method     : str   = "quality_plateau",
        routing_method   : str   = "probabilistic",
        criterion        : str   = "gini",
        max_depth        : Optional[int] = None,
        min_samples_split: int   = 2,
        min_samples_leaf : int   = 1,
        epsilon          : float = 0.05,
        percentile       : float = 10.0,
        alpha            : float = 0.10,
        n_bootstraps     : int   = 20,
        class_weight     : object = None,
        random_state     : Optional[int] = None,
    ) -> None:
        self.delta_method      = delta_method
        self.routing_method    = routing_method
        self.criterion         = criterion
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.epsilon           = epsilon
        self.percentile        = percentile
        self.alpha             = alpha
        self.n_bootstraps      = n_bootstraps
        self.class_weight      = class_weight
        self.random_state      = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryTernaryTree":
        X, y = check_X_y(X, y)
        X    = X.astype(float)

        if self.delta_method not in DELTA_METHODS:
            raise ValueError(f"Unknown delta_method '{self.delta_method}'")
        if self.routing_method not in ROUTING_METHODS:
            raise ValueError(f"Unknown routing_method '{self.routing_method}'")

        self.classes_       = np.unique(y)
        self.n_classes_     = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self._rng           = np.random.default_rng(self.random_state)

        sample_weight = compute_sample_weight(y, self.class_weight, self.classes_)

        self.tree_ = self._build(X, y, sample_weight, depth=0)
        return self

    def _build(
        self,
        X            : np.ndarray,
        y            : np.ndarray,
        sample_weight: np.ndarray,
        depth        : int,
    ) -> NodeType:
        n = len(y)

        # Leaf conditions
        max_depth_reached = (self.max_depth is not None and depth >= self.max_depth)
        too_small         = n < self.min_samples_split
        pure              = is_pure(y)

        if max_depth_reached or too_small or pure:
            return self._make_leaf(y, sample_weight)

        # Find best split
        feat, theta, score, cands, scores, sinfos = find_best_split(
            X, y, sample_weight, self.classes_, self.criterion
        )

        if feat == -1 or score <= 0:
            return self._make_leaf(y, sample_weight)

        col     = X[:, feat]
        best_idx= int(np.argmax(scores)) if len(scores) > 0 else 0

        # Compute δ
        delta = compute_delta(
            method=self.delta_method,
            col=col, y=y, sample_weight=sample_weight,
            classes=self.classes_,
            candidates=cands, scores=scores, split_infos=sinfos,
            best_score=score, best_idx=best_idx,
            criterion=self.criterion,
            epsilon=self.epsilon,
            percentile=self.percentile,
            alpha=self.alpha,
            n_bootstraps=self.n_bootstraps,
            rng=self._rng,
        )

        # Standard binary split for training (δ not used here)
        left_mask  = col <= theta
        right_mask = ~left_mask

        if (left_mask.sum() < self.min_samples_leaf or
                right_mask.sum() < self.min_samples_leaf):
            return self._make_leaf(y, sample_weight)

        # Compute parent impurity for node record
        _, _, parent_imp, _ = evaluate_all_thresholds(
            col, y, sample_weight, self.classes_, self.criterion
        )

        left_node  = self._build(X[left_mask],  y[left_mask],  sample_weight[left_mask],  depth + 1)
        right_node = self._build(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1)

        return SplitNode(
            feature_idx=feat,
            threshold=theta,
            delta=delta,
            left=left_node,
            right=right_node,
            n_samples=n,
            impurity=parent_imp,
        )

    def _make_leaf(self, y: np.ndarray, sample_weight: np.ndarray) -> LeafNode:
        proba    = compute_class_probabilities(y, sample_weight, self.classes_)
        pred_idx = int(np.argmax(proba))
        return LeafNode(
            class_proba=proba,
            predicted_class=int(self.classes_[pred_idx]),
            n_samples=len(y),
        )

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X).astype(float)
        self._check_n_features(X)
        return np.vstack([self._predict_proba_one(self.tree_, x) for x in X])

    def _predict_proba_one(self, node: NodeType, x: np.ndarray) -> np.ndarray:
        if isinstance(node, LeafNode):
            return node.class_proba

        val   = x[node.feature_idx]
        theta = node.threshold
        delta = node.delta

        if val <= theta - delta:
            return self._predict_proba_one(node.left, x)
        if val > theta + delta:
            return self._predict_proba_one(node.right, x)

        # Undecided zone — compute distance-based weights
        w_left, w_right = _undecided_weights(val, theta, delta)

        if self.routing_method == "probabilistic":
            # Recurse into both children with ternary routing
            lp = self._predict_proba_one(node.left,  x)
            rp = self._predict_proba_one(node.right, x)
        else:  # deferred
            # Children use hard (binary) routing only
            lp = self._predict_proba_hard(node.left,  x)
            rp = self._predict_proba_hard(node.right, x)

        return w_left * lp + w_right * rp

    def _predict_proba_hard(self, node: NodeType, x: np.ndarray) -> np.ndarray:
        """Standard binary routing (no ternary treatment)."""
        while isinstance(node, SplitNode):
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.class_proba  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba     = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return self.classes_[class_idx]

    # ------------------------------------------------------------------
    # predict_ternary
    # ------------------------------------------------------------------

    def predict_ternary(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (DECIDED) or 0 (UNDECIDED) for each instance."""
        check_is_fitted(self)
        X = check_array(X).astype(float)
        self._check_n_features(X)
        return np.array(
            [self._is_decided(self.tree_, x) for x in X], dtype=np.int8
        )

    def _is_decided(self, node: NodeType, x: np.ndarray) -> int:
        """Return 1 if sample traverses no undecided zone, 0 otherwise."""
        while isinstance(node, SplitNode):
            val   = x[node.feature_idx]
            theta = node.threshold
            delta = node.delta

            if val <= theta - delta:
                node = node.left
            elif val > theta + delta:
                node = node.right
            else:
                return 0  # Undecided zone hit
        return 1

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_depth(self) -> int:
        check_is_fitted(self)
        return _tree_depth(self.tree_)

    def get_n_leaves(self) -> int:
        check_is_fitted(self)
        return _count_leaves(self.tree_)

    def feature_importances(self) -> np.ndarray:
        """Impurity-based feature importances, shape (n_features,)."""
        check_is_fitted(self)
        importances = np.zeros(self.n_features_in_)
        _accumulate_importances(self.tree_, importances)
        total = importances.sum()
        return importances / total if total > 0 else importances

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_n_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}."
            )

    def __repr__(self) -> str:
        fitted = hasattr(self, "tree_")
        return (
            f"BinaryTernaryTree("
            f"delta={self.delta_method!r}, "
            f"routing={self.routing_method!r}, "
            f"depth={self.get_depth() if fitted else '?'})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _undecided_weights(val: float, theta: float, delta: float):
    """Distance-based weights for left and right children in undecided zone."""
    if delta <= 0:
        return 0.5, 0.5
    w_left  = (theta + delta - val) / (2.0 * delta)
    w_right = (val - (theta - delta)) / (2.0 * delta)
    # Clamp for floating point safety
    w_left  = float(np.clip(w_left,  0.0, 1.0))
    w_right = float(np.clip(w_right, 0.0, 1.0))
    total   = w_left + w_right
    if total > 0:
        w_left  /= total
        w_right /= total
    return w_left, w_right


def _tree_depth(node: NodeType) -> int:
    if isinstance(node, LeafNode):
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _count_leaves(node: NodeType) -> int:
    if isinstance(node, LeafNode):
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)


def _accumulate_importances(node: NodeType, importances: np.ndarray) -> None:
    if isinstance(node, LeafNode):
        return
    n     = node.n_samples
    imp   = node.impurity
    nl    = node.left.n_samples   if isinstance(node.left,  SplitNode) else node.left.n_samples
    nr    = node.right.n_samples  if isinstance(node.right, SplitNode) else node.right.n_samples
    imp_l = node.left.impurity    if isinstance(node.left,  SplitNode) else 0.0
    imp_r = node.right.impurity   if isinstance(node.right, SplitNode) else 0.0
    importances[node.feature_idx] += (
        n * imp - nl * imp_l - nr * imp_r
    )
    _accumulate_importances(node.left,  importances)
    _accumulate_importances(node.right, importances)
