"""
trinary_tree.py
===============
TrinaryTree: a decision tree with THREE children per node.

Training phase
--------------
After finding the best (feature, θ) and computing δ:

    left  branch  ← examples where x ≤ θ - δ  (clear False)
    middle branch ← examples where θ - δ < x ≤ θ + δ  (Undecided zone)
    right branch  ← examples where x > θ + δ  (clear True)

Each branch is a full subtree trained recursively on its own subset.
The middle subtree learns to classify examples that genuinely sit in the
boundary region — it can use a different feature for its own splits.

If the middle zone has fewer than min_samples_middle examples, it becomes a
leaf node using the class distribution of the examples in that zone (or of
the full node if the zone is empty).

Prediction phase
----------------
Hard deterministic routing:
  x ≤ θ - δ       →  go left
  θ - δ < x ≤ θ+δ →  go middle  [marks the instance as UNDECIDED]
  x > θ + δ        →  go right

predict_ternary(X) returns:
  1  = DECIDED  — sample never routed to a middle branch
  0  = UNDECIDED — sample routed through at least one middle branch
"""

from __future__ import annotations

from typing import Optional

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


class TrinaryTree(BaseEstimator, ClassifierMixin):
    """Decision tree with a genuine third (middle) branch at each node.

    Parameters
    ----------
    delta_method : str
        How to compute δ.  One of DELTA_METHODS.

    criterion : str
        'gini' or 'entropy'.

    max_depth : int or None

    min_samples_split : int

    min_samples_leaf : int

    min_samples_middle : int
        Minimum examples in the undecided zone to build a middle subtree.
        If fewer examples are in the zone, a leaf is created instead.
        Default 5.

    epsilon, percentile, alpha, n_bootstraps
        Hyperparameters passed to the chosen delta method.

    class_weight : None, 'balanced', or dict

    random_state : int or None
    """

    def __init__(
        self,
        delta_method       : str   = "quality_plateau",
        criterion          : str   = "gini",
        max_depth          : Optional[int] = None,
        min_samples_split  : int   = 2,
        min_samples_leaf   : int   = 1,
        min_samples_middle : int   = 5,
        epsilon            : float = 0.05,
        percentile         : float = 10.0,
        alpha              : float = 0.10,
        n_bootstraps       : int   = 20,
        class_weight       : object = None,
        random_state       : Optional[int] = None,
    ) -> None:
        self.delta_method       = delta_method
        self.criterion          = criterion
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.min_samples_middle = min_samples_middle
        self.epsilon            = epsilon
        self.percentile         = percentile
        self.alpha              = alpha
        self.n_bootstraps       = n_bootstraps
        self.class_weight       = class_weight
        self.random_state       = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrinaryTree":
        X, y = check_X_y(X, y)
        X    = X.astype(float)

        if self.delta_method not in DELTA_METHODS:
            raise ValueError(f"Unknown delta_method '{self.delta_method}'")

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

        max_depth_reached = (self.max_depth is not None and depth >= self.max_depth)
        too_small         = n < self.min_samples_split
        pure              = is_pure(y)

        if max_depth_reached or too_small or pure:
            return self._make_leaf(y, sample_weight, is_middle=False)

        # Find best split
        feat, theta, score, cands, scores, sinfos = find_best_split(
            X, y, sample_weight, self.classes_, self.criterion
        )

        if feat == -1 or score <= 0:
            return self._make_leaf(y, sample_weight, is_middle=False)

        col      = X[:, feat]
        best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0

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

        # Three-way partition for TRAINING
        left_mask   = col <= theta - delta
        middle_mask = (col > theta - delta) & (col <= theta + delta)
        right_mask  = col > theta + delta

        # Fall back if left or right is too small
        if (left_mask.sum() < self.min_samples_leaf or
                right_mask.sum() < self.min_samples_leaf):
            return self._make_leaf(y, sample_weight, is_middle=False)

        # Compute parent impurity
        _, _, parent_imp, _ = evaluate_all_thresholds(
            col, y, sample_weight, self.classes_, self.criterion
        )

        # Build left and right subtrees normally
        left_node  = self._build(
            X[left_mask],  y[left_mask],  sample_weight[left_mask],  depth + 1
        )
        right_node = self._build(
            X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1
        )

        # Build middle subtree or leaf
        n_middle = int(middle_mask.sum())
        if n_middle >= self.min_samples_middle:
            middle_node = self._build(
                X[middle_mask], y[middle_mask], sample_weight[middle_mask], depth + 1
            )
            # Mark the root of the middle subtree as a middle leaf if it is a leaf
            if isinstance(middle_node, LeafNode):
                middle_node.is_middle_leaf = True
        else:
            # Not enough examples → leaf from middle-zone distribution
            if n_middle > 0:
                middle_node = self._make_leaf(
                    y[middle_mask], sample_weight[middle_mask], is_middle=True
                )
            else:
                # Undecided zone is empty → use full node distribution
                middle_node = self._make_leaf(y, sample_weight, is_middle=True)

        return SplitNode(
            feature_idx=feat,
            threshold=theta,
            delta=delta,
            left=left_node,
            middle=middle_node,
            right=right_node,
            n_samples=n,
            impurity=parent_imp,
        )

    def _make_leaf(
        self, y: np.ndarray, sample_weight: np.ndarray, is_middle: bool
    ) -> LeafNode:
        proba    = compute_class_probabilities(y, sample_weight, self.classes_)
        pred_idx = int(np.argmax(proba))
        return LeafNode(
            class_proba=proba,
            predicted_class=int(self.classes_[pred_idx]),
            n_samples=len(y),
            is_middle_leaf=is_middle,
        )

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X).astype(float)
        self._check_n_features(X)
        return np.vstack([self._proba_one(self.tree_, x) for x in X])

    def _proba_one(self, node: NodeType, x: np.ndarray) -> np.ndarray:
        while isinstance(node, SplitNode):
            val   = x[node.feature_idx]
            theta = node.threshold
            delta = node.delta

            if val <= theta - delta:
                node = node.left
            elif val > theta + delta:
                node = node.right
            else:
                node = node.middle  # type: ignore[assignment]
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
        """Return 1 (DECIDED) or 0 (UNDECIDED) per instance.

        An instance is UNDECIDED if it was ever routed to a middle branch.
        """
        check_is_fitted(self)
        X = check_array(X).astype(float)
        self._check_n_features(X)
        return np.array(
            [self._is_decided(self.tree_, x) for x in X], dtype=np.int8
        )

    def _is_decided(self, node: NodeType, x: np.ndarray) -> int:
        """1 if sample never hits a middle branch, 0 otherwise."""
        went_middle = False
        while isinstance(node, SplitNode):
            val   = x[node.feature_idx]
            theta = node.threshold
            delta = node.delta

            if val <= theta - delta:
                node = node.left
            elif val > theta + delta:
                node = node.right
            else:
                went_middle = True
                node = node.middle  # type: ignore[assignment]

        return 0 if went_middle else 1

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_depth(self) -> int:
        check_is_fitted(self)
        return _tree_depth(self.tree_)

    def get_n_leaves(self) -> int:
        check_is_fitted(self)
        return _count_leaves(self.tree_)

    def count_middle_leaves(self) -> int:
        check_is_fitted(self)
        return _count_middle_leaves(self.tree_)

    def feature_importances(self) -> np.ndarray:
        check_is_fitted(self)
        importances = np.zeros(self.n_features_in_)
        _accumulate_importances(self.tree_, importances)
        total = importances.sum()
        return importances / total if total > 0 else importances

    def _check_n_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}."
            )

    def __repr__(self) -> str:
        fitted = hasattr(self, "tree_")
        return (
            f"TrinaryTree("
            f"delta={self.delta_method!r}, "
            f"routing='hard_middle', "
            f"depth={self.get_depth() if fitted else '?'})"
        )


# ---------------------------------------------------------------------------
# Tree utility functions
# ---------------------------------------------------------------------------

def _tree_depth(node: NodeType) -> int:
    if isinstance(node, LeafNode):
        return 0
    children = [node.left, node.right]
    if node.middle is not None:
        children.append(node.middle)
    return 1 + max(_tree_depth(c) for c in children)


def _count_leaves(node: NodeType) -> int:
    if isinstance(node, LeafNode):
        return 1
    count = _count_leaves(node.left) + _count_leaves(node.right)
    if node.middle is not None:
        count += _count_leaves(node.middle)
    return count


def _count_middle_leaves(node: NodeType) -> int:
    if isinstance(node, LeafNode):
        return 1 if node.is_middle_leaf else 0
    count = _count_middle_leaves(node.left) + _count_middle_leaves(node.right)
    if node.middle is not None:
        count += _count_middle_leaves(node.middle)
    return count


def _accumulate_importances(node: NodeType, importances: np.ndarray) -> None:
    if isinstance(node, LeafNode):
        return
    n     = node.n_samples
    imp   = node.impurity
    nl    = node.left.n_samples
    nr    = node.right.n_samples
    imp_l = node.left.impurity   if isinstance(node.left,  SplitNode) else 0.0
    imp_r = node.right.impurity  if isinstance(node.right, SplitNode) else 0.0
    importances[node.feature_idx] += n * imp - nl * imp_l - nr * imp_r
    _accumulate_importances(node.left,  importances)
    _accumulate_importances(node.right, importances)
    if node.middle is not None:
        _accumulate_importances(node.middle, importances)
