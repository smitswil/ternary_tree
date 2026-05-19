"""
node.py
=======
Node dataclasses shared by BinaryTernaryTree and TrinaryTree.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

NodeType = Union["SplitNode", "LeafNode"]

@dataclass
class LeafNode:
    class_proba    : np.ndarray
    predicted_class: int
    n_samples      : int
    is_middle_leaf : bool = False

    def __repr__(self):
        return (f"LeafNode(class={self.predicted_class}, "
                f"n={self.n_samples}, proba={self.class_proba.round(3)})")

@dataclass
class SplitNode:
    feature_idx : int
    threshold   : float
    delta       : float
    left        : NodeType
    right       : NodeType
    middle      : Optional[NodeType] = None
    n_samples   : int   = 0
    impurity    : float = 0.0

    def __repr__(self):
        return (f"SplitNode(feat={self.feature_idx}, "
                f"θ={self.threshold:.4g}, δ={self.delta:.4g}, "
                f"n={self.n_samples})")
