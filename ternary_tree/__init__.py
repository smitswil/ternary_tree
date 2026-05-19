"""
ternary_tree
============
Ternary decision trees with locally-adaptive Undecided zones.

Quick start
-----------
>>> from ternary_tree import BinaryTernaryTree, TrinaryTree
>>> clf = BinaryTernaryTree(delta_method='quality_plateau', routing_method='probabilistic')
>>> clf.fit(X_train, y_train)
>>> clf.predict(X_test)
>>> clf.predict_ternary(X_test)   # 1=decided, 0=undecided
"""

from .binary_ternary_tree import BinaryTernaryTree
from .trinary_tree         import TrinaryTree
from .metrics              import ternary_summary, decided_accuracy, undecided_rate
from .delta_methods        import DELTA_METHODS

ROUTING_METHODS = frozenset({"probabilistic", "deferred", "hard_middle"})

__version__ = "0.1.0"

__all__ = [
    "BinaryTernaryTree",
    "TrinaryTree",
    "ternary_summary",
    "decided_accuracy",
    "undecided_rate",
    "DELTA_METHODS",
    "ROUTING_METHODS",
]

def make_classifier(delta_method: str, routing_method: str, **kwargs):
    """Factory returning the right tree class for a (delta, routing) pair.

    Parameters
    ----------
    delta_method   : 'quality_plateau' | 'class_overlap' | 'gain_ratio' | 'node_bootstrap'
    routing_method : 'probabilistic' | 'deferred' | 'hard_middle'
    **kwargs       : passed to the tree constructor
    """
    if routing_method in ("probabilistic", "deferred"):
        return BinaryTernaryTree(
            delta_method=delta_method, routing_method=routing_method, **kwargs
        )
    if routing_method == "hard_middle":
        return TrinaryTree(delta_method=delta_method, **kwargs)
    raise ValueError(f"Unknown routing_method '{routing_method}'")
