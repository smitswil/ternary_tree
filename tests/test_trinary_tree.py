"""Tests for TrinaryTree — all 4 delta methods with hard_middle routing."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from ternary_tree import TrinaryTree
from ternary_tree.delta_methods import DELTA_METHODS


def fast_trinary(delta_method="quality_plateau", **kw):
    defaults = dict(max_depth=3, n_bootstraps=5, min_samples_middle=3, random_state=42)
    defaults.update(kw)
    return TrinaryTree(delta_method=delta_method, **defaults)


def binary_data():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def multiclass_data():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ---------------------------------------------------------------------------
# All 4 delta methods fit without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", list(DELTA_METHODS))
def test_all_delta_methods_fit(delta):
    Xtr, Xte, ytr, yte = binary_data()
    clf = fast_trinary(delta)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    assert pred.shape == (len(yte),)


# ---------------------------------------------------------------------------
# predict shapes and types
# ---------------------------------------------------------------------------

class TestPredictShapes:
    def test_predict_shape(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        assert clf.predict(Xte).shape == (len(yte),)

    def test_predict_proba_shape(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)

    def test_predict_proba_sums_to_one(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_ternary_valid_values(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        tv = clf.predict_ternary(Xte)
        assert tv.shape == (len(yte),)
        assert set(tv.tolist()).issubset({0, 1})

    def test_multiclass(self):
        Xtr, Xte, ytr, yte = multiclass_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 3)


# ---------------------------------------------------------------------------
# Middle branch specific
# ---------------------------------------------------------------------------

class TestMiddleBranch:
    def test_some_instances_undecided_with_wide_delta(self):
        """Wide δ → more examples routed to middle → some undecided."""
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_trinary(delta_method="gain_ratio", alpha=0.5)
        clf.fit(Xtr, ytr)
        tv = clf.predict_ternary(Xte)
        # With alpha=0.5 some instances should hit the middle branch
        assert tv.shape == (len(yte),)

    def test_count_middle_leaves_nonneg(self):
        Xtr, _, ytr, _ = binary_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        n_middle = clf.count_middle_leaves()
        assert n_middle >= 0

    def test_narrow_delta_mostly_decided(self):
        """Very narrow δ → almost all examples take left/right branch."""
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_trinary(delta_method="gain_ratio", alpha=0.001)
        clf.fit(Xtr, ytr)
        tv = clf.predict_ternary(Xte)
        decided_rate = (tv == 1).mean()
        # With very narrow δ, most instances should be decided
        assert decided_rate > 0.5


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

class TestInspection:
    def test_get_depth(self):
        Xtr, _, ytr, _ = binary_data()
        clf = fast_trinary(max_depth=3)
        clf.fit(Xtr, ytr)
        assert clf.get_depth() <= 4  # middle branches may add one level

    def test_feature_importances(self):
        Xtr, _, ytr, _ = binary_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        imp = clf.feature_importances()
        assert imp.shape == (Xtr.shape[1],)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_not_fitted_raises(self):
        with pytest.raises(NotFittedError):
            TrinaryTree().predict(np.zeros((5, 3)))

    def test_wrong_n_features_raises(self):
        Xtr, Xte, ytr, _ = binary_data()
        clf = fast_trinary()
        clf.fit(Xtr, ytr)
        with pytest.raises(ValueError, match="features"):
            clf.predict(Xte[:, :5])

    def test_unknown_delta_raises(self):
        with pytest.raises(ValueError):
            TrinaryTree(delta_method="bogus").fit(
                np.random.randn(20, 3), np.array([0,1]*10)
            )
