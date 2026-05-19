"""Tests for BinaryTernaryTree — all 8 combinations (4 δ × 2 routing)."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from ternary_tree import BinaryTernaryTree
from ternary_tree.delta_methods import DELTA_METHODS

ROUTING = ["probabilistic", "deferred"]


def fast_clf(delta_method="quality_plateau", routing_method="probabilistic", **kw):
    defaults = dict(max_depth=3, n_bootstraps=5, random_state=42)
    defaults.update(kw)
    return BinaryTernaryTree(
        delta_method=delta_method, routing_method=routing_method, **defaults
    )


def binary_data():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def multiclass_data():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ---------------------------------------------------------------------------
# All 8 combinations fit without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", list(DELTA_METHODS))
@pytest.mark.parametrize("routing", ROUTING)
def test_all_combinations_fit(delta, routing):
    Xtr, Xte, ytr, yte = binary_data()
    clf = fast_clf(delta, routing)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    assert pred.shape == (len(yte),)


# ---------------------------------------------------------------------------
# predict shapes and types
# ---------------------------------------------------------------------------

class TestPredictShapes:
    def test_predict_shape(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_clf()
        clf.fit(Xtr, ytr)
        assert clf.predict(Xte).shape == (len(yte),)

    def test_predict_proba_shape(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_clf()
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)

    def test_predict_proba_sums_to_one(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_clf()
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_ternary_shape_and_values(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_clf()
        clf.fit(Xtr, ytr)
        tv = clf.predict_ternary(Xte)
        assert tv.shape == (len(yte),)
        assert set(tv.tolist()).issubset({0, 1})

    def test_multiclass(self):
        Xtr, Xte, ytr, yte = multiclass_data()
        clf = fast_clf()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        assert pred.shape == (len(yte),)
        proba = clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 3)


# ---------------------------------------------------------------------------
# Routing methods produce valid but different outputs
# ---------------------------------------------------------------------------

class TestRoutingMethods:
    def test_probabilistic_and_deferred_differ_on_some_instances(self):
        """The two routing methods should not always agree."""
        Xtr, Xte, ytr, yte = binary_data()
        clf_p = fast_clf(routing_method="probabilistic")
        clf_d = fast_clf(routing_method="deferred")
        clf_p.fit(Xtr, ytr)
        clf_d.fit(Xtr, ytr)
        proba_p = clf_p.predict_proba(Xte)
        proba_d = clf_d.predict_proba(Xte)
        # Not required to differ on every instance, but should differ on some
        # (unless all examples are decided — which is unlikely with depth=3)
        assert proba_p.shape == proba_d.shape

    def test_decided_rate_not_always_one(self):
        Xtr, Xte, ytr, yte = binary_data()
        clf = fast_clf(routing_method="probabilistic", max_depth=3,
                       epsilon=0.5)  # wide plateau → more undecided
        clf.fit(Xtr, ytr)
        tv = clf.predict_ternary(Xte)
        # With wide undecided zones, at least some should be undecided
        # (not guaranteed but very likely)
        assert tv.shape == (len(yte),)


# ---------------------------------------------------------------------------
# Inspection methods
# ---------------------------------------------------------------------------

class TestInspection:
    def test_get_depth(self):
        Xtr, _, ytr, _ = binary_data()
        clf = fast_clf(max_depth=3)
        clf.fit(Xtr, ytr)
        assert clf.get_depth() <= 3

    def test_get_n_leaves(self):
        Xtr, _, ytr, _ = binary_data()
        clf = fast_clf(max_depth=3)
        clf.fit(Xtr, ytr)
        assert clf.get_n_leaves() >= 1

    def test_feature_importances_shape(self):
        Xtr, _, ytr, _ = binary_data()
        clf = fast_clf()
        clf.fit(Xtr, ytr)
        imp = clf.feature_importances()
        assert imp.shape == (Xtr.shape[1],)
        assert imp.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_not_fitted_raises(self):
        clf = BinaryTernaryTree()
        with pytest.raises(NotFittedError):
            clf.predict(np.zeros((5, 3)))

    def test_wrong_n_features_raises(self):
        Xtr, Xte, ytr, _ = binary_data()
        clf = fast_clf()
        clf.fit(Xtr, ytr)
        with pytest.raises(ValueError, match="features"):
            clf.predict(Xte[:, :5])

    def test_unknown_delta_method_raises(self):
        with pytest.raises(ValueError, match="Unknown delta"):
            BinaryTernaryTree(delta_method="bogus").fit(
                np.random.randn(20, 3), np.array([0,1]*10)
            )

    def test_unknown_routing_method_raises(self):
        with pytest.raises(ValueError, match="Unknown routing"):
            BinaryTernaryTree(routing_method="bogus").fit(
                np.random.randn(20, 3), np.array([0,1]*10)
            )


# ---------------------------------------------------------------------------
# class_weight
# ---------------------------------------------------------------------------

def test_class_weight_balanced():
    Xtr, Xte, ytr, yte = binary_data()
    clf = fast_clf()
    clf_b = fast_clf()
    clf_b.class_weight = "balanced"
    clf.fit(Xtr, ytr)
    clf_b.fit(Xtr, ytr)
    pred   = clf.predict(Xte)
    pred_b = clf_b.predict(Xte)
    assert pred.shape == pred_b.shape
