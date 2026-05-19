"""Tests for delta_methods.py — all four δ computation methods."""

import numpy as np
import pytest
from ternary_tree.delta_methods import (
    delta_quality_plateau,
    delta_class_overlap,
    delta_gain_ratio,
    delta_node_bootstrap,
    compute_delta,
    DELTA_METHODS,
)
from ternary_tree.splitter import evaluate_all_thresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_separable_node(n=100, seed=0):
    """Well-separated binary node — delta should be small."""
    rng = np.random.default_rng(seed)
    col = np.concatenate([rng.normal(2, 0.3, n//2), rng.normal(8, 0.3, n//2)])
    y   = np.array([0]*(n//2) + [1]*(n//2))
    sw  = np.ones(n)
    return col, y, sw


def make_overlapping_node(n=100, seed=0):
    """Heavily overlapping — delta should be larger."""
    rng = np.random.default_rng(seed)
    col = rng.normal(5, 2, n)
    y   = (col + rng.normal(0, 2, n) > 5).astype(int)
    sw  = np.ones(n)
    return col, y, sw


CLASSES = np.array([0, 1])


# ---------------------------------------------------------------------------
# Method 1: Quality Plateau
# ---------------------------------------------------------------------------

class TestQualityPlateau:
    def test_returns_nonneg(self):
        col, y, sw = make_separable_node()
        cands, scores, _, _ = evaluate_all_thresholds(col, y, sw, CLASSES, "gini")
        best_score = float(scores.max())
        delta = delta_quality_plateau(cands, scores, best_score, col, epsilon=0.05)
        assert delta >= 0.0

    def test_plateau_nonneg_both_node_types(self):
        """Quality plateau returns valid δ for both separable and overlapping nodes.

        Note: a separable node can have a WIDER plateau than an overlapping one
        because many thresholds may give equally good (perfect) separation.
        We only assert both return non-negative values.
        """
        col_s, y_s, sw_s = make_separable_node()
        col_o, y_o, sw_o = make_overlapping_node()

        cands_s, scrs_s, _, _ = evaluate_all_thresholds(col_s, y_s, sw_s, CLASSES, "gini")
        cands_o, scrs_o, _, _ = evaluate_all_thresholds(col_o, y_o, sw_o, CLASSES, "gini")

        d_s = delta_quality_plateau(cands_s, scrs_s, float(scrs_s.max()), col_s, epsilon=0.1)
        d_o = delta_quality_plateau(cands_o, scrs_o, float(scrs_o.max()), col_o, epsilon=0.1)
        assert d_s >= 0.0
        assert d_o >= 0.0

    def test_empty_candidates(self):
        col = np.array([1.0, 2.0, 3.0])
        delta = delta_quality_plateau(np.array([]), np.array([]), 0.0, col, epsilon=0.05)
        assert delta == 0.0

    def test_larger_epsilon_gives_wider_delta(self):
        col, y, sw = make_overlapping_node()
        cands, scrs, _, _ = evaluate_all_thresholds(col, y, sw, CLASSES, "gini")
        best = float(scrs.max())
        d_small = delta_quality_plateau(cands, scrs, best, col, epsilon=0.01)
        d_large = delta_quality_plateau(cands, scrs, best, col, epsilon=0.20)
        assert d_large >= d_small


# ---------------------------------------------------------------------------
# Method 2: Class Overlap
# ---------------------------------------------------------------------------

class TestClassOverlap:
    def test_returns_nonneg(self):
        col, y, sw = make_separable_node()
        delta = delta_class_overlap(col, y, CLASSES, percentile=10.0)
        assert delta >= 0.0

    def test_separated_near_zero(self):
        # Very clean separation → near-zero overlap
        col = np.array([0.0]*50 + [10.0]*50)
        y   = np.array([0]*50 + [1]*50)
        delta = delta_class_overlap(col, y, CLASSES, percentile=5.0)
        assert delta < 1.0

    def test_identical_distributions_positive(self):
        # Full overlap → large δ
        rng = np.random.default_rng(0)
        col = rng.normal(5, 1, 200)
        y   = rng.integers(0, 2, 200)
        delta = delta_class_overlap(col, y, CLASSES, percentile=10.0)
        assert delta > 0.0

    def test_single_class_returns_zero(self):
        col = np.array([1.0, 2.0, 3.0])
        y   = np.array([0, 0, 0])
        delta = delta_class_overlap(col, y, CLASSES, percentile=10.0)
        assert delta == 0.0


# ---------------------------------------------------------------------------
# Method 3: Gain Ratio
# ---------------------------------------------------------------------------

class TestGainRatio:
    def test_returns_nonneg(self):
        col, y, sw = make_separable_node()
        cands, scrs, _, sinfos = evaluate_all_thresholds(col, y, sw, CLASSES, "entropy")
        best_idx = int(np.argmax(scrs))
        delta = delta_gain_ratio(col, float(scrs[best_idx]), float(sinfos[best_idx]), alpha=0.1)
        assert delta >= 0.0

    def test_high_gain_ratio_small_delta(self):
        col, y, sw = make_separable_node()
        # High GR → small δ
        delta_hi_gr = delta_gain_ratio(col, best_score=0.9, best_split_info=0.5, alpha=0.1)
        delta_lo_gr = delta_gain_ratio(col, best_score=0.05, best_split_info=0.5, alpha=0.1)
        assert delta_hi_gr < delta_lo_gr

    def test_alpha_scales_delta(self):
        col, y, sw = make_separable_node()
        d1 = delta_gain_ratio(col, 0.5, 0.8, alpha=0.05)
        d2 = delta_gain_ratio(col, 0.5, 0.8, alpha=0.20)
        assert d2 > d1

    def test_zero_split_info(self):
        col = np.ones(10)
        delta = delta_gain_ratio(col, best_score=0.5, best_split_info=0.0, alpha=0.1)
        assert delta >= 0.0  # should not crash


# ---------------------------------------------------------------------------
# Method 4: Node Bootstrap
# ---------------------------------------------------------------------------

class TestNodeBootstrap:
    def test_returns_nonneg(self):
        col, y, sw = make_separable_node(n=60)
        cands, _, _, _ = evaluate_all_thresholds(col, y, sw, CLASSES, "gini")
        delta = delta_node_bootstrap(col, y, sw, CLASSES, cands,
                                     criterion="gini", n_bootstraps=10,
                                     rng=np.random.default_rng(0))
        assert delta >= 0.0

    def test_small_n_returns_zero(self):
        col = np.array([1.0, 2.0])
        y   = np.array([0, 1])
        sw  = np.ones(2)
        delta = delta_node_bootstrap(col, y, sw, CLASSES, np.array([]),
                                     n_bootstraps=5)
        assert delta == 0.0

    def test_reproducible_with_seed(self):
        col, y, sw = make_separable_node(n=80)
        cands, _, _, _ = evaluate_all_thresholds(col, y, sw, CLASSES, "gini")
        d1 = delta_node_bootstrap(col, y, sw, CLASSES, cands,
                                   n_bootstraps=10, rng=np.random.default_rng(42))
        d2 = delta_node_bootstrap(col, y, sw, CLASSES, cands,
                                   n_bootstraps=10, rng=np.random.default_rng(42))
        assert d1 == pytest.approx(d2)

    def test_entropy_criterion(self):
        col, y, sw = make_overlapping_node(n=80)
        cands, _, _, _ = evaluate_all_thresholds(col, y, sw, CLASSES, "entropy")
        delta = delta_node_bootstrap(col, y, sw, CLASSES, cands,
                                     criterion="entropy", n_bootstraps=10,
                                     rng=np.random.default_rng(0))
        assert delta >= 0.0


# ---------------------------------------------------------------------------
# compute_delta dispatcher
# ---------------------------------------------------------------------------

class TestComputeDelta:
    def test_all_methods_run(self):
        col, y, sw = make_separable_node(n=80)
        cands, scrs, _, sinfos = evaluate_all_thresholds(col, y, sw, CLASSES, "gini")
        best_idx = int(np.argmax(scrs))
        rng = np.random.default_rng(0)

        for method in DELTA_METHODS:
            delta = compute_delta(
                method=method, col=col, y=y, sample_weight=sw,
                classes=CLASSES, candidates=cands, scores=scrs,
                split_infos=sinfos, best_score=float(scrs[best_idx]),
                best_idx=best_idx, criterion="gini",
                n_bootstraps=5, rng=rng,
            )
            assert delta >= 0.0, f"Method {method} returned negative delta"

    def test_unknown_method_raises(self):
        col, y, sw = make_separable_node(n=40)
        cands, scrs, _, sinfos = evaluate_all_thresholds(col, y, sw, CLASSES)
        with pytest.raises(ValueError, match="Unknown delta method"):
            compute_delta(
                method="bogus", col=col, y=y, sample_weight=sw,
                classes=CLASSES, candidates=cands, scores=scrs,
                split_infos=sinfos, best_score=0.5, best_idx=0,
                criterion="gini",
            )


# ---------------------------------------------------------------------------
# Method 5: Margin
# ---------------------------------------------------------------------------

from ternary_tree.delta_methods import delta_margin

class TestMarginDelta:
    def test_returns_nonneg(self):
        col, y, sw = make_separable_node()
        cands, scrs, _, _ = evaluate_all_thresholds(col, y, sw, CLASSES, "gini")
        best_theta = float(cands[np.argmax(scrs)])
        delta = delta_margin(col, y, best_theta, CLASSES)
        assert delta >= 0.0

    def test_perfect_separation_uses_physical_gap(self):
        # theta=9.5 midpoint; physical gap on each side = 0.5
        col = np.arange(20, dtype=float)
        y   = np.array([0]*10 + [1]*10)
        delta = delta_margin(col, y, theta=9.5, classes=CLASSES)
        assert delta == pytest.approx(0.5, abs=1e-6)

    def test_clean_wide_gap_larger_delta(self):
        col = np.array([0., 1., 2., 8., 9., 10.])
        y   = np.array([0,  0,  0,  1,  1,  1 ])
        delta = delta_margin(col, y, theta=5.0, classes=CLASSES)
        # Raw physical gap = 3.0 on each side, but clamped to 25% of range:
        # 0.25 * (10 - 0) = 2.5
        assert delta == pytest.approx(2.5, abs=1e-6)

    def test_cross_class_intrusion_reduces_delta(self):
        # class-1 example at 4.9 on the left side of theta=5.0
        col = np.array([0., 1., 4.9, 5.1, 8., 9.])
        y   = np.array([0,  0,  1,   1,   1,  1 ])
        delta = delta_margin(col, y, theta=5.0, classes=CLASSES)
        assert delta == pytest.approx(0.1, abs=1e-6)

    def test_single_class_returns_zero(self):
        col = np.array([1., 2., 3., 4., 5.])
        y   = np.array([0,  0,  0,  0,  0 ])
        delta = delta_margin(col, y, theta=3.0, classes=CLASSES)
        assert delta == 0.0

    def test_clamped_to_max_fraction(self):
        col = np.array([0., 1., 100., 101.])
        y   = np.array([0,  0,  1,    1  ])
        delta = delta_margin(col, y, theta=50.0, classes=CLASSES)
        assert delta <= 0.25 * 101.0 + 1e-9
