"""
figure1_decision_surface.py
===========================
Figure 1: Decision boundary comparison — CART vs Ternary Decision Tree.

Key design choice:
  The boundary-uncertain zone is shown as HATCHING overlaid on the
  underlying two-class background, not as a solid neutral grey.
  This makes it visually unambiguous that a class prediction exists
  everywhere — the hatching qualifies the confidence of the prediction,
  not its absence.

  Solid blue/red  = decisive class prediction
  Blue+hatch      = Class 0 predicted via boundary-uncertain blending
  Red+hatch       = Class 1 predicted via boundary-uncertain blending

Run from the project root:
    uv run python figures/figure1_decision_surface.py

Outputs: figures/figure1_decision_surface.pdf / .png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from ternary_tree import BinaryTernaryTree

OUT_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES    = 250
NOISE        = 0.28
MAX_DEPTH    = 3
DELTA_METHOD = "quality_plateau"
EPSILON      = 0.15
RANDOM_STATE = 42
MESH_RES     = 400

# Colours
C0_BG  = "#AED6F1"   # class 0 background (light blue)
C1_BG  = "#F1948A"   # class 1 background (light red/salmon)
C0_PT  = "#1A5276"   # class 0 scatter points (dark blue)
C1_PT  = "#922B21"   # class 1 scatter points (dark red)
HATCH_EDGE = "#5D6D7E"   # hatch line colour


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data():
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE,
                      random_state=RANDOM_STATE)
    return StandardScaler().fit_transform(X), y


def make_meshgrid(X, res=MESH_RES, margin=0.6):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, res),
        np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, res),
    )
    return xx, yy


# ---------------------------------------------------------------------------
# Panel (a): standard CART
# ---------------------------------------------------------------------------

def plot_cart(ax, clf, X, y, xx, yy):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z,
                cmap=ListedColormap([C0_BG, C1_BG]),
                alpha=0.55, levels=[-0.5, 0.5, 1.5])
    _scatter(ax, X, y)
    _axes(ax, xx, yy, "(a) Standard CART\nHard binary decision regions")


# ---------------------------------------------------------------------------
# Panel (b): ternary decision tree — hatching version
# ---------------------------------------------------------------------------

def plot_ternary(ax, clf, X, y, xx, yy):
    X_mesh    = np.c_[xx.ravel(), yy.ravel()]
    Z_pred    = clf.predict(X_mesh)
    Z_ternary = clf.predict_ternary(X_mesh)

    Z_class = Z_pred.reshape(xx.shape)
    Z_undec = (Z_ternary == 0).reshape(xx.shape).astype(float)

    # ── Step 1: full two-class background everywhere ─────────────────────
    # Every pixel gets a class colour — this is the CLASS PREDICTION.
    # Even in the undecided zone a class is predicted; it is just formed
    # by weighted subtree blending rather than deterministic routing.
    ax.contourf(xx, yy, Z_class,
                cmap=ListedColormap([C0_BG, C1_BG]),
                alpha=0.60, levels=[-0.5, 0.5, 1.5], zorder=1)

    # ── Step 2: hatch overlay on boundary-uncertain zone only ────────────
    # Hatching qualifies the CONFIDENCE of the prediction, not its presence.
    # The underlying class colour remains visible through the hatch.
    if Z_undec.max() > 0:
        import matplotlib as _mpl
        with _mpl.rc_context({"hatch.color": HATCH_EDGE,
                              "hatch.linewidth": 0.6}):
            ax.contourf(xx, yy, Z_undec,
                        levels=[0.5, 1.5],
                        colors=["white"],
                        hatches=["////"],
                        alpha=0.30, zorder=2)

        # Dotted boundary outline of the undecided zone
        ax.contour(xx, yy, Z_undec, levels=[0.5],
                   colors=[HATCH_EDGE], linewidths=1.2,
                   linestyles="--", zorder=3)

    _scatter(ax, X, y)

    # Stats for subtitle
    tv          = clf.predict_ternary(X)
    undec_pct   = 100 * (tv == 0).mean()
    dec_mask    = tv == 1
    decided_acc = float(np.mean(clf.predict(X[dec_mask]) == y[dec_mask])) \
                  if dec_mask.any() else float("nan")

    _axes(ax, xx, yy,
          f"(b) Ternary Decision Tree (Quality-Plateau $\\delta$)\n"
          f"Hatching = boundary-uncertain zone ({undec_pct:.1f}%)  "
          f"|  Decided accuracy: {decided_acc:.3f}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scatter(ax, X, y):
    for cls, color, marker in [(0, C0_PT, "o"), (1, C1_PT, "^")]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=color, marker=marker, s=24,
                   edgecolors="white", linewidths=0.5,
                   zorder=5)


def _axes(ax, xx, yy, title):
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.set_title(title, fontsize=10, pad=8)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def make_legend(fig):
    """
    Five legend entries that together make the visual encoding explicit.
    The hatched patches make clear that background colour = predicted class
    even in the boundary-uncertain zone.
    """
    entries = [
        # Decisive regions
        mpatches.Patch(facecolor=C0_BG, edgecolor="grey", linewidth=0.8,
                       label="Class 0 — decisive prediction"),
        mpatches.Patch(facecolor=C1_BG, edgecolor="grey", linewidth=0.8,
                       label="Class 1 — decisive prediction"),
        # Boundary-uncertain: show BOTH class colours with hatching so the
        # reader understands the hatch just qualifies, not removes, prediction
        mpatches.Patch(facecolor=C0_BG, edgecolor=HATCH_EDGE,
                       hatch="////", linewidth=0.8,
                       label="Class 0 — boundary-uncertain (blended prediction)"),
        mpatches.Patch(facecolor=C1_BG, edgecolor=HATCH_EDGE,
                       hatch="////", linewidth=0.8,
                       label="Class 1 — boundary-uncertain (blended prediction)"),
        # Training points
        plt.scatter([], [], c=C0_PT, marker="o", s=30,
                    edgecolors="white", label="Class 0 training"),
        plt.scatter([], [], c=C1_PT, marker="^", s=30,
                    edgecolors="white", label="Class 1 training"),
    ]
    fig.legend(handles=entries, loc="lower center",
               ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.10),
               framealpha=0.95, borderpad=0.8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X, y = prepare_data()

    cart = DecisionTreeClassifier(max_depth=MAX_DEPTH,
                                   random_state=RANDOM_STATE)
    cart.fit(X, y)

    ternary = BinaryTernaryTree(
        delta_method=DELTA_METHOD, routing_method="probabilistic",
        max_depth=MAX_DEPTH, epsilon=EPSILON, random_state=RANDOM_STATE)
    ternary.fit(X, y)

    xx, yy = make_meshgrid(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(
        "Decision Boundaries: Standard CART vs Ternary Decision Tree\n"
        "Background colour = predicted class (every point receives a prediction)",
        fontsize=11, fontweight="bold", y=1.02)

    plot_cart(axes[0], cart, X, y, xx, yy)
    plot_ternary(axes[1], ternary, X, y, xx, yy)
    make_legend(fig)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"figure1_decision_surface.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
