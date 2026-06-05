"""
figure4_probabilistic_tree.py
==============================
Figure 4: BinaryTernaryTree — Probabilistic Routing.

Complements Figure 2 / Appendix F (TrinaryTree, hard-middle routing).
Here the tree has the same two-branch structure as standard CART.
The delta zone [theta-delta, theta+delta] is NOT a physical third branch —
instances in this zone receive predictions formed by distance-weighted
blending of BOTH left and right child subtree outputs simultaneously.

Key visual differences from the TrinaryTree figure:
  - Two branches per node (not three)
  - Green colour scheme signals a different architecture
  - Each node annotates the blend zone range [theta-delta, theta+delta]
  - A routing annotation box explains the three-zone logic
  - Edge labels mark which instances route decisively left/right

Run from the project root:
    uv run python figures/figure4_probabilistic_tree.py

Outputs: figures/figure4_probabilistic_tree.pdf / .png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import load_breast_cancer

from ternary_tree import BinaryTernaryTree
from ternary_tree.node import SplitNode, LeafNode

OUT_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_DEPTH    = 2          # same depth as TrinaryTree figure for comparability
DELTA_METHOD = "margin"
RANDOM_STATE = 42

# Node geometry
NODE_W  = 8.0
NODE_H  = 6.0    # taller than TrinaryTree figure — 5 lines of text
X_GAP   = 2.5
LEVEL_H = 12.0

# Font sizes — large for readability
FS_TITLE  = 30
FS_NODE   = 20
FS_LEAF   = 20
FS_EDGE   = 17
FS_LEGEND = 18
FS_ANNOT  = 16

# Colours — green scheme to visually differentiate from TrinaryTree (blue)
SPLIT_FC   = "#D5F5E3"    # light green
SPLIT_EC   = "#1E8449"    # dark green
LEAF0_FC   = "#D6EAF8"    # light blue (benign)
LEAF1_FC   = "#F9EBEA"    # light red (malignant)
LEAF_EC    = "#7F8C8D"
DELTA_COL  = "#C0392B"    # red for delta
BLEND_COL  = "#7F8C8D"    # grey for blend zone annotation
ARROW_COL  = "#2E4053"
ANNOT_FC   = "#FEF9E7"    # annotation box fill (light yellow)
ANNOT_EC   = "#F39C12"    # annotation box edge (orange)


# ---------------------------------------------------------------------------
# Layout (standard binary tree — same as CART structure)
# ---------------------------------------------------------------------------

def _assign_x(node):
    positions = {}
    slot = NODE_W + X_GAP

    def _recurse(n, start):
        if isinstance(n, LeafNode):
            positions[id(n)] = start + NODE_W / 2
            return start + slot
        end_l = _recurse(n.left,  start)
        end_r = _recurse(n.right, end_l)
        positions[id(n)] = (positions[id(n.left)] + positions[id(n.right)]) / 2
        return end_r

    total = _recurse(node, 0.0)
    return positions, total


def _assign_y(node, depth=0, y_map=None):
    if y_map is None:
        y_map = {}
    y_map[id(node)] = -depth * LEVEL_H
    if isinstance(node, SplitNode):
        _assign_y(node.left,  depth + 1, y_map)
        _assign_y(node.right, depth + 1, y_map)
    return y_map


def _collect_all(node, out=None):
    if out is None: out = []
    out.append(node)
    if isinstance(node, SplitNode):
        _collect_all(node.left,  out)
        _collect_all(node.right, out)
    return out


def _collect_edges(node, out=None):
    if out is None: out = []
    if isinstance(node, SplitNode):
        out.append((node, node.left,  r"$x_f \leq \theta{-}\delta$" + "\n(decisive)"))
        out.append((node, node.right, r"$x_f > \theta{+}\delta$" + "\n(decisive)"))
        _collect_edges(node.left,  out)
        _collect_edges(node.right, out)
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_split(ax, cx, cy, node, feat_names):
    fname = feat_names[node.feature_idx]
    if len(fname) > 22:
        fname = fname[:20] + ".."

    lo = node.threshold - node.delta
    hi = node.threshold + node.delta

    box = FancyBboxPatch(
        (cx - NODE_W / 2, cy - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.1",
        facecolor=SPLIT_FC, edgecolor=SPLIT_EC, linewidth=2.5, zorder=3)
    ax.add_patch(box)

    # Five lines of text evenly spaced inside the box
    gap = NODE_H / 6.5
    y0  = cy + NODE_H / 2 - gap

    ax.text(cx, y0,          fname,
            ha="center", va="center", fontsize=FS_NODE,
            fontweight="bold", zorder=4)
    ax.text(cx, y0 - gap,    f"$\\theta$ = {node.threshold:.3f}",
            ha="center", va="center", fontsize=FS_NODE, zorder=4)
    ax.text(cx, y0 - 2*gap,  f"$\\delta$ = {node.delta:.4f}",
            ha="center", va="center", fontsize=FS_NODE,
            color=DELTA_COL, fontweight="bold", zorder=4)
    ax.text(cx, y0 - 3*gap,
            f"blend: [{lo:.3f},  {hi:.3f}]",
            ha="center", va="center", fontsize=FS_NODE - 3,
            color=BLEND_COL, style="italic", zorder=4)
    ax.text(cx, y0 - 4*gap,  f"$n$ = {node.n_samples}",
            ha="center", va="center", fontsize=FS_NODE - 2,
            color=BLEND_COL, zorder=4)


def _draw_leaf(ax, cx, cy, node, class_names):
    cname = class_names[node.predicted_class]
    conf  = node.class_proba[node.predicted_class]
    fc    = LEAF0_FC if node.predicted_class == 0 else LEAF1_FC

    w = NODE_W * 0.88
    box = FancyBboxPatch(
        (cx - w / 2, cy - NODE_H / 2), w, NODE_H,
        boxstyle="round,pad=0.1",
        facecolor=fc, edgecolor=LEAF_EC, linewidth=2.0, zorder=3)
    ax.add_patch(box)

    gap = NODE_H / 4.5
    y0  = cy + NODE_H / 2 - gap

    ax.text(cx, y0,          f"Class: {cname}",
            ha="center", va="center", fontsize=FS_LEAF,
            fontweight="bold", zorder=4)
    ax.text(cx, y0 - gap,    f"conf = {conf:.2f}",
            ha="center", va="center", fontsize=FS_LEAF, zorder=4)
    ax.text(cx, y0 - 2*gap,  f"$n$ = {node.n_samples}",
            ha="center", va="center", fontsize=FS_LEAF - 2,
            color="#7F8C8D", zorder=4)


def _draw_edge(ax, x_pos, y_pos, parent, child, label):
    px, py = x_pos[id(parent)], y_pos[id(parent)]
    cx, cy = x_pos[id(child)],  y_pos[id(child)]

    ax.annotate("",
        xy=(cx, cy + NODE_H / 2), xytext=(px, py - NODE_H / 2),
        arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.4",
                        color=ARROW_COL, lw=1.8), zorder=2)

    mx   = (px + cx) / 2
    my   = (py + cy) / 2 + 0.3
    side = -1.0 if cx < px - 0.5 else 1.0

    ax.text(mx + side, my, label,
            ha="center", va="center", fontsize=FS_EDGE,
            color=ARROW_COL,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="none", alpha=0.9),
            zorder=5)


def _draw_routing_annotation(ax, x_right, y_top):
    """Explains the three-zone routing logic in a callout box."""
    text = (
        "Probabilistic routing rules:\n\n"
        r"$x_f \leq \theta - \delta$" + "\n"
        "  decisive left subtree\n\n"
        r"$\theta - \delta < x_f \leq \theta + \delta$" + "\n"
        "  BOTH subtrees (blended)\n"
        r"  $\hat{p} = w_L \hat{p}_L + w_R \hat{p}_R$" + "\n\n"
        r"$x_f > \theta + \delta$" + "\n"
        "  decisive right subtree"
    )
    ax.text(x_right, y_top, text,
            ha="left", va="top", fontsize=FS_ANNOT,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor=ANNOT_FC,
                      edgecolor=ANNOT_EC,
                      linewidth=2.0, alpha=0.97),
            zorder=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data        = load_breast_cancer()
    X, y        = data.data, data.target
    feat_names  = list(data.feature_names)
    class_names = list(data.target_names)

    clf = BinaryTernaryTree(
        delta_method=DELTA_METHOD, routing_method="probabilistic",
        max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    clf.fit(X, y)
    tree = clf.tree_

    x_pos, total_w = _assign_x(tree)
    y_pos           = _assign_y(tree)
    all_nodes       = _collect_all(tree)
    edges           = _collect_edges(tree)

    xs    = list(x_pos.values())
    ys    = list(y_pos.values())
    pad_x = NODE_W * 2.0
    pad_y = NODE_H * 2.0

    fig_w = min(46, max(30, total_w + 2 * pad_x))
    fig_h = min(28, max(16, abs(min(ys)) + NODE_H + 2 * pad_y))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Draw edges then nodes
    for parent, child, label in edges:
        _draw_edge(ax, x_pos, y_pos, parent, child, label)
    for node in all_nodes:
        cx, cy = x_pos[id(node)], y_pos[id(node)]
        if isinstance(node, SplitNode):
            _draw_split(ax, cx, cy, node, feat_names)
        else:
            _draw_leaf(ax, cx, cy, node, class_names)

    # Routing annotation box — upper right, outside the tree
    x_annot = max(xs) + NODE_W * 0.6
    y_annot = max(ys) + NODE_H * 0.3
    _draw_routing_annotation(ax, x_annot, y_annot)

    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x * 3.0)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    fig.suptitle(
        "Ternary Decision Tree  —  Probabilistic Routing\n"
        "Margin $\\delta$ Method  |  depth 2  |  Breast Cancer dataset",
        fontsize=FS_TITLE, fontweight="bold", y=0.99)

    legend_handles = [
        mpatches.Patch(facecolor=SPLIT_FC, edgecolor=SPLIT_EC, linewidth=2,
                       label=r"Split node  ($\theta$ = threshold,  "
                             r"$\delta$ = uncertainty half-width,  "
                             r"blend range = $[\theta{-}\delta,\, \theta{+}\delta]$)"),
        mpatches.Patch(facecolor=LEAF0_FC, edgecolor=LEAF_EC, linewidth=2,
                       label="Leaf: benign"),
        mpatches.Patch(facecolor=LEAF1_FC, edgecolor=LEAF_EC, linewidth=2,
                       label="Leaf: malignant"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.35, 0.01), ncol=3,
              fontsize=FS_LEGEND, framealpha=0.95, borderpad=0.9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"figure4_probabilistic_tree.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=180)
        print(f"Saved {path}")
    plt.close()

    # Print node summary
    print("\nSplit nodes (BinaryTernaryTree, probabilistic routing):")
    print(f"  {'Feature':<35} {'theta':>10} {'delta':>10} "
          f"{'blend_lo':>10} {'blend_hi':>10} {'n':>6}")
    print(f"  {'-'*83}")

    def _print(node, depth=0):
        if isinstance(node, SplitNode):
            lo = node.threshold - node.delta
            hi = node.threshold + node.delta
            print(f"  {'  '*depth}{feat_names[node.feature_idx][:33]:<35} "
                  f"{node.threshold:>10.4f} {node.delta:>10.4f} "
                  f"{lo:>10.4f} {hi:>10.4f} {node.n_samples:>6}")
            _print(node.left,  depth + 1)
            _print(node.right, depth + 1)

    _print(clf.tree_)


if __name__ == "__main__":
    main()
