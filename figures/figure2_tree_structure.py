"""
figure2_tree_structure.py
=========================
Figure 2: True ternary decision tree structure (TrinaryTree).

Each split node has THREE physical children:
  LEFT   — instances where feature <= theta - delta  (decided False)
  MIDDLE — instances in the uncertainty zone          (boundary-uncertain)
  RIGHT  — instances where feature >  theta + delta  (decided True)

The middle branch uses a dashed red edge and distinct box shading to
visually separate boundary-uncertain routing from decisive routing.

Run from project root:
    uv run python figures/figure2_tree_structure.py

Outputs: figures/figure2_tree_structure.pdf / .png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import load_breast_cancer

from ternary_tree import TrinaryTree
from ternary_tree.node import SplitNode, LeafNode

OUT_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_DEPTH    = 2          # depth 2 → 9 leaves — readable on one page
DELTA_METHOD = "margin"
RANDOM_STATE = 42

# Node geometry (data units)
NODE_W  = 8.0
NODE_H  = 5.5
X_GAP   = 2.5    # gap between adjacent leaf slots
LEVEL_H = 12.0   # vertical gap between levels

# Font sizes — deliberately large
FS_TITLE  = 32
FS_NODE   = 22
FS_LEAF   = 22
FS_EDGE   = 18
FS_LEGEND = 20

# Colours
SPLIT_FC   = "#D6EAF8"    # decided split node fill
SPLIT_EC   = "#2980B9"    # decided split node edge
MIDDLE_FC  = "#FDEBD0"    # middle-branch split node fill (boundary-uncertain)
MIDDLE_EC  = "#E67E22"    # middle-branch split node edge
LEAF0_FC   = "#D5F5E3"    # leaf class 0 (benign)
LEAF1_FC   = "#F9EBEA"    # leaf class 1 (malignant)
LEAF_EC    = "#7F8C8D"
DELTA_COL  = "#C0392B"    # delta value colour
ARROW_STD  = "#5D6D7E"    # standard edge colour
ARROW_MID  = "#E67E22"    # middle (undecided) edge colour


# ---------------------------------------------------------------------------
# Layout — in-order: left, middle, right
# ---------------------------------------------------------------------------

def _assign_x(node):
    positions = {}
    slot = NODE_W + X_GAP

    def _recurse(n, start, via_middle=False):
        positions[id(n)] = (0.0, via_middle)   # placeholder x, store via_middle flag
        if isinstance(n, LeafNode):
            cx = start + NODE_W / 2
            positions[id(n)] = (cx, via_middle)
            return start + slot

        # Recurse left, middle, right
        end_l = _recurse(n.left,   start,  via_middle=False)
        end_m = _recurse(n.middle, end_l,  via_middle=True)
        end_r = _recurse(n.right,  end_m,  via_middle=False)

        lx = positions[id(n.left)][0]
        rx = positions[id(n.right)][0]
        positions[id(n)] = ((lx + rx) / 2, via_middle)
        return end_r

    total = _recurse(node, 0.0, via_middle=False)
    return positions, total


def _assign_y(node, depth=0, y_map=None):
    if y_map is None:
        y_map = {}
    y_map[id(node)] = -depth * LEVEL_H
    if isinstance(node, SplitNode):
        _assign_y(node.left,   depth + 1, y_map)
        _assign_y(node.middle, depth + 1, y_map)
        _assign_y(node.right,  depth + 1, y_map)
    return y_map


def _collect_all(node, out=None):
    if out is None: out = []
    out.append(node)
    if isinstance(node, SplitNode):
        _collect_all(node.left,   out)
        _collect_all(node.middle, out)
        _collect_all(node.right,  out)
    return out


def _collect_edges(node, out=None):
    if out is None: out = []
    if isinstance(node, SplitNode):
        out.append((node, node.left,
                    r"$\leq\theta{-}\delta$", False))
        out.append((node, node.middle,
                    "UNDECIDED\n" + r"$(\theta{-}\delta$ to $\theta{+}\delta)$",
                    True))
        out.append((node, node.right,
                    r"$>\theta{+}\delta$", False))
        _collect_edges(node.left,   out)
        _collect_edges(node.middle, out)
        _collect_edges(node.right,  out)
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_split(ax, cx, cy, node, feat_names, via_middle):
    fname = feat_names[node.feature_idx]
    if len(fname) > 22:
        fname = fname[:20] + ".."

    fc = MIDDLE_FC if via_middle else SPLIT_FC
    ec = MIDDLE_EC if via_middle else SPLIT_EC

    box = FancyBboxPatch(
        (cx - NODE_W / 2, cy - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.1",
        facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=3)
    ax.add_patch(box)

    gap = NODE_H / 5.5
    y0  = cy + NODE_H / 2 - gap

    ax.text(cx, y0,           fname,
            ha="center", va="center", fontsize=FS_NODE,
            fontweight="bold", zorder=4)
    ax.text(cx, y0 - gap,     f"$\\theta$ = {node.threshold:.3f}",
            ha="center", va="center", fontsize=FS_NODE, zorder=4)
    ax.text(cx, y0 - 2*gap,   f"$\\delta$ = {node.delta:.4f}",
            ha="center", va="center", fontsize=FS_NODE,
            color=DELTA_COL, fontweight="bold", zorder=4)
    ax.text(cx, y0 - 3*gap,   f"$n$ = {node.n_samples}",
            ha="center", va="center", fontsize=FS_NODE - 2,
            color="#7F8C8D", zorder=4)


def _draw_leaf(ax, cx, cy, node, class_names, via_middle):
    cname = class_names[node.predicted_class]
    conf  = node.class_proba[node.predicted_class]

    if via_middle:
        fc = "#FDEBD0"   # boundary-uncertain leaf: orange tint
        ec = MIDDLE_EC
    elif node.predicted_class == 0:
        fc, ec = LEAF0_FC, LEAF_EC
    else:
        fc, ec = LEAF1_FC, LEAF_EC

    w = NODE_W * 0.88
    box = FancyBboxPatch(
        (cx - w / 2, cy - NODE_H / 2), w, NODE_H,
        boxstyle="round,pad=0.1",
        facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=3)
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


def _draw_edge(ax, x_pos, y_pos, parent, child, label, is_middle):
    px, py = x_pos[id(parent)][0], y_pos[id(parent)]
    cx, cy = x_pos[id(child)][0],  y_pos[id(child)]

    color  = ARROW_MID if is_middle else ARROW_STD
    style  = "dashed"  if is_middle else "solid"
    lw     = 2.0

    ax.annotate("",
        xy=(cx, cy + NODE_H / 2), xytext=(px, py - NODE_H / 2),
        arrowprops=dict(
            arrowstyle="->,head_width=0.5,head_length=0.4",
            color=color, lw=lw, linestyle=style),
        zorder=2)

    mx   = (px + cx) / 2
    my   = (py + cy) / 2 + 0.5
    side = -1.2 if cx < px - 0.5 else (1.2 if cx > px + 0.5 else 0)
    fw   = "bold" if is_middle else "normal"
    bbox_ec = ARROW_MID if is_middle else "none"

    ax.text(mx + side, my, label,
            ha="center", va="center", fontsize=FS_EDGE,
            color=color, fontweight=fw,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor=bbox_ec,
                      linewidth=1.5, alpha=0.92),
            zorder=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data        = load_breast_cancer()
    X, y        = data.data, data.target
    feat_names  = list(data.feature_names)
    class_names = list(data.target_names)

    clf = TrinaryTree(
        delta_method=DELTA_METHOD, max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE)
    clf.fit(X, y)
    tree = clf.tree_

    x_pos, total_w = _assign_x(tree)
    y_pos           = _assign_y(tree)
    all_nodes       = _collect_all(tree)
    edges           = _collect_edges(tree)

    xs    = [v[0] for v in x_pos.values()]
    ys    = list(y_pos.values())
    pad_x = NODE_W * 1.2
    pad_y = NODE_H * 1.8

    fig_w = min(44, max(28, total_w + 2 * pad_x))
    fig_h = min(30, max(18, abs(min(ys)) + NODE_H + 2 * pad_y))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Draw edges behind nodes
    for parent, child, label, is_mid in edges:
        _draw_edge(ax, x_pos, y_pos, parent, child, label, is_mid)

    # Draw nodes
    for node in all_nodes:
        cx = x_pos[id(node)][0]
        cy = y_pos[id(node)]
        via_mid = x_pos[id(node)][1]
        if isinstance(node, SplitNode):
            _draw_split(ax, cx, cy, node, feat_names, via_mid)
        else:
            _draw_leaf(ax, cx, cy, node, class_names, via_mid)

    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    fig.suptitle(
        "Ternary Decision Tree — Margin $\\delta$ Method\n"
        f"(TrinaryTree, depth {MAX_DEPTH}, Breast Cancer dataset)",
        fontsize=FS_TITLE, fontweight="bold", y=0.99)

    legend_handles = [
        mpatches.Patch(facecolor=SPLIT_FC, edgecolor=SPLIT_EC, linewidth=2,
                       label=r"Split node  ($\theta$ = threshold,  "
                             r"$\delta$ = uncertainty half-width)"),
        mpatches.Patch(facecolor=MIDDLE_FC, edgecolor=MIDDLE_EC, linewidth=2,
                       label="Node reached via boundary-uncertain branch"),
        mpatches.Patch(facecolor=LEAF0_FC, edgecolor=LEAF_EC, linewidth=2,
                       label="Leaf: benign (decided)"),
        mpatches.Patch(facecolor=LEAF1_FC, edgecolor=LEAF_EC, linewidth=2,
                       label="Leaf: malignant (decided)"),
        mpatches.Patch(facecolor="#FDEBD0", edgecolor=MIDDLE_EC, linewidth=2,
                       label="Leaf: boundary-uncertain prediction"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, 0.01), ncol=3,
              fontsize=FS_LEGEND, framealpha=0.95,
              borderpad=0.9, labelspacing=0.5)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"figure2_tree_structure.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=180)
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
