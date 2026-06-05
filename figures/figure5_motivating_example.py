"""
figure5_motivating_example.py
==============================
Figure 5: Motivating example — how the five delta-estimation methods
characterise the same split node from four different analytical angles.

A synthetic 1-D split node is used so that all five methods produce
visible, non-zero delta values simultaneously, clearly illustrating
that they are alternative estimators of the same underlying quantity:
the local error probability p_e(d) at distance d from theta*.

Synthetic node design:
  Class 0: N(5.0, 1.5) — 150 examples (left of threshold)
  Class 1: N(9.0, 1.5) — 150 examples (right of threshold)
  Optimal threshold theta* ~ 7.0 (midpoint of means)

Four panels:
  (a) Split quality curve          — quality-plateau delta
  (b) Class-conditional histograms — class-overlap delta + margin delta
  (c) Bootstrap threshold dist.    — node-bootstrap delta
  (d) Uncertainty zone comparison  — all five delta values

Run from project root:
    uv run python figures/figure5_motivating_example.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.tree import DecisionTreeClassifier

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_EACH      = 150     # examples per class
MU0, MU1    = 5.0, 9.0
SIGMA       = 1.5
N_BOOTSTRAP = 300
EPSILON     = 0.05
Q_OVERLAP   = 0.10
ALPHA_GR    = 0.10
RANDOM_SEED = 42

FS_TITLE = 12
FS_LABEL = 10
FS_TICK  =  8
FS_ANNOT =  9.5

C0     = "#2980B9"
C1     = "#E74C3C"
THRESH = "#2C3E50"

DELTA_STYLES = {
    "quality-plateau": {"color": "#F39C12", "label": r"$\delta_{\mathrm{QP}}$"},
    "class-overlap"  : {"color": "#8E44AD", "label": r"$\delta_{\mathrm{CO}}$"},
    "gain-ratio"     : {"color": "#E67E22", "label": r"$\delta_{\mathrm{GR}}$"},
    "node-bootstrap" : {"color": "#2471A3", "label": r"$\delta_{\mathrm{NB}}$"},
    "margin"         : {"color": "#1E8449", "label": r"$\delta_{\mathrm{M}}$"},
}


# ---------------------------------------------------------------------------
# Synthetic node data
# ---------------------------------------------------------------------------

def make_node(seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    c0  = rng.normal(MU0, SIGMA, N_EACH)
    c1  = rng.normal(MU1, SIGMA, N_EACH)
    c   = np.concatenate([c0, c1])
    y   = np.array([0] * N_EACH + [1] * N_EACH, dtype=int)
    return c, y


# ---------------------------------------------------------------------------
# Delta computations (self-contained)
# ---------------------------------------------------------------------------

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - float(np.sum(p**2))


def gini_gain(y, mask):
    n, n_l, n_r = len(y), int(mask.sum()), int((~mask).sum())
    if n_l == 0 or n_r == 0:
        return 0.0
    return (gini_impurity(y)
            - (n_l/n) * gini_impurity(y[mask])
            - (n_r/n) * gini_impurity(y[~mask]))


def quality_curve(c, y):
    splits = np.unique(c)[:-1]
    gains  = np.array([gini_gain(y, c <= t) for t in splits])
    return splits, gains


def delta_qp(thresholds, gains, epsilon=EPSILON):
    q_star   = gains.max()
    plateau  = thresholds[gains >= (1 - epsilon) * q_star]
    if len(plateau) < 2:
        return 0.0, thresholds[gains.argmax()], thresholds[gains.argmax()]
    return (plateau[-1] - plateau[0]) / 2.0, plateau[0], plateau[-1]


def delta_co(c, y, q=Q_OVERLAP):
    best = 0.0
    classes = np.unique(y)
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            v1, v2 = c[y == c1], c[y == c2]
            lo = min(np.quantile(v1, 1-q), np.quantile(v2, 1-q))
            hi = max(np.quantile(v1, q),   np.quantile(v2, q))
            if hi > lo:
                best = max(best, (hi - lo) / 2.0)
    return best


def delta_gr(c, y, theta_star, best_gain, alpha=ALPHA_GR):
    n_l  = int((c <= theta_star).sum())
    n    = len(c)
    p_l, p_r = n_l/n, 1 - n_l/n
    eps  = 1e-9
    h    = -(p_l * np.log2(p_l + eps) + p_r * np.log2(p_r + eps))
    gr   = best_gain / (h + eps)
    return alpha * (c.max() - c.min()) / (1.0 + gr)


def delta_nb(c, y, B=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng  = np.random.default_rng(seed)
    boot = []
    Xn   = c.reshape(-1, 1)
    for _ in range(B):
        idx = rng.choice(len(c), len(c), replace=True)
        Xb, yb = Xn[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        dt = DecisionTreeClassifier(max_depth=1, random_state=0)
        dt.fit(Xb, yb)
        boot.append(float(dt.tree_.threshold[0]))
    arr = np.array(boot)
    return float(np.std(arr)), arr


def delta_margin(c, y, theta_star):
    left  = c <= theta_star
    right = ~left
    dom_l = np.bincount(y[left]).argmax()
    dom_r = np.bincount(y[right]).argmax()
    xl    = c[left  & (y == dom_r)]   # cross-class on left
    xr    = c[right & (y == dom_l)]   # cross-class on right
    ml    = (theta_star - xl.max())  if len(xl) else np.inf
    mr    = (xr.min() - theta_star)  if len(xr) else np.inf
    d     = min(ml, mr)
    nl    = xl.max() if len(xl) else None
    nr    = xr.min() if len(xr) else None
    return d, nl, nr


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure():
    c, y = make_node()
    thresholds, gains = quality_curve(c, y)

    dt = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED)
    dt.fit(c.reshape(-1, 1), y)
    theta_star = float(dt.tree_.threshold[0])
    best_gain  = gains.max()

    d_qp_val, lo_plat, hi_plat = delta_qp(thresholds, gains)
    d_co_val  = delta_co(c, y)
    d_gr_val  = delta_gr(c, y, theta_star, best_gain)
    d_nb_val, boot_arr = delta_nb(c, y)
    d_m_val,  nl, nr   = delta_margin(c, y, theta_star)

    deltas_ordered = [
        ("margin",          d_m_val),
        ("node-bootstrap",  d_nb_val),
        ("gain-ratio",      d_gr_val),
        ("class-overlap",   d_co_val),
        ("quality-plateau", d_qp_val),
    ]

    print(f"Synthetic node: theta*={theta_star:.4f}  n={len(c)}")
    for name, val in deltas_ordered[::-1]:
        print(f"  delta_{name:<20} = {val:.4f}")

    # View window
    span = max(d_qp_val, d_co_val, d_gr_val, d_nb_val, d_m_val, 0.3) * 5
    lo_v = theta_star - span
    hi_v = theta_star + span

    fig = plt.figure(figsize=(18, 5.6))
    gs  = GridSpec(1, 4, figure=fig, wspace=0.62)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # --- Panel (a): Quality curve -------------------------------------------
    ax = axs[0]
    mask = (thresholds >= lo_v) & (thresholds <= hi_v)
    ax.plot(thresholds[mask], gains[mask],
            color=THRESH, lw=2.0, zorder=3)

    # Plateau band
    q_star  = gains.max()
    in_plat = gains >= (1 - EPSILON) * q_star
    pt = thresholds[in_plat & mask]
    pg = gains[in_plat & mask]
    if len(pt):
        ax.fill_between(pt, pg, alpha=0.30,
                        color=DELTA_STYLES["quality-plateau"]["color"],
                        label=f"Plateau (ε={EPSILON})")
        for xv in [pt.min(), pt.max()]:
            ax.axvline(xv, color=DELTA_STYLES["quality-plateau"]["color"],
                       lw=1.0, ls="--", alpha=0.75)

    ax.axvline(theta_star, color=THRESH, lw=2.0, ls="-",
               label=f"$\\theta^*={theta_star:.2f}$")

    # Bracket
    yb = q_star * 0.28
    col_qp = DELTA_STYLES["quality-plateau"]["color"]
    ax.annotate("", xy=(theta_star + d_qp_val, yb),
                xytext=(theta_star - d_qp_val, yb),
                arrowprops=dict(arrowstyle="<->", color=col_qp, lw=2.2))
    ax.text(theta_star, yb + q_star * 0.05,
            f"$\\delta_{{\\mathrm{{QP}}}}={d_qp_val:.3f}$",
            ha="center", va="bottom", fontsize=FS_ANNOT,
            color=col_qp, fontweight="bold")

    ax.set_xlim(lo_v, hi_v)
    ax.set_xlabel("Feature value", fontsize=FS_LABEL)
    ax.set_ylabel("Gini gain $Q(\\theta)$", fontsize=FS_LABEL)
    ax.set_title("(a) Split quality curve\n"
                 r"Quality-plateau: $\delta_{\mathrm{QP}}$",
                 fontsize=FS_TITLE, pad=8)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=7.5, loc="upper right")

    # --- Panel (b): Class histograms ----------------------------------------
    ax = axs[1]
    bins = np.linspace(lo_v, hi_v, 40)
    for cls, col, lbl in [(0, C0, "Class 0"), (1, C1, "Class 1")]:
        vals = c[(c >= lo_v) & (c <= hi_v) & (y == cls)]
        ax.hist(vals, bins=bins, alpha=0.45, color=col, label=lbl)

    ax.axvline(theta_star, color=THRESH, lw=2.0)

    ymax = ax.get_ylim()[1]

    # Class-overlap bracket
    col_co = DELTA_STYLES["class-overlap"]["color"]
    y_co   = ymax * 0.78
    ax.annotate("", xy=(theta_star + d_co_val, y_co),
                xytext=(theta_star - d_co_val, y_co),
                arrowprops=dict(arrowstyle="<->", color=col_co, lw=2.2))
    ax.text(theta_star, y_co + ymax * 0.05,
            f"$\\delta_{{\\mathrm{{CO}}}}={d_co_val:.3f}$",
            ha="center", va="bottom", fontsize=FS_ANNOT,
            color=col_co, fontweight="bold")

    # Margin — nearest cross-class examples
    col_m = DELTA_STYLES["margin"]["color"]
    y_m   = ymax * 0.60
    for xv, mk in [(nl, "|"), (nr, "|")]:
        if xv is not None:
            ax.axvline(xv, color=col_m, lw=1.3, ls=":", alpha=0.85)
    ax.annotate("", xy=(theta_star + d_m_val, y_m),
                xytext=(theta_star - d_m_val, y_m),
                arrowprops=dict(arrowstyle="<->", color=col_m, lw=2.2))
    ax.text(theta_star, y_m + ymax * 0.05,
            f"$\\delta_{{\\mathrm{{M}}}}={d_m_val:.3f}$",
            ha="center", va="bottom", fontsize=FS_ANNOT,
            color=col_m, fontweight="bold")

    ax.set_xlim(lo_v, hi_v)
    ax.set_xlabel("Feature value", fontsize=FS_LABEL)
    ax.set_ylabel("Count", fontsize=FS_LABEL)
    ax.set_title("(b) Class-conditional distributions\n"
                 r"Class-overlap: $\delta_{\mathrm{CO}}$  "
                 r"|  Margin: $\delta_{\mathrm{M}}$",
                 fontsize=FS_TITLE, pad=8)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=7.5, loc="upper right")

    # --- Panel (c): Bootstrap distribution ----------------------------------
    ax = axs[2]
    boot_v = boot_arr[(boot_arr >= lo_v) & (boot_arr <= hi_v)]
    bins_b = np.linspace(lo_v, hi_v, 40)
    col_nb = DELTA_STYLES["node-bootstrap"]["color"]
    ax.hist(boot_v, bins=bins_b, color=col_nb, alpha=0.65)
    ax.axvline(theta_star, color=THRESH, lw=2.0, ls="-",
               label=f"$\\theta^*={theta_star:.2f}$")

    ymax_b = ax.get_ylim()[1]
    y_nb   = ymax_b * 0.78
    ax.annotate("", xy=(theta_star + d_nb_val, y_nb),
                xytext=(theta_star - d_nb_val, y_nb),
                arrowprops=dict(arrowstyle="<->", color=col_nb, lw=2.2))
    ax.text(theta_star, y_nb + ymax_b * 0.05,
            f"$\\delta_{{\\mathrm{{NB}}}}={d_nb_val:.3f}$",
            ha="center", va="bottom", fontsize=FS_ANNOT,
            color=col_nb, fontweight="bold")

    ax.set_xlim(lo_v, hi_v)
    ax.set_xlabel("Bootstrap optimal threshold $\\hat{\\theta}^*$",
                  fontsize=FS_LABEL)
    ax.set_ylabel("Frequency", fontsize=FS_LABEL)
    ax.set_title(f"(c) Bootstrap threshold distribution ($B={N_BOOTSTRAP}$)\n"
                 r"Node-bootstrap: $\delta_{\mathrm{NB}}$",
                 fontsize=FS_TITLE, pad=8)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=7.5, loc="upper right")

    # --- Panel (d): All five delta comparison --------------------------------
    ax = axs[3]
    ax.set_xlim(lo_v, hi_v + span * 0.55)
    ax.set_ylim(-0.6, len(deltas_ordered) - 0.4)
    ax.axvline(theta_star, color=THRESH, lw=2.0, ls="-", zorder=5,
               label=f"$\\theta^*={theta_star:.2f}$")
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="x", color="white", lw=1.2, zorder=1)

    row_labels = [
        r"Margin $\delta_{\mathrm{M}}$",
        r"Node-bootstrap $\delta_{\mathrm{NB}}$",
        r"Gain-ratio $\delta_{\mathrm{GR}}$",
        r"Class-overlap $\delta_{\mathrm{CO}}$",
        r"Quality-plateau $\delta_{\mathrm{QP}}$",
    ]
    ax.set_yticks(range(len(deltas_ordered)))
    ax.set_yticklabels(row_labels, fontsize=FS_TICK + 1)

    for row, (name, delta) in enumerate(deltas_ordered):
        col = DELTA_STYLES[name]["color"]
        lo_z, hi_z = theta_star - delta, theta_star + delta
        ax.barh(row, hi_z - lo_z, left=lo_z, height=0.52,
                color=col, alpha=0.38, zorder=2, edgecolor=col, linewidth=2.0)
        ax.text(hi_z + span * 0.02, row, f"{delta:.3f}",
                va="center", ha="left", fontsize=FS_TICK - 0.5,
                color=col, fontweight="bold")

    ax.set_xlabel("Feature value", fontsize=FS_LABEL)
    ax.set_title("(d) Uncertainty zone comparison\n"
                 "All five $\\delta$ methods at the same node",
                 fontsize=FS_TITLE, pad=8)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.legend(fontsize=8.5, loc="upper right")

    # ── Suptitle ──────────────────────────────────────────────────────────
    fig.suptitle(
        r"Motivating example: five $\delta$ methods applied to the same"
        r" synthetic split node ($n=300$, $\theta^*=$"
        f"{theta_star:.2f})"
        "\n"
        r"Each method estimates a different observable property of "
        r"the local error probability $p_e(d)$ (Section 3.2)",
        fontsize=FS_TITLE - 1, y=1.03, fontweight="bold")

    plt.subplots_adjust(top=0.84, wspace=0.62, left=0.05, right=0.97)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"figure5_motivating_example.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    make_figure()
