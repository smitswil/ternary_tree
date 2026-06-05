"""
figure3_accuracy_coverage.py
============================
Figure 3: Accuracy-coverage tradeoff across OpenML-CC18.

Run from the project root:
    uv run python figures/figure3_accuracy_coverage.py --input results_cc18_repaired.csv
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Visual style — one entry per (delta_method, routing_method) combination
# ---------------------------------------------------------------------------

DELTA_COLORS = {
    "margin"         : "#E74C3C",   # red
    "node_bootstrap" : "#2980B9",   # blue
    "quality_plateau": "#27AE60",   # green
    "gain_ratio"     : "#E67E22",   # orange
    "class_overlap"  : "#8E44AD",   # purple
    "baseline"       : "black",
}

ROUTING_MARKERS = {
    "probabilistic": "o",
    "hard_middle"  : "D",
    "sklearn_CART" : "*",
}

DELTA_LABELS = {
    "margin"         : "Margin",
    "node_bootstrap" : "Node-Bootstrap",
    "quality_plateau": "Quality-Plateau",
    "gain_ratio"     : "Gain-Ratio",
    "class_overlap"  : "Class-Overlap",
    "baseline"       : "CART (baseline)",
}

ROUTING_LABELS_SHORT = {
    "probabilistic": "prob.",
    "hard_middle"  : "h.m.",
    "sklearn_CART" : "",
}

EFFICIENCY_LEVELS = [0.05, 0.10, 0.15]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_aggregate(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    # Drop deferred — proven identical to probabilistic in our study
    df = df[df["routing_method"] != "deferred"]
    return (
        df.groupby(["combo", "delta_method", "routing_method"])
          .agg(
              decided_accuracy=("decided_accuracy", "mean"),
              undecided_rate  =("undecided_rate",   "mean"),
              n_datasets      =("decided_accuracy", "count"),
          )
          .reset_index()
          .round(4)
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(agg):
    baseline_row = agg[agg["delta_method"] == "baseline"]
    bl_acc = float(baseline_row["decided_accuracy"].iloc[0]) \
             if not baseline_row.empty else 0.70

    # Axis limits that include EVERY combination
    x_max = float(agg["undecided_rate"].max()) * 1.12 + 0.03
    y_min = float(agg["decided_accuracy"].min()) - 0.03
    y_max = float(agg["decided_accuracy"].max()) + 0.025

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # ── Efficiency isolines ──────────────────────────────────────────────
    x_iso = np.linspace(0.001, x_max, 400)
    for eta in EFFICIENCY_LEVELS:
        y_iso = bl_acc + eta * x_iso
        vis   = (y_iso >= y_min) & (y_iso <= y_max)
        if vis.any():
            ax.plot(x_iso[vis], y_iso[vis],
                    linestyle=":", color="#BDC3C7", linewidth=0.9, zorder=1)
            xi, yi = float(x_iso[vis][-1]), float(y_iso[vis][-1])
            ax.text(xi + 0.004, yi, f"$\\eta$={eta:.2f}",
                    fontsize=8, color="#95A5A6", va="center", zorder=2)

    # ── Baseline reference line (label on LEFT side near y-axis) ────────
    ax.axhline(bl_acc, color="#BDC3C7", lw=0.9, linestyle="--", zorder=1)

    # ── Better-than-baseline shading ────────────────────────────────────
    ax.axhspan(bl_acc, y_max + 0.02, alpha=0.04, color="green", zorder=0)
    ax.text(x_max * 0.55, bl_acc + 0.008, "Improved decided accuracy",
            fontsize=8.5, color="#27AE60", style="italic", zorder=2)

    # ── Plot every combination ───────────────────────────────────────────
    for _, row in agg.sort_values("undecided_rate").iterrows():
        dm = row["delta_method"]
        rm = row["routing_method"]
        x  = float(row["undecided_rate"])
        y  = float(row["decided_accuracy"])

        color  = DELTA_COLORS.get(dm, "#95A5A6")
        marker = ROUTING_MARKERS.get(rm, "o")

        is_baseline    = dm == "baseline"
        is_recommended = dm in ("margin", "node_bootstrap") and rm == "probabilistic"

        size    = 280 if is_baseline else 110
        edge_c  = "black"
        edge_lw = 2.2 if (is_baseline or is_recommended) else 0.7
        zorder  = 10 if is_baseline else (7 if is_recommended else 4)

        ax.scatter(x, y,
                   c=color, marker=marker, s=size,
                   edgecolors=edge_c, linewidths=edge_lw,
                   zorder=zorder, alpha=0.92)

    # ── Label EVERY combination ──────────────────────────────────────────
    # Build a clean label for each point and place it without arrows
    for _, row in agg.sort_values("undecided_rate").iterrows():
        dm = row["delta_method"]
        rm = row["routing_method"]
        x  = float(row["undecided_rate"])
        y  = float(row["decided_accuracy"])

        dlabel = DELTA_LABELS.get(dm, dm)
        rlabel = ROUTING_LABELS_SHORT.get(rm, "")
        label  = f"{dlabel}" + (f"\n({rlabel})" if rlabel else "")

        is_baseline    = dm == "baseline"
        is_recommended = dm in ("margin", "node_bootstrap") and rm == "probabilistic"
        fontsize = 8.5 if (is_baseline or is_recommended) else 7.5
        fontw    = "bold" if (is_baseline or is_recommended) else "normal"

        # Offset label to avoid sitting on the marker
        # Heuristic: place text to the right if there is room, else left
        ha = "left"
        offset_x = x_max * 0.015
        offset_y = (y_max - y_min) * 0.018

        # Nudge up/down based on y position to reduce overlap
        if y < (y_min + y_max) / 2:
            offset_y = -offset_y

        ax.annotate(
            label,
            xy    =(x, y),
            xytext=(x + offset_x, y + offset_y),
            fontsize=fontsize,
            fontweight=fontw,
            ha=ha, va="center",
            zorder=9,
            arrowprops=dict(
                arrowstyle="-",
                color="#AAAAAA",
                lw=0.5,
            ),
        )

    # ── Axes ────────────────────────────────────────────────────────────
    ax.set_xlim(-0.01, x_max)
    ax.set_ylim(y_min,  y_max)
    ax.set_xlabel("Mean Boundary-Uncertain Rate", fontsize=12)
    ax.set_ylabel("Mean Decided Accuracy",        fontsize=12)
    ax.set_title(
        "Accuracy-Coverage Tradeoff — All Combinations on OpenML-CC18",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.tick_params(labelsize=10)

    # ── Legend ───────────────────────────────────────────────────────────
    # One entry per (delta_method, routing_method): color encodes method,
    # shape encodes routing.  11 entries total (5 × 2 + 1 CART).
    legend_entries = []
    for dm, color in [("margin",          "#E74C3C"),
                      ("node_bootstrap",  "#2980B9"),
                      ("quality_plateau", "#27AE60"),
                      ("gain_ratio",      "#E67E22"),
                      ("class_overlap",   "#8E44AD")]:
        dlabel = DELTA_LABELS[dm]
        legend_entries.append(
            mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                          markersize=9, markeredgecolor="black",
                          markeredgewidth=0.5,
                          label=f"{dlabel}  (prob.)")
        )
        legend_entries.append(
            mlines.Line2D([], [], color=color, marker="D", linestyle="None",
                          markersize=8, markeredgecolor="black",
                          markeredgewidth=0.5,
                          label=f"{dlabel}  (h.m.)")
        )

    legend_entries.append(
        mlines.Line2D([], [], color="black", marker="*", linestyle="None",
                      markersize=14, label="CART (baseline)")
    )

    ax.legend(handles=legend_entries,
              title="Method  |  Routing\n"
                    "prob. = probabilistic,  h.m. = hard-middle",
              title_fontsize=8, fontsize=8,
              loc="lower right", ncol=2,
              framealpha=0.93, borderpad=0.8,
              labelspacing=0.45)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"figure3_accuracy_coverage.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_cc18_repaired.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    agg = load_and_aggregate(args.input)

    print(f"Loaded {agg['combo'].nunique()} combinations (deferred excluded)\n")
    print(f"{'Combination':<45} {'x (undec)':>10} {'y (dec_acc)':>12} {'color':>10}")
    print("-"*80)
    for _, r in agg.sort_values("undecided_rate").iterrows():
        col = DELTA_COLORS.get(r["delta_method"], "grey")
        mrk = ROUTING_MARKERS.get(r["routing_method"], "o")
        print(f"  {r['combo']:<43} {r['undecided_rate']:>10.4f} "
              f"{r['decided_accuracy']:>12.4f}  {col} {mrk}")

    plot(agg)


if __name__ == "__main__":
    main()
