"""
benchmark/report.py
===================
Generate formatted comparison tables from the benchmark results CSV.

Usage
-----
    python benchmark/report.py --input results_aggregated.csv
    python benchmark/report.py --input results_aggregated.csv --metric decided_accuracy_mean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


METRIC_DESCRIPTIONS = {
    "accuracy_all_mean"     : "Overall Accuracy (all instances)",
    "decided_accuracy_mean" : "Decided Accuracy (confident predictions only)",
    "undecided_rate_mean"   : "Undecided Rate (fraction abstained)",
    "decided_rate_mean"     : "Decided Rate (fraction with confident verdict)",
    "decided_f1_mean"       : "Decided F1 (macro, confident only)",
    "f1_all_mean"           : "F1 Macro (all instances)",
    "fit_time_s_mean"       : "Mean Fit Time (seconds)",
}


def pivot_table(
    df    : pd.DataFrame,
    metric: str = "decided_accuracy_mean",
) -> pd.DataFrame:
    """Pivot to (combo × dataset) table for one metric."""
    if metric not in df.columns:
        available = [c for c in df.columns if c.endswith("_mean")]
        print(f"Metric '{metric}' not found. Available: {available}")
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=["delta_method", "routing_method", "combo"],
        columns="dataset",
        values=metric,
        aggfunc="first",
    ).round(4)

    pivot["Mean"] = pivot.mean(axis=1).round(4)
    pivot = pivot.sort_values("Mean", ascending=False)
    return pivot


def print_report(df: pd.DataFrame) -> None:
    """Print a full comparison report for key metrics."""
    datasets = sorted(df["dataset"].unique())

    print("\n" + "="*80)
    print("TERNARY DECISION TREE BENCHMARK REPORT")
    print("="*80)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Combinations: {df['combo'].nunique()} (12 ternary + 1 baseline)")

    for metric, desc in METRIC_DESCRIPTIONS.items():
        if metric not in df.columns:
            continue
        print(f"\n{'─'*80}")
        print(f"  {desc}")
        print(f"{'─'*80}")
        pivot = pivot_table(df, metric)
        if not pivot.empty:
            print(pivot.to_string())

    # Best combination per dataset
    print(f"\n{'='*80}")
    print("  BEST COMBINATION PER DATASET (by Decided Accuracy)")
    print(f"{'='*80}")
    key_metric = "decided_accuracy_mean"
    if key_metric in df.columns:
        for dataset in datasets:
            sub = df[df["dataset"] == dataset].copy()
            if sub.empty or key_metric not in sub.columns:
                continue
            best = sub.loc[sub[key_metric].idxmax()]
            print(
                f"  {dataset:<20} → {best['combo']:<45} "
                f"dec_acc={best[key_metric]:.4f}  "
                f"undec={best.get('undecided_rate_mean', float('nan')):.3f}"
            )

    # Delta method summary (average across datasets and routing methods)
    print(f"\n{'='*80}")
    print("  DELTA METHOD COMPARISON (mean across datasets & routing methods)")
    print(f"{'='*80}")
    if key_metric in df.columns and "delta_method" in df.columns:
        dm_agg = (
            df[df["delta_method"] != "baseline"]
            .groupby("delta_method")[key_metric]
            .agg(["mean", "std"])
            .round(4)
            .sort_values("mean", ascending=False)
        )
        print(dm_agg.to_string())

    # Routing method summary
    print(f"\n{'='*80}")
    print("  ROUTING METHOD COMPARISON (mean across datasets & delta methods)")
    print(f"{'='*80}")
    if key_metric in df.columns and "routing_method" in df.columns:
        rm_agg = (
            df[df["routing_method"] != "sklearn_CART"]
            .groupby("routing_method")[key_metric]
            .agg(["mean", "std"])
            .round(4)
            .sort_values("mean", ascending=False)
        )
        print(rm_agg.to_string())


def main():
    parser = argparse.ArgumentParser(description="Ternary Tree Benchmark Report")
    parser.add_argument("--input",  required=True, help="Aggregated results CSV")
    parser.add_argument(
        "--metric",
        default="decided_accuracy_mean",
        help="Primary metric for pivot table",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print_report(df)

    print(f"\n{'─'*80}")
    print(f"  Pivot table: {args.metric}")
    print(f"{'─'*80}")
    pivot = pivot_table(df, args.metric)
    if not pivot.empty:
        print(pivot.to_string())


if __name__ == "__main__":
    main()
