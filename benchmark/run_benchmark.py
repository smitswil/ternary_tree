"""
benchmark/run_benchmark.py
==========================
Runs the full 12-combination benchmark:

    4 delta methods × 3 routing methods × 4 datasets × 5-fold CV

Plus sklearn DecisionTreeClassifier as baseline.

Usage
-----
    python benchmark/run_benchmark.py [--output results.csv] [--cv 5] [--depth 4]

Results are saved to CSV and printed as a formatted table.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Add parent dir to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from ternary_tree import make_classifier, DELTA_METHODS
from ternary_tree.metrics import ternary_summary
from benchmark.datasets import load_all

DELTA_METHODS_LIST  = ["quality_plateau", "class_overlap", "gain_ratio", "node_bootstrap", "margin"]
ROUTING_METHODS_LIST= ["probabilistic", "deferred", "hard_middle"]


def run_benchmark(
    cv       : int  = 5,
    max_depth: int  = 4,
    seed     : int  = 42,
    scale    : bool = True,
    verbose  : bool = True,
) -> pd.DataFrame:
    """Run the full benchmark and return results DataFrame."""

    datasets = load_all()
    skf      = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    records  = []

    # All 12 ternary combinations
    combinations = list(product(DELTA_METHODS_LIST, ROUTING_METHODS_LIST))

    # Add sklearn baseline
    baseline = ("baseline", "sklearn_CART")

    all_combos = list(combinations) + [baseline]

    for X, y, info in datasets:
        dataset_name = info["name"]
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}  "
                  f"(n={info['n_samples']}, d={info['n_features']}, "
                  f"classes={info['n_classes']})")
            print(f"{'='*60}")

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if scale:
                scaler  = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

            for delta_m, routing_m in all_combos:
                combo_name = f"{delta_m}__{routing_m}"
                is_baseline = (delta_m == "baseline")

                t0 = time.perf_counter()

                try:
                    if is_baseline:
                        clf = DecisionTreeClassifier(
                            max_depth=max_depth, random_state=seed
                        )
                        clf.fit(X_train, y_train)
                        y_pred    = clf.predict(X_test)
                        # Baseline is always "decided"
                        ternary_v = np.ones(len(y_test), dtype=np.int8)
                    else:
                        clf = make_classifier(
                            delta_method=delta_m,
                            routing_method=routing_m,
                            max_depth=max_depth,
                            random_state=seed,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(X_train, y_train)
                        y_pred    = clf.predict(X_test)
                        ternary_v = clf.predict_ternary(X_test)

                    fit_time = time.perf_counter() - t0

                    metrics  = ternary_summary(y_test, y_pred, ternary_v)
                    record   = {
                        "dataset"       : dataset_name,
                        "fold"          : fold_idx,
                        "delta_method"  : delta_m,
                        "routing_method": routing_m,
                        "combo"         : combo_name,
                        "fit_time_s"    : round(fit_time, 4),
                        **metrics,
                    }
                    records.append(record)

                    if verbose:
                        print(
                            f"  [{combo_name[:40]:<40}] "
                            f"fold={fold_idx}  "
                            f"acc={metrics['accuracy_all']:.3f}  "
                            f"dec_acc={metrics['decided_accuracy']:.3f}  "
                            f"undec={metrics['undecided_rate']:.2f}  "
                            f"t={fit_time:.2f}s"
                        )

                except Exception as e:
                    if verbose:
                        print(f"  [{combo_name}] fold={fold_idx} ERROR: {e}")
                    records.append({
                        "dataset": dataset_name, "fold": fold_idx,
                        "delta_method": delta_m, "routing_method": routing_m,
                        "combo": combo_name, "error": str(e),
                    })

    return pd.DataFrame(records)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold results to mean ± std across folds per (dataset, combo)."""
    metric_cols = [
        "accuracy_all", "decided_rate", "undecided_rate",
        "decided_accuracy", "decided_f1", "f1_all", "fit_time_s",
    ]
    available = [c for c in metric_cols if c in df.columns]

    agg = (
        df.groupby(["dataset", "combo", "delta_method", "routing_method"])[available]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c).strip() for c in agg.columns]
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Ternary Decision Tree Benchmark")
    parser.add_argument("--output",   default="results.csv",  help="Output CSV file")
    parser.add_argument("--cv",       type=int, default=5,     help="Number of CV folds")
    parser.add_argument("--depth",    type=int, default=4,     help="Max tree depth")
    parser.add_argument("--seed",     type=int, default=42,    help="Random seed")
    parser.add_argument("--no-scale", action="store_true",     help="Skip StandardScaler")
    parser.add_argument("--quiet",    action="store_true",     help="Suppress per-fold output")
    args = parser.parse_args()

    df = run_benchmark(
        cv=args.cv,
        max_depth=args.depth,
        seed=args.seed,
        scale=not args.no_scale,
        verbose=not args.quiet,
    )

    df.to_csv(args.output, index=False)
    print(f"\nRaw results saved to {args.output}")

    agg = aggregate_results(df)
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (mean across folds)")
    print("="*80)
    print(agg.to_string(index=False))

    # Also save aggregated
    agg_path = args.output.replace(".csv", "_aggregated.csv")
    agg.to_csv(agg_path, index=False)
    print(f"\nAggregated results saved to {agg_path}")


if __name__ == "__main__":
    main()
