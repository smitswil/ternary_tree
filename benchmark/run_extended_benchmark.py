"""
run_extended_benchmark.py
=========================
Runs the full three-collection paper benchmark.

Collections
-----------
    cc18      — OpenML-CC18 (72 datasets); broad validity claim
    breiman   — waveform-5000, twonorm, ringnorm; Bayes error validation
    medical   — diabetes, credit-g, heart-c, mammography; domain narrative

Design decisions for the extended benchmark
-------------------------------------------
  * Only 'probabilistic' routing is run (deferred is proven identical in the
    pilot study on 4 datasets).  This halves computation vs running both.

  * Hard-middle routing is included only for completeness; the pilot study
    showed it is unreliable on high-dimensional multiclass problems.

  * Bootstrap replicates are scaled to dataset size:
      N < 2,000   → n_bootstraps=20
      N < 10,000  → n_bootstraps=15
      N >= 10,000 → n_bootstraps=10

  * node_bootstrap is disabled on datasets with N > NODE_BOOTSTRAP_MAX_SAMPLES
    (default 20,000).  On large datasets the bootstrap loop is too slow to
    be practical.  All other delta methods run on every dataset.  This is
    reported in the paper as a computational constraint, not a methodological
    one — reviewers routinely accept this for expensive methods.

  * Results are written to CSV incrementally (append after each fold) so that
    a crashed run can be resumed with --resume.

Usage
-----
    # Quick test (3 CC18 datasets, 2 folds)
    python benchmark/run_extended_benchmark.py --collection cc18 --max-datasets 3 --cv 2

    # Full CC18 run
    python benchmark/run_extended_benchmark.py --collection cc18 --output results_cc18.csv

    # Breiman synthetics
    python benchmark/run_extended_benchmark.py --collection breiman --output results_breiman.csv

    # Medical / financial
    python benchmark/run_extended_benchmark.py --collection medical --output results_medical.csv

    # All three (sequential)
    python benchmark/run_extended_benchmark.py --collection all
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from ternary_tree import make_classifier
from ternary_tree.metrics import ternary_summary

from benchmark.datasets_extended import (
    load_openml_cc18,
    load_breiman_synthetics,
    load_medical_financial,
)

# ---------------------------------------------------------------------------
# Combinations to evaluate
# (deferred routing is omitted — proven identical to probabilistic in pilot)
# ---------------------------------------------------------------------------

DELTA_METHODS   = ["quality_plateau", "class_overlap", "gain_ratio",
                   "node_bootstrap", "margin"]
ROUTING_METHODS = ["probabilistic", "hard_middle"]

# All ternary combinations (5 delta × 2 routing = 10) + baseline
ALL_COMBOS = [(d, r) for d in DELTA_METHODS for r in ROUTING_METHODS]
ALL_COMBOS.append(("baseline", "sklearn_CART"))

# node_bootstrap is disabled when N exceeds this threshold
NODE_BOOTSTRAP_MAX_SAMPLES = 20_000


def _get_combos_for_dataset(n_samples: int) -> List[Tuple[str, str]]:
    """Return the combination list appropriate for this dataset size.

    node_bootstrap is excluded when N > NODE_BOOTSTRAP_MAX_SAMPLES because
    running 10-20 mini-trees per node per bootstrap replicate becomes
    prohibitively slow on large datasets.  All other delta methods run
    regardless of N.  The exclusion is logged in the results CSV via the
    'node_bootstrap_excluded' column so it is transparent in the paper.
    """
    if n_samples > NODE_BOOTSTRAP_MAX_SAMPLES:
        return [
            (d, r) for d, r in ALL_COMBOS
            if d != "node_bootstrap"
        ]
    return ALL_COMBOS


# ---------------------------------------------------------------------------
# Adaptive config based on dataset size
# ---------------------------------------------------------------------------

def _tree_config(n_samples: int, max_depth: int) -> Dict:
    if n_samples >= 10_000:
        return dict(max_depth=max_depth, n_bootstraps=10)
    if n_samples >= 2_000:
        return dict(max_depth=max_depth, n_bootstraps=15)
    return dict(max_depth=max_depth, n_bootstraps=20)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_completed(output_path: str) -> Set[Tuple]:
    """Return set of (dataset_name, fold, combo) already in the output CSV."""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path)
        return set(zip(df["dataset"], df["fold"], df["combo"]))
    except Exception:
        return set()


def _append_row(output_path: str, row: Dict, write_header: bool) -> None:
    mode = "a"
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()),
                                quoting=csv.QUOTE_ALL)   # ← add this
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Single-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    X          : np.ndarray,
    y          : np.ndarray,
    info       : Dict,
    cv         : int,
    max_depth  : int,
    seed       : int,
    output_path: str,
    completed  : Set[Tuple],
    verbose    : bool,
) -> List[Dict]:
    """Run all combinations on one dataset, with checkpoint awareness."""
    skf      = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    cfg      = _tree_config(len(y), max_depth)
    results  = []
    name     = info["name"]
    first_write = not os.path.exists(output_path)

    n_samples = len(y)
    combos    = _get_combos_for_dataset(n_samples)
    nb_excluded = n_samples > NODE_BOOTSTRAP_MAX_SAMPLES
    if nb_excluded and verbose:
        print(f"    [node_bootstrap disabled: N={n_samples} > "
              f"{NODE_BOOTSTRAP_MAX_SAMPLES}]")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        for delta_m, routing_m in combos:
            combo = f"{delta_m}__{routing_m}"

            # Skip if already computed
            if (name, fold_idx, combo) in completed:
                continue

            t0 = time.perf_counter()
            try:
                if delta_m == "baseline":
                    clf = DecisionTreeClassifier(
                        max_depth=max_depth, random_state=seed
                    )
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    ternary_v = np.ones(len(y_te), dtype=np.int8)
                else:
                    clf = make_classifier(
                        delta_method=delta_m,
                        routing_method=routing_m,
                        random_state=seed,
                        **cfg,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(X_tr, y_tr)
                    y_pred    = clf.predict(X_te)
                    ternary_v = clf.predict_ternary(X_te)

                fit_time = time.perf_counter() - t0
                metrics  = ternary_summary(y_te, y_pred, ternary_v)

                row = {
                    "dataset"               : name,
                    "collection"            : info.get("collection", ""),
                    "n_samples"             : info["n_samples"],
                    "n_features"            : info["n_features"],
                    "n_classes"             : info["n_classes"],
                    "node_bootstrap_excluded": int(nb_excluded),
                    "fold"                  : fold_idx,
                    "delta_method"          : delta_m,
                    "routing_method"        : routing_m,
                    "combo"                 : combo,
                    "fit_time_s"            : round(fit_time, 5),
                    **{k: round(v, 6) if isinstance(v, float) else v
                       for k, v in metrics.items()},
                }
                # Add Bayes error if available
                if "bayes_error" in info:
                    row["bayes_error"] = info["bayes_error"]

                _append_row(output_path, row, write_header=first_write)
                first_write = False
                results.append(row)

                if verbose:
                    print(
                        f"    {combo:<42} fold={fold_idx} "
                        f"dec_acc={metrics['decided_accuracy']:.4f} "
                        f"undec={metrics['undecided_rate']:.3f} "
                        f"t={fit_time:.2f}s"
                    )

            except Exception as e:
                fit_time = time.perf_counter() - t0
                if verbose:
                    print(f"    {combo} fold={fold_idx} ERROR: {e}")
                row = {
                    "dataset": name, "collection": info.get("collection",""),
                    "fold": fold_idx, "delta_method": delta_m,
                    "routing_method": routing_m, "combo": combo,
                    "error": str(e)[:200],
                }
                _append_row(output_path, row, write_header=first_write)
                first_write = False

    return results


# ---------------------------------------------------------------------------
# Collection runners
# ---------------------------------------------------------------------------

def run_collection(
    collection  : str,
    output_path : str,
    cv          : int  = 5,
    max_depth   : int  = 4,
    seed        : int  = 42,
    max_datasets: Optional[int] = None,
    verbose     : bool = True,
    resume      : bool = True,
) -> pd.DataFrame:
    """Run one complete collection and return results DataFrame.

    Parameters
    ----------
    collection   : 'cc18' | 'breiman' | 'medical'
    output_path  : CSV file for incremental results (checkpointed)
    cv           : number of cross-validation folds
    max_depth    : maximum tree depth
    seed         : random seed
    max_datasets : cap on dataset count (None = all)
    verbose      : print progress
    resume       : skip already-computed rows in output_path
    """
    print(f"\n{'='*70}")
    print(f"Collection: {collection.upper()}")
    print(f"Output:     {output_path}")
    print(f"Settings:   cv={cv}, max_depth={max_depth}, seed={seed}")
    print(f"{'='*70}\n")

    # Load datasets
    if collection == "cc18":
        datasets = load_openml_cc18(
            max_datasets=max_datasets, verbose=verbose
        )
    elif collection == "breiman":
        datasets = load_breiman_synthetics(verbose=verbose)
    elif collection == "medical":
        datasets = load_medical_financial(verbose=verbose)
    else:
        raise ValueError(f"Unknown collection '{collection}'. "
                         "Choose: 'cc18', 'breiman', 'medical'")

    if not datasets:
        print("No datasets loaded. Check your OpenML connection.")
        return pd.DataFrame()

    completed = _load_completed(output_path) if resume else set()
    if completed and verbose:
        print(f"Resuming: {len(completed)} (dataset, fold, combo) "
              f"combinations already completed.\n")

    all_results = []
    for i, (X, y, info) in enumerate(datasets):
        name = info["name"]
        print(f"[{i+1}/{len(datasets)}] {name}  "
              f"(n={info['n_samples']}, d={info['n_features']}, "
              f"k={info['n_classes']})")
        results = evaluate_dataset(
            X, y, info, cv, max_depth, seed,
            output_path, completed, verbose,
        )
        all_results.extend(results)
        print()

    if os.path.exists(output_path):
        return pd.read_csv(output_path, engine='python', on_bad_lines='skip')
    return pd.DataFrame(all_results)


def run_all_collections(
    output_dir  : str  = ".",
    cv          : int  = 5,
    max_depth   : int  = 4,
    seed        : int  = 42,
    verbose     : bool = True,
    resume      : bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run all three collections sequentially."""
    results = {}
    collections = {
        "cc18"   : "results_cc18.csv",
        "breiman": "results_breiman.csv",
        "medical": "results_medical.csv",
    }
    for name, fname in collections.items():
        path = os.path.join(output_dir, fname)
        df = run_collection(
            collection=name, output_path=path,
            cv=cv, max_depth=max_depth, seed=seed,
            verbose=verbose, resume=resume,
        )
        results[name] = df
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ternary Decision Tree — Extended Paper Benchmark"
    )
    parser.add_argument(
        "--collection", default="all",
        choices=["cc18", "breiman", "medical", "all"],
        help="Which dataset collection to run",
    )
    parser.add_argument("--output", default=None,
                        help="Output CSV path (auto-named if not set)")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for --collection all")
    parser.add_argument("--cv",    type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None,
                        help="Limit dataset count (useful for testing)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Recompute everything (ignore existing output)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    resume  = not args.no_resume

    if args.collection == "all":
        run_all_collections(
            output_dir=args.output_dir,
            cv=args.cv, max_depth=args.depth, seed=args.seed,
            verbose=verbose, resume=resume,
        )
    else:
        output_path = args.output or f"results_{args.collection}.csv"
        run_collection(
            collection=args.collection,
            output_path=output_path,
            cv=args.cv, max_depth=args.depth, seed=args.seed,
            max_datasets=args.max_datasets,
            verbose=verbose, resume=resume,
        )


if __name__ == "__main__":
    main()