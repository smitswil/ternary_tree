"""
datasets_extended.py
====================
Dataset loaders for the three paper benchmark collections.

Collection 1 — OpenML-CC18 (72 datasets)
    The standard curated classification benchmark suite (Bischl et al. 2021).
    Provides the broad validity claim for the paper.
    Requires:  pip install openml

Collection 2 — Breiman Synthetic Benchmarks
    waveform-5000, twonorm, ringnorm  —  Breiman's classic datasets with
    analytically-known Bayes errors.  Lets us validate that the undecided rate
    tracks the theoretical classification uncertainty.

Collection 3 — Medical & Financial High-Stakes Datasets
    Pima Diabetes, German Credit, Cleveland Heart Disease, Mammography.
    Provides the domain narrative for the paper: abstention is clinically /
    economically meaningful on these problems.

Preprocessing (applied consistently to all datasets)
-----------------------------------------------------
1. OrdinalEncoder  for any categorical (object-dtype) feature columns
2. SimpleImputer(strategy='median')  for missing numerical values
3. LabelEncoder    for the target variable
4. StandardScaler  is applied in the benchmark runner, not here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ---------------------------------------------------------------------------
# Known Bayes errors for Breiman synthetic benchmarks
# (from Breiman et al. 1984 and subsequent theoretical analyses)
# ---------------------------------------------------------------------------
BAYES_ERRORS: Dict[str, float] = {
    "waveform-5000": 0.140,   # ~14.0% irreducible error
    "twonorm":       0.023,   # ~2.3%
    "ringnorm":      0.017,   # ~1.7%
}

# Domain narrative for medical / financial datasets
DOMAIN_NARRATIVE: Dict[str, str] = {
    "diabetes":      "Pima diabetes screening — abstention triggers clinical follow-up",
    "credit-g":      "German credit risk — uncertain applicants go to manual review",
    "heart-c":       "Cleveland heart disease — abstention leads to further cardiac tests",
    "mammography":   "Cancer screening — severe imbalance; abstention avoids false negatives",
}

# Max dataset size for full benchmark (rows × features); skip larger
MAX_CELLS = 100_000_000  # node_bootstrap gated separately in runner


# ---------------------------------------------------------------------------
# Internal preprocessing helper
# ---------------------------------------------------------------------------

def _preprocess(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """Convert any OpenML dataset to clean float X and int y."""
    import pandas as pd

    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    X = np.array(X, dtype=object)

    # Identify categorical columns (non-numeric)
    cat_cols = []
    for j in range(X.shape[1]):
        try:
            X[:, j].astype(float)
        except (ValueError, TypeError):
            cat_cols.append(j)

    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        X[:, cat_cols] = enc.fit_transform(X[:, cat_cols])

    X = X.astype(float)

    # Impute missing values
    if np.isnan(X).any():
        imp = SimpleImputer(strategy="median")
        X   = imp.fit_transform(X)

    # Encode target
    le = LabelEncoder()
    y  = le.fit_transform(np.array(y, dtype=str))

    return X.astype(np.float64), y.astype(int)


# ---------------------------------------------------------------------------
# Collection 1: OpenML-CC18
# ---------------------------------------------------------------------------

def load_openml_cc18(
    max_datasets : Optional[int] = None,
    cache_dir    : Optional[str] = None,
    verbose      : bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Load all (or a subset of) OpenML-CC18 datasets.

    Parameters
    ----------
    max_datasets : limit for fast testing (None = all 72)
    cache_dir    : local OpenML cache directory
    verbose      : print progress

    Returns
    -------
    list of (X, y, info_dict)
    """
    try:
        import openml
    except ImportError:
        raise ImportError(
            "OpenML-CC18 requires the openml package:\n"
            "    uv pip install openml\n"
            "or\n"
            "    pip install openml"
        )

    if cache_dir:
        openml.config.cache_directory = cache_dir

    if verbose:
        print("Fetching OpenML-CC18 task list …")

    try:
        suite    = openml.study.get_suite("OpenML-CC18")
        task_ids = suite.tasks
    except Exception as e:
        warnings.warn(f"Could not fetch CC18 suite ({e}). Using fallback task ID list.")
        task_ids = _CC18_TASK_IDS_FALLBACK

    if max_datasets:
        task_ids = task_ids[:max_datasets]

    datasets = []
    skipped  = []

    for i, task_id in enumerate(task_ids):
        if verbose:
            print(f"  [{i+1}/{len(task_ids)}] task {task_id} …", end=" ")
        try:
            task    = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            X, y, _, _ = dataset.get_data(
                target=task.target_name,
                dataset_format="array"
            )
            X, y = _preprocess(X, y)

            n, d = X.shape
            if n * d > MAX_CELLS:
                if verbose:
                    print(f"SKIP (too large: {n}×{d})")
                skipped.append(task_id)
                continue

            info = {
                "name"       : dataset.name,
                "task_id"    : task_id,
                "dataset_id" : dataset.dataset_id,
                "n_samples"  : n,
                "n_features" : d,
                "n_classes"  : len(np.unique(y)),
                "collection" : "OpenML-CC18",
            }
            datasets.append((X, y, info))
            if verbose:
                print(f"OK  (n={n}, d={d}, k={info['n_classes']})")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            skipped.append(task_id)

    if verbose:
        print(f"\nLoaded {len(datasets)} / {len(task_ids)} datasets "
              f"({len(skipped)} skipped)")

    return datasets


# ---------------------------------------------------------------------------
# Collection 2: Breiman Synthetic Benchmarks
# ---------------------------------------------------------------------------

def load_breiman_synthetics(verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Load waveform-5000, twonorm, and ringnorm from OpenML.

    These datasets have analytically-known Bayes errors, which allows
    us to validate that the undecided rate tracks theoretical uncertainty.
    """
    try:
        import openml
    except ImportError:
        raise ImportError("pip install openml")

    # OpenML dataset IDs for Breiman synthetics
    breiman_ids = {
        "waveform-5000": 60,
        "twonorm"      : 1507,
        "ringnorm"     : 1496,
    }

    datasets = []

    for name, dataset_id in breiman_ids.items():
        if verbose:
            print(f"  Loading {name} (OpenML id={dataset_id}) …", end=" ")
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="array"
            )
            X, y = _preprocess(X, y)

            info = {
                "name"       : name,
                "dataset_id" : dataset_id,
                "n_samples"  : X.shape[0],
                "n_features" : X.shape[1],
                "n_classes"  : len(np.unique(y)),
                "bayes_error": BAYES_ERRORS[name],
                "collection" : "Breiman",
            }
            datasets.append((X, y, info))
            if verbose:
                print(f"OK  (n={X.shape[0]}, d={X.shape[1]}, "
                      f"Bayes error={BAYES_ERRORS[name]:.1%})")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Collection 3: Medical & Financial Datasets
# ---------------------------------------------------------------------------

def load_medical_financial(verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Load four medical / financial high-stakes classification datasets.

    All fetched from OpenML for reproducibility.
    """
    try:
        import openml
    except ImportError:
        raise ImportError("pip install openml")

    # OpenML dataset IDs
    targets = {
        "diabetes"   : (37,  "class"),       # Pima Indians Diabetes
        "credit-g"   : (31,  "class"),       # German Credit
        "heart-c"    : (53,  "class"),       # Cleveland Heart Disease
        "mammography": (310, "class"),       # Mammography (imbalanced)
    }

    datasets = []

    for name, (dataset_id, target_col) in targets.items():
        if verbose:
            print(f"  Loading {name} (OpenML id={dataset_id}) …", end=" ")
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            target  = dataset.default_target_attribute or target_col
            X, y, _, _ = dataset.get_data(
                target=target, dataset_format="array"
            )
            X, y = _preprocess(X, y)

            info = {
                "name"           : name,
                "dataset_id"     : dataset_id,
                "n_samples"      : X.shape[0],
                "n_features"     : X.shape[1],
                "n_classes"      : len(np.unique(y)),
                "domain_narrative": DOMAIN_NARRATIVE.get(name, ""),
                "collection"     : "Medical-Financial",
            }
            datasets.append((X, y, info))
            if verbose:
                class_balance = np.bincount(y) / len(y)
                print(f"OK  (n={X.shape[0]}, d={X.shape[1]}, "
                      f"balance={class_balance.round(2).tolist()})")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Fallback CC18 task ID list (used if OpenML API is unavailable)
# ---------------------------------------------------------------------------

_CC18_TASK_IDS_FALLBACK = [
    3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 38,
    44, 46, 50, 54, 151, 182, 188, 458, 469, 554, 1049, 1050, 1053,
    1063, 1067, 1068, 1590, 4134, 4534, 4538, 6332, 7592, 9910, 9946,
    9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9981, 9985, 10093,
    10101, 14954, 14965, 14969, 14970, 125920, 125922, 167119, 167120,
    167124, 167125, 167140, 167141, 167168, 167181, 167184, 167185,
    167190, 167200, 167201,
]