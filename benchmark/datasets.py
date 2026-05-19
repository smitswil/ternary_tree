"""
benchmark/datasets.py
=====================
Load the four sklearn benchmark datasets used in the comparison study.

All datasets are returned as (X, y, metadata_dict) tuples so the
benchmark runner knows what it is working with.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_digits,
    load_iris,
)


DatasetTuple = Tuple[np.ndarray, np.ndarray, Dict]


def load_all() -> List[DatasetTuple]:
    """Return list of (X, y, info) for all four benchmark datasets."""
    return [
        _load_iris(),
        _load_breast_cancer(),
        _load_wine(),
        _load_digits(),
    ]


def _load_iris() -> DatasetTuple:
    data = load_iris()
    return data.data, data.target, {
        "name"     : "Iris",
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_classes": len(np.unique(data.target)),
        "task"     : "multiclass",
    }


def _load_breast_cancer() -> DatasetTuple:
    data = load_breast_cancer()
    return data.data, data.target, {
        "name"     : "Breast Cancer",
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_classes": 2,
        "task"     : "binary",
    }


def _load_wine() -> DatasetTuple:
    data = load_wine()
    return data.data, data.target, {
        "name"     : "Wine",
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_classes": len(np.unique(data.target)),
        "task"     : "multiclass",
    }


def _load_digits() -> DatasetTuple:
    data = load_digits()
    return data.data, data.target, {
        "name"     : "Digits",
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_classes": len(np.unique(data.target)),
        "task"     : "multiclass",
    }
