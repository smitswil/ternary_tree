# Ternary Decision Trees with Locally-Adaptive Uncertainty Zones

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Paper

This repository accompanies the paper:

**Ternary Decision Trees with Locally-Adaptive Uncertainty Zones**
William Smits — *Data Mining and Knowledge Discovery (under review)*
arXiv: [XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

> Standard decision trees make hard binary routing decisions at each node,
> assigning identical confidence to instances far from a decision boundary
> and to those directly on it. We introduce ternary decision trees, which
> augment each split node with a locally-computed uncertainty zone.
> Instances whose feature value falls within this zone receive predictions
> formed by weighted blending of both child subtrees and are flagged as
> boundary-uncertain. Five methods for computing the zone half-width δ
> are proposed and evaluated across 72 OpenML-CC18 datasets, three
> Breiman synthetic benchmarks with known Bayes errors, and four
> medical and financial datasets.

---

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

Requires Python ≥ 3.9.

---

## Quick Start

```python
from ternary_tree import BinaryTernaryTree, TrinaryTree, make_classifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Recommended: margin delta method with probabilistic routing
clf = make_classifier(delta_method="margin", routing_method="probabilistic")
clf.fit(X_train, y_train)

pred    = clf.predict(X_test)           # class predictions
ternary = clf.predict_ternary(X_test)   # 1=decided, 0=boundary-uncertain
proba   = clf.predict_proba(X_test)     # probability distributions
```

---

## Five Delta Methods

| Method | Description | Hyperparameters |
|---|---|---|
| `margin` | Distance to nearest cross-class training example (recommended) | None |
| `node_bootstrap` | Threshold variance under node-level bootstrap resampling | `n_bootstraps` |
| `quality_plateau` | Plateau width of the split quality criterion curve | `epsilon` |
| `gain_ratio` | Inverse of information gain ratio at the node | `alpha` |
| `class_overlap` | Empirical class-distribution overlap width | `percentile` |

## Two Routing Architectures

| Method | Description |
|---|---|
| `probabilistic` | Binary tree structure; uncertain instances weighted across both children |
| `hard_middle` | True third branch; uncertain instances train a separate middle subtree |

---

## Reproducing the Benchmark

Install the benchmark dependencies:

```bash
uv pip install -e ".[benchmark]"
```

Run the three dataset collections:

```bash
# Breiman synthetics (~15 min)
python benchmark/run_extended_benchmark.py --collection breiman --output results_breiman.csv

# Medical / financial (~15 min)
python benchmark/run_extended_benchmark.py --collection medical --output results_medical.csv

# OpenML-CC18 (~3-4 hours)
python benchmark/run_extended_benchmark.py --collection cc18 --output results_cc18.csv
```

Generate paper tables:

```bash
python benchmark/report_paper.py \
    --cc18    results_cc18_repaired.csv \
    --breiman results_breiman.csv \
    --medical results_medical.csv \
    --output  paper_tables.txt
```

---

## Project Structure

```
ternary_tree/
├── ternary_tree/
│   ├── splitter.py              # O(n log n) split evaluation
│   ├── delta_methods.py         # Five local δ computation methods
│   ├── node.py                  # SplitNode / LeafNode dataclasses
│   ├── binary_ternary_tree.py   # Probabilistic + deferred routing
│   ├── trinary_tree.py          # Hard middle branch (true 3-child tree)
│   └── metrics.py               # Decided accuracy, undecided rate
├── benchmark/
│   ├── datasets_extended.py     # OpenML-CC18, Breiman, medical loaders
│   ├── run_extended_benchmark.py# Full benchmark runner with checkpointing
│   └── report_paper.py          # Paper-ready tables and statistical tests
├── tests/                       # 67 unit tests
├── pyproject.toml
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{smits2026ternary,
  author  = {Smits, William},
  title   = {Ternary Decision Trees with Locally-Adaptive Uncertainty Zones},
  journal = {Data Mining and Knowledge Discovery},
  year    = {2026},
  note    = {Under review. arXiv:XXXX.XXXXX}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
