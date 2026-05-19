"""
report_paper.py
===============
Generate paper-ready analysis tables and statistics from benchmark results.

Fixes applied vs original
--------------------------
  * All file writes use encoding='utf-8' (fixes Windows cp1252 UnicodeEncodeError)
  * Wilcoxon test guards against zero-difference case (fixes RuntimeWarning)
  * All pd.read_csv calls use engine='python', on_bad_lines='skip' (robust parsing)
  * Special characters in table separators replaced with ASCII equivalents
  * Centralised _write() and _read_csv() helpers eliminate repetition

Usage
-----
    python benchmark/report_paper.py \\
        --cc18    results_cc18_clean.csv \\
        --breiman results_breiman.csv \\
        --medical results_medical.csv \\
        --output  paper_tables.txt
"""

from __future__ import annotations

import argparse
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed -- Wilcoxon tests will be skipped.")

METHOD_LABELS = {
    "node_bootstrap" : "Node-Bootstrap",
    "margin"         : "Margin (ours)",
    "quality_plateau": "Quality-Plateau",
    "gain_ratio"     : "Gain-Ratio",
    "class_overlap"  : "Class-Overlap",
    "baseline"       : "CART (baseline)",
}

ROUTING_LABELS = {
    "probabilistic": "prob.",
    "hard_middle"  : "h.m.",
    "sklearn_CART" : "--",
}

WIN_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    """Read results CSV robustly regardless of quoting or encoding issues."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, engine="python",
                               on_bad_lines="skip", encoding=enc)
        except Exception:
            continue
    raise IOError(f"Could not read {path} with any supported encoding.")


def _write(output_file: Optional[str], text: str) -> None:
    """Append text to output file using UTF-8 encoding."""
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")

def _fmt_pval(p) -> str:
    """Format p-value for LaTeX: show <0.001 instead of 0.0 for tiny values."""
    if p == "--" or (isinstance(p, float) and (p != p)):  # nan or placeholder
        return "--"
    try:
        pf = float(p)
        if pf < 0.001:
            return "$<$0.001"
        return str(p)
    except (TypeError, ValueError):
        return str(p)




# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["accuracy_all", "decided_rate", "undecided_rate",
                    "decided_accuracy", "decided_f1", "f1_all", "fit_time_s"]
    available = [c for c in numeric_cols if c in df.columns]
    return (
        df.groupby(["dataset", "combo", "delta_method", "routing_method"])[available]
        .mean()
        .reset_index()
    )


def _pivot(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    return agg.pivot_table(index="combo", columns="dataset",
                           values=metric, aggfunc="first")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_vs_baseline(
    agg     : pd.DataFrame,
    method  : str,
    baseline: str = "baseline__sklearn_CART",
    metric  : str = "decided_accuracy",
) -> Tuple[float, float, str]:
    if not HAS_SCIPY:
        return (float("nan"), float("nan"), "n/a")

    piv = _pivot(agg, metric)
    if method not in piv.index or baseline not in piv.index:
        return (float("nan"), float("nan"), "not found")

    m = piv.loc[method].dropna()
    b = piv.loc[baseline].dropna()
    shared = m.index.intersection(b.index)

    if len(shared) < 5:
        return (float("nan"), float("nan"), "too few")

    diff = m[shared] - b[shared]
    if (diff == 0).all():
        return (float("nan"), float("nan"), "identical")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat, p = scipy_stats.wilcoxon(m[shared], b[shared],
                                           alternative="greater")
        sig = ("***" if p < 0.001 else
               "**"  if p < 0.01  else
               "*"   if p < 0.05  else "n.s.")
        return (float(stat), float(p), sig)
    except Exception as e:
        return (float("nan"), float("nan"), str(e))


def win_tie_loss(
    agg     : pd.DataFrame,
    method  : str,
    baseline: str = "baseline__sklearn_CART",
    metric  : str = "decided_accuracy",
) -> Tuple[int, int, int]:
    piv = _pivot(agg, metric)
    if method not in piv.index or baseline not in piv.index:
        return (0, 0, 0)
    diff = (piv.loc[method] - piv.loc[baseline]).dropna()
    return (
        int((diff >  WIN_THRESHOLD).sum()),
        int((diff.abs() <= WIN_THRESHOLD).sum()),
        int((diff < -WIN_THRESHOLD).sum()),
    )


# ---------------------------------------------------------------------------
# Table 1: OpenML-CC18
# ---------------------------------------------------------------------------

def table1_cc18(df: pd.DataFrame, output_file=None) -> str:
    agg = _agg(df)

    combo_stats = (
        agg.groupby(["combo", "delta_method", "routing_method"])
           .agg(
               n_datasets     =("decided_accuracy", "count"),
               mean_dec_acc   =("decided_accuracy", "mean"),
               std_dec_acc    =("decided_accuracy", "std"),
               mean_undec_rate=("undecided_rate",   "mean"),
               mean_acc_all   =("accuracy_all",     "mean"),
               mean_fit_time  =("fit_time_s",       "mean"),
           )
           .round(4).reset_index()
           .sort_values("mean_dec_acc", ascending=False)
    )

    bl = "baseline__sklearn_CART"
    rows = []
    for _, row in combo_stats.iterrows():
        combo = row["combo"]
        w, t, l = win_tie_loss(agg, combo, bl)
        _, p_val, sig = wilcoxon_vs_baseline(agg, combo, bl)
        rows.append({
            **row.to_dict(),
            "W": w, "T": t, "L": l,
            "p_value": round(p_val, 4) if not np.isnan(p_val) else "--",
            "sig"    : sig,
        })
    result_df = pd.DataFrame(rows)

    lines = [
        "",
        "=" * 80,
        "TABLE 1: OpenML-CC18 Summary",
        f"Datasets: {agg['dataset'].nunique()}  |  "
        f"CV folds: {df['fold'].nunique()}  |  "
        f"Win threshold: {WIN_THRESHOLD:.1%}",
        "=" * 80,
        f"{'Combo':<45} {'DecAcc':>8} {'+-std':>6} "
        f"{'Undec%':>7} {'AccAll':>7} {'W/T/L':>9} {'p':>8} {'sig':>5}",
        "-" * 80,
    ]
    for _, r in result_df.iterrows():
        lines.append(
            f"{r['combo']:<45} "
            f"{r['mean_dec_acc']:>8.4f} "
            f"{r['std_dec_acc']:>6.4f} "
            f"{r['mean_undec_rate']:>7.3f} "
            f"{r['mean_acc_all']:>7.4f} "
            f"{r['W']:>3}/{r['T']:>2}/{r['L']:>2}  "
            f"{str(r['p_value']):>8} "
            f"{r['sig']:>5}"
        )
    lines += [
        "-" * 80,
        "W/T/L = Win/Tie/Loss vs CART baseline on decided accuracy",
        "sig: *** p<0.001  ** p<0.01  * p<0.05  (Wilcoxon signed-rank, one-sided)",
        "",
        "  Delta method summary (mean across routing methods):",
    ]
    dm_agg = (
        result_df[result_df["delta_method"] != "baseline"]
        .groupby("delta_method")
        .agg(mean_dec_acc   =("mean_dec_acc",    "mean"),
             mean_undec_rate=("mean_undec_rate",  "mean"))
        .round(4).sort_values("mean_dec_acc", ascending=False)
    )
    for dm, row in dm_agg.iterrows():
        label = METHOD_LABELS.get(dm, dm)
        lines.append(f"    {label:<25} decided_acc={row['mean_dec_acc']:.4f}  "
                     f"undec={row['mean_undec_rate']:.3f}")

    text = "\n".join(lines) + "\n\n" + _latex_table1(result_df)
    _write(output_file, text)
    print(text)
    return text


def _latex_table1(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Decided accuracy across OpenML-CC18 (72 datasets, 5-fold CV). "
        r"W/T/L = wins/ties/losses vs CART baseline on decided accuracy. "
        r"node-bootstrap excluded where $N > 20{,}000$ due to computational cost.}",
        r"\label{tab:cc18}",
        r"\centering",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Routing & Dec.Acc & Undec\% & Acc.All & W/T/L & $p$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        dm  = METHOD_LABELS.get(r["delta_method"], r["delta_method"])
        rt  = ROUTING_LABELS.get(r["routing_method"], r["routing_method"])
        lines.append(
            f"{dm} & {rt} & "
            f"{r['mean_dec_acc']:.4f} & "
            f"{r['mean_undec_rate']:.3f} & "
            f"{r['mean_acc_all']:.4f} & "
            f"{r['W']}/{r['T']}/{r['L']} & "
            f"{_fmt_pval(r['p_value'])}{r['sig']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: Breiman Synthetic Analysis
# ---------------------------------------------------------------------------

def table2_breiman(df: pd.DataFrame, output_file=None) -> str:
    from benchmark.datasets_extended import BAYES_ERRORS
    agg = _agg(df)
    agg["bayes_error"] = agg["dataset"].map(BAYES_ERRORS)
    agg["undec_bayes_ratio"] = (
        agg["undecided_rate"] / agg["bayes_error"].replace(0, np.nan)
    ).round(3)

    lines = [
        "",
        "=" * 80,
        "TABLE 2: Breiman Synthetic Benchmarks",
        "Known Bayes errors: waveform=14.0%  twonorm=2.3%  ringnorm=1.7%",
        "Undec/Bayes ratio: undecided_rate / known_Bayes_error",
        "  ratio ~1.0 means the undecided zone captures the Bayes-uncertain region",
        "=" * 80,
    ]
    for dataset in ["waveform-5000", "twonorm", "ringnorm"]:
        sub = agg[agg["dataset"] == dataset].sort_values(
            "decided_accuracy", ascending=False)
        if sub.empty:
            continue
        bayes_err = BAYES_ERRORS.get(dataset, float("nan"))
        lines.append(f"\n  {dataset}  (Bayes error = {bayes_err:.1%})")
        lines.append(
            f"  {'Combo':<45} {'DecAcc':>8} {'Undec%':>7} {'Undec/Bayes':>12}")
        lines.append("  " + "-" * 73)
        for _, r in sub.iterrows():
            ratio     = r.get("undec_bayes_ratio", float("nan"))
            ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "--"
            lines.append(
                f"  {r['combo']:<45} "
                f"{r['decided_accuracy']:>8.4f} "
                f"{r['undecided_rate']:>7.3f} "
                f"{ratio_str:>12}"
            )

    text = "\n".join(lines) + "\n\n" + _latex_table2(agg)
    _write(output_file, text)
    print(text)
    return text


def _latex_table2(agg: pd.DataFrame) -> str:
    from benchmark.datasets_extended import BAYES_ERRORS
    datasets     = ["waveform-5000", "twonorm", "ringnorm"]
    best_methods = ["node_bootstrap__probabilistic", "margin__probabilistic",
                    "quality_plateau__probabilistic", "baseline__sklearn_CART"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Breiman synthetic benchmarks. "
        r"Undec/Bayes = undecided rate / known Bayes error.}",
        r"\label{tab:breiman}",
        r"\centering",
        r"\begin{tabular}{l" + "cc" * len(datasets) + "}",
        r"\toprule",
        r"& \multicolumn{2}{c}{waveform (BE=14\%)} "
        r"& \multicolumn{2}{c}{twonorm (BE=2.3\%)} "
        r"& \multicolumn{2}{c}{ringnorm (BE=1.7\%)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Method & Dec.Acc & U/B & Dec.Acc & U/B & Dec.Acc & U/B \\",
        r"\midrule",
    ]
    for combo in best_methods:
        dm    = combo.split("__")[0]
        parts = [METHOD_LABELS.get(dm, dm)]
        for ds in datasets:
            row = agg[(agg["combo"] == combo) & (agg["dataset"] == ds)]
            if row.empty:
                parts += ["--", "--"]
            else:
                r  = row.iloc[0]
                be = BAYES_ERRORS.get(ds, 1.0)
                ratio = r["undecided_rate"] / be if be > 0 else float("nan")
                parts.append(f"{r['decided_accuracy']:.4f}")
                parts.append(f"{ratio:.2f}" if not np.isnan(ratio) else "--")
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: Medical / Financial
# ---------------------------------------------------------------------------

def table3_medical(df: pd.DataFrame, output_file=None) -> str:
    from benchmark.datasets_extended import DOMAIN_NARRATIVE
    agg = _agg(df)

    lines = [
        "",
        "=" * 80,
        "TABLE 3: Medical & Financial High-Stakes Datasets",
        "Undecided instances are flagged for human review / further testing.",
        "=" * 80,
    ]
    for dataset in agg["dataset"].unique():
        sub       = agg[agg["dataset"] == dataset].sort_values(
            "decided_accuracy", ascending=False)
        narrative = DOMAIN_NARRATIVE.get(dataset, "")
        lines.append(f"\n  {dataset}")
        if narrative:
            lines.append(f"  Context: {narrative}")
        lines.append(
            f"  {'Combo':<42} {'DecAcc':>8} {'Undec%':>7} "
            f"{'AccAll':>7} {'F1-Dec':>7}")
        lines.append("  " + "-" * 70)
        for _, r in sub.iterrows():
            f1_dec = r.get("decided_f1", float("nan"))
            f1_str = f"{f1_dec:.4f}" if not np.isnan(f1_dec) else "--"
            lines.append(
                f"  {r['combo']:<42} "
                f"{r['decided_accuracy']:>8.4f} "
                f"{r['undecided_rate']:>7.3f} "
                f"{r['accuracy_all']:>7.4f} "
                f"{f1_str:>7}"
            )
        bl = sub[sub["delta_method"] == "baseline"]
        if not bl.empty:
            bl_acc = bl.iloc[0]["decided_accuracy"]
            best   = sub[sub["delta_method"] != "baseline"].iloc[0]
            gain   = best["decided_accuracy"] - bl_acc
            lines.append(
                f"\n  Best ternary: {best['combo']}"
                f"\n  +{gain:.2%} decided accuracy vs CART by flagging "
                f"{best['undecided_rate']:.1%} of cases for review"
            )

    text = "\n".join(lines) + "\n\n" + _latex_table3(agg)
    _write(output_file, text)
    print(text)
    return text


def _latex_table3(agg: pd.DataFrame) -> str:
    selected = ["node_bootstrap__probabilistic", "margin__probabilistic",
                "quality_plateau__probabilistic", "baseline__sklearn_CART"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Medical and financial datasets. "
        r"Dec.Acc = accuracy on committed predictions only.}",
        r"\label{tab:medical}",
        r"\centering",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Method & Dec.Acc & Undec\% & Acc.All & F1-Dec \\",
        r"\midrule",
    ]
    for dataset in agg["dataset"].unique():
        sub   = agg[agg["dataset"] == dataset]
        first = True
        for combo in selected:
            row = sub[sub["combo"] == combo]
            if row.empty:
                continue
            r    = row.iloc[0]
            dm   = METHOD_LABELS.get(r["delta_method"], r["delta_method"])
            ds   = dataset if first else ""
            f1_d = r.get("decided_f1", float("nan"))
            f1_s = f"{f1_d:.4f}" if not np.isnan(f1_d) else "--"
            lines.append(
                f"{ds} & {dm} & "
                f"{r['decided_accuracy']:.4f} & "
                f"{r['undecided_rate']:.3f} & "
                f"{r['accuracy_all']:.4f} & "
                f"{f1_s} \\\\"
            )
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Efficiency analysis
# ---------------------------------------------------------------------------

def efficiency_table(dfs: Dict[str, pd.DataFrame], output_file=None) -> str:
    lines = [
        "",
        "=" * 70,
        "EFFICIENCY ANALYSIS  (decided_acc_gain / undecided_rate)",
        "(Higher = more accuracy improvement per unit of abstention)",
        "=" * 70,
    ]
    for cname, df in dfs.items():
        if df.empty:
            continue
        agg    = _agg(df)
        bl     = (agg[agg["delta_method"] == "baseline"]
                  [["dataset", "decided_accuracy"]]
                  .rename(columns={"decided_accuracy": "baseline_acc"}))
        merged = agg.merge(bl, on="dataset")
        merged["gain"] = merged["decided_accuracy"] - merged["baseline_acc"]
        merged["efficiency"] = merged.apply(
            lambda r: r["gain"] / r["undecided_rate"]
            if r["undecided_rate"] > 0.01 else float("nan"),
            axis=1,
        )
        eff = (
            merged[merged["delta_method"] != "baseline"]
            .groupby("combo")["efficiency"]
            .mean()
            .sort_values(ascending=False)
            .round(3)
        )
        lines.append(f"\n  Collection: {cname.upper()}")
        for combo, val in eff.items():
            tag = "  <- recommended" if (
                ("margin" in combo or "node_bootstrap" in combo)
                and not (isinstance(val, float) and (val < 0 or val != val))
            ) else ""
            lines.append(f"    {combo:<45} {val:>8.3f}{tag}")

    text = "\n".join(lines)
    _write(output_file, text)
    print(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper tables from benchmark results"
    )
    parser.add_argument("--cc18",    default=None, help="results_cc18_clean.csv")
    parser.add_argument("--breiman", default=None, help="results_breiman.csv")
    parser.add_argument("--medical", default=None, help="results_medical.csv")
    parser.add_argument("--output",  default="paper_tables.txt")
    args = parser.parse_args()

    if args.output:
        open(args.output, "w", encoding="utf-8").close()

    dfs = {}

    if args.cc18:
        df = _read_csv(args.cc18)
        print(f"CC18   : {len(df)} rows  |  "
              f"{df['dataset'].nunique()} datasets  |  "
              f"{df['combo'].nunique()} combos")
        dfs["cc18"] = df
        table1_cc18(df, output_file=args.output)

    if args.breiman:
        df = _read_csv(args.breiman)
        print(f"Breiman: {len(df)} rows")
        dfs["breiman"] = df
        table2_breiman(df, output_file=args.output)

    if args.medical:
        df = _read_csv(args.medical)
        print(f"Medical: {len(df)} rows")
        dfs["medical"] = df
        table3_medical(df, output_file=args.output)

    if dfs:
        efficiency_table(dfs, output_file=args.output)

    if args.output:
        print(f"\nAll tables written to {args.output}")


if __name__ == "__main__":
    main()