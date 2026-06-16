#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Subset Generator
# Institute for Protein Innovation (IPI)
#
# Creates CDR3-diverse balanced subsets from any DELPHI dataset.
#
# Usage:
#   python utils/create_subsets.py --input tests/DS1.xlsx --target psr_filter
#   python utils/create_subsets.py --input tests/DS2.xlsx --target sec_filter
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT  = Path(__file__).resolve().parent.parent
_TESTS = _ROOT / "tests"

RANDOM_STATE   = 42
CDR3_THRESHOLD = 0.8
SUBSETS        = [500, 1000, 5000]


# ══════════════════════════════════════════════════════════════════════════════
# CDR3 clustering
# ══════════════════════════════════════════════════════════════════════════════
def cluster_cdr3(df: pd.DataFrame,
                 label_col: str,
                 threshold: float = CDR3_THRESHOLD) -> pd.DataFrame:
    """
    Cluster CDR3 sequences at given identity threshold.
    Returns one representative per cluster.
    """
    try:
        sys.path.insert(0, str(_ROOT))
        from utils.clustering import greedy_clustering_by_levenshtein
    except ImportError:
        print("  WARNING: utils/clustering.py not found — "
              "skipping CDR3 clustering, using random sampling instead")
        return df

    cdr3s = df["CDR3"].fillna("").tolist()
    print(f"  Clustering {len(cdr3s):,} CDR3s at {threshold:.0%} identity...")

    cluster_ids = greedy_clustering_by_levenshtein(cdr3s, threshold)
    df = df.copy()
    df["_cluster"] = cluster_ids

    n_clusters = df["_cluster"].nunique()
    print(f"  {n_clusters:,} clusters from {len(df):,} sequences "
          f"(diversity={n_clusters/len(df):.1%})")

    rng  = np.random.default_rng(RANDOM_STATE)
    reps = (df.groupby("_cluster", group_keys=False)
              .apply(lambda g: g.sample(
                  1, random_state=int(rng.integers(1e6)))))
    reps = reps.drop(columns=["_cluster"]).reset_index(drop=True)

    pos_rate = reps[label_col].mean() if label_col in reps.columns else float("nan")
    print(f"  Representatives: {len(reps):,}  pos_rate={pos_rate:.1%}")
    return reps


# ══════════════════════════════════════════════════════════════════════════════
# Balanced subset
# ══════════════════════════════════════════════════════════════════════════════
def create_balanced_subset(pass_reps: pd.DataFrame,
                           fail_reps: pd.DataFrame,
                           n: int,
                           label_col: str) -> pd.DataFrame:
    """
    Sample n/2 PASS + n/2 FAIL from pre-clustered class representatives.
    Clustering must be done per-class BEFORE calling this function.
    """
    n_each = n // 2
    n_pass = min(n_each, len(pass_reps))
    n_fail = min(n_each, len(fail_reps))

    if n_pass < n_each:
        print(f"  WARNING: only {len(pass_reps):,} diverse PASS sequences "
              f"available (requested {n_each:,}) — subset will be smaller")
    if n_fail < n_each:
        print(f"  WARNING: only {len(fail_reps):,} diverse FAIL sequences "
              f"available (requested {n_each:,}) — subset will be smaller")

    subset = pd.concat([
        pass_reps.sample(n=n_pass, random_state=RANDOM_STATE),
        fail_reps.sample(n=n_fail, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    actual_pass = (subset[label_col] == 1).sum()
    actual_fail = (subset[label_col] == 0).sum()
    pos_rate    = subset[label_col].mean()

    ok = actual_pass == n_pass and actual_fail == n_fail
    sym = "OK" if ok else "UNBALANCED"
    print(f"  [{sym}] PASS={actual_pass:,}  FAIL={actual_fail:,}  "
          f"pos_rate={pos_rate:.1%}"
          + (f"  ← expected {n_pass:,}+{n_fail:,}" if not ok else ""))
    return subset


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Create CDR3-diverse balanced subsets from any DELPHI dataset")
    ap.add_argument("--target", default="psr_filter",
                    choices=["psr_filter", "sec_filter", "hic_filter",
                             "acsins_filter", "viscosity_filter"],
                    help="Target label column (default: psr_filter)")
    ap.add_argument("--input", default=str(_TESTS / "DS1.xlsx"),
                    help="Path to DS1.xlsx (default: tests/DS1.xlsx)")
    ap.add_argument("--outdir", default=str(_TESTS),
                    help="Output directory (default: tests/)")
    ap.add_argument("--threshold", type=float, default=CDR3_THRESHOLD,
                    help=f"CDR3 clustering threshold (default: {CDR3_THRESHOLD})")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  DELPHI — Subset Generator")
    print(f"  Input     : {input_path}")
    print(f"  Output    : {out_dir}/")
    print(f"  Target    : {args.target}")
    print(f"  Subsets   : {SUBSETS}")
    print(f"  CDR3 thr  : {args.threshold:.0%}")
    print("══════════════════════════════════════════════════════════════════")
    print()

    # ── Load ──────────────────────────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        print("Run: python utils/download_ds1_dataset.py")
        sys.exit(1)

    print(f"Loading {input_path.name}...")
    df = pd.read_excel(input_path)
    print(f"  {len(df):,} antibodies  x  {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    label_col = args.target

    # Validate required columns
    required = ["BARCODE", "HSEQ", "LSEQ", "CDR3", label_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        sys.exit(1)

    pos_rate = df[label_col].mean()
    print(f"  PASS={(df[label_col]==1).sum():,}  "
          f"FAIL={(df[label_col]==0).sum():,}  "
          f"pos_rate={pos_rate:.1%}")

    # ── Cluster PASS and FAIL ONCE, then sample all subset sizes ─────────
    print()
    print("── CDR3 clustering (per class) ─────────────────────────────")
    pass_df = df[df[label_col] == 1].copy()
    fail_df = df[df[label_col] == 0].copy()
    print(f"  PASS: {len(pass_df):,} sequences → clustering...")
    pass_reps = cluster_cdr3(pass_df, label_col=label_col, threshold=args.threshold)
    print(f"  FAIL: {len(fail_df):,} sequences → clustering...")
    fail_reps = cluster_cdr3(fail_df, label_col=label_col, threshold=args.threshold)
    print(f"  PASS representatives: {len(pass_reps):,}")
    print(f"  FAIL representatives: {len(fail_reps):,}")
    print()

    # ── Create and save each subset ───────────────────────────────────────
    for n in SUBSETS:
        print(f"── Subset n={n:,} ──────────────────────────────────────────")
        subset   = create_balanced_subset(pass_reps, fail_reps, n,
                                          label_col=label_col)
        tag      = label_col.replace('_filter', '')
        out_path = out_dir / f"DS1_{tag}_{n}.xlsx"

        # Keep only DELPHI standard columns
        cols   = [c for c in ["BARCODE", "HSEQ", "LSEQ", "CDR3", label_col]
                  if c in subset.columns]
        subset = subset[cols]

        subset.to_excel(out_path, index=False)
        print(f"  PASS={(subset[label_col]==1).sum():,}  "
              f"FAIL={(subset[label_col]==0).sum():,}  "
              f"pos_rate={subset[label_col].mean():.1%}")
        print(f"  Saved → {out_path.name}")
        print()

    print("══════════════════════════════════════════════════════════════════")
    print("  Done.")
    tag = label_col.replace('_filter', '')
    for n in SUBSETS:
        print(f"    tests/DS1_{tag}_{n}.xlsx")
    print("══════════════════════════════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()