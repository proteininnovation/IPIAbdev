#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Subset Generator
# Institute for Protein Innovation (IPI)
#
# Creates CDR3-diverse subsets from any DELPHI dataset. Clusters ALL antibodies
# once (global clustering), then writes:
#   - DS1_{tag}_rep.xlsx        one antibody per cluster — ALWAYS (default)
#                               (natural ratio, or 50/50 with --balanced)
#   - DS1_{tag}_{n}.xlsx        balanced fixed-size subsets — only if --sizes given
#
# Usage:
#   # Default: representative subset of ALL data (natural ratio):
#   python utils/create_subsets.py --input tests/DS1.xlsx --target psr_filter
#
#   # Balanced representative subset (50/50) instead of natural ratio:
#   python utils/create_subsets.py --input tests/DS1.xlsx --target psr_filter \
#       --balanced
#
#   # Also generate fixed-size balanced subsets (opt-in):
#   python utils/create_subsets.py --input tests/DS1.xlsx --target psr_filter \
#       --sizes 500 1000 5000 --threshold 0.9
#
#   # Cluster on a different sequence (Fab full-length or VHH nanobodies):
#   #   CDR3 (default, Fab) | HSEQ (VHH/nanobody) | HSEQ LSEQ (full Fab)
#   python utils/create_subsets.py --input nanobodies.xlsx --target psr_filter \
#       --cluster_col HSEQ
#   python utils/create_subsets.py --input fabs.xlsx --target psr_filter \
#       --cluster_col HSEQ LSEQ
#
#   # Only fixed-size subsets, no representative:
#   python utils/create_subsets.py --input tests/DS1.xlsx --target psr_filter \
#       --sizes 500 1000 --no-representative
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
# Global CDR3 clustering
# ══════════════════════════════════════════════════════════════════════════════
def assign_global_clusters(df: pd.DataFrame,
                           threshold: float = CDR3_THRESHOLD,
                           cluster_cols=("CDR3",)) -> pd.DataFrame:
    """
    Cluster ALL antibodies together by sequence identity and return the
    dataframe with a new '_cluster' column.

    cluster_cols selects which sequence column(s) to cluster on:
      ("CDR3",)          — CDR3 only (Fab, default; the classic DELPHI behaviour)
      ("HSEQ",)          — heavy-chain / VHH only (nanobodies, single domain)
      ("HSEQ", "LSEQ")   — full Fab variable region (heavy + light concatenated)
    When several columns are given they are concatenated per antibody into one
    string before clustering, so identity is measured over the joined sequence.

    This is done ONCE for the whole dataset. Both the balanced fixed-size
    subsets and the representative subset are drawn from these global clusters,
    so near-identical sequences (even across PASS/FAIL) share a cluster and are
    never double-counted.
    """
    cluster_cols = list(cluster_cols)
    missing = [c for c in cluster_cols if c not in df.columns]
    if missing:
        raise ValueError(f"--cluster_col columns not found in data: {missing}")

    try:
        sys.path.insert(0, str(_ROOT))
        from utils.clustering import greedy_clustering_by_levenshtein
    except ImportError:
        print("  WARNING: utils/clustering.py not found — "
              "cannot cluster; every antibody treated as its own cluster")
        out = df.copy()
        out["_cluster"] = range(len(out))
        return out

    # Build the sequence used for clustering (concatenate if multiple columns).
    if len(cluster_cols) == 1:
        seqs = df[cluster_cols[0]].fillna("").astype(str)
    else:
        seqs = (df[cluster_cols].fillna("").astype(str)
                .agg("".join, axis=1))
    seq_list = seqs.tolist()

    col_desc = "+".join(cluster_cols)
    print(f"  Global clustering {len(seq_list):,} sequences on [{col_desc}] "
          f"at {threshold:.0%} identity...")
    out = df.copy()
    out["_cluster"] = greedy_clustering_by_levenshtein(seq_list, threshold)

    n_clusters = out["_cluster"].nunique()
    print(f"  {n_clusters:,} global clusters from {len(out):,} sequences "
          f"(diversity={n_clusters/len(out):.1%})")
    return out


def _one_per_cluster(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Pick one random antibody per '_cluster' (diverse representatives)."""
    idx = [g.sample(1, random_state=int(rng.integers(1e6))).index[0]
           for _clu, g in df.groupby("_cluster", sort=False)]
    return df.loc[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Balanced subset (from global clusters)
# ══════════════════════════════════════════════════════════════════════════════
def create_balanced_subset(df_clustered: pd.DataFrame,
                           n: int,
                           label_col: str) -> pd.DataFrame:
    """
    Sample n/2 PASS + n/2 FAIL from the globally-clustered dataset, taking at
    most one antibody per CDR3 cluster within each class so the subset stays
    CDR3-diverse. Clustering is global (done once beforehand); this function
    only samples.
    """
    n_each = n // 2
    rng = np.random.default_rng(RANDOM_STATE)

    # One representative per cluster, then split by class.
    reps = _one_per_cluster(df_clustered, rng)
    pass_reps = reps[reps[label_col] == 1]
    fail_reps = reps[reps[label_col] == 0]

    n_pass = min(n_each, len(pass_reps))
    n_fail = min(n_each, len(fail_reps))

    if n_pass < n_each:
        print(f"  WARNING: only {len(pass_reps):,} diverse PASS clusters "
              f"available (requested {n_each:,}) — subset will be smaller")
    if n_fail < n_each:
        print(f"  WARNING: only {len(fail_reps):,} diverse FAIL clusters "
              f"available (requested {n_each:,}) — subset will be smaller")

    subset = pd.concat([
        pass_reps.sample(n=n_pass, random_state=RANDOM_STATE),
        fail_reps.sample(n=n_fail, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    if "_cluster" in subset.columns:
        subset = subset.drop(columns=["_cluster"])

    actual_pass = (subset[label_col] == 1).sum()
    actual_fail = (subset[label_col] == 0).sum()
    pos_rate    = subset[label_col].mean()

    ok = actual_pass == n_pass and actual_fail == n_fail
    sym = "OK" if ok else "UNBALANCED"
    print(f"  [{sym}] PASS={actual_pass:,}  FAIL={actual_fail:,}  "
          f"pos_rate={pos_rate:.1%}"
          + (f"  ← expected {n_pass:,}+{n_fail:,}" if not ok else ""))
    return subset


def create_representative_subset(df_clustered: pd.DataFrame,
                                 label_col: str,
                                 balanced: bool = False) -> pd.DataFrame:
    """
    Build the representative subset from the globally-clustered dataset: one
    randomly-chosen antibody per CDR3 cluster.

    Clustering is global and done ONCE beforehand (assign_global_clusters), so
    near-identical CDR3s across PASS/FAIL already share a cluster and are never
    double-counted.

    balanced=False (default): keep every cluster; the natural PASS/FAIL ratio of
        the clusters is reported as-is.
    balanced=True: down-sample the majority class so the subset is 50/50
        (keeps all of the minority class, samples an equal number from the
        majority). Uses fewer clusters but is class-balanced.
    """
    work = df_clustered
    if "_cluster" not in work.columns:
        raise ValueError("create_representative_subset expects a dataframe with "
                         "a '_cluster' column (call assign_global_clusters first).")

    rng = np.random.default_rng(RANDOM_STATE)
    reps = _one_per_cluster(work, rng).drop(columns=["_cluster"]).reset_index(drop=True)

    n_pass = int((reps[label_col] == 1).sum())
    n_fail = int((reps[label_col] == 0).sum())

    if balanced:
        n_each = min(n_pass, n_fail)
        reps = (pd.concat([
                    reps[reps[label_col] == 1].sample(n=n_each, random_state=RANDOM_STATE),
                    reps[reps[label_col] == 0].sample(n=n_each, random_state=RANDOM_STATE),
                ])
                .sample(frac=1, random_state=RANDOM_STATE)
                .reset_index(drop=True))
        print(f"  Representative subset (balanced): n={len(reps):,}  "
              f"PASS={n_each:,}  FAIL={n_each:,}  pos_rate=50.0% "
              f"(down-sampled majority from {max(n_pass, n_fail):,})")
    else:
        pos_rate = reps[label_col].mean() if len(reps) else float("nan")
        print(f"  Representative subset: n={len(reps):,}  "
              f"PASS={n_pass:,}  FAIL={n_fail:,}  pos_rate={pos_rate:.1%} "
              f"(one per cluster, natural ratio)")
    return reps


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
                    help=f"Sequence clustering identity threshold (default: {CDR3_THRESHOLD})")
    ap.add_argument("--cluster_col", nargs="+", default=["CDR3"],
                    metavar="COL",
                    help="Sequence column(s) to cluster on. "
                         "CDR3 (Fab, default); HSEQ (VHH/nanobody, single "
                         "domain); HSEQ LSEQ (full Fab, heavy+light "
                         "concatenated). Example: --cluster_col HSEQ LSEQ")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    metavar="N",
                    help="Balanced fixed-size subset sizes to also generate "
                         "(e.g. --sizes 500 1000 5000). If omitted, only the "
                         "representative subset is written.")
    ap.add_argument("--no-representative", action="store_true",
                    help="Skip the representative subset (one antibody per "
                         "global cluster, natural ratio). By default it is "
                         "always written.")
    ap.add_argument("--balanced", action="store_true",
                    help="Make the representative subset class-balanced (50/50 "
                         "by down-sampling the majority class) instead of "
                         "keeping the natural cluster ratio.")
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
    _rep_desc = ("no (skipped)" if args.no_representative
                 else ("yes, balanced 50/50" if args.balanced
                       else "yes, natural ratio"))
    print(f"  Subsets   : {args.sizes if args.sizes else '(none — representative only)'}")
    print(f"  Represent.: {_rep_desc}")
    print(f"  Cluster on: {'+'.join(args.cluster_col)}  (at {args.threshold:.0%} identity)")
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

    # Ignore any precomputed cluster columns (e.g. HCDR3_CLUSTER_0.8,
    # HSEQ_CLUSTER_0.85). DELPHI always recomputes CDR3 clusters fresh from the
    # CDR3 sequences at the requested --threshold, so these are dropped to make
    # sure they are never read or written into the output subsets.
    _precomputed = [c for c in df.columns if "_CLUSTER_" in c]
    if _precomputed:
        print(f"  Ignoring {len(_precomputed)} precomputed cluster column(s) "
              f"(recomputing fresh)")
        df = df.drop(columns=_precomputed)

    label_col = args.target

    # Validate required columns. BARCODE + the label + whichever sequence
    # column(s) we cluster on must be present. HSEQ/LSEQ/CDR3 are otherwise
    # optional so nanobody (VHH) datasets without LSEQ/CDR3 still work.
    required = ["BARCODE", label_col] + list(args.cluster_col)
    missing  = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  (clustering on {args.cluster_col}; those columns must exist)")
        sys.exit(1)

    pos_rate = df[label_col].mean()
    print(f"  PASS={(df[label_col]==1).sum():,}  "
          f"FAIL={(df[label_col]==0).sum():,}  "
          f"pos_rate={pos_rate:.1%}")

    # ── Global sequence clustering ONCE for the whole dataset ─────────────
    # Both the balanced fixed-size subsets and the representative subset draw
    # from these global clusters. No per-class clustering anywhere.
    print()
    print("── Global clustering ───────────────────────────────────────")
    df_clustered = assign_global_clusters(df, threshold=args.threshold,
                                          cluster_cols=args.cluster_col)
    print()

    # ── Create and save each fixed-size balanced subset ───────────────────
    tag           = label_col.replace('_filter', '')
    written_files = []
    std_cols      = ["BARCODE", "HSEQ", "LSEQ", "CDR3", label_col]

    if args.sizes:
        for n in args.sizes:
            print(f"── Subset n={n:,} ──────────────────────────────────────────")
            subset   = create_balanced_subset(df_clustered, n,
                                              label_col=label_col)
            out_path = out_dir / f"DS1_{tag}_{n}.xlsx"

            # Keep only DELPHI standard columns
            cols   = [c for c in std_cols if c in subset.columns]
            subset = subset[cols]

            subset.to_excel(out_path, index=False)
            print(f"  PASS={(subset[label_col]==1).sum():,}  "
                  f"FAIL={(subset[label_col]==0).sum():,}  "
                  f"pos_rate={subset[label_col].mean():.1%}")
            print(f"  Saved → {out_path.name}")
            print()
            written_files.append(out_path.name)

    # ── Representative subset: one per global CDR3 cluster (default ON) ────
    if not args.no_representative:
        _bal = " (balanced 50/50)" if args.balanced else " (natural ratio)"
        print(f"── Representative subset{_bal} ──────────────")
        rep = create_representative_subset(df_clustered, label_col=label_col,
                                           balanced=args.balanced)
        cols = [c for c in std_cols if c in rep.columns]
        rep  = rep[cols]
        rep_path = out_dir / f"DS1_{tag}_rep.xlsx"
        rep.to_excel(rep_path, index=False)
        print(f"  Saved → {rep_path.name}")
        print()
        written_files.append(rep_path.name)

    print("══════════════════════════════════════════════════════════════════")
    print("  Done.")
    for fname in written_files:
        print(f"    {out_dir}/{fname}")
    print("══════════════════════════════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()