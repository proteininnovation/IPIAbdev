#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — DS1 Dataset Download Script
# Institute for Protein Innovation (IPI)
#
# Downloads Dataset 1 (DS1) from Chen et al. 2024 (Cell Reports)
# and extracts a CDR3-diverse balanced 5,000-antibody test subset.
#
# Source   : https://zenodo.org/records/14735846
# DOI      : 10.5281/zenodo.14735846
# License  : MIT License (redistribution permitted with attribution)
# Citation : Chen HT, Zhang Y, Huang J et al. Human antibody polyreactivity
#            is governed primarily by the VH gene germline. Cell Reports, 2024.
#            PMC11564698
#
# Usage:
#   python utils/download_ds1_dataset.py
#
# Output:
#   data/DS1.xlsx          — full dataset (gitignored, large)
#   tests/DS1_5000.xlsx    — 5,000-antibody test subset (committed to GitHub)
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import zipfile
import urllib.request
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── ANARCI — CDR annotation (conda install -c bioconda hmmer anarci) ─────────
try:
    from anarci import anarci as _anarci_fn
    _ANARCI_AVAILABLE = True
except ImportError:
    _anarci_fn = None
    _ANARCI_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
ZENODO_URL     = ("https://zenodo.org/records/14735846/files/"
                  "Tessier-Lab-UMich/Human_Ab_Polyreactivity-v1.2.0-alpha.zip"
                  "?download=1")
SUBSET_SIZE    = 5000
RANDOM_STATE   = 42
CDR3_THRESHOLD = 0.8
_TESTS_DIR     = Path(__file__).resolve().parent.parent / "tests"
OUT_RAW        = _TESTS_DIR / "DS1_raw.xlsx"
OUT_FULL       = _TESTS_DIR / "DS1.xlsx"
OUT_SUBSET     = _TESTS_DIR / "DS1_5000.xlsx"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download
# ══════════════════════════════════════════════════════════════════════════════
def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        mb  = downloaded / 1e6
        tot = total_size / 1e6
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB",
              end="", flush=True)


def download_dataset(url: str, dest_zip: Path) -> Path:
    print(f"  Downloading from Zenodo...")
    urllib.request.urlretrieve(url, dest_zip, reporthook=_progress)
    print()
    print(f"  Downloaded: {dest_zip.name}  "
          f"({dest_zip.stat().st_size/1e6:.1f} MB)")
    return dest_zip


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract
# ══════════════════════════════════════════════════════════════════════════════
def extract_dataset(zip_path: Path, extract_dir: Path) -> Path:
    """Extract zip and return the extraction directory."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"  Contents: {len(names)} files")
        for n in names[:10]:
            print(f"    {n}")
        if len(names) > 10:
            print(f"    ... and {len(names)-10} more")
        zf.extractall(extract_dir)
    return extract_dir


def extract_hcdr3(vh_sequences: list, scheme: str = "imgt") -> list:
    """
    Extract HCDR3 from VH sequences using ANARCI.
    IMGT positions 105-117.
    Returns list of HCDR3 strings (empty string if ANARCI fails).
    """
    if not _ANARCI_AVAILABLE:
        print("  ERROR: ANARCI not installed.")
        print("  Run: conda install -c bioconda hmmer anarci")
        return [""] * len(vh_sequences)

    CDR3_START, CDR3_END = 105, 117
    print(f"  Extracting HCDR3 via ANARCI "
          f"(scheme={scheme}, IMGT {CDR3_START}-{CDR3_END})...")

    cdr3_seqs = []
    n         = len(vh_sequences)
    _debug_printed = False   # print ANARCI format once for verification

    for i in range(0, n, 500):
        batch = [(f"seq_{j}", seq.replace("-", "").upper())
                 for j, seq in enumerate(vh_sequences[i:i+500], start=i)
                 if seq and isinstance(seq, str)]
        if not batch:
            cdr3_seqs.extend([""] * min(500, n - i))
            continue
        try:
            results, _, _ = _anarci_fn(
                batch, scheme=scheme, output=False, assign_germline=False)
        except Exception as e:
            print(f"\n  WARNING: ANARCI batch failed: {e}")
            cdr3_seqs.extend([""] * len(batch))
            continue
        for result in results:
            if result is None or not result:
                cdr3_seqs.append(""); continue
            numbering, _start, _end = result[0]
            # Debug: print ANARCI format once
            if not _debug_printed and numbering:
                print(f"\n  [ANARCI debug] First residue: {numbering[0]}")
                _debug_printed = True
            cdr3 = ""
            for pos_info, aa in numbering:
                try:
                    pos = int(pos_info[1])
                    if CDR3_START <= pos <= CDR3_END and aa != "-":
                        cdr3 += aa
                except (ValueError, TypeError):
                    pass
            cdr3_seqs.append(cdr3)
        done = min(i + 500, n)
        print(f"\r  Progress: {done:,}/{n:,}  ({done/n*100:.0f}%)",
              end="", flush=True)

    print()
    n_ok = sum(1 for c in cdr3_seqs if c)
    print(f"  HCDR3: {n_ok:,} OK   {n - n_ok:,} failed/empty")
    return cdr3_seqs


def load_and_standardise(extract_dir: Path) -> pd.DataFrame:
    """
    Find and load Human Ab Poly Dataset S1_v2.xlsx from Supplemental Datasets.

    File structure (header starts at row 3):
      Row 1: Description text
      Row 2: "Sequence" label
      Row 3: Name  VL  VH  (actual column headers)
      Row 4+: data

    Label derived from Name column:
      "low"  in Name → psr_filter = 1  (PASS — low polyreactivity)
      "high" in Name → psr_filter = 0  (FAIL — high polyreactivity)

    Column mapping:
      Name → BARCODE  (spaces → underscores)
      VH   → HSEQ
      VL   → LSEQ
      CDR3 → extracted by ANARCI (IMGT 105-117)
    """
    # ── Find the Dataset S1 Excel file ───────────────────────────────────
    xlsx_files = list(extract_dir.rglob("*.xlsx"))
    print(f"  Excel files found: {[f.name for f in xlsx_files]}")

    # Prefer Human Ab Poly Dataset S1_v2.xlsx
    target = next(
        (f for f in xlsx_files if "Dataset S1" in f.name or "dataset_s1" in f.name.lower()),
        None
    )
    if target is None:
        # Fallback: any xlsx that is not Binding_data
        target = next(
            (f for f in xlsx_files if "binding" not in f.name.lower()),
            xlsx_files[0] if xlsx_files else None
        )
    if target is None:
        raise FileNotFoundError(
            f"No Excel file found in extracted archive.\n"
            f"Contents: {[f.name for f in extract_dir.rglob('*')]}"
        )

    print(f"  Loading: {target.name}  ({target.stat().st_size/1e6:.1f} MB)")

    # Header is on row 3 (0-indexed: header=2)
    df = pd.read_excel(target, header=2).copy()
    print(f"  Loaded: {len(df):,} rows  x  {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)[:10]}")

    # ── Save raw file FIRST before any processing ─────────────────────────
    import shutil
    if not OUT_RAW.exists():
        shutil.copy2(target, OUT_RAW)
        print(f"  Saved raw file → {OUT_RAW.name}")
    else:
        print(f"  Raw file already exists: {OUT_RAW.name}")

    # ── psr_filter from Name column ───────────────────────────────────────
    name_col = next((c for c in ["Name", "name"] if c in df.columns), None)
    if name_col:
        name_lower = df[name_col].astype(str).str.lower()
        df["psr_filter"] = np.where(name_lower.str.contains("low"),  1,
                           np.where(name_lower.str.contains("high"), 0, -1))
        n_invalid = (df["psr_filter"] == -1).sum()
        if n_invalid:
            print(f"  WARNING: {n_invalid:,} rows with Name containing neither 'high' nor 'low' — removed")
            df = df[df["psr_filter"] != -1].copy()
        print(f"  psr_filter: PASS/low={(df['psr_filter']==1).sum():,}  "
              f"FAIL/high={(df['psr_filter']==0).sum():,}  "
              f"pos_rate={df['psr_filter'].mean():.1%}")
    else:
        raise KeyError(f"Name column not found. Available: {list(df.columns)}")

    # ── BARCODE ───────────────────────────────────────────────────────────
    df["BARCODE"] = df[name_col].astype(str).str.replace(" ", "_", regex=False)
    print(f"  BARCODE: e.g. '{df['BARCODE'].iloc[0]}'")

    # ── HSEQ: detect VH column ────────────────────────────────────────────
    for col in ["VH", "VHaa", "vh", "heavy", "Heavy", "heavy_chain", "VH_sequence"]:
        if col in df.columns:
            df["HSEQ"] = df[col].astype(str).str.replace("-", "", regex=False)
            print(f"  HSEQ: from '{col}'")
            break
    else:
        raise KeyError(f"VH column not found. Available: {list(df.columns)}")

    # ── LSEQ: detect VL column ────────────────────────────────────────────
    for col in ["VL", "VLaa", "vl", "light", "Light", "light_chain", "VL_sequence"]:
        if col in df.columns:
            df["LSEQ"] = df[col].astype(str).str.replace("-", "", regex=False)
            print(f"  LSEQ: from '{col}'")
            break
    else:
        df["LSEQ"] = ""
        print("  LSEQ: not found — VH-only mode")

    # ── CDR3: always extract via ANARCI ───────────────────────────────────
    print(f"\n  Extracting HCDR3 via ANARCI ({len(df):,} sequences)...")
    df["CDR3"] = extract_hcdr3(df["HSEQ"].fillna("").tolist())
    n_empty = (df["CDR3"] == "").sum()
    if n_empty:
        print(f"  Removing {n_empty:,} sequences with empty CDR3")
        df = df[df["CDR3"] != ""].copy()

    # ── Final DELPHI columns ──────────────────────────────────────────────
    core = ["BARCODE", "HSEQ", "LSEQ", "CDR3", "psr_filter"]
    df   = df[[c for c in core if c in df.columns]].reset_index(drop=True)
    print(f"\n  Final: {len(df):,} antibodies  "
          f"PASS={(df['psr_filter']==1).sum():,}  "
          f"FAIL={(df['psr_filter']==0).sum():,}  "
          f"pos_rate={df['psr_filter'].mean():.1%}")
    return df


def cluster_cdr3(df: pd.DataFrame,
                 threshold: float = CDR3_THRESHOLD) -> pd.DataFrame:
    """
    Cluster CDR3 sequences at given identity threshold and return
    one representative per cluster (diverse subset).
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from utils.clustering import greedy_clustering_by_levenshtein
    except ImportError:
        print("  WARNING: utils/clustering.py not found — "
              "skipping CDR3 clustering (random sampling instead)")
        return df

    cdr3s      = df["CDR3"].fillna("").tolist()
    print(f"  Clustering {len(cdr3s):,} CDR3s at {threshold:.0%} identity...")
    cluster_ids = greedy_clustering_by_levenshtein(cdr3s, threshold)

    df          = df.copy()
    df["_cls"]  = cluster_ids
    n_clusters  = df["_cls"].nunique()
    print(f"  {n_clusters:,} clusters from {len(df):,} sequences "
          f"(diversity={n_clusters/len(df):.1%})")

    rng  = np.random.default_rng(RANDOM_STATE)
    reps = (df.groupby("_cls", group_keys=False)
              .apply(lambda g: g.sample(1, random_state=int(rng.integers(1e6)))))
    return reps.drop(columns=["_cls"]).reset_index(drop=True)


def create_subset(df: pd.DataFrame, n: int = SUBSET_SIZE,
                  label_col: str = "psr_filter") -> pd.DataFrame:
    """
    CDR3-diverse balanced 50-50 test subset:
      1. Cluster by CDR3 → pick one representative per cluster
      2. Sample n/2 PASS + n/2 FAIL from representatives
    """
    n_each = n // 2

    # Step 1: CDR3 clustering
    reps = cluster_cdr3(df)

    # Step 2: balanced 50-50
    pass_reps = reps[reps[label_col] == 1]
    fail_reps = reps[reps[label_col] == 0]

    n_pass = min(n_each, len(pass_reps))
    n_fail = min(n_each, len(fail_reps))

    if n_pass < n_each:
        print(f"  WARNING: only {len(pass_reps):,} diverse PASS sequences "
              f"(requested {n_each:,})")
    if n_fail < n_each:
        print(f"  WARNING: only {len(fail_reps):,} diverse FAIL sequences "
              f"(requested {n_each:,})")

    subset = pd.concat([
        pass_reps.sample(n=n_pass, random_state=RANDOM_STATE),
        fail_reps.sample(n=n_fail, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"\n  Test subset:")
    print(f"    PASS (low polyreactivity)  : {n_pass:,}")
    print(f"    FAIL (high polyreactivity) : {n_fail:,}")
    print(f"    Total                      : {len(subset):,}  "
          f"pos_rate={subset[label_col].mean():.1%}")
    print(f"    CDR3 diversity: cluster-based ({CDR3_THRESHOLD:.0%} threshold)")
    return subset


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  DELPHI — DS1 Dataset Download")
    print("  Source : Chen et al. 2024  (Cell Reports, PMC11564698)")
    print("  DOI    : 10.5281/zenodo.14735846  |  License: MIT")
    print("══════════════════════════════════════════════════════════════════")
    print()

    _TESTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Raw file     : tests/DS1_raw.xlsx")
    print(f"  Full dataset : tests/DS1.xlsx")
    print(f"  Test subset  : tests/DS1_5000.xlsx")

    if OUT_SUBSET.exists():
        print(f"\n  Test subset already exists: {OUT_SUBSET.name}")
        df  = pd.read_excel(OUT_SUBSET)
        pr  = df["psr_filter"].mean() if "psr_filter" in df.columns else float("nan")
        print(f"  {len(df):,} antibodies  pos_rate={pr:.1%}")
        print("  To re-download, delete the file and run again.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp      = Path(tmpdir)
        zip_path = tmp / "ds1.zip"

        print("── Step 1: Download ────────────────────────────────────────")
        download_dataset(ZENODO_URL, zip_path)

        print("\n── Step 2: Extract ─────────────────────────────────────────")
        extract_dir = extract_dataset(zip_path, tmp / "extracted")

        print("\n── Step 3: Load, standardise and extract HCDR3 ─────────────")
        df = load_and_standardise(extract_dir)

        print("\n── Step 4: Save full dataset ───────────────────────────────")
        df.to_excel(OUT_FULL, index=False)
        print(f"  Full dataset → {OUT_FULL.name}  ({len(df):,} antibodies)")

        print("\n── Step 5: Create CDR3-diverse balanced test subset ─────────")
        subset = create_subset(df)
        subset.to_excel(OUT_SUBSET, index=False)
        print(f"  Test subset  → {OUT_SUBSET.name}")

    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  Done.")
    print(f"  Raw file     : tests/DS1_raw.xlsx")
    print(f"  Full dataset : tests/DS1.xlsx")
    print(f"  Test subset  : tests/DS1_5000.xlsx")
    print()
    print("  Next step:")
    print("    python tests/test_delphi.py")
    print("══════════════════════════════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()