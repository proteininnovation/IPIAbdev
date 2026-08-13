#!/usr/bin/env python3
"""
R1: Leave-one-VH-germline-out evaluation for DELPHI antibody developability.

Reviewer objection: the headline 10-fold HCDR3-cluster CV AUC (0.959) controls
only the HCDR3 loop. All framework regions are frozen across ~5-7 VH germlines,
so two antibodies in different HCDR3 clusters still share ~95% of their sequence.
Holding out a WHOLE germline (its entire frozen-framework background) tests
whether that framework redundancy inflates the AUC.

This script:
  ANCHOR  = StratifiedGroupKFold(10) grouped by HCDR3_CLUSTER_0.8  -> AUC_cluster
  MAIN    = LeaveOneGroupOut grouped by BASE vh_scaffold germline  -> AUC_germline_out
  OPTIONAL= LeaveOneGroupOut grouped by base vh+vl pair            -> AUC_pair_out
Same classifier config for every protocol. Pooled out-of-fold AUC each time.

Every number printed is computed from this run. No values are hard-coded.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

PAPER = Path(__file__).resolve().parents[1]
LOCAL = PAPER / "data" / "local_only"
EMB_CSV = LOCAL / "embeddings" / "ipi_psr_trainset.xlsx.ablang.emb.csv"
LABEL_XLSX = LOCAL / "raw" / "ipi_psr_trainset.xlsx"

RANDOM_STATE = 42
EMB_DIM = 480
FEATURE_COLS = [str(i) for i in range(EMB_DIM)]
MIN_N_PER_GERMLINE = 100  # for the per-germline reporting table


def make_classifier():
    """Return a fresh classifier with the fixed config, plus a name string."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        )
        import xgboost
        return clf, f"xgboost.XGBClassifier (v{xgboost.__version__})"
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        return clf, f"sklearn.HistGradientBoostingClassifier (xgboost import failed: {e})"


def predict_proba_pass(clf, X):
    """P(psr_filter == 1). Robust to class ordering in clf.classes_."""
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    return proba[:, classes.index(1)]


def main():
    print("=" * 78)
    print("R1: Leave-one-VH-germline-out evaluation (DELPHI developability)")
    print("=" * 78)

    # ---- 1. Load + merge -----------------------------------------------------
    print("\n[1] Loading and merging data on BARCODE (inner join)")
    labels = pd.read_excel(LABEL_XLSX, sheet_name="Sheet1")
    needed = ["BARCODE", "psr_filter", "vh_scaffold", "vl_scaffold", "HCDR3_CLUSTER_0.8"]
    missing = [c for c in needed if c not in labels.columns]
    if missing:
        sys.exit(f"FATAL: missing label columns: {missing}")
    labels = labels[needed].copy()

    emb = pd.read_csv(EMB_CSV)
    if emb.columns[0] != "BARCODE":
        sys.exit(f"FATAL: first embedding column is {emb.columns[0]!r}, expected BARCODE")
    miss_feat = [c for c in FEATURE_COLS if c not in emb.columns]
    if miss_feat:
        sys.exit(f"FATAL: embedding missing feature cols, e.g. {miss_feat[:5]}")

    n_emb, n_lab = len(emb), len(labels)
    merged = emb.merge(labels, on="BARCODE", how="inner")
    n_matched = len(merged)
    # BARCODEs present in embeddings but not matched to labels:
    emb_unmatched = n_emb - emb["BARCODE"].isin(labels["BARCODE"]).sum()
    lab_unmatched = n_lab - labels["BARCODE"].isin(emb["BARCODE"]).sum()

    # base germline = strip everything from first underscore
    merged["vh_base"] = merged["vh_scaffold"].astype(str).str.split("_").str[0]
    merged["vl_base"] = merged["vl_scaffold"].astype(str).str.split("_").str[0]
    merged["vh_vl_pair"] = merged["vh_base"] + " | " + merged["vl_base"]

    y = merged["psr_filter"].astype(int).values
    n_pass = int((y == 1).sum())
    n_fail = int((y == 0).sum())

    print(f"    embedding rows         : {n_emb}")
    print(f"    label rows             : {n_lab}")
    print(f"    matched (inner join)   : {n_matched}")
    print(f"    emb BARCODEs unmatched : {emb_unmatched}")
    print(f"    label BARCODEs unmatched: {lab_unmatched}")
    print(f"    n_pass (psr_filter==1) : {n_pass} ({n_pass/n_matched:.4f})")
    print(f"    n_fail (psr_filter==0) : {n_fail} ({n_fail/n_matched:.4f})")
    print(f"    -> PASS is the {'MAJORITY' if n_pass > n_fail else 'MINORITY'} class "
          f"(expected MAJORITY ~0.526)")
    print(f"    distinct HCDR3 clusters: {merged['HCDR3_CLUSTER_0.8'].nunique()}")
    print(f"    distinct vh_base       : {merged['vh_base'].nunique()}  "
          f"({sorted(merged['vh_base'].unique())})")

    X = merged[FEATURE_COLS].values.astype(np.float32)

    clf_probe, clf_name = make_classifier()
    print(f"\n    Classifier (same config for ALL protocols): {clf_name}")

    # ---- 3. ANCHOR: StratifiedGroupKFold(10) by HCDR3 cluster ---------------
    print("\n" + "=" * 78)
    print("[3] ANCHOR protocol: StratifiedGroupKFold(10) grouped by HCDR3_CLUSTER_0.8")
    print("=" * 78)
    groups_clu = merged["HCDR3_CLUSTER_0.8"].values
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.full(n_matched, np.nan, dtype=float)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=groups_clu), 1):
        clf, _ = make_classifier()
        clf.fit(X[tr], y[tr])
        oof_pred[te] = predict_proba_pass(clf, X[te])
        print(f"    fold {fold:2d}: train={len(tr):5d}  test={len(te):5d}  "
              f"test_pass={int(y[te].sum()):4d}  test_fail={int((y[te]==0).sum()):4d}")
    assert not np.isnan(oof_pred).any(), "some rows never held out in ANCHOR"
    auc_cluster = roc_auc_score(y, oof_pred)
    print(f"\n    AUC_cluster (pooled OOF over all 10 folds) = {auc_cluster:.4f}")

    # ---- 4. MAIN: LeaveOneGroupOut by base VH germline ----------------------
    print("\n" + "=" * 78)
    print("[4] MAIN protocol: LeaveOneGroupOut grouped by BASE vh_scaffold germline")
    print("=" * 78)
    groups_vh = merged["vh_base"].values
    logo = LeaveOneGroupOut()
    pooled_pred, pooled_y = [], []
    per_germ = []  # rows: name, n, n_pass, n_fail, auc_or_None, included_in_table
    for tr, te in logo.split(X, y, groups=groups_vh):
        name = groups_vh[te][0]
        yte = y[te]
        clf, _ = make_classifier()
        clf.fit(X[tr], y[tr])
        pte = predict_proba_pass(clf, X[te])
        pooled_pred.append(pte)
        pooled_y.append(yte)
        npass, nfail = int((yte == 1).sum()), int((yte == 0).sum())
        both = npass > 0 and nfail > 0
        auc = roc_auc_score(yte, pte) if both else None
        per_germ.append({"germline": name, "n": len(te), "n_pass": npass,
                         "n_fail": nfail, "auc": auc})

    pooled_pred = np.concatenate(pooled_pred)
    pooled_y = np.concatenate(pooled_y)
    auc_germ_out = roc_auc_score(pooled_y, pooled_pred)

    per_germ_df = pd.DataFrame(per_germ).sort_values("n", ascending=False)
    table_df = per_germ_df[per_germ_df["n"] >= MIN_N_PER_GERMLINE].copy()
    included = sorted(table_df["germline"].tolist())
    excluded = sorted(per_germ_df[per_germ_df["n"] < MIN_N_PER_GERMLINE]["germline"].tolist())

    print(f"\n    Per-germline results (table restricted to n >= {MIN_N_PER_GERMLINE}):")
    print(f"    {'germline':<10} {'n':>6} {'pass':>6} {'fail':>6} {'pass_frac':>9} {'AUC':>8}")
    print("    " + "-" * 50)
    for _, r in table_df.iterrows():
        aucs = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A"
        print(f"    {r['germline']:<10} {r['n']:>6} {r['n_pass']:>6} {r['n_fail']:>6} "
              f"{r['n_pass']/r['n']:>9.3f} {aucs:>8}")
    if excluded:
        print(f"\n    Germlines EXCLUDED from table (n < {MIN_N_PER_GERMLINE}) but still in pooled AUC:")
        for _, r in per_germ_df[per_germ_df["n"] < MIN_N_PER_GERMLINE].iterrows():
            aucs = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A"
            print(f"      {r['germline']:<10} n={r['n']:>5} pass={r['n_pass']:>4} "
                  f"fail={r['n_fail']:>4} AUC={aucs}")

    print(f"\n    Pooled AUC_germline_out computed over ALL held-out germlines "
          f"({merged['vh_base'].nunique()} germlines, all {n_matched} antibodies).")
    print(f"    Germlines with n>={MIN_N_PER_GERMLINE} (in table): {included}")
    print(f"    Germlines with n< {MIN_N_PER_GERMLINE} (pooled only): {excluded if excluded else 'none'}")
    print(f"\n    AUC_germline_out (pooled) = {auc_germ_out:.4f}")

    # ---- 5. Headline comparison ---------------------------------------------
    drop = auc_cluster - auc_germ_out
    print("\n" + "=" * 78)
    print("[5] HEADLINE COMPARISON (identical classifier)")
    print("=" * 78)
    print(f"    AUC_cluster      (HCDR3-cluster 10-fold CV) = {auc_cluster:.4f}")
    print(f"    AUC_germline_out (leave-one-VH-germline-out)= {auc_germ_out:.4f}")
    print(f"    absolute drop                               = {drop:.4f}")
    print(f"    (manuscript ED Table 1 reports 0.959 for XGBoost+AbLang2)")

    # ---- 6. OPTIONAL: LeaveOneGroupOut by base VH+VL pair -------------------
    print("\n" + "=" * 78)
    print("[6] OPTIONAL protocol: LeaveOneGroupOut grouped by base vh+vl pair")
    print("=" * 78)
    groups_pair = merged["vh_vl_pair"].values
    n_pairs = merged["vh_vl_pair"].nunique()
    ppred, py = [], []
    pair_rows = []
    for tr, te in logo.split(X, y, groups=groups_pair):
        name = groups_pair[te][0]
        yte = y[te]
        clf, _ = make_classifier()
        clf.fit(X[tr], y[tr])
        pte = predict_proba_pass(clf, X[te])
        ppred.append(pte)
        py.append(yte)
        npass, nfail = int((yte == 1).sum()), int((yte == 0).sum())
        both = npass > 0 and nfail > 0
        pair_rows.append({"pair": name, "n": len(te), "n_pass": npass, "n_fail": nfail,
                          "auc": roc_auc_score(yte, pte) if both else None})
    ppred = np.concatenate(ppred)
    py = np.concatenate(py)
    auc_pair_out = roc_auc_score(py, ppred)
    pair_df = pd.DataFrame(pair_rows).sort_values("n", ascending=False)
    print(f"    n pairs (LOGO folds) = {n_pairs}; pooled over all {n_matched} antibodies.")
    print(f"\n    Per-pair results (n >= {MIN_N_PER_GERMLINE}):")
    print(f"    {'pair':<20} {'n':>6} {'pass':>6} {'fail':>6} {'AUC':>8}")
    print("    " + "-" * 52)
    for _, r in pair_df[pair_df["n"] >= MIN_N_PER_GERMLINE].iterrows():
        aucs = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A"
        print(f"    {r['pair']:<20} {r['n']:>6} {r['n_pass']:>6} {r['n_fail']:>6} {aucs:>8}")
    print(f"\n    AUC_pair_out (pooled) = {auc_pair_out:.4f}")
    print(f"    absolute drop vs AUC_cluster = {auc_cluster - auc_pair_out:.4f}")

    # ---- Summary block -------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY (all values from this run)")
    print("=" * 78)
    print(f"  classifier              : {clf_name}")
    print(f"  n matched               : {n_matched}  (pass={n_pass}, fail={n_fail})")
    print(f"  AUC_cluster   (anchor)  : {auc_cluster:.4f}")
    print(f"  AUC_germline_out (main) : {auc_germ_out:.4f}")
    print(f"  drop (cluster - germ)   : {drop:.4f}")
    print(f"  AUC_pair_out  (optional): {auc_pair_out:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
