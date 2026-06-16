"""
build_balanced_dataset.py
Always produces TWO output datasets from one run:

  Dataset 1 -- BALANCED
    Majority downsampled to exactly n_minority.
    Both classes equal size.
    File: <input>_<label>_balanced.xlsx

  Dataset 2 -- IMBALANCED (at least --min-total samples)
    100% minority class kept.
    Majority downsampled to (min_total - n_minority), keeping natural
    imbalance but reaching a minimum dataset size.
    File: <input>_<label>_imbalanced_<min_total>.xlsx

The expensive step (clustering or GS+OOF scoring) runs ONCE.
Both datasets are sliced from the same ranked majority pool.

Three strategies via --strategy:

  cluster  (default)
    Greedy Levenshtein clustering on CDR3.
    Cluster representatives ranked first (most diverse), then
    remaining rows fill up the imbalanced dataset if needed.

  kmer_consensus
    GridSearchCV (roc_auc) finds best RF + XGBoost params.
    cross_val_predict produces out-of-fold (OOF) probabilities --
    each majority sample scored by a model that never saw it.
    Consensus: RF_oof >= min_prob AND XGB_oof >= min_prob -> keep.
    Ranked by mean OOF confidence for both datasets.

  combined  (recommended)
    Runs BOTH cluster and kmer_consensus, then merges results into
    a single tiered priority pool:
      Tier 1: cluster rep  AND  OOF consensus  -> diverse + confident
      Tier 2: cluster rep  AND  NOT consensus  -> diverse but uncertain
      Tier 3: NOT rep      AND  OOF consensus  -> confident but redundant
      Tier 4: NOT rep      AND  NOT consensus  -> last resort fill-up
    Within each tier, ranked by mean OOF prob descending.
    Maximises both sequence diversity and label confidence simultaneously.

NA/empty label rows removed automatically. Majority class auto-detected.
Works with any binary label: sec_filter, psr_filter, spr_filter, etc.

Usage
-----
  python build_balanced_dataset.py --input ipi_sec.xlsx --label sec_filter
  python build_balanced_dataset.py --input ipi_sec.xlsx --label sec_filter --strategy combined
  python build_balanced_dataset.py --input ipi_sec.xlsx --label sec_filter --strategy combined --min-total 6000
  python build_balanced_dataset.py --input ipi_sec.xlsx --label sec_filter --strategy combined --min-prob 0.7 --cv 3
  python build_balanced_dataset.py --input ipi_sec.xlsx --label sec_filter --strategy kmer_consensus
  python build_balanced_dataset.py --input ipi_sec.xlsx --label sec_filter --strategy cluster --threshold 0.8
"""

import argparse
import os
import sys
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd


# ============================================================
# SHARED HELPERS
# ============================================================

def _clean_label(df, label_col):
    before = len(df)
    df = df[df[label_col].notna()]
    df = df[df[label_col].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print("[clean] Removed", dropped, "rows with NA/empty", repr(label_col))
    else:
        print("[clean] No NA/empty values in", repr(label_col))
    df = df.copy()
    df[label_col] = df[label_col].astype(int)
    return df.reset_index(drop=True)


def _clean_seq(df, col):
    before = len(df)
    df = df[df[col].notna()]
    df = df[df[col].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print("[clean] Removed", dropped, "rows with NA/empty", repr(col))
    return df.reset_index(drop=True)


def _detect_classes(df, label_col):
    n1 = int((df[label_col] == 1).sum())
    n0 = int((df[label_col] == 0).sum())
    if n1 >= n0:
        return 1, 0, n1, n0
    return 0, 1, n0, n1


def _print_dataset_summary(df, label_col, tag, path):
    n1  = int((df[label_col] == 1).sum())
    n0  = int((df[label_col] == 0).sum())
    tot = len(df)
    print("  [" + tag + "]")
    print("    Class 1 :", f"{n1:,}")
    print("    Class 0 :", f"{n0:,}")
    print("    Total   :", f"{tot:,}")
    print("    Ratio   :", f"{n1/tot:.1%}", "class-1")
    print("    File    :", path)


def _save_outputs(df_balanced, df_imbalanced, df_rejected,
                  label_col, bal_path, imb_path, reject_path, strategy_note):
    sep = "=" * 62
    print("")
    print(sep)
    print("  OUTPUT SUMMARY  [" + label_col + "]")
    print("-" * 62)
    print("  Strategy :", strategy_note)
    print("-" * 62)
    _print_dataset_summary(df_balanced,   label_col, "BALANCED",   bal_path)
    print("")
    _print_dataset_summary(df_imbalanced, label_col, "IMBALANCED", imb_path)
    print("")
    print("  Majority rejected :", f"{len(df_rejected):,}", " ->", reject_path)
    print(sep)
    print("")

    df_balanced.to_excel(bal_path, index=False)
    print("[save] Balanced    ->", bal_path)

    df_imbalanced.to_excel(imb_path, index=False)
    print("[save] Imbalanced  ->", imb_path)

    if len(df_rejected):
        df_rejected.to_excel(reject_path, index=False)
        print("[save] Rejected    ->", reject_path)
    print("")
    print("[done]")


# ============================================================
# STRATEGY 1 -- CLUSTER-BASED (Levenshtein)
# ============================================================

def _levenshtein(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j] + (c1 != c2), curr[-1] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def _norm_lev(s1, s2):
    return 1.0 - _levenshtein(s1, s2) / max(len(s1), len(s2), 1)


def _greedy_cluster(seq_list, threshold):
    centroids, labels = [], []
    for seq in seq_list:
        assigned = False
        for cid, centroid in enumerate(centroids):
            if _norm_lev(seq, centroid) >= threshold:
                labels.append(cid)
                assigned = True
                break
        if not assigned:
            labels.append(len(centroids))
            centroids.append(seq)
    return labels


def _cluster_build_pool(df_majority, cluster_col, threshold, random_state):
    """
    Run clustering once. Returns df_majority with two extra columns:
      _is_rep  : bool  -- True if this row is a cluster representative
      _cluster : int   -- cluster ID
    Representatives are the shortest sequence per cluster (nearest centroid).
    """
    print("")
    print("[cluster] Clustering", f"{len(df_majority):,}", "majority sequences")
    print("[cluster] Column=" + repr(cluster_col) + "  threshold=" + str(threshold))
    print("[cluster] This may take 1-3 minutes for n > 5,000 ...")
    print("")

    seqs = df_majority[cluster_col].astype(str).str.upper().str.strip().tolist()
    cluster_ids = _greedy_cluster(seqs, threshold)

    df_pool = df_majority.copy()
    df_pool["_cluster"] = cluster_ids
    n_clusters = len(set(cluster_ids))
    print("[cluster]", f"{len(df_pool):,}", "sequences ->", f"{n_clusters:,}", "clusters")
    print("[cluster] Average cluster size:", round(len(df_pool) / n_clusters, 1))

    # Mark cluster representatives (shortest CDR3 per cluster)
    df_pool["_seq_len"] = df_pool[cluster_col].str.len()
    rep_idx = (
        df_pool.sort_values("_seq_len")
        .groupby("_cluster", sort=False)
        .head(1)
        .index
    )
    df_pool["_is_rep"] = False
    df_pool.loc[rep_idx, "_is_rep"] = True
    df_pool = df_pool.drop(columns=["_seq_len"])

    n_reps = df_pool["_is_rep"].sum()
    print("[cluster]", f"{n_reps:,}", "cluster representatives identified")
    return df_pool


def _cluster_select_n(df_pool, n_target, random_state):
    """
    Select n_target rows from cluster pool.
    Representatives are prioritised; remainder filled randomly.
    Returns (selected_df, note_str).
    """
    reps      = df_pool[df_pool["_is_rep"]].copy()
    non_reps  = df_pool[~df_pool["_is_rep"]].copy()
    n_reps    = len(reps)

    if n_reps >= n_target:
        selected = reps.sample(n=n_target, random_state=random_state)
        note = f"{n_reps:,} reps -> sampled {n_target:,}"
    else:
        n_topup  = n_target - n_reps
        avail    = min(n_topup, len(non_reps))
        topup    = non_reps.sample(n=avail, random_state=random_state)
        selected = pd.concat([reps, topup], ignore_index=True)
        if avail < n_topup:
            print("  [WARN] Only", len(selected), "majority rows available (< target", n_target, ")")
        note = f"{n_reps:,} reps + {avail:,} random = {len(selected):,}"

    drop_cols = ["_cluster", "_is_rep"]
    return selected.drop(columns=drop_cols, errors="ignore").reset_index(drop=True), note


def downsample_cluster(df_majority, n_balanced, n_imbalanced, cluster_col, threshold, random_state):
    """
    Run clustering once, then select at two different sizes.
    Returns (selected_balanced, selected_imbalanced, strategy_base_note).
    """
    df_pool = _cluster_build_pool(df_majority, cluster_col, threshold, random_state)
    prefix  = "cluster (col=" + cluster_col + ", thr=" + str(threshold) + ")"

    print("[cluster] Selecting balanced  n=" + f"{n_balanced:,}")
    sel_bal,  note_bal  = _cluster_select_n(df_pool, n_balanced,   random_state)

    print("[cluster] Selecting imbalanced n=" + f"{n_imbalanced:,}")
    sel_imb,  note_imb  = _cluster_select_n(df_pool, n_imbalanced, random_state)

    return sel_bal, sel_imb, prefix + " | balanced: " + note_bal + " | imbalanced: " + note_imb


# ============================================================
# STRATEGY 2 -- RF + XGBoost K-MER CONSENSUS
# ============================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _build_vocab(k_sizes):
    vocab = []
    for k in k_sizes:
        vocab += ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]
    return vocab


def _kmer_features(sequences, vocab, k_sizes):
    """Normalised k-mer frequency matrix, shape (n_seqs, len(vocab))."""
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    X = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for row_i, seq in enumerate(sequences):
        seq = seq.upper().replace("-", "")
        for k in k_sizes:
            kmers  = [seq[j:j+k] for j in range(len(seq) - k + 1)]
            counts = Counter(kmers)
            total  = max(sum(counts.values()), 1)
            for kmer, cnt in counts.items():
                if kmer in vocab_index:
                    X[row_i, vocab_index[kmer]] = cnt / total
    return X


def _grid_search_rf(X, y, cv, random_state):
    """
    GridSearchCV for RandomForest.
    Grid is intentionally compact (9 combos) so it runs in ~1-2 min.
    Scored on roc_auc to handle class imbalance correctly.

    Param grid rationale:
      n_estimators  : 200/300/500 -- sweet spot for k-mer features;
                      >500 rarely helps but doubles runtime.
      max_depth     : None (full trees) vs 20/30 -- controls overfitting.
      class_weight  : always balanced (imbalanced dataset).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth":    [20, 30, None],
    }
    base = RandomForestClassifier(
        class_weight="balanced", min_samples_leaf=2,
        random_state=random_state, n_jobs=-1)

    gs = GridSearchCV(
        base, param_grid,
        scoring="roc_auc", cv=cv,
        n_jobs=-1, refit=True, verbose=0)
    gs.fit(X, y)

    print("[RF-GS]  best params :", gs.best_params_)
    print("[RF-GS]  best CV AUC :", round(gs.best_score_, 4))
    _print_cv_table(gs, "RF")
    return gs.best_estimator_


def _grid_search_xgb(X, y, cv, random_state):
    """
    GridSearchCV for XGBoost.
    Grid: 12 combos -- balanced between coverage and speed.

    Param grid rationale:
      max_depth      : 4/6 -- shallower trees reduce overfitting on k-mer features.
      learning_rate  : 0.05/0.1 -- lower lr + more trees generalises better.
      n_estimators   : 200/300/500 -- paired with lr.
      subsample      : 0.8 fixed (good default for antibody sequence features).
      scale_pos_weight: auto-computed from class ratio (handles imbalance).
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    param_grid = {
        "n_estimators":  [200, 300, 500],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    base = XGBClassifier(
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=random_state, verbosity=0)

    gs = GridSearchCV(
        base, param_grid,
        scoring="roc_auc", cv=cv,
        n_jobs=-1, refit=True, verbose=0)
    gs.fit(X, y)

    print("[XGB-GS] best params :", gs.best_params_)
    print("[XGB-GS] best CV AUC :", round(gs.best_score_, 4))
    _print_cv_table(gs, "XGB")
    return gs.best_estimator_


def _print_cv_table(gs, name):
    """Print a compact fold-result table from GridSearchCV results."""
    res    = gs.cv_results_
    best_i = gs.best_index_
    n_folds = sum(1 for k in res if k.startswith("split") and k.endswith("_test_score"))
    fold_scores = [res["split" + str(f) + "_test_score"][best_i] for f in range(n_folds)]
    mean = res["mean_test_score"][best_i]
    std  = res["std_test_score"][best_i]
    fold_str = "  ".join(f"F{i+1}={s:.3f}" for i, s in enumerate(fold_scores))
    print("[" + name + "-GS]  best-param folds: " + fold_str)
    print("[" + name + "-GS]  mean=" + f"{mean:.4f}" + "  std=" + f"{std:.4f}")


def _kmer_score_pool(df_majority, df_full, label_col, kmer_col,
                     random_state, k_sizes, cv):
    """
    Stage 1: GridSearchCV  -- find best RF + XGBoost hyperparameters.
    Stage 2: cross_val_predict -- generate honest OOF probabilities
             (each sample scored by a model that never saw it).

    Returns df_majority with added columns:
      _rf_oof_prob   : RF out-of-fold probability for majority class
      _xgb_oof_prob  : XGB out-of-fold probability for majority class
      _mean_oof_prob : mean of the two
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    majority_val = int(df_majority[label_col].iloc[0])
    k_str = "+".join(str(k) for k in k_sizes)

    # Identify majority-row positions within full dataset
    df_full_reset = df_full.reset_index(drop=True)
    majority_mask = (df_full_reset[label_col] == majority_val).values

    print("")
    print("[kmer] Building", k_str + "-mer vocabulary ...")
    vocab = _build_vocab(k_sizes)
    print("[kmer] Vocabulary size:", f"{len(vocab):,}", "features")

    print("[kmer] Encoding full dataset (" + f"{len(df_full_reset):,}" + " sequences) on " + repr(kmer_col) + " ...")
    X_full = _kmer_features(
        df_full_reset[kmer_col].astype(str).str.upper().str.strip().tolist(),
        vocab, k_sizes)
    y_full = df_full_reset[label_col].values.astype(int)

    n_pos = int((y_full == 1).sum())
    n_neg = int((y_full == 0).sum())
    print("")
    print("[kmer] Full dataset: n=" + f"{len(y_full):,}"
          + "  class1=" + f"{n_pos:,}" + "  class0=" + f"{n_neg:,}")
    print("[kmer] GridSearchCV + cross_val_predict: cv=" + str(cv) + "  scoring=roc_auc")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    print("")
    print("[RF]  Stage 1: GridSearchCV  (9 combos x " + str(cv) + " folds) ...")
    rf_best = _grid_search_rf(X_full, y_full, cv=skf, random_state=random_state)
    print("[RF]  Stage 2: cross_val_predict OOF probabilities ...")
    rf_oof_all = cross_val_predict(
        rf_best, X_full, y_full, cv=skf, method="predict_proba", n_jobs=-1)
    maj_col_rf = list(rf_best.classes_).index(majority_val)
    print("[RF]  OOF AUC:", round(roc_auc_score(y_full, rf_oof_all[:, 1]), 4))

    print("")
    print("[XGB] Stage 1: GridSearchCV  (12 combos x " + str(cv) + " folds) ...")
    xgb_best = _grid_search_xgb(X_full, y_full, cv=skf, random_state=random_state)
    print("[XGB] Stage 2: cross_val_predict OOF probabilities ...")
    xgb_oof_all = cross_val_predict(
        xgb_best, X_full, y_full, cv=skf, method="predict_proba", n_jobs=-1)
    maj_col_xgb = list(xgb_best.classes_).index(majority_val)
    print("[XGB] OOF AUC:", round(roc_auc_score(y_full, xgb_oof_all[:, 1]), 4))

    # Extract OOF probs for majority-class rows only
    df_scored = df_majority.copy().reset_index(drop=True)
    df_scored["_rf_oof_prob"]   = rf_oof_all[majority_mask, maj_col_rf]
    df_scored["_xgb_oof_prob"]  = xgb_oof_all[majority_mask, maj_col_xgb]
    df_scored["_mean_oof_prob"] = (df_scored["_rf_oof_prob"] + df_scored["_xgb_oof_prob"]) / 2.0
    return df_scored


def _kmer_select_n(df_scored, n_target, min_prob, label_str=""):
    """
    Select n_target rows from an OOF-scored majority pool.
    Consensus survivors (both RF+XGB >= min_prob) are prioritised,
    ranked by mean OOF confidence. Non-survivors fill up if needed.
    Returns (selected_df, note_str).
    """
    rf_pass        = df_scored["_rf_oof_prob"]  >= min_prob
    xgb_pass       = df_scored["_xgb_oof_prob"] >= min_prob
    consensus_mask = rf_pass & xgb_pass
    n_survivors    = int(consensus_mask.sum())

    survivors     = df_scored[consensus_mask].copy()
    non_survivors = df_scored[~consensus_mask].copy()

    tag = label_str + " " if label_str else ""
    print("[OOF " + tag + "filter] min_prob=" + str(min_prob)
          + "  consensus=" + f"{n_survivors:,}" + "/" + f"{len(df_scored):,}"
          + "  target=" + f"{n_target:,}")

    if n_survivors >= n_target:
        selected = survivors.sort_values("_mean_oof_prob", ascending=False).head(n_target)
        note = f"{n_survivors:,} survivors -> top-{n_target:,} by OOF confidence"
    else:
        n_topup = n_target - n_survivors
        if n_topup > 0:
            print("  [WARN " + tag + "] Only", n_survivors, "consensus survivors < target", n_target)
            print("  [WARN " + tag + "] Topping up with", n_topup, "next-highest OOF confidence rows.")
        topup    = non_survivors.sort_values("_mean_oof_prob", ascending=False).head(n_topup)
        selected = pd.concat([survivors, topup], ignore_index=True)
        note = f"{n_survivors:,} consensus + {len(topup):,} top-up = {len(selected):,}"

    drop_cols = ["_rf_oof_prob", "_xgb_oof_prob", "_mean_oof_prob"]
    return selected.drop(columns=drop_cols, errors="ignore").reset_index(drop=True), note


def downsample_kmer_consensus(df_majority, df_full, label_col, kmer_col,
                               n_balanced, n_imbalanced, min_prob, random_state,
                               k_sizes=(1, 2, 3), cv=5):
    """
    Score majority pool once with OOF probabilities (expensive).
    Select at two different sizes (cheap).
    Returns (selected_balanced, selected_imbalanced, strategy_base_note).
    """
    df_scored = _kmer_score_pool(
        df_majority, df_full, label_col, kmer_col, random_state, k_sizes, cv)

    prefix = "kmer_consensus OOF (col=" + kmer_col + ", min_prob=" + str(min_prob) + ", cv=" + str(cv) + ")"

    print("")
    print("[kmer] Selecting balanced  n=" + f"{n_balanced:,}")
    sel_bal, note_bal = _kmer_select_n(df_scored, n_balanced,   min_prob, "balanced")

    print("[kmer] Selecting imbalanced n=" + f"{n_imbalanced:,}")
    sel_imb, note_imb = _kmer_select_n(df_scored, n_imbalanced, min_prob, "imbalanced")

    return sel_bal, sel_imb, prefix + " | balanced: " + note_bal + " | imbalanced: " + note_imb

def _combined_select_n(df_pool, n_target, floor_prob=0.5, label_str=""):
    """
    Round-robin selection across CDR3 clusters with strict dual-model
    consensus filtering.

    Quality rule (per row):
        KEEP : RF_oof >= floor_prob AND XGB_oof >= floor_prob
        DROP : anything else — including cases where only one model agrees

    Cluster rule:
        Clusters with zero passing members are dropped entirely.
        No fallback, no partial rescue — both models must agree.

    Rationale:
        A high mean_oof can mask one model strongly disagreeing
        (RF=0.9, XGB=0.1 → mean=0.5, but XGB predicts FAIL).
        Partial-consensus members and their clusters add noise that
        hurts downstream model accuracy more than their diversity helps.
        Strict dual consensus sacrifices some cluster coverage but yields
        a cleaner, more label-reliable training set.

    Round-robin (across clusters that have >= 1 passing member):
        Round 1: pick best passing member from every qualifying cluster
        Round 2: pick 2nd-best from every qualifying cluster
        ...until n_target reached.
    """
    tag = "[combined" + (" " + label_str if label_str else "") + "]"
    df  = df_pool.copy().reset_index(drop=True)

    # ── Strict dual-model consensus ───────────────────────────────────────────
    df["_pass"] = ((df["_rf_oof_prob"]  >= floor_prob) &
                   (df["_xgb_oof_prob"] >= floor_prob))

    n_pass = int(df["_pass"].sum())
    n_drop = len(df) - n_pass

    # ── Cluster-level stats ───────────────────────────────────────────────────
    cluster_has_pass   = df.groupby("_cluster")["_pass"].any()
    n_clusters_total   = df["_cluster"].nunique()
    n_clusters_pass    = int(cluster_has_pass.sum())
    n_clusters_dropped = n_clusters_total - n_clusters_pass

    print("  " + tag + " floor_prob=" + str(floor_prob)
          + "  (strict: BOTH RF_oof AND XGB_oof >= floor)")
    print("  " + tag + " rows passing : " + f"{n_pass:,}"
          + " / " + f"{len(df):,}"
          + " (" + f"{n_pass/len(df):.1%}" + ")")
    print("  " + tag + " rows dropped : " + f"{n_drop:,}"
          + "  (at least one model < floor)")
    print("  " + tag + " clusters kept    : " + f"{n_clusters_pass:,}"
          + " / " + f"{n_clusters_total:,}"
          + "  (have >= 1 passing member)")
    print("  " + tag + " clusters dropped : " + f"{n_clusters_dropped:,}"
          + " / " + f"{n_clusters_total:,}"
          + "  (zero passing members -> excluded)")
    print("  " + tag + " target : " + f"{n_target:,}")

    # ── Effective pool: only passing rows from qualifying clusters ─────────────
    effective_pool = df[df["_pass"]].copy().dropna(
        subset=["_mean_oof_prob"]).reset_index(drop=True)

    if len(effective_pool) == 0:
        print("  [WARN " + tag + "] No rows passed dual consensus. "
              "Lower --floor-prob.")
        drop_cols = ["_cluster", "_rf_oof_prob", "_xgb_oof_prob",
                     "_mean_oof_prob", "_pass"]
        return df.drop(columns=drop_cols, errors="ignore").head(0), \
               "no rows passed floor=" + str(floor_prob)

    # ── Within-cluster rank by mean OOF prob descending ───────────────────────
    effective_pool["_within_rank"] = (
        effective_pool.groupby("_cluster")["_mean_oof_prob"]
        .rank(method="first", ascending=False)
        .fillna(1).astype(int)
    )

    n_effective = len(effective_pool)
    max_round   = int(effective_pool["_within_rank"].max())

    print("  " + tag + " effective pool : " + f"{n_effective:,}"
          + " rows / " + f"{n_clusters_pass:,}" + " clusters")
    print("  " + tag + " max depth      : " + f"{max_round}")

    # ── Round-robin ───────────────────────────────────────────────────────────
    selected_idx = []
    for rnd in range(1, max_round + 1):
        round_candidates = (
            effective_pool[effective_pool["_within_rank"] == rnd]
            .sort_values("_mean_oof_prob", ascending=False)
        )
        for idx in round_candidates.index:
            selected_idx.append(idx)
            if len(selected_idx) >= n_target:
                break
        if len(selected_idx) >= n_target:
            break

    selected = effective_pool.loc[selected_idx].copy()

    # ── Fill-up if strict dual-consensus pool is insufficient ─────────────────
    # Graceful degradation when n_clusters_pass * depth < n_target:
    #   Phase 1: single-model agreement (RF>=floor OR XGB>=floor, not both)
    #            ranked by mean_oof_prob descending — best of the partial
    #   Phase 2: last resort — neither model agrees — ranked by mean_oof_prob
    #            only reached if phase 1 also cannot fill the gap
    # Each phase prints a clear warning so the user knows fill-up occurred.
    if len(selected) < n_target:
        n_needed = n_target - len(selected)
        sel_barcodes = set(selected["BARCODE"]) if "BARCODE" in selected.columns else set()

        dropped_rows = (
            df[~df["_pass"]]
            .copy()
            .pipe(lambda d: d[~d["BARCODE"].isin(sel_barcodes)]
                  if "BARCODE" in d.columns else d)
            .sort_values("_mean_oof_prob", ascending=False)
        )

        # Phase 1: single-model agreement
        single_agree = dropped_rows[
            (dropped_rows["_rf_oof_prob"]  >= floor_prob) |
            (dropped_rows["_xgb_oof_prob"] >= floor_prob)
        ].head(n_needed)

        if len(single_agree) > 0:
            print("  [WARN " + tag + "] Dual consensus gave only "
                  + str(len(selected)) + " / " + str(n_target) + " needed.")
            print("  [WARN " + tag + "] Fill-up phase 1: adding "
                  + str(len(single_agree))
                  + " single-model rows (one model >= " + str(floor_prob)
                  + "), ranked by mean OOF prob.")
            selected = pd.concat([selected, single_agree], ignore_index=True)
            sel_barcodes = set(selected["BARCODE"]) if "BARCODE" in selected.columns else set()

        # Phase 2: last resort — neither model agrees
        if len(selected) < n_target:
            n_needed = n_target - len(selected)
            last_resort = (
                dropped_rows[
                    ~dropped_rows["BARCODE"].isin(sel_barcodes) &
                    (dropped_rows["_rf_oof_prob"]  < floor_prob) &
                    (dropped_rows["_xgb_oof_prob"] < floor_prob)
                ].head(n_needed)
            ) if "BARCODE" in dropped_rows.columns else dropped_rows.head(n_needed)

            if len(last_resort) > 0:
                print("  [WARN " + tag + "] Fill-up phase 2 (last resort): adding "
                      + str(len(last_resort))
                      + " rows where neither model >= " + str(floor_prob) + ".")
                print("  [WARN " + tag + "] Consider lowering --floor-prob "
                      "(currently " + str(floor_prob) + ") to avoid last-resort fill.")
                selected = pd.concat([selected, last_resort], ignore_index=True)

        if len(selected) < n_target:
            print("  [WARN " + tag + "] After all fill-up phases: "
                  + str(len(selected)) + " rows available "
                  "(< target " + str(n_target) + "). "
                  "Majority pool exhausted — lower --floor-prob or --min-total.")

    # ── Final stats ───────────────────────────────────────────────────────────
    max_round_used  = int(selected["_within_rank"].max()) if "_within_rank" in selected.columns and len(selected) else 0
    n_round1        = int((selected["_within_rank"] == 1).sum()) if "_within_rank" in selected.columns else 0
    n_sel_dual      = int(selected["_pass"].sum()) if "_pass" in selected.columns else len(selected)
    n_sel_fillup    = len(selected) - n_sel_dual
    mean_p          = float(selected["_mean_oof_prob"].mean()) if "_mean_oof_prob" in selected.columns else 0.0
    min_p           = float(selected["_mean_oof_prob"].min())  if "_mean_oof_prob" in selected.columns else 0.0

    print("  " + tag + " rounds used : 1->" + str(max_round_used)
          + "  (round-1 = " + str(n_round1)
          + "/" + str(n_clusters_pass) + " clusters)")
    print("  " + tag + " selected    : " + f"{len(selected):,}"
          + "  (dual-consensus=" + str(n_sel_dual)
          + "  fill-up=" + str(n_sel_fillup) + ")")
    print("  " + tag + " OOF prob    : mean=" + f"{mean_p:.3f}"
          + "  min=" + f"{min_p:.3f}")

    note = ("round-robin strict-dual: "
            + str(n_clusters_pass) + " clusters kept, "
            + str(n_clusters_dropped) + " dropped, "
            + "floor=" + str(floor_prob) + ", "
            + str(max_round_used) + " round(s), "
            + "selected=" + str(len(selected))
            + " [dual=" + str(n_sel_dual)
            + " fillup=" + str(n_sel_fillup) + "]"
            + " mean_oof=" + f"{mean_p:.3f}")

    drop_cols = ["_cluster", "_rf_oof_prob", "_xgb_oof_prob",
                 "_mean_oof_prob", "_within_rank", "_pass"]
    return selected.drop(columns=drop_cols, errors="ignore").reset_index(drop=True), note


def downsample_combined(df_majority, df_full, label_col, cluster_col, threshold,
                        kmer_col, n_balanced, n_imbalanced,
                        random_state, k_sizes=(1, 2, 3), cv=5, floor_prob=0.5):
    """
    Step 1: CDR3 clustering  -> assigns _cluster ID to every majority row.
    Step 2: OOF scoring      -> assigns _rf_oof_prob, _xgb_oof_prob,
                                _mean_oof_prob to every majority row.
    Step 3: Round-robin selection across clusters, ranked within each
            cluster by mean OOF probability descending, with a soft
            per-cluster floor (floor_prob) to exclude likely mislabels.

    floor_prob guideline:
      0.5  (default) -- exclude only samples both models predict as FAIL.
                        Safe for any dataset. Preserves boundary samples.
      0.6            -- stricter; exclude uncertain PASS samples.
                        Use if your SEC labels are known to be noisy.

    Both expensive steps run exactly once. Two output sizes are sliced
    from the same ranked pool.

    Returns (selected_balanced, selected_imbalanced, strategy_note).
    """
    # ── Step 1: cluster ───────────────────────────────────────────────────────
    print("")
    print("[combined] Step 1/2: CDR3 clustering ...")
    df_pool = _cluster_build_pool(df_majority, cluster_col, threshold, random_state)
    df_pool = df_pool.drop(columns=["_is_rep"], errors="ignore")

    # ── Step 2: OOF scoring ───────────────────────────────────────────────────
    print("")
    print("[combined] Step 2/2: GridSearchCV + OOF scoring ...")
    df_scored = _kmer_score_pool(
        df_majority, df_full, label_col, kmer_col, random_state, k_sizes, cv)

    df_pool = df_pool.reset_index(drop=True)
    df_pool["_rf_oof_prob"]   = df_scored["_rf_oof_prob"].values
    df_pool["_xgb_oof_prob"]  = df_scored["_xgb_oof_prob"].values
    df_pool["_mean_oof_prob"] = df_scored["_mean_oof_prob"].values

    # ── Pool summary ──────────────────────────────────────────────────────────
    n_clusters   = df_pool["_cluster"].nunique()
    n_above_floor = int((df_pool["_mean_oof_prob"] >= floor_prob).sum())
    print("")
    print("[combined] Pool: " + f"{len(df_pool):,}" + " majority rows  "
          + f"{n_clusters:,}" + " CDR3 clusters")
    print("[combined] OOF mean=" + f"{df_pool['_mean_oof_prob'].mean():.3f}"
          + "  min=" + f"{df_pool['_mean_oof_prob'].min():.3f}"
          + "  max=" + f"{df_pool['_mean_oof_prob'].max():.3f}")
    print("[combined] Above floor (>=" + str(floor_prob) + ")  : "
          + f"{n_above_floor:,}" + " / " + f"{len(df_pool):,}"
          + " (" + f"{n_above_floor/len(df_pool):.1%}" + ")")

    prefix = ("combined round-robin (cluster_col=" + cluster_col
              + ", thr=" + str(threshold)
              + ", kmer_col=" + kmer_col
              + ", floor_prob=" + str(floor_prob)
              + ", cv=" + str(cv) + ")")

    # ── Step 3: round-robin at two target sizes ───────────────────────────────
    print("")
    print("[combined] Selecting imbalanced n=" + f"{n_imbalanced:,}")
    sel_imb, note_imb = _combined_select_n(
        df_pool, n_imbalanced, floor_prob=floor_prob, label_str="imbalanced")

    print("")
    print("[combined] Selecting balanced  n=" + f"{n_balanced:,}")
    sel_bal, note_bal = _combined_select_n(
        df_pool, n_balanced, floor_prob=floor_prob, label_str="balanced")

    return sel_bal, sel_imb, prefix + " | imbalanced: " + note_imb + " | balanced: " + note_bal

def build_balanced_dataset(
    input_path, label_col,
    strategy="combined", cluster_col="CDR3", threshold=0.8,
    kmer_col="CDR3", min_prob=0.6, k_sizes=(1, 2, 3), cv=5,
    floor_prob=0.5,
    min_total=5000,
    bal_path=None, imb_path=None, reject_path=None,
    random_state=42,
):
    """
    Parameters
    ----------
    min_total  : minimum total size for the imbalanced dataset (default 5000).
                 Majority size for imbalanced = max(min_total - n_minority, n_minority + 1).
                 Capped at n_majority (cannot select more than available).
    bal_path   : output path for balanced dataset   (auto-derived if None)
    imb_path   : output path for imbalanced dataset (auto-derived if None)
    reject_path: output path for rejected rows      (auto-derived if None)
    """
    np.random.seed(random_state)

    base = os.path.splitext(input_path)[0]
    if bal_path    is None: bal_path    = base + "_" + label_col + "_balanced.xlsx"
    if imb_path    is None: imb_path    = base + "_" + label_col + "_imbalanced_" + str(min_total) + ".xlsx"
    if reject_path is None: reject_path = base + "_" + label_col + "_majority_rejected.xlsx"

    seq_col = cluster_col  # for column validation; combined needs both CDR3 and kmer_col
    sep = "=" * 62

    print("")
    print(sep)
    print("  Balanced + Imbalanced Dataset Builder")
    print("-" * 62)
    print("  Input          :", input_path)
    print("  Label col      :", label_col)
    print("  Strategy       :", strategy)
    if strategy in ("cluster", "combined"):
        print("  Cluster col    :", cluster_col)
        print("  Similarity thr :", threshold)
    if strategy in ("kmer_consensus", "combined"):
        print("  K-mer col      :", kmer_col)
        print("  K sizes        :", list(k_sizes))
        print("  CV folds (GS)  :", cv)
    if strategy == "kmer_consensus":
        print("  Min prob       :", min_prob)
    if strategy == "combined":
        print("  Floor prob     :", floor_prob,
              " (per-cluster OOF floor: 0.5=remove likely mislabels, 0.6=stricter)")
    print("  Min total      :", min_total, " (imbalanced dataset target size)")
    print("  Balanced out   :", bal_path)
    print("  Imbalanced out :", imb_path)
    print("  Rejected out   :", reject_path)
    print(sep)
    print("")

    # ── Load and validate ─────────────────────────────────────────────────────
    if not os.path.exists(input_path):
        sys.exit("[ERROR] File not found: " + input_path)

    df = pd.read_excel(input_path, dtype={"BARCODE": str})
    print("[load]", f"{len(df):,}", "rows  |  columns:", list(df.columns))

    required = {"BARCODE", "HSEQ", "CDR3", label_col}
    missing  = required - set(df.columns)
    if missing:
        available = [c for c in df.columns if c.endswith("_filter")]
        sys.exit("[ERROR] Missing columns: " + str(missing)
                 + ". Available *_filter columns: " + str(available))

    # Validate all sequence columns needed by chosen strategy
    cols_to_check = []
    if strategy in ("cluster", "combined"):
        cols_to_check.append(cluster_col)
    if strategy in ("kmer_consensus", "combined"):
        cols_to_check.append(kmer_col)
    for col in set(cols_to_check):
        if col not in df.columns:
            sys.exit("[ERROR] Column " + repr(col) + " not found.")

    # Clean on CDR3 (always present); kmer_col cleaned separately if different
    df = _clean_label(df, label_col)
    df = _clean_seq(df, "CDR3")
    if kmer_col != "CDR3" and strategy in ("kmer_consensus", "combined"):
        df = _clean_seq(df, kmer_col)

    unique_vals = set(df[label_col].unique())
    if not unique_vals.issubset({0, 1}):
        sys.exit("[ERROR] " + repr(label_col) + " has values other than 0/1: " + str(unique_vals))
    if len(unique_vals) < 2:
        sys.exit("[ERROR] " + repr(label_col) + " has only one class after cleaning: " + str(unique_vals))

    majority_val, minority_val, n_majority, n_minority = _detect_classes(df, label_col)
    n1 = int((df[label_col] == 1).sum())
    n0 = int((df[label_col] == 0).sum())

    print("")
    print("[data]  Class 1 (" + label_col + "=1) :", f"{n1:,}")
    print("[data]  Class 0 (" + label_col + "=0) :", f"{n0:,}")
    print("[data]  Majority class :", majority_val, f" ({n_majority:,})  <- downsampled")
    print("[data]  Minority class :", minority_val, f" ({n_minority:,})  <- unchanged (100%)")
    print("[data]  Imbalance ratio :", round(n_majority / max(n_minority, 1), 2), "x")

    # ── Compute two majority targets ──────────────────────────────────────────
    # Balanced: exactly n_minority majority samples
    n_maj_balanced = n_minority

    # Imbalanced: enough majority to reach min_total, but always > n_minority
    # so the dataset stays imbalanced (majority class is still larger)
    n_maj_imbalanced = max(min_total - n_minority, n_minority + 1)
    n_maj_imbalanced = min(n_maj_imbalanced, n_majority)   # cap at available
    imb_total        = n_maj_imbalanced + n_minority

    print("")
    print("[targets]  Balanced   : majority=" + f"{n_maj_balanced:,}"
          + "  minority=" + f"{n_minority:,}"
          + "  total=" + f"{n_maj_balanced + n_minority:,}")
    print("[targets]  Imbalanced : majority=" + f"{n_maj_imbalanced:,}"
          + "  minority=" + f"{n_minority:,}"
          + "  total=" + f"{imb_total:,}"
          + "  (min_total=" + str(min_total) + ")")
    if imb_total < min_total:
        print("  [WARN] Only", n_majority, "majority rows available;"
              " imbalanced total (" + str(imb_total) + ") < min_total (" + str(min_total) + ")")
    print("")

    if n_majority <= n_minority:
        print("[INFO] Classes already equal or majority <= minority -- no downsampling needed.")
        for path in [bal_path, imb_path]:
            df.to_excel(path, index=False)
            print("[save] ->", path)
        print("[done]")
        return df, df

    df_majority = df[df[label_col] == majority_val].copy().reset_index(drop=True)
    df_minority = df[df[label_col] == minority_val].copy().reset_index(drop=True)

    # ── Run strategy (expensive part runs once) ───────────────────────────────
    if strategy == "cluster":
        sel_bal, sel_imb, strategy_note = downsample_cluster(
            df_majority, n_maj_balanced, n_maj_imbalanced,
            cluster_col, threshold, random_state)

    elif strategy == "kmer_consensus":
        sel_bal, sel_imb, strategy_note = downsample_kmer_consensus(
            df_majority, df, label_col, kmer_col,
            n_maj_balanced, n_maj_imbalanced,
            min_prob, random_state, k_sizes, cv=cv)

    elif strategy == "combined":
        sel_bal, sel_imb, strategy_note = downsample_combined(
            df_majority, df, label_col,
            cluster_col, threshold,
            kmer_col,
            n_maj_balanced, n_maj_imbalanced,
            random_state, k_sizes, cv=cv, floor_prob=floor_prob)

    else:
        sys.exit("[ERROR] Unknown strategy " + repr(strategy)
                 + ". Choose: cluster | kmer_consensus | combined")

    # ── Assemble both datasets ────────────────────────────────────────────────
    def _assemble(sel_maj):
        return (
            pd.concat([sel_maj, df_minority], ignore_index=True)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    df_balanced   = _assemble(sel_bal)
    df_imbalanced = _assemble(sel_imb)

    # Rejected = majority rows not in the IMBALANCED set (larger selection)
    # (imbalanced always includes at least as many majority rows as balanced)
    imb_barcodes  = set(sel_imb["BARCODE"])
    df_rejected   = df_majority[
        ~df_majority["BARCODE"].isin(imb_barcodes)
    ].reset_index(drop=True)

    _save_outputs(df_balanced, df_imbalanced, df_rejected,
                  label_col, bal_path, imb_path, reject_path, strategy_note)
    return df_balanced, df_imbalanced


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build TWO datasets from one run: balanced + imbalanced (>= min_total). "
            "Three strategies: cluster | kmer_consensus | combined (recommended). "
            "combined = CDR3 diversity + OOF confidence, tiered priority pool. "
            "NA/empty label rows removed automatically. Majority auto-detected."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",       required=True,  help="Input Excel file")
    parser.add_argument("--label",       required=True,
                        help="Binary label column (sec_filter | psr_filter | spr_filter)")
    parser.add_argument("--strategy",    default="combined",
                        choices=["cluster", "kmer_consensus", "combined"],
                        help="Downsampling strategy (default: combined)")
    parser.add_argument("--min-total",   default=5000, type=int, dest="min_total",
                        help="Min total size for imbalanced dataset (default: 5000)")
    parser.add_argument("--cluster-col", default="CDR3", dest="cluster_col",
                        choices=["CDR3", "HSEQ"],
                        help="[cluster] Sequence column for clustering (default: CDR3)")
    parser.add_argument("--threshold",   default=0.8, type=float,
                        help="[cluster] Similarity threshold (default: 0.8)")
    parser.add_argument("--kmer-col",    default="CDR3", dest="kmer_col",
                        choices=["CDR3", "HSEQ"],
                        help="[kmer_consensus] Sequence column for k-mer encoding (default: CDR3)")
    parser.add_argument("--floor-prob",  default=0.5, type=float, dest="floor_prob",
                        help="[combined] Per-cluster OOF floor to exclude likely mislabels "
                             "(default: 0.5 = both models predict FAIL; 0.6 = stricter)")
    parser.add_argument("--min-prob",    default=0.6, type=float, dest="min_prob",
                        help="[kmer_consensus] Global OOF consensus threshold (default: 0.6)")
    parser.add_argument("--cv",          default=5, type=int,
                        help="[kmer_consensus] GridSearchCV folds (default: 5, use 3 for large datasets)")
    parser.add_argument("--bal-output",  default=None, dest="bal_path",
                        help="Balanced output path (default: <input>_<label>_balanced.xlsx)")
    parser.add_argument("--imb-output",  default=None, dest="imb_path",
                        help="Imbalanced output path (default: <input>_<label>_imbalanced_<min_total>.xlsx)")
    parser.add_argument("--rejected",    default=None,
                        help="Rejected rows path (default: <input>_<label>_majority_rejected.xlsx)")
    parser.add_argument("--seed",        default=42, type=int, help="Random seed (default: 42)")
    args = parser.parse_args()

    build_balanced_dataset(
        input_path=args.input,    label_col=args.label,
        strategy=args.strategy,   min_total=args.min_total,
        cluster_col=args.cluster_col, threshold=args.threshold,
        kmer_col=args.kmer_col,   min_prob=args.min_prob,
        floor_prob=args.floor_prob, cv=args.cv,
        bal_path=args.bal_path,   imb_path=args.imb_path,
        reject_path=args.rejected, random_state=args.seed,
    )