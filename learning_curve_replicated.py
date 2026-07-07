#!/usr/bin/env python3
"""Replicated learning curve for DELPHI TransformerLM (frozen PLM embeddings).

Repeats the SAME per-size procedure as
models.transformer_lm.evaluate_sample_size_effect
  balanced stratified subsample
  -> CDR3-cluster-stratified 80/20 split (best-balanced of 5 folds)
  -> train TransformerLMModel on frozen embeddings (batch=32, unchanged)
  -> AUC on the 20% test set
K times per training-set size with varying seeds (subsample, split, and model
init all vary), giving mean AUC and a 95% CI per size.

Three modes:
  --only-clusters FILE   compute the CDR3 cluster column once and save {BARCODE,cluster}
  --aggregate  GLOB      combine worker raw CSVs -> {out}_summary.csv (+ combined raw)
  (default)              run trainings; --work-slice I/N runs grid items where idx%N==I
                         so W workers can share one GPU in parallel.

Run from the DELPHI repo root (needs models/, utils/, config on the path).
"""
import os, sys, json, glob, argparse, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import torch

from models.transformer_lm import TransformerLMModel
from utils.clustering import greedy_clustering_by_levenshtein


def load(db_path, emb_path, target):
    ext = os.path.splitext(db_path)[1].lower()
    df = pd.read_excel(db_path) if ext in ('.xlsx', '.xls') else pd.read_csv(db_path)
    df = df.dropna(subset=[target])
    if 'antigen' in df.columns:
        df = df[~df['antigen'].astype(str).str.contains('test', na=False, case=False)]
    emb = pd.read_csv(emb_path, index_col=0)
    if 'BARCODE' in df.columns:
        df = df.set_index('BARCODE')
    common = df.index.intersection(emb.index)
    if len(common) == 0:
        sys.exit("ERROR: no overlapping BARCODEs between db and embeddings")
    df = df.loc[common]; emb = emb.loc[common]
    return df.reset_index(), emb.reset_index(drop=True)


def attach_clusters(df, cluster_col, cluster_file):
    if cluster_col in df.columns:                      # already ships in the data (IPI) -> use as-is
        print(f"[lc] cluster col already present: {df[cluster_col].nunique():,} clusters", flush=True)
        return df
    if cluster_file:                                   # precomputed -> merge (no re-cluster)
        cl = pd.read_csv(cluster_file)
        col = 'cluster' if 'cluster' in cl.columns else cluster_col
        df = df.merge(cl[['BARCODE', col]].rename(columns={col: cluster_col}),
                      on='BARCODE', how='left')
        if df[cluster_col].isna().any():
            sys.exit(f"ERROR: {int(df[cluster_col].isna().sum())} rows missing a cluster")
        print(f"[lc] clusters from {cluster_file}: {df[cluster_col].nunique():,}", flush=True)
    elif cluster_col not in df.columns:
        thr = float(cluster_col.split('_')[-1])
        print(f"[lc] clustering {len(df):,} CDR3s at {thr} (once) ...", flush=True)
        df[cluster_col] = greedy_clustering_by_levenshtein(df['CDR3'].astype(str).tolist(), thr)
        print(f"[lc] {df[cluster_col].nunique():,} clusters", flush=True)
    return df


def one_run(df, emb, y, size, seed, cluster_col, target, lm, db_stem):
    idx_all = np.arange(len(df))
    sampled = []
    for cls in (0, 1):                                 # balanced stratified subsample
        cls_idx = idx_all[y == cls]
        n_take = min(len(cls_idx), size // 2 + (size % 2 if cls == 1 else 0))
        rng = np.random.default_rng(seed)
        sampled.extend(rng.choice(cls_idx, n_take, replace=False).tolist())
    sampled = sorted(sampled)[:size]
    df_s = df.iloc[sampled]; emb_s = emb.iloc[sampled]; y_s = y[sampled]

    groups = df_s[cluster_col].values                  # CDR3-cluster-stratified 80/20
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    best, best_diff = None, float('inf')
    for tr, te in sgkf.split(np.arange(len(y_s)), y_s, groups):
        d = abs(y_s[te].mean() - y_s.mean())
        if d < best_diff:
            best_diff, best = d, (tr, te)
    tr_idx, te_idx = best
    X_tr, y_tr = emb_s.iloc[tr_idx], y_s[tr_idx]
    X_te, y_te = emb_s.iloc[te_idx], y_s[te_idx]

    torch.manual_seed(seed); np.random.seed(seed)      # model init varies too
    model = TransformerLMModel(); model._lm_name = lm
    model.train(X_tr, y_tr, val_X=X_te, val_y=y_te, epochs=None,
                target=target, db_stem=db_stem)
    probs = np.asarray(model.predict_proba(X_te)).ravel()
    try:
        auc = roc_auc_score(y_te, probs)
    except Exception:
        auc = 0.5
    return dict(size=size, seed=seed, train=len(X_tr), test=len(X_te),
                auc=float(auc),
                acc=float(accuracy_score(y_te, (probs >= 0.5).astype(int))))


def summarize(rdf, out):
    rows = []
    for size, g in rdf.groupby('size'):
        a = g['auc'].values; n = len(a); m = a.mean()
        sd = a.std(ddof=1) if n > 1 else 0.0
        h = stats.t.ppf(0.975, n - 1) * (sd / np.sqrt(n)) if n > 1 else 0.0
        rows.append(dict(size=int(size), n=n, mean_auc=round(m, 4),
                         ci_lo=round(m - h, 4), ci_hi=round(m + h, 4),
                         std=round(sd, 4), min=round(a.min(), 4), max=round(a.max(), 4),
                         aucs=json.dumps([round(float(x), 4) for x in a])))
    sdf = pd.DataFrame(rows).sort_values('size')
    sdf.to_csv(f"{out}_summary.csv", index=False)
    return sdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db')
    ap.add_argument('--emb', default=None, help='defaults to {db}.{lm}.emb.csv')
    ap.add_argument('--lm', default='ablang')
    ap.add_argument('--target', default='psr_filter')
    ap.add_argument('--sizes', default='100,200,300,500,700,900,1200,1500,1900,2400,3000,3800,4700,5800,7000,8500,0')
    ap.add_argument('--repeats', type=int, default=12)
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--cluster-col', default='HCDR3_CLUSTER_0.8')
    ap.add_argument('--cluster-file', default=None)
    ap.add_argument('--work-slice', default=None, help='I/N: run grid items where idx%%N==I')
    ap.add_argument('--only-clusters', default=None, help='compute clusters, save {BARCODE,cluster} here, exit')
    ap.add_argument('--aggregate', default=None, help='glob of *_raw.csv to combine into {out}_summary.csv')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    # ---- aggregate mode ----
    if a.aggregate:
        files = sorted(glob.glob(a.aggregate))
        if not files:
            sys.exit(f"no raw files match {a.aggregate}")
        rdf = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        rdf = rdf.drop_duplicates(['size', 'seed']).sort_values(['size', 'seed'])
        rdf.to_csv(f"{a.out}_raw.csv", index=False)
        sdf = summarize(rdf, a.out)
        print(f"[lc] aggregated {len(files)} files, {len(rdf)} runs -> {a.out}_summary.csv")
        print(sdf[['size', 'n', 'mean_auc', 'ci_lo', 'ci_hi', 'std']].to_string(index=False))
        return

    emb_path = a.emb or f"{a.db}.{a.lm}.emb.csv"
    db_stem = os.path.splitext(os.path.basename(a.db))[0]
    df, emb = load(a.db, emb_path, a.target)
    y = df[a.target].astype(int).values
    print(f"[lc] db={db_stem} n={len(df):,} pos={y.mean():.1%} emb_dim={emb.shape[1]} "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    # ---- only-clusters mode ----
    if a.only_clusters:
        df = attach_clusters(df, a.cluster_col, None)
        df[['BARCODE', a.cluster_col]].rename(columns={a.cluster_col: 'cluster'}) \
            .to_csv(a.only_clusters, index=False)
        print(f"[lc] wrote clusters -> {a.only_clusters}  ({df[a.cluster_col].nunique():,} clusters)")
        return

    df = attach_clusters(df, a.cluster_col, a.cluster_file)

    sizes = sorted({len(df) if int(s) == 0 else min(int(s), len(df))
                    for s in a.sizes.split(',')})
    seeds = [a.base_seed + i for i in range(a.repeats)]
    grid = [(sz, sd) for sz in sizes for sd in seeds]
    if a.work_slice:
        I, N = map(int, a.work_slice.split('/'))
        grid = [g for k, g in enumerate(grid) if k % N == I]
        print(f"[lc] slice {I}/{N}: {len(grid)} of {len(sizes)*len(seeds)} runs", flush=True)

    raw, t0 = [], time.time()
    for size, seed in grid:
        r = one_run(df, emb, y, size, seed, a.cluster_col, a.target, a.lm, db_stem)
        raw.append(r)
        print(f"[lc] size={size:>7} seed={seed} auc={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(raw).to_csv(f"{a.out}_raw.csv", index=False)
    print(f"[lc] worker done: {len(raw)} runs in {time.time()-t0:.0f}s -> {a.out}_raw.csv", flush=True)


if __name__ == '__main__':
    main()
