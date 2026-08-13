# -*- coding: utf-8 -*-
"""Table-grounded recomputes I (main thread) own: R8 germline reconciliation,
R7 SEC mean-AUC definition, R4 PSR multiplicity facts + holdout AUC CIs.
Every number printed here is read from source files; nothing invented."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
PAPER = Path(__file__).resolve().parents[1]
D = PAPER / "data" / "local_only" / "raw"
PLMS = {"AbLang2","AntiBERTy","AntiBERTa2","AntiBERTa2-CSSP","IgBert"}

print("="*78); print("R8 | VH germline reconciliation (replicates fig1.py load_germline EXACTLY)")
print("="*78)
df = pd.read_excel(D / "ipi_psr_trainset.xlsx").dropna(subset=["psr_filter","vh_scaffold"])
df["vh_scaffold"] = df["vh_scaffold"].astype(str).replace({"VH3-23_A":"VH3-23","VH3-7_A":"VH3-7"})
overall = df["psr_filter"].mean()
g = df.groupby("vh_scaffold")["psr_filter"].agg(rate="mean", n="count").sort_values("rate", ascending=False)
print(f"overall PSR pass rate = {overall*100:.1f}%   (n={len(df)})")
print("\nALL vh_scaffold values (after VH3-23_A/VH3-7_A merge):")
print(g.assign(rate=lambda x:(x['rate']*100).round(1)).to_string())
g100 = g[g["n"]>=100]
print(f"\nGermlines with n>=100 (what Fig 1d plots): {list(g100.index)}")
print(g100.assign(rate=lambda x:(x['rate']*100).round(1)).to_string())
claimed5 = ["VH1-69","VH3-23","VH3-7","VH4-34","VH5-51"]
present = list(g100.index)
print(f"\nMethods [105] claims 5 VH germlines: {claimed5}")
print(f"  of these, present with n>=100: {[x for x in claimed5 if x in present]}")
print(f"  EXTRA germlines present (n>=100) NOT in the claimed 5: {[x for x in present if x not in claimed5]}")
print(f"  claimed-but-absent (n>=100): {[x for x in claimed5 if x not in present]}")
# also raw (unmerged) to see sub-alleles
raw = pd.read_excel(D / "ipi_psr_trainset.xlsx").dropna(subset=["psr_filter","vh_scaffold"])
rawc = raw["vh_scaffold"].astype(str).value_counts()
print(f"\nRAW vh_scaffold value_counts (unmerged, all):\n{rawc.to_string()}")
# VL germlines
vl = pd.read_excel(D / "ipi_psr_trainset.xlsx").dropna(subset=["vl_scaffold"])
print(f"\nVL scaffold value_counts:\n{vl['vl_scaffold'].astype(str).value_counts().to_string()}")

print("\n"+"="*78); print("R7 | SEC mean-AUC definition (ED_Table3_SEC.xlsx)")
print("="*78)
t3 = pd.read_excel(D / "ED_Table3_SEC.xlsx", header=1).dropna(how="all")
t3.columns = ["Architecture","LM","Acc","Prec","Rec","F1","AUC"]
t3["Architecture"] = t3["Architecture"].ffill()
t3["AUC"] = pd.to_numeric(t3["AUC"], errors="coerce")
t3["LMs"] = t3["LM"].astype(str).str.strip()
plm = t3[t3["LMs"].isin(PLMS)]
print(f"n rows total={len(t3)}, PLM rows={len(plm)}")
print(f"SEC mean AUC over the {len(plm)} PLM combos = {plm['AUC'].mean():.4f}  (sd {plm['AUC'].std():.4f}, range {plm['AUC'].min():.4f}-{plm['AUC'].max():.4f})")
print(f"SEC mean AUC over ALL {t3['AUC'].notna().sum()} combos = {t3['AUC'].mean():.4f}")
print("per-LM SEC AUC max (minor #11 check):")
for lm in ["AbLang2","AntiBERTy","AntiBERTa2","AntiBERTa2-CSSP","IgBert"]:
    sub=t3[t3["LMs"]==lm]["AUC"]
    print(f"  {lm:16s} min {sub.min():.4f}  max {sub.max():.4f}")
nonplm = t3[~t3["LMs"].isin(PLMS) & t3["AUC"].notna()]
print("non-PLM SEC rows:"); print(nonplm[["Architecture","LMs","AUC"]].to_string(index=False))

print("\n"+"="*78); print("R4 | PSR 25-combo facts (ED_Table1_PSR.xlsx)")
print("="*78)
t1 = pd.read_excel(D / "ED_Table1_PSR.xlsx", header=1).dropna(how="all")
t1.columns = ["Architecture","LM","Acc","Prec","Rec","F1","AUC"]
t1["Architecture"]=t1["Architecture"].ffill(); t1["AUC"]=pd.to_numeric(t1["AUC"],errors="coerce")
t1["LMs"]=t1["LM"].astype(str).str.strip()
plm1=t1[t1["LMs"].isin(PLMS)]
print(f"PLM combos={len(plm1)}: mean AUC {plm1['AUC'].mean():.4f} sd {plm1['AUC'].std():.4f} range {plm1['AUC'].min():.4f}-{plm1['AUC'].max():.4f}")
best=t1.loc[t1["AUC"].idxmax()]
print(f"best overall: {best['Architecture']}+{best['LMs']} AUC {best['AUC']:.4f}")
print(f"best-minus-PLMmean = {best['AUC']-plm1['AUC'].mean():.4f}  (in SD units: {(best['AUC']-plm1['AUC'].mean())/plm1['AUC'].std():.2f})")
weakest_plm = plm1['AUC'].min(); weak_rows=plm1[plm1['AUC']<=weakest_plm+0.004][["Architecture","LMs","AUC"]]
print(f"weakest PLM AUC = {weakest_plm:.4f}; PLM combos <=weakest+0.004:\n{weak_rows.to_string(index=False)}")
nonplm1=t1[~t1["LMs"].isin(PLMS) & t1["AUC"].notna()]
print("non-PLM (baseline/one-hot) rows:"); print(nonplm1[["Architecture","LMs","AUC"]].to_string(index=False))
print("--> baselines EXCEEDING the weakest PLM combos:")
for _,r in nonplm1.iterrows():
    above=plm1[plm1['AUC']<r['AUC']]
    if len(above): print(f"   {r['Architecture']}+{r['LMs']} AUC {r['AUC']:.3f} > {len(above)} PLM combos (e.g. {above['AUC'].min():.3f})")

print("\n"+"="*78); print("R4 supp | held-out 20% AUC + bootstrap 95% CI (validation CSV; fig5 bootstrap)")
print("="*78)
v = pd.read_csv(D / "IPI_PSR_TRAINSET_validation20pct_muliple_models_output.csv")
y = v["psr_filter"].astype(int).values
rng = np.random.default_rng(0)
def auc_ci(s):
    s=np.asarray(s); pt=roc_auc_score(y,s); vals=[]
    for _ in range(2000):
        idx=rng.integers(0,len(y),len(y))
        if y[idx].sum() in (0,len(idx)): continue
        vals.append(roc_auc_score(y[idx],s[idx]))
    return pt,np.percentile(vals,2.5),np.percentile(vals,97.5)
reps={"Transformer+AbLang2":"trans_ablang","CNN+AbLang2":"cnn_ablang","XGBoost+AbLang2":"xgb_ablang",
      "RF+AbLang2":"rf_ablang","Transformer+one-hot":"trans_one_hot"}
print(f"held-out validation n={len(y)}, pass={y.sum()}, fail={(y==0).sum()}")
for nm,c in reps.items():
    if c in v.columns:
        pt,lo,hi=auc_ci(v[c].values); print(f"  {nm:22s} AUC {pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    else: print(f"  {nm}: column {c} MISSING")
# paired bootstrap: PLM (trans_ablang) vs one-hot (trans_one_hot) on holdout
sa, so = v["trans_ablang"].values, v["trans_one_hot"].values
rng2=np.random.default_rng(0); diffs=[]
for _ in range(2000):
    idx=rng2.integers(0,len(y),len(y))
    if y[idx].sum() in (0,len(idx)): continue
    diffs.append(roc_auc_score(y[idx],sa[idx])-roc_auc_score(y[idx],so[idx]))
diffs=np.array(diffs)
print(f"paired AUC diff (Transformer+AbLang2 - Transformer+one-hot) on holdout: "
      f"{roc_auc_score(y,sa)-roc_auc_score(y,so):+.4f}  95% CI [{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]  "
      f"frac(diff>0)={np.mean(diffs>0):.3f}")
print("\nDONE R7_R8_R4")
