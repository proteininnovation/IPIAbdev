"""
Recompute developability-model metrics for DELPHI revision-2 red-team fixes
R2 (dual-liability), R3/R5 (external cohorts), T2 (external-grid max AUC).

ACCURACY IS PARAMOUNT: every number printed is computed here from the data files.
Bootstrap matches figures_nature_v2/code/fig5.py EXACTLY:
  rng = np.random.default_rng(0); 2000 resamples; paired idx = rng.integers(0,n,n);
  skip a resample if resampled y is all one class; CI = 2.5 / 97.5 percentiles.

DELPHI deployed score column (all external tables):
  transformer_lm_ablang_ipi_psr_trainset_score  (Transformer + AbLang2)
  higher score => more PASS => LESS polyreactive.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact
from sklearn.metrics import roc_auc_score, average_precision_score

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
DATA = f"{DELPHI}/data"

DELPHI_SC = "transformer_lm_ablang_ipi_psr_trainset_score"
THRESH = 0.27
THRESH_ELISA = 1.9
N_BOOT = 2000

# ---------------------------------------------------------------- bootstrap (verbatim logic from fig5.py)
def boot_ci(stat_fn, n, n_boot=N_BOOT, rng=None):
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def auc_ci(y, s, rng):
    y = np.asarray(y); s = np.asarray(s)
    pt = roc_auc_score(y, s)
    def fn(idx):
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == len(yi):
            return None
        return roc_auc_score(yi, s[idx])
    lo, hi = boot_ci(fn, len(y), rng=rng)
    return pt, lo, hi

def ap_ci(yfail, sfail, rng):
    """Average precision (PR-AUC) for the rare FAIL class, with bootstrap CI."""
    yfail = np.asarray(yfail); sfail = np.asarray(sfail)
    pt = average_precision_score(yfail, sfail)
    def fn(idx):
        yi = yfail[idx]
        if yi.sum() == 0 or yi.sum() == len(yi):
            return None
        return average_precision_score(yi, sfail[idx])
    lo, hi = boot_ci(fn, len(yfail), rng=rng)
    return pt, lo, hi

def rho_signed_ci(s, a, rng):
    s = np.asarray(s); a = np.asarray(a)
    pt = spearmanr(s, a).correlation
    def fn(idx):
        c = spearmanr(s[idx], a[idx]).correlation
        return c if np.isfinite(c) else None
    lo, hi = boot_ci(fn, len(s), rng=rng)
    return pt, lo, hi

# ================================================================ PART A: external cohorts
print("=" * 90)
print("PART A: EXTERNAL COHORTS  (DELPHI = transformer_lm_ablang_ipi_psr_trainset_score)")
print("=" * 90)

jain = pd.read_excel(f"{DATA}/Jain2017_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")
g1   = pd.read_excel(f"{DATA}/GDPa1_v1.3_20251027_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")
g3   = pd.read_excel(f"{DATA}/GDPa3_20260106_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")

# Each cohort spec: (name, dataframe, assay_continuous_col, binary_pass_col_or_None)
# y_pass: 1 = PASS (less polyreactive). If binary col given use it; else (assay < THRESH).
COHORTS = [
    ("Jain 2017 PSR-SMP", jain, "PSR_SMP_Score", "psr_filter"),
    ("GDPa1 PR-CHO",      g1,   "polyreactivity_prscore_cho_avg", None),
    ("GDPa3 PR-CHO",      g3,   "polyreactivity_prscore_cho_avg", None),
    ("GDPa3 PR-Ova",      g3,   "polyreactivity_prscore_ova_avg", None),
]

rows = []
for name, df, assay_col, bin_col in COHORTS:
    cols = [DELPHI_SC, assay_col] + ([bin_col] if bin_col else [])
    sub = df[cols].dropna()
    s = sub[DELPHI_SC].values
    a = sub[assay_col].values
    if bin_col:
        y = sub[bin_col].astype(int).values          # 1 = Pass
    else:
        y = (a < THRESH).astype(int)                  # 1 = Pass
    n = len(y); npass = int(y.sum()); nfail = int((y == 0).sum())

    # fresh seed-0 rng per cohort so each metric stream is reproducible & independent of order
    rng = np.random.default_rng(0)
    auc, auc_lo, auc_hi = auc_ci(y, s, rng)

    # PR-AUC for RARE FAIL class: Fail=1, score_fail = 1 - DELPHI_SC (higher => more likely Fail)
    yfail = 1 - y
    sfail = 1.0 - s
    rng = np.random.default_rng(0)
    prauc, pr_lo, pr_hi = ap_ci(yfail, sfail, rng)
    noskill = nfail / n                               # prevalence of Fail = no-skill PR baseline

    rng = np.random.default_rng(0)
    rho, rho_lo, rho_hi = rho_signed_ci(s, a, rng)

    rows.append(dict(cohort=name, n=n, n_pass=npass, n_fail=nfail,
                     auc=auc, auc_lo=auc_lo, auc_hi=auc_hi,
                     prauc=prauc, pr_lo=pr_lo, pr_hi=pr_hi, noskill=noskill,
                     rho=rho, rho_lo=rho_lo, rho_hi=rho_hi, absrho=abs(rho)))

# tidy print
print(f"\n{'Cohort':<20}{'n':>4}{'Pass':>5}{'Fail':>5}   "
      f"{'ROC-AUC [95% CI]':<26}{'PR-AUC(Fail) [95% CI]':<28}{'no-skill':>9}   "
      f"{'signed rho [95% CI]':<26}{'|rho|':>6}")
for r in rows:
    print(f"{r['cohort']:<20}{r['n']:>4}{r['n_pass']:>5}{r['n_fail']:>5}   "
          f"{r['auc']:.3f} [{r['auc_lo']:.3f}-{r['auc_hi']:.3f}]   "
          f"{r['prauc']:.3f} [{r['pr_lo']:.3f}-{r['pr_hi']:.3f}]    "
          f"{r['noskill']:.3f}   "
          f"{r['rho']:+.3f} [{r['rho_lo']:+.3f},{r['rho_hi']:+.3f}]  "
          f"{r['absrho']:.3f}")

print("\nSanity checks (expected vs computed):")
exp_auc = {"Jain 2017 PSR-SMP": 0.73, "GDPa1 PR-CHO": 0.68, "GDPa3 PR-CHO": 0.75}
for r in rows:
    if r['cohort'] in exp_auc:
        e = exp_auc[r['cohort']]
        ok = abs(r['auc'] - e) < 0.02
        print(f"  AUC {r['cohort']:<20} expected ~{e:.2f}  computed {r['auc']:.3f}  "
              f"{'MATCH' if ok else '*** MISMATCH ***'}")
g3cho = next(r for r in rows if r['cohort'] == "GDPa3 PR-CHO")
g3ova = next(r for r in rows if r['cohort'] == "GDPa3 PR-Ova")
print(f"  GDPa3 PR-CHO Fails expected ~12  computed {g3cho['n_fail']}  "
      f"{'MATCH' if g3cho['n_fail'] == 12 else 'CHECK'}")
print(f"  GDPa3 PR-Ova Fails expected ~3   computed {g3ova['n_fail']}  "
      f"{'MATCH' if g3ova['n_fail'] == 3 else 'CHECK'}")

# ================================================================ PART B: external-grid max AUC (fix T2)
print("\n" + "=" * 90)
print("PART B: EXTERNAL-GRID MAXIMUM AUC  (reproducing ed8.py grid; claim 'up to 0.805')")
print("=" * 90)

HM_LM_ORDER   = ['ablang', 'antiberty', 'antiberta2', 'antiberta2-cssp', 'igbert', 'onehot']
HM_LM_DISPLAY = ['AbLang2', 'AntiBERTy', 'AntiBERTa2', 'AntiBERTa2-CSSP', 'IgBert', 'One-hot']
DATASETS = ['Jain PSR-SMP', 'Jain ELISA', 'GDPa1 PR-Ova', 'GDPa1 PR-CHO', 'GDPa3 PR-Ova', 'GDPa3 PR-CHO']

def score_col(lm):
    return 'transformer_onehot_onehot_ipi_psr_trainset_score' if lm == 'onehot' \
        else f'transformer_lm_{lm}_ipi_psr_trainset_score'

def safe_auc(df, sc, col, min_fail=5, thresh=None):
    sub = df[[sc, col]].dropna(); t = thresh if thresh is not None else THRESH
    y = (sub[col] < t).astype(int)
    if (y == 0).sum() < min_fail or (y == 1).sum() < min_fail:
        return np.nan
    return roc_auc_score(y, sub[sc])

def heatmap_auc(jain, g1, g3):
    n = len(HM_LM_ORDER)
    auc = np.full((n, 6), np.nan)
    for i, lm in enumerate(HM_LM_ORDER):
        c = score_col(lm)
        if c in jain.columns:
            auc[i, 0] = safe_auc(jain, c, 'PSR_SMP_Score')
            auc[i, 1] = safe_auc(jain, c, 'ELISA', thresh=THRESH_ELISA)
        if c in g1.columns:
            auc[i, 2] = safe_auc(g1, c, 'polyreactivity_prscore_ova_avg')
            auc[i, 3] = safe_auc(g1, c, 'polyreactivity_prscore_cho_avg')
        if c in g3.columns:
            auc[i, 4] = safe_auc(g3, c, 'polyreactivity_prscore_ova_avg', min_fail=3)
            auc[i, 5] = safe_auc(g3, c, 'polyreactivity_prscore_cho_avg')
    return auc

auc_mat = heatmap_auc(jain, g1, g3)
print("\nFull AUC grid (rows = language model, cols = assay readout):")
hdr = "                 " + "".join(f"{d:>15}" for d in DATASETS)
print(hdr)
for i, lm in enumerate(HM_LM_DISPLAY):
    cells = "".join((f"{auc_mat[i,j]:>15.3f}" if np.isfinite(auc_mat[i,j]) else f"{'N/A':>15}")
                    for j in range(6))
    print(f"{lm:<17}{cells}")

imax = np.nanargmax(auc_mat)
ri, ci = np.unravel_index(imax, auc_mat.shape)
maxval = auc_mat[ri, ci]
print(f"\nMAX AUC cell = {maxval:.4f}")
print(f"  language model = {HM_LM_DISPLAY[ri]}  ({HM_LM_ORDER[ri]})")
print(f"  assay readout  = {DATASETS[ci]}")
is_deployed = (HM_LM_ORDER[ri] == 'ablang')
print(f"  is this the DEPLOYED AbLang2 model? {is_deployed}")
print(f"  equals ~0.805? {'YES' if abs(maxval - 0.805) < 0.01 else 'NO (computed %.4f)' % maxval}")

# ================================================================ PART C: dual-liability cross-assay (fix R2)
print("\n" + "=" * 90)
print("PART C: DUAL-LIABILITY CROSS-ASSAY TEST  (PSR vs SEC, HCDR3 charge)")
print("=" * 90)

sec = pd.read_excel(f"{DATA}/ipi_sec_5000.xlsx", sheet_name="Sheet1")
psrtrain = pd.read_excel(f"{DATA}/ipi_psr_trainset.xlsx", sheet_name="Sheet1")

# --- VERIFY sec_filter polarity vs retention_time / peak_area (1 should = Pass = better SEC)
# peak_area_pct & retention_time_mins are object cols: most rows are single numbers, but a few
# are multi-peak strings ("58.91;  17.05;  16.09"). Coerce to numeric (drops the multi-peak
# strings) for the polarity sanity check only; this does NOT touch C1/C2/C3.
print("\n[Polarity check] sec_filter vs SEC continuous readouts:")
secv = sec[["sec_filter", "retention_time_mins", "peak_area_pct"]].copy()
secv["peak_area_pct"] = pd.to_numeric(secv["peak_area_pct"], errors="coerce")
secv["retention_time_mins"] = pd.to_numeric(secv["retention_time_mins"], errors="coerce")
for val in [0, 1]:
    grp = secv[secv["sec_filter"] == val]
    pk = grp["peak_area_pct"].dropna(); rt = grp["retention_time_mins"].dropna()
    print(f"  sec_filter=={val}: "
          f"peak_area_pct mean={pk.mean():.2f} median={pk.median():.2f} (n={len(pk)}) | "
          f"retention_time_mins mean={rt.mean():.3f} median={rt.median():.3f} (n={len(rt)})")
print("  (Higher peak_area_pct = more monomer = better SEC. If sec_filter==1 has higher peak_area_pct, polarity 1=Pass is correct.)")

# Same quick polarity sanity for psr_filter vs psr_norm readouts
print("\n[Polarity check] psr_filter vs psr_norm readouts (higher norm = more polyreactive = worse):")
psrv = sec[["psr_filter", "psr_norm_smp", "psr_norm_dna", "psr_norm_avidin", "psr_norm_insulin"]].dropna(subset=["psr_filter"])
for val in [0, 1]:
    grp = psrv[psrv["psr_filter"] == val]
    print(f"  psr_filter=={val} (n={len(grp)}): psr_norm_smp mean={grp['psr_norm_smp'].mean():.3f} "
          f"median={grp['psr_norm_smp'].median():.3f}")

# --- C1: both-measured set + 2x2 contingency PSR-fail x SEC-fail
both = sec[["BARCODE", "psr_filter", "sec_filter", "HCDR3_charge"]].copy()
both = both.dropna(subset=["psr_filter", "sec_filter"])
n_both = len(both)
psr_fail = (both["psr_filter"] == 0).astype(int)   # 1 = PSR fail
sec_fail = (both["sec_filter"] == 0).astype(int)   # 1 = SEC fail

# 2x2: rows = PSR (fail=1,pass=0), cols = SEC (fail=1,pass=0)
a11 = int(((psr_fail == 1) & (sec_fail == 1)).sum())  # PSR fail & SEC fail
a10 = int(((psr_fail == 1) & (sec_fail == 0)).sum())  # PSR fail & SEC pass
a01 = int(((psr_fail == 0) & (sec_fail == 1)).sum())  # PSR pass & SEC fail
a00 = int(((psr_fail == 0) & (sec_fail == 0)).sum())  # PSR pass & SEC pass

print(f"\n[C1] Molecules with BOTH psr_filter and sec_filter measured: n = {n_both}")
print(f"     PSR-fail total = {int(psr_fail.sum())}   SEC-fail total = {int(sec_fail.sum())}")
print("     2x2 contingency (PSR x SEC):")
print(f"                       SEC-fail   SEC-pass")
print(f"        PSR-fail   {a11:>8}   {a10:>8}")
print(f"        PSR-pass   {a01:>8}   {a00:>8}")

table = np.array([[a11, a10], [a01, a00]])
# phi coefficient from the 2x2
n = a11 + a10 + a01 + a00
r1, r2 = a11 + a10, a01 + a00
c1, c2 = a11 + a01, a10 + a00
denom = np.sqrt(r1 * r2 * c1 * c2)
phi = (a11 * a00 - a10 * a01) / denom if denom > 0 else np.nan
# odds ratio + Fisher exact
OR_fisher, p_fisher = fisher_exact(table, alternative="two-sided")
# Haldane-corrected OR if any zero cell (report both)
if min(a11, a10, a01, a00) == 0:
    c = [a11 + 0.5, a10 + 0.5, a01 + 0.5, a00 + 0.5]
    OR_hald = (c[0] * c[3]) / (c[1] * c[2])
else:
    OR_hald = (a11 * a00) / (a10 * a01)

p_sec_given_psrfail = a11 / r1 if r1 > 0 else np.nan
p_sec_given_psrpass = a01 / r2 if r2 > 0 else np.nan

print(f"\n     phi coefficient            = {phi:.4f}")
print(f"     odds ratio (Fisher)        = {OR_fisher:.4f}")
print(f"     odds ratio (direct/Haldane)= {OR_hald:.4f}")
print(f"     Fisher exact p (two-sided) = {p_fisher:.4g}")
print(f"     P(SEC fail | PSR fail)     = {p_sec_given_psrfail:.4f}  ({a11}/{r1})")
print(f"     P(SEC fail | PSR pass)     = {p_sec_given_psrpass:.4f}  ({a01}/{r2})")

# --- C2: HCDR3 net charge predicting BOTH failure modes (Fail=1)
both_ch = both.dropna(subset=["HCDR3_charge"])
psr_fail_ch = (both_ch["psr_filter"] == 0).astype(int).values
sec_fail_ch = (both_ch["sec_filter"] == 0).astype(int).values
chg = both_ch["HCDR3_charge"].values
auc_psr = roc_auc_score(psr_fail_ch, chg)   # higher charge -> predict PSR fail?
auc_sec = roc_auc_score(sec_fail_ch, chg)   # higher charge -> predict SEC fail?
rho_psr = spearmanr(chg, psr_fail_ch).correlation
rho_sec = spearmanr(chg, sec_fail_ch).correlation
print(f"\n[C2] HCDR3_charge predicting failure (Fail=1, n={len(chg)} with charge):")
print(f"     ROC-AUC(PSR_fail, HCDR3_charge) = {auc_psr:.4f}   Spearman rho = {rho_psr:+.4f}")
print(f"     ROC-AUC(SEC_fail, HCDR3_charge) = {auc_sec:.4f}   Spearman rho = {rho_sec:+.4f}")
print("     (AUC>0.5 => higher charge predicts FAIL; AUC<0.5 => higher charge predicts PASS.)")

# --- C3: BARCODE overlap between psr trainset and sec_5000
bc_train = set(psrtrain["BARCODE"].dropna().astype(str))
bc_sec   = set(sec["BARCODE"].dropna().astype(str))
overlap = bc_train & bc_sec
print(f"\n[C3] BARCODE overlap:")
print(f"     ipi_psr_trainset unique BARCODEs = {len(bc_train)}")
print(f"     ipi_sec_5000     unique BARCODEs = {len(bc_sec)}")
print(f"     shared BARCODEs                  = {len(overlap)}")

# --- C4: verdict
print("\n[C4] DUAL-LIABILITY VERDICT:")
print(f"     n both-measured = {n_both}; PSR-fail&SEC-fail co-occurrence = {a11}")
print(f"     association: phi={phi:.3f}, OR(Fisher)={OR_fisher:.2f}, Fisher p={p_fisher:.3g}")
print(f"     conditional lift: P(SEC fail|PSR fail)={p_sec_given_psrfail:.3f} vs "
      f"P(SEC fail|PSR pass)={p_sec_given_psrpass:.3f}")
print(f"     charge drives both? PSR-fail AUC={auc_psr:.3f}, SEC-fail AUC={auc_sec:.3f}")
print("     (Verdict text written in the returned report, grounded in the numbers above.)")

print("\nDONE.")
