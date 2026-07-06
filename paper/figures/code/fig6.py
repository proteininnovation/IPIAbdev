"""
Figure 6 — model interpretability.
  a/b  mean |IG| per heavy-chain position (HCDR3 + VH framework)  PSR / SEC
  c/d  CDR H3 per-residue signed IG heatmap (Pass=blue, Fail=red)   PSR / SEC
  e/f  attribution convergence across RF-SHAP / XGBoost-SHAP / Transformer-IG,
       as % of attribution mass per region (HCDR3, VH, VL)          PSR / SEC
  g/h  XGBoost SHAP value vs CDR H3 net charge, coloured Pass/Fail   PSR / SEC
  i/j  XGBoost SHAP value vs CDR H3 tryptophan count, Pass/Fail      PSR / SEC

Rebuilt from the precomputed interpretability outputs (no model re-run, no values
invented):
  GENERATED_NBT_revision/analysis_runs/interp_{psr,sec}_*/ig_FULL_*.csv
  GENERATED_NBT_revision/analysis_runs/interp_{psr,sec}_*/region_attribution_*.csv
Note: the saved IG tables contain HCDR3 + VH per-residue attributions only (no
per-residue VL), so panels a-d are heavy-chain; the VL contribution appears in the
region-level convergence panels e/f where it was saved.
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
warnings.filterwarnings("ignore")

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
AR = f"{DELPHI}/GENERATED_NBT_revision/analysis_runs"
OUT = f"{DELPHI}/revision2_redteam/figures/output"
PSR_DIR = f"{AR}/interp_psr_filter_ipi_psr"
SEC_DIR = f"{AR}/interp_sec_filter_ipi_sec_5000"
PSR_SHAP = f"{PSR_DIR}/shap_xgb_FULL_psr_filter_biophysical_ipi_psr.csv"
SEC_SHAP = f"{SEC_DIR}/shap_xgb_FULL_sec_filter_biophysical_ipi_sec_5000.csv"
ok.set_style(base_pt=6.5)

AAORD = list("RKH" "DE" "WFY" "AGILMPV" "STNQC")
HEAT_CMAP = ok.DIVERGING.reversed()   # +IG (Pass) -> blue ; -IG (Fail) -> red
REGION_HCDR3, REGION_VH, REGION_VL = ok.REGION_HCDR3, ok.REGION_VH, ok.REGION_VL
# method colours match the architecture identity used in Figure 4
# (RF=green, XGBoost=purple, Transformer=orange) so colour means the same model across the paper
METHODS = [("RF", "RF-SHAP", ok.OI_GREEN), ("XGBoost", "XGBoost-SHAP", ok.OI_PURPLE),
           ("Transformer", "Transformer-IG", ok.OI_ORANGE)]
REGIONS = ["HCDR3", "VH", "VL"]


def load(dir_, tag):
    ig = pd.read_csv(f"{dir_}/ig_FULL_{tag}_onehot_{os.path.basename(dir_).replace('interp_'+tag+'_','')}.csv")
    ra = pd.read_csv(f"{dir_}/region_attribution_{tag}_{os.path.basename(dir_).replace('interp_'+tag+'_','')}.csv")
    return ig, ra


def positional(ig, min_n=10):
    cdr3 = [c for c in ig.columns if c.startswith("ig_CDR3_")]
    vh = [c for c in ig.columns if c.startswith("ig_VH_")]
    def _mean(cols):                                  # mean |IG| per position, blanking
        sub = ig[cols].abs()                          # positions seen in <min_n antibodies
        m = sub.mean().values                         # (matches the c/d >=10 count guard)
        m[sub.notna().sum().values < min_n] = np.nan
        return m
    return _mean(cdr3), _mean(vh)


def heat(ig, pmax=20):
    M = np.full((len(AAORD), pmax), np.nan)
    seqs = ig["hcdr3_seq"].astype(str).values
    for p in range(1, pmax + 1):
        col = f"ig_CDR3_{p:02d}"
        if col not in ig: continue
        aas = np.array([s[p - 1] if len(s) >= p else "" for s in seqs])
        vals = ig[col].values
        for ai, aa in enumerate(AAORD):
            m = aas == aa
            if m.sum() >= 10:
                M[ai, p - 1] = np.nanmean(vals[m])   # +IG => Pass-associated
    return M


def region_pct(ra):
    out = {mkey: {r: 0.0 for r in REGIONS} for mkey, _, _ in METHODS}
    for _, row in ra.iterrows():
        if row["method"] in out and row["region"] in REGIONS:
            out[row["method"]][row["region"]] = row["fraction_of_mass"] * 100
    return out


def dependence(ax, shap_csv, fcol, scol, xlabel, ylabel, title, vline=True, integer_x=False,
               title_fs=6.8, lbl_fs=6.0, xlbl_fs=6.3, tick_fs=None):
    """SHAP value for one biophysical feature vs the feature value, coloured Pass/Fail,
    with a per-unit binned mean. Generalised from the original charge-only panel."""
    d = pd.read_csv(shap_csv, usecols=["true_label", fcol, scol]).dropna()
    x = d[fcol].values; y = d[scol].values; lab = d["true_label"].values
    rng = np.random.default_rng(0)
    if len(x) > 4000:
        idx = rng.choice(len(x), 4000, replace=False); x, y, lab = x[idx], y[idx], lab[idx]
    for lv, c in [(0, ok.FAIL), (1, ok.PASS)]:
        m = lab == lv
        ax.scatter(x[m], y[m], s=2, color=c, alpha=0.4, lw=0, rasterized=True)
    bins = np.arange(np.floor(x.min()) - 0.5, np.ceil(x.max()) + 1.5, 1.0)
    bc = (bins[:-1] + bins[1:]) / 2
    means = [y[(x >= bins[i]) & (x < bins[i + 1])].mean()
             if ((x >= bins[i]) & (x < bins[i + 1])).sum() > 5 else np.nan for i in range(len(bins) - 1)]
    ax.plot(bc, means, "-", color="black", lw=1.1, zorder=5)
    ax.axhline(0, color="#888888", lw=0.5)
    if vline:
        ax.axvline(0, color="#888888", lw=0.5, ls=":")
    if integer_x:                                   # count features: integer ticks, drop n<=5 tail
        vc = pd.Series(np.round(x).astype(int)).value_counts()
        keep = sorted(k for k, v in vc.items() if v > 5)
        if keep:
            ax.set_xticks(keep); ax.set_xlim(min(keep) - 0.5, max(keep) + 0.5)
    if tick_fs:
        ax.tick_params(labelsize=tick_fs)
    ax.set_xlabel(xlabel, fontsize=xlbl_fs, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=lbl_fs, labelpad=2)
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=3)


psr_ig, psr_ra = load(PSR_DIR, "psr_filter")
sec_ig, sec_ra = load(SEC_DIR, "sec_filter")

fig = plt.figure(figsize=(ok.DOUBLE, 212 * ok.MM))
outer = GridSpec(4, 1, figure=fig, left=0.070, right=0.972, top=0.965, bottom=0.058,
                 hspace=0.58, height_ratios=[0.82, 1.16, 0.90, 0.94])
row_ab = outer[0].subgridspec(1, 2, wspace=0.28)
row_cd = outer[1].subgridspec(1, 2, wspace=0.42)
row_ef = outer[2].subgridspec(1, 2, wspace=0.28)
row_gj = outer[3].subgridspec(1, 4, wspace=0.42)   # g,h,i,j in one row (y-label on g only)

# ── a/b: positional |IG| profile (heavy chain) ────────────────────────────────
for col, (ig, name) in enumerate([(psr_ig, "PSR"), (sec_ig, "SEC")]):
    ax = fig.add_subplot(row_ab[0, col])
    c, v = positional(ig)
    xc = np.arange(len(c)); xv = np.arange(len(c), len(c) + len(v))
    ax.fill_between(xc, c, color=REGION_HCDR3, alpha=0.85, lw=0, label="CDR H3")
    ax.fill_between(xv, v, color=REGION_VH, alpha=0.75, lw=0, label="VH framework")
    ax.set_xlabel("Heavy-chain position", fontsize=6.3, labelpad=2)
    ax.set_ylabel("Mean |IG|", fontsize=6.3, labelpad=2)
    ax.set_xlim(0, len(c) + len(v)); ax.set_ylim(0, max(np.nanmax(c), np.nanmax(v)) * 1.12)
    ax.set_title(f"Mean |IG| per position · {name}", fontsize=6.8, fontweight="bold", pad=3)
    if col == 0:
        ax.legend(loc="upper right", fontsize=5.8, handlelength=1.0, handletextpad=0.4)
    ok.panel_label(fig, ax, "a" if col == 0 else "b", dx=-0.05, dy=0.026, size=8.5)

# ── c/d: CDR H3 per-residue signed IG heatmap ──────────────────────────────────
for col, (ig, name) in enumerate([(psr_ig, "PSR"), (sec_ig, "SEC")]):
    ax = fig.add_subplot(row_cd[0, col])
    M = heat(ig); vmax = np.nanmax(np.abs(M)) * 0.85
    im = ax.imshow(M, cmap=HEAT_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(AAORD))); ax.set_yticklabels(AAORD, fontsize=4.6)
    ax.set_xticks(range(0, 20, 4)); ax.set_xticklabels(range(1, 21, 4), fontsize=5.6)
    ax.set_xlabel("CDR H3 position", fontsize=6.3, labelpad=2)
    ax.set_ylabel("Amino acid", fontsize=6.3, labelpad=2)
    ax.set_title(f"CDR H3 per-residue signed IG · {name}", fontsize=6.8, fontweight="bold", pad=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=4.6, length=2)
    cb.set_label("← Fail        Pass →", fontsize=5.2)
    ok.panel_label(fig, ax, "c" if col == 0 else "d", dx=-0.05, dy=0.026, size=8.5)

# ── e/f: regional |IG| mass per method ────────────────────────────────────────
for col, (ra, name) in enumerate([(psr_ra, "PSR"), (sec_ra, "SEC")]):
    ax = fig.add_subplot(row_ef[0, col])
    pct = region_pct(ra)
    x = np.arange(len(REGIONS)); w = 0.26
    for k, (mkey, mlab, mcol) in enumerate(METHODS):
        vals = [pct[mkey][r] for r in REGIONS]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=mcol, label=mlab)
        for xi, vv in zip(x + (k - 1) * w, vals):
            if vv > 1:
                ax.text(xi, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=4.8)
    ax.set_xticks(x); ax.set_xticklabels([r.replace("HCDR3", "CDR H3") for r in REGIONS], fontsize=6.3)
    if col == 0:                                     # shared y-label: leftmost panel only
        ax.set_ylabel("% of total |IG| mass", fontsize=6.3, labelpad=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"% |IG| mass per region · {name}", fontsize=6.8, fontweight="bold", pad=3)
    if col == 0:
        ax.legend(loc="upper right", fontsize=5.6, handlelength=1.0, handletextpad=0.4, labelspacing=0.25)
    ok.panel_label(fig, ax, "e" if col == 0 else "f", dx=-0.05, dy=0.026, size=8.5)

# ── g-j: SHAP value vs CDR H3 net charge (g,h) and tryptophan count (i,j) ──────
GJ = [("fval_cdr3_charge", "shap_cdr3_charge", "CDR H3 net charge", "Net charge", True,  False),
      ("fval_cdr3_W",       "shap_cdr3_W",      "CDR H3 Trp count",  "Trp count",  False, True)]
letters = ["g", "h", "i", "j"]
k = 0
for fcol, scol, xlab, tlab, vline, integer_x in GJ:
    for name, shap_csv in [("PSR", PSR_SHAP), ("SEC", SEC_SHAP)]:
        ax = fig.add_subplot(row_gj[0, k])
        dependence(ax, shap_csv, fcol, scol, xlab, "SHAP → P(Pass)" if k == 0 else "",
                   f"{tlab} · {name}", vline=vline, integer_x=integer_x,
                   title_fs=5.9, lbl_fs=5.7, xlbl_fs=5.9, tick_fs=5.2)
        if k == 0:
            ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=ok.PASS, ms=3.2, label="Pass"),
                               Line2D([0], [0], marker="o", color="w", markerfacecolor=ok.FAIL, ms=3.2, label="Fail"),
                               Line2D([0], [0], color="black", lw=1.0, label="binned mean")],
                      loc="upper right", fontsize=4.5, handlelength=1.0, handletextpad=0.3,
                      borderpad=0.3, labelspacing=0.25)
        ok.panel_label(fig, ax, letters[k], dx=-0.07, dy=0.022, size=8.5)
        k += 1

ok.save_fig(fig, "Figure6", OUT)
print("Fig6 done")
