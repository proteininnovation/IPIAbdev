"""
Figure 6 — model interpretability.
  a/b  mean |IG| per heavy-chain position (HCDR3 + VH framework)  PSR / SEC
  c/d  CDR H3 per-residue signed IG heatmap (Pass=blue, Fail=red)   PSR / SEC
  e/f  attribution convergence across RF-SHAP / XGBoost-SHAP / Transformer-IG,
       as % of attribution mass per region (HCDR3, VH, VL)          PSR / SEC

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


def positional(ig):
    cdr3 = [c for c in ig.columns if c.startswith("ig_CDR3_")]
    vh = [c for c in ig.columns if c.startswith("ig_VH_")]
    c = np.nan_to_num(ig[cdr3].abs().mean().values)   # empty HCDR3 positions -> 0
    v = np.nan_to_num(ig[vh].abs().mean().values)
    return c, v


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


def charge_dependence(ax, shap_csv, name):
    d = pd.read_csv(shap_csv, usecols=["true_label", "fval_cdr3_charge", "shap_cdr3_charge"]).dropna()
    x = d["fval_cdr3_charge"].values; y = d["shap_cdr3_charge"].values; lab = d["true_label"].values
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
    ax.axhline(0, color="#888888", lw=0.5); ax.axvline(0, color="#888888", lw=0.5, ls=":")
    ax.set_xlabel("CDR H3 net charge", fontsize=6.3, labelpad=2)
    ax.set_ylabel("SHAP (charge) → P(Pass)", fontsize=6.0, labelpad=2)
    ax.set_title(f"Charge drives the model · {name}", fontsize=6.8, fontweight="bold", pad=3)


psr_ig, psr_ra = load(PSR_DIR, "psr_filter")
sec_ig, sec_ra = load(SEC_DIR, "sec_filter")

fig = plt.figure(figsize=(ok.DOUBLE, 215 * ok.MM))
gs = GridSpec(4, 2, figure=fig, left=0.085, right=0.93, top=0.965, bottom=0.05,
              hspace=0.55, wspace=0.30, height_ratios=[0.82, 1.12, 0.88, 0.9])

# ── a/b: positional |IG| profile (heavy chain) ────────────────────────────────
for col, (ig, name) in enumerate([(psr_ig, "PSR"), (sec_ig, "SEC")]):
    ax = fig.add_subplot(gs[0, col])
    c, v = positional(ig)
    xc = np.arange(len(c)); xv = np.arange(len(c), len(c) + len(v))
    ax.fill_between(xc, c, color=REGION_HCDR3, alpha=0.85, lw=0, label="CDR H3")
    ax.fill_between(xv, v, color=REGION_VH, alpha=0.75, lw=0, label="VH framework")
    ax.set_xlabel("Heavy-chain position", fontsize=6.3, labelpad=2)
    ax.set_ylabel("Mean |IG|", fontsize=6.3, labelpad=2)
    ax.set_xlim(0, len(c) + len(v)); ax.set_ylim(0, max(c.max(), v.max()) * 1.12)
    ax.set_title(f"Transformer-IG · {name}", fontsize=6.8, fontweight="bold", pad=3)
    if col == 0:
        ax.legend(loc="upper right", fontsize=5.8, handlelength=1.0, handletextpad=0.4)
    ok.panel_label(fig, ax, "a" if col == 0 else "b", dx=-0.05, dy=0.026, size=8.5)

# ── c/d: CDR H3 per-residue signed IG heatmap ──────────────────────────────────
for col, (ig, name) in enumerate([(psr_ig, "PSR"), (sec_ig, "SEC")]):
    ax = fig.add_subplot(gs[1, col])
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

# ── e/f: attribution convergence per region ───────────────────────────────────
for col, (ra, name) in enumerate([(psr_ra, "PSR"), (sec_ra, "SEC")]):
    ax = fig.add_subplot(gs[2, col])
    pct = region_pct(ra)
    x = np.arange(len(REGIONS)); w = 0.26
    for k, (mkey, mlab, mcol) in enumerate(METHODS):
        vals = [pct[mkey][r] for r in REGIONS]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=mcol, label=mlab)
        for xi, vv in zip(x + (k - 1) * w, vals):
            if vv > 1:
                ax.text(xi, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=4.8)
    ax.set_xticks(x); ax.set_xticklabels([r.replace("HCDR3", "CDR H3") for r in REGIONS], fontsize=6.3)
    ax.set_ylabel("% of attribution mass", fontsize=6.3, labelpad=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"Attribution convergence · {name}", fontsize=6.8, fontweight="bold", pad=3)
    if col == 0:
        ax.legend(loc="upper right", fontsize=5.6, handlelength=1.0, handletextpad=0.4, labelspacing=0.25)
    ok.panel_label(fig, ax, "e" if col == 0 else "f", dx=-0.05, dy=0.026, size=8.5)

# ── g/h: charge -> SHAP dependence (the learned electrostatic rule) ────────────
for col, (shap_csv, name) in enumerate([(PSR_SHAP, "PSR"), (SEC_SHAP, "SEC")]):
    ax = fig.add_subplot(gs[3, col])
    charge_dependence(ax, shap_csv, name)
    if col == 0:
        ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=ok.PASS, ms=4, label="Pass"),
                           Line2D([0], [0], marker="o", color="w", markerfacecolor=ok.FAIL, ms=4, label="Fail"),
                           Line2D([0], [0], color="black", lw=1.1, label="binned mean")],
                  loc="upper right", fontsize=5.6, handlelength=1.2, handletextpad=0.4)
    ok.panel_label(fig, ax, "g" if col == 0 else "h", dx=-0.05, dy=0.022, size=8.5)

ok.save_fig(fig, "Figure6", OUT)
print("Fig6 done")
