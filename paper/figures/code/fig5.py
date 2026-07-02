"""
Figure 5 — Conserved CDR H3 electrostatic signature.

TOP block (3 rows x 5 cols): Pass/Fail CDR H3 feature distributions, faithful to
v1 fig2 / utils/Figure2_physicochemical.py, restyled Okabe-Ito.
  rows: IPI PSR (ELISA+NGS) | DS1 (public) | IPI SEC
  cols: Arg count | Asp count | Trp count (Arg=1) | CDR H3 length | net charge

BOTTOM strip (3 new panels):
  p) Cohen's d effect-size heatmap  — 5 features x 3 datasets, standardized
     (Fail - Pass) mean difference; shows the signature is direction-conserved.
  q) Net charge -> P(Pass) curve     — integer charge bins, fraction Pass per bin
     with Wilson 95% CIs, IPI PSR (blue) + DS1 (orange) overlaid; monotone decline.
  r) Biophysical determinant correlations — Spearman matrix among PSR sub-assays,
     SPR affinity, SEC retention, CDR H3/VH charge & hydrophobicity, titer.

Data:
  IPI PSR  data/ipi_psr_trainset.xlsx          (CDR3, psr_filter)
  IPI SEC  data/ipi_sec_5000.xlsx              (CDR3, sec_filter)
  DS1      /tmp/delphi_ds1/ds1_clean.parquet   (VH, psr_filter) — CDR H3 re-derived
           from VH via the C...WGxG motif, fixed 60k sample (seed 42).
CDR H3 net charge uses liabilities.charge_value (ProteinAnalysis.charge_at_pH 7.4),
the exact function the original figure used.
"""
import sys, os, re, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/Andre.Teixeira/temp/delphi/utils")   # liabilities
import okabe_style as ok
import liabilities
warnings.filterwarnings("ignore")

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
DATA = f"{DELPHI}/data"; OUT = f"{DELPHI}/revision2_redteam/figures/output"
DS1_PARQUET = "/tmp/delphi_ds1/ds1_clean.parquet"
ok.set_style(base_pt=6.5)
PASS, FAIL = ok.PASS, ok.FAIL
ALPHA = 0.72
HCDR3_RE = re.compile(r"C([A-Z]{3,35}?)WG[QKRPL]G")


def add_features(df, cdr3_col="CDR3"):
    df = df.copy()
    s = df[cdr3_col].astype(str)
    df["R"] = s.str.count("R"); df["D"] = s.str.count("D")
    df["W"] = s.str.count("W"); df["CDR3_len"] = s.str.len()
    df["charge"] = s.map(liabilities.charge_value)
    return df[df["CDR3_len"] < 25]


def load_psr():
    d = pd.read_excel(f"{DATA}/ipi_psr_trainset.xlsx").dropna(subset=["psr_filter"])
    return add_features(d), "psr_filter"


def load_sec():
    d = pd.read_excel(f"{DATA}/ipi_sec_5000.xlsx").dropna(subset=["sec_filter"])
    return add_features(d), "sec_filter"


def load_ds1(n=60000, seed=42):
    d = pd.read_parquet(DS1_PARQUET).dropna(subset=["psr_filter"])
    if len(d) > n:
        d = d.sample(n, random_state=seed)          # density-faithful subsample
    m = d["VH"].astype(str).map(lambda v: (HCDR3_RE.findall(v) or [None])[-1])
    d = d.assign(CDR3=m).dropna(subset=["CDR3"])
    return add_features(d), "psr_filter"


# (column, x-label, restrict to Arg==1, short feature name for effect-size heatmap)
PANELS = [
    ("R",        "Arginine count (CDR H3)",            False, "Arg count"),
    ("D",        "Aspartic acid count (CDR H3)",       False, "Asp count"),
    ("W",        "Tryptophan count\n(Arg count = 1)", True,  "Trp count\n(Arg=1)"),
    ("CDR3_len", "CDR H3 loop length",                 False, "CDR H3 length"),
    ("charge",   "Net charge (CDR H3)",                False, "Net charge"),
]
LETTERS = [list("abcde"), list("fghij"), list("klmno")]
ROW_LABELS = ["IPI PSR\n(ELISA+NGS)", "DS1\n(public)", "IPI SEC"]
DSET_NAMES = ["IPI PSR", "DS1", "IPI SEC"]


def cohens_d(fail, pas):
    """Standardized (Fail - Pass) mean difference, pooled SD."""
    fail = np.asarray(fail, float); pas = np.asarray(pas, float)
    nf, npp = len(fail), len(pas)
    if nf < 2 or npp < 2:
        return np.nan
    sf, sp = fail.std(ddof=1), pas.std(ddof=1)
    pooled = np.sqrt(((nf - 1) * sf ** 2 + (npp - 1) * sp ** 2) / (nf + npp - 2))
    if pooled == 0:
        return np.nan
    return (fail.mean() - pas.mean()) / pooled


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (centre - half, centre + half)


# ── Load ──────────────────────────────────────────────────────────────────────
print("loading PSR..."); psr, fp = load_psr()
print("loading DS1..."); ds1, fd = load_ds1()
print("loading SEC..."); sec, fs = load_sec()
rows = [(psr, fp), (ds1, fd), (sec, fs)]

# ── Effect sizes: 5 features x 3 datasets (Fail - Pass) Cohen's d ─────────────
d_mat = np.full((5, 3), np.nan)
for j, (df, fcol) in enumerate(rows):
    for i, (var, _xl, sub, _sn) in enumerate(PANELS):
        d = df[df["R"] == 1] if sub else df
        d_mat[i, j] = cohens_d(d.loc[d[fcol] == 0, var], d.loc[d[fcol] == 1, var])
print("Cohen's d (rows=features, cols=%s):" % DSET_NAMES)
for i, (_v, _xl, _s, sn) in enumerate(PANELS):
    print("  %-18s %s" % (sn.replace("\n", " "),
                          "  ".join(f"{d_mat[i, j]:+.2f}" for j in range(3))))

# ── Charge -> P(Pass) curves (IPI PSR + DS1), integer charge bins, Wilson CI ──
def passrate_curve(df, fcol):
    g = df.assign(cbin=np.rint(df["charge"]).astype(int)).groupby("cbin")
    out = []
    for cb, sub in g:
        n = len(sub); k = int((sub[fcol] == 1).sum())
        if n < 20:                              # drop bins too sparse to plot
            continue
        lo, hi = wilson_ci(k, n)
        out.append((cb, k / n, lo, hi, n))
    return pd.DataFrame(out, columns=["charge", "ppass", "lo", "hi", "n"]).sort_values("charge")

psr_curve = passrate_curve(psr, fp)
ds1_curve = passrate_curve(ds1, fd)
print("charge->P(Pass) IPI PSR:",
      "  ".join(f"{r.charge:+d}:{r.ppass:.2f}(n={r.n})" for r in psr_curve.itertuples()))

# ── Biophysical determinant correlations (SEC file), Spearman ────────────────
CORR_SPEC = [
    ("psr_norm_dna",         "PSR DNA"),
    ("psr_norm_avidin",      "PSR avidin"),
    ("psr_norm_insulin",     "PSR insulin"),
    ("psr_norm_smp",         "PSR SMP"),
    ("kd_m",                 "SPR KD"),
    ("retention_time_mins",  "SEC RT"),
    ("HCDR3_charge",         "CDR H3 charge"),
    ("HCDR3_hydrophobicity", "CDR H3 GRAVY"),
    ("VH_charge",            "VH charge"),
    ("puriftitermgl",        "Titer"),
]
sec_raw = pd.read_excel(f"{DATA}/ipi_sec_5000.xlsx")
corr_cols, corr_labels = [], []
for col, lab in CORR_SPEC:
    if col not in sec_raw.columns:
        print(f"  skip correlation var (missing column): {col}")
        continue
    corr_cols.append(col); corr_labels.append(lab)
corr_df = sec_raw[corr_cols].apply(pd.to_numeric, errors="coerce")  # RT is object dtype
corr = corr_df.corr(method="spearman")

# ══════════════════════════════════════════════════════════════════════════════
# Layout: TOP block (3x5 dist grid) over BOTTOM strip (3 panels).
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(ok.DOUBLE, 150 * ok.MM))
outer = GridSpec(2, 1, figure=fig, height_ratios=[3.05, 1.0],
                 left=0.045, right=0.99, top=0.965, bottom=0.075, hspace=0.55)

# ── TOP: row-label column + 3x5 grid ─────────────────────────────────────────
gtop = GridSpecFromSubplotSpec(3, 6, subplot_spec=outer[0],
                               width_ratios=[0.09] + [1] * 5,
                               hspace=0.62, wspace=0.55)
for r, lbl in enumerate(ROW_LABELS):
    axl = fig.add_subplot(gtop[r, 0]); axl.axis("off")
    axl.text(0.85, 0.5, lbl, transform=axl.transAxes, fontsize=7, fontweight="bold",
             rotation=90, va="center", ha="center", linespacing=1.3)

axes = [[fig.add_subplot(gtop[r, c + 1]) for c in range(5)] for r in range(3)]
for r, (df, fcol) in enumerate(rows):
    for c, (var, xlab, sub, _sn) in enumerate(PANELS):
        ax = axes[r][c]
        d = df[df["R"] == 1] if sub else df
        dp = d[d[fcol] == 1]; df_ = d[d[fcol] == 0]
        if var == "charge":
            lo, hi = np.floor(d[var].min()), np.ceil(d[var].max())
            bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
            sns.histplot(dp, x=var, bins=bins, color=PASS, ax=ax, stat="density", alpha=ALPHA, lw=0)
            sns.histplot(df_, x=var, bins=bins, color=FAIL, ax=ax, stat="density", alpha=ALPHA, lw=0)
        else:
            sns.histplot(dp, x=var, discrete=True, color=PASS, ax=ax, stat="density", alpha=ALPHA, lw=0)
            sns.histplot(df_, x=var, discrete=True, color=FAIL, ax=ax, stat="density", alpha=ALPHA, lw=0)
        ax.set_xlabel(xlab, fontsize=6.3, labelpad=2)
        ax.set_ylabel("Density" if c == 0 else "", fontsize=6.3, labelpad=2)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=5.6, length=2)
        if ax.get_legend(): ax.get_legend().remove()
        ok.panel_label(fig, ax, LETTERS[r][c], dx=-0.034, dy=0.030, size=8)

# top-block legend + caption, placed in the gap between the two blocks.
# Anchor to the rendered bottom of the lowest top-block x-labels so they never
# sit on the row-k..o distribution axis labels.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
y_lab_bot = min(  # figure-frac y of the lowest x-axis-label extent in the top block
    ax.xaxis.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
    for ax in axes[2])
y_caption = y_lab_bot - 0.012
y_legend = y_lab_bot - 0.050
fig.text(0.5, y_caption, "Pass / Fail = non-polyreactive / polyreactive (PSR rows) · "
         "monomeric / non-monomeric (SEC row)", ha="center", va="top",
         fontsize=5.8, color="#444444")
fig.legend(handles=[Line2D([0], [0], color=PASS, lw=5, alpha=ALPHA, label="Pass (1)"),
                    Line2D([0], [0], color=FAIL, lw=5, alpha=ALPHA, label="Fail (0)")],
           loc="center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, y_legend),
           columnspacing=2.2, handlelength=1.4)

# ── BOTTOM strip: 3 panels (p, q, r) ─────────────────────────────────────────
# Own GridSpec (not nested in outer[1]) so panel p's multi-char y-tick labels get
# left room the top block's thin row-label column doesn't leave. Bottom row of
# outer[1] spans top≈0.255..bottom 0.075 here; mirror that band.
b0, b1 = outer[1].get_position(fig).y0, outer[1].get_position(fig).y1
gbot = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.15, 1.35],
                left=0.085, right=0.99, top=b1, bottom=b0, wspace=0.42)

# (p) Cohen's d effect-size heatmap ------------------------------------------
axp = fig.add_subplot(gbot[0, 0])
dmax = np.nanmax(np.abs(d_mat))
norm_d = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
im = axp.imshow(d_mat, cmap=ok.DIVERGING, norm=norm_d, aspect="auto")
axp.set_xticks(range(3)); axp.set_xticklabels(DSET_NAMES, fontsize=5.8)
axp.set_yticks(range(5))
# keep the Trp label two-line so it doesn't run off the left margin
axp.set_yticklabels([sn for _v, _xl, _s, sn in PANELS], fontsize=5.8, linespacing=0.9)
axp.tick_params(length=0)
for i in range(5):
    for j in range(3):
        v = d_mat[i, j]
        if np.isnan(v):
            continue
        axp.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.6,
                 color=ok.text_on(v, -dmax, dmax, diverging=True))
axp.set_title("Effect size (Fail − Pass)", fontsize=6.5, pad=5)
cb = fig.colorbar(im, ax=axp, fraction=0.05, pad=0.04)
cb.set_label("Cohen's d", fontsize=5.8); cb.ax.tick_params(labelsize=5.2, length=1.5)
ok.panel_label(fig, axp, "p", dx=-0.040, dy=0.052, size=8)

# (q) Net charge -> P(Pass) curve --------------------------------------------
axq = fig.add_subplot(gbot[0, 1])
for cv, col, lab in [(psr_curve, PASS, "IPI PSR (designed)"),
                     (ds1_curve, FAIL, "DS1 (natural)")]:
    # Wilson interval is asymmetric and its centre is shifted from the point
    # estimate, so error arms can come out slightly <0 near p=0/1; clip to 0.
    yerr = np.clip(np.vstack([cv["ppass"] - cv["lo"], cv["hi"] - cv["ppass"]]), 0, None)
    axq.errorbar(cv["charge"], cv["ppass"], yerr=yerr, color=col, marker="o", ms=2.6,
                 lw=1.0, elinewidth=0.7, capsize=1.3, alpha=0.95, label=lab)
axq.set_xlabel("CDR H3 net charge", fontsize=6.3, labelpad=2)
axq.set_ylabel("P(Pass)", fontsize=6.3, labelpad=2)
axq.set_ylim(0, 1.02)
axq.tick_params(labelsize=5.6, length=2)
axq.xaxis.set_major_locator(ticker.MultipleLocator(2))
axq.legend(fontsize=5.4, loc="lower left", handlelength=1.1, borderaxespad=0.2)
n_psr = int(psr_curve["n"].sum()); n_ds1 = int(ds1_curve["n"].sum())
axq.text(0.97, 0.97, f"n = {n_psr:,} / {n_ds1:,}", transform=axq.transAxes,
         ha="right", va="top", fontsize=5.2, color="#444444")
axq.set_title("Charge → pass rate", fontsize=6.5, pad=5)
ok.panel_label(fig, axq, "q", dx=-0.052, dy=0.052, size=8)

# (r) Biophysical determinant correlations (Spearman) ------------------------
axr = fig.add_subplot(gbot[0, 2])
norm_c = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
imc = axr.imshow(corr.values, cmap=ok.DIVERGING, norm=norm_c, aspect="auto")
axr.set_xticks(range(len(corr_labels)))
axr.set_xticklabels(corr_labels, fontsize=5.0, rotation=45, ha="right")
axr.set_yticks(range(len(corr_labels)))
axr.set_yticklabels(corr_labels, fontsize=5.0)
axr.tick_params(length=0)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        v = corr.values[i, j]
        if np.isnan(v):
            continue
        axr.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."),
                 ha="center", va="center", fontsize=4.2,
                 color=ok.text_on(v, -1.0, 1.0, diverging=True))
axr.set_title("Biophysical determinant correlations", fontsize=6.5, pad=5)
cbc = fig.colorbar(imc, ax=axr, fraction=0.046, pad=0.03)
cbc.set_label("Spearman ρ", fontsize=5.8)
cbc.set_ticks([-1, -0.5, 0, 0.5, 1]); cbc.ax.tick_params(labelsize=5.2, length=1.5)
ok.panel_label(fig, axr, "r", dx=-0.060, dy=0.052, size=8)

ok.save_fig(fig, "Figure5", OUT)
print("Fig2 done")
