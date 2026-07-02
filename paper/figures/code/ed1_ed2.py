"""
Extended Data Figures 1 & 2 — 8-panel CDR H3 biophysical
profiling.
  ED1: IPI PSR ELISA-only (denoised)   Pass/Fail = non-polyreactive / polyreactive
  ED2: IPI SEC                         Pass/Fail = monomeric / aggregating
Panels: a Arg | b Asp | c Trp(Arg=1) | d Glu | e loop length | f net charge
        | g CDR H3 pI | h heavy-chain pI
Faithful to utils/Figure2_physicochemical.generate_extended_figure1 / generate_figure2;
restyled to Okabe-Ito. Charge & pI via the original liabilities functions.
Data: data/ipi_psr_trainset_elisa.xlsx, data/ipi_sec_5000.xlsx
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/Andre.Teixeira/temp/delphi/utils")
import okabe_style as ok
import liabilities
warnings.filterwarnings("ignore")

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
DATA = f"{DELPHI}/data"; OUT = f"{DELPHI}/revision2_redteam/figures/output"
ok.set_style(base_pt=6.5)
PASS, FAIL = ok.PASS, ok.FAIL
ALPHA = 0.72


def add_features(df):
    df = df.copy(); s = df["CDR3"].astype(str)
    df["R"] = s.str.count("R"); df["D"] = s.str.count("D")
    df["E"] = s.str.count("E"); df["W"] = s.str.count("W"); df["CDR3_len"] = s.str.len()
    df["charge"] = s.map(liabilities.charge_value)
    df["hcdr3_pi"] = s.map(liabilities.IsoelectricPoint)
    df["vh_pi"] = df["HSEQ"].astype(str).map(liabilities.IsoelectricPoint)
    return df[df["CDR3_len"] < 25]


# (col, x-label, mode)  mode: 'count' integer | 'bin' 1-unit bins | subset Arg==1 flag via 'W'
PANELS = [
    ("R", "Arginine count (CDR H3)", "count"),
    ("D", "Aspartic acid count (CDR H3)", "count"),
    ("W", "Tryptophan count (CDR H3)\n(Arg count = 1)", "count"),
    ("E", "Glutamic acid count (CDR H3)", "count"),
    ("CDR3_len", "CDR H3 loop length", "count"),
    ("charge", "Net charge (CDR H3)", "bin"),
    ("hcdr3_pi", "Isoelectric point (CDR H3)", "bin"),
    ("vh_pi", "Isoelectric point\n(heavy chain)", "bin"),
]


def build(df, fcol, meaning, outname, cohort):
    fig = plt.figure(figsize=(ok.DOUBLE, 98 * ok.MM))
    # cohort/framing line so this reads as the full per-dataset biophysical panel
    # (distinct from Fig. 2's cross-dataset signature; net charge + pI are new here)
    fig.text(0.5, 0.975, cohort, ha="center", va="top", fontsize=6.2, fontweight="bold")
    gs = GridSpec(2, 4, figure=fig, left=0.06, right=0.99, top=0.90, bottom=0.16,
                  hspace=0.62, wspace=0.40)
    letters = list("abcdefgh")
    for i, (var, xlab, mode) in enumerate(PANELS):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        d = df[df["R"] == 1] if var == "W" else df
        dp = d[d[fcol] == 1][var].dropna(); dfn = d[d[fcol] == 0][var].dropna()
        if mode == "count":
            sns.histplot(x=dp, discrete=True, color=PASS, ax=ax, stat="density", alpha=ALPHA, lw=0)
            sns.histplot(x=dfn, discrete=True, color=FAIL, ax=ax, stat="density", alpha=ALPHA, lw=0)
        else:
            lo = np.floor(min(dp.min(), dfn.min())); hi = np.ceil(max(dp.max(), dfn.max()))
            bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
            sns.histplot(x=dp, bins=bins, color=PASS, ax=ax, stat="density", alpha=ALPHA, lw=0)
            sns.histplot(x=dfn, bins=bins, color=FAIL, ax=ax, stat="density", alpha=ALPHA, lw=0)
        ax.set_xlabel(xlab, fontsize=6.3, labelpad=2)
        ax.set_ylabel("Density" if i % 4 == 0 else "", fontsize=6.3, labelpad=2)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=5.6, length=2)
        if ax.get_legend(): ax.get_legend().remove()
        ok.panel_label(fig, ax, letters[i], dx=-0.045, dy=0.028, size=8)
    fig.legend(handles=[Line2D([0], [0], color=PASS, lw=5, alpha=ALPHA, label=f"Pass (1) — {meaning[0]}"),
                        Line2D([0], [0], color=FAIL, lw=5, alpha=ALPHA, label=f"Fail (0) — {meaning[1]}")],
               loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.01),
               columnspacing=2.5, handlelength=1.4)
    ok.save_fig(fig, outname, OUT)
    plt.close(fig)


# ED1 — PSR ELISA-only (4 ELISA scores present + psr_filter)
eli = pd.read_excel(f"{DATA}/ipi_psr_trainset_elisa.xlsx")
eli = eli[eli[["psr_norm_insulin", "psr_norm_dna", "psr_norm_smp", "psr_norm_avidin"]].notna().all(axis=1)]
eli = eli[eli["psr_filter"].notna()]
print("ED1 n=", len(eli))
build(add_features(eli), "psr_filter", ("non-polyreactive", "polyreactive"), "ED_Fig1",
      f"Full CDR H3 biophysical profile — IPI PSR-ELISA only (denoised, n={len(eli):,});  "
      "net charge and isoelectric point extend the Fig. 2 signature")

# ED2 — SEC
sec = pd.read_excel(f"{DATA}/ipi_sec_5000.xlsx"); sec = sec[sec["sec_filter"].notna()]
print("ED2 n=", len(sec))
build(add_features(sec), "sec_filter", ("monomeric", "aggregating"), "ED_Fig2",
      f"Full CDR H3 biophysical profile — IPI SEC (n={len(sec):,});  "
      "net charge and isoelectric point extend the Fig. 2 signature")
print("ED1/ED2 done")
