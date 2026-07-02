"""
Extended Data Figures 5 & 6 — feature attribution beeswarms.
  a  RF-biophysical SHAP summary        b  XGBoost-biophysical SHAP summary
  c  Transformer (one-hot) IG per amino acid
  ED5 = PSR, ED6 = SEC.
Rebuilt from the saved per-sample attribution tables (no model re-run, nothing
invented). Points are subsampled per row only for rendering density; ordering and
colour use all rows.
Data: GENERATED_NBT_revision/analysis_runs/interp_{psr,sec}_*/*_beeswarm_{RF,XGBoost,Transformer}_*.csv
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
warnings.filterwarnings("ignore")

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
AR = f"{DELPHI}/GENERATED_NBT_revision/analysis_runs"
OUT = f"{DELPHI}/revision2_redteam/figures/output"
ok.set_style(base_pt=6.5)
RNG = np.random.default_rng(0)
N_SHOW = 1500

AA_CLASS = {**{a: "Cationic" for a in "RKH"}, **{a: "Anionic" for a in "DE"},
            **{a: "Aromatic" for a in "WFY"}, **{a: "Hydrophobic" for a in "AILMVPG"},
            **{a: "Polar" for a in "STNQC"}}
CLASS_COL = {"Cationic": ok.OI_VERMILION, "Anionic": ok.OI_BLUE, "Aromatic": ok.OI_GREEN,
             "Hydrophobic": ok.OI_ORANGE, "Polar": ok.OI_GREY}


def pretty(f):
    f = f.replace("cdr3_", "CDR3 ").replace("vh_", "VH ").replace("_", " ")
    for a, b in [("hydrophobicity", "hydrophob."), ("aromaticity", "aromatic."),
                 ("instability", "instab."), ("frac hydrophobic", "frac phob"),
                 ("frac charged", "frac chg"), ("VH CDR3 length ratio", "VH/CDR3 len"),
                 ("hydrophobic", "hydrophob.")]:
        f = f.replace(a, b)
    if f.endswith(" pi"):
        f = f[:-3] + " pI"
    return f


def shap_beeswarm(ax, df, n_top=20):
    order = df.groupby("feature")["shap_value"].apply(lambda s: s.abs().mean()).sort_values()
    order = order.index.tolist()[-n_top:]
    for yi, f in enumerate(order):
        sub = df[df["feature"] == f]
        fv = sub["feature_value"].values; sv = sub["shap_value"].values
        lo, hi = np.nanpercentile(fv, 5), np.nanpercentile(fv, 95)
        cn = np.clip((fv - lo) / (hi - lo + 1e-9), 0, 1)
        if len(sub) > N_SHOW:
            idx = RNG.choice(len(sub), N_SHOW, replace=False); sv, cn = sv[idx], cn[idx]
        jit = yi + (RNG.random(len(sv)) - 0.5) * 0.72
        ax.scatter(sv, jit, c=cn, cmap=ok.DIVERGING, vmin=0, vmax=1, s=1.6, alpha=0.5, lw=0, rasterized=True)
    ax.axvline(0, color="#888888", lw=0.5, zorder=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([pretty(f) for f in order], fontsize=5.0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("SHAP value  (← Fail | Pass →)", fontsize=6.0)


def ig_beeswarm(ax, df, n_top=22):
    df = df.copy(); df["key"] = df["region"].str.replace("_framework", "") + " " + df["aa"]
    order = df.groupby("key")["ig_value"].apply(lambda s: s.abs().mean()).sort_values()
    order = order.index.tolist()[-n_top:]
    for yi, k in enumerate(order):
        sub = df[df["key"] == k]; iv = sub["ig_value"].values
        aa = k.split()[-1]; col = CLASS_COL[AA_CLASS.get(aa, "Polar")]
        if len(sub) > N_SHOW:
            iv = iv[RNG.choice(len(sub), N_SHOW, replace=False)]
        jit = yi + (RNG.random(len(iv)) - 0.5) * 0.72
        ax.scatter(iv, jit, color=col, s=1.6, alpha=0.45, lw=0, rasterized=True)
    ax.axvline(0, color="#888888", lw=0.5, zorder=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5.0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("IG value  (← Fail | Pass →)", fontsize=6.0)


def build(tag, db_stem, label, outname):
    base = f"{AR}/interp_{tag}_{db_stem}/interp_{tag}_biophysical_biophysical_onehot_{db_stem}_beeswarm"
    rf = pd.read_csv(f"{base}_RF_{tag}.csv")
    xgb = pd.read_csv(f"{base}_XGBoost_{tag}.csv")
    tr = pd.read_csv(f"{base}_Transformer_{tag}.csv")

    fig = plt.figure(figsize=(ok.DOUBLE, 100 * ok.MM))
    gs = GridSpec(1, 3, figure=fig, left=0.13, right=0.965, top=0.86, bottom=0.155, wspace=0.66)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    shap_beeswarm(axa, rf); axa.set_title(f"RF-SHAP · {label}\n(biophysical)", fontsize=6.6, fontweight="bold", pad=3)
    shap_beeswarm(axb, xgb); axb.set_title(f"XGBoost-SHAP · {label}\n(biophysical)", fontsize=6.6, fontweight="bold", pad=3)
    ig_beeswarm(axc, tr); axc.set_title(f"Transformer one-hot · {label}\n(IG per amino acid)", fontsize=6.6, fontweight="bold", pad=3)

    # shared colourbar for SHAP feature value, in a figure-level slot above the titles
    sm = mpl.cm.ScalarMappable(cmap=ok.DIVERGING, norm=mpl.colors.Normalize(0, 1))
    cax = fig.add_axes([0.13, 0.955, 0.18, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=5, length=0)
    cb.set_label("SHAP feature value", fontsize=5.6, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    # AA-class legend for panel c
    axc.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=4, label=n)
                        for n, c in CLASS_COL.items()],
               loc="lower right", fontsize=4.8, handletextpad=0.2, labelspacing=0.2,
               title="AA class", title_fontsize=5.0, frameon=True, framealpha=0.9, edgecolor="#cccccc")
    for ax, l in [(axa, "a"), (axb, "b"), (axc, "c")]:
        ok.panel_label(fig, ax, l, dx=-0.085, dy=0.028, size=8.5)
    ok.save_fig(fig, outname, OUT)
    plt.close(fig)
    print(outname, "done")


build("psr_filter", "ipi_psr", "PSR", "ED_Fig6")   # renumbered: SHAP/IG beeswarm PSR -> Extended Data Fig. 6
build("sec_filter", "ipi_sec_5000", "SEC", "ED_Fig7")   # renumbered: SHAP/IG beeswarm SEC -> Extended Data Fig. 7
