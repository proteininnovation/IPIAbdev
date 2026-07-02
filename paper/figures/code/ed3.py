"""
Extended Data Figure 3.
  a  Spearman ρ correlation heatmap across all models on IPI PSR validation
  b  Spearman ρ correlation heatmap across models on DS1 (cross-library)
  c  Score-distribution KDEs, IPI PSR validation (all models)
  d  Score-distribution KDEs, DS1 (Transformer + AbLang2 architectures)
Faithful to utils/Extended_fig3.py; restyled (sequential blue heatmaps instead of
viridis; Okabe-Ito for the 9-model KDE).
Data: figures_tables/Suppl_Table2_prediction_score_val.xlsx
"""
import sys, os, re, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
warnings.filterwarnings("ignore")

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
XLSX = f"{DELPHI}/figures_tables/Suppl_Table2_prediction_score_val.xlsx"
OUT = f"{DELPHI}/revision2_redteam/figures/output"
ok.set_style(base_pt=6)
ARCH = {'Trans': 0, 'CNN': 1, 'XGB': 2, 'RF': 3}
EMB = {'AbLang2': 0, 'IgBert': 1, 'AntiBERTy': 2, 'AntiBERTa2': 3, 'AntiBERTa2-CSSP': 4,
       'OneHot': 5, 'k-mer': 6, 'Biophys': 7}


def shorten(col):
    name = re.sub(r'_ipi_psr_trainset_train_score$', '', col)
    am = {'transformer_lm': 'Trans', 'transformer_onehot': 'Trans', 'cnn': 'CNN', 'xgboost': 'XGB', 'rf': 'RF'}
    em = {'ablang': 'AbLang2', 'igbert': 'IgBert', 'antiberty': 'AntiBERTy', 'antiberta2': 'AntiBERTa2',
          'antiberta2-cssp': 'AntiBERTa2-CSSP', 'onehot': 'OneHot', 'kmer': 'k-mer', 'biophysical': 'Biophys'}
    for a in sorted(am, key=len, reverse=True):
        if name.startswith(a + '_'):
            return f"{am[a]}_{em.get(name[len(a)+1:], name[len(a)+1:])}"
    return name


def load(sheet):
    df = pd.read_excel(XLSX, sheet_name=sheet)
    sc = [c for c in df.columns if c.endswith('_train_score')]
    s = df[sc].copy(); s.columns = [shorten(c) for c in sc]
    return s


def order(cols):
    def key(c):
        p = c.split('_', 1); return (ARCH.get(p[0], 99), EMB.get(p[1] if len(p) > 1 else '', 99), c)
    return sorted(cols, key=key)


def corr_panel(ax, scores, title, annot):
    cols = order(list(scores.columns)); corr = scores[cols].corr(method='spearman')
    vmin = float(corr.values.min())
    sns.heatmap(corr, ax=ax, cmap=ok.SEQ, vmin=vmin, vmax=1.0, annot=annot, fmt='.2f',
                annot_kws={'fontsize': 4.3}, square=True, cbar=False, linewidths=0.3, linecolor='white')
    ax.set_title(title, pad=4, fontsize=6.3)
    ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_yticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=90, ha='center', fontsize=4.4)
    ax.set_yticklabels(cols, rotation=0, va='center', fontsize=4.4)
    ax.tick_params(length=0)
    bb = ax.get_position(); core = min(bb.width, bb.height) / bb.height
    cax = ax.inset_axes([1.04, (1 - core) / 2, 0.03, core])
    sm = mpl.cm.ScalarMappable(cmap=ok.SEQ, norm=mpl.colors.Normalize(vmin=vmin, vmax=1.0))
    cb = ax.figure.colorbar(sm, cax=cax); cb.set_label("Spearman ρ", fontsize=5.6); cax.tick_params(labelsize=4.6)


def kde_panel(ax, scores, title, cmap, ncol):
    cols = order(list(scores.columns)); xg = np.linspace(0, 1, 400)
    for i, m in enumerate(cols):
        v = np.clip(scores[m].dropna().values, 0, 1)
        if len(v) < 2 or np.std(v) < 1e-9: continue
        ax.plot(xg, gaussian_kde(v, bw_method=0.15)(xg), lw=0.8, color=cmap[i % len(cmap)], label=m, alpha=0.85)
    ax.set_xlabel("Predicted P(Pass)", fontsize=6.3); ax.set_ylabel("Density", fontsize=6.3)
    ax.set_xlim(0, 1); ax.set_title(title, pad=4, fontsize=6.3); ax.grid(alpha=0.3, lw=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False,
              handlelength=1.2, columnspacing=0.7, labelspacing=0.3, fontsize=4.4)


ipi, ds1 = load('ipi_psr_trainset_val'), load('DS1')
print(f"IPI {ipi.shape}  DS1 {ds1.shape}")
tab20 = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20').colors]

fig = plt.figure(figsize=(ok.DOUBLE, 200 * ok.MM))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.8], height_ratios=[1.5, 1.0],
              hspace=0.55, wspace=0.6, left=0.11, right=0.95, top=0.93, bottom=0.16)
axa, axb, axc, axd = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                      fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
corr_panel(axa, ipi, f"IPI PSR validation (n={len(ipi):,})\n{ipi.shape[1]} models", annot=False)
corr_panel(axb, ds1, f"DS1 cross-library (n={len(ds1):,})\n{ds1.shape[1]} models", annot=True)
kde_panel(axc, ipi, f"IPI PSR validation — score distributions ({ipi.shape[1]} models)", tab20, ncol=5)
kde_panel(axd, ds1, f"DS1 — score distributions ({ds1.shape[1]} models)", ok.QUALITATIVE, ncol=3)
for ax, l in [(axa, 'a'), (axb, 'b'), (axc, 'c'), (axd, 'd')]:
    ok.panel_label(fig, ax, l, dx=-0.06, dy=0.02, size=9)
ok.save_fig(fig, "ED_Fig3", OUT)
print("ED3 done")
