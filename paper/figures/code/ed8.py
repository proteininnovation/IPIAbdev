"""
Extended Data Figure 4 (publication-quality) — full external-clinical-validation grid.
  a  AUC-ROC heatmap     : every language model × two Jain 2017 readouts
  b  Spearman ρ heatmap  : P(Pass) score vs continuous assay, with significance stars

Main Figure 5b summarises external validation for the deployed model
(Transformer + AbLang2). This ED figure preserves the full grid that was in the
original Figure 4 (every LM × every assay) so nothing is lost.

Binary AUC is calculated only for Jain 2017, using its assay-specific
developability flags. GDPa1 and GDPa3 are retained only in the continuous
Spearman grid because the Jain 0.27 threshold does not apply to Ginkgo assays.
"""
import argparse, sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Generate DELPHI Extended Data Figure 4.")
parser.add_argument("--jain", required=True)
parser.add_argument("--gdpa1", required=True)
parser.add_argument("--gdpa3", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
OUT = os.path.abspath(args.output_dir)
os.makedirs(OUT, exist_ok=True)
ok.set_style(base_pt=6.5)

# ── constants (verbatim from v1 fig4.py) ──────────────────────────────────────
THRESH, THRESH_ELISA = 0.27, 1.9
HM_LM_ORDER = ['ablang', 'antiberty', 'antiberta2', 'antiberta2-cssp', 'igbert', 'onehot']
HM_LM_DISPLAY = ['AbLang2', 'AntiBERTy', 'AntiBERTa2', 'AntiBERTa2-CSSP', 'IgBert', 'One-hot']
AUC_DATASETS = ['Jain 2017\nPSR SMP', 'Jain 2017\nELISA']
RHO_DATASETS = ['Jain 2017\nPSR SMP', 'Jain 2017\nELISA', 'GDPa1\nPR Ova', 'GDPa1\nPR CHO', 'GDPa3\nPR Ova', 'GDPa3\nPR CHO']


# ── data + computation (verbatim from v1 fig4.py) ─────────────────────────────
def score_col(lm):
    return 'transformer_onehot_onehot_ipi_psr_trainset_score' if lm == 'onehot' \
        else f'transformer_lm_{lm}_ipi_psr_trainset_score'


def safe_auc(df, sc, col, min_fail=5, thresh=None):
    sub = df[[sc, col]].dropna(); t = thresh if thresh is not None else THRESH
    y = (sub[col] < t).astype(int)
    if (y == 0).sum() < min_fail or (y == 1).sum() < min_fail: return np.nan
    return roc_auc_score(y, sub[sc])


def safe_rho_p(df, sc, col):
    sub = df[[sc, col]].dropna()
    if len(sub) < 5: return np.nan, np.nan
    return spearmanr(sub[sc], sub[col])


def heatmap_matrices(jain, g1, g3):
    n = len(HM_LM_ORDER)
    auc = np.full((n, 2), np.nan); rho = np.full((n, 6), np.nan); pv = np.full((n, 6), np.nan)
    for i, lm in enumerate(HM_LM_ORDER):
        c = score_col(lm)
        if c in jain.columns:
            auc[i, 0] = safe_auc(jain, c, 'PSR_SMP_Score'); auc[i, 1] = safe_auc(jain, c, 'ELISA', thresh=THRESH_ELISA)
            rho[i, 0], pv[i, 0] = safe_rho_p(jain, c, 'PSR_SMP_Score'); rho[i, 1], pv[i, 1] = safe_rho_p(jain, c, 'ELISA')
        if c in g1.columns:
            rho[i, 2], pv[i, 2] = safe_rho_p(g1, c, 'polyreactivity_prscore_ova_avg'); rho[i, 3], pv[i, 3] = safe_rho_p(g1, c, 'polyreactivity_prscore_cho_avg')
        if c in g3.columns:
            rho[i, 4], pv[i, 4] = safe_rho_p(g3, c, 'polyreactivity_prscore_ova_avg'); rho[i, 5], pv[i, 5] = safe_rho_p(g3, c, 'polyreactivity_prscore_cho_avg')
    return auc, rho, pv


def _stars(p):
    if np.isnan(p): return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


# ── drawing (verbatim header/title placement from v1 fig4.py draw_heatmap) ─────
def draw_heatmap(ax, mat, row_labels, col_labels, title, vmin, vmax, vcenter,
                 cbar_label, fmt='.2f', cbar_ticks=None, pval=None, col_sep=(1, 3),
                 subtitle=None, letter=None):
    n_r, n_c = mat.shape
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(mat, aspect='auto', cmap=ok.DIVERGING, norm=norm, interpolation='nearest',
                   extent=[-0.5, n_c - 0.5, n_r - 0.5, -0.5])
    for i in range(n_r):
        for j in range(n_c):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=5, color='#888'); continue
            rgba = ok.DIVERGING(norm(v)); lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = 'white' if lum < 0.5 else '#222222'
            star = _stars(pval[i, j]) if pval is not None else ''
            txt = format(v, fmt) + (f"\n{star}" if star else '')
            ax.text(j, i, txt, ha='center', va='center', fontsize=5.4 if star else 6.4,
                    color=tc, fontweight='bold' if abs(v - vcenter) > 0.12 else 'normal', linespacing=1.15)
    ax.set_ylim(n_r - 0.5, -0.5)
    ax.set_xticks(range(n_c)); ax.set_xticklabels(col_labels, fontsize=6, rotation=32, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(n_r)); ax.set_yticklabels(row_labels, fontsize=6.4)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    for sep in col_sep:  # column separators between cohorts (Jain | GDPa1 | GDPa3)
        ax.axvline(sep + 0.5, color='#555555', lw=1.0, ls='--', zorder=5)
    # header in FIGURE coords so title/subtitle/letter keep fixed gaps regardless
    # of the (short) heatmap axes height
    fig = ax.figure; bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + 0.022, title, fontsize=6.8, fontweight='bold', va='baseline', ha='left')
    if subtitle:
        fig.text(bb.x0, bb.y1 + 0.008, subtitle, fontsize=5.6, color='#555555', va='baseline', ha='left')
    if letter:
        fig.text(bb.x0 - 0.052, bb.y1 + 0.022, letter, fontsize=9, fontweight='bold', va='baseline', ha='left')
    div = make_axes_locatable(ax); cax = div.append_axes('right', size='3.5%', pad=0.08)
    cb = plt.colorbar(im, cax=cax); cb.set_label(cbar_label, fontsize=6, labelpad=3)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    cb.ax.tick_params(labelsize=5.5, length=2)


# ── build ─────────────────────────────────────────────────────────────────────
jain = pd.read_excel(args.jain)
g1 = pd.read_excel(args.gdpa1)
g3 = pd.read_excel(args.gdpa3)
auc_mat, rho_mat, pv = heatmap_matrices(jain, g1, g3)

fig = plt.figure(figsize=(ok.DOUBLE, 150 * ok.MM))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0],
              hspace=0.95, left=0.135, right=0.94, top=0.90, bottom=0.105)

# AUC panel restricted to the two Jain 2017 readouts and their developability flags.
axa = fig.add_subplot(gs[0, 0])
# Keep this two-column panel compact instead of stretching two cells across the page.
axa_box = axa.get_position()
compact_width = axa_box.width * 0.46
axa.set_position([
    axa_box.x0 + (axa_box.width - compact_width) / 2,
    axa_box.y0,
    compact_width,
    axa_box.height,
])
draw_heatmap(axa, auc_mat, HM_LM_DISPLAY, AUC_DATASETS, "Jain 2017 clinical panel · AUC-ROC",
             vmin=0.20, vmax=0.90, vcenter=0.5, cbar_label='AUC-ROC', fmt='.3f', letter='a', col_sep=(),
             subtitle=f"Jain developability flags: PSR SMP < {THRESH}; ELISA < {THRESH_ELISA}")
axb = fig.add_subplot(gs[1, 0])
draw_heatmap(axb, rho_mat, HM_LM_DISPLAY, RHO_DATASETS, "External clinical validation · Spearman ρ",
             vmin=-0.80, vmax=0.20, vcenter=0.0, cbar_label='Spearman ρ', fmt='.2f',
             cbar_ticks=[-0.8, -0.4, 0.0, 0.2], pval=pv, letter='b',
             subtitle="P(Pass) vs polyreactivity assay score   (*** p<0.001  ** p<0.01  * p<0.05)")

ok.save_fig(fig, "ED_Fig4", OUT)   # renumbered: external grid is now Extended Data Fig. 4 (citation order)
print("ED(external grid -> ED_Fig4) done")
