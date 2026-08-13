"""
Extended Data Figure 5 — predicted-score distributions per
architecture, within-distribution (IPI val) vs cross-library (DS1).
  row 1 (a-e): IPI validation     row 2 (f-j): DS1 public dataset
Predicted-score densities across 4 classifier architectures (+ OneHot transformer);
per-panel AUC, accuracy at the 0.5 and Youden J-optimal thresholds, Okabe-Ito.
Demoted out of the main figure set (old main Fig 5) into Extended Data.
Data logic identical to figures_nature_v1/code/fig5.py.
Data: figures_tables/Suppl_Table2_prediction_score_val.xlsx  (sheets: ipi_psr_trainset_val, DS1)
"""
import argparse, sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Generate updated DELPHI Extended Data Figure 5.")
parser.add_argument("--xlsx", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
XLSX = os.path.abspath(args.xlsx)
OUT = os.path.abspath(args.output_dir)
os.makedirs(OUT, exist_ok=True)
ok.set_style(base_pt=6.5)
PASS, FAIL = ok.PASS, ok.FAIL
STD_COL, Y_COL = ok.OI_GREY, "#222222"
STD = 0.5

PANELS = [("transformer_lm_ablang_ipi_psr_trainset_train_score", "Transformer + AbLang2", "a"),
          ("cnn_ablang_ipi_psr_trainset_train_score", "CNN + AbLang2", "b"),
          ("xgboost_ablang_ipi_psr_trainset_train_score", "XGBoost + AbLang2", "c"),
          ("rf_ablang_ipi_psr_trainset_train_score", "Random Forest + AbLang2", "d"),
          ("transformer_onehot_onehot_ipi_psr_trainset_train_score", "Transformer + OneHot", "e")]
LETTERS2 = list("fghij")


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def acc(y, s, t):
    return float(((s >= t).astype(int) == y).mean())


def draw(ax, s, y, title, letter):
    try: auc = roc_auc_score(y, s)
    except Exception: auc = np.nan
    ty = youden(y, s); a_std, a_y = acc(y, s, STD), acc(y, s, ty)
    bins = np.linspace(0, 1, 26); c = (bins[:-1] + bins[1:]) / 2; wdt = np.diff(bins)
    pc, _ = np.histogram(np.clip(s[y == 1], 0, 1), bins=bins)
    fc, _ = np.histogram(np.clip(s[y == 0], 0, 1), bins=bins)
    ax.bar(c, fc / max(fc.max(), 1), width=wdt, color=FAIL, alpha=0.75, edgecolor='none')
    ax.bar(c, pc / max(pc.max(), 1), width=wdt, color=PASS, alpha=0.65, edgecolor='none')
    ax.axvline(STD, color=STD_COL, ls='--', lw=1.0)
    ax.axvline(ty, color=Y_COL, ls='-.', lw=0.9)
    ax.set_ylim(0, 1.22); ax.set_xlim(0, 1)
    # threshold labels: std anchored left of its line, Youden right of its line,
    # so they never sit on top of each other even when the two lines coincide
    ax.text(STD - 0.03, 1.12, "0.5", color=STD_COL, ha='right', va='bottom', fontsize=5, fontweight='bold')
    if ty > 0.82:
        ax.text(ty - 0.03, 1.12, f"Y={ty:.2f}", color=Y_COL, ha='right', va='bottom', fontsize=5)
    else:
        ax.text(ty + 0.03, 1.12, f"Y={ty:.2f}", color=Y_COL, ha='left', va='bottom', fontsize=5)
    ax.set_title(title, fontsize=6.3, fontweight='bold', pad=3)
    ax.grid(alpha=0.22, lw=0.3, axis='y')
    ax.text(0.05, 0.97, f"AUC {auc:.3f}\nAcc {a_std:.2f}/{a_y:.2f}\nn {int((y==1).sum()):,}/{int((y==0).sum()):,}",
            transform=ax.transAxes, fontsize=4.9, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#bbbbbb', alpha=0.9, lw=0.4))


# ── load + build ──────────────────────────────────────────────────────────────
ipi = pd.read_excel(XLSX, sheet_name='ipi_psr_trainset_val')
ds1 = pd.read_excel(XLSX, sheet_name='DS1')
print(f"IPI val n={len(ipi):,}  DS1 n={len(ds1):,}")

fig, axes = plt.subplots(2, 5, figsize=(ok.DOUBLE, 95 * ok.MM), sharex=True)
# Reserve enough space for the two multiline row labels at journal page size.
# The previous 0.085 margin clipped "IPI validation" and "DS1 public dataset"
# after the standalone figure was placed into the compiled Extended Data PDF.
fig.subplots_adjust(left=0.14, right=0.985, top=0.90, bottom=0.155, hspace=0.42, wspace=0.30)

for row, (df, letters) in enumerate([(ipi, [p[2] for p in PANELS]), (ds1, LETTERS2)]):
    yv = df['psr_filter'].values.astype(int)
    for col, (sc, title, _) in enumerate(PANELS):
        ax = axes[row, col]
        if sc not in df.columns:
            ax.set_axis_off(); continue
        s = df[sc].values; m = ~np.isnan(s) & ~np.isnan(yv)
        draw(ax, s[m], yv[m], title, letters[col])
        ok.panel_label(fig, ax, letters[col], dx=-0.027, dy=0.030, size=8)

axes[0, 0].set_ylabel("IPI validation\n(within-distribution)\n\nRel. frequency", fontsize=6.3)
axes[1, 0].set_ylabel("DS1 public dataset\n(cross-library)\n\nRel. frequency", fontsize=6.3)
for ax in axes[1]:
    ax.set_xlabel("Predicted P(Pass)", fontsize=6.3)

fig.legend(handles=[Patch(facecolor=PASS, alpha=0.65, label='Pass (psr_filter = 1)'),
                    Patch(facecolor=FAIL, alpha=0.75, label='Fail (psr_filter = 0)'),
                    Line2D([0], [0], color=STD_COL, ls='--', lw=1.0, label='Threshold = 0.5'),
                    Line2D([0], [0], color=Y_COL, ls='-.', lw=0.9, label="Youden J-optimal threshold")],
           loc='lower center', ncol=4, fontsize=6, bbox_to_anchor=(0.5, 0.01),
           handlelength=1.8, columnspacing=1.8)
ok.save_fig(fig, "ED_Fig5", OUT)   # renumbered: predicted-score density is now Extended Data Fig. 5
print("ED(score-density -> ED_Fig5) done")
