"""
Figure 3 — Within-distribution model benchmarking.
  a  ROC curves, 5 representative models on the 20% internal validation split
  b  Precision-Recall curves, same 5 models (legend = average precision)
  c  Calibration / reliability curves, same 5 models
  d  arch x LM cross-validation AUC bars (10-fold CV sheet)
  e  Confusion matrices at the Youden-J threshold (trans_ablang vs trans_one_hot)
  f  Learning curve: AUC vs training-set size (IPI vs DS1) with quadratic fit

Every number is read from the source files — no invented data. Curve-panel CIs
are bootstrap only (2000 resamples, fixed seed). Colourblind-safe: Okabe-Ito
colours plus a distinct line style per series.
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
from paths import data_file, ensure_output
warnings.filterwarnings("ignore")

OUT = str(ensure_output())
ok.set_style(base_pt=6.5)

# ── representative models (column -> display name), shared across a/b/c ──────────
MODELS = [
    ("trans_ablang",  "Transformer + AbLang2"),
    ("cnn_ablang",    "CNN + AbLang2"),
    ("xgb_ablang",    "XGBoost + AbLang2"),
    ("rf_ablang",     "RF + AbLang2"),
    ("trans_one_hot", "Transformer + one-hot"),
]
COLS = ok.qualitative(5)                      # blue, orange, green, purple, skyblue
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]   # greyscale-safe distinction

# ── load the authoritative 20% validation split from public Supplementary Table 4
validation_workbook = data_file(
    "manuscript_tables/DELPHI_Supplementary_Table_4.xlsx"
)
validation_source = pd.read_excel(
    validation_workbook, sheet_name="ipi_psr_trainset_val"
)
validation_columns = {
    "trans_ablang": "transformer_lm_ablang_ipi_psr_trainset_train_score",
    "cnn_ablang": "cnn_ablang_ipi_psr_trainset_train_score",
    "xgb_ablang": "xgboost_ablang_ipi_psr_trainset_train_score",
    "rf_ablang": "rf_ablang_ipi_psr_trainset_train_score",
    "trans_one_hot": "transformer_onehot_onehot_ipi_psr_trainset_train_score",
}
missing = [column for column in validation_columns.values()
           if column not in validation_source.columns]
if missing:
    raise ValueError(f"Supplementary Table 4 is missing columns: {missing}")
val = validation_source[["psr_filter", *validation_columns.values()]].rename(
    columns={source: display for display, source in validation_columns.items()}
)
Y = val["psr_filter"].astype(int).values     # 1 = Pass

RNG = np.random.default_rng(0)
N_BOOT = 2000


def boot_auc_ci(y, s, n_boot=N_BOOT):
    """Bootstrap 95% CI for ROC-AUC (percentile, paired resampling of indices)."""
    idx = np.arange(len(y))
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        r = RNG.choice(idx, size=len(idx), replace=True)
        yr = y[r]
        if yr.min() == yr.max():              # degenerate resample, skip
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(yr, s[r])
    aucs = aucs[~np.isnan(aucs)]
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def youden_threshold(y, s):
    """Threshold on the score that maximises sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return thr[int(np.argmax(j))]


# ── precompute per-model metrics ────────────────────────────────────────────────
roc_data, pr_data, cal_data = [], [], []
auc_report, ap_report = {}, {}
for (col, name), c, ls in zip(MODELS, COLS, LINESTYLES):
    s = val[col].values
    fpr, tpr, _ = roc_curve(Y, s)
    auc = roc_auc_score(Y, s)
    lo, hi = boot_auc_ci(Y, s)
    auc_report[name] = (auc, lo, hi)
    roc_data.append((name, c, ls, fpr, tpr, auc, lo, hi))

    prec, rec, _ = precision_recall_curve(Y, s)
    ap = average_precision_score(Y, s)
    ap_report[name] = ap
    pr_data.append((name, c, ls, rec, prec, ap))

    frac_pos, mean_pred = calibration_curve(Y, s, n_bins=10, strategy="uniform")
    cal_data.append((name, c, ls, mean_pred, frac_pos))


# ── panel d: 10-fold CV AUC, grouped by architecture ────────────────────────────
cv = pd.read_excel(data_file("Figure4_data.xlsx"), sheet_name="Fig4A_IPI_10fold_CV", skiprows=1)
cv.columns = cv.columns.str.strip()
cv["Architecture"] = cv["Architecture"].ffill()
cv = cv.dropna(subset=["Language Model"]).rename(columns={"Language Model": "LM"})
ARCH_ORDER = ["CNN", "Transformer", "RF", "XGBoost"]
# compact tick labels so the 24 vertical x-labels stay short (full names in legend d note)
LM_ABBR = {"AbLang2": "AbLang2", "AntiBERTy": "AntiBERTy", "AntiBERTa2": "AntiBERTa2",
           "AntiBERTa2-CSSP": "AB2-CSSP", "IgBert": "IgBert", "One-hot": "One-hot",
           "Biophysical": "Biophys.", "k-mer": "k-mer"}
best_auc = cv["AUC"].max()
best_row = cv.loc[cv["AUC"].idxmax()]


# ── panel e: confusion matrices at Youden threshold (best named model + one-hot) ─
def confusion(y, s, thr):
    pred = (s >= thr).astype(int)             # 1 = predicted Pass
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    # rows = true (Pass, Fail); cols = predicted (Pass, Fail)
    return np.array([[tp, fn], [fp, tn]])


cm_models = [("trans_ablang", "Transformer + AbLang2"), ("trans_one_hot", "Transformer + one-hot")]
cm_data = []
for col, name in cm_models:
    s = val[col].values
    thr = youden_threshold(Y, s)
    cm_data.append((name, thr, confusion(Y, s, thr)))


# ── panel f: REPLICATED learning curves — mean ± 95% CI over repeated
#    subsample + CDR3-cluster split + init per training-set size (TransformerLM + AbLang2).
#    IPI: 17 sizes × 12 reps; DS1: 12 reps (≤26k) / 4 (>26k). Replaces the old single-draw
#    points + quadratic fit, which were noise-dominated below ~2,000.
ipi = pd.read_csv(data_file("learning_curve_ipi_replicated_summary.csv")).sort_values("size")
ds1 = pd.read_csv(data_file("learning_curve_DS1_replicated_summary.csv")).sort_values("size")


# =====================================================================================
# FIGURE
# =====================================================================================
fig = plt.figure(figsize=(ok.DOUBLE, 118 * ok.MM))
gs = GridSpec(2, 3, figure=fig,
              hspace=0.95, wspace=0.34,
              left=0.065, right=0.985, top=0.91, bottom=0.16)

axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[0, 2])
axd = fig.add_subplot(gs[1, 0])
# panel e holds two small confusion heatmaps side by side
gse = gs[1, 1].subgridspec(1, 2, wspace=0.45)
axe1 = fig.add_subplot(gse[0, 0])
axe2 = fig.add_subplot(gse[0, 1])
axf = fig.add_subplot(gs[1, 2])


# ── a: ROC ──────────────────────────────────────────────────────────────────────
axa.plot([0, 1], [0, 1], color=ok.OI_GREY, lw=0.6, ls=(0, (1, 1)), zorder=1)
for name, c, ls, fpr, tpr, auc, lo, hi in roc_data:
    axa.plot(fpr, tpr, color=c, ls=ls, lw=1.1, zorder=3,
             label=f"{name}\nAUC {auc:.3f} ({lo:.3f}–{hi:.3f})")
axa.set_xlim(-0.02, 1.02); axa.set_ylim(-0.02, 1.02)
axa.set_xlabel("False positive rate", labelpad=2)
axa.set_ylabel("True positive rate", labelpad=2)
axa.set_aspect("equal")
axa.legend(loc="lower right", fontsize=4.3, handlelength=1.4, handletextpad=0.4,
           labelspacing=0.55, borderpad=0.2)
axa.set_title("ROC — internal 20% validation", fontsize=6.8, pad=4,
              fontweight="bold", loc="left")


# ── b: Precision-Recall ──────────────────────────────────────────────────────────
base = (Y == 1).mean()
axb.axhline(base, color=ok.OI_GREY, lw=0.6, ls=(0, (1, 1)), zorder=1)
for name, c, ls, rec, prec, ap in pr_data:
    axb.plot(rec, prec, color=c, ls=ls, lw=1.1, zorder=3,
             label=f"{name}\nAP {ap:.3f}")
axb.set_xlim(-0.02, 1.02); axb.set_ylim(0, 1.02)
axb.set_xlabel("Recall (Pass)", labelpad=2)
axb.set_ylabel("Precision (Pass)", labelpad=2)
axb.legend(loc="lower left", fontsize=4.5, handlelength=1.4, handletextpad=0.4,
           labelspacing=0.55, borderpad=0.2)
axb.set_title("Precision–Recall", fontsize=6.8, pad=4, fontweight="bold", loc="left")


# ── c: Calibration ───────────────────────────────────────────────────────────────
axc.plot([0, 1], [0, 1], color=ok.OI_GREY, lw=0.6, ls=(0, (1, 1)), zorder=1,
         label="Perfect calibration")
for name, c, ls, mean_pred, frac_pos in cal_data:
    axc.plot(mean_pred, frac_pos, color=c, ls=ls, lw=1.0, marker="o", ms=2.2,
             mew=0, zorder=3, label=name)
axc.set_xlim(-0.02, 1.02); axc.set_ylim(-0.02, 1.02)
axc.set_xlabel("Mean predicted P(Pass)", labelpad=2)
axc.set_ylabel("Observed Pass fraction", labelpad=2)
axc.set_aspect("equal")
# short labels so the legend fits the empty lower-right corner (below the curves)
_ch, _cl = axc.get_legend_handles_labels()
_cshort = {"Transformer + AbLang2": "Transformer", "CNN + AbLang2": "CNN",
           "XGBoost + AbLang2": "XGBoost", "RF + AbLang2": "RF",
           "Transformer + one-hot": "Transformer (1-hot)"}
axc.legend(_ch, [_cshort.get(l, l) for l in _cl], loc="lower right", fontsize=4.2,
           handlelength=1.3, handletextpad=0.35, labelspacing=0.32, borderpad=0.25,
           frameon=True, framealpha=0.9, edgecolor="#cccccc").get_frame().set_linewidth(0.4)
axc.set_title("Calibration", fontsize=6.8, pad=4, fontweight="bold", loc="left")


# ── d: CV AUC bars ───────────────────────────────────────────────────────────────
x_ticks, x_labels, centres, cursor = [], [], {}, 0.0
BAR_W, IN_GAP, GROUP_GAP = 0.8, 0.18, 0.7
for arch in ARCH_ORDER:
    sub = cv[cv["Architecture"] == arch]
    g_start = cursor
    for _, r in sub.iterrows():
        axd.bar(cursor, r["AUC"] - 0.84, BAR_W, bottom=0.84, color=ok.OI_BLUE,
                zorder=3, lw=0)
        x_ticks.append(cursor); x_labels.append(LM_ABBR.get(r["LM"], r["LM"]))
        cursor += BAR_W + IN_GAP
    centres[arch] = (g_start, cursor - BAR_W - IN_GAP + BAR_W)
    cursor += GROUP_GAP
axd.set_xlim(-0.7, cursor - GROUP_GAP + 0.1)
axd.set_ylim(0.84, 0.975)
axd.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
axd.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
axd.tick_params(axis="y", which="minor", length=1.2, width=0.4)
axd.set_ylabel("10-fold CV AUC", labelpad=2)
axd.grid(axis="y", lw=0.25, alpha=0.4, zorder=0)
axd.set_xticks(x_ticks)
axd.set_xticklabels(x_labels, rotation=90, ha="center", va="top", fontsize=4.4)
axd.tick_params(axis="x", length=0, pad=1)
axd.axhline(best_auc, color=ok.OI_VERMILION, lw=0.7, ls="--", alpha=0.85, zorder=4)
axd.text(-0.6, best_auc + 0.0015,
         f"best CV AUC {best_auc:.3f}  ({best_row['Architecture']}+{best_row['LM']})",
         color=ok.OI_VERMILION, fontsize=4.6, va="bottom", ha="left")
# architecture group brackets under the rotated labels
for arch, (s, e) in centres.items():
    axd.annotate("", xy=(s - 0.35, -0.50), xycoords=("data", "axes fraction"),
                 xytext=(e - BAR_W + 0.35, -0.50), textcoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="-", color="#333333", lw=0.9))
    axd.text((s + e - BAR_W) / 2, -0.555, arch, transform=axd.get_xaxis_transform(),
             ha="center", va="top", fontsize=5.4, fontweight="bold", color="#333333")
axd.set_title("Architecture × language model", fontsize=6.8,
              pad=4, fontweight="bold", loc="left")


# ── e: confusion matrices ────────────────────────────────────────────────────────
def draw_cm(ax, name, thr, cm):
    row_tot = cm.sum(axis=1, keepdims=True)
    pct = cm / row_tot
    ax.imshow(pct, cmap=ok.SEQ, vmin=0, vmax=1, aspect="equal")
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            tc = ok.text_on(pct[i, j], 0, 1, thresh=0.55)
            ax.text(j, i - 0.18, f"{cm[i, j]:d}", ha="center", va="center",
                    fontsize=6.6, fontweight="bold", color=tc)
            ax.text(j, i + 0.20, f"{labels[i][j]}  {pct[i, j]*100:.0f}%",
                    ha="center", va="center", fontsize=4.6, color=tc)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pass", "Fail"], fontsize=4.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Pass", "Fail"], fontsize=4.8, rotation=90, va="center")
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Predicted", fontsize=5.0, labelpad=1)
    short = "T+AbLang2" if "AbLang2" in name else "T+one-hot"
    ax.set_title(f"{short}\nJ thr {thr:.2f}", fontsize=5.2, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)


axe1.set_ylabel("True", fontsize=5.0, labelpad=1)
draw_cm(axe1, *cm_data[0])
draw_cm(axe2, *cm_data[1])
# shared panel-e header (figure coords so it sits above both heatmaps)
bbe = axe1.get_position(); bbe2 = axe2.get_position()
fig.text((bbe.x0 + bbe2.x1) / 2, bbe.y1 + 0.085,
         "Confusion @ Youden-J threshold", fontsize=6.8, fontweight="bold",
         ha="center", va="baseline")


# ── f: replicated learning curves — mean ± 95% CI per training-set size ───────────
for d, c, lab in [(ipi, ok.PASS, "IPI"), (ds1, ok.FAIL, "DS1")]:
    axf.fill_between(d["size"], d["ci_lo"], d["ci_hi"], color=c, alpha=0.22, lw=0, zorder=2)
    axf.plot(d["size"], d["mean_auc"], color=c, lw=1.3, marker="o", ms=2.6,
             zorder=4, label=f"{lab} mean ± 95% CI")
axf.set_xscale("log")
axf.set_xlim(90, 3e5)
axf.set_ylim(0.62, 1.0)
axf.set_xlabel("Training-set size", labelpad=2)
axf.set_ylabel("AUC", labelpad=2)
axf.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
axf.grid(axis="y", lw=0.25, alpha=0.4, zorder=0)
# practical-plateau band: IPI reaches ~0.95 by ~5,000 and plateaus ~0.95–0.96; DS1 higher (~0.98)
axf.axvspan(5000, 6000, color=ok.OI_GREY, alpha=0.18, zorder=1, lw=0)
axf.annotate("IPI plateau ≈0.95–0.96\nfrom ~5,000", xy=(5300, 0.951), xytext=(330, 0.80),
             fontsize=4.8, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6))
axf.legend(loc="lower right", fontsize=4.8, handlelength=1.2, handletextpad=0.4,
           labelspacing=0.4, borderpad=0.25)
axf.set_title("Learning curve — AUC vs training size", fontsize=6.8, pad=4,
              fontweight="bold", loc="left")


# ── panel letters (bold, outside axes) ───────────────────────────────────────────
ok.panel_label(fig, axa, "a", dx=-0.045, dy=0.052, size=9)
ok.panel_label(fig, axb, "b", dx=-0.045, dy=0.052, size=9)
ok.panel_label(fig, axc, "c", dx=-0.045, dy=0.052, size=9)
ok.panel_label(fig, axd, "d", dx=-0.045, dy=0.052, size=9)
ok.panel_label(fig, axe1, "e", dx=-0.060, dy=0.115, size=9)
ok.panel_label(fig, axf, "f", dx=-0.045, dy=0.052, size=9)

ok.save_fig(fig, "Figure3", OUT)

# ── console report ───────────────────────────────────────────────────────────────
print("\n=== Panel a: AUC + 95% bootstrap CI ===")
for name, (auc, lo, hi) in auc_report.items():
    print(f"  {name:24s} AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print("=== Panel b: average precision (AUC-PR) ===")
for name, ap in ap_report.items():
    print(f"  {name:24s} AP {ap:.3f}")
print(f"=== Panel d: best CV arch x LM AUC = {best_auc:.3f} "
      f"({best_row['Architecture']} + {best_row['LM']}) ===")
print("=== Panel e: Youden thresholds + confusion (rows=true, [[TP,FN],[FP,TN]]) ===")
for name, thr, cm in cm_data:
    print(f"  {name:24s} thr {thr:.3f}  TP {cm[0,0]} FN {cm[0,1]} FP {cm[1,0]} TN {cm[1,1]}")
print("Fig3 done")
