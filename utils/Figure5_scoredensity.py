#!/usr/bin/env python3
"""
Figure 5— AbLang2 embedding performance across 4 classifier
architectures on within-distribution (IPI val) vs cross-library (DS1) data.

QUICK START
───────────
  # Default (uses data/Suppl_Table2_prediction_score_val.xlsx)
  python utils/Figure5_scoredensity.py

  # Explicit path
  python utils/Figure5_scoredensity.py \
      data/Suppl_Table2_prediction_score_val.xlsx



Layout: 2 x 5 grid
  Row 1 (IPI val, n=2,253):   Trans_AbLang2, CNN_AbLang2, XGB_AbLang2,
                              RF_AbLang2, Trans_OneHot
  Row 2 (DS1,     n=246,293): Trans_AbLang2, CNN_AbLang2, XGB_AbLang2,
                              RF_AbLang2, Trans_OneHot

Each panel shows:
  - Histogram of predicted P(Pass), split by ground-truth psr_filter
    (Pass blue, Fail red; normalised per class, max bar = 1)
  - AUC in title
  - TWO threshold lines:
      * Youden's J-optimal threshold (black dash-dot)    — standard benchmark
      * DELPHI optimized threshold (green solid)        — from _optimallabel
  - Per-class mean (dashed) and median (dotted)

The DELPHI threshold is back-derived per model as the minimum score
classified as Pass by the _optimallabel column (i.e., the decision boundary
used by the DELPHI threshold-optimisation module).

Outputs: .tiff (300 DPI LZW), .pdf (vector), .png (150 DPI)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

OUTPUT_STEM = "Figure5_ablang2_arch_cross_library"
OUTPUT_DIR = Path(".")

# A4 portrait usable width ≈ 7.09 in (210 mm page − 2×15 mm margins = 180 mm)
# Keep width within this limit so fonts render at actual size in print.
FIG_WIDTH_IN = 6.75
FIG_HEIGHT_IN = 5.55
DPI_TIFF = 300
DPI_PNG = 150

COLOR_PASS = "#4e82bc"
COLOR_FAIL = "#d96c6c"
COLOR_YOUDEN = "#222222"
COLOR_STD = "#b85c00"   # orange-brown for standard 0.5 cutoff

# Panel specification: (score_col, optimallabel_col, sheet, title, letter)
# optimallabel_col=None means no DELPHI threshold (Trans_OneHot on DS1)
PANELS_ROW1_IPI = [
    ("transformer_lm_ablang_ipi_psr_trainset_train_score",
     "transformer_lm_ablang_ipi_psr_trainset_train_optimallabel",
     "Transformer + AbLang2",  "a"),
    ("cnn_ablang_ipi_psr_trainset_train_score",
     "cnn_ablang_ipi_psr_trainset_train_optimallabel",
     "CNN + AbLang2",          "b"),
    ("xgboost_ablang_ipi_psr_trainset_train_score",
     "xgboost_ablang_ipi_psr_trainset_train_optimallabel",
     "XGBoost + AbLang2",      "c"),
    ("rf_ablang_ipi_psr_trainset_train_score",
     "rf_ablang_ipi_psr_trainset_train_optimallabel",
     "Random Forest + AbLang2", "d"),
    ("transformer_onehot_onehot_ipi_psr_trainset_train_score",
     None,
     "Transformer + OneHot",   "e"),
]
PANELS_ROW2_DS1 = [
    ("transformer_lm_ablang_ipi_psr_trainset_train_score",
     "transformer_lm_ablang_ipi_psr_trainset_train_optimallabel",
     "Transformer + AbLang2",  "f"),
    ("cnn_ablang_ipi_psr_trainset_train_score",
     None,   # CNN_AbLang2 optimallabel is not in DS1 sheet — will skip DELPHI threshold line
     "CNN + AbLang2",          "g"),
    ("xgboost_ablang_ipi_psr_trainset_train_score",
     None,
     "XGBoost + AbLang2",      "h"),
    ("rf_ablang_ipi_psr_trainset_train_score",
     None,
     "Random Forest + AbLang2", "i"),
    ("transformer_onehot_onehot_ipi_psr_trainset_train_score",
     None,
     "Transformer + OneHot",   "j"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.5,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.4,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 1.8,
    "ytick.major.size": 1.8,
})


def youden_threshold(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def accuracy_at_threshold(labels, scores, threshold):
    """Classification accuracy at a given decision threshold."""
    preds = (scores >= threshold).astype(int)
    return float((preds == labels).mean())


STANDARD_THRESHOLD = 0.5   # standard binary-classification cutoff


def draw_panel(ax, scores, labels, title, opt_labels=None, show_auc_only=False):
    """Single panel: histogram + stats + two threshold lines (standard 0.5 + Youden).

    opt_labels kept in signature for API compatibility but unused.
    """
    # AUC
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float('nan')

    # Split
    pass_scores = scores[labels == 1]
    fail_scores = scores[labels == 0]
    n_pass = len(pass_scores)
    n_fail = len(fail_scores)

    # Youden threshold + accuracies at both thresholds
    t_youden = youden_threshold(labels, scores)
    acc_std = accuracy_at_threshold(labels, scores, STANDARD_THRESHOLD)
    acc_youden = accuracy_at_threshold(labels, scores, t_youden)

    # Histogram bins: 25 bins over [0, 1]
    bins = np.linspace(0, 1, 26)
    pass_counts, _ = np.histogram(np.clip(pass_scores, 0, 1), bins=bins)
    fail_counts, _ = np.histogram(np.clip(fail_scores, 0, 1), bins=bins)
    pass_norm = pass_counts / max(pass_counts.max(), 1)
    fail_norm = fail_counts / max(fail_counts.max(), 1)
    widths = np.diff(bins)
    centers = (bins[:-1] + bins[1:]) / 2

    # Draw histograms (Fail behind, Pass on top, both with transparency)
    ax.bar(centers, fail_norm, width=widths, color=COLOR_FAIL, alpha=0.75,
           edgecolor='none')
    ax.bar(centers, pass_norm, width=widths, color=COLOR_PASS, alpha=0.65,
           edgecolor='none')

    # Two threshold lines only: standard 0.5 (orange) + Youden J (black dash-dot)
    ax.axvline(STANDARD_THRESHOLD, color=COLOR_STD, linestyle='-', lw=1.3, alpha=0.9)
    ax.axvline(t_youden, color=COLOR_YOUDEN, linestyle='-.', lw=1.1, alpha=0.95)

    # Threshold text labels above the axes
    ymax = 1.18
    ax.set_ylim(0, ymax)
    ax.text(STANDARD_THRESHOLD, 1.09, f'std=0.50',
            color=COLOR_STD, ha='center', va='bottom', fontsize=5,
            fontweight='bold')
    ax.text(t_youden, 1.00, f'Y={t_youden:.2f}',
            color=COLOR_YOUDEN, ha='center', va='bottom', fontsize=5)

    # Title — keep short: just the model name
    ax.set_title(title, pad=3, fontsize=6.5, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, lw=0.3, axis='y')

    # Combined stats + counts box — centered vertically on the y-axis to avoid
    # covering the Youden threshold label (which sits at the top of the panel)
    # and to stay clear of the histogram bars (which sit at the bottom-left and
    # bottom-right edges for bimodal distributions).
    stats_str = (f"AUC   = {auc:.3f}\n"
                 f"Acc.5  = {acc_std:.3f}\n"
                 f"Acc_Y = {acc_youden:.3f}\n"
                 f"Pass  = {n_pass:,}\n"
                 f"Fail   = {n_fail:,}")
    ax.text(0.04, 0.52, stats_str,
            transform=ax.transAxes, fontsize=5.2,
            va='center', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.92, linewidth=0.4))

    # (Pass/Fail counts are included in the combined stats box above)


def build_figure(xlsx_path: Path):
    print(f"Loading {xlsx_path}")
    df_ipi = pd.read_excel(xlsx_path, sheet_name='ipi_psr_trainset_val')
    df_ds1 = pd.read_excel(xlsx_path, sheet_name='DS1')
    print(f"  IPI val: n={len(df_ipi):,}")
    print(f"  DS1:     n={len(df_ds1):,}")

    # ── Figure
    fig, axes = plt.subplots(2, 5, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                              sharex=True)

    # Row 1: IPI val
    labels_ipi = df_ipi['psr_filter'].values.astype(int)
    for ax, (score_col, opt_col, title, letter) in zip(axes[0], PANELS_ROW1_IPI):
        if score_col not in df_ipi.columns:
            ax.text(0.5, 0.5, f"MISSING\n{score_col}", ha='center', va='center',
                    transform=ax.transAxes, color='red', fontsize=6)
            ax.set_axis_off()
            continue
        scores = df_ipi[score_col].values
        opt_labels = (df_ipi[opt_col].values
                      if (opt_col and opt_col in df_ipi.columns) else None)
        mask = ~np.isnan(scores) & ~np.isnan(labels_ipi)
        draw_panel(ax, scores[mask], labels_ipi[mask], title,
                   opt_labels=(opt_labels[mask] if opt_labels is not None else None))
        ax.text(-0.08, 1.15, letter, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')

    # Row 2: DS1
    labels_ds1 = df_ds1['psr_filter'].values.astype(int)
    for ax, (score_col, opt_col, title, letter) in zip(axes[1], PANELS_ROW2_DS1):
        if score_col not in df_ds1.columns:
            ax.text(0.5, 0.5, f"MISSING\n{score_col}", ha='center', va='center',
                    transform=ax.transAxes, color='red', fontsize=6)
            ax.set_axis_off()
            continue
        scores = df_ds1[score_col].values
        opt_labels = (df_ds1[opt_col].values
                      if (opt_col and opt_col in df_ds1.columns) else None)
        mask = ~np.isnan(scores) & ~np.isnan(labels_ds1)
        draw_panel(ax, scores[mask], labels_ds1[mask], title,
                   opt_labels=(opt_labels[mask] if opt_labels is not None else None))
        ax.text(-0.08, 1.15, letter, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')

    # ── Row labels on the left
    axes[0, 0].set_ylabel('IPI validation\n(within-distribution)\n\nRel. frequency',
                          fontsize=6.5)
    axes[1, 0].set_ylabel('DS1 public dataset\n(cross-library transfer)\n\nRel. frequency',
                          fontsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel('Predicted P(Pass)', fontsize=6.5)

    # ── Shared legend below the whole figure
    legend_elements = [
        Patch(facecolor=COLOR_PASS, alpha=0.65, label='Pass (psr_filter = 1)'),
        Patch(facecolor=COLOR_FAIL, alpha=0.75, label='Fail (psr_filter = 0)'),
        Line2D([0], [0], color=COLOR_STD, linestyle='-', lw=1.3,
               label='Standard threshold = 0.5'),
        Line2D([0], [0], color=COLOR_YOUDEN, linestyle='-.', lw=1.1,
               label="Youden J-optimal threshold"),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.0),
               ncol=4, frameon=False, fontsize=6,
               handlelength=1.8, columnspacing=1.5)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14,
                        hspace=0.45, wspace=0.30)

    # ── Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_tiff = OUTPUT_DIR / f"{OUTPUT_STEM}.tiff"
    out_pdf  = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    out_png  = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(out_tiff, dpi=DPI_TIFF, format='tiff',
                pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(out_png, dpi=DPI_PNG, format='png', bbox_inches='tight')
    print(f"\nSaved:")
    print(f"  {out_tiff}  ({out_tiff.stat().st_size/1024:.1f} KB)")
    print(f"  {out_pdf}   ({out_pdf.stat().st_size/1024:.1f} KB)")
    print(f"  {out_png}   ({out_png.stat().st_size/1024:.1f} KB)")
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('xlsx', nargs='?',
                    default='data/Suppl_Table2_prediction_score_val.xlsx')
    args = ap.parse_args()
    build_figure(Path(args.xlsx))