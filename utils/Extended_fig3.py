#!/usr/bin/env python3
"""
Extended Data Figure 3 — 4-panel version (a, b, c, d).

Panels:
  a: Correlation heatmap between all 25 models on IPI PSR validation
  b: Correlation heatmap between 6 Transformer models on DS1
  c: KDE of all 25 model scores on IPI val (no label split)
  d: KDE of 6 Transformer model scores on DS1 (no label split)

Labeled score-distribution plots live in Extended Fig 3.

Outputs: .tiff (300 DPI LZW), .pdf (vector), .png (150 DPI)
Deps: pandas numpy matplotlib seaborn scipy openpyxl
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

OUTPUT_STEM = "ED_Fig3_model_correlation_and_kde"
OUTPUT_DIR = Path(".")
FIG_WIDTH_IN = 6.3
FIG_HEIGHT_IN = 7.6
DPI_TIFF = 300
DPI_PNG = 150
CMAP_CORR = "viridis"
KDE_COLORS_25 = sns.color_palette("tab20", 25)

TRANSFORMER_COLORS = {
    'Trans_AbLang2':         '#1f77b4',
    'Trans_IgBert':          '#ff7f0e',
    'Trans_AntiBERTy':       '#2ca02c',
    'Trans_AntiBERTa2':      '#d62728',
    'Trans_AntiBERTa2-CSSP': '#9467bd',
    'Trans_OneHot':          '#8c564b',
    # Non-Transformer AbLang2 models also present on DS1 sheet:
    'CNN_AbLang2':           '#e377c2',
    'XGB_AbLang2':           '#7f7f7f',
    'RF_AbLang2':            '#17becf',
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "legend.fontsize": 5, "figure.titlesize": 9,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
})


def shorten_model_name(col):
    name = re.sub(r'_ipi_psr_trainset_train_score$', '', col)
    arch_map = {
        'transformer_lm': 'Trans', 'transformer_onehot': 'Trans',
        'cnn': 'CNN', 'xgboost': 'XGB', 'rf': 'RF',
    }
    emb_map = {
        'ablang': 'AbLang2', 'igbert': 'IgBert', 'antiberty': 'AntiBERTy',
        'antiberta2': 'AntiBERTa2', 'antiberta2-cssp': 'AntiBERTa2-CSSP',
        'onehot': 'OneHot', 'kmer': 'k-mer', 'biophysical': 'Biophys',
    }
    for arch_prefix in sorted(arch_map.keys(), key=len, reverse=True):
        if name.startswith(arch_prefix + '_'):
            arch = arch_map[arch_prefix]
            rest = name[len(arch_prefix) + 1:]
            return f"{arch}_{emb_map.get(rest, rest)}"
    return name


def load_scores(xlsx_path, sheet):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    score_cols = [c for c in df.columns if c.endswith('_train_score')]
    scores = df[score_cols].copy()
    scores.columns = [shorten_model_name(c) for c in score_cols]
    labels = df['psr_filter'] if 'psr_filter' in df.columns else None
    print(f"  {sheet}: {len(scores):,} × {len(scores.columns)} models")
    return scores, labels


def sort_models(cols):
    arch_order = {'Trans': 0, 'CNN': 1, 'XGB': 2, 'RF': 3}
    emb_order = {'AbLang2': 0, 'IgBert': 1, 'AntiBERTy': 2, 'AntiBERTa2': 3,
                  'AntiBERTa2-CSSP': 4, 'OneHot': 5, 'k-mer': 6, 'Biophys': 7}
    def key(col):
        parts = col.split('_', 1)
        arch = parts[0]; emb = parts[1] if len(parts) > 1 else ''
        return (arch_order.get(arch, 99), emb_order.get(emb, 99), col)
    return sorted(cols, key=key)


def draw_corr(ax, scores, title, annot_fontsize=5, show_annot=True):
    ordered = sort_models(list(scores.columns))
    corr = scores[ordered].corr(method='spearman')

    # Let seaborn draw with cbar=False; we'll add a manual colorbar sized to
    # match the SQUARE heatmap core (not the axes frame).
    hm = sns.heatmap(
        corr, ax=ax, cmap=CMAP_CORR,
        vmin=corr.values.min(), vmax=1.0,
        annot=show_annot, fmt='.2f',
        annot_kws={'fontsize': annot_fontsize},
        square=True, cbar=False,
        linewidths=0.3, linecolor='white',
    )
    ax.set_title(title, pad=4)
    ax.set_xticks(np.arange(len(ordered)) + 0.5)
    ax.set_yticks(np.arange(len(ordered)) + 0.5)
    ax.set_xticklabels(ordered, rotation=90, ha='center')
    ax.set_yticklabels(ordered, rotation=0, va='center')

    # Build a colorbar whose height matches the heatmap core.
    # After seaborn draws with square=True, ax.get_position() still refers to
    # the axes frame, but the data coordinates span [0, N] in both x and y and
    # the visible core is square. We position the cbar in axes-fraction coords
    # so it tracks the axes frame; because seaborn leaves equal padding
    # top/bottom around the square, setting the cbar to the core height means
    # aligning it with the drawn cells.  Easiest: use ax.inset_axes with the
    # height derived from the aspect ratio of the axes bounding box.
    fig_cb = ax.figure
    fig_cb.canvas.draw()
    bbox = ax.get_position()             # axes frame bbox in figure coords
    ax_w = bbox.width
    ax_h = bbox.height
    # Heatmap core is square → side = min(ax_w, ax_h) in figure coords
    # Convert to axes-fraction height
    core_h_ax = min(ax_w, ax_h) / ax_h
    # Place cbar: just right of heatmap core, vertically centered on core
    cax_y0 = (1 - core_h_ax) / 2
    cax = ax.inset_axes([1.04, cax_y0, 0.03, core_h_ax])

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=corr.values.min(), vmax=1.0)
    sm = mpl.cm.ScalarMappable(cmap=CMAP_CORR, norm=norm)
    cbar = fig_cb.colorbar(sm, cax=cax)
    cbar.set_label("Spearman's ρ", fontsize=6)
    cax.tick_params(labelsize=5)


def draw_kde(ax, scores, title, color_map, legend_ncol=5, clip_01=True):
    """Unlabeled KDE — one curve per model."""
    from scipy.stats import gaussian_kde
    ordered = sort_models(list(scores.columns))
    x_grid = np.linspace(0, 1, 500)
    for m in ordered:
        vals = scores[m].dropna().values
        if clip_01: vals = np.clip(vals, 0, 1)
        if len(vals) < 2 or np.std(vals) < 1e-9: continue
        try:
            kde = gaussian_kde(vals, bw_method=0.15)
            ax.plot(x_grid, kde(x_grid), lw=0.9, color=color_map.get(m, 'gray'),
                    label=m, alpha=0.85)
        except Exception: continue
    ax.set_xlabel('Predicted P(Pass)'); ax.set_ylabel('Density')
    ax.set_xlim(0, 1); ax.set_title(title, pad=4); ax.grid(alpha=0.3, lw=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              ncol=legend_ncol, frameon=False, handlelength=1.4,
              columnspacing=0.6, labelspacing=0.3, fontsize=4.5)


def main(xlsx_path: Path):
    print(f"Loading {xlsx_path}")
    ipi_scores, _ = load_scores(xlsx_path, 'ipi_psr_trainset_val')
    ds1_scores, _ = load_scores(xlsx_path, 'DS1')

    # Colour maps
    ipi_colors = {m: KDE_COLORS_25[i % len(KDE_COLORS_25)]
                   for i, m in enumerate(sort_models(list(ipi_scores.columns)))}

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 0.85], height_ratios=[1.6, 1.0],
        hspace=0.40, wspace=0.75,
        left=0.12, right=0.97, top=0.90, bottom=0.10,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_corr(ax_a, ipi_scores,
        title=f'IPI PSR validation set (n={len(ipi_scores):,})\nheld-out from training | {len(ipi_scores.columns)} models',
        annot_fontsize=3.2, show_annot=False)

    draw_corr(ax_b, ds1_scores,
        title=f'DS1 public dataset (n={len(ds1_scores):,})\ncross-library transfer | {len(ds1_scores.columns)} models',
        annot_fontsize=4.5, show_annot=True)

    draw_kde(ax_c, ipi_scores,
        title=f'IPI PSR validation — score distributions ({len(ipi_scores.columns)} models)',
        color_map=ipi_colors, legend_ncol=5)

    draw_kde(ax_d, ds1_scores,
        title=f'DS1 — score distributions ({len(ds1_scores.columns)} models)',
        color_map=TRANSFORMER_COLORS, legend_ncol=3)

    # Letters positioned using figure coords so a/b align regardless of panel
    # heights (panels a and b have different aspect ratios due to square=True
    # heatmap constraint + width_ratios 1.0 : 0.85).
    #
    # For each panel, anchor at (axis_left - x_offset, axis_top + y_offset) in
    # figure-space coords. Same row → same figure-space y → visually aligned.
    fig.canvas.draw()   # ensure axes bboxes are finalized
    def place_letter(ax, letter, dx=0.015, dy=0.012):
        bbox = ax.get_position()
        fig.text(bbox.x0 - dx, bbox.y1 + dy, letter,
                 fontsize=11, fontweight='bold', va='bottom', ha='left')

    place_letter(ax_a, 'a', dx=0.060, dy=0.025)   # heatmap row — clear 2-line title
    place_letter(ax_b, 'b', dx=0.060, dy=0.025)
    place_letter(ax_c, 'c', dx=0.060, dy=0.020)   # KDE row — closer to plot than heatmaps
    place_letter(ax_d, 'd', dx=0.060, dy=0.020)

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
                    default='Suppl_Table2_prediction_score_val.xlsx')
    args = ap.parse_args()
    main(Path(args.xlsx))
