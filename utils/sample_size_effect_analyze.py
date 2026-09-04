#!/usr/bin/env python3
"""
Sample size analysis — Extended Data Figure 4
double-column format:
  - Figure width: 183 mm (double-column)
  - Font: Arial 7 pt axis labels, 6 pt tick labels
  - No panel titles (descriptions in legend text only)
  - Panel letters: bold 8 pt
  - Axis linewidth: 0.5 pt
  - 300 DPI TIFF + PDF (pdf.fonttype=42 for editable text)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind, spearmanr
from pathlib import Path

# ── User-configurable parameters ─────────────────────────────────────────────
csv_file = str(Path("data") / "sample_size" / "learning_curve_ipi_psr_trainset_transformer_lm_ablang_psr_filter_sample_size_100_200.csv")
smoothing_window = 5
low_threshold    = 1000
mid_threshold    = 5000

# ── double-column rcParams ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':         'sans-serif',
    'font.sans-serif':     ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':           7,
    'axes.titlesize':      7,
    'axes.labelsize':      7,
    'xtick.labelsize':     6,
    'ytick.labelsize':     6,
    'legend.fontsize':     6,
    'axes.linewidth':      0.5,
    'xtick.major.width':   0.5,
    'ytick.major.width':   0.5,
    'xtick.major.size':    2,
    'ytick.major.size':    2,
    'xtick.minor.width':   0.3,
    'ytick.minor.width':   0.3,
    'lines.linewidth':     0.8,
    'patch.linewidth':     0.5,
    'grid.linewidth':      0.3,
    'grid.alpha':          0.3,
    'pdf.fonttype':        42,   # editable text in Illustrator
    'ps.fonttype':         42,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_file)
print("Descriptive Statistics:")
print(df.describe())

# ── Correlations ──────────────────────────────────────────────────────────────
pearson_correlations = df.corr(method='pearson')['sample_size'].drop('sample_size')
print("\nPearson Correlations with Sample Size:")
print(pearson_correlations)

spearman_corrs = {}
for col in ['auc', 'accuracy', 'f1_score']:
    corr, pval = spearmanr(df['sample_size'], df[col])
    spearman_corrs[col] = corr
print("\nSpearman Correlations with Sample Size:")
print(pd.Series(spearman_corrs))

# ── Bucketed AUC ──────────────────────────────────────────────────────────────
buckets = {
    f'Low (<={low_threshold})':
        df[df['sample_size'] <= low_threshold]['auc'],
    f'Mid ({low_threshold}–{mid_threshold})':
        df[(df['sample_size'] > low_threshold) & (df['sample_size'] <= mid_threshold)]['auc'],
    f'High (>{mid_threshold})':
        df[df['sample_size'] > mid_threshold]['auc']
}
bucket_means = {k: v.mean() for k, v in buckets.items()}
bucket_stds  = {k: v.std()  for k, v in buckets.items()}

bucket_list = list(buckets.keys())
print("\nT-Tests between consecutive buckets:")
for i in range(len(bucket_list) - 1):
    t_stat, p_val = ttest_ind(buckets[bucket_list[i]], buckets[bucket_list[i+1]], equal_var=False)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{bucket_list[i]} vs {bucket_list[i+1]}: t={t_stat:.2f}, p={p_val:.4f} {sig}")

df['auc_diff']       = df['auc'].diff()
df['sample_size_sq'] = df['sample_size'] ** 2

# ── Regressions ───────────────────────────────────────────────────────────────
metrics                 = ['auc', 'accuracy', 'f1_score']
regression_results      = []
poly_regression_results = []

for metric in metrics:
    y     = df[metric]
    X_lin = sm.add_constant(df['sample_size'])
    mdl   = sm.OLS(y, X_lin).fit()
    regression_results.append({
        'metric':    metric, 'type': 'linear',
        'slope':     mdl.params[1], 'intercept': mdl.params[0],
        'p_value':   mdl.pvalues[1], 'r_squared': mdl.rsquared
    })
    X_poly = sm.add_constant(df[['sample_size', 'sample_size_sq']])
    pmdl   = sm.OLS(y, X_poly).fit()
    poly_regression_results.append({
        'metric':            metric, 'type': 'polynomial',
        'intercept':         pmdl.params[0],
        'slope_linear':      pmdl.params[1],
        'coeff_quadratic':   pmdl.params[2],
        'p_value_linear':    pmdl.pvalues[1],
        'p_value_quadratic': pmdl.pvalues[2],
        'r_squared':         pmdl.rsquared
    })

reg_df      = pd.DataFrame(regression_results)
poly_reg_df = pd.DataFrame(poly_regression_results)
reg_df.to_csv(f"{csv_file}.linear_regression_results.csv",      index=False)
poly_reg_df.to_csv(f"{csv_file}.polynomial_regression_results.csv", index=False)

# ── Colour palette (NB-friendly, colourblind-safe) ────────────────────────────
PALETTE = {
    'auc':       '#0072B2',   # blue
    'accuracy':  '#E69F00',   # amber
    'f1_score':  '#009E73',   # green
}
MARKERS = {'auc': 'o', 'accuracy': 's', 'f1_score': '^'}
LABELS  = {'auc': 'auc', 'accuracy': 'accuracy', 'f1_score': 'f1 score'}

# ── Panel helpers ─────────────────────────────────────────────────────────────
def _draw_panel_a(ax):
    """Performance metrics vs. training set size."""
    for metric in metrics:
        df[f'{metric}_rolling'] = df[metric].rolling(
            window=smoothing_window, min_periods=1).mean()
        ax.plot(df['sample_size'], df[f'{metric}_rolling'],
                marker=MARKERS[metric], markersize=3, linewidth=0.8,
                color=PALETTE[metric], alpha=0.9, label=LABELS[metric])

        # Linear trendline (dashed, same colour, faint)
        r = reg_df[reg_df['metric'] == metric].iloc[0]
        if r['p_value'] < 0.05:
            ax.plot(df['sample_size'],
                    r['intercept'] + r['slope'] * df['sample_size'],
                    ls='--', color=PALETTE[metric], alpha=0.30, lw=0.6)

        # Polynomial trendline (dotted, same colour, faint)
        p = poly_reg_df[poly_reg_df['metric'] == metric].iloc[0]
        if p['p_value_quadratic'] < 0.05:
            ax.plot(df['sample_size'],
                    (p['intercept'] +
                     p['slope_linear']   * df['sample_size'] +
                     p['coeff_quadratic']* df['sample_size_sq']),
                    ls=':', color=PALETTE[metric], alpha=0.30, lw=0.6)

    # Bucket means ± SD (black crosses)
    centres = [low_threshold / 2,
               (low_threshold + mid_threshold) / 2,
               mid_threshold + 1000]
    ax.errorbar(centres, list(bucket_means.values()),
                yerr=list(bucket_stds.values()),
                fmt='x', color='#000000', markersize=5, lw=0.8,
                capsize=3, zorder=5, label='Bucket mean ± s.d.')

    ax.set_xlabel('Training set size',  color='#000000')
    ax.set_ylabel('Metric value',       color='#000000')
    ax.tick_params(colors='#000000')
    ax.grid(True, axis='y')
    for sp in ax.spines.values():
        sp.set_edgecolor('#000000')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc',
              loc='lower right', handlelength=1.5)



def _add_panel_letter(ax, letter):
    """Bold panel letter — NB style (top-left, outside axes)."""
    ax.text(-0.12, 1.06, letter,
            transform=ax.transAxes,
            fontsize=8, fontweight='bold', color='#000000',
            va='bottom', ha='left', clip_on=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Individual figures (kept for supplementary use)
# ─────────────────────────────────────────────────────────────────────────────
for draw_fn, suffix, fsz in [
    (_draw_panel_a, 'performance_vs_sample_size', (89/25.4, 70/25.4)),
    (_draw_panel_b, 'correlation_heatmap',        (89/25.4, 80/25.4)),
]:
    fig, ax = plt.subplots(figsize=fsz)
    draw_fn(ax)
    plt.tight_layout()
    out = f"{csv_file}.{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {Path(out).name}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Combined Extended Data Figure 4
#    Width = 183 mm (NB double-column)
#    NO panel titles — all descriptions in figure legend text
# ─────────────────────────────────────────────────────────────────────────────
FIG_W_MM = 183
FIG_H_MM = 70

fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
gs  = gridspec.GridSpec(
    1, 2, figure=fig,
    left=0.08, right=0.97,
    top=0.88,  bottom=0.18,
    wspace=0.42)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

_draw_panel_a(ax_a)
_draw_panel_b(ax_b)
_add_panel_letter(ax_a, 'a')
_add_panel_letter(ax_b, 'b')

# ── Save ──────────────────────────────────────────────────────────────────────
stem = f"{csv_file}.extended_data_fig4_combined"
for ext, kw in [
    ('.tiff', dict(dpi=300, format='tiff')),
    ('.pdf',  dict()),
    ('.png',  dict(dpi=300)),
]:
    fig.savefig(stem + ext, bbox_inches='tight', **kw)
    print(f"Saved: {Path(stem + ext).name}")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Suggested figure legend (paste into manuscript)
# ─────────────────────────────────────────────────────────────────────────────
print("""
═══════════════════════════════════════════════════════════════
SUGGESTED FIGURE LEGEND (Extended Data Fig. 4)
═══════════════════════════════════════════════════════════════
Extended Data Figure 4 | Effect of training set size on
transformer-LM (ABlang) model performance for PSR prediction.

(a) Classification metrics (AUC, accuracy and F1 score)
as a function of training set size (n = 300–11,265
antibodies). Solid lines show 5-point rolling averages; dashed
and dotted lines indicate significant linear and quadratic
regression trends (p < 0.05), respectively. Crosses denote
mean ± s.d. within three training-size buckets (low ≤1,000;
mid 1,001–5,000; high >5,000).

(b) Pearson correlation matrix between training set size and
all performance metrics. Colour scale: red = positive
correlation, blue = negative correlation.
═══════════════════════════════════════════════════════════════
""")

# ── Scaling efficiency summary ────────────────────────────────────────────────
print("Scaling Efficiency:")
for i, res in enumerate(regression_results):
    poly_res = poly_regression_results[i]
    status   = ("Positive scaling" if res['slope'] > 0 and res['p_value'] < 0.05
                else "No/Weak scaling")
    quad_note = (f" | quadratic coeff={poly_res['coeff_quadratic']:.2e} p={poly_res['p_value_quadratic']:.4f}"
                 if poly_res['p_value_quadratic'] < 0.05 else "")
    print(f"  {res['metric']:12s}: {status} (slope={res['slope']:.3e}, p={res['p_value']:.4f}){quad_note}")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind, spearmanr
from pathlib import Path

# ── User-configurable parameters ─────────────────────────────────────────────
csv_file = str(Path("data") / "sample_size" / "learning_curve_ipi_psr_trainset_transformer_lm_ablang_psr_filter_sample_size_100_200.csv")
smoothing_window = 5
low_threshold    = 1000
mid_threshold    = 5000

# publication-quality style
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_file)
print("Descriptive Statistics:")
print(df.describe())

# ── Correlations ──────────────────────────────────────────────────────────────
pearson_correlations = df.corr(method='pearson')['sample_size'].drop('sample_size')
print("\nPearson Correlations with Sample Size:")
print(pearson_correlations)

spearman_corrs = {}
for col in ['auc', 'accuracy', 'f1_score']:
    corr, pval = spearmanr(df['sample_size'], df[col])
    spearman_corrs[col] = corr
print("\nSpearman Correlations with Sample Size:")
print(pd.Series(spearman_corrs))

# ── Bucketed AUC ──────────────────────────────────────────────────────────────
buckets = {
    f'Low (<={low_threshold})':
        df[df['sample_size'] <= low_threshold]['auc'],
    f'Mid ({low_threshold}-{mid_threshold})':
        df[(df['sample_size'] > low_threshold) & (df['sample_size'] <= mid_threshold)]['auc'],
    f'High (>{mid_threshold})':
        df[df['sample_size'] > mid_threshold]['auc']
}
bucket_means = {k: v.mean() for k, v in buckets.items()}
bucket_stds  = {k: v.std()  for k, v in buckets.items()}
print("\nBucketed AUC Averages:")
for k, mean in bucket_means.items():
    print(f"{k}: {mean:.4f}")

bucket_list = list(buckets.keys())
print("\nT-Tests between consecutive buckets:")
for i in range(len(bucket_list) - 1):
    t_stat, p_val = ttest_ind(buckets[bucket_list[i]], buckets[bucket_list[i+1]], equal_var=False)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{bucket_list[i]} vs {bucket_list[i+1]}: t={t_stat:.2f}, p={p_val:.4f} {sig}")

df['auc_diff'] = df['auc'].diff()

# ── Regressions ───────────────────────────────────────────────────────────────
regression_results      = []
poly_regression_results = []
metrics = ['auc', 'accuracy', 'f1_score']

for metric in metrics:
    X_lin = sm.add_constant(df['sample_size'])
    y     = df[metric]
    model_lin = sm.OLS(y, X_lin).fit()
    regression_results.append({
        'metric':    metric,
        'type':      'linear',
        'slope':     model_lin.params[1],
        'intercept': model_lin.params[0],
        'p_value':   model_lin.pvalues[1],
        'r_squared': model_lin.rsquared
    })

    df['sample_size_sq'] = df['sample_size'] ** 2
    X_poly     = sm.add_constant(df[['sample_size', 'sample_size_sq']])
    model_poly = sm.OLS(y, X_poly).fit()
    poly_regression_results.append({
        'metric':           metric,
        'type':             'polynomial',
        'intercept':        model_poly.params[0],
        'slope_linear':     model_poly.params[1],
        'coeff_quadratic':  model_poly.params[2],
        'p_value_linear':   model_poly.pvalues[1],
        'p_value_quadratic':model_poly.pvalues[2],
        'r_squared':        model_poly.rsquared
    })

reg_df      = pd.DataFrame(regression_results)
poly_reg_df = pd.DataFrame(poly_regression_results)
reg_df.to_csv(f"{csv_file}.linear_regression_results.csv",     index=False)
poly_reg_df.to_csv(f"{csv_file}.polynomial_regression_results.csv", index=False)
print("\nRegression results saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw panel a (line plot)
# ─────────────────────────────────────────────────────────────────────────────
METRIC_COLORS = {
    'auc':       '#1F77B4',
    'accuracy':  '#FF7F0E',
    'f1_score':  '#2CA02C',
}
METRIC_MARKERS = {'auc': 'o', 'accuracy': 's', 'f1_score': '^'}
METRIC_LABELS  = {'auc': 'auc', 'accuracy': 'accuracy', 'f1_score': 'f1 score'}


def _draw_panel_a(ax):
    for i, metric in enumerate(metrics):
        df[f'{metric}_rolling'] = df[metric].rolling(window=smoothing_window, min_periods=1).mean()
        ax.plot(df['sample_size'], df[f'{metric}_rolling'],
                marker=METRIC_MARKERS[metric], markersize=4, linewidth=1.2,
                color=METRIC_COLORS[metric], alpha=0.85,
                label=METRIC_LABELS[metric])

        # Linear trendline (dashed)
        filtered_reg = reg_df[(reg_df['metric'] == metric) & (reg_df['type'] == 'linear')]
        if not filtered_reg.empty:
            reg = filtered_reg.iloc[0]
            if reg['p_value'] < 0.05:
                trend_lin = reg['intercept'] + reg['slope'] * df['sample_size']
                ax.plot(df['sample_size'], trend_lin,
                        linestyle='--', color=METRIC_COLORS[metric], alpha=0.35, linewidth=0.8,
                        label='auc (linear)' if metric == 'auc' else None)

        # Polynomial trendline (dotted)
        filtered_poly = poly_reg_df[poly_reg_df['metric'] == metric]
        if not filtered_poly.empty:
            poly_reg = filtered_poly.iloc[0]
            if poly_reg['p_value_quadratic'] < 0.05:
                trend_poly = (poly_reg['intercept'] +
                              poly_reg['slope_linear']    * df['sample_size'] +
                              poly_reg['coeff_quadratic'] * df['sample_size_sq'])
                ax.plot(df['sample_size'], trend_poly,
                        linestyle=':', color=METRIC_COLORS[metric], alpha=0.35, linewidth=0.8,
                        label='auc (polynomial)' if metric == 'auc' else None)

    # ── AUC-specific annotations ──────────────────────────────────────────────
    auc_lin  = reg_df[reg_df['metric'] == 'auc'].iloc[0]
    auc_poly = poly_reg_df[poly_reg_df['metric'] == 'auc'].iloc[0]

    # Find AUC crossing n where polynomial fit crosses 0.90
    AUC_TARGET = 0.90
    a2  = auc_poly['coeff_quadratic']
    a1  = auc_poly['slope_linear']
    a0  = auc_poly['intercept']
    coeffs = [a2, a1, a0 - AUC_TARGET]
    roots  = np.roots(coeffs)
    real_roots = sorted([r.real for r in roots if np.isreal(r) and r.real > 0])
    n_cross = int(real_roots[0]) if real_roots else None

    if n_cross:
        ax.axvline(n_cross, color='#000000', linewidth=0.8, linestyle='--', alpha=0.6, zorder=4)
        ax.text(n_cross - 120, ax.get_ylim()[0] + 0.02,
                f'n ≈ {n_cross:,}\nAUC > 0.90',
                fontsize=6, color='#000000', va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#999', lw=0.4, alpha=0.85))

    # Annotation box: AUC regression summary
    annot_lines = [
        'auc regression summary:',
        f'  Linear:  slope={auc_lin["slope"]:.2e}  R²={auc_lin["r_squared"]:.2f}',
        f'  Poly:    coeff={auc_poly["coeff_quadratic"]:.2e}  R²={auc_poly["r_squared"]:.2f}',
        f'  Tapering p = {auc_poly["p_value_quadratic"]:.0e}',
    ]
    ax.text(0.02, 0.99, '\n'.join(annot_lines),
            transform=ax.transAxes,
            fontsize=5.0, va='top', ha='left', color='#000000',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#aaaaaa', lw=0.4, alpha=0.95))

    # Bucket means
    bucket_centers = [low_threshold / 2,
                      (low_threshold + mid_threshold) / 2,
                      mid_threshold + 1000]
    ax.errorbar(bucket_centers, list(bucket_means.values()),
                yerr=list(bucket_stds.values()),
                fmt='x', color='black', markersize=6, linewidth=1.0,
                capsize=4, label='Bucket mean ± s.d.', zorder=5)

    ax.set_xlabel('Training set size', fontsize=9, fontweight='bold', color='#000000')
    ax.set_ylabel('Metric value',      fontsize=9, fontweight='bold', color='#000000')
    ax.tick_params(axis='both', labelsize=8, colors='#000000')
    ax.grid(True, alpha=0.25, linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#000000'); sp.set_linewidth(0.8)

    # Legend: metrics + AUC linear/poly trendlines
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6.5, frameon=True, framealpha=0.9,
              edgecolor='#cccccc', loc='lower right', ncol=1,
              title='― metric  -- linear  ··· poly',
              title_fontsize=5.5)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw panel b (correlation heatmap — all columns incl. rolling + diff)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_panel_b(ax):
    # Build column list from whatever exists in df at call time
    _base    = ['sample_size', 'auc', 'accuracy', 'f1_score']
    _extra   = [c for c in ['auc_diff', 'sample_size_sq'] if c in df.columns]
    _rolling = [c for c in df.columns
                if c.endswith('_rolling') and not c.startswith(('precision', 'recall'))]
    _cols    = _base + _extra + _rolling

    nice = {
        'sample_size':       'sample\nsize',
        'auc':               'auc',
        'accuracy':          'accuracy',
        'f1_score':          'f1 score',
        'auc_diff':          'auc\ndiff',
        'sample_size_sq':    'sample\nsize²',
        'auc_rolling':       'auc_rolling',
        'accuracy_rolling':  'accuracy_rolling',
        'f1_score_rolling':  'f1_rolling',

    }

    corr_mat = df[_cols].corr(method='pearson')
    corr_mat.columns = [nice.get(c, c) for c in corr_mat.columns]
    corr_mat.index   = [nice.get(c, c) for c in corr_mat.index]

    mask_diag = np.eye(len(corr_mat), dtype=bool)
    off_diag  = corr_mat.values[~mask_diag]
    # Anchor to actual data range but with enough spread for contrast
    _vmin = 0.0   # start from 0 so full colormap range is used
    _vmax = 1.0

    sns.heatmap(corr_mat, annot=True, cmap='RdYlBu_r',
                fmt='.2f', linewidths=0.3, linecolor='#e0e0e0',
                annot_kws={'size': 6, 'color': '#000000'},
                vmin=_vmin, vmax=_vmax, ax=ax,
                cbar_kws={'shrink': 0.80, 'pad': 0.02})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', color='#000000', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  color='#000000', fontsize=6)
    ax.tick_params(length=0)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Pearson's r", color='#000000', fontsize=7)
    cbar.ax.tick_params(colors='#000000', labelsize=6)
    cbar.outline.set_linewidth(0.5)



# ─────────────────────────────────────────────────────────────────────────────
# 1. Individual figures (no titles — NB style)
# ─────────────────────────────────────────────────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(6, 5))
_draw_panel_a(ax_a)
plt.tight_layout()
out_a = f"{csv_file}.performance_vs_sample_size_enhanced.png"
fig_a.savefig(out_a, dpi=300, bbox_inches='tight')
print(f"\nSaved: {Path(out_a).name}")

fig_b, ax_b = plt.subplots(figsize=(8, 7))
_draw_panel_b(ax_b)
plt.tight_layout()
out_b = f"{csv_file}.correlation_heatmap.png"
fig_b.savefig(out_b, dpi=300, bbox_inches='tight')
print(f"Saved: {Path(out_b).name}")

plt.close('all')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Combined Extended Data Figure 4 — panels a and b, NO titles
# ─────────────────────────────────────────────────────────────────────────────
FIG_W_MM = 183
FIG_H_MM = 70

fig_comb = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
gs = gridspec.GridSpec(1, 2, figure=fig_comb,
                       left=0.08, right=0.97,
                       top=0.88, bottom=0.18,
                       wspace=0.42)

ax_ca = fig_comb.add_subplot(gs[0, 0])
ax_cb = fig_comb.add_subplot(gs[0, 1])

_draw_panel_a(ax_ca)
_draw_panel_b(ax_cb)
_add_panel_letter(ax_ca, 'a')
_add_panel_letter(ax_cb, 'b')

out_comb_png  = f"{csv_file}.extended_data_fig4_combined.png"
out_comb_pdf  = f"{csv_file}.extended_data_fig4_combined.pdf"
out_comb_tiff = f"{csv_file}.extended_data_fig4_combined.tiff"

fig_comb.savefig(out_comb_png,  dpi=300, bbox_inches='tight')
fig_comb.savefig(out_comb_pdf,  bbox_inches='tight')
fig_comb.savefig(out_comb_tiff, dpi=300, bbox_inches='tight', format='tiff')

print(f"\nCombined figure saved:")
print(f"  PNG  → {Path(out_comb_png).name}")
print(f"  PDF  → {Path(out_comb_pdf).name}")
print(f"  TIFF → {Path(out_comb_tiff).name}")
plt.close('all')

# ── Scaling efficiency summary ────────────────────────────────────────────────
print("\nScaling Efficiency (Generalization Check):")
for i, res in enumerate(regression_results):
    if i < len(poly_regression_results):
        poly_res = poly_regression_results[i]
        status   = "Positive scaling" if (res['slope'] > 0 and res['p_value'] < 0.05) else "No/Weak scaling"
        quad_note = (f" (Quadratic tapering: coeff={poly_res['coeff_quadratic']:.2e}, "
                     f"p={poly_res['p_value_quadratic']:.4f})"
                     if poly_res['p_value_quadratic'] < 0.05 else "")
        risk = " — Low memorization risk." if "Positive" in status else " — Potential plateau or memorization."
        print(f"{res['metric']}: {status} (slope={res['slope']:.6e}, p={res['p_value']:.4f}){quad_note}{risk}")
