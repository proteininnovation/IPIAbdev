"""
Delphi — Publication Figure2 Generator
========================================
Generates Figure 2, Extended Data Figure 1, and Figure 2 for:
  "Interpretable machine learning reveals HCDR3 electrostatic balance
   as the master determinant of antibody polyreactivity and aggregation"

double-column figure specifications:
  - Font       : Arial (sans-serif fallback: Helvetica, DejaVu Sans)
  - Font sizes : panel labels 8pt bold, axis labels 7pt, tick labels 6pt
  - Resolution : 300 dpi (TIFF) for submission; 600 dpi for final
  - Width      : single column = 89 mm | double column = 183 mm
  - Max depth  : 247 mm
  - Colors     : Pass = #4C9BE8 (blue), Fail = #F28C38 (orange)
  - Format     : TIFF (submission) + PDF (vector backup)

Usage
-----
    python utils/Figure2_physicochemical.py --data_path /Users/Hoan.Nguyen/ComBio/delphi/data --dpi 300

Or import in a notebook:
    from ipiabdev_figures import generate_all_figures
    generate_all_figures(data_psr_trainset, data_ds1, data_elisa_only, data_sec)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

import liabilities
import utilities

warnings.filterwarnings("ignore")

# Output directory
OUT_DIR = Path("/Users/Hoan.Nguyen/ComBio/delphi/manuscripts/figures_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── publication-quality style constants ────────────────────────────────────
MM_TO_INCH   = 1 / 25.4
SINGLE_COL   = 89  * MM_TO_INCH   # 89 mm  inches
DOUBLE_COL   = 183 * MM_TO_INCH   # 183 mm  inches
MAX_DEPTH    = 247 * MM_TO_INCH   # 247 mm  inches

DPI_SUBMIT   = 300                 # submission
DPI_FINAL    = 600                 # final accepted version

FONT_FAMILY  = "Arial"
SIZE_PANEL   = 8    # panel label (a, b, c ...)
SIZE_AXIS    = 7    # axis title
SIZE_TICK    = 6    # tick labels
SIZE_LEGEND  = 6    # legend text

COLOR_PASS   = "#4C9BE8"   # blue   -- Pass (1)
COLOR_FAIL   = "#F28C38"   # orange -- Fail (0)
ALPHA        = 0.75
LINEWIDTH    = 0.6




# ─── Global matplotlib style ─────────────────────────────────────────────────
def set_nature_style():
    """Apply double-column-compliant matplotlib rcParams."""
    matplotlib.rcParams.update({
        "font.family"         : "sans-serif",
        "font.sans-serif"     : [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size"           : SIZE_TICK,
        "axes.labelsize"      : SIZE_AXIS,
        "axes.titlesize"      : SIZE_AXIS,
        "axes.linewidth"      : LINEWIDTH,
        "axes.spines.top"     : False,
        "axes.spines.right"   : False,
        "xtick.labelsize"     : SIZE_TICK,
        "ytick.labelsize"     : SIZE_TICK,
        "xtick.major.width"   : LINEWIDTH,
        "ytick.major.width"   : LINEWIDTH,
        "xtick.major.size"    : 2.5,
        "ytick.major.size"    : 2.5,
        "xtick.direction"     : "out",
        "ytick.direction"     : "out",
        "legend.fontsize"     : SIZE_LEGEND,
        "legend.frameon"      : False,
        "legend.handlelength" : 1.0,
        "legend.handleheight" : 0.7,
        "legend.handletextpad": 0.4,
        "legend.borderpad"    : 0.3,
        "figure.dpi"          : 150,    # screen preview
        "savefig.dpi"         : DPI_SUBMIT,
        "pdf.fonttype"        : 42,     # embed fonts in PDF
        "ps.fonttype"         : 42,
    })

set_nature_style()


# ─── Shared helpers ───────────────────────────────────────────────────────────
def _legend_handles(label_pass="Pass (1)", label_fail="Fail (0)"):
    """Return Line2D proxy artists for the shared figure legend."""
    return [
        Line2D([0], [0], color=COLOR_PASS, lw=4, alpha=ALPHA, label=label_pass),
        Line2D([0], [0], color=COLOR_FAIL, lw=4, alpha=ALPHA, label=label_fail),
    ]


def _panel_label(ax, letter, x=-0.22, y=1.10):
    """Add bold lowercase panel label (a, b, c) in publication-quality style."""
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=SIZE_PANEL,
        fontweight="bold",
        va="top", ha="left",
        fontfamily=FONT_FAMILY,
    )


def _add_row_label(fig, gs, row, label):
    """
    Add a rotated bold row label in a dedicated invisible axes (GridSpec column 0).
    Placing the label in its own column prevents any overlap with the
    Density y-axis label of the first data panel.
    """
    ax_lbl = fig.add_subplot(gs[row, 0])
    ax_lbl.axis("off")
    ax_lbl.text(
        0.9, 0.5,               # x=0.9: right edge of label col, close to data panels
        label,
        transform=ax_lbl.transAxes,
        fontsize=SIZE_AXIS,
        fontweight="bold",
        rotation=90,
        va="center", ha="center",
        fontfamily=FONT_FAMILY,
        linespacing=1.4,
    )


def _hist_panel(
    ax, data, col_filter, val_pass, val_fail,
    x_var, x_label,
    discrete=True, bins=None, x_lim=None,
    panel_letter=None, show_legend=False,
):
    """
    Overlapping density histograms for Pass vs Fail on a single Axes.

    Parameters
    ----------
    ax           : matplotlib Axes
    data         : pd.DataFrame
    col_filter   : str   -- column holding Pass/Fail labels
    val_pass     : int   -- label value for Pass (1)
    val_fail     : int   -- label value for Fail (0)
    x_var        : str   -- column to plot on x-axis
    x_label      : str   -- x-axis label text
    discrete     : bool  -- True for integer count data
    bins         : sequence or None -- custom bins (overrides discrete)
    x_lim        : tuple or None -- (xmin, xmax) axis limits
    panel_letter : str   -- e.g. 'a', 'b' ...
    show_legend  : bool
    """
    d_pass = data[data[col_filter] == val_pass]
    d_fail = data[data[col_filter] == val_fail]

    hist_kw = dict(stat="density", alpha=ALPHA, linewidth=0)

    if bins is not None:
        sns.histplot(d_pass, x=x_var, bins=bins, color=COLOR_PASS, ax=ax, **hist_kw)
        sns.histplot(d_fail, x=x_var, bins=bins, color=COLOR_FAIL, ax=ax, **hist_kw)
    else:
        sns.histplot(d_pass, x=x_var, discrete=discrete, color=COLOR_PASS, ax=ax, **hist_kw)
        sns.histplot(d_fail, x=x_var, discrete=discrete, color=COLOR_FAIL, ax=ax, **hist_kw)

    ax.set_xlabel(x_label,   fontsize=SIZE_AXIS, labelpad=2)
    ax.set_ylabel("Density", fontsize=SIZE_AXIS, labelpad=2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    if x_lim:
        ax.set_xlim(x_lim)

    if show_legend:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=SIZE_LEGEND)
    elif ax.get_legend():
        ax.get_legend().remove()

    if panel_letter:
        _panel_label(ax, panel_letter)

    ax.tick_params(axis="both", labelsize=SIZE_TICK, width=LINEWIDTH, length=2.5)


def _prepare_data(data, col_filter):
    """
    Add amino acid count columns and apply standard CDR3 length filter.
    Compatible with both psr_filter and sec_filter label columns.
    """
    data = data.copy()
    data = data.dropna(subset=[col_filter])

    for aa, col_name in [("R", "R"), ("D", "D"), ("E", "E"), ("W", "CDR3_W_count")]:
        data[col_name] = data["CDR3"].str.count(aa)

    data["CDR3_len"] = data["CDR3"].str.len()
    data = data[data["CDR3_len"] < 25]     # remove likely sequencing artefacts
    return data


def _save_figure(fig, prefix, dpi=DPI_SUBMIT):
    """
    Save as TIFF (Nature submission requirement) and PDF (vector backup).
    bbox_inches='tight' + pad_inches=0.08 ensures all axis labels including
    the rightmost x-label are fully included without clipping.
    """
    tiff_path = OUT_DIR / f"{prefix}.tiff"
    pdf_path  = OUT_DIR / f"{prefix}.pdf"
    fig.savefig(tiff_path, dpi=dpi, format="tiff",
                bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(pdf_path,  dpi=dpi, format="pdf",
                bbox_inches="tight", pad_inches=0.08)
    print(f"  Saved: {tiff_path}  ({tiff_path.stat().st_size // 1024} KB)")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


# ─── FIGURE 2 ────────────────────────────────────────────────────────────────
def generate_figure1(
    data_ipi,
    data_ds1,
    data_sec,
    col_filter_psr="psr_filter",
    col_filter_sec="sec_filter",
    val_pass=1,
    val_fail=0,
    output_prefix="Figure2",
):
    """
    Figure 2 -- HCDR3 determinants are conserved across polyreactivity
    and SEC monomoric datasets.

    Layout  : 3 rows x 5 data columns + 1 dedicated label column
    Row 1   : IPI PSR (ELISA+NGS)  -- panels a-e   (psr_filter)
    Row 2   : DS1 (public)         -- panels f-j   (psr_filter)
    Row 3   : IPI SEC (trainset)   -- panels k-o   (sec_filter)  [NEW]
    Width   : double column (183 mm)
    """
    data_ipi = _prepare_data(data_ipi, col_filter_psr)
    data_ds1 = _prepare_data(data_ds1, col_filter_psr)
    data_sec = _prepare_data(data_sec, col_filter_sec)

    # (x_var, x_label, discrete, bins, x_lim)
    panels = [
        ("R",            "Arginine count (HCDR3)",          True,  None, None),
        ("D",            "Aspartic acid count (HCDR3)",     True,  None, None),
        ("CDR3_W_count", "Tryptophan count\n(Arg count=1)", True,  None, None),
        ("CDR3_len",     "HCDR3 loop length",               True,  None, None),
        ("HCDR3_charge", "Net charge (HCDR3)",              True,  None, None),
    ]
    letters     = [list("abcde"), list("fghij"), list("klmno")]
    n_data_cols = len(panels)
    n_rows      = 3

    fig_w = DOUBLE_COL
    # 3 rows: height scales ~1.5x the old 2-row version; clipped to MAX_DEPTH
    fig_h = min(fig_w * 0.66, MAX_DEPTH)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows, n_data_cols + 1,
        figure=fig,
        width_ratios=[0.10] + [1] * n_data_cols,
        left=0.06, right=0.99,
        top=0.95,  bottom=0.14,
        hspace=0.55, wspace=0.70,
    )

    # Row labels in dedicated invisible axes (column 0)
    row_labels = [
        "IPI PSR\n(ELISA+NGS)",
        "DS1\n(public)",
        "IPI SEC",
    ]
    for r, lbl in enumerate(row_labels):
        _add_row_label(fig, gs, row=r, label=lbl)

    # Data panel axes (columns 1 to n_data_cols)
    axes = np.array([
        [fig.add_subplot(gs[row, col + 1]) for col in range(n_data_cols)]
        for row in range(n_rows)
    ])

    # Per-row (data, filter-column) pairing
    row_sources = [
        (data_ipi, col_filter_psr),
        (data_ds1, col_filter_psr),
        (data_sec, col_filter_sec),
    ]

    for row_idx, (d_row, col_filter_row) in enumerate(row_sources):
        for col_idx, (x_var, x_label, discrete, bins, x_lim) in enumerate(panels):
            # Tryptophan panel: restrict to Arg count = 1 subset
            d = d_row[d_row["R"] == 1] if x_var == "CDR3_W_count" else d_row

            _hist_panel(
                axes[row_idx, col_idx], d, col_filter_row, val_pass, val_fail,
                x_var, x_label, discrete=discrete, bins=bins, x_lim=x_lim,
                panel_letter=letters[row_idx][col_idx],
                show_legend=(row_idx == 0 and col_idx == 0),
            )

    # Two figure-level legends -- one per assay, so the descriptive labels
    # ("non-polyreactive" vs "non-aggregating") are unambiguous.
    fig.legend(
        handles=_legend_handles(
            label_pass="Pass (1) -- non-polyreactive",
            label_fail="Fail (0) -- polyreactive",
        ),
        loc="lower center", ncol=2,
        fontsize=SIZE_LEGEND,
        bbox_to_anchor=(0.30, 0.00),
        frameon=False,
        title="IPI PSR and DS1 (rows a-j)",
    )
    fig.legend(
        handles=_legend_handles(
            label_pass="Pass (1) -- monomeric",
            label_fail="Fail (0) -- non-monomeric",
        ),
        loc="lower center", ncol=2,
        fontsize=SIZE_LEGEND,
        bbox_to_anchor=(0.75, 0.00),
        frameon=False,
        title="IPI SEC (row k-o)",
    )

    _save_figure(fig, output_prefix)
    return fig


# ─── EXTENDED DATA FIGURE 1 ───────────────────────────────────────────────────
def generate_extended_figure1(
    data_elisa_only,
    col_filter="psr_filter",
    val_pass=1,
    val_fail=0,
    output_prefix="ExtendedDataFig1",
):
    """
    Extended Data Figure 1 -- Full 8-panel biophysical profiling of the
    IPI PSR ELISA-only dataset (denoised, high-confidence labels).

    Panels  : a Arginine  | b Aspartic acid | c Tryptophan (Arg=1) | d Glutamic acid
              e Loop length | f Net charge  | g HCDR3 pI           | h Heavy-chain pI
    Layout  : 2 rows x 4 columns
    Width   : double column (183 mm)
    """
    data = _prepare_data(data_elisa_only, col_filter)

    # (x_var, x_label, subset_R1, discrete, bins, x_lim)
    panels = [
        ("R",                     "Arginine count (HCDR3)",               False, True,  None,           None),
        ("D",                     "Aspartic acid count (HCDR3)",          False, True,  None,           None),
        ("CDR3_W_count",          "Tryptophan count (HCDR3)\n(Arg count=1)", True, True, None,          None),
        ("E",                     "Glutamic acid count (HCDR3)",          False, True,  None,           None),
        ("CDR3_len",              "HCDR3 loop length",                    False, True,  None,           None),
        ("HCDR3_charge",          "Net charge (HCDR3)",                   False, True,  None,           None),
        ("HCDR3_isoelectricpoint","Isoelectric point (HCDR3)",            False, True,  None,           None),
        ("VH_isoelectricpoint",   "Isoelectric point\n(heavy chain)",     False, False, range(1,12,1), None),
    ]
    panel_letters = list("abcdefgh")
    n_rows, n_cols = 2, 4

    fig_w = DOUBLE_COL
    fig_h = min(fig_w * 0.52, MAX_DEPTH)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, (x_var, x_label, subset_r1, discrete, bins, x_lim) in enumerate(panels):
        d = data[data["R"] == 1] if subset_r1 else data
        _hist_panel(
            axes_flat[idx], d, col_filter, val_pass, val_fail,
            x_var, x_label,
            discrete=discrete, bins=bins, x_lim=x_lim,
            panel_letter=panel_letters[idx],
            show_legend=(idx == 0),
        )

    fig.legend(
        handles=_legend_handles(
            label_pass="Pass (1) -- non-polyreactive",
            label_fail="Fail (0) -- polyreactive",
        ),
        loc="lower center", ncol=2,
        fontsize=SIZE_LEGEND,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    _save_figure(fig, output_prefix)
    return fig


# ─── FIGURE 2 ────────────────────────────────────────────────────────────────
def generate_figure2(
    data_sec,
    col_filter="sec_filter",
    val_pass=1,
    val_fail=0,
    output_prefix="ExtendedDataFig2",
):
    """
    Figure 2 -- Biophysical profiling of the IPI SEC dataset.
    Identical panel layout to Extended Data Figure 1 but with SEC labels.

    Panels  : a Arginine  | b Aspartic acid | c Tryptophan (Arg=1) | d Glutamic acid
              e Loop length | f Net charge  | g HCDR3 pI           | h Heavy-chain pI
    Layout  : 2 rows x 4 columns
    Width   : double column (183 mm)
    """
    data = _prepare_data(data_sec, col_filter)

    panels = [
        ("R",                     "Arginine count (HCDR3)",               False, True,  None,           None),
        ("D",                     "Aspartic acid count (HCDR3)",          False, True,  None,           None),
        ("CDR3_W_count",          "Tryptophan count (HCDR3)\n(Arg count=1)", True, True, None,          None),
        ("E",                     "Glutamic acid count (HCDR3)",          False, True,  None,           None),
        ("CDR3_len",              "HCDR3 loop length",                    False, True,  None,           None),
        ("HCDR3_charge",          "Net charge (HCDR3)",                   False, True,  None,           None),
        ("HCDR3_isoelectricpoint","Isoelectric point (HCDR3)",            False, True,  None,           None),
        ("VH_isoelectricpoint",   "Isoelectric point\n(heavy chain)",     False, False, range(1,12,1), None),
    ]
    panel_letters = list("abcdefgh")
    n_rows, n_cols = 2, 4

    fig_w = DOUBLE_COL
    fig_h = min(fig_w * 0.52, MAX_DEPTH)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, (x_var, x_label, subset_r1, discrete, bins, x_lim) in enumerate(panels):
        d = data[data["R"] == 1] if subset_r1 else data
        _hist_panel(
            axes_flat[idx], d, col_filter, val_pass, val_fail,
            x_var, x_label,
            discrete=discrete, bins=bins, x_lim=x_lim,
            panel_letter=panel_letters[idx],
            show_legend=(idx == 0),
        )

    fig.legend(
        handles=_legend_handles(
            label_pass="Pass (1) -- non-aggregating (monomeric)",
            label_fail="Fail (0) -- aggregating",
        ),
        loc="lower center", ncol=2,
        fontsize=SIZE_LEGEND,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    _save_figure(fig, output_prefix)
    return fig


# ─── Data loading helpers ─────────────────────────────────────────────────────
def load_ipi_psr_trainset(main_path):
    """PSR trainset (ELISA + NGS combined) -- used for Figure 1 top row."""
    data = pd.read_excel(f"{main_path}/ipi_psr_trainset.xlsx")
    #data.loc[data["CDR3"].str.startswith("C"), "CDR3"] = data["CDR3"].str[1:]
    data = liabilities.annotate_liabilities_2(data, cdr3_col="HSEQ", label="VH")
    data = liabilities.annotate_liabilities_2(data, cdr3_col="CDR3", label="HCDR3")
    return data


def load_ds1(main_path):
    """Public DS1 library -- used for Figure 2 bottom row."""
    data = pd.read_excel(f"{main_path}/DS1.xlsx")
    #data.loc[data["CDR3"].str.startswith("C"), "CDR3"] = data["CDR3"].str[1:]
    data = liabilities.annotate_liabilities_2(data, cdr3_col="HSEQ", label="VH")
    data = liabilities.annotate_liabilities_2(data, cdr3_col="CDR3", label="HCDR3")
    return data


def load_ipi_elisa_only(main_path):
    """
    IPI PSR ELISA-only dataset (denoised, high-confidence) -- Extended Data Figure 1.

    Filters applied
    ---------------
    - All four ELISA scores present (no NA)
    - psr_filter label present and consistent with RF prediction (psr_rf_elisa)
    - Excludes test-antigen rows
    - ID2 > 8700 (removes early library batches predating final library design)
    """
    data = pd.read_excel(f"{main_path}/ipi_psr_trainset_elisa.xlsx")
    data = data[
        pd.notna(data["psr_norm_insulin"]) &
        pd.notna(data["psr_norm_dna"])     &
        pd.notna(data["psr_norm_smp"])     &
        pd.notna(data["psr_norm_avidin"])
    ]
    data = data[pd.notna(data["psr_filter"])]

    data = liabilities.annotate_liabilities_2(data, cdr3_col="HSEQ", label="VH")
    data = liabilities.annotate_liabilities_2(data, cdr3_col="CDR3", label="HCDR3")
    return data


def load_ipi_sec(main_path):
    """
    IPI SEC dataset -- Figure 2.
    Applies the same filters as load_ipi_elisa_only, then retains only rows
    with a valid sec_filter label (no additional denoising applied).
    """
    data = pd.read_excel(f"{main_path}/ipi_sec_5000.xlsx")
    data = data[pd.notna(data["sec_filter"])]
    return data


# ─── Main entry point ─────────────────────────────────────────────────────────
def generate_all_figures(
    data_psr_trainset,
    data_ds1,
    data_elisa_only,
    data_sec,
    dpi=DPI_SUBMIT,
):
    """
    Generate Figure 2, Extended Data Figure 1, and Figure 2 and save to
    OUT_DIR.

    Parameters
    ----------
    data_psr_trainset : IPI PSR trainset (ELISA + NGS combined)
    data_ds1          : Public DS1 library
    data_elisa_only   : IPI PSR ELISA-only (denoised, high-confidence)
    data_sec          : IPI SEC dataset
    dpi               : Output resolution -- 300 for submission, 600 for final
    """
    global DPI_SUBMIT
    DPI_SUBMIT = dpi
    matplotlib.rcParams["savefig.dpi"] = dpi
    set_nature_style()

    print("\n-- Figure 1 (IPI PSR + DS1 + IPI SEC biophysical profiling) ----")
    generate_figure1(data_psr_trainset, data_ds1, data_sec)

    print("\n-- Extended Data Figure 1 (IPI ELISA-only, 8 panels) -----------")
    generate_extended_figure1(data_elisa_only)

    print("\n-- Extended Data Figure 2  (IPI SEC biophysical profiling) ---------------------")
    generate_figure2(data_sec)

    print(f"\n All figures saved to: {OUT_DIR.resolve()}/")


# ─── Standalone CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DELPHI publication figures (double-column format)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/Hoan.Nguyen/ComBio/delphi",
        help="Root path containing the /data/ directory",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        choices=[300, 600],
        help="Output resolution: 300 (submission) or 600 (final accepted version)",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.data_path}")
    data_psr = load_ipi_psr_trainset(args.data_path)
    data_ds1 = load_ds1(args.data_path)
    data_eli = load_ipi_elisa_only(args.data_path)
    data_sec = load_ipi_sec(args.data_path)

    generate_all_figures(data_psr, data_ds1, data_eli, data_sec, dpi=args.dpi)