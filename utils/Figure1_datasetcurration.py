"""
Delphi — Dataset Summary Figure (double-column format)
================================================================
Single composite figure:
  Panel a  : Dataset summary table (colour-coded header + Pass/Fail tinted cells)
  Panel b  : IPI PSR normalized scores — 4 antigens × Pass/Fail boxplots
  Panel c  : Spearman correlation heatmap of the 4 PSR antigens
  Panel d  : IPI SEC Retention Time    — Pass vs Fail boxplot

Style
-----
  Font       : Arial, panel labels 8pt bold, axis 7pt, ticks 6pt, table 6pt
  Colors     : Pass = #4C9BE8 (blue), Fail = #F28C38 (orange)
               Table header = #1F3A5F (deep navy)
  Width      : double column = 183 mm
  Format     : TIFF (LZW) + PDF

Usage
-----
    python utils/Figure1_datasetcurration.py \
        --elisa_path /Users/Hoan.Nguyen/ComBio/delphi/manuscripts/data/elisa_score_figure1.xlsx  \
        --sec_path   /Users/Hoan.Nguyen/ComBio/delphi/manuscripts/data/sec_retention_time_figure1.xlsx \
        --output_prefix Figure1 \
        --dpi 300
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu

warnings.filterwarnings("ignore")


OUT_DIR = Path("/Users/Hoan.Nguyen/ComBio/delphi/manuscripts/figures_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── publication-quality style constants ────────────────────────────────────
MM_TO_INCH  = 1 / 25.4
SINGLE_COL  = 89  * MM_TO_INCH
DOUBLE_COL  = 183 * MM_TO_INCH
MAX_DEPTH   = 247 * MM_TO_INCH

DPI_SUBMIT  = 300
DPI_FINAL   = 600

FONT_FAMILY = "Arial"
SIZE_PANEL  = 8    # panel letters
SIZE_AXIS   = 7
SIZE_TICK   = 6
SIZE_LEGEND = 6
SIZE_TABLE  = 6    # table body
SIZE_THEAD  = 6    # table header (kept at 6 so headers fit in narrow cols)

COLOR_PASS  = "#4C9BE8"   # blue   — Pass (1)
COLOR_FAIL  = "#F28C38"   # orange — Fail (0)
ALPHA       = 0.85
LINEWIDTH   = 0.6

# Table palette
TBL_HEADER_BG  = "#1F3A5F"   # deep navy
TBL_HEADER_FG  = "#FFFFFF"
TBL_ROW_A      = "#FFFFFF"   # even rows
TBL_ROW_B      = "#F3F6FA"   # odd rows (very pale blue-grey)
TBL_PASS_TINT  = "#E8F1FB"   # pale blue  (Pass column data cells)
TBL_FAIL_TINT  = "#FCEFE0"   # pale peach (Fail column data cells)
TBL_RULE       = "#C9D2DC"   # horizontal rule colour




def set_nature_style():
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
        "figure.dpi"          : 150,
        "savefig.dpi"         : DPI_SUBMIT,
        "pdf.fonttype"        : 42,
        "ps.fonttype"         : 42,
    })

set_nature_style()


# ─── Dataset summary table content ───────────────────────────────────────────
# Each row: (Dataset, Total, Pass, Fail, Clusters@80%, Singletons, Library)
TABLE_HEADERS = [
    "Dataset",
    "Total",
    "Pass (n)",
    "Fail (n)",
    "Heavy CDR3\nclusters\n(80% identity)",
    "Heavy CDR3\nsingletons",
    "Experimental library",
]

TABLE_ROWS = [
    ["IPI PSR-ELISA",           "7,494",   "5,925",   "1,569",   "5,046",  "3,895",
     "IPI PSR-ELISA assay (DNA, insulin,\navidin, SMP/ovalbumin)"],
    ["IPI PSR-NGS-ssDNA",       "3,771",   "0",       "3,771",   "2,291",  "1,648",
     "NGS ssDNA"],
    ["IPI PSR train set",       "11,265",  "5,925",   "5,340",   "7263",  "5443",
     "IPI PSR-ELISA + NGS-ssDNA"],
    ["IPI SEC train set",       "5045",   "3210",   "1835",   "3272",  "2468",
     "SEC-HPLC assay"],
    ["DS1 (public dataset #1)", "246,293", "131,255", "115,038", "6,311",  "1,665",
     "Cell Reports, Chen et al. (2024)"],
]

# Relative column widths (must sum to 1.0)
COL_WIDTHS = [0.18, 0.07, 0.08, 0.08, 0.14, 0.14, 0.31]

# Column alignments: 'l' = left, 'r' = right, 'c' = center
COL_ALIGN = ['l', 'r', 'r', 'r', 'r', 'r', 'l']


def _draw_table(ax, headers, rows, col_widths, col_align,
                header_height=0.20, row_height=0.16):
    """
    Draw a styled dataset-summary table using Rectangles + text on a
    normalized (0-1) axes area.  The axes itself is made invisible.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_rows = len(rows)
    total_h = header_height + n_rows * row_height
    # Anchor to the top of the axes; any excess space falls below.
    y_top = 1.0
    y_bot = max(0.0, y_top - total_h)

    # Column x-edges
    x_edges = np.concatenate(([0.0], np.cumsum(col_widths)))

    # Identify tinted data columns
    pass_col = 2   # "Pass (n)"
    fail_col = 3   # "Fail (n)"

    # ── Header row ────────────────────────────────────────────────────────────
    y_header = y_top - header_height
    ax.add_patch(Rectangle(
        (0, y_header), 1.0, header_height,
        facecolor=TBL_HEADER_BG, edgecolor="none", zorder=1,
    ))
    for i, h in enumerate(headers):
        xc = x_edges[i] + col_widths[i] / 2
        ax.text(
            xc, y_header + header_height / 2, h,
            ha="center", va="center",
            color=TBL_HEADER_FG, fontsize=SIZE_THEAD,
            fontweight="bold", fontfamily=FONT_FAMILY,
            linespacing=1.1, zorder=3,
        )

    # ── Data rows ─────────────────────────────────────────────────────────────
    for r, row in enumerate(rows):
        y = y_header - (r + 1) * row_height
        base_color = TBL_ROW_B if r % 2 else TBL_ROW_A

        # Full-row background
        ax.add_patch(Rectangle(
            (0, y), 1.0, row_height,
            facecolor=base_color, edgecolor="none", zorder=1,
        ))
        # Tint Pass / Fail columns
        for col_idx, tint in [(pass_col, TBL_PASS_TINT), (fail_col, TBL_FAIL_TINT)]:
            ax.add_patch(Rectangle(
                (x_edges[col_idx], y), col_widths[col_idx], row_height,
                facecolor=tint, edgecolor="none", zorder=2,
            ))
        # Thin horizontal rule above each row (except the first)
        if r > 0:
            ax.plot([0, 1.0], [y + row_height, y + row_height],
                    color=TBL_RULE, lw=0.4, zorder=3)

        # Cell text
        for i, cell in enumerate(row):
            align = col_align[i]
            pad   = 0.008
            if align == 'l':
                x_text, ha = x_edges[i] + pad,                       "left"
            elif align == 'r':
                x_text, ha = x_edges[i] + col_widths[i] - pad,       "right"
            else:
                x_text, ha = x_edges[i] + col_widths[i] / 2,         "center"
            ax.text(
                x_text, y + row_height / 2, cell,
                ha=ha, va="center",
                fontsize=SIZE_TABLE, fontfamily=FONT_FAMILY,
                linespacing=1.1, zorder=4,
            )

    # Top and bottom dark rules for clean framing
    ax.plot([0, 1.0], [y_top, y_top],   color=TBL_HEADER_BG, lw=0.8, zorder=5)
    ax.plot([0, 1.0], [y_bot, y_bot],   color=TBL_HEADER_BG, lw=0.8, zorder=5)


# ─── Box-plot helpers ─────────────────────────────────────────────────────────
def _style_boxplot(ax, xlabel, ylabel, panel_letter=None, panel_title=None,
                   show_legend=False, legend_labels=("Pass (1)", "Fail (0)")):
    ax.set_xlabel(xlabel, fontsize=SIZE_AXIS, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=SIZE_AXIS, labelpad=2)
    ax.tick_params(axis="both", labelsize=SIZE_TICK, width=LINEWIDTH, length=2.5)
    if ax.get_legend():
        ax.get_legend().remove()
    if show_legend:
        handles = [
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=COLOR_PASS, markeredgecolor="none",
                   markersize=6, alpha=ALPHA, label=legend_labels[0]),
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=COLOR_FAIL, markeredgecolor="none",
                   markersize=6, alpha=ALPHA, label=legend_labels[1]),
        ]
        ax.legend(handles=handles, loc="upper right",
                  fontsize=SIZE_LEGEND, frameon=False)
    if panel_letter:
        ax.text(
            -0.14, 1.06, panel_letter,
            transform=ax.transAxes,
            fontsize=SIZE_PANEL, fontweight="bold",
            va="top", ha="left", fontfamily=FONT_FAMILY,
        )
        if panel_title:
            # Title placed immediately to the right of the panel letter
            ax.text(
                -0.04, 1.06, panel_title,
                transform=ax.transAxes,
                fontsize=SIZE_AXIS, fontweight="bold",
                va="top", ha="left", fontfamily=FONT_FAMILY,
            )


def _boxplot_grouped(ax, df, x, y, hue, order, palette,
                     showfliers=False, showmeans=False):
    """Seaborn grouped boxplot with Nature-consistent thin lines."""
    sns.boxplot(
        data=df, x=x, y=y, hue=hue, order=order,
        palette=palette,
        ax=ax,
        showfliers=showfliers,
        showmeans=showmeans,
        meanprops={
            "marker"         : "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize"     : 2.5,
            "markeredgewidth": 0.5,
        },
        width=0.7,
        linewidth=0.6,
        fliersize=1.0,
        boxprops   ={"alpha": ALPHA, "edgecolor": "black"},
        medianprops={"color" : "black", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.5},
        capprops    ={"linewidth": 0.5},
    )


# ─── Significance brackets and correlation heatmap ───────────────────────────
def _pval_stars(p):
    """Convert a p-value to the conventional significance-star notation."""
    if   p < 1e-4: return "****"
    elif p < 1e-3: return "***"
    elif p < 1e-2: return "**"
    elif p < 5e-2: return "*"
    else:          return "ns"


def _sig_bracket(ax, x1, x2, y, height, p_value,
                 lw=0.5, fontsize=6):
    """
    Draw a significance bracket spanning [x1, x2] at y, with a short vertical
    tick of `height` at each end and the significance-star string centred
    above the bracket.
    """
    stars = _pval_stars(p_value)
    ax.plot(
        [x1, x1, x2, x2],
        [y,  y + height, y + height, y],
        color="black", lw=lw,
        solid_capstyle="butt", zorder=6,
    )
    ax.text(
        (x1 + x2) / 2, y + height, stars,
        ha="center", va="bottom",
        fontsize=fontsize, fontfamily=FONT_FAMILY,
        color="black", zorder=6,
    )


def _draw_correlation_heatmap(ax, df, cols, col_labels,
                              cmap_name="Blues", vmin=0.5, vmax=1.0):
    """
    Compact Spearman correlation heatmap for a set of numeric columns.
    Values annotated in each cell; diagonal forced to 1.00.
    Panel letter + title should be added by the caller (via _panel_header).
    """
    rho = df[cols].corr(method="spearman").values
    n = rho.shape[0]

    ax.imshow(rho, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="equal")
    for i in range(n):
        for j in range(n):
            v = rho[i, j]
            norm = (v - vmin) / max(vmax - vmin, 1e-9)
            text_col = "white" if norm > 0.65 else "black"
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=SIZE_TICK, color=text_col,
                fontfamily=FONT_FAMILY,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(col_labels, fontsize=SIZE_TICK, rotation=0)
    ax.set_yticklabels(col_labels, fontsize=SIZE_TICK, rotation=0)
    ax.tick_params(axis="both", which="major", length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel_header(ax, letter, title="", y=1.03, x_letter=-0.14, x_title=-0.04):
    """
    Place bold panel letter + (optional, possibly multi-line) title to its
    right, both sitting on the same baseline above the axes.
    """
    ax.text(
        x_letter, y, letter, transform=ax.transAxes,
        fontsize=SIZE_PANEL, fontweight="bold",
        va="bottom", ha="left", fontfamily=FONT_FAMILY,
    )
    if title:
        ax.text(
            x_title, y, title, transform=ax.transAxes,
            fontsize=SIZE_AXIS, fontweight="bold",
            va="bottom", ha="left", fontfamily=FONT_FAMILY,
            linespacing=1.1,
        )


def _panel_header_stacked(ax, letter, top_line, bottom_line,
                          y_bottom=1.03, line_gap=0.10,
                          x_letter=-0.14, x_title=-0.04):
    """
    Two-line panel header where the panel letter sits at the *beginning*
    of the TOP line (same line as the dataset prefix), e.g.

        c  IPI PSR-ELISA
           antigen correlation
    """
    y_top = y_bottom + line_gap
    # Top line: bold letter + dataset prefix
    ax.text(
        x_letter, y_top, letter, transform=ax.transAxes,
        fontsize=SIZE_PANEL, fontweight="bold",
        va="bottom", ha="left", fontfamily=FONT_FAMILY,
    )
    ax.text(
        x_title, y_top, top_line, transform=ax.transAxes,
        fontsize=SIZE_AXIS, fontweight="bold",
        va="bottom", ha="left", fontfamily=FONT_FAMILY,
    )
    # Bottom line: specific-metric name (no letter)
    ax.text(
        x_title, y_bottom, bottom_line, transform=ax.transAxes,
        fontsize=SIZE_AXIS, fontweight="bold",
        va="bottom", ha="left", fontfamily=FONT_FAMILY,
    )


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_elisa(path):
    df = pd.read_excel(path)
    df = df.dropna(subset=["psr_filter"])
    df["psr_filter"] = df["psr_filter"].astype(int)
    for c in ["psr_norm_dna", "psr_norm_avidin", "psr_norm_insulin", "psr_norm_smp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pick_main_peak(rt_raw, pa_raw):
    """
    Extract the main-peak (Retention Time, Peak Area %) pair from a SEC row.

    SEC cells may contain either a single number (pure-monomer antibody, one
    peak) or a comma-separated list (aggregating antibody, multiple peaks),
    e.g. Retention Time '3.313, 3.55' paired with Peak Area '46.54, 53.46'.
    The convention is that the peak with the largest area is the 'main peak'
    (usually the monomer), and its area % equals the conventional monomer
    purity metric.  We return the RT and PA of that max-area peak.

    Returns (np.nan, np.nan) if either entry is missing or unparseable.
    """
    if pd.isna(rt_raw) or pd.isna(pa_raw):
        return np.nan, np.nan
    try:
        rts = [float(x.strip()) for x in str(rt_raw).split(",") if x.strip()]
        pas = [float(x.strip()) for x in str(pa_raw).split(",") if x.strip()]
        if len(rts) != len(pas) or len(pas) == 0:
            return np.nan, np.nan
        idx = int(np.argmax(pas))
        return rts[idx], pas[idx]
    except (ValueError, TypeError):
        return np.nan, np.nan


def load_sec(path):
    """
    Load the SEC dataset and resolve multi-peak rows by selecting the main
    (largest-area) peak per antibody.  All 82 multi-peak rows in the current
    file belong to the Fail class (sec_filter = 0) — as expected, since
    aggregation is what produces multiple peaks — and are the most
    diagnostic examples of the aggregation signal.
    """
    df = pd.read_excel(path)
    df["sec_filter"] = pd.to_numeric(df["sec_filter"], errors="coerce")
    df = df.dropna(subset=["sec_filter"])
    df["sec_filter"] = df["sec_filter"].astype(int)

    rts, pas = [], []
    for rt_raw, pa_raw in zip(df["retention_time_mins"], df["Peak Area Percent"]):
        rt, pa = _pick_main_peak(rt_raw, pa_raw)
        rts.append(rt)
        pas.append(pa)
    df["Retention Time_SEC"] = rts
    df["Peak Area Percent"]  = pas
    return df


# ─── Figure builder ───────────────────────────────────────────────────────────
def generate_dataset_summary_figure(
    elisa_df,
    sec_df,
    output_prefix="DatasetSummary",
    dpi=DPI_SUBMIT,
):
    """
    Build the composite dataset-summary figure and save TIFF + PDF.
    """
    fig_w = DOUBLE_COL
    fig_h = min(fig_w * 0.70, MAX_DEPTH)   # ~5.0" for 7.2" wide
    fig = plt.figure(figsize=(fig_w, fig_h))

    # 2 rows × 3 cols.  Row 0 = table (spans all 3 cols).  Row 1 = 3 panels:
    #   col 0  panel b  PSR boxplot (wider)
    #   col 1  panel c  antigen Spearman heatmap
    #   col 2  panel d  SEC retention time
    gs = GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.15, 1.0],
        width_ratios =[1.7, 1.3, 1.15],
        left=0.05, right=0.995,
        top=0.94,  bottom=0.11,
        hspace=0.55, wspace=0.42,
    )

    # ── Panel a: table ────────────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[0, :])
    _draw_table(
        ax_table, TABLE_HEADERS, TABLE_ROWS, COL_WIDTHS, COL_ALIGN,
        header_height=0.24, row_height=0.145,
    )
    # Align panel-a letter at the same absolute figure-x as panel b.  The
    # table axes is ~3.1× wider than panel b in the 3-column layout, so to
    # land at the same figure-x the axes-fraction offset must be scaled down
    # proportionally: x_a ≈ -0.14 × (panel_b_width / table_width) ≈ -0.044.
    _panel_header(
        ax_table,
        letter="a",
        title="Polyreactivity and monomer purity dataset curation",
        y=1.02, x_letter=-0.044, x_title=-0.022,
    )

    # ── Panel b: PSR 4 antigens × Pass/Fail boxplot ──────────────────────────
    ax_psr = fig.add_subplot(gs[1, 0])
    psr_long = elisa_df.melt(
        id_vars   = ["psr_filter"],
        value_vars= ["psr_norm_dna", "psr_norm_avidin",
                     "psr_norm_insulin", "psr_norm_smp"],
        var_name  = "Antigen",
        value_name= "Normalized PSR score",
    )
    pretty = {
        "psr_norm_dna"    : "DNA",
        "psr_norm_avidin" : "Avidin",
        "psr_norm_insulin": "Insulin",
        "psr_norm_smp"    : "SMP",
    }
    psr_long["Antigen"] = psr_long["Antigen"].map(pretty)
    order = ["DNA", "Avidin", "Insulin", "SMP"]
    _boxplot_grouped(
        ax_psr, psr_long,
        x="Antigen", y="Normalized PSR score", hue="psr_filter",
        order=order,
        palette={1: COLOR_PASS, 0: COLOR_FAIL},
        showmeans=True,
    )
    q_lo, q_hi = psr_long["Normalized PSR score"].quantile([0.01, 0.99])
    pad = 0.15 * (q_hi - q_lo)
    ax_psr.set_ylim(q_lo - pad, q_hi + pad)

    _style_boxplot(
        ax_psr,
        xlabel="",
        ylabel="Normalized PSR score",
        panel_letter=None,          # header added explicitly below
        show_legend=False,
    )
    _panel_header(ax_psr, letter="b", title="IPI PSR-ELISA data", y=1.13)
    # Pass / Fail legend at the top-right INSIDE the axes — title on top-left
    # and legend on top-right share the top band without overlapping.
    legend_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COLOR_PASS, markersize=6,
               alpha=ALPHA, label="Pass (1)"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COLOR_FAIL, markersize=6,
               alpha=ALPHA, label="Fail (0)"),
    ]
    ax_psr.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=SIZE_LEGEND, frameon=False,
        ncol=1, handletextpad=0.3,
        borderpad=0.2, labelspacing=0.25,
    )

    # ── Panel c: Spearman correlation heatmap of the 4 PSR antigens ──────────
    ax_hm = fig.add_subplot(gs[1, 1])
    antigen_cols   = ["psr_norm_dna", "psr_norm_avidin",
                      "psr_norm_insulin", "psr_norm_smp"]
    antigen_labels = ["DNA", "Avidin", "Insulin", "SMP"]
    _draw_correlation_heatmap(
        ax_hm, elisa_df, antigen_cols, antigen_labels,
        cmap_name="Blues", vmin=0.5, vmax=1.0,
    )
    _panel_header_stacked(
        ax_hm, letter="c",
        top_line="IPI PSR-ELISA",
        bottom_line="antigen correlation",
    )

    # ── Panel d: SEC Retention Time ──────────────────────────────────────────
    ax_rt = fig.add_subplot(gs[1, 2])
    _boxplot_grouped(
        ax_rt, sec_df,
        x="sec_filter", y="Retention Time_SEC", hue="sec_filter",
        order=[1, 0],
        palette={1: COLOR_PASS, 0: COLOR_FAIL},
        showmeans=True,
    )
    ax_rt.set_xticklabels(["Pass (1)", "Fail (0)"])
    _style_boxplot(
        ax_rt, xlabel="",
        ylabel="Retention time (min)",
        panel_letter=None,
    )
    _panel_header_stacked(
        ax_rt, letter="d",
        top_line="IPI SEC",
        bottom_line="Retention time",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    tiff_path = OUT_DIR / f"{output_prefix}.tiff"
    pdf_path  = OUT_DIR / f"{output_prefix}.pdf"
    fig.savefig(
        tiff_path, dpi=dpi, format="tiff",
        bbox_inches="tight", pad_inches=0.08,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(pdf_path, dpi=dpi, format="pdf",
                bbox_inches="tight", pad_inches=0.08)
    print(f"  Saved: {tiff_path}  ({tiff_path.stat().st_size // 1024} KB)")
    print(f"  Saved: {pdf_path}")
    return fig


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Delphi dataset-summary figure "
                    "(double-column format)"
    )
    parser.add_argument("--elisa_path", type=str, required=True,
                        help="Path to public_elisa_score.xlsx")
    parser.add_argument("--sec_path",   type=str, required=True,
                        help="Path to public_sec.xlsx")
    parser.add_argument("--output_prefix", type=str, default="Figure1")
    parser.add_argument("--dpi", type=int, default=300, choices=[300, 600])
    args = parser.parse_args()

    print("Loading ELISA data...")
    elisa_df = load_elisa(args.elisa_path)
    print(f"  {len(elisa_df):,} rows")
    print("Loading SEC data...")
    sec_df = load_sec(args.sec_path)
    print(f"  {len(sec_df):,} rows")

    print("\nBuilding figure...")
    generate_dataset_summary_figure(
        elisa_df, sec_df,
        output_prefix=args.output_prefix,
        dpi=args.dpi,
    )
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}/")