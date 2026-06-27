"""
Shared Okabe-Ito / Nature style for the DELPHI paper figures.
=============================================================

Single source of truth for colour and typography across the improved figure set
(Figure1..Figure6). Every figure imports its colours and style from HERE, so a
reader who learns "blue = Pass" in Figure 1 reads it the same way everywhere, and
the whole set is colourblind- and greyscale-safe.

Okabe-Ito (Okabe & Ito 2008) is the reference colourblind-safe qualitative set;
it survives deuteranopia/protanopia/tritanopia and greyscale. We avoid red-green
as the meaningful contrast and never use rainbow/jet.

Build figures at FINAL print size (figsize in inches = mm / 25.4) so 6-7 pt text
on screen is 6-7 pt on the page. Do NOT build big and shrink.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# ── Print geometry (Nature Biotechnology) ─────────────────────────────────────
MM        = 1 / 25.4
SINGLE    = 89  * MM      # single column
COL1_5    = 120 * MM      # 1.5 column
DOUBLE    = 183 * MM      # double column
MAX_DEPTH = 247 * MM      # page depth limit

# ── Raw Okabe-Ito palette (do not edit these hexes) ───────────────────────────
OI_BLACK    = "#000000"
OI_ORANGE   = "#E69F00"
OI_SKYBLUE  = "#56B4E9"
OI_GREEN    = "#009E73"   # bluish green
OI_YELLOW   = "#F0E442"
OI_BLUE     = "#0072B2"
OI_VERMILION= "#D55E00"
OI_PURPLE   = "#CC79A7"   # reddish purple
OI_GREY     = "#999999"

# ── Semantic tokens — ONE colour, ONE meaning, reused across every figure ──────
PASS   = OI_BLUE         # Pass / designed / "good" outcome
FAIL   = OI_ORANGE       # Fail / "bad" outcome
NEUTRAL= OI_GREY         # baseline / reference / "all"

# Antibody-region tokens (Figure 6 a-d): region = colour
REGION_HCDR3 = OI_PURPLE
REGION_VH    = OI_BLUE
REGION_VL    = OI_ORANGE

# Attribution-method tokens (Figure 6 e-f): chosen distinct from the region hues
METHOD_RF  = OI_GREEN
METHOD_XGB = OI_VERMILION
METHOD_IG  = "#444444"

# Qualitative order for unordered categories (germlines, etc.). Front-loads the
# most separable pairs; ends on grey for an "other"/baseline bucket.
QUALITATIVE = [OI_BLUE, OI_ORANGE, OI_GREEN, OI_PURPLE, OI_SKYBLUE,
               OI_VERMILION, OI_YELLOW, OI_GREY, OI_BLACK]

# Table chrome (Figure 1) — neutral structural colour, not a data encoding.
TBL_HEADER_BG = "#22384F"
TBL_HEADER_FG = "#FFFFFF"
TBL_ROW_A     = "#FFFFFF"
TBL_ROW_B     = "#F2F5F8"
TBL_PASS_TINT = "#E3EEF8"   # pale blue   (Pass column)
TBL_FAIL_TINT = "#FBEBD8"   # pale orange (Fail column)
TBL_RULE      = "#C9D2DC"


def qualitative(n: int) -> list[str]:
    if n > len(QUALITATIVE):
        raise ValueError(f"{n} categorical colours exceeds the safe set; group into 'other'.")
    return QUALITATIVE[:n]


# ── Colormaps. Match the map to the DATA. ─────────────────────────────────────
#   SEQ       -> one-directional magnitude (counts, %, |correlation|, intensity)
#   DIVERGING -> signed values with a meaningful zero/centre (corr, ρ, AUC vs 0.5)
SEQ = LinearSegmentedColormap.from_list(
    "oi_seq_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
DIVERGING = LinearSegmentedColormap.from_list(
    "oi_div", ["#08519C", "#6BAED6", "#F7F7F7", "#FDA56B", "#D55E00"])


def text_on(value, vmin, vmax, diverging=False, thresh=0.62) -> str:
    """Legible label colour for text drawn ON a heatmap cell."""
    if diverging:
        norm = abs(value) / max(abs(vmin), abs(vmax), 1e-9)
    else:
        norm = (value - vmin) / max(vmax - vmin, 1e-9)
    return "white" if norm >= thresh else "black"


# ── Typography ────────────────────────────────────────────────────────────────
_PREFERRED = ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"]


def _register_arial() -> str | None:
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"):
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in _PREFERRED:
        if name in avail:
            return name
    return None


def set_style(base_pt: float = 6.5) -> None:
    """Apply journal rcParams at FINAL on-page point sizes. Call once per figure."""
    fam = _register_arial() or "sans-serif"
    matplotlib.rcParams.update({
        "font.family"        : fam,
        "font.size"          : base_pt,
        "axes.labelsize"     : base_pt + 0.5,
        "axes.titlesize"     : base_pt + 0.5,
        "xtick.labelsize"    : base_pt - 0.5,
        "ytick.labelsize"    : base_pt - 0.5,
        "legend.fontsize"    : base_pt - 0.5,
        "axes.linewidth"     : 0.5,
        "xtick.major.width"  : 0.5,
        "ytick.major.width"  : 0.5,
        "xtick.major.size"   : 2.0,
        "ytick.major.size"   : 2.0,
        "xtick.direction"    : "out",
        "ytick.direction"    : "out",
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "legend.frameon"     : False,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.4,
        "legend.borderpad"   : 0.3,
        "figure.dpi"         : 150,
        "savefig.dpi"        : 300,
        "pdf.fonttype"       : 42,   # editable embedded text
        "ps.fonttype"        : 42,
        "svg.fonttype"       : "none",
    })


def panel_label(fig, ax, letter, dx=-0.020, dy=0.022, size=8.5):
    """Bold panel tag in FIGURE coords, just outside the axis top-left so it never
    sits on the data. Nudge dx/dy if a particular panel needs it."""
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, letter, ha="left", va="top",
             fontsize=size, fontweight="bold")


def save_fig(fig, stem, outdir, dpi=300, tiff=True):
    """Save vector PDF + 300 dpi PNG (+ LZW TIFF for submission). No bbox='tight'
    on multipanel figures — it shifts panels and breaks alignment."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=dpi)
    saved = [pdf, png]
    if tiff:
        tif = outdir / f"{stem}.tiff"
        fig.savefig(tif, format="tiff", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
        saved.append(tif)
    print("  saved:", ", ".join(p.name for p in saved))
    return saved
