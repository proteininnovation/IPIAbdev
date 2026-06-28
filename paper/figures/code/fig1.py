"""
Figure 1 (Nature-style v2) — DELPHI platform & datasets.

  a  Platform architecture flowchart (input -> curation -> representations ->
     models -> CV/threshold -> output, + interpretability branch)
  b  Dataset summary table
  c  CDR H3 diversity (clusters per sequence): designed (IPI) vs natural (DS1)
  d  PSR pass rate by VH germline (data/ipi_psr_trainset.xlsx)
  e  IPI PSR-ELISA normalized scores, 4 antigens x Pass/Fail
  f  Antigen Spearman correlation heatmap
  g  IPI SEC retention time, Pass vs Fail

Data (local): data/ipi_psr_trainset.xlsx, data/elisa_score_figure1.xlsx,
              data/sec_retention_time_figure1.xlsx
All numbers come from the data files or the curated dataset constants below.
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_nature as ok
warnings.filterwarnings("ignore")

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
DATA = f"{DELPHI}/data"
OUT  = f"{DELPHI}/revision2_redteam/figures/output"
ok.set_style(base_pt=7)
PASS, FAIL, NEUTRAL = ok.PASS, ok.FAIL, ok.NEUTRAL

# muted Okabe-Ito tints for the flowchart box groups (pale fill, dark text)
TINT_INPUT  = "#EAEAEA"   # neutral grey
TINT_CURATE = "#FBEFDB"   # pale orange
TINT_REP    = "#E3EEF8"   # pale blue   (representations)
TINT_MODEL  = "#E0F0E8"   # pale green  (models)
TINT_CV     = "#F0E9F4"   # pale purple (CV band)
TINT_OUT    = "#FBE3CC"   # warm orange (output)
TINT_INTERP = "#F2F2F2"   # interpretability
EDGE = "#5A5A5A"

# ─────────────────────────── PANEL a : flowchart ─────────────────────────────
def _box(ax, x, y, w, h, text, fc, fontsize=5.6, weight="normal",
         ec=EDGE, lw=0.6, style="round,pad=0.012,rounding_size=0.018",
         tc="#1A1A1A", va="center", linespacing=1.05):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
                                mutation_aspect=0.55))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va=va, fontsize=fontsize,
            fontweight=weight, color=tc, zorder=4, linespacing=linespacing)
    return (x, y, w, h)


def _arrow(ax, p0, p1, color=EDGE, lw=0.8, ls="-", rad=0.0, mut=4.0,
           zorder=3, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                                 connectionstyle=f"arc3,rad={rad}", color=color,
                                 lw=lw, linestyle=ls, zorder=zorder, alpha=alpha,
                                 shrinkA=0.5, shrinkB=0.5))


def R(b):  # right-mid
    return (b[0] + b[2], b[1] + b[3] / 2)
def L(b):  # left-mid
    return (b[0], b[1] + b[3] / 2)
def T(b):  # top-mid
    return (b[0] + b[2] / 2, b[1] + b[3])
def B(b):  # bottom-mid
    return (b[0] + b[2] / 2, b[1])


def draw_flowchart(ax):
    # small margins so the leftmost/rightmost boxes are never clipped at the figure edge
    ax.set_xlim(-0.025, 1.025); ax.set_ylim(0, 1); ax.axis("off")

    def stitle(x, s):
        ax.text(x, 0.985, s, ha="left", va="top", fontsize=5.6,
                fontweight="bold", color="#555555")

    # ---- Stage 1 : INPUT ----
    x1, w1 = 0.010, 0.150
    stitle(x1, "INPUT")
    b_in = _box(ax, x1, 0.66, w1, 0.215,
                "Antibody sequences\nVH / VL / CDR H3\n+ assay label",
                TINT_INPUT, fontsize=5.6, weight="bold")
    _box(ax, x1, 0.560, w1, 0.072, "PSR → polyreactivity\n(psr_filter)", "#FFFFFF",
         fontsize=5.0, ec="#9AA7B4", lw=0.5)
    _box(ax, x1, 0.478, w1, 0.072, "SEC → monomericity\n(sec_filter)", "#FFFFFF",
         fontsize=5.0, ec="#9AA7B4", lw=0.5)
    _box(ax, x1, 0.408, w1, 0.058, "label = PASS(1) / FAIL(0)", "#FFFFFF",
         fontsize=5.0, ec="#9AA7B4", lw=0.5)

    # ---- Stage 2 : CURATION ----
    x2, w2 = 0.202, 0.144
    stitle(x2, "CURATION")
    b_cur = _box(ax, x2, 0.66, w2, 0.215,
                 "CDR H3 clustering\n(80% identity)\n+ out-of-fold\nbalancing",
                 TINT_CURATE, fontsize=5.6)
    _arrow(ax, R(b_in), L(b_cur), lw=0.9)

    # ---- Stage 3 : REPRESENTATIONS (PLMs on top, one-hot at the bottom) ----
    x3, w3 = 0.388, 0.150
    stitle(x3, "REPRESENTATION")
    reps = ["AbLang2", "IgBert", "AntiBERTy", "AntiBERTa2", "AntiBERTa2-CSSP",
            "Biophysical", "K-mer", "One-hot"]
    rep_h, rep_gap, rep_top = 0.055, 0.0080, 0.905
    split_after = 5     # gap between the 5 PLMs and the classical reps
    plm_split = 0.034
    rb = {}
    y = rep_top
    for i, name in enumerate(reps):
        if i == split_after:
            y -= plm_split
        y -= rep_h
        is_plm = i < 5
        rb[name] = _box(ax, x3, y, w3, rep_h, name, TINT_REP,
                        fontsize=5.2, weight="normal" if is_plm else "bold")
        y -= rep_gap
    # PLM group label, in the split gap (labels the block above it)
    plm_label_y = (rb["AntiBERTa2-CSSP"][1] + rb["Biophysical"][1] + rep_h) / 2
    ax.text(x3 + w3 / 2, plm_label_y, "protein language models",
            ha="center", va="center", fontsize=4.7, style="italic", color="#3A5A78")
    # curation -> the WHOLE representation stack (straight arrow to its vertical centre)
    rep_centre_y = (rb["AbLang2"][1] + rep_h + rb["One-hot"][1]) / 2
    _arrow(ax, R(b_cur), (x3 - 0.004, rep_centre_y), lw=0.9)

    # ---- Stage 4 : MODELS (Transformer-OneHot at the bottom, feeds the CV band) ----
    x4, w4 = 0.612, 0.150
    stitle(x4, "MODELS")
    models = ["Transformer-LM", "CNN", "Random Forest", "XGBoost", "Transformer-OneHot"]
    mod_h, mod_gap, mod_top = 0.072, 0.0175, 0.840   # stack lowered to clear the title band
    mb = {}
    for i, name in enumerate(models):
        y = mod_top - (i + 1) * mod_h - i * mod_gap
        mb[name] = _box(ax, x4, y, w4, mod_h, name, TINT_MODEL,
                        fontsize=5.3, weight="bold")

    # ---- grouped (bus) compatibility arrows; ordering avoids crossings ----
    bus_x = x3 + w3 + 0.022
    # PLM bus -> Transformer-LM + CNN
    plm_names = ["AbLang2", "IgBert", "AntiBERTy", "AntiBERTa2", "AntiBERTa2-CSSP"]
    plm_ys = [rb[n][1] + rep_h / 2 for n in plm_names]
    for n in plm_names:
        _arrow(ax, R(rb[n]), (bus_x, rb[n][1] + rep_h / 2), lw=0.5, color="#7E97B0", mut=2.6)
    ax.plot([bus_x, bus_x], [min(plm_ys), max(plm_ys)], color="#7E97B0", lw=1.0, zorder=3)
    plm_mid = (min(plm_ys) + max(plm_ys)) / 2
    _arrow(ax, (bus_x, plm_mid), L(mb["Transformer-LM"]), lw=1.0, color="#3A5A78", rad=0.05)
    _arrow(ax, (bus_x, plm_mid), L(mb["CNN"]), lw=1.0, color="#3A5A78", rad=-0.05)
    # classical bus -> Random Forest + XGBoost
    bk_ys = [rb["Biophysical"][1] + rep_h / 2, rb["K-mer"][1] + rep_h / 2]
    for n in ["Biophysical", "K-mer"]:
        _arrow(ax, R(rb[n]), (bus_x, rb[n][1] + rep_h / 2), lw=0.5, color="#6FA98F", mut=2.6)
    ax.plot([bus_x, bus_x], [min(bk_ys), max(bk_ys)], color="#6FA98F", lw=1.0, zorder=3)
    bk_mid = sum(bk_ys) / 2
    _arrow(ax, (bus_x, bk_mid), L(mb["Random Forest"]), lw=1.0, color="#3F7D5F", rad=0.05)
    _arrow(ax, (bus_x, bk_mid), L(mb["XGBoost"]), lw=1.0, color="#3F7D5F", rad=-0.04)
    # one-hot -> Transformer-OneHot (both at the bottom)
    _arrow(ax, R(rb["One-hot"]), L(mb["Transformer-OneHot"]), lw=0.8, color="#7E97B0", rad=0.0)

    # 'deployed' note just above the (top) Transformer-LM box — readable, dark, with pointer
    tl = mb["Transformer-LM"]
    ax.text(tl[0] + tl[2] / 2, tl[1] + tl[3] + 0.030, "deployed · best generalization",
            ha="center", va="bottom", fontsize=5.0, fontweight="bold", color="#A65A00")
    _arrow(ax, (tl[0] + tl[2] / 2, tl[1] + tl[3] + 0.028),
           (tl[0] + tl[2] / 2, tl[1] + tl[3] + 0.003), lw=0.7, color="#A65A00", mut=3.0)

    # ---- Stage 5 : CV band beneath models ----
    cv_y, cv_h = 0.075, 0.092
    b_cv = _box(ax, x4, cv_y, w4, cv_h,
                "10-fold CDR H3-stratified CV\n→ OOF pooling\n→ Youden threshold",
                TINT_CV, fontsize=5.1)
    for n in models:
        _arrow(ax, B(mb[n]), (mb[n][0] + w4 / 2, cv_y + cv_h), lw=0.5, color="#8B6FA8", mut=2.4)

    # ---- Stage 6 : OUTPUT ----
    x6, w6 = 0.812, 0.168
    stitle(x6, "OUTPUT")
    b_out = _box(ax, x6, 0.585, w6, 0.270,
                 "P(Pass) score\n↓\nthreshold\n↓\ndevelopability call\n(PASS / FAIL)",
                 TINT_OUT, fontsize=5.5, weight="bold")
    _arrow(ax, R(b_cv), (x6 - 0.004, b_out[1] + 0.045), lw=1.0, rad=-0.18)

    # ---- interpretability branch (dashed) ----
    b_int = _box(ax, x6, 0.375, w6, 0.160,
                 "INTERPRETABILITY\nIntegrated Gradients (Transformer)\nSHAP (RF / XGBoost)\nCDR3 mutagenesis",
                 TINT_INTERP, fontsize=4.9, ec="#8A8A8A", lw=0.5)
    _arrow(ax, B(b_out), T(b_int), lw=0.8, ls=(0, (3, 2)), color="#7A7A7A")


# ─────────────────────────── PANEL b : table ─────────────────────────────────
HEADERS = ["Dataset", "Total", "Pass\n(n)", "Fail\n(n)",
           "CDR H3\nclust.", "CDR H3\nsing.", "Library"]
ROWS = [
    ["IPI PSR-ELISA",  "7,494",  "5,925",  "1,569",  "5,046", "3,895",
     "PSR-ELISA"],
    ["IPI PSR-NGS", "3,771", "0", "3,771", "2,291", "1,648", "NGS ssDNA"],
    ["IPI PSR train", "11,265", "5,925", "5,340", "7,263", "5,443",
     "ELISA+NGS"],
    ["IPI SEC train", "5,045", "3,210", "1,835", "3,272", "2,468", "SEC-HPLC"],
    ["DS1 (public)", "246,293", "131,255", "115,038", "6,311", "1,665",
     "Chen 2024"],
]
COL_W = [0.185, 0.135, 0.120, 0.135, 0.115, 0.105, 0.205]
COL_AL = ['l', 'r', 'r', 'r', 'r', 'r', 'l']


def draw_table(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    hh, rh = 0.255, 0.142
    y_top = 0.99
    y_bot = y_top - (hh + len(ROWS) * rh)
    xe = np.concatenate(([0.0], np.cumsum(COL_W)))
    yh = y_top - hh
    ax.add_patch(Rectangle((0, yh), 1, hh, facecolor=ok.TBL_HEADER_BG, edgecolor="none", zorder=1))
    for i, h in enumerate(HEADERS):
        ax.text(xe[i] + COL_W[i] / 2, yh + hh / 2, h, ha="center", va="center",
                color=ok.TBL_HEADER_FG, fontsize=4.6, fontweight="bold",
                linespacing=1.0, zorder=3)
    for r, row in enumerate(ROWS):
        y = yh - (r + 1) * rh
        ax.add_patch(Rectangle((0, y), 1, rh, facecolor=ok.TBL_ROW_B if r % 2 else ok.TBL_ROW_A,
                               edgecolor="none", zorder=1))
        for ci, tint in [(2, ok.TBL_PASS_TINT), (3, ok.TBL_FAIL_TINT)]:
            ax.add_patch(Rectangle((xe[ci], y), COL_W[ci], rh, facecolor=tint, edgecolor="none", zorder=2))
        if r > 0:
            ax.plot([0, 1], [y + rh, y + rh], color=ok.TBL_RULE, lw=0.4, zorder=3)
        for i, cell in enumerate(row):
            al = COL_AL[i]; pad = 0.006
            if al == 'l':   xt, ha = xe[i] + pad, "left"
            elif al == 'r': xt, ha = xe[i] + COL_W[i] - pad, "right"
            else:           xt, ha = xe[i] + COL_W[i] / 2, "center"
            ax.text(xt, y + rh / 2, cell, ha=ha, va="center", fontsize=4.7,
                    linespacing=1.0, zorder=4)
    ax.plot([0, 1], [y_top, y_top], color=ok.TBL_HEADER_BG, lw=0.8, zorder=5)
    ax.plot([0, 1], [y_bot, y_bot], color=ok.TBL_HEADER_BG, lw=0.8, zorder=5)


# ─────────────────────────── shared boxplot ─────────────────────────────────
def boxplot(ax, df, x, y, hue, order):
    sns.boxplot(data=df, x=x, y=y, hue=hue, order=order,
                palette={1: PASS, 0: FAIL}, ax=ax, showfliers=False, showmeans=True,
                width=0.7, linewidth=0.6, gap=0.12,
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                               markersize=2.4, markeredgewidth=0.5),
                boxprops=dict(alpha=0.9, edgecolor="black"),
                medianprops=dict(color="black", linewidth=0.8),
                whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5))
    if ax.get_legend():
        ax.get_legend().remove()


# ─────────────────────────── data loaders ─────────────────────────────────
def load_elisa():
    df = pd.read_excel(f"{DATA}/elisa_score_figure1.xlsx").dropna(subset=["psr_filter"])
    df["psr_filter"] = df["psr_filter"].astype(int)
    for c in ["psr_norm_dna", "psr_norm_avidin", "psr_norm_insulin", "psr_norm_smp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _main_peak(rt, pa):
    if pd.isna(rt) or pd.isna(pa): return np.nan, np.nan
    try:
        rts = [float(x) for x in str(rt).split(",") if x.strip()]
        pas = [float(x) for x in str(pa).split(",") if x.strip()]
        if len(rts) != len(pas) or not pas: return np.nan, np.nan
        i = int(np.argmax(pas)); return rts[i], pas[i]
    except (ValueError, TypeError):
        return np.nan, np.nan


def load_sec():
    df = pd.read_excel(f"{DATA}/sec_retention_time_figure1.xlsx")
    df["sec_filter"] = pd.to_numeric(df["sec_filter"], errors="coerce")
    df = df.dropna(subset=["sec_filter"]); df["sec_filter"] = df["sec_filter"].astype(int)
    rts = [_main_peak(a, b)[0] for a, b in zip(df["retention_time_mins"], df["Peak Area Percent"])]
    df["RT"] = rts
    return df


def load_germline():
    df = pd.read_excel(f"{DATA}/ipi_psr_trainset.xlsx")
    df["psr_filter"] = pd.to_numeric(df["psr_filter"], errors="coerce")
    df = df.dropna(subset=["psr_filter", "vh_scaffold"])
    # merge split germline sub-alleles into their parent germline (consistent with Fig 3)
    df["vh_scaffold"] = df["vh_scaffold"].astype(str).replace({"VH3-23_A": "VH3-23", "VH3-7_A": "VH3-7"})
    overall = df["psr_filter"].mean()
    g = df.groupby("vh_scaffold")["psr_filter"].agg(rate="mean", n="count")
    g = g[g["n"] >= 100].sort_values("rate", ascending=False)
    return g, overall


# ─────────────────────────── figure ─────────────────────────────────
elisa, sec = load_elisa(), load_sec()
germ, overall_rate = load_germline()

fig = plt.figure(figsize=(ok.DOUBLE, 175 * ok.MM))
gs = GridSpec(3, 3, figure=fig,
              height_ratios=[1.42, 1.0, 1.0],
              width_ratios=[1.55, 1.18, 1.05],
              left=0.052, right=0.985, top=0.955, bottom=0.075,
              hspace=0.66, wspace=0.50)

# a: flowchart (spans all columns, row 0)
axA = fig.add_subplot(gs[0, :]); draw_flowchart(axA)
fig.text(0.012, 0.988, "a", fontsize=8.5, fontweight="bold", va="top")
fig.text(0.052, 0.985, "DELPHI developability-prediction platform",
         fontsize=7.2, fontweight="bold", va="top", color="#222222")

# b: table
axB = fig.add_subplot(gs[1, 0]); draw_table(axB)
axB.set_title("Curated developability datasets", fontsize=7, fontweight="bold",
              loc="left", pad=4)

# c: CDR H3 diversity (clusters per sequence)
axC = fig.add_subplot(gs[1, 1])
cdiv = [
    ("IPI PSR\ntrain", 7263 / 11265, PASS),
    ("IPI SEC\ntrain", 3272 / 5045, PASS),
    ("DS1\n(natural)", 6311 / 246293, FAIL),
]
xs = np.arange(len(cdiv))
vals = [v for _, v, _ in cdiv]
cols = [c for _, _, c in cdiv]
axC.bar(xs, vals, color=cols, width=0.66, edgecolor="black", linewidth=0.5)
for x, v in zip(xs, vals):
    axC.text(x, v + 0.012, f"{v:.2f}" if v >= 0.1 else f"{v:.3f}",
             ha="center", va="bottom", fontsize=5.6)
axC.set_xticks(xs); axC.set_xticklabels([l for l, _, _ in cdiv], fontsize=5.8)
axC.set_ylabel("CDR H3 clusters / sequence", fontsize=6.6)
axC.set_ylim(0, 0.82)
axC.set_title("CDR H3 sequence diversity", fontsize=7, fontweight="bold", loc="left", pad=4)
gap = (7263 / 11265) / (6311 / 246293)
axC.annotate("", xy=(0.30, 0.645), xytext=(1.92, 0.06),
             arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.7))
# annotation parked in the white space above the tiny DS1 bar
axC.text(1.98, 0.40, f"~{gap:.0f}× more\ndiverse per\nsequence",
         ha="center", va="center", fontsize=5.3, color="#333333", linespacing=1.1)
axC.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=PASS, markersize=6, label="designed (IPI)"),
                    Line2D([0], [0], marker="s", color="w", markerfacecolor=FAIL, markersize=6, label="natural (DS1)")],
           loc="upper right", fontsize=5.3, handletextpad=0.3, labelspacing=0.25, borderpad=0.2)

# d: PSR pass rate by VH germline
axD = fig.add_subplot(gs[1, 2])
gl = germ.iloc[::-1]            # ascending so highest is at top
ys = np.arange(len(gl))
rates = gl["rate"].values * 100
ns = gl["n"].values
bar_cols = [PASS if r >= overall_rate * 100 else FAIL for r in rates]
axD.barh(ys, rates, color=bar_cols, height=0.7, edgecolor="black", linewidth=0.4)
for y, r, n in zip(ys, rates, ns):
    axD.text(r + 1.8, y, f"n={int(n)}", va="center", ha="left", fontsize=4.7, color="#333333")
axD.axvline(overall_rate * 100, color="#444444", lw=0.7, ls=(0, (3, 2)))
# label near the x-axis where the short low-pass-rate bars leave white space
axD.text(overall_rate * 100 + 2, 0.0, f"overall {overall_rate*100:.0f}%",
         ha="left", va="center", fontsize=4.8, color="#444444")
axD.set_yticks(ys); axD.set_yticklabels(gl.index, fontsize=5.2)
axD.set_xlabel("PSR pass rate (%)", fontsize=6.6)
axD.set_xlim(0, 120)
axD.set_xticks([0, 25, 50, 75, 100])
axD.set_title("PSR pass rate by VH germline", fontsize=7, fontweight="bold", loc="left", pad=4)

# e: PSR boxplots
axE = fig.add_subplot(gs[2, 0])
longp = elisa.melt(id_vars=["psr_filter"],
                   value_vars=["psr_norm_dna", "psr_norm_avidin", "psr_norm_insulin", "psr_norm_smp"],
                   var_name="Antigen", value_name="score")
longp["Antigen"] = longp["Antigen"].map({"psr_norm_dna": "DNA", "psr_norm_avidin": "Avidin",
                                         "psr_norm_insulin": "Insulin", "psr_norm_smp": "SMP"})
boxplot(axE, longp, "Antigen", "score", "psr_filter", ["DNA", "Avidin", "Insulin", "SMP"])
qlo, qhi = longp["score"].quantile([0.01, 0.99]); pad = 0.15 * (qhi - qlo)
axE.set_ylim(qlo - pad, qhi + pad); axE.set_xlabel("")
axE.set_ylabel("Normalized PSR score", fontsize=6.6)
axE.set_title("IPI PSR-ELISA by antigen", fontsize=7, fontweight="bold", loc="left", pad=4)
axE.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=PASS, markersize=6, label="Pass (1)"),
                    Line2D([0], [0], marker="s", color="w", markerfacecolor=FAIL, markersize=6, label="Fail (0)")],
           loc="upper right", fontsize=5.6, handletextpad=0.3, labelspacing=0.25, borderpad=0.2)

# f: correlation heatmap
axF = fig.add_subplot(gs[2, 1])
cols_h = ["psr_norm_dna", "psr_norm_avidin", "psr_norm_insulin", "psr_norm_smp"]
lab = ["DNA", "Avidin", "Insulin", "SMP"]
rho = elisa[cols_h].corr(method="spearman").values
im = axF.imshow(rho, cmap=ok.SEQ, vmin=0.5, vmax=1.0, aspect="equal")
for i in range(4):
    for j in range(4):
        axF.text(j, i, f"{rho[i,j]:.2f}", ha="center", va="center", fontsize=5.8,
                 color=ok.text_on(rho[i, j], 0.5, 1.0))
axF.set_xticks(range(4)); axF.set_xticklabels(lab, fontsize=5.8)
axF.set_yticks(range(4)); axF.set_yticklabels(lab, fontsize=5.8)
axF.tick_params(length=0, pad=1)
for s in axF.spines.values(): s.set_visible(False)
axF.set_title("Antigen score correlation", fontsize=7, fontweight="bold", loc="left", pad=4)
cb = fig.colorbar(im, ax=axF, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=5, length=2)
cb.set_label("Spearman ρ", fontsize=6)

# g: SEC retention
axG = fig.add_subplot(gs[2, 2])
boxplot(axG, sec, "sec_filter", "RT", "sec_filter", [1, 0])
axG.set_xticks([0, 1]); axG.set_xticklabels(["Pass (1)", "Fail (0)"], fontsize=5.8)
axG.set_xlabel(""); axG.set_ylabel("Retention time (min)", fontsize=6.6)
axG.set_title("IPI SEC retention", fontsize=7, fontweight="bold", loc="left", pad=4)

ok.panel_label(fig, axB, "b"); ok.panel_label(fig, axC, "c"); ok.panel_label(fig, axD, "d")
ok.panel_label(fig, axE, "e"); ok.panel_label(fig, axF, "f"); ok.panel_label(fig, axG, "g")
ok.save_fig(fig, "Figure1", OUT)
print("Fig1 v2 done | overall PSR pass rate %.1f%%" % (overall_rate * 100))
