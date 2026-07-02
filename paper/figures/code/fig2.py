"""
Figure 2 — representation learning / PLM embeddings.
  Row 0 (a-e): t-SNE per PLM, coloured by VH germline.
  Row 1 (f-j): SAME t-SNE coordinates, coloured by Pass/Fail (psr_filter).
               Reveals whether polyreactivity-class structure is present.
  Row 2 (k):   quantification — per-PLM grouped bars of
                 label separation = 5-fold CV ROC-AUC of logistic regression
                                    predicting psr_filter from the 2 t-SNE coords,
                 germline purity  = 15-NN accuracy of predicting VH germline
                                    from the 2 t-SNE coords.
  columns: AbLang2 | IgBert | AntiBERTy | AntiBERTa2 | AntiBERTa2-CSSP
t-SNE faithful to v1 (perplexity=7, init=pca, seed=42); computed ONCE per PLM
and reused for every colouring and the quantification.
Data: data/ipi_psr_trainset.xlsx + data/ipi_psr_trainset.xlsx.<plm>.emb.csv
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
warnings.filterwarnings("ignore"); np.random.seed(42)

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
DATA = f"{DELPHI}/data"; OUT = f"{DELPHI}/revision2_redteam/figures/output"
DATA_FILE = "ipi_psr_trainset.xlsx"
ok.set_style(base_pt=6.5)
PLMS   = ["ablang", "igbert", "antiberty", "antiberta2", "antiberta2-cssp"]
LABELS = ["AbLang2", "IgBert", "AntiBERTy", "AntiBERTa2", "AntiBERTa2-CSSP"]
VH_COL, VL_COL = "vh_scaffold", "vl_scaffold"
MS, ALPHA = 2.0, 0.75
PF_ALPHA = 0.45                      # low alpha for Pass/Fail overlay (row 1)


# 10 VH germlines exceed the 9-colour Okabe-Ito safe set. Rather than wrap the
# 10th onto OI_BLUE (two classes sharing a colour), extend with one brown that
# stays distinguishable from all 9 under deuteranopia/protanopia/greyscale
# (ColorBrewer Dark2 brown). hard-coded 10th hue, revisit if a class is added.
SAFE10 = ok.QUALITATIVE + ["#A65628"]


def color_map(values):
    uniq = sorted(pd.Series(values).dropna().astype(str).unique())
    if len(uniq) > len(SAFE10):
        raise ValueError(f"{len(uniq)} categories exceed the safe palette ({len(SAFE10)}).")
    return {g: SAFE10[i] for i, g in enumerate(uniq)}


def tsne(X):
    return TSNE(n_components=2, perplexity=7, learning_rate="auto", max_iter=1000,
               early_exaggeration=12.0, metric="euclidean", random_state=42,
               init="pca").fit_transform(X)


def scatter_cat(ax, XY, labels, cmap, title=None, alpha=ALPHA):
    for g in sorted(labels.unique()):
        m = labels == g
        ax.scatter(XY[m, 0], XY[m, 1], c=cmap[g], s=MS, alpha=alpha, linewidths=0,
                   rasterized=True)
    _frame(ax, title)


def scatter_pf(ax, XY, psr):
    # Pass (blue) first, Fail (orange) ON TOP — positives (the minority in most
    # germlines) must never be buried under Pass.
    for v, col in [(1, ok.PASS), (0, ok.FAIL)]:
        m = psr.values == v
        ax.scatter(XY[m, 0], XY[m, 1], c=col, s=MS, alpha=PF_ALPHA, linewidths=0,
                   rasterized=True)
    _frame(ax, None)


def _frame(ax, title):
    ax.set_xlabel("t-SNE 1", fontsize=6.3, labelpad=1.5)
    ax.set_ylabel("t-SNE 2", fontsize=6.3, labelpad=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect(1)            # square panel: x and y axes the same length
    if title:
        ax.set_title(title, fontsize=6.8, fontweight="bold", pad=3)


# ── load + t-SNE (ONCE per PLM) ───────────────────────────────────────────────
data = pd.read_excel(f"{DATA}/{DATA_FILE}")
data = data[pd.notna(data["psr_filter"])]
# merge split germline sub-alleles into their parent germline
data[VH_COL] = data[VH_COL].astype(str).replace({"VH3-23_A": "VH3-23", "VH3-7_A": "VH3-7"})
vh_cmap = color_map(data[VH_COL])

tsne_xy, aligned = {}, {}
for plm in PLMS:
    print("t-SNE", plm)
    emb = pd.read_csv(f"{DATA}/{DATA_FILE}.{plm}.emb.csv").set_index("BARCODE")
    want = set(data["BARCODE"])
    missing = want - set(emb.index)
    if missing:                       # NO silent drop: every antibody must be embedded
        raise SystemExit(f"[{plm}] {len(missing)} of {len(want)} antibodies have NO embedding — "
                         f"refusing to silently drop them (e.g. {list(missing)[:5]}). "
                         f"All sequences must be represented regardless of provenance.")
    common = sorted(want)             # ALL antibodies, deterministic order
    d = data.set_index("BARCODE").loc[common].reset_index()
    tsne_xy[plm] = tsne(emb.loc[common].values); aligned[plm] = d
    print(f"  {plm}: {len(common)} antibodies embedded (0 dropped)")

# ── quantification on the 2-D t-SNE coords (reuse, do not recompute t-SNE) ─────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sep_score, pur_score = {}, {}
for plm in PLMS:
    XY = tsne_xy[plm]; d = aligned[plm]
    y_pf = d["psr_filter"].astype(int).values
    y_vh = d[VH_COL].astype(str).values
    auc = cross_val_score(LogisticRegression(max_iter=1000), XY, y_pf,
                          cv=cv5, scoring="roc_auc")
    pur = cross_val_score(KNeighborsClassifier(n_neighbors=15), XY, y_vh,
                          cv=cv5, scoring="accuracy")
    sep_score[plm], pur_score[plm] = float(auc.mean()), float(pur.mean())
    print(f"  {plm:16s}  label-sep AUC={sep_score[plm]:.3f}  germ-purity={pur_score[plm]:.3f}")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(ok.DOUBLE, 135 * ok.MM))
outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 0.62],
                 left=0.045, right=0.845, top=0.935, bottom=0.085, hspace=0.42)

# two t-SNE rows share one 5-column grid each
gs0 = outer[0].subgridspec(1, 6, width_ratios=[0.10] + [1] * 5, wspace=0.28)
gs1 = outer[1].subgridspec(1, 6, width_ratios=[0.10] + [1] * 5, wspace=0.28)

# row side labels
for gs, lbl in [(gs0, "VH germline"), (gs1, "Pass / Fail")]:
    axl = fig.add_subplot(gs[0, 0]); axl.axis("off")
    axl.text(0.9, 0.5, lbl, transform=axl.transAxes, fontsize=7, fontweight="bold",
             rotation=90, va="center", ha="center")

ax_top = [fig.add_subplot(gs0[0, c + 1]) for c in range(5)]
ax_bot = [fig.add_subplot(gs1[0, c + 1]) for c in range(5)]
top_letters, bot_letters = list("abcde"), list("fghij")
for c, (plm, lab) in enumerate(zip(PLMS, LABELS)):
    d = aligned[plm]; XY = tsne_xy[plm]
    scatter_cat(ax_top[c], XY, d[VH_COL].astype(str), vh_cmap, title=lab)
    scatter_pf(ax_bot[c], XY, d["psr_filter"].astype(int))
    ok.panel_label(fig, ax_top[c], top_letters[c], dx=-0.012, dy=0.020, size=8)
    ok.panel_label(fig, ax_bot[c], bot_letters[c], dx=-0.012, dy=0.020, size=8)

# ── bottom quantification panel (spans full width) ────────────────────────────
axq = fig.add_subplot(outer[2])
x = np.arange(len(PLMS)); w = 0.38
sep = [sep_score[p] for p in PLMS]; pur = [pur_score[p] for p in PLMS]
b1 = axq.bar(x - w / 2, sep, w, color=ok.OI_BLUE, label="Label separation (ROC-AUC, Pass/Fail)")
b2 = axq.bar(x + w / 2, pur, w, color=ok.OI_ORANGE, label="Germline purity (15-NN VH accuracy)")
axq.axhline(0.5, color=ok.NEUTRAL, lw=0.6, ls=(0, (3, 3)), zorder=0)
axq.text(x[-1] + 0.52, 0.5, "chance\n(AUC)", fontsize=5.2, color=ok.NEUTRAL,
         va="center", ha="left")
for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        axq.text(r.get_x() + r.get_width() / 2, h + 0.012, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=5.3)
axq.set_xticks(x); axq.set_xticklabels(LABELS, fontsize=6.3)
axq.set_ylim(0, 1.0); axq.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axq.set_ylabel("Score", fontsize=6.5, labelpad=2)
axq.tick_params(axis="y", labelsize=5.6)
axq.legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), ncol=2, fontsize=5.8,
           handlelength=1.0, handletextpad=0.4, columnspacing=1.2, frameon=False)
ok.panel_label(fig, axq, "k", dx=-0.022, dy=0.052, size=8)

# ── right-hand legends: VH germline (rows a-e) + Pass/Fail (rows f-j) ──────────
vh_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=vh_cmap[g],
                     markersize=4, label=g) for g in sorted(vh_cmap)]
legv = fig.legend(handles=vh_handles, loc="upper left", bbox_to_anchor=(0.852, 0.93),
                  fontsize=5.6, handlelength=0.8, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.22, frameon=False)
legv.set_title("VH germline", prop={"size": 6, "weight": "bold"})

pf_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ok.PASS,
                     markersize=4, label="Pass (1)"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=ok.FAIL,
                     markersize=4, label="Fail (0)")]
legf = fig.legend(handles=pf_handles, loc="upper left", bbox_to_anchor=(0.852, 0.40),
                  fontsize=5.6, handlelength=0.8, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.22, frameon=False)
legf.set_title("Polyreactivity", prop={"size": 6, "weight": "bold"})

ok.save_fig(fig, "Figure2", OUT, dpi=600)   # t-SNE is rasterized; 600 dpi keeps it crisp
print("Fig3 done")
