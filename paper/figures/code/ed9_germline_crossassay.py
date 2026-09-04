"""Extended Data Figure 9 (NEW, revision2) - two analyses added in response to review.
  a  Framework-driven leakage: AUC under CDR H3-cluster CV vs leave-one-VH-germline-out
     vs VH+VL-pair-out, plus per-held-out-germline AUC (R1).
  b  Dual-liability co-occurrence: P(SEC fail) for PSR-Pass vs PSR-Fail antibodies (R2).
  c  CDR H3 net charge predicts BOTH failure modes within the both-measured set (R2).

R1 values are loaded from NEW_RESULTS/R1_results.json (produced + verified by
R1_germline_out.py, XGBoost+AbLang2). R2 is recomputed here from ipi_sec_5000.xlsx.
No invented numbers.
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
from paths import data_file, ensure_output
warnings.filterwarnings("ignore")

DATA = str(data_file("ipi_sec_5000.xlsx").parent)
OUT = str(ensure_output())
R1J = str(data_file("R1_results.json"))
ok.set_style(base_pt=6.5)
PASS, FAIL, NEUTRAL = ok.PASS, ok.FAIL, ok.NEUTRAL
CHARGE = ok.OI_PURPLE

# ---- R1 (load verified) ----
r1 = json.load(open(R1J))
prot = [("CDR H3-cluster CV", r1["anchor_cluster_auc"], NEUTRAL),
        ("VH+VL-pair-out",   r1["pair_out_auc"],       ok.OI_SKYBLUE),
        ("leave-one-VH-\ngermline-out", r1["germline_out_auc"], FAIL)]
germ = sorted(r1["per_germline"], key=lambda g: g["auc"], reverse=True)  # best on top
def f3(v):  # round-half-up to 3 dp so labels match the manuscript text (e.g. 0.9025 -> 0.903)
    return f"{int(v*1000 + 0.5)/1000:.3f}"

# ---- R2 (recompute from data) ----
sec = pd.read_excel(f"{DATA}/ipi_sec_5000.xlsx")[["psr_filter", "sec_filter", "HCDR3_charge"]].dropna(
    subset=["psr_filter", "sec_filter"])
psr_fail = (sec["psr_filter"] == 0).values
sec_fail = (sec["sec_filter"] == 0).values
p_secfail_psrpass = sec_fail[~psr_fail].mean()
p_secfail_psrfail = sec_fail[psr_fail].mean()
a = int((psr_fail & sec_fail).sum()); b = int((psr_fail & ~sec_fail).sum())
c = int((~psr_fail & sec_fail).sum()); dd = int((~psr_fail & ~sec_fail).sum())
OR = (a * dd) / (b * c)
or_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
or_ci = np.exp(np.log(OR) + np.array([-1, 1]) * 1.959963984540054 * or_se)
_, fisher_p = fisher_exact([[a, b], [c, dd]])   # substantiates the "p < 1e-200" stated in the ED9 legend
ch = sec.dropna(subset=["HCDR3_charge"])
pf = (ch["psr_filter"] == 0).values; sf = (ch["sec_filter"] == 0).values
auc_psr = roc_auc_score(pf, ch["HCDR3_charge"].values)
auc_sec = roc_auc_score(sf, ch["HCDR3_charge"].values)
print(f"[R2 recomputed] P(SECfail|PSRpass)={p_secfail_psrpass:.3f} P(SECfail|PSRfail)={p_secfail_psrfail:.3f} "
      f"OR={OR:.2f} Fisher p={fisher_p:.2e} | charge AUC PSR={auc_psr:.3f} SEC={auc_sec:.3f}")

# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(ok.DOUBLE, 88 * ok.MM))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.45, 1.0], height_ratios=[1, 1],
              left=0.205, right=0.975, top=0.88, bottom=0.135, hspace=0.85, wspace=0.62)
axa = fig.add_subplot(gs[:, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1])

# ---- panel a: R1 ----
labels = [p[0] for p in prot] + [g["name"] for g in germ]
vals   = [p[1] for p in prot] + [g["auc"] for g in germ]
cols   = [p[2] for p in prot] + [PASS] * len(germ)
y = np.arange(len(labels))[::-1]          # first label on top
# gap between protocol block and germline block
y = y.astype(float)
y[3:] -= 0.7
axa.barh(y, vals, height=0.62, color=cols, edgecolor="none", zorder=2)
axa.axvline(r1["anchor_cluster_auc"], color="#555555", lw=0.7, ls="--", zorder=1)
axa.text(r1["anchor_cluster_auc"] + 0.002, y[0] + 0.7,
         "within-library\nCDR H3-cluster CV", fontsize=4.6,
         color="#555555", va="bottom", ha="left")
for yi, v in zip(y, vals):
    axa.text(v - 0.004, yi, f3(v), ha="right", va="center", fontsize=5.0,
             color="white", fontweight="bold")
axa.set_yticks(y); axa.set_yticklabels(labels, fontsize=5.4)
axa.set_xlim(0.5, 1.0); axa.set_xlabel("ROC-AUC", fontsize=6.5)
axa.set_ylim(y.min() - 0.6, y.max() + 1.2)
axa.text(0.505, y[0] + 1.05, "Evaluation protocol (same classifier)", fontsize=5.2,
         color="#333333", fontweight="bold", va="bottom")
axa.text(0.505, y[3] + 0.55, "Per held-out VH germline", fontsize=5.2,
         color="#333333", fontweight="bold", va="bottom")
axa.grid(axis="x", lw=0.25, alpha=0.4)
axa.set_title("AUC decreases when germlines are held out", fontsize=6.6,
              fontweight="bold", loc="left", pad=4)

# ---- panel b: R2 co-occurrence ----
xb = [0, 1]
axb.bar(xb, [p_secfail_psrpass, p_secfail_psrfail], width=0.62,
        color=[PASS, FAIL], edgecolor="none", zorder=2)
for xi, v in zip(xb, [p_secfail_psrpass, p_secfail_psrfail]):
    axb.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, fontweight="bold")
axb.set_xticks(xb); axb.set_xticklabels(["PSR\nPass", "PSR\nFail"], fontsize=5.6)
axb.set_ylim(0, 0.85); axb.set_ylabel("P(SEC fail)", fontsize=6.3)
axb.grid(axis="y", lw=0.25, alpha=0.4)
axb.set_title("Co-failure (n = 5,045)", fontsize=6.6, fontweight="bold", loc="left", pad=4)
axb.text(0.5, 0.82, f"odds ratio {OR:.1f}\n95% CI {or_ci[0]:.1f}-{or_ci[1]:.1f}", transform=axb.transAxes,
         ha="center", va="top", fontsize=5.2, color="#333333")

# ---- panel c: R2 charge predicts both ----
xc = [0, 1]
axc.bar(xc, [auc_psr, auc_sec], width=0.62, color=CHARGE, alpha=0.9, edgecolor="none", zorder=2)
axc.axhline(0.5, color="#999999", lw=0.6, ls="--", zorder=1)
for xi, v in zip(xc, [auc_psr, auc_sec]):
    axc.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, fontweight="bold")
axc.set_xticks(xc); axc.set_xticklabels(["PSR\nfailure", "SEC\nfailure"], fontsize=5.6)
axc.set_ylim(0.4, 0.9); axc.set_ylabel("ROC-AUC\n(CDR H3 net charge)", fontsize=6.0)
axc.grid(axis="y", lw=0.25, alpha=0.4)
axc.set_title("Net charge separates both labels", fontsize=6.6, fontweight="bold", loc="left", pad=4)

for ax, L, dx in [(axa, "a", -0.155), (axb, "b", -0.075), (axc, "c", -0.075)]:
    ok.panel_label(fig, ax, L, dx=dx, dy=0.035, size=9)

ok.save_fig(fig, "ED_Fig9", OUT)
print("ED_Fig9 done")
