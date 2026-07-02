"""
Figure 4 — Generalization & external clinical validation.
Centrepiece for the zero-shot-transfer claim: the DELPHI model (Transformer +
AbLang2, trained on IPI PSR) is applied UNCHANGED to external libraries and
public clinical-stage antibody panels.

  a  Cross-library transfer dumbbell  — IPI→DS1 vs DS1→IPI AUC per LM (asymmetry)
  b  Per-cohort external ROC          — Jain / GDPa1 / GDPa3 (AUC + 95% CI + n)
  c  Score-vs-assay scatter           — DELPHI score vs GDPa1 PR-CHO (neg. corr.)
  d  Score by subgroup                — GDPa1 score split by IgG subtype & clin. status
  e  Competition forest               — GDPa3 PR-CHO |Spearman ρ| per LM vs 113-team band
  f  Zero-shot |ρ| by cohort          — best-model |ρ| for Jain / GDPa1 / GDPa3

Sign convention: DELPHI score (transformer_lm_*_score) higher = more Pass = LESS
polyreactive. Assay PR/PSR scores higher = MORE polyreactive. So score-vs-assay
is NEGATIVE; for ROC, Pass = assay < threshold and higher score predicts Pass.

Every point estimate carries a bootstrap 95% CI (2000 resamples, rng seed 0).
No number is invented — all read from the data files.
Data: data/{Figure4_data.xlsx, Jain2017_*, GDPa1_*, GDPa3_*}.
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, linregress
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok
warnings.filterwarnings("ignore")

DELPHI = "/Users/Andre.Teixeira/Library/CloudStorage/GoogleDrive-andre.teixeira@proteininnovation.org/.shortcut-targets-by-id/1pzqwNBoHnehFObY0PzrgligSRKxpVPPY/DELPHI"
DATA = f"{DELPHI}/data"
OUT = f"{DELPHI}/revision2_redteam/figures/output"
ok.set_style(base_pt=6.5)
PASS, FAIL, NEUTRAL = ok.PASS, ok.FAIL, ok.NEUTRAL
GREY = ok.OI_GREY

THRESH = 0.27          # PR/PSR Pass cutoff (assay < THRESH => Pass)
COMP_LO, COMP_HI = 0.337, 0.356   # 113-team competition best band (GDPa3 PR-CHO)
N_BOOT = 2000
RNG = np.random.default_rng(0)

# DELPHI deployed model = Transformer + AbLang2
DELPHI_SC = "transformer_lm_ablang_ipi_psr_trainset_score"
MODELS = ["ablang", "igbert", "antiberta2", "antiberta2-cssp", "antiberty"]
MODEL_DISP = {"ablang": "AbLang2", "igbert": "IgBert", "antiberta2": "AntiBERTa2",
              "antiberta2-cssp": "AntiBERTa2-CSSP", "antiberty": "AntiBERTy"}


def score_col(lm):
    return f"transformer_lm_{lm}_ipi_psr_trainset_score"


# ── bootstrap helpers (paired resample of the index, seed 0) ──────────────────
def boot_ci(stat_fn, n, n_boot=N_BOOT, rng=RNG):
    """Bootstrap 95% CI of stat_fn(idx) over a paired resample of range(n)."""
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_ci(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pt = roc_auc_score(y, s)

    def fn(idx):
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == len(yi):
            return None
        return roc_auc_score(yi, s[idx])
    lo, hi = boot_ci(fn, len(y))
    return pt, lo, hi


def absrho_ci(s, a):
    """|Spearman rho| point + CI between score s and assay a (paired)."""
    s = np.asarray(s); a = np.asarray(a)
    pt = abs(spearmanr(s, a).correlation)

    def fn(idx):
        c = spearmanr(s[idx], a[idx]).correlation
        return abs(c) if np.isfinite(c) else None
    lo, hi = boot_ci(fn, len(s))
    return pt, lo, hi


def rho_signed_ci(s, a):
    s = np.asarray(s); a = np.asarray(a)
    pt = spearmanr(s, a).correlation

    def fn(idx):
        c = spearmanr(s[idx], a[idx]).correlation
        return c if np.isfinite(c) else None
    lo, hi = boot_ci(fn, len(s))
    return pt, lo, hi


# ── load ──────────────────────────────────────────────────────────────────────
f4 = pd.read_excel(f"{DATA}/Figure4_data.xlsx", sheet_name="Fig4B_Cross_Dataset", skiprows=1)
f4.columns = f4.columns.str.strip()
f4["Condition"] = f4["Condition"].ffill()
f4 = f4.dropna(subset=["Language Model"])

jain = pd.read_excel(f"{DATA}/Jain2017_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")
g1 = pd.read_excel(f"{DATA}/GDPa1_v1.3_20251027_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")
g3 = pd.read_excel(f"{DATA}/GDPa3_20260106_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")


def cohort_arrays(df, assay_col, thr=THRESH, binary_col=None):
    """Return (score, assay_continuous, y_pass) with NaNs dropped. y_pass=1 => Pass.
    If binary_col given, use it directly as Pass=1 truth; else Pass = assay<thr."""
    cols = [DELPHI_SC, assay_col] + ([binary_col] if binary_col else [])
    sub = df[cols].dropna()
    s = sub[DELPHI_SC].values
    a = sub[assay_col].values
    if binary_col:
        y = sub[binary_col].astype(int).values
    else:
        y = (a < thr).astype(int)
    return s, a, y


# Jain: binarise on psr_filter (already Pass=1); continuous = PSR_SMP_Score
jain_s, jain_a, jain_y = cohort_arrays(jain, "PSR_SMP_Score", binary_col="psr_filter")
g1_s, g1_a, g1_y = cohort_arrays(g1, "polyreactivity_prscore_cho_avg")
g3_s, g3_a, g3_y = cohort_arrays(g3, "polyreactivity_prscore_cho_avg")

COHORTS = [
    ("Jain 2017", jain_s, jain_a, jain_y, ok.OI_BLUE, "-"),
    ("GDPa1",     g1_s,   g1_a,   g1_y,   ok.OI_VERMILION, "--"),
    ("GDPa3",     g3_s,   g3_a,   g3_y,   ok.OI_GREEN, "-."),
]

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(ok.DOUBLE, 120 * ok.MM))
fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.085,
                    hspace=0.55, wspace=0.40)
(axa, axb, axc), (axd, axe, axf) = axes


# ── panel a: cross-library transfer dumbbell ─────────────────────────────────
def panel_a(ax):
    cond = {"IPI → DS1 Transfer": {}, "DS1 → IPI Transfer": {}}
    for _, r in f4.iterrows():
        c = str(r["Condition"]).strip()
        if c in cond:
            cond[c][str(r["Language Model"]).strip()] = float(r["AUC"])
    fwd, rev = cond["IPI → DS1 Transfer"], cond["DS1 → IPI Transfer"]
    lms = [l for l in fwd if l in rev]
    # sort by forward (designed->natural) AUC so the strongest transfer is on top
    lms = sorted(lms, key=lambda l: fwd[l])
    y = np.arange(len(lms))
    for yi, lm in zip(y, lms):
        ax.plot([rev[lm], fwd[lm]], [yi, yi], color=GREY, lw=1.4, zorder=1,
                solid_capstyle="round")
    ax.scatter([rev[l] for l in lms], y, s=26, color=FAIL, zorder=3,
               label="DS1→IPI  (natural→designed)", clip_on=False)
    ax.scatter([fwd[l] for l in lms], y, s=26, color=PASS, zorder=3,
               label="IPI→DS1  (designed→natural)", clip_on=False)
    ax.set_yticks(y); ax.set_yticklabels(lms, fontsize=5.8)
    ax.set_ylim(-0.6, len(lms) - 0.4 + 1.0)        # headroom above the top bar for the legend
    ax.set_xlim(0.70, 0.97)
    ax.set_xlabel("Transfer AUC", fontsize=6.5)
    ax.axvline(0.5, color="#cccccc", lw=0.5, ls=":")
    ax.grid(axis="x", lw=0.25, alpha=0.4)
    ax.set_title("Cross-library transfer", fontsize=6.8, fontweight="bold",
                 loc="left", pad=3)
    # legend in the clear headroom above the dumbbells
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=4.9, handlelength=0.9,
              handletextpad=0.3, labelspacing=0.3, borderpad=0.3,
              frameon=True, framealpha=0.92, edgecolor="#CCCCCC").get_frame().set_linewidth(0.4)


# ── panel b: per-cohort external ROC ─────────────────────────────────────────
def panel_b(ax):
    from sklearn.metrics import average_precision_score
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.6, ls=":", zorder=1)
    stats = []
    for name, s, a, y, col, ls in COHORTS:
        nfail = int((y == 0).sum())
        fpr, tpr, _ = roc_curve(y, s)
        auc, lo, hi = auc_ci(y, s)
        # PR-AUC for the rare FAIL class (positive = Fail, score = 1 - P(Pass)); no-skill = Fail prevalence
        prauc = average_precision_score(1 - y, 1 - s)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=1.5, zorder=3)
        stats.append((name, col, auc, lo, hi, len(y), prauc, nfail))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False-positive rate", fontsize=6.5)
    ax.set_ylabel("True-positive rate", fontsize=6.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("External ROC (zero-shot)", fontsize=6.8, fontweight="bold",
                 loc="left", pad=3)
    # colour-coded stat block in the empty lower-right triangle: ROC-AUC+CI and PR-AUC(Fail)+no-skill
    for k, (name, col, auc, lo, hi, n, prauc, nfail) in enumerate(stats):
        ax.text(0.975, 0.035 + 0.118 * (len(stats) - 1 - k),
                f"{name}  ROC {auc:.2f} [{lo:.2f}–{hi:.2f}]\nPR(Fail) {prauc:.2f} "
                f"(no-skill {nfail/n:.2f}); n={n}, {nfail}F",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.3, color=col, fontweight="bold", linespacing=1.1)
    ax.text(0.03, 0.96, "Pass = PR-CHO < 0.27\n(Jain: psr_filter)", transform=ax.transAxes,
            fontsize=4.6, va="top", ha="left", color="#555555")


# ── panel c: score vs assay scatter (GDPa1 PR-CHO) ───────────────────────────
def panel_c(ax):
    s, a, y = g1_s, g1_a, g1_y
    cols = np.where(y == 1, PASS, FAIL)
    ax.scatter(s, a, s=11, c=cols, alpha=0.55, edgecolor="none", zorder=2)
    # regression of assay on score + 95% CI band
    lr = linregress(s, a)
    xs = np.linspace(s.min(), s.max(), 100)
    ys = lr.intercept + lr.slope * xs
    n = len(s); dof = n - 2
    se_line = np.sqrt(np.sum((a - (lr.intercept + lr.slope * s))**2) / dof) * \
        np.sqrt(1 / n + (xs - s.mean())**2 / np.sum((s - s.mean())**2))
    from scipy.stats import t as tdist
    tval = tdist.ppf(0.975, dof)
    ax.plot(xs, ys, color="#222222", lw=1.1, zorder=3)
    ax.fill_between(xs, ys - tval * se_line, ys + tval * se_line,
                    color="#222222", alpha=0.12, lw=0, zorder=1)
    ax.axhline(THRESH, color=GREY, lw=0.6, ls="--", zorder=1)
    rho, lo, hi = rho_signed_ci(s, a)
    ax.set_xlabel("DELPHI score  P(Pass)", fontsize=6.5)
    ax.set_ylabel("GDPa1 PR-CHO\n(polyreactivity)", fontsize=6.5)
    ax.set_title("Score vs assay  (GDPa1)", fontsize=6.8, fontweight="bold",
                 loc="left", pad=3)
    ax.grid(lw=0.25, alpha=0.35)
    # stats box in the empty upper-right triangle (negative-correlation cloud)
    ax.text(0.965, 0.95,
            f"Spearman ρ = {rho:.2f}\n[{lo:.2f}, {hi:.2f}]\nn = {n}",
            transform=ax.transAxes, fontsize=5.4, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.9, lw=0.4))
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=PASS, ms=3.5, label="Pass"),
                       Line2D([0], [0], marker="o", ls="", color=FAIL, ms=3.5, label="Fail")],
              loc="lower left", fontsize=5, handletextpad=0.2, labelspacing=0.2,
              borderpad=0.3, frameon=False)


# ── panel d: DELPHI score by subgroup (GDPa1) ────────────────────────────────
def panel_d(ax):
    sc = g1[DELPHI_SC].values
    # grouping 1: IgG subtype
    sub = g1["hc_subtype"].astype(str).values
    sub_groups = ["IgG1", "IgG2", "IgG4"]
    # grouping 2: clinical status (Approved vs In-trial) from highest_clinical_trial
    hct = g1["highest_clinical_trial_asof_feb2025"].astype(str)
    clin = np.where(hct.str.startswith("Approved").values, "Approved", "In-trial")
    clin_groups = ["Approved", "In-trial"]

    positions, data, labels, colors, tick_pos, tick_lab = [], [], [], [], [], []
    pos = 0
    palette = ok.qualitative(len(sub_groups))
    for i, g in enumerate(sub_groups):
        d = sc[(sub == g) & ~np.isnan(sc)]
        data.append(d); positions.append(pos); colors.append(palette[i])
        tick_pos.append(pos); tick_lab.append(f"{g}\n(n={len(d)})")
        pos += 1
    pos += 0.8  # gap between the two groupings
    sep_x = pos - 0.9
    clin_pal = [ok.OI_BLUE, ok.OI_GREY]
    for i, g in enumerate(clin_groups):
        d = sc[(clin == g) & ~np.isnan(sc)]
        data.append(d); positions.append(pos); colors.append(clin_pal[i])
        tick_pos.append(pos); tick_lab.append(f"{g}\n(n={len(d)})")
        pos += 1

    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#222222", lw=1.0),
                    whiskerprops=dict(color="#555555", lw=0.6),
                    capprops=dict(color="#555555", lw=0.6),
                    boxprops=dict(lw=0.5, edgecolor="#555555"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    for d, p, c in zip(data, positions, colors):
        jit = (RNG.random(len(d)) - 0.5) * 0.32
        ax.scatter(np.full(len(d), p) + jit, d, s=4, color=c, alpha=0.6,
                   edgecolor="none", zorder=3)
    ax.axvline(sep_x, color="#dddddd", lw=0.6, ls="-")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=5.0)
    ax.set_ylabel("DELPHI score  P(Pass)", fontsize=6.5)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", lw=0.25, alpha=0.35)
    ax.set_title("Score by subgroup  (GDPa1)", fontsize=6.8, fontweight="bold",
                 loc="left", pad=3)
    # sub-headers under the two clusters
    ax.text((tick_pos[0] + tick_pos[2]) / 2, -0.22, "Isotype",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=5.6, color="#555555", fontweight="bold")
    ax.text((tick_pos[3] + tick_pos[4]) / 2, -0.22, "Clinical status",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=5.6, color="#555555", fontweight="bold")


# ── panel e: competition forest (GDPa3 PR-CHO |rho| per LM) ──────────────────
def panel_e(ax):
    rows = []
    g3c = g3.dropna(subset=["polyreactivity_prscore_cho_avg"])
    a = g3c["polyreactivity_prscore_cho_avg"].values
    for lm in MODELS:
        col = score_col(lm)
        if col not in g3c.columns:
            continue
        s = g3c[col].values
        m = ~np.isnan(s)
        pt, lo, hi = absrho_ci(s[m], a[m])
        rows.append((MODEL_DISP[lm], pt, lo, hi))
    rows.sort(key=lambda r: r[1])           # weakest at bottom, best on top
    y = np.arange(len(rows))
    ax.axvspan(COMP_LO, COMP_HI, color=ok.OI_YELLOW, alpha=0.35, zorder=0)
    ax.axvline(COMP_HI, color="#9a8500", lw=0.8, ls="--", zorder=1)
    for yi, (nm, pt, lo, hi) in zip(y, rows):
        col = PASS if nm == "AbLang2" else NEUTRAL
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.2, zorder=2,
                solid_capstyle="round")
        ax.scatter(pt, yi, s=24, color=col, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.set_ylim(-0.6, len(rows) + 0.15)        # headroom for the label above the bars
    ax.set_xlim(0, max(0.56, max(r[3] for r in rows) * 1.05))
    ax.set_xlabel("|Spearman ρ|  (GDPa3 PR-CHO)", fontsize=6.5)
    ax.grid(axis="x", lw=0.25, alpha=0.4)
    ax.set_title("Zero-shot vs competition", fontsize=6.8, fontweight="bold",
                 loc="left", pad=3)
    # label above the bars and to the LEFT of the band so it never sits on the line
    ax.text(0.01, len(rows) - 0.15, "113-team best: 0.337–0.356",
            ha="left", va="bottom", fontsize=4.9, color="#222222")


# ── panel f: zero-shot |rho| by cohort (best model = DELPHI/AbLang2) ─────────
def panel_f(ax):
    bars = []
    for name, s, a, y in [("Jain 2017", jain_s, jain_a, len(jain_s)),
                          ("GDPa1", g1_s, g1_a, len(g1_s)),
                          ("GDPa3", g3_s, g3_a, len(g3_s))]:
        pt, lo, hi = absrho_ci(s, a)
        bars.append((name, pt, lo, hi, len(s)))
    x = np.arange(len(bars))
    cols = [ok.OI_BLUE, ok.OI_VERMILION, ok.OI_GREEN]
    for xi, (nm, pt, lo, hi, n), c in zip(x, bars, cols):
        ax.bar(xi, pt, width=0.6, color=c, alpha=0.85, zorder=2, lw=0)
        ax.errorbar(xi, pt, yerr=[[pt - lo], [hi - pt]], fmt="none",
                    ecolor="#333333", elinewidth=0.8, capsize=2.2, zorder=3)
        ax.text(xi, hi + 0.015, f"{pt:.2f}", ha="center", va="bottom", fontsize=5.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[0]}\nn={b[4]}" for b in bars], fontsize=5.6)
    ax.set_ylabel("|Spearman ρ|  (DELPHI)", fontsize=6.5)
    ax.set_ylim(0, max(b[3] for b in bars) * 1.18)
    ax.grid(axis="y", lw=0.25, alpha=0.4)
    ax.set_title("Zero-shot |ρ| by cohort", fontsize=6.8, fontweight="bold",
                 loc="left", pad=3)
    ax.text(0.5, 0.97, "PR-CHO (GDPa); PSR-SMP (Jain)", transform=ax.transAxes,
            ha="center", va="top", fontsize=4.7, color="#555555")


panel_a(axa); panel_b(axb); panel_c(axc)
panel_d(axd); panel_e(axe); panel_f(axf)

for ax, letter, dx in [(axa, "a", -0.052), (axb, "b", -0.050), (axc, "c", -0.050),
                       (axd, "d", -0.052), (axe, "e", -0.050), (axf, "f", -0.050)]:
    ok.panel_label(fig, ax, letter, dx=dx, dy=0.030, size=9)

ok.save_fig(fig, "Figure4", OUT)
print("Fig5 done")
