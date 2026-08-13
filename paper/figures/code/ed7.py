"""
Extended Data Figure 8 — per-antibody interpretability.
  rows: PSR | SEC
  cols: a/d  example PASS antibody — per-residue IG waterfall
        b/e  example FAIL antibody — per-residue IG waterfall
        c/f  same FAIL antibody    — CDR3 single-point mutagenesis heatmap

Runs the DELPHI one-hot Transformer (pretrained_202605) directly:
  * IG (waterfalls) via captum IntegratedGradients — exactly the call the original
    `_waterfall_single_ab` uses (baselines=zero, target=1).
  * Mutagenesis via model.predict_single over every position × 20 amino acids — the
    exact computation in the original `_render_mutagenesis_heatmap`.
Both come from the SAME loaded model, so the panels are internally consistent. No
values invented. Example antibodies are selected by predicted score (PASS = highest
true-pass; FAIL = a true-fail in P(Pass) 0.2-0.45, the informative-mutagenesis band).
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO_ROOT, DATA_ROOT, data_file, ensure_output
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
import numpy as np, pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from captum.attr import IntegratedGradients
from models.transformer_onehot import TransformerOneHotModel, one_hot_encode_sequence_2d
import okabe_style as ok
warnings.filterwarnings("ignore")

MODELS = str(DATA_ROOT / "local_only" / "models")
OUT = str(ensure_output())
ok.set_style(base_pt=6.5)
AMINO = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AMINO)}
MUT_CMAP = ok.DIVERGING.reversed()   # P(Pass): 0->vermilion(Fail), 0.5->white, 1->blue(Pass)
IG_STEPS = 100


def seqs(d):
    vh = str(d.HSEQ).upper().replace("-", "")
    vl = str(d.LSEQ).upper().replace("-", "") if "LSEQ" in d.index else ""
    cdr3 = str(d.CDR3).upper().replace("-", "")
    return vh, vl, cdr3


def compute_ig(m, vh, vl, cdr3):
    # torch.from_numpy can fail in environments where PyTorch and NumPy were
    # loaded from different ABI builds. Constructing from the nested values is
    # slower for these tiny tensors but portable and numerically identical.
    def as_float_tensor(encoded):
        return torch.tensor(encoded.tolist(), dtype=torch.float32)

    enc_h = as_float_tensor(one_hot_encode_sequence_2d(vh, m.max_heavy_len))
    enc_c = as_float_tensor(one_hot_encode_sequence_2d(cdr3, m.max_hcdr3_len))
    if m._vh_only():
        enc_hl = enc_h.unsqueeze(0).to(m.device)
    else:
        enc_l = as_float_tensor(one_hot_encode_sequence_2d(vl or "", m.max_light_len))
        enc_hl = torch.cat([enc_h, enc_l], 0).unsqueeze(0).to(m.device)
    enc_c = enc_c.unsqueeze(0).to(m.device)
    m.model.eval()
    attr = IntegratedGradients(m.model).attribute(
        (enc_hl, enc_c), baselines=(torch.zeros_like(enc_hl), torch.zeros_like(enc_c)),
        target=1, n_steps=IG_STEPS)
    return attr[0].squeeze(0).detach().cpu().numpy(), attr[1].squeeze(0).detach().cpu().numpy()


def waterfall_rows(attr_enc, attr_cdr3, vh, cdr3, n_vh=5):
    rows = []  # (label, ig)
    for pos, aa in enumerate(cdr3):
        if pos >= attr_cdr3.shape[0]: break
        ig = float(attr_cdr3[pos, AA_IDX[aa]]) if aa in AA_IDX else 0.0
        rows.append((f"CDR3 {pos+1:02d} {aa}", ig))
    vh_rows = [(f"VH {pos+1} {aa}", float(attr_enc[pos, AA_IDX[aa]]))
               for pos, aa in enumerate(vh) if pos < attr_enc.shape[0] and aa in AA_IDX]
    vh_rows = sorted(vh_rows, key=lambda r: abs(r[1]), reverse=True)[:n_vh]
    return rows + vh_rows   # HCDR3 block (sequence order) then top VH


def mutagenesis(m, bc, vh, vl, cdr3):
    n = len(cdr3); H = np.full((len(AMINO), n), np.nan, dtype=np.float32)
    cs = vh.find(cdr3)
    for pos in range(n):
        for ai, aa in enumerate(AMINO):
            mc = cdr3[:pos] + aa + cdr3[pos+1:]
            vm = vh[:cs] + mc + vh[cs+n:] if cs >= 0 else vh
            H[ai, pos] = m.predict_single(bc, vm, vl, mc)
    return H


def pick_examples(m, df, fcol, seed=0, max_len=18):
    rng = np.random.default_rng(seed)
    pid = df[df[fcol] == 1].index.values; fid = df[df[fcol] == 0].index.values
    pc = rng.choice(pid, min(250, len(pid)), replace=False)
    fc = rng.choice(fid, min(450, len(fid)), replace=False)
    def sc(bc):
        vh, vl, cdr3 = seqs(df.loc[bc]); return m.predict_single(bc, vh, vl, cdr3), len(cdr3)
    ps = [(bc, *sc(bc)) for bc in pc]
    fs = [(bc, *sc(bc)) for bc in fc]
    pass_bc = max([r for r in ps if r[2] <= max_len] or ps, key=lambda r: r[1])[0]
    band = [r for r in fs if 0.2 <= r[1] <= 0.45 and r[2] <= max_len]
    fail_bc = (min(band, key=lambda r: abs(r[1] - 0.33)) if band
               else min(fs, key=lambda r: abs(r[1] - 0.33)))[0]
    return pass_bc, fail_bc


def draw_waterfall(ax, rows, prob, true_lab, bc):
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    y = np.arange(len(rows))[::-1]
    cols = [ok.PASS if v >= 0 else ok.FAIL for v in vals]
    ax.barh(y, vals, color=cols, height=0.7, lw=0)
    ax.axvline(0, color="#888888", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=4.4, fontfamily="monospace")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("IG  (← Fail | Pass →)", fontsize=6.0)
    out = "PASS" if true_lab == 1 else "FAIL"
    ax.set_title(f"{bc} · actual {out}\nP(Pass) = {prob:.3f}", fontsize=6.0, fontweight="bold", pad=3)


def draw_mut(ax, H, cdr3, fig):
    im = ax.imshow(H, cmap=MUT_CMAP, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    for pos, aa in enumerate(cdr3):
        if aa in AA_IDX:
            ax.plot(pos, AA_IDX[aa], "s", ms=2.6, mfc="#000000", mec="white", mew=0.4, zorder=5)
    ax.set_xticks(range(len(cdr3))); ax.set_xticklabels(list(cdr3), fontsize=5.0, fontfamily="monospace")
    ax.set_yticks(range(len(AMINO))); ax.set_yticklabels(list(AMINO), fontsize=4.6, fontfamily="monospace")
    ax.set_xlabel("CDR3 position (WT residue)", fontsize=6.0); ax.set_ylabel("Mutant AA", fontsize=6.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["Fail", "0.5", "Pass"], fontsize=5)
    cb.set_label("P(Pass) after substitution", fontsize=5.4)


# ── build ─────────────────────────────────────────────────────────────────────
print("loading models + data ...")
m_psr = TransformerOneHotModel.load(f"{MODELS}/FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt")
m_sec = TransformerOneHotModel.load(f"{MODELS}/FINAL_sec_filter_onehot_transformer_onehot_ipi_sec_5000.pt")
psr = pd.read_excel(data_file("ipi_psr_trainset.xlsx")).dropna(subset=["psr_filter"]).set_index("BARCODE")
sec = pd.read_excel(data_file("ipi_sec_5000.xlsx")).dropna(subset=["sec_filter"]).set_index("BARCODE")

fig = plt.figure(figsize=(ok.DOUBLE, 165 * ok.MM))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.25], left=0.075, right=0.95,
              top=0.91, bottom=0.075, hspace=0.40, wspace=0.55)
letters = [["a", "b", "e"], ["c", "d", "f"]]  # legend groups waterfalls (a-d) then mutagenesis (e,f)

for ri, (m, df, fcol, name) in enumerate([(m_psr, psr, "psr_filter", "PSR"),
                                          (m_sec, sec, "sec_filter", "SEC")]):
    # pinned to the two antibodies named in the manuscript ED Fig. 8 legend: pass TAB0012562,
    # fail TAB0015016 — both present and correctly labeled (Pass / Fail) in the PSR and SEC
    # datasets. These are BARCODE identifiers, the same ones printed in the manuscript; the
    # sequences they resolve to are read from the (non-committed) local datasets, not stored here.
    pass_bc, fail_bc = "TAB0012562", "TAB0015016"
    print(f"{name}: pass={pass_bc} (P={m.predict_single(pass_bc, *seqs(df.loc[pass_bc])):.3f}) "
          f"fail={fail_bc} (P={m.predict_single(fail_bc, *seqs(df.loc[fail_bc])):.3f})")
    for ci, bc in enumerate([pass_bc, fail_bc]):
        d = df.loc[bc]; vh, vl, cdr3 = seqs(d)
        ae, ac = compute_ig(m, vh, vl, cdr3)
        prob = m.predict_single(bc, vh, vl, cdr3)
        ax = fig.add_subplot(gs[ri, ci])
        draw_waterfall(ax, waterfall_rows(ae, ac, vh, cdr3), prob, int(d[fcol]), bc)
        ok.panel_label(fig, ax, letters[ri][ci], dx=-0.058, dy=0.022, size=8.5)
    # mutagenesis on the FAIL example
    d = df.loc[fail_bc]; vh, vl, cdr3 = seqs(d)
    H = mutagenesis(m, fail_bc, vh, vl, cdr3)
    axm = fig.add_subplot(gs[ri, 2])
    draw_mut(axm, H, cdr3, fig)
    axm.set_title(f"CDR3 mutagenesis · {name}\n{fail_bc} (actual FAIL)", fontsize=6.0, fontweight="bold", pad=3)
    ok.panel_label(fig, axm, letters[ri][2], dx=-0.06, dy=0.022, size=8.5)

fig.text(0.5, 0.965, "Per-antibody attribution — DELPHI one-hot Transformer",
         ha="center", fontsize=7.5, fontweight="bold")
fig.legend(handles=[Line2D([0], [0], color=ok.PASS, lw=5, label="IG → Pass"),
                    Line2D([0], [0], color=ok.FAIL, lw=5, label="IG → Fail")],
           loc="lower left", bbox_to_anchor=(0.075, 0.005), ncol=2, fontsize=6, frameon=False)
ok.save_fig(fig, "ED_Fig8", OUT)   # renumbered: per-antibody interpretation is now Extended Data Fig. 8
print("ED(per-antibody -> ED_Fig8) done")
