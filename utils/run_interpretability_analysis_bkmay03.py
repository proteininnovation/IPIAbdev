#!/usr/bin/env python3
"""
run_interpretability_analysis.py
─────────────────────────────────────────────────────────────────────────────
MLAbDev — Nature Biotech interpretability figure generator.

Given a training database (e.g. ipi_psr_trainset.xlsx), this script:

  1. Auto-locates FINAL_*.pkl / FINAL_*.pt checkpoints in MODEL_DIR for:
       • Random Forest + biophysical features        (SHAP)
       • XGBoost       + biophysical features        (SHAP)
       • Transformer   + one-hot sequences           (Integrated Gradients)

  2. Computes interpretability attributions on the full training set.

  3. Renders a 5-panel Extended-Data-style figure matching Nature Biotech
     style (same specs as Extended Figs 2/3/4):
       • 6.3 × 10.2 inch  (160 × 260 mm)  — single-column, stacks 5 panels
       • 300 DPI TIFF (LZW) + 300 DPI PDF (vector) + 150 DPI PNG
       • DejaVu Sans 6pt / 0.5pt axis linewidth
       • Panel letters (a, b, c, d, e) lowercase

  4. Writes raw attribution arrays (CSV + NPZ) alongside the figure.

Usage
─────
    # PSR figure
    python run_interpretability_analysis.py \
        --db data/ipi_psr_trainset.xlsx --target psr_filter

    # SEC figure
    python run_interpretability_analysis.py \
        --db data/ipi_sec_trainset.xlsx --target sec_filter

    # Custom LMs per model (defaults shown):
    python run_interpretability_analysis.py \
        --db data/ipi_psr_trainset.xlsx --target psr_filter \
        --rf-lm biophysical --xgb-lm biophysical \
        --transformer-lm onehot \
        --model-dir build/pretrained_models \
        --outdir outputs/interp_psr

Output naming convention (matches predict_developability.py)
────────────────────────────────────────────────────────────
    interp_{target}_{rf_lm}_{xgb_lm}_{tr_lm}_{db_stem}.{tiff|pdf|png}
    shap_rf_{target}_{rf_lm}_{db_stem}.csv
    shap_xgb_{target}_{xgb_lm}_{db_stem}.csv
    ig_{target}_{tr_lm}_{db_stem}.npz
    region_attribution_{target}_{db_stem}.csv
    interp_log_{target}_{db_stem}.txt

If a model is missing, the corresponding panel is rendered blank with a
note, and the script continues — so you can run it before all three
models are trained.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path so models.* imports work ─────────────
# Walk up from this file until we find a directory containing `models/` —
# that's the project root. Works whether the script lives in utils/, scripts/,
# the project root, or anywhere else.
_HERE = Path(__file__).resolve().parent
_search = _HERE
for _ in range(6):                              # walk up to 6 levels
    if (_search / "models").is_dir():
        sys.path.insert(0, str(_search))
        break
    if _search.parent == _search:               # hit filesystem root
        break
    _search = _search.parent
else:
    # Fallback — also cover current working directory in case user runs
    # from project root but the script lives elsewhere (symlink / install).
    sys.path.insert(0, str(Path.cwd()))


# ══════════════════════════════════════════════════════════════════════════════
# NATURE BIOTECH STYLING — matches Extended Figures 2/3/4 exactly
# ══════════════════════════════════════════════════════════════════════════════

# Single-column, 5 stacked panels. Panel E narrower than A-D.
FIG_WIDTH_IN  = 6.3
FIG_HEIGHT_IN = 11.0
DPI_TIFF = 300
DPI_PDF  = 300
DPI_PNG  = 150

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":          6,
    "axes.titlesize":     8,
    "axes.labelsize":     7,
    "xtick.labelsize":    5,
    "ytick.labelsize":    5,
    "legend.fontsize":    5,
    "figure.titlesize":   9,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2,
    "ytick.major.size":   2,
    "pdf.fonttype":      42,   # TrueType — editable in Illustrator
    "ps.fonttype":       42,
})

# Colour scheme — same palette as existing ED figures for consistency
COLOR_RF     = '#1f77b4'   # blue
COLOR_XGB    = '#ff7f0e'   # orange
COLOR_TRANS  = '#2ca02c'   # green
COLOR_CDR3   = '#C94A8C'   # magenta — HCDR3 (paper's key region)
COLOR_VH_FR  = '#378ADD'   # light blue — VH framework
COLOR_VL_FR  = '#F5B942'   # yellow-orange — VL framework
COLOR_VH_ALL = '#7F77DD'   # muted purple — whole VH (no region split)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# ── AA physicochemical classes — 4 categories ────────────────────────────────
_AA_CLASS = {
    'R': 0, 'K': 0, 'H': 0,                         # cationic
    'D': 1, 'E': 1,                                  # anionic
    'W': 2, 'F': 2, 'Y': 2, 'L': 2, 'I': 2, 'V': 2, 'M': 2,  # hydrophobic
    'A': 3, 'G': 3, 'S': 3, 'T': 3, 'C': 3, 'P': 3, 'Q': 3, 'N': 3,
}
_AA_CLASS_COLORS = ['#C0392B', '#2471A3', '#E67E22', '#AAAAAA']
_AA_CLASS_LABELS = [
    'Cationic (R,K,H) — +charge',
    'Anionic (D,E) — −charge',
    'Hydrophobic/Aromatic (W,F,Y,L,I,V,M)',
    'Small/Polar (A,G,S,T,C,P,Q,N)',
]

def _aa_color(aa): return _AA_CLASS_COLORS[_AA_CLASS.get(aa, 3)]

def _aa_class_legend():
    import matplotlib.patches as _mp
    return [_mp.Patch(color=c, label=l)
            for c,l in zip(_AA_CLASS_COLORS, _AA_CLASS_LABELS)]


# ══════════════════════════════════════════════════════════════════════════════
# LOG HELPER
# ══════════════════════════════════════════════════════════════════════════════

class _Log:
    """Simple tee to stdout + file. Identical to predict_developability.py style."""
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, 'w', buffering=1, encoding='utf-8')
        self.path = path

    def __call__(self, *msg):
        text = " ".join(str(m) for m in msg)
        print(text)
        self._f.write(text + "\n")

    def close(self):
        if self._f and not self._f.closed:
            self._f.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY — matches predict_developability.py naming exactly:
#   FINAL_{target}_{lm}_{model}_{db_stem}{_regression?}{ext}
#     ext = .pkl for rf/xgboost, .pt for transformer_onehot
# ══════════════════════════════════════════════════════════════════════════════

def _find_final(model_dir: str, target: str, lm: str, model_type: str,
                db_stem: str, ext: str) -> Optional[str]:
    """
    Find a FINAL_* checkpoint. Preference order:
      1. exact match:          FINAL_{target}_{lm}_{model}_{db_stem}{ext}
      2. exact + _regression:  FINAL_{target}_{lm}_{model}_{db_stem}_regression{ext}
      3. any suffix match:     FINAL_{target}_{lm}_{model}_{db_stem}*{ext}
      4. None
    """
    stem = f"FINAL_{target}_{lm}_{model_type}_{db_stem}"
    candidates = [
        os.path.join(model_dir, f"{stem}{ext}"),
        os.path.join(model_dir, f"{stem}_regression{ext}"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback: glob for any suffix (e.g. _lora8, _ft_*)
    pattern = os.path.join(model_dir, f"{stem}*{ext}")
    found = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return found[0] if found else None


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — keep it minimal & aligned with predict_developability.py
# ══════════════════════════════════════════════════════════════════════════════

def _load_db(db_path: str, target: str, log: _Log) -> pd.DataFrame:
    """Load training database, set BARCODE index, drop rows missing target."""
    ext = Path(db_path).suffix.lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(db_path)
    elif ext == '.csv':
        df = pd.read_csv(db_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    log(f"[load] {Path(db_path).name}: {len(df):,} rows × {len(df.columns)} cols")

    if 'BARCODE' not in df.columns:
        df['BARCODE'] = range(len(df))
    df['BARCODE'] = df['BARCODE'].astype(str).str.strip()

    for c in ['HSEQ', 'LSEQ', 'CDR3']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)

    n_before = len(df)
    df = df.dropna(subset=[target]).set_index('BARCODE')
    log(f"[load] after dropna({target}): {len(df):,} rows (−{n_before - len(df):,})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SHAP COMPUTATION — reuses the model's own fb_ (FeatureBuilder) + TreeExplainer
# ══════════════════════════════════════════════════════════════════════════════

def _compute_tree_shap(model, X_df: pd.DataFrame, max_samples: int, log: _Log
                       ) -> Optional[dict]:
    """
    Compute SHAP values on non-embedding (biophysical / kmer / onehot) features.

    Returns dict with keys:
      names         : list[str]          feature names (non-embedding only)
      mean_abs_shap : np.ndarray (F,)    mean |SHAP| per feature
      shap_matrix   : np.ndarray (n, F)  raw SHAP for class-1
      X_matrix      : np.ndarray (n, F)  feature values (for beeswarm colour)
      expected      : float              baseline
    or None if SHAP can't be computed.
    """
    try:
        import shap as _shap
    except ImportError:
        log("[SHAP] shap not installed — pip install shap"); return None

    if getattr(model, 'fb_', None) is None:
        log("[SHAP] model has no FeatureBuilder (fb_ missing) — skipping")
        return None

    fb = model.fb_
    ne_idx   = fb.non_embedding_indices
    ne_names = fb.non_embedding_feature_names
    if not ne_idx:
        log("[SHAP] embedding-only mode — no interpretable features. "
            "Use --rf-lm biophysical or --rf-lm kmer.")
        return None

    # Transform full dataset to feature matrix
    X_feat = fb.transform(X_df, None)   # seq-only mode — embeddings=None
    X_ne   = X_feat[:, np.array(ne_idx)]

    # Subsample if huge
    if len(X_ne) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_ne), max_samples, replace=False)
        X_shap = X_ne[idx]
        log(f"[SHAP] subsampled {max_samples:,} of {len(X_ne):,} rows")
    else:
        X_shap = X_ne

    explainer = _shap.TreeExplainer(model.model)
    try:
        sv_raw = explainer.shap_values(X_shap, check_additivity=False)
    except Exception as e:
        log(f"[SHAP] shap_values failed: {e}"); return None

    # Normalise SHAP output shape — varies across shap versions
    if isinstance(sv_raw, list):
        sv = np.asarray(sv_raw[1], dtype=np.float64)
    elif isinstance(sv_raw, np.ndarray):
        if sv_raw.ndim == 3:
            sv = (sv_raw[:, :, 1] if sv_raw.shape[0] == len(X_shap)
                  else sv_raw[1])
        else:
            sv = sv_raw
    else:
        sv = np.asarray(sv_raw, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    if sv.ndim != 2:
        log(f"[SHAP] unexpected shape {sv.shape} — skipping"); return None

    expected = float(
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    return {
        'names':         ne_names,
        'mean_abs_shap': np.mean(np.abs(sv), axis=0),
        'shap_matrix':   sv,
        'X_matrix':      X_shap,
        'expected':      expected,
    }


# ══════════════════════════════════════════════════════════════════════════════
# IG COMPUTATION — uses transformer_onehot.TransformerOneHotModel.global_ig_analysis
# but we re-implement the accumulation here so we can save arrays.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ig(model, df: pd.DataFrame, n_samples: int, n_steps: int,
                log: _Log) -> Optional[dict]:
    """
    Compute Integrated Gradients on the trained Transformer-onehot model.

    Returns dict with:
      attr_enc     : (n, L1, 20)   signed IG on VH+VL (or VH) branch
      attr_cdr3    : (n, 25, 20)   signed IG on HCDR3 branch
      hcdr3_seqs   : list[str]
      labels, probs, preds
    or None on failure.
    """
    try:
        import torch
        from captum.attr import IntegratedGradients
        from models.transformer_onehot import AntibodyDataset
        from torch.utils.data import DataLoader
    except ImportError as e:
        log(f"[IG] cannot import captum/torch/models — {e}"); return None

    fb_model = model.model
    if fb_model is None:
        log("[IG] model.model is None — not loaded properly"); return None

    vh_only = model._vh_only() if hasattr(model, '_vh_only') else False

    # Select balanced subsample (preserves class distribution for IG)
    n_avail = len(df)
    if n_samples > 0 and n_avail > n_samples:
        rng = np.random.default_rng(42)
        if 'label' in df.columns:
            by_cls = [df[df['label'] == c].index.tolist() for c in (0, 1)]
        else:
            by_cls = [df.index.tolist()]
        half = n_samples // max(len(by_cls), 1)
        chosen = []
        for bucket in by_cls:
            k = min(half, len(bucket))
            chosen.extend(rng.choice(bucket, size=k, replace=False).tolist())
        df = df.loc[chosen]
    log(f"[IG] using {len(df):,} antibodies  n_steps={n_steps}  vh_only={vh_only}")

    heavy = df['HSEQ'].tolist()
    light = [''] * len(df) if vh_only else df['LSEQ'].tolist()
    hcdr3 = df['CDR3'].tolist()
    # For IG we just need to build the dataset — labels are only for colouring
    labels   = np.zeros(len(df), dtype=np.int64)
    barcodes = df.index.astype(str).tolist()

    ds = AntibodyDataset(heavy, light, hcdr3, labels, barcodes,
                         model.max_heavy_len, model.max_light_len,
                         model.max_hcdr3_len, vh_only=vh_only)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    fb_model.eval()
    ig = IntegratedGradients(fb_model)

    attrs_enc  = []
    attrs_cdr3 = []
    probs_all  = []
    with torch.no_grad():
        pass   # IG itself needs grad; torch.no_grad() just here for structure

    for enc, cdr3_enc, lbl, *_ in loader:
        enc      = enc.to(model.device)
        cdr3_enc = cdr3_enc.to(model.device)

        # Baseline = zeros (standard for one-hot; Captum/Alibi default)
        base_enc  = torch.zeros_like(enc)
        base_cdr3 = torch.zeros_like(cdr3_enc)

        attr = ig.attribute(
            (enc, cdr3_enc),
            baselines=(base_enc, base_cdr3),
            target=1, n_steps=n_steps
        )
        attrs_enc.append(attr[0].detach().cpu().numpy())
        attrs_cdr3.append(attr[1].detach().cpu().numpy())

        # Prediction probability for record
        with torch.no_grad():
            logits = fb_model(enc, cdr3_enc)
            p1 = torch.softmax(logits, dim=1)[:, 1]
        probs_all.extend(p1.cpu().numpy().tolist())

    attr_enc  = np.concatenate(attrs_enc,  axis=0)   # (n, L1, 20)
    attr_cdr3 = np.concatenate(attrs_cdr3, axis=0)   # (n, 25, 20)
    probs     = np.asarray(probs_all, dtype=np.float64)

    log(f"[IG] done  attr_enc={attr_enc.shape}  attr_cdr3={attr_cdr3.shape}")

    return {
        'attr_enc':   attr_enc,
        'attr_cdr3':  attr_cdr3,
        'hcdr3_seqs': hcdr3,
        'barcodes':   barcodes,
        'probs':      probs,
        'vh_only':    vh_only,
        'max_vh':     model.max_heavy_len,
        'max_vl':     model.max_light_len,
        'max_cdr3':   model.max_hcdr3_len,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE-NAME → REGION LABEL (for Panel E)
#
# Biophysical feature names (from randomforest.py / xgboost.py):
#   cdr3_*      → HCDR3
#   vh_*        → VH framework (whole chain minus CDR3)
#   single AA   → CDR3-loop composition (counted on stripped loop)
#
# K-mer feature names:
#   1mer_X, 2mer_XY, 3mer_XYZ → source depends on kmer.sequence
#     CDR3 → HCDR3   VH → VH (framework)   VHVL → VH+VL
#
# One-hot feature names:
#   oh_hcdr3_{pos}_{AA} → HCDR3
#   oh_vh_{pos}_{AA}    → VH framework
#   oh_vl_{pos}_{AA}    → VL framework
# ══════════════════════════════════════════════════════════════════════════════

def _feature_region(name: str, kmer_source: str = 'CDR3') -> str:
    """Return one of: 'HCDR3', 'VH', 'VL', 'other'."""
    if name.startswith('cdr3_'):       return 'HCDR3'
    if name.startswith('vh_'):         return 'VH'
    if name.startswith('oh_hcdr3'):    return 'HCDR3'
    if name.startswith('oh_cdr3'):     return 'HCDR3'
    if name.startswith('oh_vh'):       return 'VH'
    if name.startswith('oh_vl'):       return 'VL'
    if re.match(r'^\d+mer_', name):
        # k-mer region depends on which sequence it was computed from
        src = kmer_source.upper()
        if src == 'CDR3': return 'HCDR3'
        if src == 'VH':   return 'VH'
        if src == 'VHVL': return 'VH+VL'   # indistinguishable from name alone
        return 'HCDR3'
    # Single amino acid (e.g. 'R', 'K' in biophysical feature list)
    if len(name) == 1 and name in AMINO_ACIDS:
        return 'HCDR3'   # biophysical single-AA features are on stripped CDR3 loop
    return 'other'


def _region_attribution_tree(shap_data: dict, kmer_source: str = 'CDR3') -> dict:
    """Aggregate mean |SHAP| by region. Returns {region: fraction}."""
    names  = shap_data['names']
    ma_shap = shap_data['mean_abs_shap']
    tot = float(ma_shap.sum()) or 1.0
    buckets = {}
    for n, v in zip(names, ma_shap):
        r = _feature_region(n, kmer_source)
        buckets[r] = buckets.get(r, 0.0) + float(v)
    return {r: v / tot for r, v in buckets.items()}


def _region_attribution_ig(ig_data: dict) -> dict:
    """
    Aggregate mean |IG| by region for the Transformer model.
    Branch-1 is split into VH (framework) and VL (framework) sections.
    """
    # |IG| per position (summed over AA dim), averaged over antibodies
    pos_enc  = np.abs(ig_data['attr_enc']).sum(axis=-1).mean(axis=0)   # (L1,)
    pos_cdr3 = np.abs(ig_data['attr_cdr3']).sum(axis=-1).mean(axis=0)  # (25,)

    if ig_data['vh_only']:
        vh_mass = float(pos_enc.sum())
        vl_mass = 0.0
    else:
        vh_mass = float(pos_enc[:ig_data['max_vh']].sum())
        vl_mass = float(pos_enc[ig_data['max_vh']:].sum())
    cdr3_mass = float(pos_cdr3.sum())
    tot = vh_mass + vl_mass + cdr3_mass or 1.0
    return {
        'HCDR3': cdr3_mass / tot,
        'VH':    vh_mass   / tot,
        'VL':    vl_mass   / tot,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PANEL RENDERERS — each takes an axis + data, returns None.
# Each panel guards against missing data and renders a blank-with-note.
# ══════════════════════════════════════════════════════════════════════════════

def _render_blank(ax, note: str, title: str):
    ax.text(0.5, 0.5, note, ha='center', va='center',
            transform=ax.transAxes, fontsize=7, color='#888',
            style='italic', wrap=True)
    ax.set_title(title, fontsize=8, loc='left', fontweight='bold', pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _panel_shap_bar(ax, shap_data: Optional[dict], title: str,
                    top_n: int = 20, kmer_source: str = 'CDR3'):
    """Horizontal bar — top-N mean |SHAP|, colour-coded by region."""
    if shap_data is None:
        _render_blank(ax,
                      "Model not found — rerun predict_developability.py\n"
                      "    --train --lm biophysical  to generate.",
                      title)
        return

    names   = shap_data['names']
    ma_shap = shap_data['mean_abs_shap']
    order   = np.argsort(ma_shap)[::-1][:top_n]
    labels  = [names[i] for i in order]
    vals    = ma_shap[order]

    region_color = {'HCDR3': COLOR_CDR3, 'VH': COLOR_VH_FR,
                    'VL': COLOR_VL_FR, 'VH+VL': COLOR_VH_ALL, 'other': '#bbb'}
    regions = [_feature_region(l, kmer_source) for l in labels]
    colors  = [region_color.get(r, '#bbb') for r in regions]

    pretty = [l.replace('cdr3_charge_ph7', 'CDR3 charge')
               .replace('cdr3_', 'CDR3 ').replace('vh_', 'VH ')
               .replace('_', ' ').strip() for l in labels]

    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(pretty, fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|', fontsize=6)
    ax.set_title(title, fontsize=8, loc='left', fontweight='bold', pad=4)
    ax.tick_params(axis='x', labelsize=5)
    ax.grid(axis='x', alpha=0.25, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    present = []
    for r in ['HCDR3', 'VH', 'VL', 'VH+VL']:
        if r in regions:
            present.append(mpatches.Patch(color=region_color[r], label=r))
    if present:
        ax.legend(handles=present, loc='lower right', fontsize=5,
                  frameon=False, handlelength=1.0, handleheight=0.8)


def _panel_shap_beeswarm(ax, shap_data: Optional[dict], title: str,
                         top_n: int = 20, kmer_source: str = 'CDR3'):
    """
    SHAP beeswarm — shows BOTH magnitude AND direction of each feature.

    Dot position on x-axis  = SHAP value
      negative (left)  → feature pushes prediction toward FAIL (PSR/SEC positive)
      positive (right) → feature pushes prediction toward PASS

    Dot colour = feature value (RdBu_r: red = high, blue = low)
      e.g. CDR3 charge: red dots on RIGHT  → high charge protects (→ PASS)
           CDR3 R:       red dots on LEFT   → many Arg drives polyreactivity (→ FAIL)

    Each row = one feature (ordered by mean |SHAP|, highest at top).
    Dots are jittered vertically within the row to show density.
    """
    if shap_data is None:
        _render_blank(ax,
                      "Model not found — rerun predict_developability.py\n"
                      "    --train --lm biophysical  to generate.",
                      title)
        return

    names    = shap_data['names']
    ma_shap  = shap_data['mean_abs_shap']
    sv       = shap_data['shap_matrix']    # (n, F)
    Xm       = shap_data['X_matrix']       # (n, F) — feature values for colour

    order  = np.argsort(ma_shap)[::-1][:top_n]
    n_feat = len(order)

    # Humanise labels
    def _pretty(l):
        l = (l.replace('cdr3_charge_ph7', 'CDR3 charge')
              .replace('cdr3_', 'CDR3 ').replace('vh_', 'VH ')
              .replace('_ph7', '').replace('_', ' '))
        return l.strip()

    pretty  = [_pretty(names[i]) for i in order]
    rng     = np.random.default_rng(0)

    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    cmap    = _cm.RdBu_r
    dot_s   = 3      # point size — small for NB density
    alpha   = 0.55

    for row, feat_idx in enumerate(order):
        sv_col  = sv[:, feat_idx]             # SHAP values for this feature
        xv_col  = Xm[:, feat_idx]             # raw feature values

        # Normalise feature value to [0, 1] for colour mapping
        lo, hi  = xv_col.min(), xv_col.max()
        span    = hi - lo
        if span > 0:
            norm_xv = (xv_col - lo) / span
        else:
            norm_xv = np.full_like(xv_col, 0.5)

        # Vertical jitter within the row — amount proportional to n samples
        n       = len(sv_col)
        jitter  = rng.uniform(-0.38, 0.38, size=n)

        colors_dot = cmap(norm_xv)

        ax.scatter(sv_col, row + jitter,
                   c=colors_dot, s=dot_s, alpha=alpha,
                   linewidths=0, rasterized=True, zorder=2)

    # Zero-SHAP reference line
    ax.axvline(0, color='#999', lw=0.5, ls='--', zorder=1)

    # Axis labels and direction annotation
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(pretty, fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value  (← toward FAIL  |  toward PASS →)', fontsize=6)
    ax.set_title(title, fontsize=8, loc='left', fontweight='bold', pad=4)
    ax.tick_params(axis='x', labelsize=5)
    ax.grid(axis='x', alpha=0.20, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Colourbar — feature value (low → high)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=_mc.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.025,
                        aspect=30, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=4)
    cbar.set_label('Feature value', fontsize=4.5, labelpad=2)
    cbar.ax.tick_params(width=0.4, length=1.5)


def _panel_ig_positions(ax, ig_data: Optional[dict], title: str):
    """Per-position mean |IG| across VH / VL / HCDR3."""
    if ig_data is None:
        _render_blank(ax,
                      "Transformer (onehot) model not found.\n"
                      "rerun predict_developability.py --train --lm onehot\n"
                      "    --model transformer_onehot", title)
        return

    pos_enc  = np.abs(ig_data['attr_enc']).sum(axis=-1).mean(axis=0)
    pos_cdr3 = np.abs(ig_data['attr_cdr3']).sum(axis=-1).mean(axis=0)

    if ig_data['vh_only']:
        x_vh   = np.arange(1, ig_data['max_vh'] + 1)
        vh_y   = pos_enc[:ig_data['max_vh']]
        vl_x   = vl_y = None
        cdr3_x = np.arange(ig_data['max_vh'] + 5,
                           ig_data['max_vh'] + 5 + ig_data['max_cdr3'])
    else:
        x_vh   = np.arange(1, ig_data['max_vh'] + 1)
        vh_y   = pos_enc[:ig_data['max_vh']]
        vl_x   = np.arange(ig_data['max_vh'] + 1,
                           ig_data['max_vh'] + ig_data['max_vl'] + 1)
        vl_y   = pos_enc[ig_data['max_vh']:]
        cdr3_x = np.arange(ig_data['max_vh'] + ig_data['max_vl'] + 5,
                           ig_data['max_vh'] + ig_data['max_vl'] + 5 +
                           ig_data['max_cdr3'])

    ax.fill_between(x_vh, vh_y, color=COLOR_VH_FR, alpha=0.85, lw=0)
    if vl_x is not None:
        ax.fill_between(vl_x, vl_y, color=COLOR_VL_FR, alpha=0.85, lw=0)
    ax.fill_between(cdr3_x, pos_cdr3, color=COLOR_CDR3, alpha=0.85, lw=0)

    # Region boundaries
    ymax = float(max(vh_y.max(),
                     pos_cdr3.max(),
                     vl_y.max() if vl_y is not None else 0.0))
    for xr in (x_vh[-1] + 0.5,
               (vl_x[-1] + 0.5 if vl_x is not None else x_vh[-1] + 4),
               cdr3_x[0] - 0.5):
        ax.axvline(xr, color='#ccc', lw=0.3, ls=':')

    # Region label strips above the plot — placed well clear of data peaks
    mid_vh   = (x_vh[0] + x_vh[-1]) / 2
    mid_cdr3 = (cdr3_x[0] + cdr3_x[-1]) / 2
    label_y  = ymax * 1.40
    ax.text(mid_vh,   label_y, 'VH framework',
            ha='center', va='bottom', fontsize=6, color=COLOR_VH_FR,
            fontweight='bold')
    if vl_x is not None:
        mid_vl = (vl_x[0] + vl_x[-1]) / 2
        ax.text(mid_vl, label_y, 'VL framework',
                ha='center', va='bottom', fontsize=6, color=COLOR_VL_FR,
                fontweight='bold')
    ax.text(mid_cdr3, label_y, 'HCDR3',
            ha='center', va='bottom', fontsize=6, color=COLOR_CDR3,
            fontweight='bold')

    ax.set_xlabel('Position', fontsize=6)
    ax.set_ylabel('Mean |IG|', fontsize=6)
    ax.set_title(title, fontsize=8, loc='left', fontweight='bold', pad=4)
    ax.set_ylim(0, ymax * 1.58)
    ax.tick_params(axis='both', labelsize=5)
    ax.grid(axis='y', alpha=0.25, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _panel_ig_cdr3_heatmap(ax, ig_data: Optional[dict], title: str):
    """HCDR3 AA × position signed IG heatmap."""
    if ig_data is None:
        _render_blank(ax,
                      "Transformer (onehot) model not found.\n"
                      "rerun predict_developability.py --train --lm onehot\n"
                      "    --model transformer_onehot", title)
        return

    # Mean signed IG across antibodies — (25, 20)
    mat = ig_data['attr_cdr3'].mean(axis=0).T   # (20, 25) for imshow
    # Show the first N positions actually used by most HCDR3 sequences
    hcdr3_seqs = ig_data['hcdr3_seqs']
    avg_len    = int(np.median([len(s) for s in hcdr3_seqs if s]))
    show_cols  = min(max(avg_len + 4, 16), mat.shape[1])
    mat_show = mat[:, :show_cols]

    vmax = float(np.abs(mat_show).max()) or 1e-8
    im = ax.imshow(mat_show, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto', interpolation='nearest')

    ax.set_yticks(range(20))
    ax.set_yticklabels(list(AMINO_ACIDS), fontsize=4.5)
    ax.set_xticks(range(show_cols))
    ax.set_xticklabels([str(i + 1) for i in range(show_cols)], fontsize=5)
    ax.set_xlabel('HCDR3 position', fontsize=6)
    ax.set_ylabel('Amino acid', fontsize=6)
    ax.set_title(title, fontsize=8, loc='left', fontweight='bold', pad=4)

    cbar = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label('Mean signed IG', fontsize=5)
    cbar.ax.tick_params(labelsize=4)


def _panel_region_convergence(ax,
                               rf_reg:  Optional[dict],
                               xgb_reg: Optional[dict],
                               ig_reg:  Optional[dict],
                               title: str):
    """Grouped bars: per method, % attribution mass in each region."""
    data = {
        'RF':          rf_reg  or {},
        'XGBoost':     xgb_reg or {},
        'Transformer': ig_reg  or {},
    }
    # Include VL only if ANY method gives it mass
    regions = ['HCDR3', 'VH']
    if any((d.get('VL', 0) or 0) > 0.01 for d in data.values()):
        regions.append('VL')

    x = np.arange(len(regions))
    width = 0.27

    method_colors = {'RF': COLOR_RF, 'XGBoost': COLOR_XGB,
                     'Transformer': COLOR_TRANS}
    for i, (method, reg_frac) in enumerate(data.items()):
        vals = [reg_frac.get(r, 0.0) * 100 for r in regions]
        offs = (i - 1) * width
        bars = ax.bar(x + offs, vals, width,
                      color=method_colors[method],
                      edgecolor='white', linewidth=0.4,
                      label=method)
        for b, v in zip(bars, vals):
            if v > 3:   # only annotate bars > 3%
                ax.text(b.get_x() + b.get_width() / 2,
                        v + 1.5, f"{v:.0f}%",
                        ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=6)
    ax.set_ylabel('% of |attribution| mass', fontsize=6)
    ax.set_title(title, fontsize=8, loc='left', fontweight='bold', pad=4)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=5, frameon=False,
              handlelength=1.0, handleheight=0.8, ncol=3,
              columnspacing=0.8)
    ax.tick_params(axis='both', labelsize=5)
    ax.grid(axis='y', alpha=0.25, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Caption below
    ax.text(0.5, -0.22,
            "Proportion of total absolute attribution mass assigned to each region "
            "by each model. All three converge on HCDR3 despite operating in "
            "different feature spaces (biophysical vs. one-hot).",
            transform=ax.transAxes, ha='center', va='top', fontsize=5,
            color='#555', style='italic', wrap=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def _make_fig_scaffold(target, db_stem):
    """Shared scaffold for both figure variants."""
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                     constrained_layout=False)
    gs = gridspec.GridSpec(5, 1, figure=fig,
                           height_ratios=[1.2, 1.2, 0.85, 1.15, 1.0],
                           left=0.17, right=0.96,
                           top=0.955, bottom=0.055,
                           hspace=0.95)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    fig.suptitle(
        f"Interpretability convergence — {target}  ·  {db_stem}",
        fontsize=9, y=0.985)
    return fig, axes


def _save_fig(fig, out_stem: str, log: _Log):
    for ext, dpi in [('tiff', DPI_TIFF), ('pdf', DPI_PDF), ('png', DPI_PNG)]:
        path = f"{out_stem}.{ext}"
        save_kw = dict(dpi=dpi, bbox_inches='tight')
        if ext == 'tiff':
            save_kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **save_kw)
        log(f"[figure] {path}  ({dpi} DPI)")
    plt.close(fig)


def build_figure_bar(rf_shap, xgb_shap, ig_data,
                     rf_reg, xgb_reg, ig_reg,
                     target: str, db_stem: str,
                     rf_lm: str, xgb_lm: str, tr_lm: str,
                     out_stem: str, log: _Log):
    """
    Main figure — panels a/b are SHAP bar charts (mean |SHAP|, region-coloured).
    Same format as Extended Figs 2/3/4. Use this for consistent figure style
    across all analyses.

    Output: {out_stem}.{tiff|pdf|png}
    """
    fig, (ax_a, ax_b, ax_c, ax_d, ax_e) = _make_fig_scaffold(target, db_stem)

    _panel_shap_bar(ax_a, rf_shap,
                    f"a   Random Forest  ·  SHAP top features  ({rf_lm})",
                    top_n=20, kmer_source=rf_lm)
    _panel_shap_bar(ax_b, xgb_shap,
                    f"b   XGBoost  ·  SHAP top features  ({xgb_lm})",
                    top_n=20, kmer_source=xgb_lm)
    _panel_ig_positions(ax_c, ig_data,
                    f"c   Transformer  ·  Integrated Gradients per position  ({tr_lm})")
    _panel_ig_cdr3_heatmap(ax_d, ig_data,
                    f"d   Transformer  ·  HCDR3 amino-acid × position signed IG")
    _panel_region_convergence(ax_e, rf_reg, xgb_reg, ig_reg,
                    f"e   Cross-method convergence on HCDR3")

    _save_fig(fig, out_stem, log)


def build_figure_beeswarm(rf_shap, xgb_shap, ig_data,
                           rf_reg, xgb_reg, ig_reg,
                           target: str, db_stem: str,
                           rf_lm: str, xgb_lm: str, tr_lm: str,
                           out_stem: str, log: _Log):
    """
    Directional figure — panels a/b are SHAP beeswarms.
    Dot colour = feature value (red=high, blue=low).
    Dot x-position = SHAP value (left=FAIL, right=PASS).
    Use this to identify which biophysical properties CAUSE PSR/SEC failure
    and in which direction (e.g. high CDR3 charge → PASS, high R count → FAIL).

    Output: {out_stem}_beeswarm.{tiff|pdf|png}
    """
    fig, (ax_a, ax_b, ax_c, ax_d, ax_e) = _make_fig_scaffold(target, db_stem)

    _panel_shap_beeswarm(ax_a, rf_shap,
                    f"a   Random Forest  ·  SHAP beeswarm  ({rf_lm})  "
                    f"— dot colour = feature value  (red=high, blue=low)",
                    top_n=20, kmer_source=rf_lm)
    _panel_shap_beeswarm(ax_b, xgb_shap,
                    f"b   XGBoost  ·  SHAP beeswarm  ({xgb_lm})",
                    top_n=20, kmer_source=xgb_lm)
    _panel_ig_positions(ax_c, ig_data,
                    f"c   Transformer  ·  Integrated Gradients per position  ({tr_lm})")
    _panel_ig_cdr3_heatmap(ax_d, ig_data,
                    f"d   Transformer  ·  HCDR3 amino-acid × position signed IG")
    _panel_region_convergence(ax_e, rf_reg, xgb_reg, ig_reg,
                    f"e   Cross-method convergence on HCDR3")

    _save_fig(fig, f"{out_stem}_beeswarm", log)


def _ig_residue_rows(ig_data: dict, top_n: int = 25,
                     mode: str = 'per_aa') -> list:
    """
    Extract IG rows for beeswarm plotting.

    mode = 'per_aa'      (DEFAULT)
        20 rows — one per amino acid, IG summed over all HCDR3 positions.
        Every AA is visible (W, E, D, R, K all shown).
        Colour = fixed charge of that AA (+1/-1/0).
        Ordered by mean |IG| across antibodies.
        Best for: "which amino acid identity drives PSR/SEC?"

    mode = 'per_residue'
        top_n (pos, AA) pairs ranked by mean |IG|.
        Position-specific — shows WHERE in HCDR3 + VH/VL the signal is.
        Warning: high-frequency AAs (R) can dominate all top-N slots.
        Best for: "which exact residue at which position matters?"

    Returns list of tuples:
        (mean_abs_ig, label, ig_vec, charge_vec)
    """
    _FULL_NAME = {
        'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe',
        'G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu',
        'M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg',
        'S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr',
    }

    attr_cdr3  = ig_data['attr_cdr3']    # (n, max_cdr3, 20)
    attr_enc   = ig_data['attr_enc']     # (n, max_vh[+max_vl], 20)
    hcdr3_seqs = ig_data['hcdr3_seqs']
    max_vh     = ig_data['max_vh']
    max_vl     = ig_data['max_vl']
    vh_only    = ig_data['vh_only']
    n          = attr_cdr3.shape[0]

    avg_cdr3_len = int(np.median([len(s.replace('-',''))
                                  for s in hcdr3_seqs if s]) or 12)
    use_cdr3     = min(avg_cdr3_len + 4, attr_cdr3.shape[1])

    # ── MODE: per_aa ─────────────────────────────────────────────────────────
    if mode == 'per_aa':
        rows = []
        for aa_i, aa in enumerate(AMINO_ACIDS):
            # Sum signed IG over all active HCDR3 positions for this AA
            ig_vec = attr_cdr3[:, :use_cdr3, aa_i].sum(axis=1)   # (n,)
            ma     = float(np.abs(ig_vec).mean())
            label  = f"{aa}  ({_FULL_NAME.get(aa, aa)})"
            rows.append((ma, label, ig_vec, aa))
        # Always show all 20 AAs, ordered by mean |IG|
        rows.sort(key=lambda x: x[0], reverse=True)
        return rows   # all 20 — ignore top_n for per_aa mode

    # ── MODE: per_aa_by_region ────────────────────────────────────────────────
    # Returns 3 × 20 rows — one group per region (HCDR3, VH, VL).
    # Each group contains all 20 AAs with IG summed within that region only.
    # Returned as flat list with sentinel rows marking region boundaries:
    #   sentinel = (None, '__SEP__:{label}', None, None)
    if mode == 'per_aa_by_region':
        groups = []

        # ── HCDR3 ─────────────────────────────────────────────────────────
        hcdr3_rows = []
        for aa_i, aa in enumerate(AMINO_ACIDS):
            ig_vec = attr_cdr3[:, :use_cdr3, aa_i].sum(axis=1)
            ma     = float(np.abs(ig_vec).mean())
            hcdr3_rows.append((ma, f"{aa}  ({_FULL_NAME.get(aa,aa)})", ig_vec, aa))
        hcdr3_rows.sort(key=lambda x: x[0], reverse=True)
        groups.append(('HCDR3', hcdr3_rows))

        # ── VH framework ──────────────────────────────────────────────────
        if attr_enc.shape[1] >= max_vh:
            vh_rows = []
            for aa_i, aa in enumerate(AMINO_ACIDS):
                ig_vec = attr_enc[:, :max_vh, aa_i].sum(axis=1)
                ma     = float(np.abs(ig_vec).mean())
                vh_rows.append((ma, f"{aa}  ({_FULL_NAME.get(aa,aa)})", ig_vec, aa))
            vh_rows.sort(key=lambda x: x[0], reverse=True)
            groups.append(('VH framework', vh_rows))

        # ── VL framework ──────────────────────────────────────────────────
        if not vh_only and attr_enc.shape[1] > max_vh:
            vl_rows = []
            for aa_i, aa in enumerate(AMINO_ACIDS):
                ig_vec = attr_enc[:, max_vh:max_vh+max_vl, aa_i].sum(axis=1)
                ma     = float(np.abs(ig_vec).mean())
                vl_rows.append((ma, f"{aa}  ({_FULL_NAME.get(aa,aa)})", ig_vec, aa))
            vl_rows.sort(key=lambda x: x[0], reverse=True)
            groups.append(('VL framework', vl_rows))

        # Flatten with sentinel separators
        flat = []
        for region_name, rrows in groups:
            flat.append((None, f'__SEP__:{region_name}', None, None))
            flat.extend(rrows)
        return flat



    rows = []
    for pos in range(use_cdr3):
        for aa_i, aa in enumerate(AMINO_ACIDS):
            ig_vec = attr_cdr3[:, pos, aa_i]
            ma     = float(np.abs(ig_vec).mean())
            rows.append((ma, f"CDR3·{pos+1:02d}·{aa}", ig_vec, aa))

    # ── VH branch — only top positions by mean |IG| summed over AAs ───────
    pos_level_vh = np.abs(attr_enc[:, :max_vh, :]).sum(axis=-1).mean(axis=0)
    top_vh_pos   = np.argsort(pos_level_vh)[::-1][:min(30, max_vh)]

    for pos in top_vh_pos:
        for aa_i, aa in enumerate(AMINO_ACIDS):
            ig_vec = attr_enc[:, pos, aa_i]
            ma     = float(np.abs(ig_vec).mean())
            rows.append((ma, f"VH·{pos+1:03d}·{aa}", ig_vec, aa))

    if not vh_only and attr_enc.shape[1] > max_vh:
        pos_level_vl = (np.abs(attr_enc[:, max_vh:max_vh+max_vl, :])
                        .sum(axis=-1).mean(axis=0))
        top_vl_pos   = np.argsort(pos_level_vl)[::-1][:min(20, max_vl)]
        for pos in top_vl_pos:
            enc_pos = pos + max_vh
            for aa_i, aa in enumerate(AMINO_ACIDS):
                ig_vec = attr_enc[:, enc_pos, aa_i]
                ma     = float(np.abs(ig_vec).mean())
                rows.append((ma, f"VL·{pos+1:03d}·{aa}", ig_vec, aa))

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:top_n]

def _ig_scatter(ax, rows, rng, s=12, alpha=0.65):
    """Render IG beeswarm dots. Colour = AA physicochemical class."""
    for row_idx, (_, label, ig_vec, aa_id) in enumerate(rows):
        jitter = rng.uniform(-0.40, 0.40, size=len(ig_vec))
        col = _aa_color(aa_id) if isinstance(aa_id, str) else _AA_CLASS_COLORS[int(aa_id)%4]
        ax.scatter(ig_vec, row_idx + jitter, c=col,
                   s=s, alpha=alpha, linewidths=0, rasterized=True, zorder=2)

def _ig_ax_style(ax, rows, xlabel='IG value  (← FAIL  |  PASS →)',
                 fsz_y=7, monospace=True):
    """Apply common axis styling to an IG beeswarm axis."""
    ax.axvline(0, color='#777', lw=0.8, ls='-', zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=fsz_y,
                       fontfamily='monospace' if monospace else 'sans-serif')
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=fsz_y - 1)
    ax.grid(axis='x', alpha=0.20, lw=0.3)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fsz_y + 1, labelpad=2)

def _ig_legend(ax, fontsize=6.5, loc='lower right', outside=False):
    """4-class AA physicochemical legend.
    outside=True places the legend below the x-axis — avoids covering data.
    """
    kwargs = dict(handles=_aa_class_legend(), fontsize=fontsize,
                  frameon=True, framealpha=0.92, edgecolor='#ccc',
                  handlelength=1.1, handleheight=0.85,
                  title='AA physicochemical class', title_fontsize=fontsize)
    if outside:
        # Place below the x-axis — never overlaps data
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                  ncol=2, **kwargs)
    else:
        ax.legend(loc=loc, **kwargs)


def _render_3region_beeswarm(ig_data: Optional[dict],
                              title: str, out_stem: str, log: _Log):
    """
    3-region IG beeswarm — HCDR3 / VH / VL stacked sub-panels.
    Each panel shows all 20 AAs, ordered by mean |IG| within that region.
    Colour = AA physicochemical class (4 categories).
    Shared x-axis so magnitude is directly comparable across regions.
    """
    if ig_data is None:
        log("[3region] skipped (no IG data)"); return

    _FULL = {'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe','G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu','M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg','S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr'}

    attr_cdr3  = ig_data['attr_cdr3']
    attr_enc   = ig_data['attr_enc']
    hcdr3_seqs = ig_data['hcdr3_seqs']
    max_vh     = ig_data['max_vh']
    max_vl     = ig_data['max_vl']
    vh_only    = ig_data['vh_only']
    n          = attr_cdr3.shape[0]
    avg_len    = int(np.median([len(s.replace('-','')) for s in hcdr3_seqs if s]) or 12)
    use_cdr3   = min(avg_len + 4, attr_cdr3.shape[1])
    rng        = np.random.default_rng(42)

    def _build_region(mat):
        rows = []
        for aa_i, aa in enumerate(AMINO_ACIDS):
            ig_vec = mat[:, :, aa_i].sum(axis=1) if mat.ndim == 3 else mat[:, aa_i]
            rows.append((float(np.abs(ig_vec).mean()),
                         f"{aa}  {_FULL.get(aa,aa)}", ig_vec, aa))
        return sorted(rows, key=lambda x: x[0], reverse=True)

    region_specs = [('HCDR3', COLOR_CDR3,
                     _build_region(attr_cdr3[:, :use_cdr3, :]))]
    if attr_enc.shape[1] >= max_vh:
        region_specs.append(('VH framework', COLOR_VH_FR,
                              _build_region(attr_enc[:, :max_vh, :])))
    if not vh_only and attr_enc.shape[1] > max_vh:
        region_specs.append(('VL framework', COLOR_VL_FR,
                              _build_region(attr_enc[:, max_vh:max_vh+max_vl, :])))

    # Shared x-range (99th percentile across all regions)
    all_ig = np.concatenate([np.concatenate([r[2] for r in rows])
                              for _, _, rows in region_specs])
    x_abs = float(np.percentile(np.abs(all_ig), 99)) * 1.08
    xlim  = (-x_abs, x_abs)

    n_panels = len(region_specs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, n_panels * 4.5),
                              gridspec_kw={'hspace': 0.60})
    if n_panels == 1: axes = [axes]

    for ax, (region_name, region_col, rows) in zip(axes, region_specs):
        _ig_scatter(ax, rows, rng, s=11)
        _ig_ax_style(ax, rows, fsz_y=7)
        ax.set_xlim(xlim)
        ax.set_title(f"  {region_name}", fontsize=8, loc='left',
                     fontweight='bold', pad=4, color=region_col,
                     backgroundcolor='#f8f8f8')
        _ig_legend(ax, fontsize=6)

    fig.suptitle(title, fontsize=9, y=1.005)
    for ext, dpi in [('tiff', DPI_TIFF), ('pdf', DPI_PDF), ('png', DPI_PNG)]:
        path = f"{out_stem}.{ext}"
        kw = dict(dpi=dpi, bbox_inches='tight')
        if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kw)
        log(f"[3region] {path}  ({dpi} DPI)")
    plt.close(fig)


def _standalone_ig_residue_beeswarm(ig_data: Optional[dict],
                                     title: str, out_stem: str, log: _Log,
                                     top_n: int = 30):
    """
    Produces TWO standalone IG residue beeswarms:

    Figure 1 — per_aa  ({out_stem}_residue_beeswarm_perAA)
        20 rows, one per amino acid (W and E always visible).
        IG summed over all active HCDR3 positions for each AA.
        Answers: "which amino acid identity drives PSR/SEC?"
        Colour = fixed AA charge (red=R/K/H, blue=D/E, grey=neutral)

    Figure 2 — per_residue  ({out_stem}_residue_beeswarm_perPos)
        top_n rows, one per (pos, AA) pair ranked by mean |IG|.
        Position-specific — where exactly in HCDR3 + VH/VL the signal is.
        Answers: "which exact residue at which position matters?"
        Colour = AA charge at that position.
    """
    if ig_data is None:
        log(f"[ig_res_beeswarm] skipped (no IG data)")
        return

    rng      = np.random.default_rng(42)

    def _render_beeswarm(rows, out_path, fig_title, fsz_y=7.5):
        n_feat = len(rows)
        fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 0.44)))
        _ig_scatter(ax, rows, rng, s=13)
        _ig_ax_style(ax, rows, fsz_y=fsz_y)
        ax.set_title(fig_title, fontsize=9, pad=8)
        _ig_legend(ax, fontsize=6.5)
        plt.tight_layout()
        for ext, dpi in [('tiff', DPI_TIFF), ('pdf', DPI_PDF), ('png', DPI_PNG)]:
            path = f"{out_path}.{ext}"
            save_kw = dict(dpi=dpi, bbox_inches='tight')
            if ext == 'tiff':
                save_kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
            fig.savefig(path, **save_kw)
            log(f"[ig_res_beeswarm] {path}  ({dpi} DPI)")
        plt.close(fig)

    # ── Figure 1: per-AA (all 20 AAs, W and E always shown) ─────────────
    rows_aa = _ig_residue_rows(ig_data, top_n=top_n, mode='per_aa')
    _render_beeswarm(
        rows_aa,
        out_path  = f"{out_stem}_residue_beeswarm_perAA",
        fig_title = (f"{title}\n"
                     f"Per amino acid — IG summed over all HCDR3 positions  "
                     f"(all 20 AAs shown, W and E visible)"),
        fsz_y     = 8.5,
    )

    # ── Figure 2: per-residue (pos, AA) pairs ────────────────────────────
    rows_pos = _ig_residue_rows(ig_data, top_n=top_n, mode='per_residue')
    _render_beeswarm(
        rows_pos,
        out_path  = f"{out_stem}_residue_beeswarm_perPos",
        fig_title = (f"{title}\n"
                     f"Per residue — top {top_n} (chain·pos·AA) pairs by mean |IG|"),
        fsz_y     = 6.5,
    )

    # ── Figure 3: 3-region layout (HCDR3 / VH / VL stacked) ─────────────
    # Shows WHERE in the sequence each AA effect comes from
    _render_3region_beeswarm(
        ig_data,
        title    = f"{title}\nPer amino acid by region — HCDR3 / VH / VL",
        out_stem = f"{out_stem}_residue_beeswarm_byRegion",
        log      = log,
    )


def _split_ig_rows_by_region(ig_data: dict, top_n: int):
    """Return (hcdr3_rows, vh_rows) from per_aa_by_region mode. VL excluded."""
    all_rows = _ig_residue_rows(ig_data, top_n=top_n, mode='per_aa_by_region')
    hcdr3_rows, vh_rows = [], []
    current = None
    for r in all_rows:
        if isinstance(r[1], str) and r[1].startswith('__SEP__'):
            current = r[1].split(':')[1]
            continue
        if current == 'HCDR3':
            hcdr3_rows.append(r)
        elif current == 'VH framework':
            vh_rows.append(r)
    return hcdr3_rows, vh_rows


def _render_ig_combined(ax, ig_data, rng,
                        vh_top_n: int = 7,
                        fsz_y: float = 6.5,
                        dot_size: int = 8):
    """
    Draw HCDR3 (all 20 AA) + VH framework (top vh_top_n AA) in a SINGLE axis.

    Layout
    ──────
    • HCDR3 rows at y = 0 … n_cdr3-1  (top of axis)
    • Visual separator : dashed line + region labels at y ≈ n_cdr3 + GAP/2
    • VH rows at y = n_cdr3 + GAP … n_cdr3 + GAP + n_vh-1  (bottom)
    • y-axis inverted so row 0 appears at top
    • Y-tick labels coloured by region (CDR3 = magenta, VH = steel-blue)
    • Shared x-axis scale — both regions use the same range so magnitudes
      are directly comparable without the visual discontinuity of split axes.
    • Physicochemical class legend drawn as a standard ax.legend() inside
      the axis — no FancyBboxPatch / clip_on hacks.
    """
    import matplotlib.transforms as _mt

    GAP = 1.8   # blank rows between HCDR3 and VH sections

    if ig_data is None:
        ax.text(0.5, 0.5, 'Transformer model not found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=7, color='#888', style='italic')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        return

    hcdr3_rows, vh_rows = _split_ig_rows_by_region(ig_data, top_n=20)
    vh_rows = vh_rows[:vh_top_n]
    n_cdr3  = len(hcdr3_rows)
    n_vh    = len(vh_rows)

    # ── Scatter dots ─────────────────────────────────────────────────────
    def _y(region_row_i, is_vh):
        """Convert per-region row index to unified axis y coordinate."""
        return region_row_i if not is_vh else n_cdr3 + GAP + region_row_i

    for row_i, (_, label, ig_vec, aa_id) in enumerate(hcdr3_rows):
        jitter = rng.uniform(-0.40, 0.40, size=len(ig_vec))
        col = _aa_color(aa_id) if isinstance(aa_id, str) else _AA_CLASS_COLORS[int(aa_id) % 4]
        ax.scatter(ig_vec, _y(row_i, False) + jitter,
                   c=col, s=dot_size, alpha=0.65, linewidths=0,
                   rasterized=True, zorder=2)

    for row_i, (_, label, ig_vec, aa_id) in enumerate(vh_rows):
        jitter = rng.uniform(-0.40, 0.40, size=len(ig_vec))
        col = _aa_color(aa_id) if isinstance(aa_id, str) else _AA_CLASS_COLORS[int(aa_id) % 4]
        ax.scatter(ig_vec, _y(row_i, True) + jitter,
                   c=col, s=dot_size, alpha=0.65, linewidths=0,
                   rasterized=True, zorder=2)

    # ── Y-axis ticks ─────────────────────────────────────────────────────
    yticks  = ([_y(i, False) for i in range(n_cdr3)] +
               [_y(i, True)  for i in range(n_vh)])
    ylabels = [r[1] for r in hcdr3_rows] + [r[1] for r in vh_rows]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=fsz_y, fontfamily='monospace')

    # Colour tick labels by region
    tick_colors = [COLOR_CDR3] * n_cdr3 + [COLOR_VH_FR] * n_vh
    for tick, col in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(col)

    # ── Separator ─────────────────────────────────────────────────────────
    sep_y = n_cdr3 + GAP / 2
    ax.axhline(sep_y, color='#ccc', lw=0.8, ls='--', zorder=1)

    # Region labels to the left of the y-axis (in axes-x, data-y coordinates)
    trans = _mt.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-0.22, n_cdr3 / 2 - 0.5,
            'HCDR3', transform=trans,
            fontsize=6.5, fontweight='bold', color=COLOR_CDR3,
            va='center', ha='right', clip_on=False,
            rotation=90)
    ax.text(-0.22, n_cdr3 + GAP + n_vh / 2 - 0.5,
            'VH framework', transform=trans,
            fontsize=6.5, fontweight='bold', color=COLOR_VH_FR,
            va='center', ha='right', clip_on=False,
            rotation=90)

    # ── Axis styling ──────────────────────────────────────────────────────
    ax.axvline(0, color='#888', lw=0.6, ls='-', zorder=1)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=max(4.5, fsz_y - 1.5))
    ax.grid(axis='x', alpha=0.18, lw=0.3)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('IG value\n(← FAIL  |  PASS →)', fontsize=6, labelpad=3)

    # ── Physicochemical class legend — proper ax.legend() ────────────────
    _short_labels = [
        'Cationic (R,K,H)  +charge',
        'Anionic (D,E)  −charge',
        'Hydrophobic/Aromatic (W,F,Y,L,I,V,M)',
        'Small/Polar (A,G,S,T,C,P,Q,N)',
    ]
    handles = [mpatches.Patch(facecolor=c, edgecolor='none', label=l)
               for c, l in zip(_AA_CLASS_COLORS, _short_labels)]
    ax.legend(handles=handles,
              title='AA physicochemical class',
              title_fontsize=5.0,
              fontsize=4.5,
              loc='lower right',
              frameon=True, framealpha=0.92, edgecolor='#ccc',
              handlelength=1.0, handleheight=0.85,
              borderpad=0.6, labelspacing=0.4)


def build_figure_3beeswarms(rf_shap, xgb_shap, ig_data,
                             target: str, db_stem: str,
                             rf_lm: str, xgb_lm: str, tr_lm: str,
                             out_stem: str, log: _Log,
                             top_n: int = 20):
    """
    3-column beeswarm figure — NB Extended Data style.

    Col a  RF SHAP beeswarm      (biophysical features)
    Col b  XGBoost SHAP beeswarm (same row order as RF)
    Col c  Transformer IG — ONE combined axis:
             HCDR3 (all 20 AA, top) ── dashed separator ── VH framework (top 7 AA)
             Shared x-axis scale; y-tick labels coloured by region.
             Physicochemical legend drawn as ax.legend() inside the panel.

    All three panels use the same y-tick font size (YTICK_FSZ).
    No figure super-title (not NB standard).
    Output: {out_stem}_3beeswarms.{tiff|pdf|png}
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    import matplotlib.transforms as _mt
    from matplotlib.gridspec import GridSpec

    YTICK_FSZ = 6.5   # unified y-tick font across ALL panels
    VH_TOP_N  = 7     # AAs shown for VH framework region

    # Figure height driven by the taller panel (RF rows vs HCDR3+VH rows)
    ig_total_rows = 20 + VH_TOP_N + 2   # 20 HCDR3 + 7 VH + ~2 gap
    FIG_H = max(7.0, max(top_n, ig_total_rows) * 0.33 + 2.5)
    FIG_W = 13.0

    gs  = GridSpec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.58)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.09)

    ax_rf  = fig.add_subplot(gs[0, 0])
    ax_xgb = fig.add_subplot(gs[0, 1])
    ax_ig  = fig.add_subplot(gs[0, 2])   # single combined IG axis

    cmap_feat = _cm.get_cmap('RdBu_r')
    rng = np.random.default_rng(42)

    def _pretty(l):
        return l.replace('cdr3_charge_ph7', 'cdr3_charge').replace('_ph7', '')

    # ── Panel A : RF SHAP ─────────────────────────────────────────────────
    if rf_shap is None:
        _render_blank(ax_rf, "RF model not found", "")
        rf_order  = list(range(top_n))
        rf_labels = [f"feature_{i}" for i in range(top_n)]
    else:
        names_rf  = rf_shap['names']
        ma_rf     = rf_shap['mean_abs_shap']
        sv_rf     = rf_shap['shap_matrix']
        Xm_rf     = rf_shap['X_matrix']
        rf_order  = list(np.argsort(ma_rf)[::-1][:top_n])
        rf_labels = [_pretty(names_rf[i]) for i in rf_order]

        for row, feat_idx in enumerate(rf_order):
            sv_col  = sv_rf[:, feat_idx]
            xv_col  = Xm_rf[:, feat_idx]
            lo, hi  = xv_col.min(), xv_col.max()
            norm_xv = (xv_col - lo) / (hi - lo + 1e-10)
            jitter  = rng.uniform(-0.36, 0.36, size=len(sv_col))
            ax_rf.scatter(sv_col, row + jitter,
                          c=cmap_feat(norm_xv), s=7, alpha=0.6,
                          linewidths=0, rasterized=True)

        ax_rf.axvline(0, color='#888', lw=0.6, ls='-')
        ax_rf.set_yticks(range(top_n))
        ax_rf.set_yticklabels(rf_labels, fontsize=YTICK_FSZ)
        ax_rf.invert_yaxis()
        ax_rf.tick_params(axis='x', labelsize=YTICK_FSZ - 1)
        ax_rf.grid(axis='x', alpha=0.18, lw=0.3)
        for s in ('top', 'right'): ax_rf.spines[s].set_visible(False)

    ax_rf.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                     fontsize=YTICK_FSZ, labelpad=3)

    # Feature-value colorbar — upper-right inset, above data
    import matplotlib.colors as _mc2
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
    _axins = _inset(ax_rf, width='36%', height='2.4%',
                    loc='upper right', borderpad=1.0)
    sm_feat = plt.cm.ScalarMappable(cmap=cmap_feat,
                                     norm=_mc2.Normalize(vmin=0, vmax=1))
    sm_feat.set_array([])
    cbar_ab = plt.colorbar(sm_feat, cax=_axins, orientation='horizontal')
    cbar_ab.set_ticks([0, 1])
    cbar_ab.set_ticklabels(['Low', 'High'], fontsize=5)
    cbar_ab.set_label('Feature value', fontsize=5, labelpad=1)
    cbar_ab.ax.xaxis.set_label_position('top')
    cbar_ab.ax.xaxis.tick_top()
    cbar_ab.ax.tick_params(width=0.4, length=1.5, labelsize=5)

    # ── Panel B : XGBoost — same row order as RF ──────────────────────────
    if xgb_shap is None or rf_shap is None:
        _render_blank(ax_xgb, "XGBoost model not found", "")
    else:
        xgb_name_to_idx = {n: i for i, n in enumerate(xgb_shap['names'])}
        sv_xgb = xgb_shap['shap_matrix']
        Xm_xgb = xgb_shap['X_matrix']

        for row, rf_feat_idx in enumerate(rf_order):
            feat_name = rf_shap['names'][rf_feat_idx]
            xgb_idx   = xgb_name_to_idx.get(feat_name)
            if xgb_idx is None:
                continue
            sv_col  = sv_xgb[:, xgb_idx]
            xv_col  = Xm_xgb[:, xgb_idx]
            lo, hi  = xv_col.min(), xv_col.max()
            norm_xv = (xv_col - lo) / (hi - lo + 1e-10)
            jitter  = rng.uniform(-0.36, 0.36, size=len(sv_col))
            ax_xgb.scatter(sv_col, row + jitter,
                           c=cmap_feat(norm_xv), s=7, alpha=0.6,
                           linewidths=0, rasterized=True)

        ax_xgb.axvline(0, color='#888', lw=0.6, ls='-')
        ax_xgb.tick_params(axis='x', labelsize=YTICK_FSZ - 1)
        ax_xgb.grid(axis='x', alpha=0.18, lw=0.3)
        for s in ('top', 'right'): ax_xgb.spines[s].set_visible(False)

    n_rf_rows = len(rf_labels) if rf_shap is not None else top_n
    ax_xgb.set_yticks(range(n_rf_rows))
    ax_xgb.set_yticklabels(
        rf_labels if rf_shap is not None else [f"feature_{i}" for i in range(top_n)],
        fontsize=YTICK_FSZ)
    ax_xgb.invert_yaxis()
    ax_xgb.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                      fontsize=YTICK_FSZ, labelpad=3)

    # ── Panel C : Transformer IG — single combined axis ───────────────────
    _render_ig_combined(ax_ig, ig_data, rng,
                        vh_top_n=VH_TOP_N, fsz_y=YTICK_FSZ, dot_size=8)

    # ── Panel titles via fig.text after canvas.draw() ─────────────────────
    # Filter short name (PSR / SEC) derived from target column name
    _filter = target.upper().split('_')[0]   # 'psr_filter' → 'PSR'

    fig.canvas.draw()
    _panel_defs = [
        (ax_rf,  'a', f'RF-{_filter}',
                      '(biophysical features)'),
        (ax_xgb, 'b', f'XGBoost-{_filter}',
                      '(same row order as RF — compare directly)'),
        (ax_ig,  'c', f'Transformer onehot-{_filter}  ·  IG per amino acid',
                      '(HCDR3 + VH framework, one row per AA)'),
    ]
    # Lift titles ~0.5 inch above axis top (0.5 in / FIG_H in ≈ 0.045 fig-frac)
    TITLE_LIFT = 0.048
    for ax, letter, line1, line2 in _panel_defs:
        pos = ax.get_position()
        x0, y1 = pos.x0, pos.y1
        ty = y1 + TITLE_LIFT
        fig.text(x0, ty, letter,
                 fontsize=10, fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        fig.text(x0 + 0.022, ty, line1,
                 fontsize=7, fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        if line2:
            fig.text(x0 + 0.022, ty - 0.010, line2,
                     fontsize=5.5, va='top', ha='left', color='#555',
                     transform=fig.transFigure)

    _save_fig(fig, f"{out_stem}_3beeswarms", log)


def _standalone_shap_beeswarm(shap_data: Optional[dict],
                               title: str, out_stem: str, log: _Log,
                               top_n: int = 25, kmer_source: str = 'CDR3'):
    """
    Standalone SHAP beeswarm — exact style of existing shap_analysis() output.

    • Figure size : 10 × 8 inch  (generous for standalone use)
    • Y-axis      : top-N features ordered by mean |SHAP|
    • X-axis      : signed SHAP value  (← FAIL  |  PASS →)
    • Dot colour  : feature value  (blue=low, red/pink=high)
                    same RdBu_r scale used by shap.summary_plot()
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc

    if shap_data is None:
        log(f"[beeswarm] {Path(out_stem).name} — skipped (no data)")
        return

    names   = shap_data['names']
    ma_shap = shap_data['mean_abs_shap']
    sv      = shap_data['shap_matrix']    # (n, F)
    Xm      = shap_data['X_matrix']       # (n, F)

    order  = np.argsort(ma_shap)[::-1][:top_n]
    n_feat = len(order)

    def _pretty(l):
        return (l.replace('cdr3_charge_ph7', 'cdr3_charge')
                 .replace('_ph7', '').replace('_', '_'))

    pretty = [_pretty(names[i]) for i in order]
    cmap   = _cm.get_cmap('RdBu_r')   # matches shap.summary_plot default
    rng    = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 0.35)))

    for row, feat_idx in enumerate(order):
        sv_col  = sv[:, feat_idx]
        xv_col  = Xm[:, feat_idx]
        lo, hi  = xv_col.min(), xv_col.max()
        norm_xv = (xv_col - lo) / (hi - lo + 1e-10)
        n       = len(sv_col)
        jitter  = rng.uniform(-0.38, 0.38, size=n)
        ax.scatter(sv_col, row + jitter,
                   c=cmap(norm_xv), s=12, alpha=0.65,
                   linewidths=0, rasterized=True, zorder=2)

    ax.axvline(0, color='#777', lw=0.8, ls='-', zorder=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(pretty, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value  (← toward FAIL  |  toward PASS →)', fontsize=8)
    ax.set_title(title, fontsize=9, pad=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', alpha=0.20, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Colourbar — matches shap.summary_plot style
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=_mc.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.018, aspect=40)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=7)
    cbar.set_label('Feature value', fontsize=7, labelpad=4)
    cbar.ax.tick_params(width=0.5, length=2)

    plt.tight_layout()
    for ext, dpi in [('tiff', DPI_TIFF), ('pdf', DPI_PDF), ('png', DPI_PNG)]:
        path = f"{out_stem}.{ext}"
        save_kw = dict(dpi=dpi, bbox_inches='tight')
        if ext == 'tiff':
            save_kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **save_kw)
        log(f"[beeswarm] {path}  ({dpi} DPI)")
    plt.close(fig)


def _standalone_ig_beeswarm(ig_data: Optional[dict],
                              title: str, out_stem: str, log: _Log,
                              top_n: int = 25):
    """
    Standalone IG beeswarm — Transformer onehot analogue of SHAP beeswarm.

    Each row = one HCDR3 position, ordered by mean |IG| (summed over AA dim).
    X-axis   = signed IG at that position, per antibody.
    Colour   = net charge of the actual amino acid present at that position
               in each antibody's HCDR3 sequence:
                 red  (positive, +1) : R, K, H  — cationic, polyreactivity drivers
                 blue (negative, −1) : D, E      — anionic, protective
                 grey (neutral,   0) : all others

    Reading: if red dots are LEFT (negative IG) → positively charged AAs at
             this position push toward FAIL (polyreactive).
             If blue dots are RIGHT → anionic residues here are protective.
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc

    if ig_data is None:
        log(f"[ig_beeswarm] {Path(out_stem).name} — skipped (no IG data)")
        return

    attr_cdr3  = ig_data['attr_cdr3']    # (n, max_cdr3, 20)
    hcdr3_seqs = ig_data['hcdr3_seqs']
    n_ab, max_cdr3, _ = attr_cdr3.shape

    # Per-antibody, per-position signed IG (sum over 20 AA channels)
    pos_ig = attr_cdr3.sum(axis=-1)     # (n, max_cdr3)

    # Only include positions present in majority of sequences
    avg_len  = int(np.median([len(s.replace('-', '')) for s in hcdr3_seqs if s]))
    use_pos  = min(avg_len + 4, max_cdr3)

    # AA charge lookup: +1 cationic, -1 anionic, 0 neutral
    AA_IDX  = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    # Per-antibody AA charge at each position (from one-hot: find which AA is present)
    # attr_cdr3 shape gives us which AA has the largest gradient — but simpler:
    # read directly from hcdr3_seqs
    def _charge_at(seqs, pos):
        """Return charge (+1/0/-1) for each antibody at CDR3 position pos."""
        out = np.zeros(len(seqs), dtype=np.float32)
        for i, s in enumerate(seqs):
            if s and pos < len(s):
                out[i] = float(_AA_CLASS.get(s[pos], 3))
        return out

    # Build rows: (mean_abs_ig, position_label, ig_vec, charge_vec)
    rows = []
    for pos in range(use_pos):
        ig_vec     = pos_ig[:, pos]
        mean_abs   = float(np.abs(ig_vec).mean())
        charge_vec = _charge_at(hcdr3_seqs, pos)
        rows.append((mean_abs, f"HCDR3 pos {pos+1}", ig_vec, charge_vec))

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:top_n]
    n_feat = len(rows)

    # Colormap: blue (−1, anionic) → grey (0, neutral) → red (+1, cationic)
    # Use TwoSlopeNorm centred at 0
    cmap = _cm.get_cmap('RdBu_r')
    norm = _mc.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    rng  = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 0.38)))

    for row_idx, (mean_abs, label, ig_vec, charge_vec) in enumerate(rows):
        n      = len(ig_vec)
        jitter = rng.uniform(-0.38, 0.38, size=n)
        ax.scatter(ig_vec, row_idx + jitter,
                   c=cmap(norm(charge_vec)), s=12, alpha=0.65,
                   linewidths=0, rasterized=True, zorder=2)

    ax.axvline(0, color='#777', lw=0.8, ls='-', zorder=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([r[1] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('IG value  (← toward FAIL  |  toward PASS →)', fontsize=8)
    ax.set_title(title, fontsize=9, pad=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', alpha=0.20, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Colourbar: red=cationic, blue=anionic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.018, aspect=40)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Anionic\n(D, E)', 'Neutral', 'Cationic\n(R, K, H)'],
                        fontsize=6)
    cbar.set_label('AA charge at position', fontsize=7, labelpad=4)
    cbar.ax.tick_params(width=0.5, length=2)

    plt.tight_layout()
    for ext, dpi in [('tiff', DPI_TIFF), ('pdf', DPI_PDF), ('png', DPI_PNG)]:
        path = f"{out_stem}.{ext}"
        save_kw = dict(dpi=dpi, bbox_inches='tight')
        if ext == 'tiff':
            save_kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **save_kw)
        log(f"[ig_beeswarm] {path}  ({dpi} DPI)")
    plt.close(fig)




def _plot_pass_fail_ig(ig_data: Optional[dict],
                       df: pd.DataFrame,
                       target: str,
                       db_stem: str,
                       out_stem: str,
                       log: _Log,
                       top_n: int = 60):
    """
    Reproduce the existing global_ig_analysis() plot format from transformer_onehot.py,
    but split by PASS (label=1) and FAIL (label=0).

    Format matches your existing IG plots exactly:
      • Horizontal bar chart
      • Blue  = positive mean signed IG  (pushes toward PASS / developable)
      • Red   = negative mean signed IG  (pushes toward FAIL / polyreactive)
      • Y-axis: position labels  H_1 … H_135 | L_1 … L_135 | CDR3_1 … CDR3_25
      • Two panels side by side: PASS group (left) | FAIL group (right)
      • Top-N positions by mean |signed IG| across the whole dataset

    Output: {out_stem}_pass_fail_ig.{tiff|pdf|png}
    """
    if ig_data is None:
        log("[pass_fail_ig] skipped — no IG data"); return

    attr_enc  = ig_data['attr_enc']    # (n, max_vh+max_vl, 20)  or (n, max_vh, 20)
    attr_cdr3 = ig_data['attr_cdr3']   # (n, 25, 20)
    barcodes  = ig_data['barcodes']
    max_vh    = ig_data['max_vh']
    max_vl    = ig_data['max_vl']
    vh_only   = ig_data['vh_only']
    n         = attr_enc.shape[0]

    # ── Position-level signed IG (sum over AA dim) ────────────────────────
    pos_enc  = attr_enc.sum(axis=-1)    # (n, L1)  — signed
    pos_cdr3 = attr_cdr3.sum(axis=-1)  # (n, 25)  — signed
    all_pos  = np.concatenate([pos_enc, pos_cdr3], axis=1)   # (n, L1+25)

    # ── Position labels — matching your existing H_/L_/CDR3_ convention ──
    if vh_only:
        labels = ([f"H_{i+1}" for i in range(max_vh)] +
                  [f"CDR3_{i+1}" for i in range(attr_cdr3.shape[1])])
    else:
        labels = ([f"H_{i+1}" for i in range(max_vh)] +
                  [f"L_{i+1}" for i in range(max_vl)] +
                  [f"CDR3_{i+1}" for i in range(attr_cdr3.shape[1])])

    # ── Get labels for antibodies in ig_data from df ─────────────────────
    bc_to_label = {}
    if target in df.columns:
        for bc, row in df.iterrows():
            bc_to_label[str(bc)] = int(row[target])

    y_arr = np.array([bc_to_label.get(bc, -1) for bc in barcodes])
    pass_idx = np.where(y_arr == 1)[0]
    fail_idx = np.where(y_arr == 0)[0]

    if len(pass_idx) == 0 or len(fail_idx) == 0:
        log(f"[pass_fail_ig] WARNING: PASS={len(pass_idx)} FAIL={len(fail_idx)} "
            f"— labels not found in df. Check BARCODE alignment.")
        # Fall back to whole-dataset mean
        pass_idx = np.arange(n)
        fail_idx = np.arange(n)

    log(f"[pass_fail_ig] PASS n={len(pass_idx):,}  FAIL n={len(fail_idx):,}")

    # ── Mean signed IG per group ──────────────────────────────────────────
    mean_pass = all_pos[pass_idx].mean(axis=0)   # (L1+25,)  signed
    mean_fail = all_pos[fail_idx].mean(axis=0)

    # ── Select top-N by mean |IG| across full dataset ─────────────────────
    mean_all   = all_pos.mean(axis=0)
    top_idx    = np.argsort(np.abs(mean_all))[::-1][:top_n]
    top_labels = [labels[i] for i in top_idx]

    # ── Plot: 2 panels (PASS left, FAIL right) ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.22)),
                              sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.18, right=0.97,
                        top=0.93, bottom=0.06)

    pos_rate = float((y_arr >= 0).mean()) if len(y_arr) > 0 else 0.5
    n_pass   = len(pass_idx)
    n_fail   = len(fail_idx)

    for ax, vals, title_str, group_n in [
        (axes[0], mean_pass[top_idx],
         f"PASS  (n={n_pass:,})", n_pass),
        (axes[1], mean_fail[top_idx],
         f"FAIL  (n={n_fail:,})", n_fail),
    ]:
        colors = ['#1F77B4' if v >= 0 else '#D62728' for v in vals]
        y = np.arange(len(vals))
        ax.barh(y, vals, color=colors, edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='#555', lw=0.7, ls='-')
        ax.set_yticks(y)
        ax.set_yticklabels(top_labels, fontsize=6.5, fontfamily='monospace')
        ax.invert_yaxis()
        ax.set_xlabel('Mean signed IG attribution', fontsize=7)
        ax.set_title(title_str, fontsize=8, fontweight='bold', pad=4)
        ax.tick_params(axis='x', labelsize=6)
        ax.grid(axis='x', alpha=0.20, lw=0.3)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)

    # Shared legend
    from matplotlib.patches import Patch as _P
    axes[0].legend(
        handles=[_P(color='#1F77B4', label='Positive contribution (→ PASS)'),
                 _P(color='#D62728', label='Negative contribution (→ FAIL)')],
        fontsize=6, loc='lower right', frameon=False)

    fig.suptitle(
        f"IG attribution by outcome — {target}  ·  {db_stem}\n"
        f"Mean signed IG across PASS vs FAIL antibodies  "
        f"(top {top_n} positions by mean |IG|)",
        fontsize=8, y=0.98)

    for ext, dpi in [('tiff', DPI_TIFF), ('pdf', DPI_PDF), ('png', DPI_PNG)]:
        path = f"{out_stem}_pass_fail_ig.{ext}"
        kw = dict(dpi=dpi, bbox_inches='tight')
        if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kw)
        log(f"[pass_fail_ig] {path}  ({dpi} DPI)")
    plt.close(fig)

def _run_one_dataset(args, db_path: str, target: str,
                     outdir_base: str = None,
                     _inject_df=None) -> dict:
    """
    Compute SHAP (RF + XGB) and IG (Transformer) for one database/target pair.
    Returns a dict with all computed data + metadata used by both the
    single-dataset run() and the 2-row combined figure.
    """
    db_stem = Path(db_path).stem
    outdir  = Path(outdir_base or f"outputs/interp_{target}_{db_stem}")
    outdir.mkdir(parents=True, exist_ok=True)

    log = _Log(outdir / f"interp_log_{target}_{args.rf_lm}_{args.xgb_lm}_"
                         f"{args.transformer_lm}_{db_stem}.txt")
    result = dict(db_path=db_path, target=target, db_stem=db_stem,
                  outdir=outdir, log=log,
                  rf_shap=None, xgb_shap=None, ig_data=None,
                  rf_reg=None, xgb_reg=None, ig_reg=None, df=None,
                  transformer_model_path=None,
                  xgb_proba={}, rf_proba={},
                  rf_model=None, xgb_model=None)   # kept for per-antibody SHAP
    try:
        log("═" * 62)
        log(f"  MLAbDev INTERPRETABILITY ANALYSIS")
        log(f"  db         : {db_path}")
        log(f"  target     : {target}")
        log(f"  RF  lm     : {args.rf_lm}")
        log(f"  XGB lm     : {args.xgb_lm}")
        log(f"  Trans lm   : {args.transformer_lm}")
        log(f"  model_dir  : {args.model_dir}")
        log(f"  outdir     : {outdir}")
        log("═" * 62)

        if _inject_df is not None:
            df = _inject_df
            log(f"[load] using pre-labelled inject DF: {len(df):,} rows")
        else:
            df = _load_db(db_path, target, log)
        if 'label' not in df.columns:
            df['label'] = df[target].astype(int, errors='ignore')
        result['df'] = df

        # ── RF ──────────────────────────────────────────────────────────
        rf_path = _find_final(args.model_dir, target, args.rf_lm,
                              'rf', db_stem, '.pkl')
        log(f"\n[RF]  checkpoint → {rf_path or 'NOT FOUND'}")
        if rf_path:
            try:
                from models.randomforest import RandomForestModel
                rf_model = RandomForestModel.load(rf_path)
                result['rf_model'] = rf_model
                rf_shap  = _compute_tree_shap(rf_model, df,
                                              args.shap_max_samples, log)
                if rf_shap is not None:
                    result['rf_shap'] = rf_shap
                    result['rf_reg']  = _region_attribution_tree(
                        rf_shap, kmer_source=args.rf_lm)
                    pd.DataFrame({
                        'feature': rf_shap['names'],
                        'mean_abs_shap': rf_shap['mean_abs_shap'],
                        'region': [_feature_region(n, args.rf_lm)
                                   for n in rf_shap['names']],
                    }).sort_values('mean_abs_shap', ascending=False).to_csv(
                        outdir / f"shap_rf_{target}_{args.rf_lm}_{db_stem}.csv",
                        index=False)
                    log(f"[RF]  csv → shap_rf_{target}_{args.rf_lm}_{db_stem}.csv")

                # ── RF predicted probs for ALL antibodies (for triple-model selection)
                try:
                    fb_rf  = rf_model.fb_
                    ne_rf  = np.array(fb_rf.non_embedding_indices)
                    X_rf   = fb_rf.transform(df, None)[:, ne_rf]
                    probs_rf = rf_model.model.predict_proba(X_rf)[:, 1]
                    result['rf_proba'] = {
                        str(bc): float(p)
                        for bc, p in zip(df.index, probs_rf)
                    }
                    log(f"[RF]  rf_proba computed for {len(result['rf_proba']):,} antibodies")
                except Exception as _ep:
                    log(f"[RF]  rf_proba skipped ({_ep})")
            except Exception as e:
                log(f"[RF]  FAILED: {e}"); log(traceback.format_exc())

        # ── XGBoost ─────────────────────────────────────────────────────
        xgb_path = _find_final(args.model_dir, target, args.xgb_lm,
                               'xgboost', db_stem, '.pkl')
        log(f"\n[XGB] checkpoint → {xgb_path or 'NOT FOUND'}")
        if xgb_path:
            try:
                from models.xgboost import XGBoostModel
                xgb_model = XGBoostModel.load(xgb_path)
                result['xgb_model'] = xgb_model
                xgb_shap  = _compute_tree_shap(xgb_model, df,
                                               args.shap_max_samples, log)
                if xgb_shap is not None:
                    result['xgb_shap'] = xgb_shap
                    result['xgb_reg']  = _region_attribution_tree(
                        xgb_shap, kmer_source=args.xgb_lm)
                    pd.DataFrame({
                        'feature': xgb_shap['names'],
                        'mean_abs_shap': xgb_shap['mean_abs_shap'],
                        'region': [_feature_region(n, args.xgb_lm)
                                   for n in xgb_shap['names']],
                    }).sort_values('mean_abs_shap', ascending=False).to_csv(
                        outdir / f"shap_xgb_{target}_{args.xgb_lm}_{db_stem}.csv",
                        index=False)
                    log(f"[XGB] csv → shap_xgb_{target}_{args.xgb_lm}_{db_stem}.csv")

                # ── XGBoost predicted probs for ALL antibodies (for per-ab titles)
                try:
                    fb = xgb_model.fb_
                    ne_idx = np.array(fb.non_embedding_indices)
                    X_all  = fb.transform(df, None)[:, ne_idx]
                    probs_all = xgb_model.model.predict_proba(X_all)[:, 1]
                    result['xgb_proba'] = {
                        str(bc): float(p)
                        for bc, p in zip(df.index, probs_all)
                    }
                    log(f"[XGB] xgb_proba computed for {len(result['xgb_proba']):,} antibodies")
                except Exception as _ep:
                    log(f"[XGB] xgb_proba skipped ({_ep})")
            except Exception as e:
                log(f"[XGB] FAILED: {e}"); log(traceback.format_exc())

        # ── Transformer IG ───────────────────────────────────────────────
        tr_path = _find_final(args.model_dir, target, args.transformer_lm,
                              'transformer_onehot', db_stem, '.pt')
        log(f"\n[IG]  checkpoint → {tr_path or 'NOT FOUND'}")
        if tr_path:
            result['transformer_model_path'] = tr_path   # for per-antibody IG
            try:
                from models.transformer_onehot import TransformerOneHotModel
                tr_model = TransformerOneHotModel.load(tr_path)
                tr_model.set_lm_mode(args.transformer_lm
                                     if args.transformer_lm in ('onehot', 'onehot_vh')
                                     else 'onehot')
                ig_data = _compute_ig(tr_model, df,
                                      args.ig_max_samples, args.ig_steps, log)
                if ig_data is not None:
                    result['ig_data'] = ig_data
                    result['ig_reg']  = _region_attribution_ig(ig_data)
                    np.savez_compressed(
                        outdir / f"ig_{target}_{args.transformer_lm}_{db_stem}.npz",
                        attr_enc   = ig_data['attr_enc'],
                        attr_cdr3  = ig_data['attr_cdr3'],
                        barcodes   = np.array(ig_data['barcodes']),
                        hcdr3_seqs = np.array(ig_data['hcdr3_seqs']),
                        probs      = ig_data['probs'],
                        vh_only    = ig_data['vh_only'],
                        max_vh     = ig_data['max_vh'],
                        max_vl     = ig_data['max_vl'],
                        max_cdr3   = ig_data['max_cdr3'],
                    )
                    log(f"[IG]  npz → ig_{target}_{args.transformer_lm}_{db_stem}.npz")
            except Exception as e:
                log(f"[IG]  FAILED: {e}"); log(traceback.format_exc())

        # ── Region attribution table ─────────────────────────────────────
        region_rows = []
        for method, r in [('RF', result['rf_reg']),
                          ('XGBoost', result['xgb_reg']),
                          ('Transformer', result['ig_reg'])]:
            if r is not None:
                for reg_name, frac in r.items():
                    region_rows.append({'method': method, 'region': reg_name,
                                        'fraction_of_mass': frac})
        if region_rows:
            pd.DataFrame(region_rows).to_csv(
                outdir / f"region_attribution_{target}_{db_stem}.csv", index=False)
            log("\n  Method       HCDR3    VH     VL")
            log("  ───────────  ─────  ─────  ─────")
            for method, r in [('RF', result['rf_reg']),
                              ('XGBoost', result['xgb_reg']),
                              ('Transformer', result['ig_reg'])]:
                if r is None:
                    log(f"  {method:11s}    —      —      — (not loaded)")
                else:
                    log(f"  {method:11s}  {r.get('HCDR3',0)*100:5.1f}%  "
                        f"{r.get('VH',0)*100:5.1f}%  "
                        f"{r.get('VL',0)*100:5.1f}%")

    except Exception as e:
        log(f"[ERROR] {e}"); log(traceback.format_exc())

    return result


def _load_predict_db(predict_path: str, pseudo_target: str, log: _Log) -> pd.DataFrame:
    """
    Load a new unseen antibody file (xlsx or csv).
    Expected columns: BARCODE (or ID), HSEQ, LSEQ, CDR3.
    A pseudo-label column (pseudo_target) is added after prediction — we set it
    to 0 initially so _run_one_dataset can index without error; it is replaced
    by model consensus predictions before any figure is drawn.
    """
    ext = Path(predict_path).suffix.lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(predict_path)
    elif ext == '.csv':
        df = pd.read_csv(predict_path)
    else:
        raise ValueError(f"Unsupported prediction file format: {ext}")

    log(f"[predict] loaded {Path(predict_path).name}: {len(df):,} rows × {len(df.columns)} cols")

    # Normalise BARCODE column (accept ID, barcode, Barcode, etc.)
    col_map = {c.lower(): c for c in df.columns}
    for alias in ('barcode', 'id', 'sample_id', 'sample'):
        if alias in col_map:
            df = df.rename(columns={col_map[alias]: 'BARCODE'})
            break
    if 'BARCODE' not in df.columns:
        df.insert(0, 'BARCODE', [f"AB{i+1:06d}" for i in range(len(df))])
        log("[predict] No BARCODE/ID column found — auto-generated sequential IDs")

    df['BARCODE'] = df['BARCODE'].astype(str).str.strip()

    for c in ['HSEQ', 'LSEQ', 'CDR3']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
        else:
            if c != 'LSEQ':
                log(f"[predict] WARNING: column '{c}' not found in prediction file")

    # Add placeholder pseudo-label — will be overwritten by model predictions
    df[pseudo_target] = 0
    df = df.set_index('BARCODE')
    log(f"[predict] {len(df):,} antibodies ready for prediction")
    return df


def _run_predict_dataset(args, predict_path: str, target: str,
                          outdir_base: str, train_r: dict) -> dict:
    """
    Run all three models on an unseen antibody set.

    1. Load prediction file via _load_predict_db
    2. Predict with RF, XGBoost, Transformer — majority-vote pseudo-labels
    3. Store pseudo-labels in df[target]
    4. Re-use _run_one_dataset to compute SHAP/IG and generate all figures

    train_r : result dict from training set run (same filter) — needed for
              transformer_model_path and model objects.
    """
    outdir = Path(outdir_base)
    outdir.mkdir(parents=True, exist_ok=True)
    log = _Log(outdir / f"predict_log_{target}.txt")

    log("═" * 62)
    log(f"  PREDICTION MODE")
    log(f"  predict file : {predict_path}")
    log(f"  target       : {target}")
    log(f"  outdir       : {outdir}")
    log("═" * 62)

    df = _load_predict_db(predict_path, target, log)

    # ── Predict with each available model ─────────────────────────────────
    prob_rf  = {}; prob_xgb = {}; prob_tr = {}

    rf_model  = train_r.get('rf_model')
    xgb_model = train_r.get('xgb_model')

    if rf_model is not None:
        try:
            fb     = rf_model.fb_
            ne_idx = np.asarray(fb.non_embedding_indices, dtype=int)
            X      = fb.transform(df, None)[:, ne_idx]
            probs  = rf_model.model.predict_proba(X)[:, 1]
            prob_rf = {str(bc): float(p) for bc, p in zip(df.index, probs)}
            log(f"[predict] RF: {len(prob_rf):,} predictions")
        except Exception as e:
            log(f"[predict] RF prediction failed: {e}")

    if xgb_model is not None:
        try:
            fb     = xgb_model.fb_
            ne_idx = np.asarray(fb.non_embedding_indices, dtype=int)
            X      = fb.transform(df, None)[:, ne_idx]
            probs  = xgb_model.model.predict_proba(X)[:, 1]
            prob_xgb = {str(bc): float(p) for bc, p in zip(df.index, probs)}
            log(f"[predict] XGBoost: {len(prob_xgb):,} predictions")
        except Exception as e:
            log(f"[predict] XGBoost prediction failed: {e}")

    tr_path = train_r.get('transformer_model_path')
    if tr_path:
        try:
            from models.transformer_onehot import TransformerOneHotModel
            m = TransformerOneHotModel.load(tr_path)
            ig_out = _compute_ig(m, df, args.ig_max_samples, args.ig_steps, log)
            if ig_out is not None:
                prob_tr = {bc: float(p)
                           for bc, p in zip(ig_out['barcodes'], ig_out['probs'])}
            log(f"[predict] Transformer: {len(prob_tr):,} predictions")
        except Exception as e:
            log(f"[predict] Transformer prediction failed: {e}")

    # ── Majority-vote pseudo-labels ────────────────────────────────────────
    THRESH = 0.5
    n_pass = n_fail = n_split = 0
    for bc in df.index:
        bc_s = str(bc)
        votes = []
        if bc_s in prob_rf:  votes.append(1 if prob_rf[bc_s]  >= THRESH else 0)
        if bc_s in prob_xgb: votes.append(1 if prob_xgb[bc_s] >= THRESH else 0)
        if bc_s in prob_tr:  votes.append(1 if prob_tr[bc_s]  >= THRESH else 0)
        pseudo = int(round(sum(votes) / len(votes))) if votes else 0
        df.at[bc, target] = pseudo
        if pseudo == 1: n_pass += 1
        elif pseudo == 0: n_fail += 1
        else: n_split += 1

    log(f"[predict] Pseudo-labels: {n_pass} PASS  {n_fail} FAIL  {n_split} split")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'barcode':          df.index,
        f'pseudo_{target}': df[target].values,
        'prob_rf':    [prob_rf.get(str(bc), float('nan')) for bc in df.index],
        'prob_xgb':   [prob_xgb.get(str(bc), float('nan')) for bc in df.index],
        'prob_transformer': [prob_tr.get(str(bc), float('nan')) for bc in df.index],
    })
    csv_path = outdir / f"predictions_{target}.csv"
    pred_df.to_csv(csv_path, index=False)
    log(f"[predict] predictions → {csv_path.name}")

    # ── Run full interpretability pipeline on prediction set ───────────────
    # Temporarily inject df + model objects + paths into a fake result dict
    # and call _run_one_dataset with the prediction file (which now has pseudo-labels)
    result = _run_one_dataset(args, predict_path, target, str(outdir),
                               _inject_df=df)
    log.close()
    return result


def run(args) -> int:
    # ── Resolve root output directory once ───────────────────────────────
    root_dir = Path(args.outdir) if args.outdir else Path('outputs')

    # ── Unified sample limit ──────────────────────────────────────────────
    _max = getattr(args, 'max_samples', 3000) or 3000
    _shap_legacy = getattr(args, 'shap_max_samples', None)
    _ig_legacy   = getattr(args, 'ig_max_samples',   None)
    args.shap_max_samples = _shap_legacy if _shap_legacy is not None else _max
    args.ig_max_samples   = _ig_legacy   if _ig_legacy   is not None else _max

    # ── PREDICT MODE ──────────────────────────────────────────────────────
    # If --predict is given, first run the normal training pipeline to load
    # models, then run prediction on the unseen set using those models.
    predict_path  = getattr(args, 'predict',  None)
    predict_path2 = getattr(args, 'predict2', None)

    if predict_path:
        # Step 1 — run training pipeline (needed to load models into r1/r2)
        outdir1_tr = root_dir / f"interp_{args.target}_{Path(args.db).stem}"
        r1_train   = _run_one_dataset(args, args.db, args.target, str(outdir1_tr))
        log1 = r1_train['log']

        r2_train = None
        if args.db2 and args.target2:
            outdir2_tr = root_dir / f"interp_{args.target2}_{Path(args.db2).stem}"
            r2_train   = _run_one_dataset(args, args.db2, args.target2, str(outdir2_tr))

        # Step 2 — predict on unseen set
        pred_stem1 = Path(predict_path).stem
        pred_stem2 = Path(predict_path2 or predict_path).stem
        outdir1_pr = root_dir / f"predict_{args.target}_{pred_stem1}"
        log1(f"\n[predict] Running prediction pipeline on {predict_path} ...")
        r1 = _run_predict_dataset(args, predict_path, args.target,
                                   str(outdir1_pr), r1_train)

        r2 = None
        if args.db2 and args.target2:
            p2_path    = predict_path2 or predict_path
            outdir2_pr = root_dir / f"predict_{args.target2}_{pred_stem2}"
            log1(f"\n[predict] Running prediction pipeline on {p2_path} ...")
            r2 = _run_predict_dataset(args, p2_path, args.target2,
                                       str(outdir2_pr), r2_train)

        # Step 3 — combined figures (same as training mode)
        if r2 is not None:
            combined_dir = root_dir / f"predict_{r1['target']}_{r2['target']}_combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            out_stem = str(combined_dir / f"predict_{r1['target']}_{r2['target']}")
            log1(f"\n[predict] Generating combined figures → {out_stem}")
            build_figure_2row_6panels(r1, r2, args, out_stem, log1)
            build_figure_xgb_dual_filter(
                xgb_shap_1=r1['xgb_shap'], xgb_shap_2=r2['xgb_shap'],
                label_1=r1['target'].upper().replace('_','-'),
                label_2=r2['target'].upper().replace('_','-'),
                out_stem=out_stem, log=log1, top_n=20)
            _n_pairs = getattr(args, 'n_pairs', 20)
            fail_list, pass_list = _find_all_example_antibodies(r1, r2, log1,
                                                                  n_each=_n_pairs)
            build_figure_per_antibody_ig(
                r1=r1, r2=r2, out_stem=out_stem, log=log1,
                n_vhvl_top=6, ig_steps=getattr(args, 'ig_steps', 100),
                n_pairs=_n_pairs, _precomputed_lists=(fail_list, pass_list))
            build_figure_shap_per_antibody(
                r1=r1, r2=r2, out_stem=out_stem, log=log1,
                fail_list=fail_list, pass_list=pass_list,
                model_key='xgb_model', n_pairs=_n_pairs, top_n=25)
            build_figure_shap_per_antibody(
                r1=r1, r2=r2, out_stem=out_stem, log=log1,
                fail_list=fail_list, pass_list=pass_list,
                model_key='rf_model', n_pairs=_n_pairs, top_n=25)

        log1(f"\n[predict] all outputs in {root_dir}/")
        log1.close()
        return 0

    # ── Run dataset 1 (always) ────────────────────────────────────────────
    outdir1 = root_dir / f"interp_{args.target}_{Path(args.db).stem}"
    r1 = _run_one_dataset(args, args.db, args.target, str(outdir1))
    log1 = r1['log']

    # ── Run dataset 2 (when --db2 / --target2 provided) ─────────────────
    r2 = None
    if args.db2 and args.target2:
        outdir2 = root_dir / f"interp_{args.target2}_{Path(args.db2).stem}"
        r2 = _run_one_dataset(args, args.db2, args.target2, str(outdir2))

    # ── Per-dataset standalone figures ───────────────────────────────────
    for r in ([r1] if r2 is None else [r1, r2]):
        _render_dataset_figures(args, r)

    # ── 2-row combined figure (PSR + SEC) ────────────────────────────────
    if r2 is not None:
        combined_dir = root_dir / f"interp_{r1['target']}_{r2['target']}_combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        out_stem = str(combined_dir /
                       f"combined_{r1['target']}_{r2['target']}_"
                       f"{args.rf_lm}_{args.transformer_lm}")
        log1(f"\n[combined] Rendering 2-row PSR+SEC figure → {out_stem}")
        build_figure_2row_6panels(r1, r2, args, out_stem, log1)

        # ── XGBoost dual-filter beeswarm (PSR vs SEC, same row order) ────
        label_1 = r1['target'].upper().replace('_', '-')
        label_2 = r2['target'].upper().replace('_', '-')
        log1(f"\n[combined] Rendering XGBoost dual-filter figure "
             f"({label_1} vs {label_2}) → {out_stem}")
        build_figure_xgb_dual_filter(
            xgb_shap_1 = r1['xgb_shap'],
            xgb_shap_2 = r2['xgb_shap'],
            label_1    = label_1,
            label_2    = label_2,
            out_stem   = out_stem,
            log        = log1,
            top_n      = 20,
        )

        # ── Per-antibody IG waterfall (2 abs × 2 filters = 4 panels) ─────
        _n_pairs = getattr(args, 'n_pairs', 20)

        log1(f"\n[per_ab] Rendering per-antibody IG waterfall figure ...")
        fail_list, pass_list = _find_all_example_antibodies(r1, r2, log1, n_each=_n_pairs)
        build_figure_per_antibody_ig(
            r1=r1, r2=r2, out_stem=out_stem, log=log1,
            n_vhvl_top=6, ig_steps=getattr(args, 'ig_steps', 100), n_pairs=_n_pairs,
            _precomputed_lists=(fail_list, pass_list))

        log1(f"\n[per_ab] Rendering per-antibody XGBoost-biophysical SHAP waterfall ...")
        build_figure_shap_per_antibody(
            r1=r1, r2=r2, out_stem=out_stem, log=log1,
            fail_list=fail_list, pass_list=pass_list,
            model_key='xgb_model', n_pairs=_n_pairs, top_n=25,
            ig_steps=getattr(args, 'ig_steps', 100))

        log1(f"\n[per_ab] Rendering per-antibody RF-biophysical SHAP waterfall ...")
        build_figure_shap_per_antibody(
            r1=r1, r2=r2, out_stem=out_stem, log=log1,
            fail_list=fail_list, pass_list=pass_list,
            model_key='rf_model', n_pairs=_n_pairs, top_n=25,
            ig_steps=getattr(args, 'ig_steps', 100))

    log1(f"\n[done] all outputs in {root_dir}/")
    log1.close()
    if r2 is not None:
        r2['log'](f"\n[done] all outputs in {root_dir}/")
        r2['log'].close()
    return 0


def _render_dataset_figures(args, r: dict):
    """Render all per-dataset standalone figures for one result dict."""
    rf_shap = r['rf_shap']; xgb_shap = r['xgb_shap']; ig_data = r['ig_data']
    rf_reg  = r['rf_reg'];  xgb_reg  = r['xgb_reg'];  ig_reg  = r['ig_reg']
    target  = r['target'];  db_stem  = r['db_stem']
    outdir  = r['outdir'];  df       = r['df'];        log     = r['log']
    fig_stem = str(outdir / f"interp_{target}_{args.rf_lm}_{args.xgb_lm}_"
                             f"{args.transformer_lm}_{db_stem}")

    log(f"\n[figure] Rendering bar-chart figure ...")
    build_figure_bar(rf_shap, xgb_shap, ig_data, rf_reg, xgb_reg, ig_reg,
                     target, db_stem, args.rf_lm, args.xgb_lm,
                     args.transformer_lm, fig_stem, log)

    log(f"\n[figure] Rendering beeswarm figure (directional) ...")
    build_figure_beeswarm(rf_shap, xgb_shap, ig_data, rf_reg, xgb_reg, ig_reg,
                          target, db_stem, args.rf_lm, args.xgb_lm,
                          args.transformer_lm, fig_stem, log)

    log(f"\n[figure] Rendering 3-column beeswarm comparison figure ...")
    build_figure_3beeswarms(rf_shap, xgb_shap, ig_data, target, db_stem,
                             args.rf_lm, args.xgb_lm, args.transformer_lm,
                             fig_stem, log, top_n=20)

    log(f"\n[pass_fail_ig] Rendering PASS vs FAIL signed IG bar plots ...")
    _plot_pass_fail_ig(ig_data=ig_data, df=df, target=target, db_stem=db_stem,
                       out_stem=str(outdir / f"ig_{target}_{args.transformer_lm}_{db_stem}"),
                       log=log, top_n=60)

    log(f"\n[beeswarm] Rendering 3 standalone beeswarm figures ...")
    _standalone_shap_beeswarm(
        rf_shap,
        title   = f"SHAP beeswarm — {target}  ·  RF ({args.rf_lm})  ·  {db_stem}",
        out_stem= str(outdir / f"beeswarm_{target}_{args.rf_lm}_rf_{db_stem}"),
        log=log, top_n=25, kmer_source=args.rf_lm)
    _standalone_shap_beeswarm(
        xgb_shap,
        title   = f"SHAP beeswarm — {target}  ·  XGBoost ({args.xgb_lm})  ·  {db_stem}",
        out_stem= str(outdir / f"beeswarm_{target}_{args.xgb_lm}_xgboost_{db_stem}"),
        log=log, top_n=25, kmer_source=args.xgb_lm)
    _standalone_ig_beeswarm(
        ig_data,
        title   = f"IG beeswarm — {target}  ·  Transformer onehot ({args.transformer_lm})  ·  {db_stem}",
        out_stem= str(outdir / f"beeswarm_{target}_{args.transformer_lm}_transformer_onehot_{db_stem}"),
        log=log, top_n=25)
    _standalone_ig_residue_beeswarm(
        ig_data,
        title   = f"IG residue — {target}  ·  Transformer onehot  ·  {db_stem}",
        out_stem= str(outdir / f"beeswarm_{target}_{args.transformer_lm}_transformer_onehot_{db_stem}"),
        log=log, top_n=30)


def build_figure_xgb_dual_filter(
        xgb_shap_1: Optional[dict],
        xgb_shap_2: Optional[dict],
        label_1: str,
        label_2: str,
        out_stem: str,
        log: _Log,
        top_n: int = 20,
):
    """
    Nature Biotech – style 2-panel XGBoost SHAP beeswarm figure.

    Panels
    ──────
      a  XGBoost SHAP beeswarm — first filter  (e.g. PSR-filter)
      b  XGBoost SHAP beeswarm — second filter (e.g. SEC-filter)

    Row ordering
    ────────────
    The displayed feature set is the union of the top-N features from each
    filter, deduplicated and ranked by the *combined* mean |SHAP| (sum of
    both filters' mean |SHAP| per feature, so features important in either
    assay float to the top).  Both panels share this identical row order,
    enabling direct row-by-row comparison of magnitude and direction.

    Visual style
    ────────────
    • Full NB width: 7.1 in × adaptive height
    • 300 DPI TIFF (LZW) + 300 DPI PDF (vector) + 150 DPI PNG
    • DejaVu Sans, axis linewidth 0.5 pt — matches existing NB rcParams
    • Dot colour = feature value (RdBu_r: red = high, blue = low)
    • Dot x-position = signed SHAP value  (left = FAIL, right = PASS)
    • Feature-value colourbar: inset lower-right of panel a
    • Panel letters a / b via fig.text — zero-overlap guarantee
    • No figure super-title (not NB standard for Extended Data figures)

    Output
    ──────
    {out_stem}_xgb_dual_filter.{tiff|pdf|png}
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    from matplotlib.gridspec import GridSpec

    # ── Guard: need at least one panel ───────────────────────────────────
    if xgb_shap_1 is None and xgb_shap_2 is None:
        log("[xgb_dual] both SHAP dicts are None — skipping figure")
        return

    rng = np.random.default_rng(42)
    cmap = _cm.get_cmap('RdBu_r')

    # ── Build shared feature row order ───────────────────────────────────
    # Collect feature names appearing in either panel
    all_names: dict[str, float] = {}   # name → combined mean |SHAP|
    for sd in (xgb_shap_1, xgb_shap_2):
        if sd is None:
            continue
        for name, val in zip(sd['names'], sd['mean_abs_shap']):
            all_names[name] = all_names.get(name, 0.0) + float(val)

    # Keep top_n from each panel's own ranking, then union + sort by combined score
    def _top_names(sd, n):
        if sd is None:
            return set()
        idx = np.argsort(sd['mean_abs_shap'])[::-1][:n]
        return {sd['names'][i] for i in idx}

    union_names = _top_names(xgb_shap_1, top_n) | _top_names(xgb_shap_2, top_n)
    # Sort union by combined mean |SHAP|, highest first
    ordered_names = sorted(union_names,
                           key=lambda n: all_names.get(n, 0.0),
                           reverse=True)

    def _pretty(l):
        return (l.replace('cdr3_charge_ph7', 'cdr3_charge')
                  .replace('_ph7', ''))

    pretty_labels = [_pretty(n) for n in ordered_names]
    n_rows = len(ordered_names)

    # ── Figure geometry ───────────────────────────────────────────────────
    FIG_W  = 7.1    # 180 mm — full NB width
    ROW_H  = 0.32   # inches per feature row
    FIG_H  = max(5.0, n_rows * ROW_H + 1.8)

    gs  = GridSpec(1, 2, wspace=0.52)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.09)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ── Helper: render one beeswarm panel ────────────────────────────────
    def _render_panel(ax, shap_data, panel_label, title_str):
        if shap_data is None:
            _render_blank(ax, f"XGBoost model not found\n({panel_label})", "")
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(pretty_labels, fontsize=5.5)
            ax.invert_yaxis()
            return

        name_to_idx = {n: i for i, n in enumerate(shap_data['names'])}
        sv  = shap_data['shap_matrix']    # (n_samples, n_features)
        Xm  = shap_data['X_matrix']

        for row_i, feat_name in enumerate(ordered_names):
            col_idx = name_to_idx.get(feat_name)
            if col_idx is None:
                continue   # feature absent in this model — leave row blank
            sv_col  = sv[:, col_idx]
            xv_col  = Xm[:, col_idx]
            lo, hi  = xv_col.min(), xv_col.max()
            norm_xv = (xv_col - lo) / (hi - lo + 1e-10)
            jitter  = rng.uniform(-0.38, 0.38, size=len(sv_col))
            ax.scatter(sv_col, row_i + jitter,
                       c=cmap(norm_xv), s=7, alpha=0.60,
                       linewidths=0, rasterized=True, zorder=2)

        ax.axvline(0, color='#888', lw=0.6, ls='-', zorder=1)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(pretty_labels, fontsize=5.5)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=5)
        ax.grid(axis='x', alpha=0.18, lw=0.3)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                      fontsize=6, labelpad=3)

    _render_panel(ax_a, xgb_shap_1, label_1, label_1)
    _render_panel(ax_b, xgb_shap_2, label_2, label_2)

    # Panel b: hide y-tick labels (shared axis — labels already on left)
    ax_b.set_yticklabels(['' for _ in pretty_labels])

    # ── Feature-value colourbar — inset upper-right of panel a ───────────
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
    _axins = _inset(ax_a, width='36%', height='2.6%',
                    loc='upper right', borderpad=1.2)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=_mc.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=_axins, orientation='horizontal')
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'], fontsize=4.5)
    cb.set_label('Feature value', fontsize=4.5, labelpad=1)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.tick_top()
    cb.ax.tick_params(width=0.4, length=1.5, labelsize=4.5)

    # ── Panel titles via fig.text — aligned baseline ──────────────────────
    fig.canvas.draw()

    _panels = [
        (ax_a, 'a', f'XGBoost  ·  SHAP beeswarm',
                    f'({label_1}  |  biophysical features)'),
        (ax_b, 'b', f'XGBoost  ·  SHAP beeswarm',
                    f'({label_2}  |  biophysical features)'),
    ]
    for ax, letter, line1, line2 in _panels:
        pos = ax.get_position()
        x0  = pos.x0
        y1  = pos.y1
        fig.text(x0, y1 + 0.010, letter,
                 fontsize=10, fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        fig.text(x0 + 0.025, y1 + 0.011, line1,
                 fontsize=7, fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        if line2:
            fig.text(x0 + 0.025, y1 + 0.002, line2,
                     fontsize=6, va='top', ha='left', color='#555',
                     transform=fig.transFigure)

    # ── Save ──────────────────────────────────────────────────────────────
    _save_fig(fig, f"{out_stem}_xgb_dual_filter", log)
    log(f"[xgb_dual] → {out_stem}_xgb_dual_filter  "
        f"(n_features={n_rows}, n1={len(xgb_shap_1['shap_matrix']) if xgb_shap_1 else 0}"
        f", n2={len(xgb_shap_2['shap_matrix']) if xgb_shap_2 else 0})")


def _find_all_example_antibodies(r1, r2, log, n_each=20):
    """
    Find up to n_each antibodies correctly predicted by ALL THREE models
    (RF, XGBoost, Transformer) in BOTH PSR and SEC filters.

    Correct = predicted class matches true label:
      FAIL: true=0, RF pred=0, XGB pred=0, Transformer prob < 0.5
      PASS: true=1, RF pred=1, XGB pred=1, Transformer prob >= 0.5

    Returns (fail_list, pass_list) where each element =
      (barcode, tr_psr_prob, tr_sec_prob, xgb_psr_prob, xgb_sec_prob, rf_psr_pred, rf_sec_pred)
    """
    for tag, r in [('PSR', r1), ('SEC', r2)]:
        if r['ig_data'] is None or r['df'] is None:
            log(f"[per_ab] {tag} ig_data or df missing — skipping per-antibody figure")
            return [], []

    ig1, df1, t1 = r1['ig_data'], r1['df'], r1['target']
    ig2, df2, t2 = r2['ig_data'], r2['df'], r2['target']

    # Transformer predicted probs (from IG computation)
    tr_prob1 = {bc: float(p) for bc, p in zip(ig1['barcodes'], ig1['probs'])}
    tr_prob2 = {bc: float(p) for bc, p in zip(ig2['barcodes'], ig2['probs'])}

    # XGBoost predicted probs (pre-computed in _run_one_dataset)
    xgb_prob1 = r1.get('xgb_proba', {})
    xgb_prob2 = r2.get('xgb_proba', {})

    # RF predicted probs (pre-computed in _run_one_dataset)
    rf_prob1 = r1.get('rf_proba', {})
    rf_prob2 = r2.get('rf_proba', {})

    def _true(df, col, bc):
        try: return int(df.loc[bc, col])
        except Exception: return None

    common = set(tr_prob1.keys()) & set(tr_prob2.keys())
    log(f"[per_ab] {len(common):,} barcodes in both Transformer ig_data sets")

    THRESH = 0.5
    fail_list, pass_list = [], []

    for bc in common:
        y1 = _true(df1, t1, bc); y2 = _true(df2, t2, bc)
        if y1 is None or y2 is None: continue

        tp1 = tr_prob1[bc];  tp2 = tr_prob2[bc]
        xp1 = xgb_prob1.get(bc); xp2 = xgb_prob2.get(bc)
        rp1 = rf_prob1.get(bc);  rp2 = rf_prob2.get(bc)

        # Transformer predictions
        tr_pred1 = 1 if tp1 >= THRESH else 0
        tr_pred2 = 1 if tp2 >= THRESH else 0

        # XGBoost predictions (skip check if XGB not available)
        xgb_pred1 = (1 if xp1 >= THRESH else 0) if xp1 is not None else y1
        xgb_pred2 = (1 if xp2 >= THRESH else 0) if xp2 is not None else y2

        # RF predictions (skip check if RF not available)
        rf_pred1 = (1 if rp1 >= THRESH else 0) if rp1 is not None else y1
        rf_pred2 = (1 if rp2 >= THRESH else 0) if rp2 is not None else y2

        # All models must agree with true label
        if (y1 == 0 and y2 == 0
                and tr_pred1 == 0 and tr_pred2 == 0
                and xgb_pred1 == 0 and xgb_pred2 == 0
                and rf_pred1 == 0 and rf_pred2 == 0):
            fail_list.append((bc, tp1, tp2, xp1, xp2))
        elif (y1 == 1 and y2 == 1
                and tr_pred1 == 1 and tr_pred2 == 1
                and xgb_pred1 == 1 and xgb_pred2 == 1
                and rf_pred1 == 1 and rf_pred2 == 1):
            pass_list.append((bc, tp1, tp2, xp1, xp2))

    fail_list.sort(key=lambda x: x[1])       # most confident FAIL first (lowest PSR prob)
    pass_list.sort(key=lambda x: -x[1])      # most confident PASS first (highest PSR prob)
    log(f"[per_ab] All-3-model agreement: {len(fail_list)} FAIL  {len(pass_list)} PASS "
        f"-> generating up to {min(len(fail_list), len(pass_list), n_each)} pairs")
    return fail_list[:n_each], pass_list[:n_each]


def _waterfall_single_ab(ax, bc, r, n_vhvl_top=6,
                          ig_steps=100, bar_height=0.50,
                          xgb_prob=None):
    """
    Horizontal waterfall for ONE antibody x ONE filter.

    Row order (top -> bottom, y inverted):
      HCDR3  — every CDR3 position with actual AA
      VHVL   — top-n_vhvl_top positions from VH+VL by |IG|

    No dashed separator.  No rotated region labels (y-axis labels carry that info).

    Returns (true_label, tr_prob, xgb_prob, filter_name, outcome_str, title_col, cdr3_seq)
    """
    import matplotlib.transforms as _mt

    ig_data = r['ig_data']
    df      = r['df']
    target  = r['target']
    tr_path = r.get('transformer_model_path')

    try:
        true_label = int(df.loc[bc, target])
    except Exception:
        true_label = None

    tr_prob = None
    if ig_data is not None and bc in ig_data['barcodes']:
        idx     = ig_data['barcodes'].index(bc)
        tr_prob = float(ig_data['probs'][idx])

    try:
        vh   = str(df.loc[bc, 'HSEQ']).upper().replace('-', '')
        vl   = str(df.loc[bc, 'LSEQ']).upper().replace('-', '') if 'LSEQ' in df.columns else ''
        cdr3 = str(df.loc[bc, 'CDR3']).upper().replace('-', '')
    except Exception:
        ax.text(0.5, 0.5, f'Barcode not found\n{bc}',
                ha='center', va='center', transform=ax.transAxes, fontsize=7, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return (None, tr_prob, xgb_prob, target.upper().replace('_','-'), '?', '#555', '')

    # 2-D IG attribution
    attr_enc = attr_cdr3 = None
    max_vh = max_vl = max_cdr3 = 0
    vh_only = False

    if ig_data is not None and bc in ig_data['barcodes']:
        idx       = ig_data['barcodes'].index(bc)
        attr_enc  = ig_data['attr_enc'][idx]
        attr_cdr3 = ig_data['attr_cdr3'][idx]
        max_vh    = ig_data['max_vh']
        max_vl    = ig_data['max_vl']
        max_cdr3  = ig_data['max_cdr3']
        vh_only   = ig_data['vh_only']
    elif tr_path is not None:
        try:
            import torch
            from captum.attr import IntegratedGradients
            from models.transformer_onehot import TransformerOneHotModel, one_hot_encode_sequence_2d
            m = TransformerOneHotModel.load(tr_path)
            max_vh = m.max_heavy_len; max_vl = m.max_light_len; max_cdr3 = m.max_hcdr3_len
            vh_only = m._vh_only()
            enc_h = torch.from_numpy(one_hot_encode_sequence_2d(vh, max_vh)).float()
            enc_c = torch.from_numpy(one_hot_encode_sequence_2d(cdr3, max_cdr3)).float()
            if vh_only:
                enc_hl = enc_h.unsqueeze(0).to(m.device)
            else:
                enc_l  = torch.from_numpy(one_hot_encode_sequence_2d(vl or '', max_vl)).float()
                enc_hl = torch.cat([enc_h, enc_l], dim=0).unsqueeze(0).to(m.device)
            enc_c = enc_c.unsqueeze(0).to(m.device)
            m.model.eval()
            attr = IntegratedGradients(m.model).attribute(
                (enc_hl, enc_c),
                baselines=(torch.zeros_like(enc_hl), torch.zeros_like(enc_c)),
                target=1, n_steps=ig_steps)
            attr_enc  = attr[0].squeeze(0).detach().cpu().numpy()
            attr_cdr3 = attr[1].squeeze(0).detach().cpu().numpy()
        except Exception as e:
            ax.text(0.5, 0.5, f'IG failed:\n{e}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=6, color='#888')
            ax.set_xticks([]); ax.set_yticks([])
            return (true_label, tr_prob, xgb_prob, target.upper().replace('_','-'), '?', '#555', cdr3)

    if attr_enc is None:
        ax.text(0.5, 0.5, 'IG data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=7, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return (true_label, tr_prob, xgb_prob, target.upper().replace('_','-'), '?', '#555', cdr3)

    AA_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    all_rows = []
    y = 0

    # ── TOP: HCDR3 — all positions, labelled "CDR3-XX (AA)" ──────────────
    for pos, aa in enumerate(cdr3):
        if pos >= attr_cdr3.shape[0]: break
        if aa in AA_IDX:
            all_rows.append((y, float(attr_cdr3[pos, AA_IDX[aa]]),
                             _aa_color(aa), f"CDR3_{pos+1:02d}_{aa}"))
        else:
            all_rows.append((y, 0.0, '#BBBBBB', f"CDR3_{pos+1:02d}_{aa}"))
        y += 1

    # ── BOTTOM: VHVL — top-N by |IG|, labelled "VH-XXX (AA)" / "VL-XXX (AA)" ──
    vhvl_items = []
    for i, aa in enumerate(vh[:max_vh]):
        if aa in AA_IDX and i < attr_enc.shape[0]:
            vhvl_items.append((f"VH_{i+1:03d}_{aa}",
                                float(attr_enc[i, AA_IDX[aa]]), _aa_color(aa)))
    if not vh_only and vl and attr_enc.shape[0] > max_vh:
        for i, aa in enumerate(vl[:max_vl]):
            enc_pos = max_vh + i
            if aa in AA_IDX and enc_pos < attr_enc.shape[0]:
                vhvl_items.append((f"VL_{i+1:03d}_{aa}",
                                    float(attr_enc[enc_pos, AA_IDX[aa]]), _aa_color(aa)))
    vhvl_items.sort(key=lambda x: abs(x[1]), reverse=True)

    for label, ig_val, col in vhvl_items[:n_vhvl_top]:
        all_rows.append((y, ig_val, col, label))
        y += 1

    # ── Render ────────────────────────────────────────────────────────────
    ys     = [r[0] for r in all_rows]
    vals   = [r[1] for r in all_rows]
    colors = [r[2] for r in all_rows]
    labels = [r[3] for r in all_rows]

    ax.barh(ys, vals, color=colors, height=bar_height,
            edgecolor='none', linewidth=0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.5, fontfamily='monospace', color='#111')
    ax.invert_yaxis()
    ax.axvline(0, color='#888', lw=0.7, ls='-', zorder=3)
    ax.tick_params(axis='x', labelsize=6.0)
    ax.grid(axis='x', alpha=0.18, lw=0.3, zorder=0)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('Transformer IG attribution\n(<- FAIL  |  PASS ->)', fontsize=6.5, labelpad=2)

    # Compact AA class legend inside panel (lower right)
    _short = ['Cationic (R,K,H)', 'Anionic (D,E)', 'Hydrophobic/Aromatic', 'Small/Polar']
    handles = [mpatches.Patch(facecolor=c, edgecolor='none', label=l)
               for c, l in zip(_AA_CLASS_COLORS, _short)]
    ax.legend(handles=handles, title='AA class', title_fontsize=4.5,
              fontsize=4.0, loc='lower right', frameon=True, framealpha=0.92,
              edgecolor='#ccc', handlelength=0.9, handleheight=0.75,
              borderpad=0.5, labelspacing=0.3)

    filter_name = target.upper().replace('_', '-')
    if true_label is not None:
        outcome_str = 'FAIL' if true_label == 0 else 'PASS'
        title_col   = '#C0392B' if true_label == 0 else '#27AE60'
    else:
        outcome_str, title_col = '?', '#555'

    return (true_label, tr_prob, xgb_prob, filter_name, outcome_str, title_col, cdr3)


def _render_mutagenesis_heatmap(ax, bc, r, title_prefix=''):
    """
    CDR3 single-point mutagenesis heatmap — rectangular cells.

    Rows    = CDR3 positions (x-axis)
    Columns = 20 mutant AAs  (y-axis, ACDEFGHIKLMNPQRSTVWY order)
    Colour  = Transformer PASS probability after mutation.
              Palette: white→navy  (white=low=FAIL, dark blue=high=PASS)
              This avoids the distracting red since FAIL abs start red anyway.
    Wild-type AA at each position: white circle marker.
    """
    import matplotlib.colors as _mc

    df      = r['df']
    target  = r['target']
    tr_path = r.get('transformer_model_path')

    try:
        vh   = str(df.loc[bc, 'HSEQ']).upper().replace('-', '')
        vl   = str(df.loc[bc, 'LSEQ']).upper().replace('-', '') if 'LSEQ' in df.columns else ''
        cdr3 = str(df.loc[bc, 'CDR3']).upper().replace('-', '')
    except Exception:
        ax.text(0.5, 0.5, 'Barcode not found', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return

    if not tr_path:
        ax.text(0.5, 0.5, 'Transformer model\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return

    try:
        from models.transformer_onehot import TransformerOneHotModel
        m = TransformerOneHotModel.load(tr_path)
    except Exception as e:
        ax.text(0.5, 0.5, f'Model load failed:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=6, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return

    n_pos = len(cdr3)
    n_aa  = len(AMINO_ACIDS)
    # hmap shape: (n_aa, n_pos) — rows=AAs, cols=CDR3 positions
    hmap  = np.zeros((n_aa, n_pos), dtype=np.float32)
    wt_prob = m.predict_single(bc, vh, vl, cdr3)

    cdr3_start = vh.find(cdr3)
    for pos in range(n_pos):
        for aa_i, mut_aa in enumerate(AMINO_ACIDS):
            mut_cdr3 = cdr3[:pos] + mut_aa + cdr3[pos+1:]
            vh_mut   = (vh[:cdr3_start] + mut_cdr3 + vh[cdr3_start + n_pos:]
                        if cdr3_start >= 0 else vh)
            try:
                hmap[aa_i, pos] = m.predict_single(bc, vh_mut, vl, mut_cdr3)
            except Exception:
                hmap[aa_i, pos] = wt_prob

    # ── Palette: light salmon (=0, FAIL) → white (=0.5) → steel blue (=1, PASS)
    # Softer red at the FAIL end avoids an all-red heatmap for low-scoring antibodies
    # while still clearly conveying fail (warm) vs pass (cool) directionality.
    cmap = _mc.LinearSegmentedColormap.from_list(
        'fail_pass',
        ['#F4CCCC',   # 0.00 — very light red/salmon  (FAIL)
         '#E88080',   # 0.25 — medium salmon
         '#FFFFFF',   # 0.50 — white (neutral)
         '#90BDD9',   # 0.75 — light steel blue
         '#1A5276'],  # 1.00 — dark navy blue           (PASS)
        N=256)

    # Force rectangular aspect: width per cell = height per cell
    # We set aspect='equal' then let imshow clip to the axes box
    im = ax.imshow(hmap, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect='auto', interpolation='nearest')

    # Wild-type marker — filled dark circle so it stands out on white cells
    aa_idx_map = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for pos, wt_aa in enumerate(cdr3):
        if wt_aa in aa_idx_map:
            ax.plot(pos, aa_idx_map[wt_aa], 's',
                    ms=3.5, markerfacecolor='#222222',
                    markeredgecolor='white', markeredgewidth=0.4, zorder=5)

    # Axes
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([cdr3[p] for p in range(n_pos)],
                       fontsize=6.5, fontfamily='monospace', color='#111')
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(list(AMINO_ACIDS), fontsize=6.0, fontfamily='monospace', color='#111')
    ax.set_xlabel('CDR3 position  (WT amino acid)', fontsize=6.5, labelpad=2)
    ax.set_ylabel('Mutant AA', fontsize=6.5, labelpad=2)
    ax.tick_params(axis='both', length=1.5, width=0.5)
    for sp in ax.spines.values(): sp.set_linewidth(0.4)

    # Slim vertical colourbar on the right
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025, aspect=30)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0\n(FAIL)', '0.5', '1\n(PASS)'], fontsize=4.0)
    cbar.ax.tick_params(width=0.4, length=1.5)
    cbar.outline.set_linewidth(0.4)
    # Add colour hint labels
    cbar.ax.text(0.5, -0.02, '← FAIL', transform=cbar.ax.transAxes,
                 fontsize=3.8, ha='center', va='top', color='#C0392B')
    cbar.ax.text(0.5, 1.02, 'PASS →', transform=cbar.ax.transAxes,
                 fontsize=3.8, ha='center', va='bottom', color='#1A5276')

    # (WT probability not shown — method difference vs XGBoost score already
    #  explained by "Transformer onehot" label in the panel title)


def _build_single_pair_figure(bc_fail, bc_pass, r1, r2,
                               n_vhvl_top, ig_steps,
                               tr_proba_1=None, tr_proba_2=None):
    """
    One A4 waterfall figure — 2 rows × 3 columns.

    Layout (figure fraction):
      col a  PASS waterfall   (left)
      col b  FAIL waterfall   (right of centre — extra gap from col a)
      col c  CDR3 mutagenesis (FAIL antibody)

    Plots fill top 75% of A4; bottom 25% is blank for legend text.

    Row geometry:
      ROW0  y = 0.750 → 0.415   PSR
      GAP        0.415 → 0.365  (titles c/d/f sit here)
      ROW1  y = 0.365 → 0.045   SEC
      LEGEND     0.045 → 0.000  blank
    """
    A4_W, A4_H = 8.27, 11.69

    # ── Column geometry ───────────────────────────────────────────────────
    LEFT  = 0.08
    RIGHT = 0.99

    # Widths
    COL_A_W = 0.275   # PASS waterfall
    COL_B_W = 0.275   # FAIL waterfall
    COL_C_W = 0.290   # mutagenesis heatmap

    # Gaps: normal gap between a and b; EXTRA gap before b to push FAIL right
    GAP_AB = 0.048    # wider than normal — "moves FAIL column right"
    GAP_BC = 0.028

    col_a_l = LEFT
    col_b_l = col_a_l + COL_A_W + GAP_AB
    col_c_l = col_b_l + COL_B_W + GAP_BC

    # ── Row geometry (plots = top 75% of A4) ─────────────────────────────
    ROW0_TOP    = 0.955
    ROW0_BOTTOM = 0.545   # row 0 height ~0.410
    ROW1_TOP    = 0.450   # gap = 0.095 (~1.1 in) for half-inch visual space
    ROW1_BOTTOM = 0.055   # row 1 height = 0.420
    # legend area: y = 0.000 → 0.055  (≈ 64 mm on A4 portrait)

    row0_h = ROW0_TOP - ROW0_BOTTOM
    row1_h = ROW1_TOP - ROW1_BOTTOM

    TITLE_Y_ROW0 = ROW0_TOP + 0.050   # ~0.5 inch above row 0 top
    TITLE_Y_ROW1 = ROW1_TOP + 0.030
    BC_Y_ROW0    = TITLE_Y_ROW0 - 0.014
    BC_Y_ROW1    = TITLE_Y_ROW1 - 0.014

    fig = plt.figure(figsize=(A4_W, A4_H))

    ax_a = fig.add_axes([col_a_l, ROW0_BOTTOM, COL_A_W, row0_h])  # PSR PASS
    ax_b = fig.add_axes([col_b_l, ROW0_BOTTOM, COL_B_W, row0_h])  # PSR FAIL
    ax_e = fig.add_axes([col_c_l, ROW0_BOTTOM, COL_C_W, row0_h])  # PSR muta
    ax_c = fig.add_axes([col_a_l, ROW1_BOTTOM, COL_A_W, row1_h])  # SEC PASS
    ax_d = fig.add_axes([col_b_l, ROW1_BOTTOM, COL_B_W, row1_h])  # SEC FAIL
    ax_f = fig.add_axes([col_c_l, ROW1_BOTTOM, COL_C_W, row1_h])  # SEC muta

    axes = {
        ('psr','pass'): ax_a, ('psr','fail'): ax_b,
        ('sec','pass'): ax_c, ('sec','fail'): ax_d,
    }

    def _xp(proba_dict, bc):
        return float(proba_dict[bc]) if proba_dict and bc and bc in proba_dict else None

    def _fs(r):
        return r['target'].upper().split('_')[0]

    TITLE_Y = {
        ('psr','pass'): TITLE_Y_ROW0, ('psr','fail'): TITLE_Y_ROW0,
        ('sec','pass'): TITLE_Y_ROW1, ('sec','fail'): TITLE_Y_ROW1,
    }
    BC_Y = {
        ('psr','pass'): BC_Y_ROW0, ('psr','fail'): BC_Y_ROW0,
        ('sec','pass'): BC_Y_ROW1, ('sec','fail'): BC_Y_ROW1,
    }

    panel_map = [
        (('psr','pass'), r1, bc_pass, 'a', _xp(tr_proba_1, bc_pass), _fs(r1)),
        (('psr','fail'), r1, bc_fail, 'b', _xp(tr_proba_1, bc_fail), _fs(r1)),
        (('sec','pass'), r2, bc_pass, 'c', _xp(tr_proba_2, bc_pass), _fs(r2)),
        (('sec','fail'), r2, bc_fail, 'd', _xp(tr_proba_2, bc_fail), _fs(r2)),
    ]

    # ── Waterfall panels ──────────────────────────────────────────────────
    meta = {}
    for key, r, bc, letter, xgb_p, fshort in panel_map:
        ax = axes[key]
        if bc is None:
            ax.text(0.5, 0.5, 'No antibody\nfound',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=7, color='#888', style='italic')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            meta[key] = (None, None, None, fshort, '?', '#555', '')
        else:
            meta[key] = _waterfall_single_ab(
                ax, bc, r, n_vhvl_top=n_vhvl_top,
                ig_steps=ig_steps, bar_height=0.38, xgb_prob=xgb_p)
        if key[0] == 'psr':
            ax.set_xlabel('')

    # ── Mutagenesis panels ────────────────────────────────────────────────
    _render_mutagenesis_heatmap(ax_e, bc_fail, r1)
    _render_mutagenesis_heatmap(ax_f, bc_fail, r2)

    # ── Titles + barcodes via fig.text ────────────────────────────────────
    for key, r, bc, letter, xgb_p, fshort in panel_map:
        ax  = axes[key]
        x0  = ax.get_position().x0
        true_lbl, tr_prob, xgb_prob, fname, outcome, tcol, cdr3_seq = meta[key]
        tr_sc    = meta[key][1]   # tr_prob from _waterfall_single_ab
        score_s  = f"={tr_sc:.4f}" if tr_sc is not None else ''
        title_ln = f"trans_onehot {fshort}{score_s}  |  actual={outcome}"
        bc_line  = f"{bc}:{cdr3_seq}" if cdr3_seq else (bc or '')

        ty = TITLE_Y[key]
        fig.text(x0, ty, letter,
                 fontsize=9, fontweight='bold', va='bottom', ha='left',
                 color='#111', transform=fig.transFigure)
        fig.text(x0 + 0.018, ty, title_ln,
                 fontsize=6.5, fontweight='bold', va='bottom', ha='left',
                 color=tcol, transform=fig.transFigure)
        # ── Barcode + CDR3 — larger font, bold barcode ─────────────────
        fig.text(x0 + 0.018, BC_Y[key], bc_line,
                 fontsize=6.5, va='top', ha='left',        # was 5.0
                 color='#333', fontfamily='monospace',
                 transform=fig.transFigure)

    # ── Mutagenesis titles — same format as waterfall panels ─────────────
    # e/f titles use the FAIL antibody info and the respective filter
    fail_meta_psr = meta.get(('psr','fail'), (None, None, None, _fs(r1), '?', '#C0392B', ''))
    fail_meta_sec = meta.get(('sec','fail'), (None, None, None, _fs(r2), '?', '#C0392B', ''))

    for ax_mut, row_key, ltr, fail_meta, xgb_p_fail, fshort in [
        (ax_e, ('psr','fail'), 'e', fail_meta_psr, _xp(tr_proba_1, bc_fail), _fs(r1)),
        (ax_f, ('sec','fail'), 'f', fail_meta_sec, _xp(tr_proba_2, bc_fail), _fs(r2)),
    ]:
        x0  = ax_mut.get_position().x0
        ty  = TITLE_Y[row_key]
        true_lbl, tr_prob, xgb_prob, fname, outcome, tcol, cdr3_seq = fail_meta

        tr_sc_mut = fail_meta[1]  # Transformer PSR or SEC prob
        score_s   = f"={tr_sc_mut:.4f}" if tr_sc_mut is not None else ''
        title_ln  = f"CDR3 mutagenesis  trans_onehot {fshort}{score_s}  |  actual={outcome}"
        bc_line  = f"{bc_fail}:{cdr3_seq}" if cdr3_seq else (bc_fail or '')

        fig.text(x0, ty, ltr,
                 fontsize=9, fontweight='bold', va='bottom', ha='left',
                 color='#111', transform=fig.transFigure)
        fig.text(x0 + 0.018, ty, title_ln,
                 fontsize=6.5, fontweight='bold', va='bottom', ha='left',
                 color=tcol, transform=fig.transFigure)
        fig.text(x0 + 0.018, BC_Y[row_key], bc_line,
                 fontsize=6.5, va='top', ha='left',
                 color='#333', fontfamily='monospace',
                 transform=fig.transFigure)

    return fig


def _waterfall_shap_single_ab(ax, bc: str, r: dict, top_n: int = 25,
                               model_key: str = 'xgb_model'):
    """
    Horizontal SHAP waterfall for ONE antibody × ONE filter using tree model.

    Computes per-sample SHAP for the biophysical features of antibody bc.
    Rows sorted by |SHAP| (most important first, top_n shown).
    Bar direction: negative = FAIL, positive = PASS.
    Dot colour = feature value (RdBu_r: red=high, blue=low).

    Returns (true_label, pred_prob, filter_name, outcome_str, title_col)
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc2

    df     = r['df']
    target = r['target']
    model  = r.get(model_key)

    try:
        true_label = int(df.loc[bc, target])
    except Exception:
        true_label = None

    if model is None:
        ax.text(0.5, 0.5, f'{model_key} not available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=7, color='#888', style='italic')
        ax.set_xticks([]); ax.set_yticks([])
        filter_name = target.upper().replace('_', '-')
        return (true_label, None, filter_name, '?', '#555')

    try:
        import shap as _shap
        fb     = model.fb_
        ne_idx = np.asarray(fb.non_embedding_indices, dtype=int)
        names  = [fb.non_embedding_feature_names[i] for i in range(len(ne_idx))]

        # Transform single antibody to feature vector
        X_full = fb.transform(df.loc[[bc]], None)        # (1, all_feats)
        X_row  = X_full[:, ne_idx]                       # (1, n_biophys)
        pred_prob = float(model.model.predict_proba(X_row)[0, 1])

        # Per-sample SHAP — use same robust extraction as _compute_tree_shap
        explainer = _shap.TreeExplainer(model.model)
        sv_raw    = explainer.shap_values(X_row, check_additivity=False)

        # Normalise output shape (varies: list, 2D, 3D)
        if isinstance(sv_raw, list):
            # list[n_classes][n_samples, n_feats] — take class 1
            sv = np.asarray(sv_raw[1], dtype=np.float64).reshape(-1)
        elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 3:
            # (n_samples, n_feats, n_classes) or (n_classes, n_samples, n_feats)
            sv_arr = np.asarray(sv_raw, dtype=np.float64)
            if sv_arr.shape[0] == 1:      # (1, n_feats, n_classes)
                sv = sv_arr[0, :, 1]
            else:                          # (n_classes, 1, n_feats)
                sv = sv_arr[1, 0, :]
        elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 2:
            sv = np.asarray(sv_raw[0], dtype=np.float64)
        else:
            sv = np.asarray(sv_raw, dtype=np.float64).reshape(-1)

        xv = X_row[0]   # feature values for colour

        # Sort by |SHAP|, show top_n
        order  = np.argsort(np.abs(sv))[::-1][:top_n]
        sv_top = sv[order]
        xv_top = xv[order]
        labels = [names[i].replace('cdr3_charge_ph7', 'cdr3_charge').replace('_ph7', '')
                  for i in order]

        # Colour by feature value
        cmap = _cm.get_cmap('RdBu_r')
        lo, hi = xv_top.min(), xv_top.max()
        norm_xv = (xv_top - lo) / (hi - lo + 1e-10)
        colors  = [cmap(v) for v in norm_xv]

        ys = list(range(len(sv_top)))
        ax.barh(ys, sv_top, color=colors, height=0.60,
                edgecolor='none', linewidth=0, zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=6.5, fontfamily='monospace', color='#111')
        ax.invert_yaxis()
        ax.axvline(0, color='#888', lw=0.7, ls='-', zorder=3)
        ax.tick_params(axis='x', labelsize=6.0)
        ax.grid(axis='x', alpha=0.18, lw=0.3, zorder=0)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        ax.set_xlabel('SHAP value\n(<- FAIL  |  PASS ->)', fontsize=6.5, labelpad=2)

        # Feature-value colourbar
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
        _axins = _inset(ax, width='36%', height='2.4%',
                        loc='upper right', borderpad=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=_mc2.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=_axins, orientation='horizontal')
        cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High'], fontsize=4.5)
        cb.set_label('Feature value', fontsize=4.5, labelpad=1)
        cb.ax.xaxis.set_label_position('top'); cb.ax.xaxis.tick_top()
        cb.ax.tick_params(width=0.4, length=1.5)

    except Exception as e:
        ax.text(0.5, 0.5, f'SHAP failed:\n{e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=6, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        pred_prob = None

    filter_name = target.upper().replace('_', '-')
    if true_label is not None:
        outcome_str = 'FAIL' if true_label == 0 else 'PASS'
        title_col   = '#C0392B' if true_label == 0 else '#27AE60'
    else:
        outcome_str, title_col = '?', '#555'
    return (true_label, pred_prob, filter_name, outcome_str, title_col)


def _render_mutagenesis_heatmap_tree(ax, bc, r, model_key='xgb_model'):
    """
    CDR3 single-point mutagenesis heatmap using a tree model (XGBoost or RF).

    For each CDR3 position × each of 20 AAs:
      - Replace CDR3[pos] with mutant AA in the full sequence
      - Recompute biophysical features
      - Predict PASS probability with the tree model

    Colour scale: light salmon (FAIL) → white → navy (PASS)  — same as Transformer version.
    Wild-type position: filled dark square marker.
    """
    import matplotlib.colors as _mc

    df      = r['df']
    target  = r['target']
    model   = r.get(model_key)

    try:
        vh   = str(df.loc[bc, 'HSEQ']).upper().replace('-', '')
        vl   = str(df.loc[bc, 'LSEQ']).upper().replace('-', '') if 'LSEQ' in df.columns else ''
        cdr3 = str(df.loc[bc, 'CDR3']).upper().replace('-', '')
    except Exception:
        ax.text(0.5, 0.5, 'Barcode not found', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return

    if model is None:
        ax.text(0.5, 0.5, f'{model_key}\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return

    n_pos = len(cdr3)
    n_aa  = len(AMINO_ACIDS)
    hmap  = np.zeros((n_aa, n_pos), dtype=np.float32)

    try:
        fb     = model.fb_
        ne_idx = np.array(fb.non_embedding_indices)

        def _predict_row(bc_mut, hseq, lseq, cdr3_mut):
            """Build a 1-row DataFrame matching the original, predict prob."""
            row = df.loc[[bc]].copy()
            row['HSEQ'] = hseq
            if 'LSEQ' in row.columns: row['LSEQ'] = lseq
            row['CDR3']  = cdr3_mut
            row.index    = [bc_mut]
            X = fb.transform(row, None)[:, ne_idx]
            return float(model.model.predict_proba(X)[0, 1])

        wt_prob = _predict_row(bc, vh, vl, cdr3)
        cdr3_start = vh.find(cdr3)

        for pos in range(n_pos):
            for aa_i, mut_aa in enumerate(AMINO_ACIDS):
                mut_cdr3 = cdr3[:pos] + mut_aa + cdr3[pos+1:]
                vh_mut   = (vh[:cdr3_start] + mut_cdr3 + vh[cdr3_start + n_pos:]
                            if cdr3_start >= 0 else vh)
                try:
                    hmap[aa_i, pos] = _predict_row(f'{bc}_mut', vh_mut, vl, mut_cdr3)
                except Exception:
                    hmap[aa_i, pos] = wt_prob

    except Exception as e:
        ax.text(0.5, 0.5, f'Mutagenesis failed:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=6, color='#888')
        ax.set_xticks([]); ax.set_yticks([])
        return

    cmap = _mc.LinearSegmentedColormap.from_list(
        'fail_pass',
        ['#F4CCCC', '#E88080', '#FFFFFF', '#90BDD9', '#1A5276'], N=256)

    im = ax.imshow(hmap, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect='auto', interpolation='nearest')

    aa_idx_map = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for pos, wt_aa in enumerate(cdr3):
        if wt_aa in aa_idx_map:
            ax.plot(pos, aa_idx_map[wt_aa], 's',
                    ms=3.5, markerfacecolor='#222222',
                    markeredgecolor='white', markeredgewidth=0.4, zorder=5)

    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([cdr3[p] for p in range(n_pos)],
                       fontsize=6.5, fontfamily='monospace', color='#111')
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(list(AMINO_ACIDS), fontsize=6.0, fontfamily='monospace', color='#111')
    ax.set_xlabel('CDR3 position  (WT amino acid)', fontsize=6.5, labelpad=2)
    ax.set_ylabel('Mutant AA', fontsize=6.5, labelpad=2)
    ax.tick_params(axis='both', length=1.5, width=0.5)
    for sp in ax.spines.values(): sp.set_linewidth(0.4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025, aspect=30)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0\n(FAIL)', '0.5', '1\n(PASS)'], fontsize=4.0)
    cbar.ax.tick_params(width=0.4, length=1.5)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.text(0.5, -0.02, '← FAIL', transform=cbar.ax.transAxes,
                 fontsize=3.8, ha='center', va='top', color='#C0392B')
    cbar.ax.text(0.5, 1.02, 'PASS →', transform=cbar.ax.transAxes,
                 fontsize=3.8, ha='center', va='bottom', color='#1A5276')


def _build_shap_pair_figure(bc_fail, bc_pass, r1, r2,
                             model_key, top_n, ig_steps):
    """
    One A4 figure — 2 rows × 3 columns, same layout as _build_single_pair_figure.

    col a  PASS SHAP waterfall  (biophysical features, per-sample SHAP)
    col b  FAIL SHAP waterfall
    col c  CDR3 mutagenesis     (Transformer one-hot, same as IG figure)

    model_key : 'xgb_model' → XGBoost biophysical
                'rf_model'  → Random Forest biophysical
    """
    model_tag = model_key.replace('_model', '').upper()   # 'XGB' or 'RF'

    A4_W, A4_H = 8.27, 11.69
    LEFT  = 0.08; RIGHT = 0.99
    GAP_W = 0.048
    COL_A_W = 0.275; COL_B_W = 0.275; COL_C_W = 0.290
    col_a_l = LEFT
    col_b_l = col_a_l + COL_A_W + GAP_W
    col_c_l = col_b_l + COL_B_W + 0.028

    ROW0_TOP    = 0.955; ROW0_BOTTOM = 0.545   # row 0 height = 0.410
    ROW1_TOP    = 0.455; ROW1_BOTTOM = 0.055   # gap = 0.090 ≈ 0.5 inch on A4
    row0_h = ROW0_TOP - ROW0_BOTTOM
    row1_h = ROW1_TOP - ROW1_BOTTOM

    TITLE_Y_ROW0 = ROW0_TOP + 0.050
    TITLE_Y_ROW1 = ROW1_TOP + 0.030
    BC_Y_ROW0    = TITLE_Y_ROW0 - 0.014
    BC_Y_ROW1    = TITLE_Y_ROW1 - 0.014

    fig = plt.figure(figsize=(A4_W, A4_H))

    ax_a = fig.add_axes([col_a_l, ROW0_BOTTOM, COL_A_W, row0_h])   # PSR PASS
    ax_b = fig.add_axes([col_b_l, ROW0_BOTTOM, COL_B_W, row0_h])   # PSR FAIL
    ax_e = fig.add_axes([col_c_l, ROW0_BOTTOM, COL_C_W, row0_h])   # PSR muta
    ax_c = fig.add_axes([col_a_l, ROW1_BOTTOM, COL_A_W, row1_h])   # SEC PASS
    ax_d = fig.add_axes([col_b_l, ROW1_BOTTOM, COL_B_W, row1_h])   # SEC FAIL
    ax_f = fig.add_axes([col_c_l, ROW1_BOTTOM, COL_C_W, row1_h])   # SEC muta

    def _fs(r): return r['target'].upper().split('_')[0]

    def _cdr3(bc, r):
        try: return str(r['df'].loc[bc, 'CDR3']).upper().replace('-', '')
        except: return ''

    # ── SHAP waterfall panels ─────────────────────────────────────────────
    meta_a = _waterfall_shap_single_ab(ax_a, bc_pass, r1, top_n, model_key) if bc_pass else (None,None,_fs(r1),'?','#555')
    meta_b = _waterfall_shap_single_ab(ax_b, bc_fail, r1, top_n, model_key) if bc_fail else (None,None,_fs(r1),'?','#555')
    meta_c = _waterfall_shap_single_ab(ax_c, bc_pass, r2, top_n, model_key) if bc_pass else (None,None,_fs(r2),'?','#555')
    meta_d = _waterfall_shap_single_ab(ax_d, bc_fail, r2, top_n, model_key) if bc_fail else (None,None,_fs(r2),'?','#555')

    # Suppress x-label on row 0 to keep gap clean
    ax_a.set_xlabel(''); ax_b.set_xlabel('')

    # ── Mutagenesis panels — scores from the same SHAP model (XGB or RF) ──
    _render_mutagenesis_heatmap_tree(ax_e, bc_fail, r1, model_key=model_key)
    _render_mutagenesis_heatmap_tree(ax_f, bc_fail, r2, model_key=model_key)

    # ── Titles via fig.text ───────────────────────────────────────────────
    for ax, meta, bc, ty, bcy, fshort, r in [
        (ax_a, meta_a, bc_pass, TITLE_Y_ROW0, BC_Y_ROW0, _fs(r1), r1),
        (ax_b, meta_b, bc_fail, TITLE_Y_ROW0, BC_Y_ROW0, _fs(r1), r1),
        (ax_c, meta_c, bc_pass, TITLE_Y_ROW1, BC_Y_ROW1, _fs(r2), r2),
        (ax_d, meta_d, bc_fail, TITLE_Y_ROW1, BC_Y_ROW1, _fs(r2), r2),
    ]:
        true_lbl, prob, fname, outcome, tcol = meta
        score_s  = f"={prob:.4f}" if prob is not None else ''
        title_ln = f"{model_tag}-biophysical {fshort}{score_s}  |  actual={outcome}"
        bc_line  = f"{bc}:{_cdr3(bc, r)}" if bc else ''
        ltr      = {ax_a:'a', ax_b:'b', ax_c:'c', ax_d:'d'}[ax]
        x0 = ax.get_position().x0
        fig.text(x0, ty, ltr,
                 fontsize=9, fontweight='bold', va='bottom', ha='left',
                 color='#111', transform=fig.transFigure)
        fig.text(x0+0.018, ty, title_ln,
                 fontsize=6.5, fontweight='bold', va='bottom', ha='left',
                 color=tcol, transform=fig.transFigure)
        fig.text(x0+0.018, bcy, bc_line,
                 fontsize=6.5, va='top', ha='left',
                 color='#333', fontfamily='monospace', transform=fig.transFigure)

    # Mutagenesis titles (e/f) — score from the SHAP model, not Transformer
    fail_prob_psr = meta_b[1]; fail_prob_sec = meta_d[1]
    for ax_mut, row_ty, ltr, fshort, fp in [
        (ax_e, TITLE_Y_ROW0, 'e', _fs(r1), fail_prob_psr),
        (ax_f, TITLE_Y_ROW1, 'f', _fs(r2), fail_prob_sec),
    ]:
        score_s = f"={fp:.4f}" if fp is not None else ''
        mut_lbl = f"CDR3 mutagenesis  {model_tag}-biophysical {fshort}{score_s}  |  actual=FAIL"
        x0 = ax_mut.get_position().x0
        fig.text(x0, row_ty, ltr,
                 fontsize=9, fontweight='bold', va='bottom', ha='left',
                 color='#111', transform=fig.transFigure)
        fig.text(x0+0.018, row_ty, mut_lbl,
                 fontsize=6.5, fontweight='bold', va='bottom', ha='left',
                 color='#C0392B', transform=fig.transFigure)
        if bc_fail:
            fig.text(x0+0.018, row_ty - 0.014,
                     f"{bc_fail}:{_cdr3(bc_fail, r1)}",
                     fontsize=6.5, va='top', ha='left',
                     color='#333', fontfamily='monospace', transform=fig.transFigure)

    return fig


def build_figure_shap_per_antibody(r1, r2, out_stem, log,
                                    fail_list, pass_list,
                                    model_key='xgb_model',
                                    n_pairs=20, top_n=25, ig_steps=100):
    """
    Generate per-antibody SHAP waterfall figures — same A4 2×3 layout as IG figure:
      col a  PASS SHAP (biophysical)
      col b  FAIL SHAP (biophysical)
      col c  CDR3 mutagenesis (Transformer)

    model_key : 'xgb_model' → XGBoost  |  'rf_model' → Random Forest
    Output: {out_stem}_{model_tag}_biophysical_pair{i:02d}.{tiff|pdf|png}
    """
    if not fail_list and not pass_list:
        log(f"[{model_key}] No candidates — skipping"); return

    model_tag = model_key.replace('_model', '')
    n_fig = min(len(fail_list), len(pass_list), n_pairs)
    if n_fig == 0: n_fig = max(len(fail_list), len(pass_list), 1)

    log(f"[{model_key}] Generating {n_fig} SHAP waterfall figure(s) "
        f"(layout: PASS | FAIL | CDR3 mutagenesis)...")

    for i in range(n_fig):
        bc_fail = fail_list[i][0] if i < len(fail_list) else None
        bc_pass = pass_list[i][0] if i < len(pass_list) else None

        fig = _build_shap_pair_figure(
            bc_fail=bc_fail, bc_pass=bc_pass,
            r1=r1, r2=r2,
            model_key=model_key, top_n=top_n, ig_steps=ig_steps)

        pair_stem = f"{out_stem}_{model_tag}_biophysical_pair{i+1:02d}_shap"
        _save_fig(fig, pair_stem, log)
        log(f"[{model_key}] pair {i+1}/{n_fig} → {pair_stem}"
            f"  FAIL={bc_fail}  PASS={bc_pass}")


def build_figure_per_antibody_ig(r1, r2, out_stem, log,
                                   n_vhvl_top=6,
                                   ig_steps=100, n_pairs=20,
                                   _precomputed_lists=None):
    """
    Generate one A4 Transformer IG waterfall figure per (FAIL ab, PASS ab) pair.
    _precomputed_lists : (fail_list, pass_list) from _find_all_example_antibodies
                         — avoids recomputing if already done.
    """
    if _precomputed_lists is not None:
        fail_list, pass_list = _precomputed_lists
    else:
        fail_list, pass_list = _find_all_example_antibodies(r1, r2, log, n_each=n_pairs)

    if not fail_list and not pass_list:
        log("[per_ab] No suitable antibodies found — skipping")
        return

    # Build Transformer prob dicts from the new tuple format
    # Each element: (bc, tr_psr_prob, tr_sec_prob, xgb_psr_prob, xgb_sec_prob)
    tr_proba_1 = {}   # PSR Transformer probs
    tr_proba_2 = {}   # SEC Transformer probs
    for item in fail_list + pass_list:
        bc, tp1, tp2 = item[0], item[1], item[2]
        tr_proba_1[bc] = tp1
        tr_proba_2[bc] = tp2

    n_fig = min(len(fail_list), len(pass_list), n_pairs)
    if n_fig == 0:
        n_fig = max(len(fail_list), len(pass_list), 1)

    log(f"[per_ab] Generating {n_fig} per-antibody waterfall figure(s)...")

    for i in range(n_fig):
        bc_fail = fail_list[i][0] if i < len(fail_list) else None
        bc_pass = pass_list[i][0] if i < len(pass_list) else None

        fig = _build_single_pair_figure(
            bc_fail=bc_fail, bc_pass=bc_pass,
            r1=r1, r2=r2,
            n_vhvl_top=n_vhvl_top,
            ig_steps=ig_steps,
            tr_proba_1=tr_proba_1,
            tr_proba_2=tr_proba_2)

        pair_stem = f"{out_stem}_per_antibody_ig_pair{i+1:02d}"
        _save_fig(fig, pair_stem, log)
        log(f"[per_ab] pair {i+1}/{n_fig} -> {pair_stem}"
            f"  FAIL={bc_fail}  PASS={bc_pass}")


def build_figure_2row_6panels(r1: dict, r2: dict, args,
                               out_stem: str, log: _Log):
    """
    2-row x 3-column interpretability figure.
    Row 1 = PSR,  Row 2 = SEC.
    Col a/d = RF SHAP, col b/e = XGBoost SHAP, col c/f = Transformer IG.

    Panel c / f: SINGLE combined axis -- HCDR3 (all 20 AA) above dashed separator,
    VH framework (top 7 AA) below.  Shared x-axis; y-tick labels coloured by
    region; physicochemical legend inside the panel via ax.legend().

    All panels share the same y-tick font size (YTICK_FSZ = 6.5 pt).
    No figure super-title. No external legend strip.
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    from matplotlib.gridspec import GridSpec

    YTICK_FSZ = 6.5
    VH_TOP_N  = 7
    top_n     = 20

    # Height: each row holds ~(top_n + 20 HCDR3 + VH_TOP_N) rows worth of data
    ig_total  = 20 + VH_TOP_N + 2
    row_px    = max(top_n, ig_total) * 0.33 + 1.8   # inches per dataset row
    FIG_W     = 13.0
    FIG_H     = row_px * 2 + 1.2

    cmap_feat = _cm.get_cmap('RdBu_r')
    rng       = np.random.default_rng(42)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs_outer = GridSpec(2, 3, figure=fig,
                        width_ratios=[1.05, 1.05, 1.0],
                        wspace=0.58, hspace=0.32,
                        left=0.12, right=0.97,
                        top=0.97, bottom=0.04)

    row_labels = ['PSR', 'SEC']
    row_colors = ['#C0392B', '#2471A3']
    panel_abc  = [['a', 'b', 'c'], ['d', 'e', 'f']]
    ig_axes    = []

    for row_idx, (r, row_lbl, row_col, panels) in enumerate(
            zip([r1, r2], row_labels, row_colors, panel_abc)):

        rf_shap  = r['rf_shap']
        xgb_shap = r['xgb_shap']
        ig_data  = r['ig_data']

        ax_rf  = fig.add_subplot(gs_outer[row_idx, 0])
        ax_xgb = fig.add_subplot(gs_outer[row_idx, 1])
        ax_ig  = fig.add_subplot(gs_outer[row_idx, 2])
        ig_axes.append(ax_ig)

        def _pretty(l):
            return l.replace('cdr3_charge_ph7', 'cdr3_charge').replace('_ph7', '')

        # Row label
        ax_rf.annotate(
            row_lbl, xy=(0, 0.5), xytext=(-0.44, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=13, fontweight='bold', color=row_col,
            rotation=90, va='center', ha='center', annotation_clip=False)

        # RF SHAP
        if rf_shap is None:
            _render_blank(ax_rf, "RF model not found", "")
            rf_order  = list(range(top_n))
            rf_labels = [f"feature_{i}" for i in range(top_n)]
        else:
            names_rf = rf_shap['names']
            ma_rf    = rf_shap['mean_abs_shap']
            sv_rf    = rf_shap['shap_matrix']
            Xm_rf    = rf_shap['X_matrix']
            rf_order = list(np.argsort(ma_rf)[::-1][:top_n])
            rf_labels = [_pretty(names_rf[i]) for i in rf_order]
            for row, feat_idx in enumerate(rf_order):
                sv_col  = sv_rf[:, feat_idx]
                xv_col  = Xm_rf[:, feat_idx]
                lo, hi  = xv_col.min(), xv_col.max()
                norm_xv = (xv_col - lo) / (hi - lo + 1e-10)
                jitter  = rng.uniform(-0.36, 0.36, size=len(sv_col))
                ax_rf.scatter(sv_col, row + jitter,
                              c=cmap_feat(norm_xv), s=6, alpha=0.55,
                              linewidths=0, rasterized=True)
            ax_rf.axvline(0, color='#888', lw=0.6)
            ax_rf.set_yticks(range(top_n))
            ax_rf.set_yticklabels(rf_labels, fontsize=YTICK_FSZ)
            ax_rf.invert_yaxis()
            ax_rf.tick_params(axis='x', labelsize=YTICK_FSZ - 1)
            ax_rf.grid(axis='x', alpha=0.18, lw=0.3)
            for s in ('top', 'right'): ax_rf.spines[s].set_visible(False)

        ax_rf.set_xlabel('SHAP value\n(\u2190 FAIL  |  PASS \u2192)',
                         fontsize=YTICK_FSZ, labelpad=2)

        # Colorbar upper-right
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
        import matplotlib.colors as _mc2
        _axins = _inset(ax_rf, width='36%', height='2.2%',
                        loc='upper right', borderpad=1.0)
        sm_f = plt.cm.ScalarMappable(cmap=cmap_feat,
                                      norm=_mc2.Normalize(vmin=0, vmax=1))
        sm_f.set_array([])
        cb = plt.colorbar(sm_f, cax=_axins, orientation='horizontal')
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Low', 'High'], fontsize=4.5)
        cb.set_label('Feature value', fontsize=4.5, labelpad=1)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.tick_top()
        cb.ax.tick_params(width=0.4, length=1.5, labelsize=4.5)

        # XGBoost SHAP
        if xgb_shap is None or rf_shap is None:
            _render_blank(ax_xgb, "XGBoost not found", "")
        else:
            xgb_name_to_idx = {n: i for i, n in enumerate(xgb_shap['names'])}
            for row, rf_feat_idx in enumerate(rf_order):
                feat_name = rf_shap['names'][rf_feat_idx]
                xgb_idx   = xgb_name_to_idx.get(feat_name)
                if xgb_idx is None: continue
                sv_col  = xgb_shap['shap_matrix'][:, xgb_idx]
                xv_col  = xgb_shap['X_matrix'][:, xgb_idx]
                lo, hi  = xv_col.min(), xv_col.max()
                norm_xv = (xv_col - lo) / (hi - lo + 1e-10)
                jitter  = rng.uniform(-0.36, 0.36, size=len(sv_col))
                ax_xgb.scatter(sv_col, row + jitter,
                               c=cmap_feat(norm_xv), s=6, alpha=0.55,
                               linewidths=0, rasterized=True)
            ax_xgb.axvline(0, color='#888', lw=0.6)
            ax_xgb.set_yticks(range(len(rf_labels)))
            ax_xgb.set_yticklabels(rf_labels, fontsize=YTICK_FSZ)
            ax_xgb.invert_yaxis()
            ax_xgb.tick_params(axis='x', labelsize=YTICK_FSZ - 1)
            ax_xgb.grid(axis='x', alpha=0.18, lw=0.3)
            for s in ('top', 'right'): ax_xgb.spines[s].set_visible(False)

        ax_xgb.set_xlabel('SHAP value\n(\u2190 FAIL  |  PASS \u2192)',
                          fontsize=YTICK_FSZ, labelpad=2)

        # Transformer IG — single combined axis
        _render_ig_combined(ax_ig, ig_data, rng,
                            vh_top_n=VH_TOP_N, fsz_y=YTICK_FSZ, dot_size=6)

    # Panel titles via fig.text — filter name + lifted position
    fig.canvas.draw()
    TITLE_LIFT_2ROW = 0.036   # 0.5 in on the taller 2-row figure (FIG_H ≈ 18 in)
    for row_idx, (panels, row_r) in enumerate(zip(panel_abc, [r1, r2])):
        lbl_a, lbl_b, lbl_c = panels
        _filt = row_r['target'].upper().split('_')[0]   # 'PSR' or 'SEC'
        ax_ig = ig_axes[row_idx]
        ax_rf_row  = fig.axes[row_idx * 3]
        ax_xgb_row = fig.axes[row_idx * 3 + 1]
        for ax, letter, line1, line2 in [
            (ax_rf_row,  lbl_a, f'RF-{_filt}',
                                '(biophysical features)'),
            (ax_xgb_row, lbl_b, f'XGBoost-{_filt}',
                                '(same row order as RF — compare directly)'),
            (ax_ig,      lbl_c,
             f'Transformer onehot-{_filt}  ·  IG per amino acid',
             '(HCDR3 + VH framework, one row per AA)'),
        ]:
            pos = ax.get_position()
            x0, y1 = pos.x0, pos.y1
            ty = y1 + TITLE_LIFT_2ROW
            fig.text(x0, ty, letter,
                     fontsize=9, fontweight='bold', va='bottom', ha='left',
                     transform=fig.transFigure)
            fig.text(x0 + 0.018, ty, line1,
                     fontsize=6.5, fontweight='bold', va='bottom', ha='left',
                     transform=fig.transFigure)
            if line2:
                fig.text(x0 + 0.018, ty - 0.010, line2,
                         fontsize=5.5, va='top', ha='left', color='#555',
                         transform=fig.transFigure)

    _save_fig(fig, f"{out_stem}_2row", log)


def main():
    ap = argparse.ArgumentParser(
        description="MLAbDev — generate Nature Biotech interpretability figure")
    ap.add_argument('--db',             required=True,
                    help="Training database (.xlsx or .csv), "
                         "e.g. data/ipi_psr_trainset.xlsx")
    ap.add_argument('--target',         required=True,
                    help="Label column, e.g. psr_filter or sec_filter")

    ap.add_argument('--rf-lm',          default='biophysical',
                    choices=['biophysical', 'kmer'],
                    help="Feature mode for the RF model (default: biophysical)")
    ap.add_argument('--xgb-lm',         default='biophysical',
                    choices=['biophysical', 'kmer'],
                    help="Feature mode for the XGBoost model (default: biophysical)")
    ap.add_argument('--transformer-lm', default='onehot',
                    choices=['onehot', 'onehot_vh'],
                    help="Branch-1 mode for the Transformer model (default: onehot)")

    ap.add_argument('--model-dir',      default='build/pretrained_models',
                    help="Directory holding FINAL_*.pkl / FINAL_*.pt "
                         "(default: build/pretrained_models)")
    ap.add_argument('--outdir',         default=None,
                    help="Output directory. "
                         "Default: outputs/interp_{target}_{db_stem}")

    ap.add_argument('--max-samples',      type=int, default=3000,
                    help="Max antibodies used for SHAP (RF/XGBoost) AND IG (Transformer) "
                         "computation. Applied uniformly to all models. "
                         "0 = use ALL antibodies (recommended for final figures). "
                         "Set to e.g. 500 for fast exploratory runs. (default: 3000)")
    # Legacy aliases — still accepted so existing scripts don't break
    ap.add_argument('--shap-max-samples', type=int, default=None,
                    help="[legacy] Use --max-samples instead.")
    ap.add_argument('--ig-max-samples',   type=int, default=None,
                    help="[legacy] Use --max-samples instead. "
                         "0 = use ALL (default, recommended for final figures).")
    ap.add_argument('--ig-steps',         type=int, default=200,
                    help="IG integration steps (default: 200)")
    ap.add_argument('--n-pairs',          type=int, default=20,
                    help="Number of per-antibody figures to generate for IG, "
                         "XGBoost-SHAP and RF-SHAP waterfall sets. "
                         "Each pair = 1 FAIL antibody + 1 PASS antibody. "
                         "(default: 20)")
    ap.add_argument('--predict',          default=None,
                    help="Path to NEW unseen antibody file (.xlsx or .csv) with columns: "
                         "BARCODE (or ID), HSEQ, LSEQ, CDR3. "
                         "All three models predict on this set; the same interpretability "
                         "plots are produced using predicted labels (no true labels needed). "
                         "Requires --db/--target (used for model lookup only) and --db2/--target2 "
                         "for a second filter. Outputs go to --outdir/predict_{stem}/.")
    ap.add_argument('--predict2',         default=None,
                    help="Second unseen antibody file for the second filter (--target2). "
                         "If omitted, --predict is used for both filters.")
    ap.add_argument('--target2',          default=None,
                    help="Target column for second database (e.g. sec_filter)")
    ap.add_argument('--db2',              default=None,
                    help="Second database (e.g. sec/ipi_sec.xlsx). "
                         "When provided with --target2, produces a 2-row "
                         "PSR+SEC combined figure in addition to all "
                         "individual per-dataset figures.")

    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == '__main__':
    main()