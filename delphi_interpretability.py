"""
delphi_interpretability.py
─────────────────────────────────────────────────────────────────────────
DELPHI — Deep End-to-end Learning Platform for antibody Developability
         with High Interpretability.

Generates interpretability figures for antibody developability models across
three architectures:
  - Transformer + one-hot sequences   -> Integrated Gradients (IG)
  - Random Forest + biophysical       -> SHAP
  - XGBoost       + biophysical       -> SHAP

Three modes (auto-detected from arguments)
────────────────────────────────────────────
MODE A — Dual-target manuscript figures (PSR + SEC):
    python delphi_interpretability.py \\
        --target psr_filter --target2 sec_filter \\
        --db data/ipi_psr_trainset.xlsx --db2 data/ipi_sec_5000.xlsx \\
        --model-dir pretrained_202605 \\
        --ig-max-samples 0 --ig-steps 200 --n-pairs 20 \\
        --outdir outputs/interp_psr_sec

MODE B — Single-target, all three architectures on one database:
    python delphi_interpretability.py \\
        --target psr_filter --db data/ipi_psr_trainset.xlsx \\
        --model-dir pretrained_202605 \\
        --ig-max-samples 500 --n-antibodies 20 \\
        --outdir outputs/interp_psr

MODE C — Predict new antibodies with an existing model, then interpret.
    The model is located either by --db stem OR by direct --model-path.
    Both forms produce the SAME prediction-set interpretation.

    # Transformer (one-hot) - Integrated Gradients
    python delphi_interpretability.py --predict tests/DS1_5000.xlsx \\
        --target psr_filter --model transformer_onehot --lm onehot \\
        --model-path pretrained_202605/FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt \\
        --ig-max-samples 500 --n-antibodies 20 \\
        --outdir interpret_out_transformer

    # Random Forest (biophysical) - SHAP
    python delphi_interpretability.py --predict tests/DS1_5000.xlsx \\
        --target psr_filter --model rf --lm biophysical \\
        --model-path pretrained_202605/FINAL_psr_filter_biophysical_rf_ipi_psr_trainset.pkl \\
        --n-antibodies 20 \\
        --outdir interpret_out_rf

    # XGBoost (biophysical) - SHAP
    python delphi_interpretability.py --predict tests/DS1_5000.xlsx \\
        --target psr_filter --model xgboost --lm biophysical \\
        --model-path pretrained_202605/FINAL_psr_filter_biophysical_xgboost_ipi_psr_trainset.pkl \\
        --n-antibodies 20 \\
        --outdir interpret_out_xgboost

    # Equivalent lookup by --db stem instead of --model-path:
    python delphi_interpretability.py --predict tests/DS1_5000.xlsx \\
        --db data/ipi_psr_trainset.xlsx --target psr_filter \\
        --model transformer_onehot --lm onehot \\
        --model-dir pretrained_202605 \\
        --ig-max-samples 500 --n-antibodies 20 \\
        --outdir interpret_out_transformer

Per-antibody figures (Modes B and C)
────────────────────────────────────
  --n-antibodies N   N PASS + N FAIL antibodies (0 = ALL), NO score filter.
  For each selected antibody, one figure per architecture:
    - waterfall   IG (transformer) or SHAP (rf/xgboost), AA-class coloured
    - CDR3 mutagenesis heatmap  (P(PASS) per single substitution)
  Each labelled with BARCODE, actual label (PASS=1/FAIL=0 if available),
  and the DELPHI predicted score.
  For rf/xgboost the IG amino-acid heatmap panel is skipped (IG is
  transformer-only).

Output naming convention (matches delphi.py)
──────────────────────────────────────────────────────
    interp_{target}_{rf_lm}_{xgb_lm}_{tr_lm}_{db_stem}.{tiff|pdf|png}
    shap_rf_{target}_{rf_lm}_{db_stem}.csv
    shap_xgb_{target}_{xgb_lm}_{db_stem}.csv
    ig_{target}_{tr_lm}_{db_stem}.npz
    region_attribution_{target}_{db_stem}.csv
    per_antibody_{target}_{db_stem}/ab_{barcode}_{PASS|FAIL}_{model}_*.{tiff|pdf|png}
    interp_log_{target}_{db_stem}.txt

If a model is missing, the corresponding panel is rendered blank with a
note, and the script continues.
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

def _safe_cmap(name):
    """matplotlib>=3.9 removed cm.get_cmap() — use colormaps[] instead."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]   # matplotlib >= 3.7
    except (AttributeError, KeyError):
        import matplotlib.cm as _cm
        return _cm.get_cmap(name)           # matplotlib < 3.7
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
# PUBLICATION-QUALITY STYLING — matches Extended Figures 2/3/4 exactly
# ══════════════════════════════════════════════════════════════════════════════

# Single-column, 5 stacked panels. Panel E narrower than A-D.
FIG_WIDTH_IN  = 6.3
FIG_HEIGHT_IN = 11.0
DPI_PNG  = 300   # [FIX] PNG-only output at 300 DPI (tiff/pdf removed to save disk space)

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

# ── Custom diverging colormap: dark red (FAIL) ↔ white ↔ dark blue (PASS) ─
# Steeper than RdBu/seismic — more colour even at small values
import matplotlib.colors as _mcolors
_HEATMAP_CMAP = _mcolors.LinearSegmentedColormap.from_list(
    'fail_pass_heatmap',
    [
        (0.00, '#67001F'),   # dark red        — strong FAIL
        (0.15, '#B2182B'),   # red
        (0.30, '#D6604D'),   # light red
        (0.45, '#F7C4AC'),   # pale salmon
        (0.50, '#FFFFFF'),   # pure white      — neutral
        (0.55, '#AECDE1'),   # pale blue
        (0.70, '#4393C3'),   # blue
        (0.85, '#2166AC'),   # dark blue
        (1.00, '#053061'),   # very dark blue  — strong PASS
    ],
    N=512)


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
    """Simple tee to stdout + file. Identical to delphi.py style."""
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
# MODEL DISCOVERY — matches delphi.py naming exactly:
#   FINAL_{target}_{lm}_{model}_{db_stem}{_regression?}{ext}
#     ext = .pkl for rf/xgboost, .pt for transformer_onehot
# ══════════════════════════════════════════════════════════════════════════════

def _find_final(model_dir: str, target: str, lm: str, model_type: str,
                db_stem: str, ext: str) -> Optional[str]:
    """
    Find a FINAL_* checkpoint. Preference order:
      1. exact match:            FINAL_{target}_{lm}_{model}_{db_stem}{ext}
      2. classification suffix:  FINAL_{target}_{lm}_{model}_{db_stem}_classification{ext}
      3. any non-regression glob: FINAL_{target}_{lm}_{model}_{db_stem}*{ext}
                                   (files containing '_regression' are excluded)
      4. None
    """
    stem = f"FINAL_{target}_{lm}_{model_type}_{db_stem}"
    # Try exact match, then explicit classification suffix
    for candidate in [
        os.path.join(model_dir, f"{stem}{ext}"),
        os.path.join(model_dir, f"{stem}_classification{ext}"),
    ]:
        if os.path.exists(candidate):
            return candidate

    # Glob fallback — prefer classification, skip regression entirely
    pattern = os.path.join(model_dir, f"{stem}*{ext}")
    found   = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    # Filter out regression models
    non_reg = [f for f in found if '_regression' not in os.path.basename(f)]
    if non_reg:
        return non_reg[0]
    # Last resort: warn and return None (don't silently use regression for classification)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — keep it minimal & aligned with delphi.py
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
    all_barcodes = list(X_df.index.astype(str))
    if max_samples > 0 and len(X_ne) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_ne), max_samples, replace=False)
        X_shap    = X_ne[idx]
        barcodes  = [all_barcodes[i] for i in idx]
        log(f"[SHAP] subsampled {max_samples:,} of {len(X_ne):,} rows")
    else:
        X_shap   = X_ne
        barcodes = all_barcodes

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
        'barcodes':      barcodes,
        'expected':      expected,
    }


# ══════════════════════════════════════════════════════════════════════════════
# IG COMPUTATION — uses transformer_onehot.TransformerOneHotModel.global_ig_analysis
# but we re-implement the accumulation here so we can save arrays.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ig(model, df: pd.DataFrame, n_samples: int, n_steps: int,
                log: _Log, ig_baseline: str = 'uniform') -> Optional[dict]:
    """
    Compute Integrated Gradients on the trained Transformer-onehot model.

    ig_baseline : 'uniform' (default) — 1/20 at each observed residue position,
                                        zero at true padding positions
                  'zero'              — legacy all-zero reference (unsafe for
                                        models that infer masks from zero rows)
                  'mean'              — average one-hot encoding across all antibodies
                                     suppresses attribution at conserved positions (e.g. CAR)

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
        from models.transformer_onehot import AntibodyDataset, length_matched_uniform_baseline
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
    log(f"[IG] using {len(df):,} antibodies  n_steps={n_steps}  "
        f"vh_only={vh_only}  baseline={ig_baseline}")

    heavy = df['HSEQ'].tolist()
    light = [''] * len(df) if vh_only else df['LSEQ'].tolist()
    hcdr3 = df['CDR3'].tolist()
    labels   = np.zeros(len(df), dtype=np.int64)
    barcodes = df.index.astype(str).tolist()

    ds = AntibodyDataset(heavy, light, hcdr3, labels, barcodes,
                         model.max_heavy_len, model.max_light_len,
                         model.max_hcdr3_len, vh_only=vh_only)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    fb_model.eval()
    ig = IntegratedGradients(fb_model)

    # ── Pre-compute mean baseline if requested ────────────────────────────
    mean_base_enc  = None
    mean_base_cdr3 = None
    if ig_baseline == 'mean':
        log("[IG] Computing mean one-hot baseline across all antibodies ...")
        import torch
        all_enc  = []
        all_cdr3 = []
        for enc, cdr3_enc, *_ in loader:
            all_enc.append(enc.float())
            all_cdr3.append(cdr3_enc.float())
        mean_base_enc  = torch.cat(all_enc,  dim=0).mean(dim=0, keepdim=True)  # (1, L1, 20)
        mean_base_cdr3 = torch.cat(all_cdr3, dim=0).mean(dim=0, keepdim=True)  # (1, 25, 20)
        log(f"[IG] Mean baseline shape: enc={mean_base_enc.shape}  "
            f"cdr3={mean_base_cdr3.shape}")

    attrs_enc  = []
    attrs_cdr3 = []
    probs_all  = []
    deltas_all = []

    for enc, cdr3_enc, lbl, *_ in loader:
        enc      = enc.to(model.device)
        cdr3_enc = cdr3_enc.to(model.device)

        if ig_baseline == 'mean' and mean_base_enc is not None:
            # Broadcast mean baseline to batch size
            base_enc  = mean_base_enc.expand_as(enc).to(model.device)
            base_cdr3 = mean_base_cdr3.expand_as(cdr3_enc).to(model.device)
        elif ig_baseline == 'zero':
            # Retained only for explicit legacy comparisons. An all-zero
            # sequence can be interpreted as fully padded by this architecture.
            base_enc  = torch.zeros_like(enc)
            base_cdr3 = torch.zeros_like(cdr3_enc)
        else:
            base_enc, base_cdr3 = length_matched_uniform_baseline(enc, cdr3_enc)

        with torch.no_grad():
            base_logits = fb_model(base_enc, base_cdr3)
        if not torch.isfinite(base_logits).all():
            raise RuntimeError(
                f"IG baseline '{ig_baseline}' produced non-finite model output; "
                "use the padding-safe uniform baseline")

        attr, delta = ig.attribute(
            (enc, cdr3_enc),
            baselines=(base_enc, base_cdr3),
            target=1, n_steps=n_steps,
            internal_batch_size=max(enc.shape[0] * 4, enc.shape[0]),
            return_convergence_delta=True,
        )
        if not torch.isfinite(delta).all():
            raise RuntimeError("Non-finite Integrated Gradients convergence delta")
        attrs_enc.append(attr[0].detach().cpu().numpy())
        attrs_cdr3.append(attr[1].detach().cpu().numpy())
        deltas_all.append(delta.detach().cpu().numpy())

        with torch.no_grad():
            logits = fb_model(enc, cdr3_enc)
            p1 = torch.softmax(logits, dim=1)[:, 1]
        probs_all.extend(p1.cpu().numpy().tolist())

    attr_enc  = np.concatenate(attrs_enc,  axis=0)   # (n, L1, 20)
    attr_cdr3 = np.concatenate(attrs_cdr3, axis=0)   # (n, 25, 20)
    probs     = np.asarray(probs_all, dtype=np.float64)
    deltas    = np.concatenate(deltas_all, axis=0).astype(np.float64)

    log(f"[IG] done  attr_enc={attr_enc.shape}  attr_cdr3={attr_cdr3.shape}")
    abs_delta = np.abs(deltas)
    log(f"[IG] convergence |delta|: median={np.median(abs_delta):.3g}  "
        f"p95={np.quantile(abs_delta, 0.95):.3g}  max={np.max(abs_delta):.3g}")

    return {
        'attr_enc':   attr_enc,
        'attr_cdr3':  attr_cdr3,
        'hcdr3_seqs': hcdr3,
        'barcodes':   barcodes,
        'probs':      probs,
        'convergence_delta': deltas,
        'baseline':   ig_baseline,
        'n_steps':    n_steps,
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
            transform=ax.transAxes, fontsize=9.5, color='#000000',
            style='italic', wrap=True)
    ax.set_title(title, fontsize=10.8, loc='left', fontweight='bold', pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _panel_shap_bar(ax, shap_data: Optional[dict], title: str,
                    top_n: int = 20, kmer_source: str = 'CDR3'):
    """Horizontal bar — top-N mean |SHAP|, colour-coded by region."""
    if shap_data is None:
        _render_blank(ax,
                      "Model not found — rerun delphi.py\n"
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
    ax.set_yticklabels(pretty, fontsize=10.8, color='#000000')
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|', fontsize=12.2, color='#000000')
    ax.set_title(title, fontsize=13.5, loc='left', fontweight='bold', pad=4)
    ax.tick_params(axis='x', labelsize=7, colors='#000000')
    ax.grid(axis='x', alpha=0.25, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    present = []
    for r in ['HCDR3', 'VH', 'VL', 'VH+VL']:
        if r in regions:
            present.append(mpatches.Patch(color=region_color[r], label=r))
    if present:
        ax.legend(handles=present, loc='lower right', fontsize=10.8,
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
                      "Model not found — rerun delphi.py\n"
                      "    --train --lm biophysical  to generate.",
                      title)
        return

    # Defensive: beeswarm needs full SHAP + feature-value matrices.
    # If only mean_abs_shap was loaded (e.g. partial CSV), fall back to a note.
    if ('shap_matrix' not in shap_data or shap_data.get('shap_matrix') is None
            or 'X_matrix' not in shap_data or shap_data.get('X_matrix') is None):
        _render_blank(ax,
                      "SHAP value matrix not available for beeswarm.\n"
                      "(Only mean |SHAP| was loaded — re-run without --csv-exist\n"
                      " to regenerate the full SHAP matrix.)",
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
    dot_s   = 7      # point size — larger for readability
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
    ax.axvline(0, color='#000000', lw=0.5, ls='--', zorder=1)

    # Axis labels and direction annotation
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(pretty, fontsize=10.8, color='#000000')
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value  (← toward FAIL  |  toward PASS →)', fontsize=12.2, color='#000000')
    ax.set_title(title, fontsize=13.5, loc='left', fontweight='bold', pad=4)
    ax.tick_params(axis='x', labelsize=10.8, colors='#000000')
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
    cbar.set_ticklabels(['Low', 'High'], fontsize=9.5, color='#000000')
    cbar.set_label('Feature value', fontsize=10.8, labelpad=2)
    cbar.ax.tick_params(width=0.4, length=1.5)


def _panel_ig_positions(ax, ig_data: Optional[dict], title: str):
    """Per-position mean |IG| across VH / VL / HCDR3."""
    if ig_data is None:
        _render_blank(ax,
                      "Transformer (onehot) model not found.\n"
                      "rerun delphi.py --train --lm onehot\n"
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

    # Compute separate maxima for framework vs HCDR3
    vh_max   = float(vh_y.max())
    vl_max   = float(vl_y.max() if vl_y is not None else 0.0)
    cdr3_max = float(pos_cdr3.max())
    fw_max   = max(vh_max, vl_max)   # framework max
    ymax_all = max(fw_max, cdr3_max)

    # If HCDR3 dominates (>3× framework), clip y-axis to show VH/VL clearly
    # and annotate the HCDR3 peak with its actual value
    if cdr3_max > 3 * fw_max and fw_max > 0:
        y_clip = fw_max * 2.2   # show VH/VL with headroom
        ax.set_ylim(0, y_clip * 1.55)

        # Annotate HCDR3 peak with arrow + value (peak is clipped)
        peak_x = float(cdr3_x[np.argmax(pos_cdr3)])
        ax.annotate(
            f'HCDR3 peak\n{cdr3_max:.2f}',
            xy=(peak_x, y_clip), xytext=(peak_x - 15, y_clip * 1.15),
            fontsize=9.5, color=COLOR_CDR3, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_CDR3, lw=0.8),
            ha='center', va='bottom')
        # Draw a clipping indicator (zigzag) on top of HCDR3 fill at clip line
        ax.axhline(y_clip, xmin=(cdr3_x[0] - x_vh[0]) / (cdr3_x[-1] - x_vh[0] + 10),
                   xmax=1.0, color=COLOR_CDR3, lw=0.8, ls='--', alpha=0.5)
    else:
        ax.set_ylim(0, ymax_all * 1.58)

    ymax = ax.get_ylim()[1]

    # Region boundaries
    for xr in (x_vh[-1] + 0.5,
               (vl_x[-1] + 0.5 if vl_x is not None else x_vh[-1] + 4),
               cdr3_x[0] - 0.5):
        ax.axvline(xr, color='#000000', lw=0.3, ls=':')

    # Region label strips above the plot
    mid_vh   = (x_vh[0] + x_vh[-1]) / 2
    mid_cdr3 = (cdr3_x[0] + cdr3_x[-1]) / 2
    label_y  = ymax * 0.88
    ax.text(mid_vh,   label_y, 'VH framework',
            ha='center', va='bottom', fontsize=12.2, color=COLOR_VH_FR,
            fontweight='bold')
    if vl_x is not None:
        mid_vl = (vl_x[0] + vl_x[-1]) / 2
        ax.text(mid_vl, label_y, 'VL framework',
                ha='center', va='bottom', fontsize=12.2, color=COLOR_VL_FR,
                fontweight='bold')
    ax.text(mid_cdr3, label_y, 'HCDR3',
            ha='center', va='bottom', fontsize=12.2, color=COLOR_CDR3,
            fontweight='bold')
    ax.tick_params(axis='both', labelsize=7, colors='#000000')
    ax.grid(axis='y', alpha=0.25, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _panel_ig_cdr3_heatmap(ax, ig_data: Optional[dict], title: str):
    """HCDR3 AA × position signed IG heatmap."""
    if ig_data is None:
        _render_blank(ax,
                      "Transformer (onehot) model not found.\n"
                      "rerun delphi.py --train --lm onehot\n"
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
    im = ax.imshow(mat_show, cmap=_HEATMAP_CMAP, vmin=-vmax * 0.85, vmax=vmax * 0.85,
                   aspect='auto', interpolation='nearest')

    ax.set_yticks(range(20))
    ax.set_yticklabels(list(AMINO_ACIDS), fontsize=11.5, color='#000000',
                       fontfamily='monospace', fontweight='bold')
    ax.set_xticks(range(show_cols))
    _nc = mat_show.shape[1]
    ax.set_xticks(range(_nc))
    ax.set_xticklabels([str(i + 1) for i in range(_nc)], fontsize=10.8,
                       color='#000000')
    ax.set_xlabel('HCDR3 position', fontsize=12.2, color='#000000')
    ax.set_ylabel('Amino acid', fontsize=12.2, color='#000000')
    ax.set_title(title, fontsize=13.5, loc='left', fontweight='bold', pad=4,
                 color='#000000')
    ax.tick_params(axis='both', colors='#000000', length=2, width=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#000000')

    cbar = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label('Mean signed IG', fontsize=12.2, color='#000000')
    cbar.ax.tick_params(labelsize=10.8, colors='#000000')


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
                        ha='center', va='bottom', fontsize=10.8, color='#000000')

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=12.2, color='#000000')
    ax.set_ylabel('% of |attribution| mass', fontsize=12.2, color='#000000')
    ax.set_title(title, fontsize=13.5, loc='left', fontweight='bold', pad=4)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=10.8, frameon=False,
              handlelength=1.0, handleheight=0.8, ncol=3,
              columnspacing=0.8)
    ax.tick_params(axis='both', labelsize=7, colors='#000000')
    ax.grid(axis='y', alpha=0.25, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Caption below
    ax.text(0.5, -0.22,
            "Proportion of total absolute attribution mass assigned to each region "
            "by each model. All three converge on HCDR3 despite operating in "
            "different feature spaces (biophysical vs. one-hot).",
            transform=ax.transAxes, ha='center', va='top', fontsize=9.5,
            color='#000000', style='italic', wrap=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def _make_fig_scaffold(target, db_stem):
    """Shared scaffold for both figure variants — manuscript-quality layout."""
    # Taller figure (was 14.0) so panel-d heatmap rows are readable and the
    # suptitle has clearance above panel-a (was overlapping panel-a title).
    fig = plt.figure(figsize=(9.0, 16.0),
                     constrained_layout=False)
    # Panel-d (heatmap) height raised 1.15 → 1.70 so amino-acid letters are legible.
    gs = gridspec.GridSpec(5, 1, figure=fig,
                           height_ratios=[1.3, 1.3, 0.70, 1.70, 0.75],
                           left=0.22, right=0.97,
                           top=0.935, bottom=0.045,   # top lowered 0.965 → 0.935
                           hspace=0.45)   # more gap between panels
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    fig.suptitle(
        f"Interpretability convergence — {target}  ·  {db_stem}",
        fontsize=14.9, fontweight='bold', color='#000000', y=0.975)
    return fig, axes


def _save_fig(fig, out_stem: str, log: _Log):
    # [FIX] PNG-only output (300 DPI) — tiff and pdf removed to save disk space.
    # Replace with: for ext, dpi in [('tiff', 300), ('pdf', 300), ('png', 300)]
    # if you need publication-quality tiff/pdf for a specific figure.
    path = f"{out_stem}.png"
    fig.savefig(path, dpi=DPI_PNG, bbox_inches='tight')
    log(f"[figure] {path}  ({DPI_PNG} DPI)")
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
                    top_n=15, kmer_source=rf_lm)
    _panel_shap_bar(ax_b, xgb_shap,
                    f"b   XGBoost  ·  SHAP top features  ({xgb_lm})",
                    top_n=15, kmer_source=xgb_lm)
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
                    top_n=15, kmer_source=rf_lm)
    _panel_shap_beeswarm(ax_b, xgb_shap,
                    f"b   XGBoost  ·  SHAP beeswarm  ({xgb_lm})",
                    top_n=15, kmer_source=xgb_lm)
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
            label  = f"{aa}"
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
            hcdr3_rows.append((ma, f"{aa}", ig_vec, aa))
        hcdr3_rows.sort(key=lambda x: x[0], reverse=True)
        groups.append(('HCDR3', hcdr3_rows))

        # ── VH framework ──────────────────────────────────────────────────
        if attr_enc.shape[1] >= max_vh:
            vh_rows = []
            for aa_i, aa in enumerate(AMINO_ACIDS):
                ig_vec = attr_enc[:, :max_vh, aa_i].sum(axis=1)
                ma     = float(np.abs(ig_vec).mean())
                vh_rows.append((ma, f"{aa}", ig_vec, aa))
            vh_rows.sort(key=lambda x: x[0], reverse=True)
            groups.append(('VH framework', vh_rows))

        # ── VL framework ──────────────────────────────────────────────────
        if not vh_only and attr_enc.shape[1] > max_vh:
            vl_rows = []
            for aa_i, aa in enumerate(AMINO_ACIDS):
                ig_vec = attr_enc[:, max_vh:max_vh+max_vl, aa_i].sum(axis=1)
                ma     = float(np.abs(ig_vec).mean())
                vl_rows.append((ma, f"{aa}", ig_vec, aa))
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
    ax.axvline(0, color='#000000', lw=0.8, ls='-', zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=fsz_y,
                       fontfamily='monospace' if monospace else 'sans-serif', color='#000000')
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=fsz_y - 1, colors='#000000')
    ax.grid(axis='x', alpha=0.20, lw=0.3)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fsz_y + 1, labelpad=2, color='#000000')

def _ig_legend(ax, fontsize=8.8, loc='lower right', outside=False):
    """4-class AA physicochemical legend.
    outside=True places the legend below the x-axis — avoids covering data.
    """
    kwargs = dict(handles=_aa_class_legend(), fontsize=fontsize,
                  frameon=True, framealpha=0.92, edgecolor='#000000',
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
                         f"{aa}", ig_vec, aa))
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
        _ig_ax_style(ax, rows, fsz_y=9)
        ax.set_xlim(xlim)
        ax.set_title(f"  {region_name}", fontsize=12.2, loc='left',
                     fontweight='bold', pad=4, color='#000000')
        _ig_legend(ax, fontsize=6.0)

    fig.suptitle(title, fontsize=12.2, y=1.005)
    for ext, dpi in [('png', DPI_PNG)]:
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
        ax.set_title(fig_title, fontsize=12.2, pad=8)
        _ig_legend(ax, fontsize=6.0)
        plt.tight_layout()
        for ext, dpi in [('png', DPI_PNG)]:
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
        elif current in ('VH framework', 'VH'):
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

    GAP = 0.8   # small gap between HCDR3 and VH rows

    if ig_data is None:
        ax.text(0.5, 0.5, 'Transformer model not found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9.5, color='#000000', style='italic')
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
        jitter = rng.uniform(-0.28, 0.28, size=len(ig_vec))
        col = _aa_color(aa_id) if isinstance(aa_id, str) else _AA_CLASS_COLORS[int(aa_id) % 4]
        ax.scatter(ig_vec, _y(row_i, False) + jitter,
                   c=col, s=dot_size, alpha=0.65, linewidths=0,
                   rasterized=True, zorder=2)

    for row_i, (_, label, ig_vec, aa_id) in enumerate(vh_rows):
        jitter = rng.uniform(-0.28, 0.28, size=len(ig_vec))
        col = _aa_color(aa_id) if isinstance(aa_id, str) else _AA_CLASS_COLORS[int(aa_id) % 4]
        ax.scatter(ig_vec, _y(row_i, True) + jitter,
                   c=col, s=dot_size, alpha=0.65, linewidths=0,
                   rasterized=True, zorder=2)

    # ── Y-axis ticks — prefix each label with region ─────────────────────
    yticks  = ([_y(i, False) for i in range(n_cdr3)] +
               [_y(i, True)  for i in range(n_vh)])
    ylabels = ([f"HCDR3  {r[1]}" for r in hcdr3_rows] +
               [f"VH  {r[1]}" for r in vh_rows])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=fsz_y, fontfamily='monospace', color='#000000')

    # Colour HCDR3 labels vs VH labels differently
    for tick, label in zip(ax.yaxis.get_major_ticks(), ylabels):
        tick.label1.set_color(COLOR_CDR3 if label.startswith('HCDR3') else COLOR_VH_FR)

    # ── Axis styling ──────────────────────────────────────────────────────
    ax.axvline(0, color='#000000', lw=0.6, ls='-', zorder=1)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=max(4.5, fsz_y - 1.5), colors='#000000')
    ax.grid(axis='x', alpha=0.18, lw=0.3)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('IG value\n(← FAIL  |  PASS →)', fontsize=8.1, labelpad=3)

    # ── Physicochemical class legend — compact ───────────────────────────
    _short_labels = [
        'Cationic (R,K,H)',
        'Anionic (D,E)',
        'Hydrophobic/Aromatic (W,F,Y,L,I,V,M)',
        'Small/Polar (A,G,S,T,C,P,Q,N)',
    ]
    handles = [mpatches.Patch(facecolor=c, edgecolor='none', label=l)
               for c, l in zip(_AA_CLASS_COLORS, _short_labels)]
    ax.legend(handles=handles,
              title='AA class',
              title_fontsize=6.5,
              fontsize=6.0,
              loc='lower left',
              bbox_to_anchor=(1.02, 0.0),
              frameon=True, framealpha=0.92, edgecolor='#000000',
              handlelength=0.7, handleheight=0.55,
              borderpad=0.3, labelspacing=0.15, handletextpad=0.4)


def build_figure_3beeswarms(rf_shap, xgb_shap, ig_data,
                             target: str, db_stem: str,
                             rf_lm: str, xgb_lm: str, tr_lm: str,
                             out_stem: str, log: _Log,
                             top_n: int = 20,
                             fig_width_cm: float = 18.0,
                             font_scale: float = 1.0):
    """
    3-column beeswarm figure — NB Extended Data style.
    fig_width_cm : total figure width in cm (18 = double-column)
    font_scale   : multiply all font sizes (1.0 = default, 1.2 = larger)
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    import matplotlib.transforms as _mt
    from matplotlib.gridspec import GridSpec

    # ── Font sizes — all scale together ──────────────────────────────────
    FS = font_scale
    YTICK_FSZ  = 8.0  * FS   # y-tick feature labels
    XTICK_FSZ  = 7.5  * FS   # x-axis tick numbers
    XLABEL_FSZ = 8.0  * FS   # x-axis label text
    TITLE_FSZ1 = 9.5  * FS   # panel letter + model name
    TITLE_FSZ2 = 7.0  * FS   # subtitle line
    CB_FSZ     = 8.5  * FS   # colourbar labels
    DOT_SIZE   = 6    * FS   # scatter dot size
    VH_TOP_N   = 7

    # ── Figure size — width from parameter, height from content ──────────
    FIG_W  = fig_width_cm / 2.54
    ig_total_rows = 20 + VH_TOP_N + 2
    row_h  = 0.22 * FS   # inches per feature row — tighter spacing
    FIG_H  = max(6.0, max(top_n, ig_total_rows) * row_h + 2.0)

    gs  = GridSpec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=1.20)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.09)

    ax_rf  = fig.add_subplot(gs[0, 0])
    ax_xgb = fig.add_subplot(gs[0, 1])
    ax_ig  = fig.add_subplot(gs[0, 2])

    cmap_feat = _safe_cmap('RdBu_r')
    rng = np.random.default_rng(42)

    def _pretty(l):
        return l.replace('cdr3_charge_ph7', 'cdr3_charge').replace('_ph7', '')

    # ── CSV export helpers ────────────────────────────────────────────────
    def _save_shap_beeswarm_csv(shap_data, feat_order, feat_labels, path, model_name):
        if shap_data is None: return
        sv  = shap_data['shap_matrix']
        xv  = shap_data['X_matrix']
        names = shap_data['names']
        rows = []
        for rank, fi in enumerate(feat_order):
            sv_col = sv[:, fi]
            xv_col = xv[:, fi]
            for j in range(len(sv_col)):
                rows.append({
                    'model':       model_name,
                    'rank':        rank + 1,
                    'feature':     _pretty(names[fi]),
                    'shap_value':  float(sv_col[j]),
                    'feature_value': float(xv_col[j]),
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        log(f"[CSV] beeswarm {model_name} → {Path(path).name}  ({len(rows):,} rows)")

    def _save_ig_beeswarm_csv(ig_data, path, model_name, vh_top_n):
        if ig_data is None: return
        AA_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        barcodes = ig_data['barcodes']
        attr_c   = ig_data['attr_cdr3']   # (n, max_cdr3, 20)
        attr_e   = ig_data['attr_enc']    # (n, max_vh, 20)
        max_cdr3 = ig_data['max_cdr3']
        max_vh   = ig_data['max_vh']
        rows = []
        # CDR3 — all 20 AAs
        for aa_i, aa in enumerate(AMINO_ACIDS):
            mean_ig = float(attr_c[:, :max_cdr3, aa_i].mean())
            for j in range(attr_c.shape[0]):
                ig_vals = attr_c[j, :max_cdr3, aa_i]
                rows.append({
                    'model':    model_name,
                    'region':   'HCDR3',
                    'aa':       aa,
                    'ig_value': float(ig_vals.sum()),    # sum over positions — matches 2-row figure
                    'barcode':  barcodes[j],
                })
        # VH top-N
        pos_enc = np.abs(attr_e).sum(axis=-1).mean(axis=0)
        top_vh  = np.argsort(pos_enc)[::-1][:vh_top_n]
        for aa_i, aa in enumerate(AMINO_ACIDS):
            for j in range(attr_e.shape[0]):
                ig_vals = attr_e[j, top_vh, aa_i]
                rows.append({
                    'model':    model_name,
                    'region':   'VH_framework',
                    'aa':       aa,
                    'ig_value': float(ig_vals.sum()),
                    'barcode':  barcodes[j],
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        log(f"[CSV] beeswarm {model_name} → {Path(path).name}  ({len(rows):,} rows)")

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
            jitter  = rng.uniform(-0.28, 0.28, size=len(sv_col))
            ax_rf.scatter(sv_col, row + jitter,
                          c=cmap_feat(norm_xv), s=DOT_SIZE, alpha=0.6,
                          linewidths=0, rasterized=True)

        # Export CSV
        _save_shap_beeswarm_csv(rf_shap, rf_order, rf_labels,
                                 f"{out_stem}_beeswarm_RF_{target}.csv",
                                 f"RF-{rf_lm}")

        ax_rf.axvline(0, color='#000000', lw=0.7, ls='-')
        ax_rf.set_yticks(range(top_n))
        ax_rf.set_yticklabels(rf_labels, fontsize=YTICK_FSZ, color='#000000')
        ax_rf.invert_yaxis()
        ax_rf.tick_params(axis='x', labelsize=XTICK_FSZ, colors='#000000')
        ax_rf.grid(axis='x', alpha=0.18, lw=0.3)
        for s in ('top', 'right'): ax_rf.spines[s].set_visible(False)

    ax_rf.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                     fontsize=XLABEL_FSZ, labelpad=3, color='#000000')

    # Feature-value colorbar — shared vertical bar between col a and col b
    # (drawn after canvas.draw() so positions are final)
    import matplotlib.colors as _mc2

    # ── Panel B : XGBoost ─────────────────────────────────────────────────
    if xgb_shap is None or rf_shap is None:
        _render_blank(ax_xgb, "XGBoost model not found", "")
    else:
        xgb_name_to_idx = {n: i for i, n in enumerate(xgb_shap['names'])}
        sv_xgb = xgb_shap['shap_matrix']
        Xm_xgb = xgb_shap['X_matrix']
        xgb_order_mapped = []

        for row, rf_feat_idx in enumerate(rf_order):
            feat_name = rf_shap['names'][rf_feat_idx]
            xgb_idx   = xgb_name_to_idx.get(feat_name)
            if xgb_idx is None:
                continue
            xgb_order_mapped.append(xgb_idx)
            sv_col  = sv_xgb[:, xgb_idx]
            xv_col  = Xm_xgb[:, xgb_idx]
            lo, hi  = xv_col.min(), xv_col.max()
            norm_xv = (xv_col - lo) / (hi - lo + 1e-10)
            jitter  = rng.uniform(-0.28, 0.28, size=len(sv_col))
            ax_xgb.scatter(sv_col, row + jitter,
                           c=cmap_feat(norm_xv), s=DOT_SIZE, alpha=0.6,
                           linewidths=0, rasterized=True)

        # Export CSV (same row order as RF)
        _save_shap_beeswarm_csv(xgb_shap, xgb_order_mapped,
                                 [rf_labels[i] for i in range(len(xgb_order_mapped))],
                                 f"{out_stem}_beeswarm_XGBoost_{target}.csv",
                                 f"XGBoost-{xgb_lm}")

        ax_xgb.axvline(0, color='#000000', lw=0.7, ls='-')
        ax_xgb.tick_params(axis='x', labelsize=XTICK_FSZ, colors='#000000')
        ax_xgb.grid(axis='x', alpha=0.18, lw=0.3)
        for s in ('top', 'right'): ax_xgb.spines[s].set_visible(False)

    n_rf_rows = len(rf_labels) if rf_shap is not None else top_n
    ax_xgb.set_yticks(range(n_rf_rows))
    ax_xgb.set_yticklabels(
        ['' for _ in range(n_rf_rows)],   # labels shown in RF panel only
        fontsize=YTICK_FSZ, color='#000000')
    ax_xgb.invert_yaxis()
    ax_xgb.tick_params(axis='x', colors='#000000')
    ax_xgb.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                      fontsize=XLABEL_FSZ, labelpad=3, color='#000000')

    # ── Panel C : Transformer IG ──────────────────────────────────────────
    _render_ig_combined(ax_ig, ig_data, rng,
                        vh_top_n=VH_TOP_N, fsz_y=YTICK_FSZ, dot_size=DOT_SIZE)
    ax_ig.set_xlabel('IG value\n(← FAIL  |  PASS →)', fontsize=XLABEL_FSZ, labelpad=3, color='#000000')
    ax_ig.tick_params(axis='x', labelsize=XTICK_FSZ, colors='#000000')

    # Export IG CSV
    _save_ig_beeswarm_csv(ig_data,
                           f"{out_stem}_beeswarm_Transformer_{target}.csv",
                           f"Transformer-{tr_lm}", VH_TOP_N)

    # ── Panel titles ──────────────────────────────────────────────────────
    _filter = target.upper().split('_')[0]
    fig.canvas.draw()
    TITLE_LIFT = 0.052

    # ── Shared vertical Feature Value colorbar between panels a and b ────
    _sm_feat = plt.cm.ScalarMappable(cmap=cmap_feat, norm=_mc2.Normalize(0, 1))
    _sm_feat.set_array([])
    _pos_a   = ax_rf.get_position()
    _pos_b   = ax_xgb.get_position()
    _gap_x   = (_pos_a.x1 + _pos_b.x0) / 2
    _cb_w    = 0.008
    _cb_h    = (_pos_a.y1 - _pos_a.y0) * 0.50
    _cb_y    = _pos_a.y0 + (_pos_a.y1 - _pos_a.y0) * 0.25
    _cax_ab  = fig.add_axes([_gap_x - _cb_w / 2, _cb_y, _cb_w, _cb_h])
    _cb_ab   = fig.colorbar(_sm_feat, cax=_cax_ab, orientation='vertical')
    _cb_ab.set_ticks([0, 1])
    _cb_ab.set_ticklabels(['Low', 'High'], fontsize=CB_FSZ, color='#000000')
    _cb_ab.set_label('Feature\nvalue', fontsize=CB_FSZ, labelpad=3, color='#000000')
    _cb_ab.ax.tick_params(width=0.5, length=2, colors='#000000')
    for ax, letter, line1, line2 in [
        (ax_rf,  'a', f'RF-{_filter}',                          '(biophysical features)'),
        (ax_xgb, 'b', f'XGBoost-{_filter}',                     ''),
        (ax_ig,  'c', f'Transformer onehot-{_filter}',
                      'IG per amino acid'),
    ]:
        pos = ax.get_position()
        x0, y1 = pos.x0, pos.y1
        ty = y1 + TITLE_LIFT
        fig.text(x0, ty, letter,
                 fontsize=TITLE_FSZ1, fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        fig.text(x0 + 0.022, ty, line1,
                 fontsize=TITLE_FSZ1 * 0.85, fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        if line2:
            fig.text(x0 + 0.022, ty - 0.011, line2,
                     fontsize=TITLE_FSZ2, va='top', ha='left', color='#000000',
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
    cmap   = _safe_cmap('RdBu_r')   # matches shap.summary_plot default
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

    ax.axvline(0, color='#000000', lw=0.8, ls='-', zorder=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(pretty, fontsize=9.5, color='#000000')
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value  (← toward FAIL  |  toward PASS →)', fontsize=10.8)
    ax.set_title(title, fontsize=12.2, pad=8)
    ax.tick_params(axis='x', labelsize=9.5, colors='#000000')
    ax.grid(axis='x', alpha=0.20, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Colourbar — matches shap.summary_plot style
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=_mc.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.018, aspect=40)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=10.8, color='#000000')
    cbar.set_label('Feature value', fontsize=10.8, labelpad=4, color='#000000')
    cbar.ax.tick_params(width=0.5, length=2)

    plt.tight_layout()
    for ext, dpi in [('png', DPI_PNG)]:
        path = f"{out_stem}.{ext}"
        save_kw = dict(dpi=dpi, bbox_inches='tight')
        # tiff compression removed (PNG-only output)
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
    cmap = _safe_cmap('RdBu_r')
    norm = _mc.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    rng  = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 0.38)))

    for row_idx, (mean_abs, label, ig_vec, charge_vec) in enumerate(rows):
        n      = len(ig_vec)
        jitter = rng.uniform(-0.38, 0.38, size=n)
        ax.scatter(ig_vec, row_idx + jitter,
                   c=cmap(norm(charge_vec)), s=12, alpha=0.65,
                   linewidths=0, rasterized=True, zorder=2)

    ax.axvline(0, color='#000000', lw=0.8, ls='-', zorder=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([r[1] for r in rows], fontsize=9.5, color='#000000')
    ax.invert_yaxis()
    ax.set_xlabel('IG value  (← toward FAIL  |  toward PASS →)', fontsize=10.8)
    ax.set_title(title, fontsize=12.2, pad=8)
    ax.tick_params(axis='x', labelsize=9.5, colors='#000000')
    ax.grid(axis='x', alpha=0.20, lw=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Colourbar: red=cationic, blue=anionic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.018, aspect=40)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Anionic\n(D, E)', 'Neutral', 'Cationic\n(R, K, H)'],
                        fontsize=10.8, color='#000000')
    cbar.set_label('AA charge at position', fontsize=10.8, labelpad=4, color='#000000')
    cbar.ax.tick_params(width=0.5, length=2)

    plt.tight_layout()
    for ext, dpi in [('png', DPI_PNG)]:
        path = f"{out_stem}.{ext}"
        save_kw = dict(dpi=dpi, bbox_inches='tight')
        # tiff compression removed (PNG-only output)
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

    # ── Plot: single diverging panel (PASS right, FAIL left) ─────────────
    fig_h  = max(5, top_n * 0.28)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.91, bottom=0.07)

    n_pass = len(pass_idx)
    n_fail = len(fail_idx)

    mp = mean_pass[top_idx]   # (+) means drives PASS
    mf = mean_fail[top_idx]   # signed from FAIL antibodies — negate so FAIL bars go left

    y  = np.arange(len(top_idx))
    bh = 0.38   # bar half-height

    # PASS bars (right side) — colour by sign
    for i, v in enumerate(mp):
        ax.barh(y[i] - bh/2, v, height=bh,
                color='#1F77B4' if v >= 0 else '#D62728',
                edgecolor='white', linewidth=0.2, zorder=2, label='_')

    # FAIL bars (left side, negated so they extend leftward)
    for i, v in enumerate(mf):
        ax.barh(y[i] + bh/2, v, height=bh,
                color='#D62728' if v <= 0 else '#1F77B4',
                edgecolor='white', linewidth=0.2, zorder=2, label='_',
                alpha=0.75)

    ax.axvline(0, color='#000000', lw=0.9, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top_labels, fontsize=8.8, fontfamily='monospace', color='#000000')
    ax.invert_yaxis()
    ax.set_xlabel('Mean signed IG attribution  (← FAIL  |  PASS →)', fontsize=9.5)
    ax.tick_params(axis='x', labelsize=8.1, colors='#000000')
    ax.grid(axis='x', alpha=0.18, lw=0.3, zorder=0)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

    # Bracket annotations above axis
    ax.text(0.02, 1.01, f'PASS (n={n_pass:,})',
            transform=ax.transAxes, fontsize=12.2, fontweight='bold',
            color='#1F77B4', ha='left', va='bottom')
    ax.text(0.98, 1.01, f'FAIL (n={n_fail:,})',
            transform=ax.transAxes, fontsize=12.2, fontweight='bold',
            color='#D62728', ha='right', va='bottom')

    from matplotlib.patches import Patch as _P
    ax.legend(
        handles=[_P(facecolor='#1F77B4', label=f'PASS (n={n_pass:,})  — upper bar'),
                 _P(facecolor='#D62728', label=f'FAIL (n={n_fail:,})  — lower bar')],
        fontsize=12.2, loc='lower right', frameon=True, framealpha=0.9)

    fig.suptitle(
        f"IG attribution by outcome — {target}  ·  {db_stem}\n"
        f"Mean signed IG  PASS (upper) vs FAIL (lower)  "
        f"(top {top_n} positions by mean |IG|)",
        fontsize=13.5, y=0.98)

    for ext, dpi in [('png', DPI_PNG)]:
        path = f"{out_stem}_pass_fail_ig.{ext}"
        kw = dict(dpi=dpi, bbox_inches='tight')
        if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kw)
        log(f"[pass_fail_ig] {path}  ({dpi} DPI)")
    plt.close(fig)

def _export_full_shap_csv(shap_data: dict, df: pd.DataFrame, target: str,
                           proba_dict: dict, out_path: Path, log: _Log,
                           model_name: str = '') -> None:
    """
    Save full SHAP matrix as CSV.
    Columns: barcode, true_label, pred_prob,
             shap_{feat}, ..., fval_{feat}, ...
    """
    try:
        names  = shap_data['names']
        sv     = shap_data['shap_matrix']    # (n_subsample, F)
        X_mat  = shap_data['X_matrix']       # (n_subsample, F)
        # barcodes list tracked during subsampling (may not exist in older data)
        barcodes = shap_data.get('barcodes', None)

        n = sv.shape[0]
        if barcodes is None or len(barcodes) != n:
            barcodes = [f'row_{i}' for i in range(n)]

        rows = {
            'barcode':   barcodes,
            'model':     [model_name] * n,
            'true_label': [int(df.loc[bc, target])
                           if bc in df.index else -1 for bc in barcodes],
            'pred_prob':  [float(proba_dict.get(str(bc), float('nan')))
                           for bc in barcodes],
        }
        for j, feat in enumerate(names):
            rows[f'shap_{feat}'] = sv[:, j].tolist()
        for j, feat in enumerate(names):
            rows[f'fval_{feat}'] = X_mat[:, j].tolist()

        pd.DataFrame(rows).to_csv(out_path, index=False)
        log(f"[CSV] full SHAP ({n:,} rows × {len(names)} feats) → {out_path.name}")
    except Exception as e:
        log(f"[CSV] full SHAP export failed: {e}")


def _export_full_ig_csv(ig_data: dict, df: pd.DataFrame, target: str,
                         out_path: Path, log: _Log, model_name: str = '') -> None:
    """
    Save full IG matrix as CSV.

    For each antibody the signed IG of the ACTUAL amino acid at each position
    is extracted (shape n × (n_cdr3_pos + n_vh_pos)), making it directly
    comparable to SHAP: positive = drives PASS, negative = drives FAIL.

    Columns:
      barcode, true_label, transformer_prob,
      ig_CDR3_01, ig_CDR3_02, ...,  (signed IG of actual AA at CDR3 position)
      ig_VH_001,  ig_VH_002, ...,   (signed IG of actual AA at VH position)
      hcdr3_seq, cdr3_length
    """
    try:
        AA_IDX    = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        barcodes  = ig_data['barcodes']
        attr_cdr3 = ig_data['attr_cdr3']   # (n, max_cdr3, 20)
        attr_enc  = ig_data['attr_enc']    # (n, max_vh[+vl], 20)
        hcdr3_seqs = ig_data['hcdr3_seqs']
        probs     = ig_data['probs']
        max_cdr3  = ig_data['max_cdr3']
        max_vh    = ig_data['max_vh']

        rows = []
        for idx, bc in enumerate(barcodes):
            row = {
                'barcode': bc,
                'model':   model_name,
                'true_label': int(df.loc[bc, target]) if bc in df.index else 'unknown',
                'transformer_prob': float(probs[idx]),
                'ig_convergence_delta': float(ig_data['convergence_delta'][idx]),
                'ig_baseline': ig_data.get('baseline', 'unknown'),
                'ig_steps': int(ig_data.get('n_steps', 0)),
                'hcdr3_seq': hcdr3_seqs[idx] if idx < len(hcdr3_seqs) else '',
            }
            # CDR3 positions
            cdr3_seq = hcdr3_seqs[idx] if idx < len(hcdr3_seqs) else ''
            for pos in range(max_cdr3):
                aa = cdr3_seq[pos] if pos < len(cdr3_seq) else ''
                if aa in AA_IDX:
                    ig_val = float(attr_cdr3[idx, pos, AA_IDX[aa]])
                else:
                    ig_val = float('nan')
                row[f'ig_CDR3_{pos+1:02d}'] = ig_val

            # VH positions (actual AA from df)
            try:
                vh_seq = str(df.loc[bc, 'HSEQ']).upper().replace('-', '')
            except Exception:
                vh_seq = ''
            for pos in range(max_vh):
                aa = vh_seq[pos] if pos < len(vh_seq) else ''
                if aa in AA_IDX:
                    ig_val = float(attr_enc[idx, pos, AA_IDX[aa]])
                else:
                    ig_val = float('nan')
                row[f'ig_VH_{pos+1:03d}'] = ig_val

            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        log(f"[CSV] full IG ({len(rows):,} rows × {len(out_df.columns)} cols) → {out_path.name}")
    except Exception as e:
        log(f"[CSV] full IG export failed: {e}")


def build_figure_4panel_manuscript(
        shap_csv_psr: str, shap_csv_sec: str,
        ig_csv_psr: str,   ig_csv_sec: str,
        out_stem: str, log: _Log,
        top_n: int = 20,
        model_label_shap: str = 'XGBoost-SHAP',
        model_label_ig:   str = 'Transformer onehot-IG',
        fig_width: float = 18.0,
        fig_height: float = 22.0,
        font_scale: float = 1.0) -> None:
    """
    Generate a publication-quality 4-panel beeswarm figure from CSV files.

    Panels:
      a  {model_label_shap}  PSR   b  {model_label_shap}  SEC
      c  {model_label_ig}    PSR   d  {model_label_ig}    SEC

    Each panel is a beeswarm: rows = top_n features sorted by mean |attribution|,
    dots = individual antibodies, x = attribution value, colour = feature value
    (RdBu_r: red=high, blue=low).

    Parameters
    ----------
    shap_csv_psr/sec : path to shap_xgb_FULL_*.csv
    ig_csv_psr/sec   : path to ig_FULL_*.csv
    out_stem         : output prefix (no extension)
    top_n            : number of features/positions to show per panel
    fig_width/height : figure size in cm (converted to inches internally)
    font_scale       : scale all fonts (1.0 = default, 1.5 = larger for poster)
    """
    import matplotlib.colors as _mc

    W_IN = fig_width  / 2.54
    H_IN = fig_height / 2.54
    FS   = 7 * font_scale   # base font size

    fig, axes = plt.subplots(2, 2, figsize=(W_IN, H_IN))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.06,
                        wspace=0.45, hspace=0.35)

    def _load(csv_path, attr_prefix):
        """
        Load CSV for beeswarm. Handles three formats:

        1. SHAP wide (shap_ prefix):
           columns: barcode, shap_{feat}, fval_{feat}, true_label, pred_prob

        2. IG beeswarm long (ig_ prefix, 'aa' column present):
           columns: model, region, aa, ig_value, barcode
           → pivot to (n_barcodes × 20 AAs), colour by AA class

        3. IG full wide (ig_ prefix, CDR3/VH columns):
           columns: ig_CDR3_01, ig_CDR3_02, … hcdr3_seq
           → aggregate per AA by summing positions
        """
        d = pd.read_csv(csv_path)
        aa_colors_per_col = None

        # ── Format 2: beeswarm long-format (preferred for IG) ─────────────
        if 'aa' in d.columns and 'ig_value' in d.columns:
            _FULL = {'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe',
                     'G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu',
                     'M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg',
                     'S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr'}

            # Split by region, pivot each separately
            rows_out = []   # (feat_label, ig_vec, aa_color, region)
            for region_tag, region_label, top_k in [
                ('HCDR3',        'HCDR3',        20),   # all 20 AAs, sorted by |IG|
                ('VH_framework', 'VH framework',  3),   # top 3 VH AAs
                ('VL_framework', 'VL framework',  3),   # top 3 VL AAs
            ]:
                sub = d[d['region'] == region_tag]
                if sub.empty:
                    continue
                piv = sub.pivot_table(index='barcode', columns='aa',
                                      values='ig_value', aggfunc='sum')
                piv = piv.reindex(columns=list(AMINO_ACIDS), fill_value=0.0)
                mean_abs = np.nanmean(np.abs(piv.values), axis=0)
                order = np.argsort(mean_abs)[::-1][:top_k]
                for j in order:
                    aa = AMINO_ACIDS[j]
                    label = f"{aa}  ({_FULL.get(aa, aa)})"
                    ig_vec = piv.values[:, j].astype(float)
                    rows_out.append((label, ig_vec, _aa_color(aa), region_label))

            if not rows_out:
                # fallback
                rows_out = [(f"{aa}  ({_FULL.get(aa,aa)})",
                             np.zeros(len(d.groupby('barcode').ngroups)),
                             _aa_color(aa), 'HCDR3') for aa in AMINO_ACIDS]

            n_rows = len(rows_out)
            n_ab   = len(rows_out[0][1]) if rows_out else 1
            feat_names = [r[0] for r in rows_out]
            sv  = np.column_stack([r[1] for r in rows_out]).T   # (n_rows, n_ab) → transpose → (n_ab, n_rows)
            sv  = np.array([r[1] for r in rows_out]).T           # (n_ab, n_rows)
            xv  = np.zeros_like(sv)
            colour_label = 'AA class'
            aa_colors_per_col = np.empty((n_ab, n_rows), dtype=object)
            for j, (_, _, col, _) in enumerate(rows_out):
                aa_colors_per_col[:, j] = col
            yl = np.full(n_ab, -1, dtype=int)
            pb = np.full(n_ab, float('nan'))
            return feat_names, sv, xv, yl, pb, colour_label, aa_colors_per_col

        feat_cols = [c for c in d.columns if c.startswith(attr_prefix)]
        fval_cols = [c.replace(attr_prefix, 'fval_') for c in feat_cols
                     if c.replace(attr_prefix, 'fval_') in d.columns]

        if fval_cols:
            # ── Format 1: SHAP wide ───────────────────────────────────────
            feat_names = [c[len(attr_prefix):] for c in feat_cols]
            sv  = d[feat_cols].values.astype(float)
            xv  = d[fval_cols].values.astype(float)
            colour_label = 'Feature value'

        else:
            # ── Format 3: IG full wide — aggregate per AA ─────────────────
            _FULL = {'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe',
                     'G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu',
                     'M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg',
                     'S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr'}
            AA_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
            n_ab   = len(d)
            sv_all = d[feat_cols].values.astype(float)
            sv_aa  = np.zeros((n_ab, 20), dtype=float)
            hcdr3_seqs = (d['hcdr3_seq'].fillna('').tolist()
                          if 'hcdr3_seq' in d.columns else [''] * n_ab)
            for j, col in enumerate(feat_cols):
                if 'CDR3' not in col: continue
                try: pos = int(''.join(filter(str.isdigit, col))) - 1
                except Exception: continue
                for i, seq in enumerate(hcdr3_seqs):
                    if seq and 0 <= pos < len(seq):
                        aa = seq[pos]
                        if aa in AA_IDX:
                            v = sv_all[i, j]
                            if not np.isnan(v): sv_aa[i, AA_IDX[aa]] += v
            feat_names = [f"{aa}  ({_FULL.get(aa,aa)})" for aa in AMINO_ACIDS]
            sv  = sv_aa
            xv  = np.zeros_like(sv)
            colour_label = 'AA class'
            aa_colors_per_col = np.empty((n_ab, 20), dtype=object)
            for j, aa in enumerate(AMINO_ACIDS):
                aa_colors_per_col[:, j] = _aa_color(aa)

        yl_raw = d['true_label'] if 'true_label' in d.columns else pd.Series(['unknown']*len(d))
        yl  = pd.to_numeric(yl_raw, errors='coerce').fillna(-1).astype(int).values
        pb_col = 'transformer_prob' if 'transformer_prob' in d.columns else 'pred_prob'
        pb  = pd.to_numeric(d.get(pb_col, pd.Series([float('nan')]*len(d))),
                            errors='coerce').values.astype(float)
        return feat_names, sv, xv, yl, pb, colour_label, aa_colors_per_col

    def _beeswarm(ax, feat_names, sv, xv, true_labels,
                  top_n, title, xlabel, fs, colour_label='Feature value',
                  aa_colors=None):
        """
        Draw beeswarm.
        aa_colors: if not None, (n_ab, n_feat) array of hex colors — used for IG panels
        """
        valid = ~np.all(np.isnan(sv), axis=0)
        sv_v  = sv[:, valid]
        xv_v  = xv[:, valid] if xv.shape == sv.shape else np.zeros_like(sv_v)
        aac_v = aa_colors[:, valid] if aa_colors is not None else None
        names_v = [n for n, vv in zip(feat_names, valid) if vv]

        # Define is_ig early — used for both row ordering and dot styling
        is_ig = aa_colors is not None

        mean_abs = np.nanmean(np.abs(sv_v), axis=0)
        # For IG panels preserve loaded order (HCDR3 sorted by |IG|, then VH/VL)
        # For SHAP panels sort by mean |attribution| descending
        if is_ig:
            top_idx = np.arange(min(sv_v.shape[1], top_n))
        else:
            top_idx = np.argsort(mean_abs)[::-1][:top_n]
        cmap = plt.cm.RdBu_r

        dot_s = (3 * font_scale) if not is_ig else (6 * font_scale)
        dot_a = 0.55 if not is_ig else 0.50

        for row_idx, fi in enumerate(top_idx):
            vals  = sv_v[:, fi]
            ok    = ~np.isnan(vals)
            if not ok.any(): continue

            rng    = np.random.default_rng(fi)
            jitter = rng.uniform(-0.28, 0.28, size=ok.sum())
            ys     = row_idx + jitter

            if is_ig and aac_v is not None:
                # Color by AA physicochemical class
                colors = list(aac_v[ok, fi])
                ax.scatter(vals[ok], ys, c=colors, s=dot_s,
                           alpha=dot_a, linewidths=0, zorder=2)
            else:
                fvals   = xv_v[:, fi]
                lo, hi  = np.nanmin(fvals), np.nanmax(fvals)
                norm_fv = (fvals - lo) / (hi - lo + 1e-10)
                colors  = cmap(norm_fv[ok])
                ax.scatter(vals[ok], ys, c=colors, s=dot_s,
                           alpha=dot_a, linewidths=0, zorder=2)

        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([names_v[i] for i in top_idx],
                           fontsize=fs * 0.95, fontfamily='monospace', color='#000000')
        ax.invert_yaxis()
        ax.axvline(0, color='#000000', lw=0.8, zorder=3)
        ax.grid(axis='x', alpha=0.15, lw=0.3)
        ax.set_xlabel(xlabel, fontsize=fs * 1.05, labelpad=3, color='#000000')
        ax.tick_params(axis='x', labelsize=fs * 0.95, colors='#000000')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

        # Colourbar / legend
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _ins
        if is_ig:
            # AA class legend — shifted right by 0.4 in using bbox_to_anchor
            _short = ['Cationic (R,K,H)', 'Anionic (D,E)',
                      'Hydrophobic/Aromatic', 'Small/Polar']
            handles = [mpatches.Patch(facecolor=c, edgecolor='none', label=l)
                       for c, l in zip(_AA_CLASS_COLORS, _short)]
            ax.legend(handles=handles, title='AA class',
                      title_fontsize=fs * 0.6, fontsize=fs * 0.6,
                      loc='lower right',
                      bbox_to_anchor=(1.22, 0.0),   # shift 0.4 in right of axis edge
                      bbox_transform=ax.transAxes,
                      frameon=True, framealpha=0.9,
                      edgecolor='#000000', handlelength=0.9, handleheight=0.75,
                      borderpad=0.4, labelspacing=0.25)
        else:
            _ax_cb = _ins(ax, width='38%', height='2.2%',
                          loc='upper right', borderpad=0.8)
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=_mc.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cb = plt.colorbar(sm, cax=_ax_cb, orientation='horizontal')
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'], fontsize=fs * 0.90, color='#000000')
            cb.set_label(colour_label, fontsize=fs * 0.90, labelpad=1, color='#000000')
            cb.ax.xaxis.set_label_position('top')
            cb.ax.xaxis.tick_top()
            cb.ax.tick_params(width=0.4, length=1.5, labelsize=fsz * 0.80, colors='#000000')

    # ── Load and render each panel ────────────────────────────────────────
    panel_defs = [
        (axes[0, 0], shap_csv_psr, 'shap_', 'a',
         f'{model_label_shap}  PSR',
         'SHAP value\n(← FAIL  |  PASS →)'),
        (axes[0, 1], shap_csv_sec, 'shap_', 'b',
         f'{model_label_shap}  SEC',
         'SHAP value\n(← FAIL  |  PASS →)'),
        (axes[1, 0], ig_csv_psr,   'ig_',   'c',
         f'{model_label_ig}  PSR',
         'IG attribution\n(← FAIL  |  PASS →)'),
        (axes[1, 1], ig_csv_sec,   'ig_',   'd',
         f'{model_label_ig}  SEC',
         'IG attribution\n(← FAIL  |  PASS →)'),
    ]

    for ax, csv_path, prefix, letter, title, xlabel in panel_defs:
        try:
            feat_names, sv, xv, yl, pb, colour_label, aa_colors = _load(csv_path, prefix)
            _beeswarm(ax, feat_names, sv, xv, yl, top_n, title, xlabel, FS,
                      colour_label=colour_label, aa_colors=aa_colors)
        except Exception as e:
            ax.text(0.5, 0.5, f'Load failed:\n{csv_path}\n{e}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=FS * 0.8, color='#000000', wrap=True)

        # Panel letter + title above axis
        # Panels a/b (SHAP, row 0) lifted 0.2 in extra vs c/d (IG, row 1)
        pos     = ax.get_position()
        H_IN    = fig.get_size_inches()[1]
        extra   = 0.2 / H_IN if letter in ('a', 'b') else 0.0
        title_y = pos.y1 + 0.012 + extra
        fig.text(pos.x0, title_y, letter,
                 fontsize=FS * 1.4, fontweight='bold', color='#000000',
                 va='bottom', ha='left', transform=fig.transFigure)
        fig.text(pos.x0 + 0.022, title_y, title,
                 fontsize=FS, fontweight='bold', color='#000000',
                 va='bottom', ha='left', transform=fig.transFigure)

    _save_fig(fig, out_stem, log)
    log(f"[4panel] manuscript figure → {out_stem}.{{tiff,pdf,png}}")


def build_figure_6panel_manuscript(
        rf_csv_psr: str,   xgb_csv_psr: str,   ig_csv_psr: str,
        rf_csv_sec: str,   xgb_csv_sec: str,   ig_csv_sec: str,
        out_stem: str, log: _Log,
        top_n: int = 20,
        fig_width_cm: float = 18.0,
        font_scale: float = 1.0) -> None:
    """
    Publication-quality 6-panel beeswarm figure from CSV files.

    Layout (2 rows × 3 columns):
      a  RF-SHAP   PSR   |  b  XGBoost-SHAP  PSR   |  c  Transformer-IG  PSR
      d  RF-SHAP   SEC   |  e  XGBoost-SHAP  SEC   |  f  Transformer-IG  SEC

    Row 1 (PSR) and Row 2 (SEC) share the same feature row order within each model,
    so direct row-by-row comparison is possible.

    Parameters
    ----------
    *_csv_psr/sec : paths to beeswarm CSVs generated by build_figure_3beeswarms
                    (shap_ prefix for RF/XGBoost, ig_ prefix for Transformer)
    top_n         : features per panel  (use 12 for compact version)
    fig_width_cm  : total width in cm (18 = double-column)
    font_scale    : multiplier for all font sizes
    """
    import matplotlib.colors as _mc2

    FS    = font_scale
    W_IN  = fig_width_cm / 2.54
    # Height: 2 rows, each row sized to top_n features
    H_IN  = max(8.0,  top_n * 0.26 * FS * 2 + 3.0)

    fig, axes = plt.subplots(2, 3, figsize=(W_IN, H_IN))
    # wspace=0.75 gives generous horizontal breathing room between columns
    # left=0.18 gives space for y-tick labels
    fig.subplots_adjust(left=0.18, right=0.88, top=0.92, bottom=0.07,
                        wspace=0.75, hspace=0.22)

    cmap = plt.cm.RdBu_r

    def _load_beeswarm_csv(csv_path, attr_prefix):
        """
        Load beeswarm CSV. Handles two formats:

        SHAP format (RF / XGBoost):
          columns: model, rank, feature, shap_value, feature_value

        Transformer IG format:
          columns: model, region, aa, ig_value, barcode
        """
        d = pd.read_csv(csv_path)

        # ── Transformer IG format ─────────────────────────────────────────
        if 'aa' in d.columns and 'ig_value' in d.columns:
            # One row per (antibody × AA); group by AA label
            d['feature'] = d['region'].str.replace('_', ' ', regex=False) + \
                           '  ' + d['aa'].astype(str)
            val_col   = 'ig_value'
            fval_col  = None
            colour_label = 'Transformer prob'
        # ── SHAP format ───────────────────────────────────────────────────
        elif 'feature' in d.columns and 'shap_value' in d.columns:
            val_col   = 'shap_value'
            fval_col  = 'feature_value' if 'feature_value' in d.columns else None
            colour_label = 'Feature value'
        else:
            raise ValueError(f"Unrecognised CSV format in {csv_path}. "
                             f"Columns: {list(d.columns)}")

        feat_col = 'feature'
        features  = d[feat_col].unique().tolist()
        sv_dict   = {}; xv_dict = {}
        for feat in features:
            mask = d[feat_col] == feat
            sv_dict[feat]  = d.loc[mask, val_col].values.astype(float)
            if fval_col and fval_col in d.columns:
                xv_dict[feat] = d.loc[mask, fval_col].values.astype(float)
            else:
                xv_dict[feat] = np.zeros(mask.sum(), dtype=float)

        return features, sv_dict, xv_dict, colour_label

    def _rank_features(sv_dict, top_n, is_ig=False):
        """Sort features by mean |value|, return top_n names.
        For IG panels: top 14 HCDR3 rows first, then top 4 VH rows.
        For SHAP panels: global sort by mean |value|.
        """
        scored = {f: float(np.nanmean(np.abs(sv_dict[f]))) for f in sv_dict}
        if is_ig:
            hcdr3 = sorted([f for f in scored if 'HCDR3' in f],
                           key=lambda x: scored[x], reverse=True)
            vh    = sorted([f for f in scored if ('VH' in f or 'vh' in f) and 'HCDR3' not in f],
                           key=lambda x: scored[x], reverse=True)
            # Return ORIGINAL keys (needed for sv_dict lookup in _draw_panel)
            # "framework" is stripped only at display time in set_yticklabels
            return hcdr3[:14] + vh[:4]
        return sorted(scored, key=lambda x: scored[x], reverse=True)[:top_n]

    def _draw_panel(ax, features_ordered, sv_dict, xv_dict, xlabel, fsz,
                    dot_size, colour_label, show_yticks=True):
        """Draw one beeswarm panel with given feature order."""
        rng_panel = np.random.default_rng(42)

        # Detect IG panel: feature names contain AA letter after double-space
        # e.g. "HCDR3  R", "VH  G"
        is_ig = (colour_label in ('Transformer prob', 'AA class'))

        for row_i, feat in enumerate(features_ordered):
            if feat not in sv_dict: continue
            vals  = sv_dict[feat]
            fvals = xv_dict.get(feat, np.zeros_like(vals))
            ok    = ~np.isnan(vals)
            if not ok.any(): continue
            lo, hi  = np.nanmin(fvals), np.nanmax(fvals)
            norm_fv = (fvals - lo) / (hi - lo + 1e-10)
            jitter  = rng_panel.uniform(-0.28, 0.28, size=ok.sum())

            if is_ig:
                # Extract AA from feature name: "HCDR3  R" → 'R'
                parts = feat.strip().split()
                aa = parts[-1] if parts else ''
                dot_color = _aa_color(aa) if len(aa) == 1 and aa in AMINO_ACIDS else '#AAAAAA'
                ax.scatter(vals[ok], row_i + jitter,
                           c=dot_color, s=dot_size, alpha=0.55,
                           linewidths=0, zorder=2, rasterized=True)
            else:
                colors = plt.cm.RdBu_r(norm_fv[ok])
                ax.scatter(vals[ok], row_i + jitter,
                           c=colors, s=dot_size, alpha=0.6,
                           linewidths=0, zorder=2, rasterized=True)

        n = len(features_ordered)
        ax.set_yticks(range(n))
        if show_yticks:
            _display_labels = [f.replace(' framework', '') for f in features_ordered]
            ax.set_yticklabels(_display_labels, fontsize=fsz * 0.82,
                               fontfamily='monospace', color='#000000')
        else:
            ax.set_yticklabels(['' for _ in features_ordered])
        ax.invert_yaxis()
        ax.axvline(0, color='#000000', lw=0.8, zorder=3)
        ax.grid(axis='x', alpha=0.15, lw=0.3)
        ax.set_xlabel(xlabel, fontsize=fsz, labelpad=2, color='#000000')
        ax.tick_params(axis='x', labelsize=fsz * 0.85, colors='#000000')
        ax.tick_params(axis='y', colors='#000000')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('#333333')
            ax.spines[sp].set_linewidth(0.6)

        # Colourbar (SHAP) or AA class legend (IG)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _ins
        if is_ig:
            _short = ['Cationic (R,K,H)', 'Anionic (D,E)',
                      'Hydrophobic/Aromatic', 'Small/Polar']
            handles = [mpatches.Patch(facecolor=c, edgecolor='none', label=l)
                       for c, l in zip(_AA_CLASS_COLORS, _short)]
            ax.legend(handles=handles, title='AA class',
                      title_fontsize=fsz * 0.85, fontsize=fsz * 0.80,
                      loc='lower left',
                      bbox_to_anchor=(1.02, 0.0),
                      bbox_transform=ax.transAxes,
                      frameon=True, framealpha=0.9,
                      edgecolor='#000000', handlelength=0.7, handleheight=0.55,
                      borderpad=0.3, labelspacing=0.15, handletextpad=0.4)
        else:
            pass  # Shared colorbar drawn after all panels (see below)

    # ── Load all 6 CSVs ───────────────────────────────────────────────────
    panel_specs = [
        # (row, col, csv_path, attr_prefix, xlabel, panel_letter, title)
        (0, 0, rf_csv_psr,   'shap_', 'SHAP value\n(← FAIL | PASS →)', 'a', 'RF-PSR'),
        (0, 1, xgb_csv_psr,  'shap_', 'SHAP value\n(← FAIL | PASS →)', 'b', 'XGBoost-PSR'),
        (0, 2, ig_csv_psr,   'ig_',   'IG value\n(← FAIL | PASS →)',   'c', 'Transformer-IG PSR'),
        (1, 0, rf_csv_sec,   'shap_', 'SHAP value\n(← FAIL | PASS →)', 'd', 'RF-SEC'),
        (1, 1, xgb_csv_sec,  'shap_', 'SHAP value\n(← FAIL | PASS →)', 'e', 'XGBoost-SEC'),
        (1, 2, ig_csv_sec,   'ig_',   'IG value\n(← FAIL | PASS →)',   'f', 'Transformer-IG SEC'),
    ]

    # Build per-model shared row order: rank by PSR, reuse for SEC
    shared_order = {}
    for ri, ci, csv_path, prefix, xlabel, letter, title in panel_specs:
        try:
            feats, sv_d, xv_d, clbl = _load_beeswarm_csv(csv_path, prefix)
            # SHAP panels (RF+XGBoost) share row order per dataset row
            # IG panels keep their own order (different feature space)
            if prefix == 'ig_':
                key = (ri, ci, prefix)   # IG: each panel independent
            else:
                key = (ri, 'shap_')      # SHAP: RF and XGBoost share same order per row
            if key not in shared_order:
                _is_ig = (prefix == 'ig_')
                shared_order[key] = _rank_features(sv_d, top_n, is_ig=_is_ig)
            panel_specs[panel_specs.index((ri,ci,csv_path,prefix,xlabel,letter,title))] = \
                (ri, ci, csv_path, prefix, xlabel, letter, title, feats, sv_d, xv_d, clbl)
        except Exception as e:
            log(f"[6panel] Load failed {csv_path}: {e}")
            panel_specs[panel_specs.index((ri,ci,csv_path,prefix,xlabel,letter,title))] = \
                (ri, ci, csv_path, prefix, xlabel, letter, title, [], {}, {}, 'value')

    DOT = max(4, int(8 * FS))    # larger dots — more visible at manuscript size
    FSZ = 8.5 * FS               # larger base font

    for spec in panel_specs:
        ri, ci, csv_path, prefix, xlabel, letter, title = spec[:7]
        feats = spec[7] if len(spec) > 7 else []
        sv_d  = spec[8] if len(spec) > 8 else {}
        xv_d  = spec[9] if len(spec) > 9 else {}
        clbl  = spec[10] if len(spec) > 10 else 'value'
        ax    = axes[ri, ci]
        if prefix == 'ig_':
            key = (ri, ci, prefix)
        else:
            key = (ri, 'shap_')
        feat_order = shared_order.get(key, feats[:top_n])

        if not feat_order:
            ax.text(0.5, 0.5, f'Load failed\n{Path(csv_path).name}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9.5, color='#000000', style='italic')
            continue

        _show_yticks = (ci != 1) or (prefix == 'ig_')  # hide y-labels for XGBoost cols
        _draw_panel(ax, feat_order, sv_d, xv_d, xlabel, FSZ, DOT, clbl, show_yticks=_show_yticks)

        # Panel letter + title via fig.text
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1 + 0.012, letter,
                 fontsize=FSZ * 1.3, fontweight='bold', color='#000000',
                 va='bottom', ha='left', transform=fig.transFigure)
        fig.text(pos.x0 + 0.022, pos.y1 + 0.013, title,
                 fontsize=FSZ * 1.0, fontweight='bold', color='#000000',
                 va='bottom', ha='left', transform=fig.transFigure)

    # ── Shared vertical Feature Value colorbars (between cols a-b and d-e) ─
    import matplotlib.colors as _mc3
    _cmap_cb = plt.cm.RdBu_r
    _sm      = plt.cm.ScalarMappable(cmap=_cmap_cb, norm=_mc3.Normalize(0, 1))
    _sm.set_array([])
    fig.canvas.draw()   # finalize axes positions

    for _ri in range(2):
        _ax_a  = axes[_ri, 0]
        _ax_b  = axes[_ri, 1]
        _pos_a = _ax_a.get_position()
        _pos_b = _ax_b.get_position()
        _gap_x = (_pos_a.x1 + _pos_b.x0) / 2
        _cb_w  = 0.008
        _cb_h  = (_pos_a.y1 - _pos_a.y0) * 0.50
        _cb_y  = _pos_a.y0 + (_pos_a.y1 - _pos_a.y0) * 0.25
        _cax   = fig.add_axes([_gap_x - _cb_w / 2, _cb_y, _cb_w, _cb_h])
        _cb    = fig.colorbar(_sm, cax=_cax, orientation='vertical')
        _cb.set_ticks([0, 1])
        _cb.set_ticklabels(['Low', 'High'], fontsize=FSZ * 0.75, color='#000000')
        _cb.set_label('Feature\nvalue', fontsize=FSZ * 0.75, labelpad=3, color='#000000')
        _cb.ax.tick_params(width=0.4, length=2, colors='#000000')

    _save_fig(fig, out_stem, log)
    log(f"[6panel] → {out_stem}.{{tiff,pdf,png}}  (top_n={top_n})")


def _load_result_from_disk(args, db_path: str, target: str,
                            outdir_base: str) -> dict:
    """
    Reconstruct a result dict from previously saved CSV/NPZ files.
    Used by --csv-exist mode to skip model re-computation.

    Loads:
      - shap_rf_FULL_*.csv     → rf_shap dict
      - shap_xgb_FULL_*.csv    → xgb_shap dict
      - ig_{target}_*.npz      → ig_data dict
      - region_attribution_*.csv → rf_reg / xgb_reg / ig_reg
    """
    db_stem = Path(db_path).stem
    outdir  = Path(outdir_base)

    log_path = outdir / f"interp_log_{target}_{args.rf_lm}_{args.xgb_lm}_{args.transformer_lm}_{db_stem}.txt"
    log = _Log(str(log_path))
    log(f"[csv-exist] Loading saved results from {outdir}")

    result = dict(db_path=db_path, target=target, db_stem=db_stem,
                  outdir=outdir, log=log,
                  rf_shap=None, xgb_shap=None, ig_data=None,
                  rf_reg=None, xgb_reg=None, ig_reg=None, df=None,
                  transformer_model_path=None,
                  xgb_proba={}, rf_proba={},
                  rf_model=None, xgb_model=None)

    # ── Load df ───────────────────────────────────────────────────────────
    try:
        result['df'] = _load_db(db_path, target, log)
    except Exception as e:
        log(f"[csv-exist] df load failed: {e}")

    # ── Load RF SHAP ──────────────────────────────────────────────────────
    rf_csv = list(outdir.glob(f"shap_rf_FULL_{target}_{args.rf_lm}_{db_stem}.csv"))
    if rf_csv:
        try:
            d = pd.read_csv(rf_csv[0])
            shap_cols = [c for c in d.columns if c.startswith('shap_')]
            fval_cols = [c.replace('shap_', 'fval_') for c in shap_cols
                         if c.replace('shap_', 'fval_') in d.columns]
            names = [c[5:] for c in shap_cols]
            sv    = d[shap_cols].values.astype(float)
            xv    = d[fval_cols].values.astype(float) if fval_cols else np.zeros_like(sv)
            result['rf_shap'] = {
                'names':         names,
                'mean_abs_shap': np.nanmean(np.abs(sv), axis=0),
                'shap_matrix':   sv,
                'X_matrix':      xv,
                'barcodes':      d['barcode'].astype(str).tolist() if 'barcode' in d.columns else [],
                'expected':      0.0,
            }
            result['rf_reg'] = _region_attribution_tree(result['rf_shap'], args.rf_lm)
            log(f"[csv-exist] RF SHAP loaded: {sv.shape}")
        except Exception as e:
            log(f"[csv-exist] RF SHAP load failed: {e}")

    # ── Load XGBoost SHAP ─────────────────────────────────────────────────
    xgb_csv = list(outdir.glob(f"shap_xgb_FULL_{target}_{args.xgb_lm}_{db_stem}.csv"))
    if xgb_csv:
        try:
            d = pd.read_csv(xgb_csv[0])
            shap_cols = [c for c in d.columns if c.startswith('shap_')]
            fval_cols = [c.replace('shap_', 'fval_') for c in shap_cols
                         if c.replace('shap_', 'fval_') in d.columns]
            names = [c[5:] for c in shap_cols]
            sv    = d[shap_cols].values.astype(float)
            xv    = d[fval_cols].values.astype(float) if fval_cols else np.zeros_like(sv)
            result['xgb_shap'] = {
                'names':         names,
                'mean_abs_shap': np.nanmean(np.abs(sv), axis=0),
                'shap_matrix':   sv,
                'X_matrix':      xv,
                'barcodes':      d['barcode'].astype(str).tolist() if 'barcode' in d.columns else [],
                'expected':      0.0,
            }
            result['xgb_reg'] = _region_attribution_tree(result['xgb_shap'], args.xgb_lm)
            log(f"[csv-exist] XGBoost SHAP loaded: {sv.shape}")
        except Exception as e:
            log(f"[csv-exist] XGBoost SHAP load failed: {e}")

    # ── Load Transformer IG from NPZ ──────────────────────────────────────
    npz_files = list(outdir.glob(f"ig_{target}_{args.transformer_lm}_{db_stem}.npz"))
    if npz_files:
        try:
            z = np.load(npz_files[0], allow_pickle=True)
            result['ig_data'] = {
                'attr_enc':   z['attr_enc'],
                'attr_cdr3':  z['attr_cdr3'],
                'hcdr3_seqs': list(z['hcdr3_seqs']),
                'barcodes':   list(z['barcodes']),
                'probs':      z['probs'],
                'vh_only':    bool(z['vh_only']),
                'max_vh':     int(z['max_vh']),
                'max_vl':     int(z['max_vl']),
                'max_cdr3':   int(z['max_cdr3']),
            }
            result['ig_reg'] = _region_attribution_ig(result['ig_data'])
            log(f"[csv-exist] IG NPZ loaded: attr_cdr3={z['attr_cdr3'].shape}")
        except Exception as e:
            log(f"[csv-exist] IG NPZ load failed: {e}")

    # ── Find transformer model path (for per-antibody IG) ─────────────────
    tr_path = _find_final(args.model_dir, target, args.transformer_lm,
                          'transformer_onehot', db_stem, '.pt')
    if tr_path:
        result['transformer_model_path'] = tr_path

    # ── Load RF model (for per-antibody SHAP + rf_proba) ──────────────────
    rf_path = _find_final(args.model_dir, target, args.rf_lm, 'rf', db_stem, '.pkl')
    if rf_path:
        try:
            from models.randomforest import RandomForestModel
            rf_model = RandomForestModel.load(rf_path)
            result['rf_model'] = rf_model
            if result['df'] is not None:
                fb  = rf_model.fb_
                ne  = np.array(fb.non_embedding_indices)
                X   = fb.transform(result['df'], None)[:, ne]
                probs = rf_model.model.predict_proba(X)[:, 1]
                result['rf_proba'] = {str(bc): float(p)
                                      for bc, p in zip(result['df'].index, probs)}
                log(f"[csv-exist] RF model loaded + rf_proba computed")
        except Exception as e:
            log(f"[csv-exist] RF model load failed: {e}")

    # ── Load XGB model (for per-antibody SHAP + xgb_proba) ───────────────
    xgb_path = _find_final(args.model_dir, target, args.xgb_lm, 'xgboost', db_stem, '.pkl')
    if xgb_path:
        try:
            from models.xgboost import XGBoostModel
            xgb_model = XGBoostModel.load(xgb_path)
            result['xgb_model'] = xgb_model
            if result['df'] is not None:
                fb  = xgb_model.fb_
                ne  = np.array(fb.non_embedding_indices)
                X   = fb.transform(result['df'], None)[:, ne]
                probs = xgb_model.model.predict_proba(X)[:, 1]
                result['xgb_proba'] = {str(bc): float(p)
                                       for bc, p in zip(result['df'].index, probs)}
                log(f"[csv-exist] XGBoost model loaded + xgb_proba computed")
        except Exception as e:
            log(f"[csv-exist] XGBoost model load failed: {e}")

    # ── Load region attribution ───────────────────────────────────────────
    reg_csv = list(outdir.glob(f"region_attribution_{target}_{db_stem}.csv"))
    if reg_csv and result['rf_reg'] is None:
        try:
            d = pd.read_csv(reg_csv[0])
            for method, key in [('RF','rf_reg'),('XGBoost','xgb_reg'),('Transformer','ig_reg')]:
                sub = d[d['method'] == method]
                if not sub.empty:
                    result[key] = dict(zip(sub['region'], sub['fraction_of_mass']))
        except Exception as e:
            log(f"[csv-exist] region_attribution load failed: {e}")

    log(f"[csv-exist] Done. rf_shap={'OK' if result['rf_shap'] else 'MISSING'}  "
        f"xgb_shap={'OK' if result['xgb_shap'] else 'MISSING'}  "
        f"ig_data={'OK' if result['ig_data'] else 'MISSING'}")
    return result


def _run_one_dataset(args, db_path: str, target: str,
                     outdir_base: str = None,
                     _inject_df=None, _inject_models: dict = None,
                     _model_only: bool = False) -> dict:
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
                  rf_path=None, xgb_path=None,
                  xgb_proba={}, rf_proba={},
                  rf_model=None, xgb_model=None)   # kept for per-antibody SHAP
    try:
        log("═" * 62)
        log(f"  DELPHI INTERPRETABILITY ANALYSIS")
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
        elif _model_only or not os.path.exists(db_path):
            # Model-load-only mode: skip training-set analysis entirely.
            # Used in predict mode (Step 1) and when only --model-path is given.
            # We still locate + load the model objects below; there is simply
            # no training dataframe to compute SHAP/IG on.
            _reason = ("predict mode — models only" if _model_only
                       else f"db file not found ({db_path})")
            log(f"[load] {_reason} → model-load-only mode")
            log(f"[load] training-set SHAP/IG skipped; loading model objects "
                f"for the prediction step")
            result['_model_only'] = True
            df = None
        else:
            df = _load_db(db_path, target, log)
        if df is not None and 'label' not in df.columns:
            df['label'] = df[target].astype(int, errors='ignore')
        result['df'] = df

        # ── Single-architecture gating (Mode C) ──────────────────────────
        # When --model is given, only load/run that architecture. In Mode A/B
        # (single_model is None) all three are loaded as before.
        _only = getattr(args, 'single_model', None)
        def _want(arch: str) -> bool:
            return (_only is None) or (_only == arch)

        # ── RF ──────────────────────────────────────────────────────────
        rf_path = (((_inject_models or {}).get('rf_path')
                    or _find_final(args.model_dir, target, args.rf_lm,
                                   'rf', db_stem, '.pkl'))
                   if _want('rf') else None)
        if not _want('rf'):
            log(f"\n[RF]  skipped (--model {_only})")
        else:
            log(f"\n[RF]  checkpoint → {rf_path or 'NOT FOUND'}")
        if rf_path:
            try:
                from models.randomforest import RandomForestModel
                rf_model = RandomForestModel.load(rf_path)
                result['rf_model'] = rf_model
                result['rf_path']  = rf_path
                rf_shap  = (_compute_tree_shap(rf_model, df,
                                               args.shap_max_samples, log)
                            if df is not None else None)
                if rf_shap is not None:
                    result['rf_shap'] = rf_shap
                    result['rf_reg']  = _region_attribution_tree(
                        rf_shap, kmer_source=args.rf_lm)
                    # Summary CSV (mean |SHAP| per feature)
                    pd.DataFrame({
                        'feature': rf_shap['names'],
                        'mean_abs_shap': rf_shap['mean_abs_shap'],
                        'region': [_feature_region(n, args.rf_lm)
                                   for n in rf_shap['names']],
                    }).sort_values('mean_abs_shap', ascending=False).to_csv(
                        outdir / f"shap_rf_{target}_{args.rf_lm}_{db_stem}.csv",
                        index=False)
                    log(f"[RF]  csv → shap_rf_{target}_{args.rf_lm}_{db_stem}.csv")

                    # Full SHAP matrix CSV (n_antibodies × n_features)
                    # Rows: barcodes  |  Cols: shap_{feature}, fval_{feature}, true_label, pred_prob
                    _export_full_shap_csv(
                        rf_shap, df, target, result.get('rf_proba', {}),
                        outdir / f"shap_rf_FULL_{target}_{args.rf_lm}_{db_stem}.csv",
                        log, model_name=f"RF-{args.rf_lm}")

                # ── RF predicted probs for ALL antibodies (for triple-model selection)
                try:
                    if df is not None:
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
        xgb_path = (((_inject_models or {}).get('xgb_path')
                     or _find_final(args.model_dir, target, args.xgb_lm,
                                    'xgboost', db_stem, '.pkl'))
                    if _want('xgboost') else None)
        if not _want('xgboost'):
            log(f"\n[XGB] skipped (--model {_only})")
        else:
            log(f"\n[XGB] checkpoint → {xgb_path or 'NOT FOUND'}")
        if xgb_path:
            try:
                from models.xgboost import XGBoostModel
                xgb_model = XGBoostModel.load(xgb_path)
                result['xgb_model'] = xgb_model
                result['xgb_path']  = xgb_path
                xgb_shap  = (_compute_tree_shap(xgb_model, df,
                                                args.shap_max_samples, log)
                             if df is not None else None)
                if xgb_shap is not None:
                    result['xgb_shap'] = xgb_shap
                    result['xgb_reg']  = _region_attribution_tree(
                        xgb_shap, kmer_source=args.xgb_lm)
                    # Summary CSV
                    pd.DataFrame({
                        'feature': xgb_shap['names'],
                        'mean_abs_shap': xgb_shap['mean_abs_shap'],
                        'region': [_feature_region(n, args.xgb_lm)
                                   for n in xgb_shap['names']],
                    }).sort_values('mean_abs_shap', ascending=False).to_csv(
                        outdir / f"shap_xgb_{target}_{args.xgb_lm}_{db_stem}.csv",
                        index=False)
                    log(f"[XGB] csv → shap_xgb_{target}_{args.xgb_lm}_{db_stem}.csv")

                    # Full SHAP matrix CSV
                    _export_full_shap_csv(
                        xgb_shap, df, target, result.get('xgb_proba', {}),
                        outdir / f"shap_xgb_FULL_{target}_{args.xgb_lm}_{db_stem}.csv",
                        log, model_name=f"XGBoost-{args.xgb_lm}")

                # ── XGBoost predicted probs for ALL antibodies (for per-ab titles)
                try:
                    if df is not None:
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
        tr_path = (((_inject_models or {}).get('tr_path')
                    or _find_final(args.model_dir, target, args.transformer_lm,
                                   'transformer_onehot', db_stem, '.pt'))
                   if _want('transformer_onehot') else None)
        if not _want('transformer_onehot'):
            log(f"\n[IG]  skipped (--model {_only})")
        else:
            log(f"\n[IG]  checkpoint → {tr_path or 'NOT FOUND'}")
        if tr_path:
            result['transformer_model_path'] = tr_path   # for per-antibody IG
            try:
                from models.transformer_onehot import TransformerOneHotModel
                tr_model = TransformerOneHotModel.load(tr_path)
                tr_model.set_lm_mode(args.transformer_lm
                                     if args.transformer_lm in ('onehot', 'onehot_vh')
                                     else 'onehot')
                ig_data = (_compute_ig(tr_model, df,
                                       args.ig_max_samples, args.ig_steps, log,
                                       ig_baseline=getattr(args, 'ig_baseline', 'uniform'))
                           if df is not None else None)
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
                        convergence_delta = ig_data['convergence_delta'],
                        baseline   = ig_data['baseline'],
                        n_steps    = ig_data['n_steps'],
                        vh_only    = ig_data['vh_only'],
                        max_vh     = ig_data['max_vh'],
                        max_vl     = ig_data['max_vl'],
                        max_cdr3   = ig_data['max_cdr3'],
                    )
                    log(f"[IG]  npz → ig_{target}_{args.transformer_lm}_{db_stem}.npz")

                    # Full IG matrix CSV (n_antibodies × n_positions)
                    _export_full_ig_csv(
                        ig_data, df, target,
                        outdir / f"ig_FULL_{target}_{args.transformer_lm}_{db_stem}.csv",
                        log, model_name=f"Transformer-{args.transformer_lm}")
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
            ig_out = _compute_ig(m, df, args.ig_max_samples, args.ig_steps, log,
                                    ig_baseline=getattr(args, 'ig_baseline', 'uniform'))
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

    # ── Prediction summary — PASS/FAIL rate per model ─────────────────────
    n_total = len(pred_df)
    THRESH  = 0.5
    sep  = "═" * 58
    sep2 = "─" * 58
    summary_lines = [
        "",
        sep,
        f"  PREDICTION SUMMARY — {target}",
        f"  Input   : {Path(predict_path).name}  (n={n_total:,})",
        f"  Threshold: {THRESH}  (PASS = score >= {THRESH})",
        sep2,
        f"  {'Model':<28}  {'PASS':>6}  {'FAIL':>6}  {'% PASS':>7}",
        f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*7}",
    ]
    for col, label in [
        ('rf_biophysical_score',      'RF (biophysical)'),
        ('xgboost_biophysical_score', 'XGBoost (biophysical)'),
        ('transformer_onehot_score',  'Transformer (onehot)'),
        (f'predicted_{target}',        'Ensemble (majority vote)'),
    ]:
        if col not in pred_df.columns:
            continue
        vals  = pred_df[col].dropna()
        if col == f'predicted_{target}':
            n_pass = int((vals == 1).sum())
        else:
            n_pass = int((vals >= THRESH).sum())
        n_fail   = len(vals) - n_pass
        pct_pass = 100 * n_pass / max(len(vals), 1)
        summary_lines.append(
            f"  {label:<28}  {n_pass:>6,}  {n_fail:>6,}  {pct_pass:>6.1f}%")
    summary_lines += [sep2, ""]
    for line in summary_lines:
        log(line)
        print(line)

    # ── Run full interpretability pipeline on prediction set ───────────────
    # Inject df (with pseudo-labels) + the model paths resolved in the training
    # step, so the prediction-set run uses the SAME models regardless of the
    # prediction file's stem (which would not match the model db_stem).
    _inject_models = {
        'rf_path':  train_r.get('rf_path'),
        'xgb_path': train_r.get('xgb_path'),
        'tr_path':  train_r.get('transformer_model_path'),
    }
    result = _run_one_dataset(args, predict_path, target, str(outdir),
                               _inject_df=df, _inject_models=_inject_models)
    log.close()
    return result


def _find_beeswarm_csv(outdir, model_tag: str, target_col: str) -> str:
    """Return path to beeswarm CSV for a given model tag and target, or '' if not found."""
    candidates = list(Path(str(outdir)).glob(f"*beeswarm_{model_tag}_{target_col}.csv"))
    return str(candidates[0]) if candidates else ''


def run(args) -> int:
    # ── Resolve root output directory once ───────────────────────────────
    root_dir = Path(args.outdir) if args.outdir else Path('outputs')

    # ── Unified sample limit ──────────────────────────────────────────────
    _max = getattr(args, 'max_samples', 3000)
    if _max is None: _max = 3000
    # 0 = use ALL antibodies (no subsampling)
    _shap_legacy = getattr(args, 'shap_max_samples', None)
    _ig_legacy   = getattr(args, 'ig_max_samples',   None)
    args.shap_max_samples = _shap_legacy if _shap_legacy is not None else _max
    args.ig_max_samples   = _ig_legacy   if _ig_legacy   is not None else _max

    # ── PREDICT MODE ──────────────────────────────────────────────────────
    # If --predict is given, first run the normal training pipeline to load
    # models, then run prediction on the unseen set using those models.
    predict_path  = getattr(args, 'predict',  None)
    predict_path2 = getattr(args, 'predict2', None)

    # ── CSV-EXIST MODE: regenerate all figures from saved CSVs/NPZs ───────
    csv_exist = getattr(args, 'csv_exist', False)
    if csv_exist:
        print("[csv-exist] Skipping model computation — loading from saved CSVs/NPZs")
        outdir1 = root_dir / f"interp_{args.target}_{Path(args.db).stem}"
        r1 = _load_result_from_disk(args, args.db, args.target, str(outdir1))
        log1 = r1['log']

        r2 = None
        if args.db2 and args.target2:
            outdir2 = root_dir / f"interp_{args.target2}_{Path(args.db2).stem}"
            r2 = _load_result_from_disk(args, args.db2, args.target2, str(outdir2))

        for r in ([r1] if r2 is None else [r1, r2]):
            _render_dataset_figures(args, r)

        if r2 is not None:
            combined_dir = root_dir / f"interp_{r1['target']}_{r2['target']}_combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            out_stem = str(combined_dir /
                           f"combined_{r1['target']}_{r2['target']}_"
                           f"{args.rf_lm}_{args.transformer_lm}")
            log1(f"\n[csv-exist] Regenerating combined figures → {out_stem}")

            build_figure_2row_6panels(r1, r2, args, out_stem, log1)
            build_figure_xgb_dual_filter(
                xgb_shap_1=r1['xgb_shap'], xgb_shap_2=r2['xgb_shap'],
                label_1=r1['target'].upper().replace('_','-'),
                label_2=r2['target'].upper().replace('_','-'),
                out_stem=out_stem, log=log1, top_n=20)
            try:
                build_figure_transformer_ig_6panel(
                    r1=r1, r2=r2, out_stem=out_stem, log=log1,
                    fig_width_cm=22.0, font_scale=1.2)
            except Exception as _e6:
                log1(f"[transformer_ig_6panel] ERROR: {_e6}\n{__import__('traceback').format_exc()}")
            build_figure_cdr3_heatmap_psr_sec(
                ig_data_psr=r1.get('ig_data'), ig_data_sec=r2.get('ig_data'),
                out_stem=out_stem, log=log1, fig_width_cm=18.0, font_scale=1.2)

            rf_psr_f  = _find_beeswarm_csv(r1['outdir'], 'RF',         r1['target'])
            xgb_psr_f = _find_beeswarm_csv(r1['outdir'], 'XGBoost',    r1['target'])
            tr_psr_f  = _find_beeswarm_csv(r1['outdir'], 'Transformer', r1['target'])
            rf_sec_f  = _find_beeswarm_csv(r2['outdir'], 'RF',         r2['target'])
            xgb_sec_f = _find_beeswarm_csv(r2['outdir'], 'XGBoost',    r2['target'])
            tr_sec_f  = _find_beeswarm_csv(r2['outdir'], 'Transformer', r2['target'])
            if all([rf_psr_f, xgb_psr_f, tr_psr_f, rf_sec_f, xgb_sec_f, tr_sec_f]):
                for top_n_6, suffix in [(20,'all'),(15,'top15')]:
                    build_figure_6panel_manuscript(
                        rf_csv_psr=rf_psr_f, xgb_csv_psr=xgb_psr_f, ig_csv_psr=tr_psr_f,
                        rf_csv_sec=rf_sec_f, xgb_csv_sec=xgb_sec_f, ig_csv_sec=tr_sec_f,
                        out_stem=str(combined_dir / f"manuscript_6panel_{r1['target']}_{r2['target']}_{suffix}"),
                        log=log1, top_n=top_n_6, fig_width_cm=22.0, font_scale=1.3)

            shap_psr = Path(str(r1['outdir'])) / f"shap_xgb_FULL_{r1['target']}_{args.xgb_lm}_{r1['db_stem']}.csv"
            shap_sec = Path(str(r2['outdir'])) / f"shap_xgb_FULL_{r2['target']}_{args.xgb_lm}_{r2['db_stem']}.csv"
            ig_psr   = (_find_beeswarm_csv(r1['outdir'], 'Transformer', r1['target']) or
                        str(Path(str(r1['outdir'])) / f"ig_FULL_{r1['target']}_{args.transformer_lm}_{r1['db_stem']}.csv"))
            ig_sec   = (_find_beeswarm_csv(r2['outdir'], 'Transformer', r2['target']) or
                        str(Path(str(r2['outdir'])) / f"ig_FULL_{r2['target']}_{args.transformer_lm}_{r2['db_stem']}.csv"))
            build_figure_4panel_manuscript(
                shap_csv_psr=str(shap_psr), shap_csv_sec=str(shap_sec),
                ig_csv_psr=ig_psr, ig_csv_sec=ig_sec,
                out_stem=str(combined_dir / f"manuscript_4panel_{r1['target']}_{r2['target']}"),
                log=log1, top_n=20,
                model_label_shap=f'XGBoost-SHAP ({args.xgb_lm})',
                model_label_ig='Transformer onehot-IG',
                fig_width=18.0, fig_height=22.0, font_scale=1.0)

        log1(f"\n[csv-exist] All figures regenerated → {root_dir}/")

        # ── Per-antibody waterfall figures (IG + SHAP) ────────────────────
        if r2 is not None:
            _n_pairs = getattr(args, 'n_pairs', 20)
            log1(f"\n[csv-exist] Finding example antibody pairs (n={_n_pairs}) ...")
            fail_list, pass_list = _find_all_example_antibodies(r1, r2, log1, n_each=_n_pairs)

            if fail_list or pass_list:
                log1(f"\n[csv-exist] Rendering per-antibody Transformer-IG waterfalls ...")
                build_figure_per_antibody_ig(
                    r1=r1, r2=r2, out_stem=out_stem, log=log1,
                    n_vhvl_top=6, ig_steps=getattr(args, 'ig_steps', 100),
                    n_pairs=_n_pairs,
                    _precomputed_lists=(fail_list, pass_list))

                log1(f"\n[csv-exist] Rendering per-antibody XGBoost-SHAP waterfalls ...")
                build_figure_shap_per_antibody(
                    r1=r1, r2=r2, out_stem=out_stem, log=log1,
                    fail_list=fail_list, pass_list=pass_list,
                    model_key='xgb_model', n_pairs=_n_pairs, top_n=25,
                    ig_steps=getattr(args, 'ig_steps', 100))

                log1(f"\n[csv-exist] Rendering per-antibody RF-SHAP waterfalls ...")
                build_figure_shap_per_antibody(
                    r1=r1, r2=r2, out_stem=out_stem, log=log1,
                    fail_list=fail_list, pass_list=pass_list,
                    model_key='rf_model', n_pairs=_n_pairs, top_n=25,
                    ig_steps=getattr(args, 'ig_steps', 100))
            else:
                log1(f"\n[csv-exist] No triple-model agreement pairs found — skipping per-antibody figures")
                log1(f"           (Hint: RF/XGB models need to be loaded — check --model-dir)")

        log1(f"\n[csv-exist] Done. All outputs in {root_dir}/")
        log1.close()
        return 0

    if predict_path:
        # Step 1 — LOAD MODELS ONLY (no training-set analysis).
        # Both --db and --model-path resolve to the same model here; we only
        # need the model objects/paths so the predict step can apply them.
        # Training-set SHAP/IG is intentionally skipped — the user wants
        # interpretation of the PREDICTED antibodies, not the training set.
        outdir1_tr = root_dir / f"interp_{args.target}_{Path(args.db).stem}"
        r1_train   = _run_one_dataset(args, args.db, args.target,
                                       str(outdir1_tr), _model_only=True)
        log1 = r1_train['log']

        r2_train = None
        if args.db2 and args.target2:
            outdir2_tr = root_dir / f"interp_{args.target2}_{Path(args.db2).stem}"
            r2_train   = _run_one_dataset(args, args.db2, args.target2,
                                           str(outdir2_tr), _model_only=True)

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

        # ── Single-target predict (Mode C) ───────────────────────────────
        # Render per-dataset standalone figures + per-antibody figures for the
        # requested architecture(s) on the PREDICTION set.
        if r2 is None:
            single_model = getattr(args, 'single_model', None)
            _render_dataset_figures(args, r1)   # standalone per-arch figures

            # Decide which architectures to render per-antibody figures for
            if single_model:
                models = [single_model]
                log1(f"\n[predict][Mode C] single architecture: {single_model}")
            else:
                models = []
                if r1.get('ig_data')   is not None: models.append('transformer_onehot')
                if r1.get('xgb_model') is not None: models.append('xgboost')
                if r1.get('rf_model')  is not None: models.append('rf')
                log1(f"\n[predict][Mode B] all architectures: {', '.join(models)}")

            # Filter to only architectures whose model actually loaded
            avail = []
            for mdl in models:
                if mdl == 'transformer_onehot' and r1.get('ig_data')   is not None: avail.append(mdl)
                elif mdl == 'xgboost'          and r1.get('xgb_model') is not None: avail.append(mdl)
                elif mdl == 'rf'               and r1.get('rf_model')  is not None: avail.append(mdl)
                else:
                    log1(f"[predict] {mdl} model not available — skipping")
            if avail:
                _render_per_antibody_single_target(args, r1, avail)

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

    # ── Single-target per-antibody figures (Mode B) ──────────────────────
    # Waterfall + CDR3 mutagenesis per antibody, N PASS + N FAIL (--n-antibodies).
    # Only for single-target; dual-target uses its own frozen per-antibody path.
    if r2 is None:
        # Determine which architectures have models loaded
        _models = []
        if r1.get('ig_data')   is not None: _models.append('transformer_onehot')
        if r1.get('xgb_model') is not None: _models.append('xgboost')
        if r1.get('rf_model')  is not None: _models.append('rf')
        if _models:
            log1(f"\n[per_ab] Generating single-target per-antibody figures "
                 f"({', '.join(_models)}) ...")
            _render_per_antibody_single_target(args, r1, _models)
        else:
            log1("\n[per_ab] No models available for per-antibody figures — skipping")

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
        log1(f"\n[4panel] Generating 4-panel manuscript figure from CSV ...")
        db1_stem = r1['db_stem']
        db2_stem = r2['db_stem']
        shap_psr = Path(str(r1['outdir'])) / f"shap_xgb_FULL_{r1['target']}_{args.xgb_lm}_{db1_stem}.csv"
        shap_sec = Path(str(r2['outdir'])) / f"shap_xgb_FULL_{r2['target']}_{args.xgb_lm}_{db2_stem}.csv"
        # Use beeswarm Transformer CSV (same per-AA aggregation as 2-row figure)
        ig_psr   = _find_beeswarm_csv(r1['outdir'], 'Transformer', r1['target'])
        ig_sec   = _find_beeswarm_csv(r2['outdir'], 'Transformer', r2['target'])
        if not ig_psr:
            ig_psr = str(Path(str(r1['outdir'])) / f"ig_FULL_{r1['target']}_{args.transformer_lm}_{db1_stem}.csv")
        if not ig_sec:
            ig_sec = str(Path(str(r2['outdir'])) / f"ig_FULL_{r2['target']}_{args.transformer_lm}_{db2_stem}.csv")
        manuscript_stem = str(combined_dir / f"manuscript_4panel_{r1['target']}_{r2['target']}")
        build_figure_4panel_manuscript(
            shap_csv_psr     = str(shap_psr),
            shap_csv_sec     = str(shap_sec),
            ig_csv_psr       = str(ig_psr),
            ig_csv_sec       = str(ig_sec),
            out_stem         = manuscript_stem,
            log              = log1,
            top_n            = 20,
            model_label_shap = f'XGBoost-SHAP  ({args.xgb_lm})',
            model_label_ig   = 'Transformer onehot-IG',
            fig_width        = 18.0,
            fig_height       = 22.0,
            font_scale       = 1.0)

        # ── 6-panel manuscript figure (RF + XGBoost + Transformer × PSR + SEC) ──
        rf_psr   = Path(str(r1['outdir'])) / f"combined_..._beeswarm_RF_{r1['target']}.csv"
        rf_sec   = Path(str(r2['outdir'])) / f"combined_..._beeswarm_RF_{r2['target']}.csv"
        xgb_psr  = Path(str(r1['outdir'])) / f"combined_..._beeswarm_XGBoost_{r1['target']}.csv"
        xgb_sec  = Path(str(r2['outdir'])) / f"combined_..._beeswarm_XGBoost_{r2['target']}.csv"
        tr_psr_b = Path(str(r1['outdir'])) / f"combined_..._beeswarm_Transformer_{r1['target']}.csv"
        tr_sec_b = Path(str(r2['outdir'])) / f"combined_..._beeswarm_Transformer_{r2['target']}.csv"

        # Resolve actual filenames (stem varies by outdir name)
        rf_psr_f  = _find_beeswarm_csv(r1['outdir'], 'RF',          r1['target'])
        xgb_psr_f = _find_beeswarm_csv(r1['outdir'], 'XGBoost',     r1['target'])
        tr_psr_f  = _find_beeswarm_csv(r1['outdir'], 'Transformer',  r1['target'])
        rf_sec_f  = _find_beeswarm_csv(r2['outdir'], 'RF',          r2['target'])
        xgb_sec_f = _find_beeswarm_csv(r2['outdir'], 'XGBoost',     r2['target'])
        tr_sec_f  = _find_beeswarm_csv(r2['outdir'], 'Transformer',  r2['target'])

        if all([rf_psr_f, xgb_psr_f, tr_psr_f, rf_sec_f, xgb_sec_f, tr_sec_f]):
            for top_n_6, suffix in [(20, 'all'), (15, 'top15')]:
                log1(f"\n[6panel] Generating 6-panel manuscript figure (top_n={top_n_6}) ...")
                build_figure_6panel_manuscript(
                    rf_csv_psr  = rf_psr_f,   xgb_csv_psr = xgb_psr_f,  ig_csv_psr = tr_psr_f,
                    rf_csv_sec  = rf_sec_f,   xgb_csv_sec = xgb_sec_f,  ig_csv_sec = tr_sec_f,
                    out_stem    = str(combined_dir / f"manuscript_6panel_{r1['target']}_{r2['target']}_{suffix}"),
                    log         = log1,
                    top_n       = top_n_6,
                    fig_width_cm = 22.0,   # wider than double-column to fit 3 cols + labels
                    font_scale   = 1.3)
        else:
            log1(f"[6panel] Some beeswarm CSVs missing — skipping 6-panel figure")

        # ── Combined PSR+SEC HCDR3 heatmap (main manuscript figure) ──────
        log1(f"\n[cdr3_heatmap] Rendering combined PSR+SEC HCDR3 heatmap ...")
        build_figure_cdr3_heatmap_psr_sec(
            ig_data_psr  = r1.get('ig_data'),
            ig_data_sec  = r2.get('ig_data'),
            out_stem     = out_stem,
            log          = log1,
            fig_width_cm = 18.0,
            font_scale   = 1.2)

        # ── 6-panel Transformer IG figure (main interpretability figure) ──
        log1(f"\n[transformer_ig_6panel] Rendering 6-panel Transformer IG figure ...")
        try:
            build_figure_transformer_ig_6panel(
                r1           = r1,
                r2           = r2,
                out_stem     = out_stem,
                log          = log1,
                fig_width_cm = 22.0,
                font_scale   = 1.2)
        except Exception as _e6:
            log1(f"[transformer_ig_6panel] ERROR: {_e6}")
            import traceback as _tb
            log1(_tb.format_exc())

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


def _select_single_target_antibodies(r: dict, n_each: int, log):
    """
    Select N PASS + N FAIL antibodies for single-target per-antibody figures.

    NO predictive-score filter. Selection is purely by true label:
      PASS = true label 1, FAIL = true label 0.
    n_each = 0 → return ALL antibodies of each class.

    Returns (fail_bcs, pass_bcs) — two lists of barcodes.
    """
    df     = r['df']
    target = r['target']
    if df is None or target not in df.columns:
        log(f"[per_ab] df or target '{target}' missing — cannot select antibodies")
        return [], []

    # Use the antibodies that have IG data (so waterfall + mutagenesis work)
    ig_data = r.get('ig_data')
    if ig_data is not None and 'barcodes' in ig_data:
        available = [str(b) for b in ig_data['barcodes']]
    else:
        available = [str(b) for b in df.index]

    fail_bcs, pass_bcs = [], []
    for bc in available:
        try:
            y = int(df.loc[bc, target])
        except Exception:
            continue
        if   y == 0: fail_bcs.append(bc)
        elif y == 1: pass_bcs.append(bc)

    if n_each and n_each > 0:
        fail_bcs = fail_bcs[:n_each]
        pass_bcs = pass_bcs[:n_each]

    log(f"[per_ab] selected {len(pass_bcs)} PASS + {len(fail_bcs)} FAIL "
        f"antibodies (n_antibodies={n_each if n_each else 'ALL'})")
    return fail_bcs, pass_bcs


def _predicted_score(r: dict, bc: str, model: str):
    """Return DELPHI predicted PASS-probability for one antibody, or None."""
    if model == 'transformer_onehot':
        ig_data = r.get('ig_data')
        if ig_data and 'barcodes' in ig_data and 'probs' in ig_data:
            pm = {str(b): float(p)
                  for b, p in zip(ig_data['barcodes'], ig_data['probs'])}
            return pm.get(str(bc))
    elif model == 'xgboost':
        return r.get('xgb_proba', {}).get(str(bc))
    elif model == 'rf':
        return r.get('rf_proba', {}).get(str(bc))
    return None


def _render_per_antibody_single_target(args, r: dict, models: list):
    """
    Generate one figure per antibody: waterfall + CDR3 mutagenesis, for each
    requested architecture. Single-target (Mode B/C).

    models : subset of ['transformer_onehot', 'xgboost', 'rf'] to render.
             For each architecture present, each antibody gets:
               - 1 waterfall (IG for transformer, SHAP for xgb/rf)
               - 1 CDR3 mutagenesis heatmap (transformer or tree)
    NO predictive-score filter. Each panel labeled with BARCODE, actual label
    (PASS=1 / FAIL=0), and DELPHI predicted score.
    """
    df      = r['df']
    target  = r['target']
    outdir  = r['outdir']
    db_stem = r['db_stem']
    log     = r['log']

    n_each  = getattr(args, 'n_antibodies', 20)
    fail_bcs, pass_bcs = _select_single_target_antibodies(r, n_each, log)
    all_bcs = [(bc, 'PASS') for bc in pass_bcs] + [(bc, 'FAIL') for bc in fail_bcs]
    if not all_bcs:
        log("[per_ab] no antibodies selected — skipping per-antibody figures")
        return

    # Output subdirectory for the many per-antibody figures
    ab_dir = Path(str(outdir)) / f"per_antibody_{target}_{db_stem}"
    ab_dir.mkdir(parents=True, exist_ok=True)
    log(f"[per_ab] writing per-antibody figures → {ab_dir}/")

    def _cdr3_of(bc):
        try: return str(df.loc[bc, 'CDR3']).upper().replace('-', '')
        except Exception: return ''

    for bc, outcome in all_bcs:
        cdr3 = _cdr3_of(bc)
        for model in models:
            # ── Build a 1-row, 2-column figure: waterfall | mutagenesis ──
            fig = plt.figure(figsize=(11.0, 5.2))
            gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0],
                                    left=0.10, right=0.97, top=0.84, bottom=0.12,
                                    wspace=0.42)
            ax_w = fig.add_subplot(gs[0, 0])   # waterfall
            ax_m = fig.add_subplot(gs[0, 1])   # mutagenesis

            score = _predicted_score(r, bc, model)
            score_s = f"{score:.4f}" if score is not None else "n/a"

            if model == 'transformer_onehot':
                _waterfall_single_ab(ax_w, bc, r,
                                     ig_steps=getattr(args, 'ig_steps', 100))
                _render_mutagenesis_heatmap(ax_m, bc, r)
                model_tag = 'Transformer onehot · IG'
            elif model == 'xgboost':
                _waterfall_shap_single_ab(ax_w, bc, r, top_n=25,
                                          model_key='xgb_model')
                _render_mutagenesis_heatmap_tree(ax_m, bc, r, model_key='xgb_model')
                model_tag = f'XGBoost · SHAP ({args.xgb_lm})'
            elif model == 'rf':
                _waterfall_shap_single_ab(ax_w, bc, r, top_n=25,
                                          model_key='rf_model')
                _render_mutagenesis_heatmap_tree(ax_m, bc, r, model_key='rf_model')
                model_tag = f'Random Forest · SHAP ({args.rf_lm})'
            else:
                plt.close(fig)
                continue

            # ── Figure title: BARCODE, actual label, predicted score ──────
            fig.suptitle(
                f"{bc}  ·  {target}  ·  {model_tag}\n"
                f"CDR3: {cdr3}   |   actual = {outcome} ({1 if outcome=='PASS' else 0})"
                f"   |   DELPHI score = {score_s}",
                fontsize=11.0, fontweight='bold', color='#000000', y=0.985)

            out_stem = str(ab_dir /
                           f"ab_{bc}_{outcome}_{model}_{target}_{db_stem}")
            _save_fig(fig, out_stem, log)

    log(f"[per_ab] done — {len(all_bcs)} antibodies × {len(models)} "
        f"architecture(s) = {len(all_bcs) * len(models)} figures")


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
                             fig_stem, log, top_n=20,
                             fig_width_cm=18.0, font_scale=1.2)

    log(f"\n[pass_fail_ig] Rendering PASS vs FAIL signed IG bar plots ...")
    _plot_pass_fail_ig(ig_data=ig_data, df=df, target=target, db_stem=db_stem,
                       out_stem=str(outdir / f"ig_{target}_{args.transformer_lm}_{db_stem}"),
                       log=log, top_n=60)

    # Single-dataset HCDR3 heatmap (PSR-only or SEC-only)
    if ig_data is not None:
        log(f"\n[cdr3_heatmap] Rendering single-dataset HCDR3 heatmap ...")
        build_figure_cdr3_heatmap_psr_sec(
            ig_data_psr  = ig_data,
            ig_data_sec  = None,
            out_stem     = fig_stem,
            log          = log,
            fig_width_cm = 12.0,
            font_scale   = 1.2)

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
    publication-quality style 2-panel XGBoost SHAP beeswarm figure.

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
    cmap = _safe_cmap('RdBu_r')

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
            ax.set_yticklabels(pretty_labels, fontsize=11.5, color='#000000')
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

        ax.axvline(0, color='#000000', lw=0.6, ls='-', zorder=1)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(pretty_labels, fontsize=11.5, color='#000000')
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=7, colors='#000000')
        ax.grid(axis='x', alpha=0.18, lw=0.3)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                      fontsize=12.2, labelpad=3, color='#000000')

    _render_panel(ax_a, xgb_shap_1, label_1, label_1)
    _render_panel(ax_b, xgb_shap_2, label_2, label_2)

    # Panel b: hide y-tick labels (shared axis — labels already on left)
    ax_b.set_yticklabels(['' for _ in pretty_labels], fontsize=11.5, color='#000000')

    # ── Feature-value colourbar — inset upper-right of panel a ───────────
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
    _axins = _inset(ax_a, width='36%', height='2.6%',
                    loc='upper right', borderpad=1.2)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=_mc.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=_axins, orientation='horizontal')
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'], fontsize=10.1, color='#000000')
    cb.set_label('Feature value', fontsize=10.1, labelpad=1, color='#000000')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.tick_top()
    cb.ax.tick_params(width=0.4, length=1.5, labelsize=7)

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
                 fontsize=13.5, fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        fig.text(x0 + 0.025, y1 + 0.011, line1,
                 fontsize=12.2, fontweight='bold', color='#000000', va='bottom', ha='left',
                 transform=fig.transFigure)
        if line2:
            fig.text(x0 + 0.025, y1 + 0.002, line2,
                     fontsize=10.1, va='top', ha='left', color='#000000',
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
        # FAIL: require transformer score in 0.20–0.45 range for informative mutagenesis
        # (too low <0.20 = flat mutagenesis; too high >0.45 = borderline)
        FAIL_SCORE_MIN = 0.10
        FAIL_SCORE_MAX = 0.40
        if (y1 == 0 and y2 == 0
                and tr_pred1 == 0 and tr_pred2 == 0
                and xgb_pred1 == 0 and xgb_pred2 == 0
                and rf_pred1 == 0 and rf_pred2 == 0
                and FAIL_SCORE_MIN <= tp1 <= FAIL_SCORE_MAX
                and FAIL_SCORE_MIN <= tp2 <= FAIL_SCORE_MAX):
            fail_list.append((bc, tp1, tp2, xp1, xp2))
        elif (y1 == 1 and y2 == 1
                and tr_pred1 == 1 and tr_pred2 == 1
                and xgb_pred1 == 1 and xgb_pred2 == 1
                and rf_pred1 == 1 and rf_pred2 == 1):
            pass_list.append((bc, tp1, tp2, xp1, xp2))

    # ── Fallback: relax FAIL score range if not enough found ─────────────────
    if len(fail_list) < n_each:
        log(f"[per_ab] Only {len(fail_list)} FAIL antibodies in 0.20–0.45 range "
            f"(need {n_each}) — relaxing to 0.05–0.49 ...")
        fail_list_relaxed = []
        for bc in common:
            y1 = _true(df1, t1, bc); y2 = _true(df2, t2, bc)
            if y1 is None or y2 is None: continue
            tp1 = tr_prob1[bc]; tp2 = tr_prob2[bc]
            xp1 = xgb_prob1.get(bc); xp2 = xgb_prob2.get(bc)
            rp1 = rf_prob1.get(bc);  rp2 = rf_prob2.get(bc)
            tr_pred1  = 1 if tp1 >= THRESH else 0
            tr_pred2  = 1 if tp2 >= THRESH else 0
            xgb_pred1 = (1 if xp1 >= THRESH else 0) if xp1 is not None else y1
            xgb_pred2 = (1 if xp2 >= THRESH else 0) if xp2 is not None else y2
            rf_pred1  = (1 if rp1 >= THRESH else 0) if rp1 is not None else y1
            rf_pred2  = (1 if rp2 >= THRESH else 0) if rp2 is not None else y2
            already   = any(x[0] == bc for x in fail_list)
            if (not already and y1 == 0 and y2 == 0
                    and tr_pred1 == 0 and tr_pred2 == 0
                    and xgb_pred1 == 0 and xgb_pred2 == 0
                    and rf_pred1 == 0 and rf_pred2 == 0
                    and 0.05 <= tp1 <= 0.49
                    and 0.05 <= tp2 <= 0.49):
                fail_list_relaxed.append((bc, tp1, tp2, xp1, xp2))
        fail_list_relaxed.sort(key=lambda x: abs(x[1] - 0.35))
        n_needed = n_each - len(fail_list)
        fail_list = fail_list + fail_list_relaxed[:n_needed]
        log(f"[per_ab] After relaxed range: {len(fail_list)} FAIL antibodies")

    if len(fail_list) < n_each:
        log(f"[per_ab] WARNING: only {len(fail_list)} FAIL antibodies found "
            f"(even with relaxed range 0.05–0.49). "
            f"Consider reducing --n-pairs or checking model calibration.")

    fail_list.sort(key=lambda x: abs(x[1] - 0.35))  # closest to 0.35 first (most informative)
    pass_list.sort(key=lambda x: -x[1])               # most confident PASS first
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
                ha='center', va='center', transform=ax.transAxes, fontsize=9.5, color='#000000')
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
                    ha='center', va='center', transform=ax.transAxes, fontsize=8.1, color='#000000')
            ax.set_xticks([]); ax.set_yticks([])
            return (true_label, tr_prob, xgb_prob, target.upper().replace('_','-'), '?', '#555', cdr3)

    if attr_enc is None:
        ax.text(0.5, 0.5, 'IG data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=9.5, color='#000000')
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
    ax.set_yticklabels(labels, fontsize=9.5, fontfamily='monospace', color='#000000', fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(0, color='#000000', lw=0.7, ls='-', zorder=3)
    ax.tick_params(axis='x', labelsize=8.1, colors='#000000')
    ax.grid(axis='x', alpha=0.18, lw=0.3, zorder=0)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('Transformer IG attribution\n(<- FAIL  |  PASS ->)', fontsize=10.1, labelpad=2, color='#000000', fontweight='bold')

    # Compact AA class legend inside panel (lower right)
    _short = ['Cationic (R,K,H)', 'Anionic (D,E)', 'Hydrophobic/Aromatic', 'Small/Polar']
    handles = [mpatches.Patch(facecolor=c, edgecolor='none', label=l)
               for c, l in zip(_AA_CLASS_COLORS, _short)]
    # Bars for FAIL antibodies go leftward — legend fits at lower right (positive side)
    # Check which side has more space
    _xmin, _xmax = ax.get_xlim()
    _legend_loc = 'lower right' if abs(_xmax) >= abs(_xmin) else 'lower left'
    ax.legend(handles=handles, title='AA class', title_fontsize=7.0,
              fontsize=6.0, loc=_legend_loc, frameon=True, framealpha=0.88,
              edgecolor='#000000', handlelength=0.7, handleheight=0.55,
              borderpad=0.3, labelspacing=0.15, handletextpad=0.4)

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
                transform=ax.transAxes, fontsize=9.5, color='#000000')
        ax.set_xticks([]); ax.set_yticks([])
        return

    if not tr_path:
        ax.text(0.5, 0.5, 'Transformer model\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=9.5, color='#000000')
        ax.set_xticks([]); ax.set_yticks([])
        return

    try:
        from models.transformer_onehot import TransformerOneHotModel
        m = TransformerOneHotModel.load(tr_path)
    except Exception as e:
        ax.text(0.5, 0.5, f'Model load failed:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=8.1, color='#000000')
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
                    ms=3.5, markerfacecolor='#000000',
                    markeredgecolor='white', markeredgewidth=0.4, zorder=5)

    # Axes
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([cdr3[p] for p in range(n_pos)],
                       fontsize=8.8, fontfamily='monospace', color='#000000')
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(list(AMINO_ACIDS), fontsize=8.1, fontfamily='monospace', color='#000000')
    ax.set_xlabel('CDR3 position  (WT amino acid)', fontsize=8.8, labelpad=2)
    ax.set_ylabel('Mutant AA', fontsize=8.8, labelpad=2, color='#000000')
    ax.tick_params(axis='both', length=1.5, width=0.5, colors='#000000')
    for sp in ax.spines.values(): sp.set_linewidth(0.4)

    # Slim vertical colourbar on the right
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025, aspect=30)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['FAIL', '0.5', 'PASS'],
                         fontsize=10.8, color='#000000', fontweight='bold')
    cbar.ax.yaxis.get_ticklabels()[0].set_color('#C0392B')   # FAIL = red
    cbar.ax.yaxis.get_ticklabels()[-1].set_color('#1A5276')  # PASS = blue
    cbar.ax.tick_params(width=0.6, length=2, colors='#000000')
    cbar.outline.set_linewidth(0.8)
    cbar.set_label('P(PASS)', fontsize=10.8, color='#000000', fontweight='bold')

    # (WT probability not shown — method difference vs XGBoost score already
    #  explained by "Transformer onehot" label in the panel title)


def _build_single_row_figure(bc_fail, bc_pass, r, n_vhvl_top, ig_steps,
                              tr_proba=None):
    """
    One A4 landscape half-page figure — 1 row × 3 columns for a single filter.
    col a = PASS waterfall, col b = FAIL waterfall, col c = CDR3 mutagenesis
    """
    A4_W, A4_H = 11.69, 5.85   # A4 landscape half height

    LEFT  = 0.08
    COL_A_W = 0.250
    COL_B_W = 0.250
    COL_C_W = 0.240
    GAP_AB  = 0.108
    GAP_BC  = 0.103

    col_a_l = LEFT
    col_b_l = col_a_l + COL_A_W + GAP_AB
    col_c_l = col_b_l + COL_B_W + GAP_BC

    ROW_TOP    = 0.920
    ROW_BOTTOM = 0.120
    TITLE_Y    = ROW_TOP + 0.055
    BC_Y       = TITLE_Y - 0.030

    fig = plt.figure(figsize=(A4_W, A4_H))

    ax_a = fig.add_axes([col_a_l, ROW_BOTTOM, COL_A_W, ROW_TOP - ROW_BOTTOM])
    ax_b = fig.add_axes([col_b_l, ROW_BOTTOM, COL_B_W, ROW_TOP - ROW_BOTTOM])
    ax_c = fig.add_axes([col_c_l, ROW_BOTTOM, COL_C_W, ROW_TOP - ROW_BOTTOM])

    def _xp(proba_dict, bc):
        return float(proba_dict[bc]) if proba_dict and bc and bc in proba_dict else None

    def _cdr3(bc):
        try: return str(r['df'].loc[bc, 'CDR3']).upper().replace('-', '')
        except: return ''

    def _fs(): return r['target'].upper().split('_')[0]

    meta_a = _waterfall_single_ab(ax_a, bc_pass, r, n_vhvl_top=n_vhvl_top,
                                   ig_steps=ig_steps, bar_height=0.38,
                                   xgb_prob=_xp(tr_proba, bc_pass)) if bc_pass else (None, None, None, _fs(), '?', '#555', '')
    meta_b = _waterfall_single_ab(ax_b, bc_fail, r, n_vhvl_top=n_vhvl_top,
                                   ig_steps=ig_steps, bar_height=0.38,
                                   xgb_prob=_xp(tr_proba, bc_fail)) if bc_fail else (None, None, None, _fs(), '?', '#555', '')
    ax_a.set_xlabel('')

    _render_mutagenesis_heatmap(ax_c, bc_fail, r)

    for ax, meta, bc, ltr in [(ax_a, meta_a, bc_pass, 'a'),
                               (ax_b, meta_b, bc_fail, 'b')]:
        true_lbl, tr_prob, xgb_prob, fname, outcome, tcol, cdr3_seq = meta
        score_s  = f"={tr_prob:.4f}" if tr_prob is not None else ''
        title_ln = f"{_fs()}{score_s}  |  actual={outcome}"
        bc_line  = f"{bc}:{cdr3_seq}" if cdr3_seq else (bc or '')
        x0 = ax.get_position().x0
        fig.text(x0, TITLE_Y, ltr,
                 fontsize=12.2, fontweight='bold', color='#000000',
                 va='bottom', ha='left', transform=fig.transFigure)
        fig.text(x0 + 0.018, TITLE_Y, title_ln,
                 fontsize=10.1, fontweight='bold', color=tcol,
                 va='bottom', ha='left', transform=fig.transFigure)
        fig.text(x0 + 0.018, BC_Y, bc_line,
                 fontsize=9.5, va='top', ha='left', color='#000000',
                 fontfamily='monospace', transform=fig.transFigure)

    # Mutagenesis title
    true_lbl, tr_prob, *_, cdr3_seq = meta_b
    score_s = f"={tr_prob:.4f}" if tr_prob is not None else ''
    x0 = ax_c.get_position().x0
    fig.text(x0, TITLE_Y, 'e',
             fontsize=12.2, fontweight='bold', color='#000000',
             va='bottom', ha='left', transform=fig.transFigure)
    fig.text(x0 + 0.018, TITLE_Y,
             f"CDR3 mutagenesis  {_fs()}{score_s}  |  actual=FAIL",
             fontsize=10.1, fontweight='bold', color='#C0392B',
             va='bottom', ha='left', transform=fig.transFigure)
    if bc_fail:
        fig.text(x0 + 0.018, BC_Y, f"{bc_fail}:{_cdr3(bc_fail)}",
                 fontsize=9.5, va='top', ha='left', color='#000000',
                 fontfamily='monospace', transform=fig.transFigure)

    return fig


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
    COL_A_W = 0.250   # PASS waterfall
    COL_B_W = 0.250   # FAIL waterfall
    COL_C_W = 0.270   # mutagenesis heatmap

    # Gaps: normal gap between a and b; EXTRA gap before b to push FAIL right
    GAP_AB = 0.108    # 0.5 inch gap between PASS and FAIL columns
    GAP_BC = 0.060

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
                    fontsize=9.5, color='#000000', style='italic')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            meta[key] = (None, None, None, fshort, '?', '#555', '')
        else:
            meta[key] = _waterfall_single_ab(
                ax, bc, r, n_vhvl_top=n_vhvl_top,
                ig_steps=ig_steps, bar_height=0.38, xgb_prob=xgb_p)
        if key[0] == 'psr':
            ax.set_xlabel('', color='#000000')

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
        title_ln = f"{fshort}{score_s}  |  actual={outcome}"
        bc_line  = f"{bc}:{cdr3_seq}" if cdr3_seq else (bc or '')

        ty = TITLE_Y[key]
        fig.text(x0, ty, letter,
                 fontsize=12.2, fontweight='bold', va='bottom', ha='left',
                 color='#000000', transform=fig.transFigure)
        fig.text(x0 + 0.018, ty, title_ln,
                 fontsize=10.1, fontweight='bold', va='bottom', ha='left',
                 color=tcol, transform=fig.transFigure)
        # ── Barcode + CDR3
        fig.text(x0 + 0.018, BC_Y[key], bc_line,
                 fontsize=9.5, va='top', ha='left',
                 color='#000000', fontfamily='monospace',
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
        title_ln  = f"CDR3 mutagenesis  {fshort}{score_s}  |  actual={outcome}"
        bc_line  = f"{bc_fail}:{cdr3_seq}" if cdr3_seq else (bc_fail or '')

        fig.text(x0, ty, ltr,
                 fontsize=12.2, fontweight='bold', va='bottom', ha='left',
                 color='#000000', transform=fig.transFigure)
        fig.text(x0 + 0.018, ty, title_ln,
                 fontsize=8.8, fontweight='bold', va='bottom', ha='left',
                 color=tcol, transform=fig.transFigure)
        fig.text(x0 + 0.018, BC_Y[row_key], bc_line,
                 fontsize=8.8, va='top', ha='left',
                 color='#000000', fontfamily='monospace',
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
                fontsize=9.5, color='#000000', style='italic')
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
            # (n_samples, n_feats) — single class output
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
        cmap = _safe_cmap('RdBu_r')
        lo, hi = xv_top.min(), xv_top.max()
        norm_xv = (xv_top - lo) / (hi - lo + 1e-10)
        colors  = [cmap(v) for v in norm_xv]

        ys = list(range(len(sv_top)))
        ax.barh(ys, sv_top, color=colors, height=0.60,
                edgecolor='none', linewidth=0, zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9.5, fontfamily='monospace', color='#000000', fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(0, color='#000000', lw=0.7, ls='-', zorder=3)
        ax.tick_params(axis='x', labelsize=8.1, colors='#000000')
        ax.grid(axis='x', alpha=0.18, lw=0.3, zorder=0)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        ax.set_xlabel('SHAP value\n(<- FAIL  |  PASS ->)', fontsize=10.1, labelpad=2, color='#000000', fontweight='bold')

        # Feature-value colourbar
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
        _axins = _inset(ax, width='36%', height='2.4%',
                        loc='upper right', borderpad=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=_mc2.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=_axins, orientation='horizontal')
        cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High'], fontsize=10.1, color='#000000')
        cb.set_label('Feature value', fontsize=10.1, labelpad=1, color='#000000')
        cb.ax.xaxis.set_label_position('top'); cb.ax.xaxis.tick_top()
        cb.ax.tick_params(width=0.4, length=1.5)

    except Exception as e:
        ax.text(0.5, 0.5, f'SHAP failed:\n{e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=8.1, color='#000000')
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
                transform=ax.transAxes, fontsize=9.5, color='#000000')
        ax.set_xticks([]); ax.set_yticks([])
        return

    if model is None:
        ax.text(0.5, 0.5, f'{model_key}\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=9.5, color='#000000')
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
                transform=ax.transAxes, fontsize=8.1, color='#000000')
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
                    ms=3.5, markerfacecolor='#000000',
                    markeredgecolor='white', markeredgewidth=0.4, zorder=5)

    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([cdr3[p] for p in range(n_pos)],
                       fontsize=8.8, fontfamily='monospace', color='#000000')
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(list(AMINO_ACIDS), fontsize=8.1, fontfamily='monospace', color='#000000')
    ax.set_xlabel('CDR3 position  (WT amino acid)', fontsize=8.8, labelpad=2)
    ax.set_ylabel('Mutant AA', fontsize=8.8, labelpad=2, color='#000000')
    ax.tick_params(axis='both', length=1.5, width=0.5, colors='#000000')
    for sp in ax.spines.values(): sp.set_linewidth(0.4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025, aspect=30)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['FAIL', '0.5', 'PASS'],
                         fontsize=10.8, color='#000000', fontweight='bold')
    cbar.ax.yaxis.get_ticklabels()[0].set_color('#C0392B')
    cbar.ax.yaxis.get_ticklabels()[-1].set_color('#1A5276')
    cbar.ax.tick_params(width=0.6, length=2, colors='#000000')
    cbar.outline.set_linewidth(0.8)
    cbar.set_label('P(PASS)', fontsize=10.8, color='#000000', fontweight='bold')
    # PASS/FAIL already in tick labels — no floating text needed


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
    GAP_W = 0.108   # 0.5 inch gap between PASS and FAIL columns
    COL_A_W = 0.250; COL_B_W = 0.250; COL_C_W = 0.270
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
    ax_a.set_xlabel('', color='#000000'); ax_b.set_xlabel('', color='#000000')

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
                 fontsize=12.2, fontweight='bold', va='bottom', ha='left',
                 color='#000000', transform=fig.transFigure)
        fig.text(x0+0.018, ty, title_ln,
                 fontsize=8.8, fontweight='bold', va='bottom', ha='left',
                 color=tcol, transform=fig.transFigure)
        fig.text(x0+0.018, bcy, bc_line,
                 fontsize=8.8, va='top', ha='left',
                 color='#000000', fontfamily='monospace', transform=fig.transFigure)

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
                 fontsize=12.2, fontweight='bold', va='bottom', ha='left',
                 color='#000000', transform=fig.transFigure)
        fig.text(x0+0.018, row_ty, mut_lbl,
                 fontsize=8.8, fontweight='bold', va='bottom', ha='left',
                 color='#C0392B', transform=fig.transFigure)
        if bc_fail:
            fig.text(x0+0.018, row_ty - 0.014,
                     f"{bc_fail}:{_cdr3(bc_fail, r1)}",
                     fontsize=8.8, va='top', ha='left',
                     color='#000000', fontfamily='monospace', transform=fig.transFigure)

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

        # ── Also save split single-row figures ───────────────────────────
        # PSR-only
        fig_psr = _build_single_row_figure(
            bc_fail, bc_pass, r1,
            n_vhvl_top=n_vhvl_top, ig_steps=ig_steps,
            tr_proba={bc: tp for bc, tp, *_ in fail_list + pass_list
                      if bc in (bc_fail, bc_pass)})
        _save_fig(fig_psr, f"{pair_stem}_PSR", log)

        # SEC-only
        fig_sec = _build_single_row_figure(
            bc_fail, bc_pass, r2,
            n_vhvl_top=n_vhvl_top, ig_steps=ig_steps,
            tr_proba={bc: tp for bc, _, tp, *_ in fail_list + pass_list
                      if bc in (bc_fail, bc_pass)})
        _save_fig(fig_sec, f"{pair_stem}_SEC", log)


def build_figure_transformer_ig_6panel(
        r1: dict,
        r2: dict,
        out_stem: str,
        log: _Log,
        fig_width_cm: float = 22.0,
        font_scale:   float = 1.0) -> None:
    """
    6-panel Transformer-IG figure — 3 rows × 2 columns.

    Col 1 = PSR filter   Col 2 = SEC filter

    Row a/b : IG per position (area plot, VH/VL/HCDR3 coloured regions)
    Row c/d : HCDR3 AA × position signed IG heatmap
    Row e/f : Cross-method convergence bar chart

    All fonts large and black. Shared y-axis on heatmap rows.
    """
    FS   = font_scale
    W_IN = fig_width_cm / 2.54   # double-column: 18 cm = 7.1in double-col
    H_IN = W_IN * 1.10

    LABEL_FSZ  = 8  * FS    # NB axis labels: 7-8pt
    TICK_FSZ   = 7  * FS    # NB tick labels: 6-7pt
    TITLE_FSZ  = 8  * FS    # NB panel titles
    PANEL_FSZ  = 9  * FS    # NB panel letters: 8-9pt bold
    CB_FSZ     = 7  * FS    # NB colourbar: 6-7pt

    _FULL = {'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe',
             'G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu',
             'M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg',
             'S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr'}

    if r1 is None or r2 is None:
        log(f"[transformer_ig_6panel] skipped — need both r1 and r2 (PSR + SEC)")
        return

    # publication-quality style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2,    'ytick.major.size': 2,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    fig = plt.figure(figsize=(W_IN, H_IN))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[0.65, 2.2, 1.0],
                            hspace=0.38, wspace=0.38,
                            left=0.13, right=0.91,
                            top=0.95, bottom=0.07)

    axes = [[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(3)]

    datasets = [(r1, 'Transformer-IG · PSR', 'a', 'c', 'e'),
                (r2, 'Transformer-IG · SEC', 'b', 'd', 'f')]

    # ── Pre-compute INDEPENDENT colour scales per filter ─────────────────
    # PSR and SEC have different IG magnitudes — each panel uses its own vmax
    vmax_per = {}
    for col_idx, (r, *_) in enumerate(datasets):
        ig = r.get('ig_data')
        if ig is not None:
            mat = ig['attr_cdr3'].mean(axis=0).T
            vmax_per[col_idx] = float(np.abs(mat).max()) or 0.3
        else:
            vmax_per[col_idx] = 0.3

    # Pre-compute shared row order for heatmap (combined mean |IG|)
    combined = np.zeros(20)
    for r, *_ in datasets:
        ig = r.get('ig_data')
        if ig is not None:
            mat = ig['attr_cdr3'].mean(axis=0).T
            seqs    = ig['hcdr3_seqs']
            avg_len = int(np.median([len(s) for s in seqs if s]))
            n_cols  = min(max(avg_len + 2, 12), mat.shape[1])
            combined += np.abs(mat[:, :n_cols]).mean(axis=1)
    row_order = np.argsort(combined)[::-1]
    ylabels   = [f"{AMINO_ACIDS[i]}  ({_FULL[AMINO_ACIDS[i]]})"
                 for i in row_order]

    im_per_col = {}   # store im handle per column for independent colourbars

    for col, (r, filter_label, let_a, let_c, let_e) in enumerate(datasets):
        ig_data = r.get('ig_data')
        rf_reg  = r.get('rf_reg');  xgb_reg = r.get('xgb_reg'); ig_reg = r.get('ig_reg')
        ax_pos  = axes[0][col]   # row a/b : IG per position
        ax_heat = axes[1][col]   # row c/d : heatmap
        ax_conv = axes[2][col]   # row e/f : convergence

        letter_top  = let_a
        letter_heat = let_c
        letter_conv = let_e

        def _label(ax, letter, text='', fsz=None, y=1.08):
            fsz = fsz or TITLE_FSZ * 0.80
            ax.text(-0.01, y, letter, transform=ax.transAxes,
                    fontsize=PANEL_FSZ, fontweight='bold', color='#000000',
                    va='bottom', ha='left', clip_on=False)
            if text:
                ax.text(0.07, y, text, transform=ax.transAxes,
                        fontsize=fsz, fontweight='bold', color='#000000',
                        va='bottom', ha='left', clip_on=False)

        # ── Row 1: IG per position ────────────────────────────────────────
        if ig_data is None:
            ax_pos.text(0.5, 0.5, 'No IG data', ha='center', va='center',
                        transform=ax_pos.transAxes, fontsize=TICK_FSZ, color='#000000')
        else:
            pos_enc  = np.abs(ig_data['attr_enc']).sum(axis=-1).mean(axis=0)
            pos_cdr3 = np.abs(ig_data['attr_cdr3']).sum(axis=-1).mean(axis=0)
            max_vh   = ig_data['max_vh']
            max_vl   = ig_data['max_vl']
            vh_only  = ig_data['vh_only']

            x_vh  = np.arange(1, max_vh + 1)
            vh_y  = pos_enc[:max_vh]
            fw_max = float(vh_y.max())

            if not vh_only:
                vl_x = np.arange(max_vh + 1, max_vh + max_vl + 1)
                vl_y = pos_enc[max_vh:]
                fw_max = max(fw_max, float(vl_y.max()))
                cdr3_x = np.arange(max_vh + max_vl + 5,
                                   max_vh + max_vl + 5 + ig_data['max_cdr3'])
            else:
                vl_x = vl_y = None
                cdr3_x = np.arange(max_vh + 5, max_vh + 5 + ig_data['max_cdr3'])

            cdr3_max = float(pos_cdr3.max())
            ax_pos.fill_between(x_vh, vh_y, color=COLOR_VH_FR, alpha=0.85, lw=0)
            if vl_x is not None:
                ax_pos.fill_between(vl_x, vl_y, color=COLOR_VL_FR, alpha=0.85, lw=0)
            ax_pos.fill_between(cdr3_x, pos_cdr3, color=COLOR_CDR3, alpha=0.85, lw=0)

            # Clip if HCDR3 dominates — no annotation, just clip
            if cdr3_max > 3 * fw_max and fw_max > 0:
                y_clip = fw_max * 2.2
                ax_pos.set_ylim(0, y_clip * 1.5)
            else:
                ax_pos.set_ylim(0, max(cdr3_max, fw_max) * 1.55)

            # Region labels
            # Place region labels via axes-fraction coords (always above data)
            _ylim = ax_pos.get_ylim()
            _yrange = _ylim[1] - _ylim[0]
            for x_mid, label_txt, col_r in [
                ((x_vh[0]+x_vh[-1])/2,    'VH', COLOR_VH_FR),
                ((cdr3_x[0]+cdr3_x[-1])/2, 'HCDR3', COLOR_CDR3),
            ] + ([(((vl_x[0]+vl_x[-1])/2), 'VL', COLOR_VL_FR)]
                 if vl_x is not None else []):
                import matplotlib.transforms as _mbt
                _trans = _mbt.blended_transform_factory(ax_pos.transData, ax_pos.transAxes)
                ax_pos.text(x_mid, 1.02, label_txt,
                            transform=_trans,
                            ha='center', va='bottom', fontsize=TICK_FSZ * 0.9,
                            color=col_r, fontweight='bold', clip_on=False)

            ax_pos.set_xlabel('Position', fontsize=LABEL_FSZ,
                              color='#000000', fontweight='bold')
            ax_pos.set_ylabel('Mean |IG|', fontsize=LABEL_FSZ,
                              color='#000000', fontweight='bold')
            ax_pos.tick_params(axis='both', labelsize=TICK_FSZ,
                               colors='#000000')
            ax_pos.grid(axis='y', alpha=0.2, lw=0.3)
            for sp in ('top', 'right'): ax_pos.spines[sp].set_visible(False)

        _label(ax_pos, letter_top, filter_label, fsz=TITLE_FSZ * 0.80, y=1.22)

        # ── Row 2: HCDR3 heatmap ─────────────────────────────────────────
        if ig_data is None:
            ax_heat.text(0.5, 0.5, 'No IG data', ha='center', va='center',
                         transform=ax_heat.transAxes, fontsize=TICK_FSZ, color='#000000')
        else:
            mat = ig_data['attr_cdr3'].mean(axis=0).T    # (20, max_cdr3)
            seqs    = ig_data['hcdr3_seqs']
            avg_len = int(np.median([len(s) for s in seqs if s]))
            n_cols  = min(max(avg_len + 2, 12), mat.shape[1])

            # Pad matrix to always reach position 20 (so tick labels 18,20 exist)
            MAX_POS = 20   # always show up to CDR3 position 20
            _mat_full = mat[:, :MAX_POS]   # (20, 20) — zero-pad to pos 20
            mat_s = _mat_full[row_order, :]   # all positions 1-20

            _nc_show = mat_s.shape[1]   # = 20 columns (pos 1..20)

            im = ax_heat.imshow(mat_s, cmap=_HEATMAP_CMAP,
                                vmin=-vmax_per[col] * 0.85,
                                vmax= vmax_per[col] * 0.85,
                                aspect='auto', interpolation='nearest',
                                extent=[-0.5, _nc_show - 0.5, 19.5, -0.5])
            im_per_col[col] = im

            # Explicit tick positions (CDR3 position numbers)
            if col == 0:
                # Panel c (PSR): 1,2,3,4,6,8,10,12,14,16,18,20
                _want_pos = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            else:
                # Panel d (SEC): 1,2,3,4,6,8,10,12,14,16,18,20 — NO 17
                _want_pos = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]

            # Convert position → index in mat_s (index = pos - 1)
            _tick_idx = [p - 1 for p in _want_pos if 0 <= (p - 1) < _nc_show]
            _tick_lbl = [str(p) for p in _want_pos if 0 <= (p - 1) < _nc_show]

            ax_heat.set_xticks(_tick_idx)
            ax_heat.set_xticklabels(_tick_lbl,
                                    fontsize=TICK_FSZ * 0.80, color='#000000',
                                    fontweight='bold')
            ax_heat.set_xlabel('HCDR3 position', fontsize=LABEL_FSZ,
                               color='#000000', fontweight='bold', labelpad=5)
            ax_heat.tick_params(axis='x', colors='#000000', length=3, width=0.8)
            ax_heat.tick_params(axis='y', colors='#000000', length=3, width=0.8)
            for sp in ax_heat.spines.values():
                sp.set_edgecolor('#000000'); sp.set_linewidth(0.8)

            ax_heat.set_yticks(range(20))
            if col == 0:
                # Panel c: single-letter AA only (no full names) — saves horizontal space
                ax_heat.set_yticklabels([AMINO_ACIDS[i] for i in row_order],
                                        fontsize=TICK_FSZ, color='#000000',
                                        fontfamily='monospace', fontweight='bold')
                ax_heat.set_ylabel('Amino acid',
                                   fontsize=LABEL_FSZ, color='#000000',
                                   fontweight='bold', labelpad=4)
            else:
                # Panel d: single-letter AA labels
                ax_heat.set_yticklabels([AMINO_ACIDS[i] for i in row_order],
                                        fontsize=TICK_FSZ, color='#000000',
                                        fontfamily='monospace', fontweight='bold')

        _heat_title = f"HCDR3 per-residue IG  ({'PSR' if col==0 else 'SEC'})"
        _label(ax_heat, letter_heat, _heat_title, fsz=TITLE_FSZ * 0.80, y=1.03)

        # ── Row 3: Convergence bar chart ──────────────────────────────────
        data = {'RF': rf_reg or {}, 'XGBoost': xgb_reg or {}, 'Transformer': ig_reg or {}}
        regions = ['HCDR3', 'VH']
        if any(d.get('VL', 0) > 0.01 for d in data.values()):
            regions.append('VL')
        x = np.arange(len(regions))
        w = 0.26
        method_labels = {'RF': 'RF-SHAP', 'XGBoost': 'XGBoost-SHAP', 'Transformer': 'Transformer-IG'}
        method_colors = {'RF': COLOR_RF, 'XGBoost': COLOR_XGB, 'Transformer': COLOR_TRANS}
        for mi, (method, reg_frac) in enumerate(data.items()):
            vals = [reg_frac.get(rg, 0.0) * 100 for rg in regions]
            bars = ax_conv.bar(x + (mi-1)*w, vals, w,
                               color=method_colors[method],
                               edgecolor='white', linewidth=0.4,
                               label=method_labels.get(method, method))
            for b, v in zip(bars, vals):
                if v > 3:
                    ax_conv.text(b.get_x() + b.get_width()/2,
                                 v + 1.5, f"{v:.0f}%",
                                 ha='center', va='bottom',
                                 fontsize=TICK_FSZ * 0.65,
                                 color='#000000', fontweight='bold')

        ax_conv.set_xticks(x)
        ax_conv.set_xticklabels(regions, fontsize=LABEL_FSZ,
                                color='#000000', fontweight='bold')
        ax_conv.set_ylabel('% of |attribution| mass',
                           fontsize=LABEL_FSZ * 0.9,
                           color='#000000', fontweight='bold')
        ax_conv.set_ylim(0, 105)
        ax_conv.tick_params(axis='both', labelsize=TICK_FSZ, colors='#000000')
        ax_conv.grid(axis='y', alpha=0.2, lw=0.3)
        for sp in ('top', 'right'): ax_conv.spines[sp].set_visible(False)
        if col == 0:
            ax_conv.legend(fontsize=TICK_FSZ * 0.85, frameon=True,
                           framealpha=0.9, edgecolor='#000000',
                           loc='upper right')
        else:
            ax_conv.get_legend() and ax_conv.get_legend().remove() if ax_conv.get_legend() else None

        _conv_title = f"Attribution convergence  ({'PSR' if col==0 else 'SEC'})"
        _label(ax_conv, letter_conv, _conv_title, fsz=TITLE_FSZ * 0.80, y=1.08)

    # ── Independent colourbars — one per filter ───────────────────────────
    if im_per_col:
        fig.canvas.draw()
        for col_idx, im in im_per_col.items():
            pos = axes[1][col_idx].get_position()
            cb_ax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.016, pos.height])
            cb = fig.colorbar(im, cax=cb_ax)
            cb.set_label('Mean signed IG', fontsize=CB_FSZ * 0.85,
                         color='#000000', fontweight='bold', labelpad=6)
            cb.ax.tick_params(labelsize=CB_FSZ * 0.80, colors='#000000', width=0.7)
            for tick in cb.ax.yaxis.get_ticklabels():
                tick.set_color('#000000'); tick.set_fontweight('bold')
            cb.ax.text(0.5, 1.04, 'PASS', transform=cb.ax.transAxes,
                       fontsize=CB_FSZ * 0.80, ha='center', va='bottom',
                       color='#1A5276', fontweight='bold', clip_on=False)
            cb.ax.text(0.5, -0.04, 'FAIL', transform=cb.ax.transAxes,
                       fontsize=CB_FSZ * 0.80, ha='center', va='top',
                       color='#C0392B', fontweight='bold', clip_on=False)

    out = f"{out_stem}_transformer_ig_6panel"
    _save_fig(fig, out, log)
    log(f"[transformer_ig_6panel] -> {out}.{{tiff,pdf,png}}")


def build_figure_cdr3_heatmap_psr_sec(
        ig_data_psr, ig_data_sec, out_stem, log,
        fig_width_cm=22.0, font_scale=1.0):
    """
    2-panel HCDR3 heatmap: PSR left, SEC right.
    - Independent colour scale per panel (PSR and SEC have different IG magnitudes)
    - Independent colourbar per panel
    - Same AA row order (natural ACDEFGHIKLMNPQRSTVWY) in both panels
    - Shared y-axis labels on left panel only
    """
    FS = font_scale
    _FULL = {'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe',
             'G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu',
             'M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg',
             'S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr'}

    def _get_mat(ig):
        if ig is None: return None, 0
        mat = ig['attr_cdr3'].mean(axis=0).T
        avg_len = int(np.median([len(s) for s in ig['hcdr3_seqs'] if s]))
        nc = min(max(avg_len + 2, 12), mat.shape[1])
        return mat[:, :nc], nc

    mat_psr, nc_psr = _get_mat(ig_data_psr)
    mat_sec, nc_sec = _get_mat(ig_data_sec)

    # Natural AA order — same for both panels
    row_order = np.arange(20)
    ylabels   = [f"{AMINO_ACIDS[i]}  ({_FULL[AMINO_ACIDS[i]]})" for i in row_order]

    # Independent vmax per panel
    vmax_psr = float(np.abs(mat_psr).max()) * 0.85 if mat_psr is not None else 0.3
    vmax_sec = float(np.abs(mat_sec).max()) * 0.85 if mat_sec is not None else 0.3

    n_panels = sum([mat_psr is not None, mat_sec is not None])
    N_AA   = 20
    nc_max = max(nc_psr or 0, nc_sec or 0, 1)
    cell_h = 0.30 * FS
    cell_w = 0.30 * FS
    FIG_H  = N_AA * cell_h + 2.5 * FS
    FIG_W  = nc_max * cell_w * n_panels + 6.5 * FS  # extra room for 2 colorbars

    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(FIG_W, FIG_H), sharey=True)
    if n_panels == 1: axes = [axes]
    fig.subplots_adjust(left=0.22, right=0.82, top=0.92, bottom=0.11, wspace=0.08)

    panel_defs = []
    if mat_psr is not None: panel_defs.append((mat_psr, nc_psr, 'a', 'PSR', vmax_psr))
    if mat_sec is not None: panel_defs.append((mat_sec, nc_sec, 'b', 'SEC', vmax_sec))

    im_handles = []
    for ax, (mat, nc, letter, label, vmax) in zip(axes, panel_defs):
        im = ax.imshow(mat[row_order, :], cmap=_HEATMAP_CMAP,
                       vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest',
                       extent=[-0.5, nc-0.5, 19.5, -0.5])
        im_handles.append((ax, im, vmax))

        step = 2 if nc > 16 else 1
        ticks = sorted(set(list(range(0, nc, step)) + [nc-1]))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t+1) for t in ticks],
                           fontsize=12.2*FS, color='#000000', fontweight='bold')
        ax.set_xlabel('HCDR3 position', fontsize=14.9*FS,
                      color='#000000', fontweight='bold', labelpad=5)
        ax.tick_params(axis='x', colors='#000000', length=3, width=0.8)
        ax.tick_params(axis='y', colors='#000000', length=3, width=0.8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#000000'); sp.set_linewidth(0.8)
        ax.text(-0.01, 1.025, letter, transform=ax.transAxes,
                fontsize=17.6*FS, fontweight='bold', color='#000000',
                va='bottom', ha='left', clip_on=False)
        ax.text(0.10, 1.025, label, transform=ax.transAxes,
                fontsize=16.2*FS, fontweight='bold', color='#000000',
                va='bottom', ha='left', clip_on=False)

    # Y-axis labels on left panel only
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels(ylabels, fontsize=12.2*FS, color='#000000',
                             fontfamily='monospace', fontweight='bold')
    axes[0].set_ylabel('Amino acid', fontsize=14.9*FS, color='#000000',
                       fontweight='bold', labelpad=6)

    # Independent colourbar per panel
    fig.canvas.draw()
    for ax, im, vmax in im_handles:
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x1 + 0.010, pos.y0, 0.018, pos.height])
        cb = fig.colorbar(im, cax=cb_ax)
        cb.set_label('Mean signed IG', fontsize=12.2*FS, color='#000000',
                     fontweight='bold', labelpad=6)
        cb.ax.tick_params(labelsize=10.8*FS, colors='#000000', width=0.7)
        for tick in cb.ax.yaxis.get_ticklabels():
            tick.set_color('#000000'); tick.set_fontweight('bold')
        cb.ax.text(0.5, 1.04, 'PASS', transform=cb.ax.transAxes,
                   fontsize=10.8*FS, ha='center', va='bottom',
                   color='#1A5276', fontweight='bold', clip_on=False)
        cb.ax.text(0.5, -0.04, 'FAIL', transform=cb.ax.transAxes,
                   fontsize=10.8*FS, ha='center', va='top',
                   color='#C0392B', fontweight='bold', clip_on=False)

    # Single-target (ig_data_sec is None) → "_cdr3_heatmap"
    # Dual-target (both present)          → "_cdr3_heatmap_psr_sec"
    suffix = "_cdr3_heatmap" if ig_data_sec is None else "_cdr3_heatmap_psr_sec"
    out = f"{out_stem}{suffix}"
    _save_fig(fig, out, log)
    log(f"[cdr3_heatmap] -> {out}.{{tiff,pdf,png}}")

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

    YTICK_FSZ = 8.5    # y-tick feature labels
    XTICK_FSZ = 8.0    # x-tick numbers
    XLABEL_FSZ= 8.5    # x-axis label
    TITLE_FSZ = 9.5    # panel letter
    CB_FSZ    = 7.0    # colourbar labels
    DOT_SIZE  = 8      # scatter dot size
    VH_TOP_N  = 7
    top_n     = 20

    ig_total  = 20 + VH_TOP_N + 2
    row_px    = max(top_n, ig_total) * 0.30 + 1.5
    FIG_W     = 16.0   # wider for larger labels
    FIG_H     = row_px * 2 + 1.5

    cmap_feat = _safe_cmap('RdBu_r')
    rng       = np.random.default_rng(42)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs_outer = GridSpec(2, 3, figure=fig,
                        width_ratios=[1.05, 1.05, 1.0],
                        wspace=0.62, hspace=0.28,
                        left=0.14, right=0.97,
                        top=0.96, bottom=0.04)

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

        # Row label — larger, bold black
        ax_rf.annotate(
            row_lbl, xy=(0, 0.5), xytext=(-0.42, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=15, fontweight='bold', color=row_col,
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
                jitter  = rng.uniform(-0.28, 0.28, size=len(sv_col))
                ax_rf.scatter(sv_col, row + jitter,
                              c=cmap_feat(norm_xv), s=DOT_SIZE, alpha=0.55,
                              linewidths=0, rasterized=True)
            ax_rf.axvline(0, color='#000000', lw=0.7)
            ax_rf.set_yticks(range(top_n))
            ax_rf.set_yticklabels(rf_labels, fontsize=YTICK_FSZ, color='#000000')
            ax_rf.invert_yaxis()
            ax_rf.tick_params(axis='x', labelsize=XTICK_FSZ, colors='#000000')
            ax_rf.grid(axis='x', alpha=0.18, lw=0.3)
            for s in ('top', 'right'): ax_rf.spines[s].set_visible(False)

        ax_rf.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                         fontsize=XLABEL_FSZ, labelpad=2, color='#000000')

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
        cb.set_ticklabels(['Low', 'High'], fontsize=CB_FSZ, color='#000000')
        cb.set_label('Feature value', fontsize=CB_FSZ, labelpad=1, color='#000000')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.tick_top()
        cb.ax.tick_params(width=0.4, length=1.5, labelsize=CB_FSZ, colors='#000000')

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
                jitter  = rng.uniform(-0.28, 0.28, size=len(sv_col))
                ax_xgb.scatter(sv_col, row + jitter,
                               c=cmap_feat(norm_xv), s=DOT_SIZE, alpha=0.55,
                               linewidths=0, rasterized=True)
            ax_xgb.axvline(0, color='#000000', lw=0.7)
            ax_xgb.set_yticks(range(len(rf_labels)))
            ax_xgb.set_yticklabels(rf_labels, fontsize=YTICK_FSZ, color='#000000')
            ax_xgb.invert_yaxis()
            ax_xgb.tick_params(axis='x', labelsize=XTICK_FSZ, colors='#000000')
            ax_xgb.grid(axis='x', alpha=0.18, lw=0.3)
            for s in ('top', 'right'): ax_xgb.spines[s].set_visible(False)

        ax_xgb.set_xlabel('SHAP value\n(← FAIL  |  PASS →)',
                          fontsize=XLABEL_FSZ, labelpad=2, color='#000000')

        # Transformer IG — single combined axis
        _render_ig_combined(ax_ig, ig_data, rng,
                            vh_top_n=VH_TOP_N, fsz_y=YTICK_FSZ, dot_size=DOT_SIZE)

    # Panel titles via fig.text — filter name + lifted position
    fig.canvas.draw()
    TITLE_LIFT_2ROW = 0.030
    for row_idx, (panels, row_r) in enumerate(zip(panel_abc, [r1, r2])):
        lbl_a, lbl_b, lbl_c = panels
        _filt = row_r['target'].upper().split('_')[0]
        ax_ig = ig_axes[row_idx]
        ax_rf_row  = fig.axes[row_idx * 3]
        ax_xgb_row = fig.axes[row_idx * 3 + 1]
        for ax, letter, line1, line2 in [
            (ax_rf_row,  lbl_a, f'RF-{_filt}',
                                '(biophysical features)'),
            (ax_xgb_row, lbl_b, f'XGBoost-{_filt}',
                                ''),
            (ax_ig,      lbl_c,
             f'Transformer onehot-{_filt}',
             'IG per amino acid'),
        ]:
            pos = ax.get_position()
            x0, y1 = pos.x0, pos.y1
            ty = y1 + TITLE_LIFT_2ROW
            fig.text(x0, ty, letter,
                     fontsize=TITLE_FSZ, fontweight='bold', color='#000000',
                     va='bottom', ha='left', transform=fig.transFigure)
            fig.text(x0 + 0.018, ty, line1,
                     fontsize=TITLE_FSZ * 0.85, fontweight='bold', color='#000000',
                     va='bottom', ha='left', transform=fig.transFigure)
            if line2:
                fig.text(x0 + 0.018, ty - 0.009, line2,
                         fontsize=TITLE_FSZ * 0.65, va='top', ha='left',
                         color='#000000', transform=fig.transFigure)

    _save_fig(fig, f"{out_stem}_2row", log)


def main():
    ap = argparse.ArgumentParser(
        description="DELPHI — generate interpretability figures",
        allow_abbrev=False)   # exact option names only — --model never matches --model-dir
    ap.add_argument('--db',             default=None,
                    help="Training database (.xlsx or .csv), "
                         "e.g. data/ipi_psr_trainset.xlsx. "
                         "Used to locate models by db_stem. Optional when "
                         "--model-path is given (db_stem is parsed from the "
                         "checkpoint filename instead).")
    ap.add_argument('--model-path',     default=None,
                    help="Direct path to a FINAL_*.pt/.pkl checkpoint. "
                         "Bypasses --db model lookup. The model-dir and db_stem "
                         "are parsed from the filename, so this produces the "
                         "same output as the equivalent --db invocation. "
                         "Filename must follow "
                         "FINAL_{target}_{lm}_{model}_{db_stem}.{pt|pkl}.")
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

    # ── Mode C: single-architecture interpretability ─────────────────────
    ap.add_argument('--model',          default=None,
                    choices=['transformer_onehot', 'rf', 'xgboost'],
                    help="[Mode C] Restrict interpretability to ONE architecture. "
                         "Use with --predict for single-model prediction + "
                         "interpretability. For rf/xgboost the IG heatmap panel "
                         "is skipped (IG is transformer-only). If omitted, all "
                         "three architectures are used (Mode B).")
    ap.add_argument('--lm',             default=None,
                    help="[Mode C] Feature mode for the single --model. "
                         "'onehot' for transformer_onehot; 'biophysical' or "
                         "'kmer' for rf/xgboost. Defaults per architecture if omitted.")

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
    ap.add_argument('--ig-baseline',      default='uniform',
                    choices=['uniform', 'zero', 'mean'],
                    help="IG baseline: 'uniform' (padding-safe length-matched amino-acid "
                         "reference, default), 'zero' (legacy; unsafe when zero rows define "
                         "padding), or "
                         "'mean' (average antibody baseline — suppresses attribution "
                         "at conserved positions like CAR anchor). "
                         "Use 'mean' as a sensitivity analysis. (default: uniform)")
    ap.add_argument('--ig-steps',         type=int, default=200,
                    help="IG integration steps (default: 200)")
    ap.add_argument('--n-pairs',          type=int, default=20,
                    help="[dual-target only] Number of per-antibody figures "
                         "for IG, XGBoost-SHAP and RF-SHAP waterfall sets. "
                         "Each pair = 1 FAIL antibody + 1 PASS antibody. "
                         "(default: 20)")
    ap.add_argument('--n-antibodies',     type=int, default=20,
                    help="[single-target, Mode B/C] Number of per-antibody "
                         "waterfall + CDR3 mutagenesis figures to generate. "
                         "N = N PASS + N FAIL antibodies (no predictive-score "
                         "filter). 0 = ALL antibodies. (default: 20)")
    ap.add_argument('--csv-exist', action='store_true', default=False,
                    help="Skip model loading and SHAP/IG computation entirely. "
                         "Load all data from previously saved CSV and NPZ files "
                         "and regenerate all figures. "
                         "Requires --db/--target (and --db2/--target2 for combined figures) "
                         "to locate the output directories. "
                         "Use this when you only want to update figure styling "
                         "without waiting for SHAP/IG to rerun. "
                         "(default: False)")
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

    # ── --model-path: derive model-dir + db_stem from the checkpoint name ─
    # Lets users pass a direct checkpoint instead of --db. The model lookup
    # (_find_final) reconstructs FINAL_{target}_{lm}_{model}_{db_stem}{ext},
    # so we parse db_stem from the filename and point model-dir at its folder.
    # This makes `--model-path X` produce identical output to the matching
    # `--db <db_stem>.xlsx --model-dir <folder>` invocation.
    mp = getattr(args, 'model_path', None)
    if mp:
        import os as _os
        if not _os.path.exists(mp):
            ap.error(f"--model-path not found: {mp}")
        mp_dir  = _os.path.dirname(_os.path.abspath(mp))
        mp_name = _os.path.splitext(_os.path.basename(mp))[0]
        # Expected stem: FINAL_{target}_{lm}_{model}_{db_stem}
        _model_for_parse = getattr(args, 'model', None) or 'transformer_onehot'
        _lm_for_parse    = getattr(args, 'lm', None)
        if _lm_for_parse is None:
            _lm_for_parse = ('onehot' if _model_for_parse == 'transformer_onehot'
                             else 'biophysical')
        prefix = f"FINAL_{args.target}_{_lm_for_parse}_{_model_for_parse}_"
        if mp_name.startswith(prefix):
            db_stem = mp_name[len(prefix):]
        else:
            # Fallback: take the trailing token group after the model name
            db_stem = mp_name.split(f"_{_model_for_parse}_")[-1] \
                      if f"_{_model_for_parse}_" in mp_name else mp_name
        # Point the lookup at this checkpoint's folder, and synthesise a --db
        # whose stem matches db_stem so Path(db).stem == db_stem.
        args.model_dir = mp_dir
        if not args.db:
            args.db = _os.path.join(mp_dir, f"{db_stem}.xlsx")
        print(f"[model-path] checkpoint   : {mp}")
        print(f"[model-path] → model-dir  : {mp_dir}")
        print(f"[model-path] → db_stem    : {db_stem}")

    # ── Require --db OR --model-path ──────────────────────────────────────
    if not args.db and not mp:
        ap.error("one of --db or --model-path is required")

    # ── Mode C: map --model/--lm onto the per-architecture lm fields ──────
    # When --model is given, set that architecture's lm from --lm (or a sane
    # default) so _find_final() locates the right checkpoint. The single
    # architecture is recorded in args.single_model for the render path.
    args.single_model = getattr(args, 'model', None)
    if args.single_model:
        _lm = getattr(args, 'lm', None)
        if args.single_model == 'transformer_onehot':
            args.transformer_lm = _lm or 'onehot'
        elif args.single_model == 'rf':
            args.rf_lm = _lm or 'biophysical'
        elif args.single_model == 'xgboost':
            args.xgb_lm = _lm or 'biophysical'

    sys.exit(run(args))


if __name__ == '__main__':
    main()
