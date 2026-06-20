#!/usr/bin/env python3
# delphi.py
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Deep End-to-end Learning Platform for antibody developability
#          with High Interpretability
#
# Module      : delphi_interpretability.py
# Description : Nature Biotechnology interpretability figure generator.
#               Computes SHAP (RF, XGBoost) and Integrated Gradients
#               (Transformer onehot) attributions and renders 5-panel
#               Extended-Data-style figures. Models are resolved via
#               config/model_registry.yaml or explicit --model-path flags.
#               --db is optional: if omitted, models are found via
#               --model-id args or registry (target + lm + model).
#               Applicable to any binary antibody property label
#               (PSR, SEC, HIC, AC-SINS, viscosity, expression, ...).
# Author      : Hoan Nguyen, PhD
# Company     : Institute for Protein Innovation (IPI)
# Date        : 2026-05
# Version     : 1.0.0
#
# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMANDS  (mutually exclusive — pick one per run)
# ══════════════════════════════════════════════════════════════════════════════
#
#   --build-dataset FILE   Balance / curate raw training data
#                          (calls utils/build_balanced_dataset_v4.py)
#   --kfold        N       Cross-validate model on training database
#   --train                Train final model on full training database
#   --predict      FILE    Score new antibodies with a trained model
#   --correlate    FILE+   Correlate IPI scores against experimental assays
#                          (calls utils/developability_correlation.py)
#   --build-embedding FILE Pre-compute PLM embeddings for a database
#   --split-dataset        Split --db into group-stratified train + val files
#
# ══════════════════════════════════════════════════════════════════════════════
# MODELS  (--model)
# ══════════════════════════════════════════════════════════════════════════════
#
#   transformer_lm      Transformer + PLM embeddings  (best AUC)
#   transformer_onehot  Transformer + one-hot sequences (no PLM needed)
#   rf                  Random Forest (fast, SHAP, CDR3 mutagenesis)
#   xgboost             XGBoost
#   cnn                 CNN
#
# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS  (--lm)
# ══════════════════════════════════════════════════════════════════════════════
#
#   biophysical       charge, pI, hydrophobicity, R/K/W counts (26 dims)
#   kmer              1-mer + 2-mer AA frequencies (~440 dims)
#   onehot            VH+VL one-hot position encoding
#   onehot_vh         VH one-hot only
#   onehot_cdr3       HCDR3 one-hot only
#   ablang            ABlang2 480-dim        pip install ablang2
#   antiberty         AntiBERTy 512-dim      pip install antiberty
#   antiberta2        AntiBERTa2 1024-dim    pip install transformers
#   antiberta2-cssp   AntiBERTa2-CSSP 1024-dim
#   igbert            IgBERT 1024-dim
#
# ══════════════════════════════════════════════════════════════════════════════
# TRAINING MODES  (transformer_lm)
# ══════════════════════════════════════════════════════════════════════════════
#
#   MODE 1  Frozen embeddings  (DEFAULT, recommended)
#   ─────────────────────────────────────────────────────────────────────────
#   sequences → PLM (frozen) → pre-computed .emb.csv → classifier trains
#   PLM weights: never updated  |  Trained: ~200k classifier params only
#   Best for: n < 10,000  |  CPU  |  same domain as pretraining
#
#   MODE 2  PLM layer unfreezing  (--train --finetune_plm)
#   ─────────────────────────────────────────────────────────────────────────
#   sequences → PLM (top layers update via backprop) → classifier trains
#   PLM weights: top N layers updated  |  Trained: ~20M params
#   Best for: n > 20,000  |  GPU  |  domain-shifted sequences
#
#   MODE 3  LoRA  (--train --finetune_plm --peft lora)  [RECOMMENDED]
#   ─────────────────────────────────────────────────────────────────────────
#   sequences → PLM (W frozen, only LoRA A×B updated) → classifier trains
#   PLM weights: W NEVER changes, only low-rank A×B matrices (~400k params)
#   Best for: n > 1,000  |  CPU feasible  |  low forgetting risk
#
#   LEVEL 2  Collaborator fine-tune  (--finetune --pretrained path.pt)
#   ─────────────────────────────────────────────────────────────────────────
#   Load YOUR pretrained .pt → fine-tune classifier on THEIR small dataset
#   Best for: 50–2,000 new antibodies from a collaborator lab
#
# ══════════════════════════════════════════════════════════════════════════════
# CDR3 CONVENTION  (2026)
# ══════════════════════════════════════════════════════════════════════════════
#
#   CDR3 does NOT include the leading framework cysteine (Kabat/IMGT pos 104).
#   Correct CDR3 starts with A-R (Ala-Arg germline anchor), NOT C-A-R.
#
#   Example:
#     HSEQ CDR3 region:  ...C | ARGGPGYAVFDY | WG...
#                             ^--- CDR3 starts here (not at C)
#
#   strip_cdr3_c_prefix() removes the leading 'C' if present.
#   This replaces the old fix_cdr3_c_prefix() which incorrectly ADDED 'C'.
#
#   All transformer_onehot and RF/XGBoost models must be retrained after
#   this change as CDR3 sequences are 1 residue shorter.
#
# ══════════════════════════════════════════════════════════════════════════════
# TYPICAL WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
#
#   Step 1  Build a balanced training set (optional but recommended)
#   Step 2  Cross-validate to find best hyperparameters + epoch count
#   Step 3  Train final model on full dataset  (auto-registered in model_registry.yaml)
#   Step 4  Predict on new cohort  (registry resolves model automatically)
#   Step 5  Correlate predictions against experimental assays
#
# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY  (config/model_registry.yaml)
# ══════════════════════════════════════════════════════════════════════════════
#
#   # List all registered models
#   python delphi.py --list-models
#
#   # Filter by target, lm, model, or db
#   python delphi.py --list-models --target psr_filter --lm onehot
#
#   # Predict using registry lookup (no --db needed)
#   python delphi.py --predict data/new_cohort.xlsx \
#       --target psr_filter --lm onehot --model transformer_onehot
#
#   # Predict using a specific model_id (any checkpoint, any name)
#   python delphi.py --predict data/new_cohort.xlsx \
#       --model_id FINAL_psr_filter_onehot_transformer_onehot_ipi_psr.pt
#
#   # To register a best_fold checkpoint manually:
#   # 1. Open config/model_registry.yaml
#   # 2. Add an entry with type: best_fold, point model_path to the BEST_*.pt file
#   # 3. The model_id can be any name you want
#
# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1 — build-dataset  (curate + balance raw data)
# ══════════════════════════════════════════════════════════════════════════════
#
#   # Recommended: combined strategy (CDR3 diversity + OOF confidence)
#   python delphi.py --build-dataset data/ipi_psr_raw.xlsx \
#       --target psr_filter --strategy combined --min-total 6000
#
#   # Cluster-only (fast, diversity only)
#   python delphi.py --build-dataset data/ipi_psr_raw.xlsx \
#       --target psr_filter --strategy cluster --cluster 0.8
#
#   # kmer_consensus (OOF confidence only, stricter threshold)
#   python delphi.py --build-dataset data/ipi_psr_raw.xlsx \
#       --target psr_filter --strategy kmer_consensus --min-prob 0.7 --cv 3
#
#   Output: data/ipi_psr_raw_psr_filter_balanced.xlsx
#           data/ipi_psr_raw_psr_filter_imbalanced_6000.xlsx
#           data/ipi_psr_raw_psr_filter_majority_rejected.xlsx
#
# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2 — RF biophysical  (fastest, interpretable, no PLM needed)
# ══════════════════════════════════════════════════════════════════════════════
#
#   python delphi.py --kfold 10 \
#       --target psr_filter --lm biophysical --model rf \
#       --db data/ipi_psr_trainset.xlsx \
#       --cost_fn 3.0
#
#   python delphi.py --train \
#       --target psr_filter --lm biophysical --model rf \
#       --db data/ipi_psr_trainset.xlsx
#
#   python delphi.py --predict data/new_cohort.xlsx \
#       --target psr_filter --lm biophysical --model rf \
#       --db data/ipi_psr_trainset.xlsx \
#       --mutagenesis 50
#
# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3 — transformer_onehot  (VH+VL+CDR3 one-hot, no PLM)
# ══════════════════════════════════════════════════════════════════════════════
#
#   python delphi.py --kfold 10 \
#       --target psr_filter --lm onehot --model transformer_onehot \
#       --db data/ipi_psr_trainset.xlsx --cluster 0.8
#
#   python delphi.py --train \
#       --target psr_filter --lm onehot --model transformer_onehot \
#       --db data/ipi_psr_trainset.xlsx
#
#   python delphi.py --predict data/new_cohort.xlsx \
#       --target psr_filter --lm onehot --model transformer_onehot \
#       --db data/ipi_psr_trainset.xlsx
#
# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4 — transformer_lm  (PLM embeddings, best AUC)
# ══════════════════════════════════════════════════════════════════════════════
#
#   # Pre-compute embeddings once (reused across kfold + train + predict)
#   python delphi.py --build-embedding data/ipi_psr_trainset.xlsx \
#       --lm igbert
#
#   python delphi.py --kfold 10 \
#       --target psr_filter --lm igbert --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx --cluster 0.8
#
#   python delphi.py --train \
#       --target psr_filter --lm igbert --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx
#
#   python delphi.py --predict data/new_cohort.xlsx \
#       --target psr_filter --lm igbert --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx
#
# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5 — correlate  (IPI score vs experimental assay)
# ══════════════════════════════════════════════════════════════════════════════
#
#   # Discover which score columns are in the prediction file
#   python delphi.py --correlate data/new_cohort_pred.xlsx \
#       --target psr_filter --list-scores
#
#   # Single assay
#   python delphi.py --correlate data/new_cohort_pred.xlsx \
#       --target psr_filter --assay psr_norm_smp
#
#   # Multiple assays + logit transform + custom title
#   python delphi.py --correlate data/new_cohort_pred.xlsx \
#       --target psr_filter \
#       --assay psr_norm_dna psr_norm_avidin psr_norm_smp psr_norm_mean \
#       --logit-trans \
#       --title "IPI PSR model vs normalised PSR panel" \
#       --out results/psr_correlation
#
#   # Multiple prediction files (e.g. compare LMs)
#   python delphi.py \
#       --correlate pred_igbert.xlsx pred_ablang.xlsx pred_antiberta2.xlsx \
#       --target psr_filter --assay psr_norm_mean \
#       --title "GDPa3 (n=80) — IPI PSR vs PROPHET-Ab polyreactivity" \
#       --out results/GDPa3_psr_correlation
#
#   # t-SNE on embedding space
#   python delphi.py --correlate data/new_cohort_pred.xlsx \
#       --target psr_filter --assay psr_norm_mean \
#       --tsne-source embedding \
#       --embedding-file data/ipi_psr_trainset.xlsx.igbert.emb.csv \
#       --out results/psr_tsne
#
# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6 — LoRA fine-tuning  (large dataset, GPU)
# ══════════════════════════════════════════════════════════════════════════════
#
#   python delphi.py --train --finetune_plm --peft lora \
#       --target psr_filter --lm igbert --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx \
#       --lora_r 8 --lora_alpha 16
#
# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 7 — collaborator fine-tune  (small external dataset)
# ══════════════════════════════════════════════════════════════════════════════
#
#   python delphi.py --finetune \
#       --pretrained build/pretrained_models/FINAL_psr_filter_igbert_transformer_lm_ipi_psr_trainset.pt \
#       --finetune_db data/collaborator_50ab.xlsx \
#       --target psr_filter --lm igbert --model transformer_lm \
#       --finetune_epochs 10 --finetune_lr 1e-6
#
# ══════════════════════════════════════════════════════════════════════════════
# ── Platform fixes — must be FIRST, before any imports ───────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# ─────────────────────────────────────────────────────────────────────────────
"""
Delphi — IPI Antibody Developability Prediction Platform
Production Version 2026

Subcommands: --build-dataset | --kfold | --train | --predict |
             --correlate | --build-embedding | --split-dataset

Changes (2026-05):
  * Renamed predict_developability.py -> delphi.py
  * --build-dataset: balance/curate raw training data (utils/build_balanced_dataset_v4.py)
  * --correlate: assay correlation plots (utils/developability_correlation.py)
  * strip_cdr3_c_prefix(): removes leading framework C from CDR3 (replaces fix_cdr3_c_prefix)
    CDR3 = AR... not CAR... The C belongs to VH framework (Kabat pos 104).
  * Auto-extract CDR3 from HSEQ when CDR3 column is missing (ANARCI -> regex fallback)
  * RF model: fixed kfold/predict/train calls to pass X_df + embeddings separately
"""

from config import MODEL_DIR, PREDICTION_DIR
import argparse
import os
import re
import warnings
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from embedding_generator import generate_embedding

import sys
import datetime


class _Tee:
    def __init__(self, log_path: str, mode: str = 'w'):
        self._terminal = sys.__stdout__
        self._log = open(log_path, mode, buffering=1, encoding='utf-8')
        print(f"[log] Writing to: {log_path}" + (" (append)" if mode == 'a' else ""), flush=True)

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        if self._log and not self._log.closed:
            self._log.close()

    @property
    def encoding(self):
        return self._terminal.encoding

    def isatty(self):
        return False


def _setup_logging(args, db_path: str) -> str:
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lm_tag  = args.lm.replace(",", "_")
    _LM_NORM = {"onehot_cdr3": "onehot_hcdr3"}
    args.lm  = _LM_NORM.get(args.lm, args.lm)
    lm_tag   = args.lm.replace(",", "_")
    db_stem = Path(db_path).stem if db_path else "default"

    if args.train:
        name     = f"train_{args.target}_{lm_tag}_{args.model}_{db_stem}_{ts}.log"
        log_path = os.path.join(MODEL_DIR, name)
        mode_w   = 'w'
    elif args.kfold:
        name     = f"kfold_{args.target}_{lm_tag}_{args.model}_{db_stem}_k{args.kfold}_{ts}.log"
        log_path = os.path.join(MODEL_DIR, name)
        mode_w   = 'w'
    elif args.predict:
        p        = Path(args.predict)
        name     = f"predict_{p.stem}_{args.target}_{lm_tag}_{args.model}_{db_stem}_{ts}.log"
        log_path = str(p.parent / name)
        mode_w   = 'w'
    elif args.build_embedding:
        p        = Path(args.build_embedding)
        name     = f"embedding_{p.stem}_{lm_tag}_{ts}.log"
        log_path = str(p.parent / name)
        mode_w   = 'w'
    elif getattr(args, 'split_dataset', False):
        name     = f"split_{Path(db_path).stem}_{ts}.log"
        log_path = str(Path(db_path).parent / name)
        mode_w   = 'w'
    elif getattr(args, 'build_dataset', None):
        p        = Path(args.build_dataset)
        name     = f"build_dataset_{p.stem}_{args.target}_{ts}.log"
        log_path = str(p.parent / name)
        mode_w   = 'w'
    elif getattr(args, 'correlate', None):
        p        = Path(args.correlate[0])
        name     = f"correlate_{args.target}_{ts}.log"
        log_path = str(p.parent / name)
        mode_w   = 'w'
    else:
        log_path = f"ipi_{ts}.log"
        mode_w   = 'w'

    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    sys.stdout = _Tee(log_path, mode=mode_w)
    ts_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[log] {'='*58}")
    print(f"[log] Started  : {ts_str}")
    print(f"[log] Command  : {' '.join(sys.argv)}")
    print(f"[log] Platform : {sys.platform}  Python {sys.version.split()[0]}")
    print(f"[log] {'='*58}")
    print()
    return log_path


from models.xgboost import XGBoostModel
from models.randomforest import RandomForestModel
from models.cnn import CNNModel
from models.transformer_onehot import TransformerOneHotModel
from models.transformer_lm import TransformerLMModel

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

REGISTRY_PATH = os.path.join('config', 'model_registry.yaml')

_REGISTRY_HEADER = """\
# ══════════════════════════════════════════════════════════════════════════════
# Delphi — Model Registry
# config/model_registry.yaml
# ══════════════════════════════════════════════════════════════════════════════
#
# REQUIRED FIELDS  (delphi cannot load the model without these)
#   model_id (key)  unique label — used by --model_id. Defaults to filename of
#                   model_path. Can be renamed freely. Must be unique.
#   trainset        path to the training database
#   target          label column  (psr_filter, sec_filter, ...)
#   lm              embedding     (onehot, igbert, biophysical, ...)
#   model           architecture  (transformer_onehot, transformer_lm, rf, ...)
#   model_path      actual .pt/.pkl file loaded. Independent of model_id.
#
# OPTIONAL FIELDS  (add as many or as few as you like)
#   type             full_train | best_fold
#   trained_at       timestamp
#   kfold_auc        mean AUC from --kfold
#   kfold_std        AUC std across folds
#   kfold_folds      number of folds
#   kfold_best_fold  fold with highest AUC
#   threshold        classification threshold (default 0.5 if omitted)
#   epochs           epochs used for --train
#   notes            free text — delphi never overwrites this field
#
# LOOKUP ORDER during  delphi --predict
#   1. --model_path               use directly, skip registry
#   2. --model_id NAME            find by model_id, load its model_path
#   3. --db + target + lm + model most recent full_train for that trainset
#   4. target + lm + model        most recent full_train across all trainsets
#   5. nothing found              error: run --train or add entry manually
#
# ══════════════════════════════════════════════════════════════════════════════

"""


def _registry_load() -> dict:
    """Load registry YAML. Returns {'models': {}} if file does not exist."""
    import yaml
    if not os.path.exists(REGISTRY_PATH):
        return {'models': {}}
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as _f:
        _data = yaml.safe_load(_f) or {}
    if 'models' not in _data:
        _data['models'] = {}
    return _data


def _registry_save(reg: dict) -> None:
    """Write registry to YAML with fixed header block."""
    import yaml
    os.makedirs(os.path.dirname(REGISTRY_PATH) or '.', exist_ok=True)
    _body = yaml.dump(reg, default_flow_style=False, sort_keys=False,
                      allow_unicode=True, width=120)
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as _f:
        _f.write(_REGISTRY_HEADER)
        _f.write(_body)


def _registry_add(
    model_id:        str,
    trainset:        str,
    target:          str,
    lm:              str,
    model:           str,
    model_path:      str,
    entry_type:      str   = 'full_train',
    kfold_auc:       float = None,
    kfold_std:       float = None,
    kfold_folds:     int   = None,
    kfold_best_fold: int   = None,
    threshold:       float = None,
    epochs:          int   = None,
    notes:           str   = '',
) -> str:
    """
    Append one entry to the registry after --train.
    If model_id already exists, appends a timestamp suffix to ensure uniqueness.
    Returns the model_id actually used.
    """
    import datetime as _dt
    reg    = _registry_load()
    models = reg['models']

    # Enforce uniqueness
    if model_id in models:
        _ts    = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        _stem  = Path(model_id).stem
        _suf   = Path(model_id).suffix
        model_id = f"{_stem}_{_ts}{_suf}"
        print(f"  [registry] duplicate model_id — using '{model_id}'")

    entry = {
        'trainset':   str(trainset),
        'target':     str(target),
        'lm':         str(lm),
        'model':      str(model),
        'model_path': str(model_path),
        'type':       entry_type,
        'trained_at': _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Optional metadata — only written when available
    if kfold_auc is not None:
        entry['kfold_auc'] = round(float(kfold_auc), 4)
    if kfold_std is not None:
        entry['kfold_std'] = round(float(kfold_std), 4)
    if kfold_folds is not None:
        entry['kfold_folds'] = int(kfold_folds)
    if kfold_best_fold is not None:
        entry['kfold_best_fold'] = int(kfold_best_fold)
    # Only write threshold when it differs meaningfully from 0.5
    if threshold is not None and abs(float(threshold) - 0.5) > 1e-4:
        entry['threshold'] = round(float(threshold), 4)
    if epochs is not None:
        entry['epochs'] = int(epochs)
    entry['notes'] = notes or ''

    models[model_id] = entry
    _registry_save(reg)

    _auc_str   = f"  kfold_auc={kfold_auc:.4f}" if kfold_auc is not None else ""
    _thr_str   = f"  threshold={threshold:.4f}" if (threshold and abs(threshold-0.5)>1e-4) else ""
    print(f"\n[registry] Registered: {model_id}{_auc_str}{_thr_str}")
    print(f"[registry] → {REGISTRY_PATH}")
    return model_id


def _registry_lookup(
    model_id: str  = None,
    trainset: str  = None,
    target:   str  = None,
    lm:       str  = None,
    model:    str  = None,
) -> dict:
    """
    Find one registry entry. Returns the entry dict (with 'model_id' injected)
    or None if not found.

    Priority:
      1. model_id given  →  exact key match
      2. trainset + target + lm + model  →  most recent full_train for that db
      3. target + lm + model (no trainset)  →  most recent full_train overall
    """
    import datetime as _dt
    reg    = _registry_load()
    models = reg.get('models', {})
    if not models:
        return None

    # 1. Exact model_id lookup
    if model_id:
        entry = models.get(model_id)
        if entry:
            return {**entry, 'model_id': model_id}
        print(f"  [registry] model_id '{model_id}' not found in registry")
        return None

    # 2 + 3. Filter by fields
    db_stem = Path(trainset).stem if trainset else None

    candidates = []
    for mid, entry in models.items():
        if target and entry.get('target') != target:
            continue
        if lm and entry.get('lm') != lm:
            continue
        if model and entry.get('model') != model:
            continue
        if db_stem:
            _e_stem = Path(entry.get('trainset', '')).stem
            if _e_stem != db_stem:
                continue
        candidates.append((mid, entry))

    if not candidates:
        return None

    # Sort by trained_at descending — most recent first
    def _ts(pair):
        try:
            return _dt.datetime.strptime(
                pair[1].get('trained_at', '1970-01-01 00:00:00'),
                '%Y-%m-%d %H:%M:%S')
        except Exception:
            return _dt.datetime.min

    candidates.sort(key=_ts, reverse=True)
    best_id, best_entry = candidates[0]
    return {**best_entry, 'model_id': best_id}


def _registry_list(target: str = None, lm: str = None,
                   model: str = None, trainset: str = None) -> None:
    """Print registry entries, optionally filtered."""
    reg    = _registry_load()
    models = reg.get('models', {})
    if not models:
        print("[registry] Empty — no models registered yet. Run --train to add entries.")
        return

    db_stem = Path(trainset).stem if trainset else None
    W = 66
    print(f"\n{'═'*W}")
    print(f"  Delphi Model Registry")
    print(f"  {REGISTRY_PATH}")
    print(f"{'─'*W}")

    shown = 0
    for mid, entry in models.items():
        if target   and entry.get('target') != target:   continue
        if lm       and entry.get('lm')     != lm:       continue
        if model    and entry.get('model')  != model:    continue
        if db_stem:
            if Path(entry.get('trainset', '')).stem != db_stem: continue

        _auc = (f"  kfold_auc={entry['kfold_auc']:.4f}" if entry.get('kfold_auc') else "")
        _thr = (f"  threshold={entry['threshold']:.4f}" if entry.get('threshold')
                else "  threshold=0.5")
        print(f"  {mid}")
        print(f"    target={entry.get('target')}  lm={entry.get('lm')}  "
              f"model={entry.get('model')}  type={entry.get('type','?')}")
        print(f"    trained_at={entry.get('trained_at','?')}{_auc}{_thr}")
        print(f"    model_path={entry.get('model_path')}")
        if entry.get('notes'):
            print(f"    notes: {entry['notes']}")
        print()
        shown += 1

    if shown == 0:
        print("  No entries match the given filter.")
    print(f"  Total: {shown} / {len(models)} entries shown")
    print(f"{'═'*W}\n")


def get_default_db_path():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None
    files = [f for f in os.listdir(data_dir)
             if f.startswith("ipi_antibodydb") and f.endswith(".xlsx")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    return os.path.join(data_dir, files[0])


# ===========================================================================
# CDR3 EXTRACTION + PREFIX HANDLING
# ===========================================================================

def _extract_cdr3_single_anarci(hseq: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from anarci import anarci as _anarci
            res = _anarci([("s", hseq)], scheme="imgt", output=False)
        numbered_seqs = res[0]
        if not numbered_seqs or numbered_seqs[0] is None:
            return ""
        domains = numbered_seqs[0]
        if not domains:
            return ""
        numbering, chain_type = domains[0]
        return "".join(
            aa for (pos, ins), aa in numbering
            if 105 <= pos <= 117 and aa != "-"
        )
    except Exception:
        return ""


def _extract_cdr3_regex(hseq: str) -> str:
    hseq = hseq.upper().replace("-", "").replace(" ", "")
    for pat in [r"C([A-Z]{3,30}?)WG[A-Z]G", r"C([A-Z]{3,35}?)WG"]:
        m = list(re.finditer(pat, hseq))
        if m:
            return m[-1].group(1)
    return ""


def extract_cdr3_from_hseq(hseq_series: pd.Series, verbose: bool = True) -> pd.Series:
    n = len(hseq_series)
    cdr3s = []
    n_anarci = n_regex = n_fail = 0

    if verbose:
        print(f"\n  [CDR3] CDR3 column not found — extracting from {n:,} HSEQ sequences ...")
        print(f"  [CDR3] Method: ANARCI (IMGT 105-117) with regex fallback")

    for hseq in hseq_series:
        if not isinstance(hseq, str) or len(hseq.strip()) < 20:
            cdr3s.append("")
            n_fail += 1
            continue
        cdr3 = _extract_cdr3_single_anarci(hseq)
        if cdr3:
            n_anarci += 1
        else:
            cdr3 = _extract_cdr3_regex(hseq)
            if cdr3:
                n_regex += 1
            else:
                n_fail += 1
        cdr3s.append(cdr3)

    if verbose:
        print(f"  [CDR3] Done: ANARCI={n_anarci:,}  regex={n_regex:,}  failed={n_fail:,}")
        if n_fail > 0:
            pct = 100 * n_fail / max(n, 1)
            print(f"  [CDR3] WARNING: {n_fail:,} sequences ({pct:.1f}%) could not be extracted — CDR3 set to ''.")
        if n_regex > 0:
            print(f"  [CDR3] NOTE: {n_regex:,} sequences used regex fallback.")

    return pd.Series(cdr3s, index=hseq_series.index, name="CDR3")


def _ensure_cdr3(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if "HSEQ" not in df.columns:
        if verbose:
            print("  [CDR3] WARNING: No HSEQ column — cannot extract CDR3.")
        return df

    df = df.copy()
    _nan_like = {"nan", "none", "null", "n/a", "na", ""}

    if "CDR3" not in df.columns:
        if verbose:
            print("  [CDR3] 'CDR3' column not found.")
        df["CDR3"] = ""
    else:
        df["CDR3"] = df["CDR3"].fillna("").astype(str)
        df.loc[df["CDR3"].str.strip().str.lower().isin(_nan_like), "CDR3"] = ""

    empty_mask = df["CDR3"].str.len() == 0
    n_missing  = empty_mask.sum()

    if n_missing > 0:
        if verbose and n_missing < len(df):
            print(f"  [CDR3] {n_missing:,} rows have empty CDR3 — extracting from HSEQ.")
        extracted = extract_cdr3_from_hseq(df.loc[empty_mask, "HSEQ"], verbose=verbose)
        df.loc[empty_mask, "CDR3"] = extracted.values

    return df


def strip_cdr3_c_prefix(df: pd.DataFrame, cdr3_col: str = "CDR3",
                        verbose: bool = True) -> pd.DataFrame:
    """
    Remove the leading 'C' (VH framework cysteine, Kabat/IMGT pos 104) from
    CDR3 sequences if present.

    The conserved cysteine is NOT part of CDR3. Correct CDR3 starts with A-R
    (Ala-Arg germline anchor), not C-A-R.

    Example:
        CARGGPGYAVFDY  ->  ARGGPGYAVFDY   (leading C removed)
        ARGGPGYAVFDY   ->  ARGGPGYAVFDY   (unchanged — no leading C)

    This function replaces the old fix_cdr3_c_prefix() which incorrectly
    ADDED a 'C' prefix. All models trained after this change will use CDR3
    sequences that are 1 residue shorter and correctly numbered:
        Position 1 = A (Ala)   — first true CDR3 residue
        Position 2 = R (Arg)   — conserved germline anchor
        Position 3+ = variable — developability-relevant positions
    """
    if cdr3_col not in df.columns:
        return df

    df   = df.copy()
    mask = (df[cdr3_col].str.len() > 1) & (df[cdr3_col].str.startswith("C"))
    n    = mask.sum()

    if n > 0 and verbose:
        print(f"\n  [CDR3-C] Removing framework 'C' from {n:,} CDR3 sequence(s).")
        print(f"  [CDR3-C] The leading C is the VH framework cysteine (Kabat pos 104), not CDR3.")
        print(f"  [CDR3-C] Correct CDR3 starts with AR, e.g. ARGGPGYAVFDY (not CARGGPGYAVFDY).")
        shown = 0
        for bc, row in df[mask].iterrows():
            old = row[cdr3_col]
            print(f"    BARCODE={bc}  '{old}'  ->  '{old[1:]}'")
            shown += 1
            if shown >= 5:
                break
        if n > 5:
            print(f"    ... and {n - 5:,} more")
    elif verbose and n == 0:
        print(f"  [CDR3-C] No leading 'C' found — CDR3 sequences already correctly formatted.")

    df.loc[mask, cdr3_col] = df.loc[mask, cdr3_col].str[1:]
    return df


def fix_cdr3_c_prefix(df: pd.DataFrame, cdr3_col: str = "CDR3",
                      verbose: bool = True) -> pd.DataFrame:
    """
    Deprecated — now calls strip_cdr3_c_prefix() instead.

    Old behaviour was to ADD a 'C' prefix if missing, which was incorrect.
    The leading C belongs to VH framework, not CDR3. This wrapper is kept
    for backward compatibility only — it now removes the C instead.
    """
    return strip_cdr3_c_prefix(df, cdr3_col=cdr3_col, verbose=verbose)


# ===========================================================================
# FILE I/O HELPERS
# ===========================================================================

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if path.lower().endswith('.csv'):
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    print(f"  Saved: {path}  ({len(df):,} rows)")


def read_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _split_embedding(emb_path, train_barcodes, val_barcodes, train_out, val_out):
    emb = pd.read_csv(emb_path, index_col=0)
    emb.index = emb.index.astype(str).str.strip()
    db_base = str(Path(emb_path).name)
    _parts  = db_base.split(".")
    emb_idx = next((i for i, p in enumerate(_parts) if p == "emb"), None)
    if emb_idx is None:
        print(f"  [emb-split] Cannot parse LM tag from {db_base} — skipping")
        return
    lm_tag = ".".join(_parts[emb_idx-1:])
    tr_out = f"{train_out}.{lm_tag}"
    va_out = f"{val_out}.{lm_tag}"
    tr_bcs = [b for b in train_barcodes.astype(str) if b in emb.index]
    va_bcs = [b for b in val_barcodes.astype(str)   if b in emb.index]
    n_missing_tr = len(train_barcodes) - len(tr_bcs)
    n_missing_va = len(val_barcodes)   - len(va_bcs)
    if n_missing_tr or n_missing_va:
        print(f"  [emb-split] WARNING: {n_missing_tr} train + {n_missing_va} val BARCODEs not found — skipped")
    emb.loc[tr_bcs].to_csv(tr_out)
    emb.loc[va_bcs].to_csv(va_out)
    print(f"  [emb-split] {os.path.basename(tr_out)}  ({len(tr_bcs):,} rows)")
    print(f"  [emb-split] {os.path.basename(va_out)}  ({len(va_bcs):,} rows)")


def _find_embedding_files(db_path: str) -> list:
    import glob
    return sorted(glob.glob(f"{db_path}.*.emb.csv"))


def _align_embedding(df: pd.DataFrame, embedding: pd.DataFrame, context: str = "") -> tuple:
    tag = f"[{context}] " if context else ""
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    embedding = embedding.copy()
    embedding.index      = embedding.index.astype(str).str.strip()
    embedding.index.name = "BARCODE"
    _sample = embedding.index[:5].tolist()
    if all(str(v).isdigit() for v in _sample):
        print(f"  {tag}WARNING: embedding index looks numeric ({_sample}) — expected BARCODE strings.")
    merged = df.join(embedding, how="inner")
    n_missing = len(df) - len(merged)
    if len(merged) == 0:
        raise ValueError(
            f"{tag}No overlapping BARCODEs between data ({len(df):,} rows) "
            f"and embedding ({len(embedding):,} rows)."
        )
    if n_missing > 0:
        print(f"  {tag}WARNING: {n_missing:,} / {len(df):,} rows ({n_missing/len(df):.1%}) have no embedding — excluded.")
    emb_cols = embedding.columns.tolist()
    return merged[emb_cols], merged.drop(columns=emb_cols)


def split_and_save(db_path: str, split: float = 0.8,
                   cluster_thresh: float = 0.8,
                   cluster_col: str = "CDR3",
                   label_col: str = "psr_filter") -> tuple:
    from sklearn.model_selection import StratifiedGroupKFold
    from utils.clustering import greedy_clustering_by_levenshtein

    df = read_dataframe(db_path)
    p  = Path(db_path)
    print(f"\n[split] {p.name}  ({len(df):,} rows)  split={split:.0%}/{1-split:.0%}  cluster_col={cluster_col}")

    _cc = cluster_col.upper()
    if _cc == 'HSEQ':  _cc = 'VH'
    if _cc == 'HVHVL': _cc = 'VHVL'

    col_map = {'CDR3': f'HCDR3_CLUSTER_{cluster_thresh}',
               'VH':   f'VH_CLUSTER_{cluster_thresh}',
               'VHVL': f'VHVL_CLUSTER_{cluster_thresh}'}
    seq_map = {'CDR3': 'CDR3', 'VH': 'HSEQ', 'VHVL': 'HSEQ'}
    grp_col = col_map.get(_cc, f'HCDR3_CLUSTER_{cluster_thresh}')
    seq_col = seq_map.get(_cc, 'CDR3')

    if grp_col not in df.columns:
        if seq_col not in df.columns:
            raise ValueError(f"Sequence column '{seq_col}' not found.")
        print(f"[split] Computing {grp_col} (threshold={cluster_thresh}) ...")
        seqs = (df['HSEQ'].fillna('').astype(str) + '_' +
                df['LSEQ'].fillna('').astype(str)).tolist() if _cc == 'VHVL' \
               else df[seq_col].fillna('').astype(str).tolist()
        df[grp_col] = greedy_clustering_by_levenshtein(seqs, cluster_thresh)
        n_clust = df[grp_col].nunique()
        print(f"[split] {n_clust:,} clusters  (mean {len(df)/n_clust:.1f} sequences/cluster)")
        try:
            _df_save = read_dataframe(db_path)
            _df_save[grp_col] = df[grp_col].values
            save_dataframe(_df_save, db_path)
            print(f"[split] Saved {grp_col} → {os.path.basename(db_path)}")
        except Exception as _e:
            print(f"[split] WARNING: could not save {grp_col} back to {os.path.basename(db_path)}: {_e}")
    else:
        print(f"[split] Using existing {grp_col}  ({df[grp_col].nunique():,} clusters)")

    _lc     = label_col if label_col in df.columns else None
    has_lbl = _lc is not None

    if has_lbl:
        _lbl_mask = df[_lc].notna()
        n_unl     = (~_lbl_mask).sum()
        if n_unl > 0:
            print(f"[split] {_lbl_mask.sum():,} labelled + {n_unl:,} unlabelled rows — ALL preserved")
    else:
        print(f"[split] WARNING: '{label_col}' not found — splitting by cluster only")
        _lbl_mask = pd.Series([True] * len(df), index=df.index)

    df_lbl  = df[_lbl_mask].copy()
    y_arr   = df_lbl[_lc].values.astype(int) if has_lbl else np.zeros(len(df_lbl), int)
    groups  = df_lbl[grp_col].values

    n_splits = max(2, round(1.0 / (1.0 - split)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_split, best_diff = None, float('inf')
    for tr, va in sgkf.split(np.arange(len(y_arr)), y_arr, groups):
        diff = abs(y_arr[va].mean() - y_arr.mean()) if has_lbl else 0.0
        if diff < best_diff: best_diff, best_split = diff, (tr, va)

    tr_idx, va_idx = best_split
    tr_clusters = set(groups[tr_idx])
    va_clusters = set(groups[va_idx])
    leaked = tr_clusters & va_clusters
    print(f"[split] {'[WARN] ' + str(len(leaked)) + ' cluster(s) leaked' if leaked else '✓ No cluster leakage'}")

    if not _lbl_mask.all():
        df_unl = df[~_lbl_mask].copy()
        unl_in_train = df_unl[grp_col].apply(lambda g: g in tr_clusters or g not in va_clusters)
        df_unl_train = df_unl[unl_in_train]
        df_unl_val   = df_unl[~unl_in_train]
    else:
        df_unl_train = df_unl_val = pd.DataFrame(columns=df.columns)

    train_df = pd.concat([df_lbl.iloc[tr_idx], df_unl_train], ignore_index=True)
    val_df   = pd.concat([df_lbl.iloc[va_idx],   df_unl_val], ignore_index=True)

    _pos_tr = f"  pos={y_arr[tr_idx].mean():.1%}" if has_lbl else ""
    _pos_va = f"  pos={y_arr[va_idx].mean():.1%}" if has_lbl else ""
    print(f"[split] Train={len(train_df):,}{_pos_tr}  Val={len(val_df):,}{_pos_va}")

    ext       = p.suffix
    train_out = str(p.parent / f"{p.stem}_train{ext}")
    val_out   = str(p.parent / f"{p.stem}_val{ext}")
    save_dataframe(train_df.reset_index(drop=True), train_out)
    save_dataframe(val_df.reset_index(drop=True),   val_out)

    bc_col = 'BARCODE'
    if bc_col in df.columns:
        train_bcs = train_df[bc_col].astype(str)
        val_bcs   = val_df[bc_col].astype(str)
    else:
        train_bcs = pd.Index(train_df.index.astype(str))
        val_bcs   = pd.Index(val_df.index.astype(str))

    emb_files = _find_embedding_files(db_path)
    if emb_files:
        print(f"\n[split] Found {len(emb_files)} embedding file(s) — splitting by BARCODE ...")
        for emb_path in emb_files:
            lm_name = os.path.basename(emb_path).split('.emb.csv')[0].split('.')[-1]
            print(f"  [emb-split] {lm_name}: {os.path.basename(emb_path)}")
            try:
                _split_embedding(emb_path, train_bcs, val_bcs, train_out, val_out)
            except Exception as e:
                print(f"  [emb-split] WARNING: failed for {lm_name} — {e}")
    else:
        print(f"\n[split] No embedding files found for {p.name}")

    return train_out, val_out, train_df, val_df


# ===========================================================================
# LOAD DATA
# ===========================================================================

def load_data(db_path, lm="antiberta2", label_col="psr_filter"):
    print(f"\nLoading database : {os.path.basename(db_path)}")
    print(f"Target           : {label_col}")
    print(f"Embedding        : {lm}")

    df = pd.read_excel(db_path)
    print(f"Total rows       : {len(df):,}  |  columns: {len(df.columns)}")

    required = ["BARCODE", "HSEQ", "LSEQ", label_col]
    if lm in ("onehot", "onehot_vh"):
        required += ["CDR3"]

    if "HSEQ" in df.columns:
        df = _ensure_cdr3(df, verbose=True)
        df = strip_cdr3_c_prefix(df, verbose=True)   # ← strip C, not add C

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).set_index("BARCODE")
    print(f"After dropna     : {len(df):,} rows")

    _SEQ_ONLY_MODES = {"onehot", "onehot_vh", "onehot_cdr3", "onehot_hcdr3",
                       "biophysical", "kmer", "seq", "none"}

    if lm in _SEQ_ONLY_MODES:
        X    = df[["HSEQ", "LSEQ", "CDR3"]]
        y    = df[label_col].values
        data = df.copy()
        if lm not in ("onehot", "onehot_vh"):
            print(f"[load_data] lm='{lm}' — sequence-only mode, no PLM embedding loaded")
    else:
        possible = [f"{db_path}.{lm}.emb.csv"]
        emb_file = next((f for f in possible if os.path.exists(f)), None)
        if not emb_file:
            print(f"Embedding not found -> generating {lm}...")
            emb_file = generate_embedding(db_path, lm=lm)
        if not emb_file or not os.path.exists(str(emb_file or '')):
            _SEQ_ONLY = {"biophysical","kmer","onehot","onehot_vh",
                         "onehot_cdr3","onehot_hcdr3","none","seq"}
            _PLM_LIST = "ablang | antiberty | antiberta2 | antiberta2-cssp | igbert | igt5 | abmap"
            raise ValueError(
                f"\n[load_data] Cannot load embedding for lm='{lm}'.\n"
                f"  Embedding file not found and generation failed.\n"
                f"  If you intended a sequence-only model, use one of:\n"
                f"    --lm biophysical  --lm kmer  --lm onehot\n"
                f"  If you intended a PLM, supported values are:\n"
                f"    {_PLM_LIST}\n"
                f"  Note: --model rf/xgboost/cnn is NOT a valid --lm value.\n"
                f"  Example: --lm igbert --model rf")
        print(f"Embedding file   : {emb_file}")
        print(f"Embedding size   : {os.path.getsize(emb_file) / 1024 / 1024:.1f} MB")
        embedding = pd.read_csv(emb_file, index_col=0)
        print(f"Embedding shape  : {embedding.shape[0]:,} samples × {embedding.shape[1]:,} dims")
        X, data = _align_embedding(df, embedding, context="load_data")
        y = data[label_col].values

    print(f"Samples loaded   : {len(y):,}")
    print(f"Target stats     : mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")
    return X, data, y


# ── Universal CDR3 mutagenesis ────────────────────────────────────────────────

def _run_cdr3_mutagenesis(
        data, model, model_type, lm, db_stem,
        input_file, target, test_target,
        embeddings_fn=None, n_override=None):
    import traceback as _tb
    import matplotlib.pyplot as plt
    import matplotlib.colors as _mc
    import numpy as np

    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    _cfg = getattr(model, 'config', {}) if hasattr(model, 'config') else {}
    _mut_cfg  = _cfg.get('mutagenesis', {})
    _fmt      = _mut_cfg.get('format',   'tiff')
    _dpi      = _mut_cfg.get('pub_dpi',   300)
    _make_ppt = _mut_cfg.get('make_ppt',  True)
    _max_n    = _mut_cfg.get('max_samples', 50)
    if n_override is not None:
        _max_n = len(data) if n_override == 0 else int(n_override)

    _path     = Path(input_file)
    _stem     = f"{_path.stem}_{target}_{lm}_{model_type}_{db_stem}"
    _mut_dir  = str(_path.parent / f"{_stem}_mutagenesis")
    import os; os.makedirs(_mut_dir, exist_ok=True)

    _label_col = test_target if (test_target and test_target in data.columns) \
                 else (target if target in data.columns else None)

    _n = min(len(data), _max_n)
    if _n < len(data):
        print(f"[Mutagenesis] Limited to first {_n}/{len(data)} antibodies")

    _needs_emb = (hasattr(model, 'fb_') and model.fb_ is not None and
                  model.fb_.feat_cfg.get('embedding', False))

    if _needs_emb:
        print(f"\n[Mutagenesis] WARNING: PLM mode '{lm}' — heatmap may be flat.")
        print(f"  Recommended: --lm biophysical or --lm kmer for meaningful mutagenesis.")
        try:
            from embedding_generator import EmbeddingGenerator as _EmbGen
            _emb_gen = _EmbGen(lm=lm)
            def _embeddings_fn(df_row):
                _hseq = str(df_row.iloc[0].get('HSEQ', '') or '')
                _lseq = str(df_row.iloc[0].get('LSEQ', '') or '')
                _bc   = str(df_row.index[0])
                _emb  = _emb_gen.embed_single(_hseq, _lseq, barcode=_bc)
                return _emb.reshape(1, -1).astype('float32')
        except Exception as _ee:
            try:
                import tempfile, os as _os, pandas as _pd
                from embedding_generator import generate_embedding as _gen_emb
                def _embeddings_fn(df_row):
                    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as _tf:
                        _tmp = _tf.name
                    try:
                        df_row.reset_index().to_excel(_tmp, index=False)
                        _emb_csv = _gen_emb(_tmp, lm=lm)
                        _emb_df  = _pd.read_csv(_emb_csv, index_col=0)
                        return _emb_df.values.astype('float32')
                    finally:
                        _os.unlink(_tmp)
                        for _ext in ['.emb.csv', f'.{lm}.emb.csv']:
                            _c = _tmp + _ext
                            if _os.path.exists(_c): _os.unlink(_c)
            except Exception:
                _embeddings_fn = None
    else:
        _embeddings_fn = None

    print(f"[Mutagenesis] {model_type.upper()} | {lm} | {_n} antibodies | output → {_mut_dir}/")

    task = getattr(model, 'task', 'classification')
    _saved = []

    for _s in range(_n):
        _row  = data.iloc[_s]
        _bc   = str(data.index[_s])
        _cdr3 = str(_row.get('CDR3', '') or '').upper().replace('-', '')
        if not _cdr3:
            print(f"  [{_s+1}/{_n}] {_bc}: CDR3 missing — skipped")
            continue

        _n_pos = len(_cdr3)
        _n_aa  = len(AMINO_ACIDS)

        _actual = None
        if _label_col:
            try:
                _v = _row.get(_label_col)
                if _v is not None and str(_v) not in ('', 'nan'):
                    _actual = int(float(_v))
            except Exception:
                pass

        print(f"  [{_s+1}/{_n}] {_bc}  CDR3={_cdr3}  ({_n_pos}×{_n_aa} mutants)  "
              + (f"Actual={'PASS' if _actual==1 else 'FAIL'}" if _actual is not None else ""))

        _wt_df = data.iloc[[_s]].copy()
        try:
            _wt_score = float(_predict_single(model, model_type, _wt_df,
                                              _embeddings_fn, task))
        except Exception as _e:
            print(f"    WT score failed: {_e}"); _wt_score = float('nan')

        _mat = np.full((_n_aa, _n_pos), np.nan, dtype=np.float32)
        _vh  = str(_row.get('HSEQ', '') or '')
        _cdr3_start = _vh.find(_cdr3) if _vh else -1

        _first_err = True
        for _pi in range(_n_pos):
            for _ai, _mut_aa in enumerate(AMINO_ACIDS):
                _mcdr3 = _cdr3[:_pi] + _mut_aa + _cdr3[_pi+1:]
                _mrow  = _row.to_dict()
                _mrow['CDR3'] = _mcdr3
                if _cdr3_start >= 0:
                    _mrow['HSEQ'] = (_vh[:_cdr3_start] + _mcdr3 +
                                     _vh[_cdr3_start + _n_pos:])
                _mut_df = pd.DataFrame([_mrow], index=[_bc])
                try:
                    _mat[_ai, _pi] = float(_predict_single(
                        model, model_type, _mut_df, _embeddings_fn, task))
                except Exception as _me:
                    if _first_err:
                        import traceback as _tb
                        print(f"    [Mutagenesis] mutant scoring failed: {_me}")
                        _first_err = False
                    _mat[_ai, _pi] = _wt_score

        _unique_scores = np.unique(_mat[~np.isnan(_mat)])
        if len(_unique_scores) <= 1:
            print(f"    [Mutagenesis] WARNING: all mutant scores = WT ({_wt_score:.4f}) — PLM embedding insensitive")

        _fw = max(9, _n_pos * 0.55 + 3)
        fig, ax = plt.subplots(figsize=(_fw, 7))

        if task == 'classification':
            _norm = _mc.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
            _cmap = 'RdBu'
        else:
            _vmin = float(np.nanmin(_mat))
            _vmax = float(np.nanmax(_mat))
            _norm = _mc.TwoSlopeNorm(vmin=_vmin, vcenter=(_vmin+_vmax)/2, vmax=_vmax)
            _cmap = 'coolwarm'

        im = ax.imshow(_mat, cmap=_cmap, norm=_norm, aspect='auto')

        _fsz_cell = max(4.0, min(7.0, 120.0 / max(_n_pos, 1)))
        for _ai in range(_n_aa):
            for _pi in range(_n_pos):
                _v = _mat[_ai, _pi]
                if np.isnan(_v): continue
                _tc = 'white' if (_v < 0.35 or _v > 0.65) else '#333'
                ax.text(_pi, _ai, f"{_v:.2f}", ha='center', va='center',
                        fontsize=_fsz_cell, color=_tc,
                        fontweight='bold' if abs(_v - 0.5) > 0.30 else 'normal')

        for _pi, _wt in enumerate(_cdr3):
            if _wt in AMINO_ACIDS:
                ax.add_patch(plt.Rectangle(
                    (_pi - 0.5, AMINO_ACIDS.index(_wt) - 0.5), 1, 1,
                    fill=False, edgecolor='black', lw=2.0, zorder=5))

        ax.set_xticks(range(_n_pos))
        ax.set_xticklabels([f"{_cdr3[i]}\n{i+1}" for i in range(_n_pos)], fontsize=8.5)
        ax.set_yticks(range(_n_aa))
        ax.set_yticklabels(list(AMINO_ACIDS), fontsize=8)
        ax.set_xlabel('CDR3 position  (WT residue shown above position number)', fontsize=9)
        ax.set_ylabel('Substituted AA', fontsize=9)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('P(PASS)' if task=='classification' else 'Score', fontsize=9, labelpad=6)
        if task == 'classification':
            cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['0.0\nFAIL', '0.25', '0.50\nborder', '0.75', '1.0\nPASS'])
        cbar.ax.tick_params(labelsize=7.5)

        _score_str = (f"WT P(PASS)={_wt_score:.4f}" if not np.isnan(_wt_score) else "WT score=N/A")
        _act_str   = (f"  |  Actual ({_label_col}) = {'PASS' if _actual==1 else 'FAIL'}"
                      if _actual is not None else "")
        ax.set_title(
            f"IPI MLAbDev · CDR3 Mutagenesis Heatmap\n"
            f"ID: {_bc}   {_score_str}{_act_str}\n"
            f"{model_type.upper()} | {lm} | {db_stem}",
            fontsize=9, loc='center', pad=8)
        plt.tight_layout()

        _bc_safe = _bc.replace('/', '_').replace(' ', '_')
        _img_path = os.path.join(_mut_dir, f"{_s+1:04d}_{_bc_safe}_cdr3_mutagenesis.{_fmt}")
        _save_kw = dict(dpi=_dpi, bbox_inches='tight')
        if _fmt == 'tiff':    _save_kw['format'] = 'tiff'
        elif _fmt in ('jpeg','jpg'):
            _save_kw['format'] = 'jpeg'
            _save_kw['pil_kwargs'] = {'quality': 95}
        plt.savefig(_img_path, **_save_kw)
        plt.close()
        _saved.append((_bc, _img_path, _wt_score))
        print(f"    → {os.path.basename(_img_path)}")

    print(f"[Mutagenesis] {len(_saved)} heatmaps → {_mut_dir}/")

    if _make_ppt and _saved:
        try:
            from pptx import Presentation as _Prs
            from pptx.util import Inches, Pt
            from pptx.enum.text import PP_ALIGN
            from pptx.dml.color import RGBColor

            prs = _Prs()
            prs.slide_width  = Inches(13.33)
            prs.slide_height = Inches(7.5)
            blank = prs.slide_layouts[6]

            for _bc, _img, _wt in _saved:
                slide = prs.slides.add_slide(blank)
                _iw = Inches(11); _ih = Inches(6.5)
                slide.shapes.add_picture(_img,
                    (prs.slide_width - _iw) / 2, Inches(0.4), width=_iw, height=_ih)
                txb = slide.shapes.add_textbox(Inches(0.15), Inches(7.1), Inches(13.0), Inches(0.35))
                tf  = txb.text_frame
                tf.text = (f"{_bc}  |  {model_type.upper()} | {lm} | {db_stem}  |  "
                           f"WT={'%.3f'%_wt if not np.isnan(_wt) else 'N/A'}")
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.CENTER
                p.runs[0].font.size = Pt(7)
                p.runs[0].font.color.rgb = RGBColor(0x88, 0x87, 0x80)

            _ppt = os.path.join(_mut_dir, "cdr3_mutagenesis_all.pptx")
            prs.save(_ppt)
            print(f"[Mutagenesis] PPT ({len(_saved)} slides) → {_ppt}")
        except ImportError:
            print("[Mutagenesis] pip install python-pptx for PPT")
        except Exception as _pe:
            print(f"[Mutagenesis] PPT failed — {_pe}")


def _predict_single(model, model_type, df_row, embeddings_fn, task):
    import numpy as np
    if model_type == 'rf':
        _needs_emb = (hasattr(model, 'fb_') and model.fb_ is not None and
                      model.fb_.feat_cfg.get('embedding', False))
        if _needs_emb and embeddings_fn is not None:
            _emb = embeddings_fn(df_row)
        else:
            _emb = None
        return model.predict_proba(df_row, embeddings=_emb)[0]
    elif model_type == 'xgboost':
        _xgb_row_df = df_row[['HSEQ','LSEQ','CDR3']].copy() if all(
            c in df_row.columns for c in ['HSEQ','LSEQ','CDR3']) else df_row.copy()
        _emb_r = embeddings_fn(df_row) if embeddings_fn else None
        if _emb_r is not None and hasattr(_emb_r, 'reshape'):
            _emb_r = _emb_r.reshape(1, -1)
        if getattr(model, 'task', 'classification') == 'regression':
            return float(model.predict(_xgb_row_df, embeddings=_emb_r)[0])
        return model.predict_proba(_xgb_row_df, embeddings=_emb_r)[0]
    elif model_type in ('transformer_onehot',):
        return model.predict_proba(df_row)[0]
    elif model_type in ('transformer_lm', 'cnn'):
        _emb = embeddings_fn(df_row) if embeddings_fn else None
        if _emb is not None:
            return model.predict_proba(_emb)[0]
        return model.predict_proba(df_row)[0]
    else:
        return model.predict_proba(df_row)[0]


def _find_matching_checkpoints(model_dir: str, target: str, lm: str,
                                model_type: str, db_stem: str,
                                ext: str) -> list:
    import glob, os
    _prefix  = f"FINAL_{target}_{lm}_{model_type}_{db_stem}"
    _pattern = os.path.join(model_dir, f"{_prefix}*{ext}")
    _found   = glob.glob(_pattern)
    _found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return _found


def auto_predict(input_file, target="sec_filter", lm="antiberta2",
                 model_type="xgboost", db_path=None, test_target=None,
                 run_mutagenesis=False, mutagenesis_n=None, threshold=None,
                 model_path=None, model_id=None, **kwargs):
    kwargs['model_path'] = model_path
    print(f"\nPREDICTING: {os.path.basename(input_file)}")
    print(f"Target: {target.upper()} | Model: {model_type.upper()} | LM: {lm}")
    if db_path:
        print(f"Using model trained on: {os.path.basename(db_path)}")

    if input_file.lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(input_file)
    elif input_file.lower().endswith(".csv"):
        data = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    if "BARCODE" not in data.columns:
        data["BARCODE"] = range(len(data))
    data["BARCODE"] = data["BARCODE"].astype(str).str.strip()
    data = data.set_index("BARCODE")

    if "HSEQ" in data.columns:
        data = data.reset_index()
        data = _ensure_cdr3(data, verbose=True)
        data = strip_cdr3_c_prefix(data, verbose=True)   # ← strip C
        data = data.set_index("BARCODE")

    _SEQ_ONLY = {"onehot", "onehot_vh", "onehot_cdr3", "onehot_hcdr3",
                 "biophysical", "kmer", "seq", "none"}

    if lm in _SEQ_ONLY and lm not in ("onehot", "onehot_vh"):
        print(f"[predict] lm='{lm}' — sequence-only mode, no PLM embedding loaded")
        X = data[["HSEQ", "LSEQ", "CDR3"]] if all(
            c in data.columns for c in ["HSEQ", "LSEQ", "CDR3"]) else data
    elif lm not in ["onehot", "onehot_vh"]:
        emb_file = f"{input_file}.{lm}.emb.csv"
        if not os.path.exists(emb_file):
            print("Generating embedding from input file...")
            generate_embedding(input_file, lm=lm)
        print(f"Using embedding: {emb_file}")
        embedding = pd.read_csv(emb_file, index_col=0)
        X, data = _align_embedding(data, embedding, context="predict")
    else:
        required = ["HSEQ", "LSEQ", "CDR3"]
        missing  = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for one-hot: {missing}")

        _nan_like = {"nan", "none", "null", "n/a", "na", "#n/a", "#value!", "#ref!"}
        for _col in required:
            data[_col] = data[_col].fillna("").astype(str)
            _bad = data[_col].str.strip().str.lower().isin(_nan_like)
            if _bad.any():
                print(f"  [fix-nan] {_col}: {_bad.sum():,} nan-like -> ''")
                data.loc[_bad, _col] = ""

        _empty_lseq = data["LSEQ"].str.len() == 0
        if _empty_lseq.any():
            print(f"  [warn] {_empty_lseq.sum():,} antibodies have empty LSEQ — encoded with VL=zeros.")

        _no_hseq = data["HSEQ"].str.strip().str.len() < 5
        if _no_hseq.any():
            print(f"  [fix-nan] Dropping {_no_hseq.sum():,} rows with missing HSEQ")
            data = data[~_no_hseq]
        if len(data) == 0:
            raise ValueError("No valid sequences after cleaning")

        X = data[required]

    _LM_ALIASES = {"onehot_cdr3": "onehot_hcdr3"}
    lm = _LM_ALIASES.get(lm, lm)

    if db_path:
        db_stem = Path(db_path).stem
    elif kwargs.get('model_path'):
        import re as _re2
        _ckpt_stem = Path(kwargs['model_path']).stem
        _m2 = _re2.search(
            r'(?:transformer_lm|transformer_onehot|rf|xgboost|cnn)_(.+?)(?:_ft_|_lora|_plmft|$)',
            _ckpt_stem)
        db_stem = _m2.group(1) if _m2 else _ckpt_stem
    else:
        db_stem = "default"

    ext        = ".pt" if model_type in ["cnn", "transformer_onehot", "transformer_lm"] else ".pkl"
    _chain_tag = ""
    if model_type == "rf":
        _SEQ_MODES = {"onehot", "kmer"}
        if lm in _SEQ_MODES or lm in ("biophysical", "none", "seq"):
            try:
                from models.randomforest import RandomForestModel as _RFM_tmp
                _tmp_cfg   = _RFM_tmp("config/random_forest.yaml").config
                _tmp_fb    = __import__('models.randomforest', fromlist=['FeatureBuilder']).FeatureBuilder(_tmp_cfg)
                _chain_tag = _tmp_fb.chain_tag
                if _chain_tag:
                    _chain_tag = f"_{_chain_tag}"
            except Exception:
                _chain_tag = ""

    _explicit_path = kwargs.get('model_path', None) or model_path
    if _explicit_path:
        model_path = _explicit_path
        print(f"[load] Using explicit model path: {model_path}")
    else:
        # ── Registry lookup (steps 2-4) ────────────────────────────────────
        _reg_entry = _registry_lookup(
            model_id = model_id,
            trainset = db_path,
            target   = target,
            lm       = lm,
            model    = model_type,
        )
        if _reg_entry:
            model_path = _reg_entry['model_path']
            _reg_id    = _reg_entry['model_id']
            _reg_thr   = _reg_entry.get('threshold')
            print(f"[load] Registry: {_reg_id}")
            print(f"[load] model_path: {model_path}")
            if _reg_thr and threshold is None:
                threshold = _reg_thr
                print(f"[load] threshold={threshold:.4f}  (from registry)")
        else:
            # ── Fallback: reconstruct FINAL_* path from args ────────────────
            _base_path = f"{MODEL_DIR}/FINAL_{target}_{lm}{_chain_tag}_{model_type}_{db_stem}"
            if os.path.exists(f"{_base_path}_regression{ext}"):
                model_path = f"{_base_path}_regression{ext}"
            else:
                model_path = f"{_base_path}{ext}"
            if not os.path.exists(model_path):
                _candidates = _find_matching_checkpoints(
                    MODEL_DIR, target, lm, model_type, db_stem, ext)
                if _candidates:
                    print(f"\n[load] Exact checkpoint not found: {Path(model_path).name}")
                    print(f"[load] Found {len(_candidates)} matching checkpoint(s):")
                    for i, c in enumerate(_candidates):
                        print(f"  [{i}] {Path(c).name}")
                    print(f"[load] Using: [{0}] {Path(_candidates[0]).name}")
                    model_path = _candidates[0]
                else:
                    raise FileNotFoundError(
                        f"Model not found: {model_path}\n"
                        f"Registry: no match for "
                        f"(target={target}, lm={lm}, model={model_type})\n"
                        f"Run --train first, or use --model_path / --model_id.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_type == "xgboost":
        model = XGBoostModel.load(model_path)
    elif model_type == "rf":
        model = RandomForestModel.load(model_path)
    elif model_type == "cnn":
        model = CNNModel.load(model_path,
                              embedding_dim=X.shape[1] if lm not in ["onehot","onehot_vh","onehot_hcdr3","onehot_cdr3"] else None)
    elif model_type == "transformer_onehot":
        model = TransformerOneHotModel.load(model_path)
    elif model_type == "transformer_lm":
        model = TransformerLMModel.load(model_path, embedding_dim=X.shape[1])

    if model_type == "rf":
        _SEQ_ONLY_P = {"onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","biophysical","kmer","seq","none"}
        _rf_X_df_p  = data[['HSEQ','LSEQ','CDR3']].copy() if all(
            c in data.columns for c in ['HSEQ','LSEQ','CDR3']) else data.copy()
        _rf_emb_p   = None if lm in _SEQ_ONLY_P else (X.values if hasattr(X, 'values') else X)
        if getattr(model, 'task', 'classification') == 'regression':
            scores = model.predict(_rf_X_df_p, embeddings=_rf_emb_p)
        else:
            scores = model.predict_proba(_rf_X_df_p, embeddings=_rf_emb_p)
    elif model_type == "xgboost":
        _SEQ_ONLY_P = {"onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","biophysical","kmer","seq","none"}
        _xgb_X_df_p = data[['HSEQ','LSEQ','CDR3']].copy() if all(
            c in data.columns for c in ['HSEQ','LSEQ','CDR3']) else data.copy()
        _xgb_emb_p  = None if lm in _SEQ_ONLY_P else (X.values if hasattr(X, 'values') else X)
        if getattr(model, 'task', 'classification') == 'regression':
            scores = model.predict(_xgb_X_df_p, embeddings=_xgb_emb_p)
        else:
            scores = model.predict_proba(_xgb_X_df_p, embeddings=_xgb_emb_p)
    else:
        if getattr(model, 'task', 'classification') == 'regression':
            scores = model.predict(X)
        else:
            scores = model.predict_proba(X)

    _is_regression = getattr(model, 'task', 'classification') == 'regression'
    _reg_tag       = "_regression" if _is_regression else ""
    if _is_regression:
        labels = scores
        print(f"[predict] regression mode — continuous scores")
    elif threshold is not None:
        _thresh = float(threshold)
        labels  = (scores >= _thresh).astype(int)
        print(f"[predict] threshold={_thresh:.4f}  (--threshold override)")
    else:
        _thresh = getattr(model, "recommended_threshold", 0.5)
        labels  = (scores >= _thresh).astype(int)
        print(f"[predict] threshold={_thresh:.4f}")

    data[f"{model_type}_{lm}_{db_stem}_score"] = scores
    data[f"{model_type}_{lm}_{db_stem}_label"] = labels

    if model_type in ("rf", "xgboost") and hasattr(model, 'shap_analysis'):
        try:
            import yaml as _yaml
            _yaml_path = ("config/xgboost.yaml" if model_type == "xgboost"
                          else "config/random_forest.yaml")
            if os.path.exists(_yaml_path):
                with open(_yaml_path) as _yf:
                    _yaml_shap = (_yaml.safe_load(_yf) or {}).get('shap', {})
                if _yaml_shap:
                    model.config['shap'] = _yaml_shap
        except Exception:
            pass
        _shap_cfg = model.config.get('shap', {})
        _shap_ok  = _shap_cfg.get('enabled', True)
        if not _shap_ok:
            print("[SHAP] Skipped — shap.enabled=false in config")
        else:
            try:
                _SEQ_ONLY_SH = {"onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","biophysical","kmer","seq","none"}
                _shap_emb    = None if lm in _SEQ_ONLY_SH else (X.values if hasattr(X, 'values') else X)
                _shap_X_df   = data[['HSEQ','LSEQ','CDR3']].copy() if all(
                    c in data.columns for c in ['HSEQ','LSEQ','CDR3']) else data.copy()
                _shap_top    = _shap_cfg.get('top_features', 30)
                _shap_prefix = f"{Path(input_file).stem}_{target}_{lm}_{model_type}_{db_stem}{_reg_tag}"

                print(f"\n[SHAP] Running on PREDICT set  n={len(_shap_X_df):,}  top={_shap_top}")

                if model.fb_ is None:
                    print("[SHAP] ERROR: FeatureBuilder not found in loaded model.")
                else:
                    if model_type == "xgboost":
                        import models.xgboost as _shap_mod
                    else:
                        import models.randomforest as _shap_mod
                    _orig_mdir = _shap_mod.MODEL_DIR
                    _shap_mod.MODEL_DIR = str(Path(input_file).parent)
                    _shap_actual = None
                    _label_col   = test_target if test_target and test_target in data.columns \
                                   else (target if target in data.columns else None)
                    if _label_col:
                        try:
                            _shap_actual = []
                            for v in data[_label_col].values:
                                if isinstance(v, float) and v != v:
                                    _shap_actual.append(None)
                                else:
                                    try:    _shap_actual.append(int(v))
                                    except: _shap_actual.append(None)
                        except Exception:
                            _shap_actual = None
                    _shap_col = test_target if test_target else target
                    try:
                        model.shap_analysis(
                            _shap_X_df, _shap_emb,
                            output_prefix   = _shap_prefix,
                            split_tag       = "predict",
                            top_n           = _shap_top,
                            barcodes        = list(_shap_X_df.index.astype(str)),
                            actual_labels   = _shap_actual,
                            actual_col_name = _shap_col,
                            lm_name         = lm,
                            db_name         = db_stem,
                        )
                    finally:
                        _shap_mod.MODEL_DIR = _orig_mdir
            except Exception as _se:
                import traceback
                print(f"[SHAP] predict SHAP failed: {_se}")
                print(traceback.format_exc())

    if run_mutagenesis:
        try:
            _run_cdr3_mutagenesis(
                data=data, model=model, model_type=model_type, lm=lm,
                db_stem=db_stem, input_file=input_file, target=target,
                test_target=test_target, embeddings_fn=None, n_override=mutagenesis_n)
        except Exception as _me:
            import traceback
            print(f"[Mutagenesis] failed — {_me}")
            print(traceback.format_exc())

    path        = Path(input_file)
    output_file = path.with_name(
        f"{path.stem}_pred_{target}_{lm}_{model_type}_{db_stem}{_reg_tag}{path.suffix}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        data.reset_index().to_excel(output_file, index=False)
    else:
        data.reset_index().to_csv(output_file, index=False)

    print(f"Saved predictions to: {output_file}")
    if not _is_regression:
        print(f"Positive rate: {labels.mean():.1%}")

    _eval_label_col  = test_target if (test_target and test_target != target) else target
    _data_with_index = data.reset_index()
    _eval_col_found  = _eval_label_col in _data_with_index.columns
    if not _eval_col_found and target in _data_with_index.columns:
        _eval_label_col = target
        _eval_col_found = True

    if _eval_col_found and not _is_regression:
        try:
            from utils.evaluate_model import evaluate
            _score_col  = f"{model_type}_{lm}_{db_stem}_score"
            _eval_stem  = str(path.with_name(f"{path.stem}_pred_{target}_{lm}_{model_type}_{db_stem}{_reg_tag}"))
            evaluate(file=str(output_file), target=_eval_label_col, score_col=_score_col,
                     cost_fp=1.0, cost_fn=3.0, out=_eval_stem, test_target=test_target,
                     model_type=model_type, lm=lm, db_stem=db_stem,
                     dataset_name=Path(input_file).stem)
        except ImportError:
            print("[eval] utils/evaluate_model.py not found — skipping.")
        except Exception as _e:
            print(f"[eval] WARNING: evaluation failed — {_e}")

    if not _is_regression:
        try:
            from utils.plot_biophysical import plot_biophysical_report
            _bio_stem = str(path.with_name(
                f"{path.stem}_pred_{target}_{lm}_{model_type}_{db_stem}{_reg_tag}"))
            plot_biophysical_report(file=str(output_file), target=target,
                                    test_target=_eval_label_col if _eval_col_found else None,
                                    out=_bio_stem, dataset_name=Path(input_file).stem)
        except ImportError:
            print("[biophys] utils/plot_biophysical.py not found — skipping.")
        except Exception as _e:
            print(f"[biophys] WARNING: {_e}")


_ALL_PLM_LMS = ["ablang", "antiberty", "antiberta2", "antiberta2-cssp", "igbert"]


def auto_predict_multi_lm(input_file, target="psr_filter",
                           lms=None, lm_tag="all",
                           model_type="transformer_lm",
                           db_path=None, test_target=None):
    if lms is None:
        lms = _ALL_PLM_LMS

    print(f"\n{'═'*62}")
    print(f"  MULTI-LM PREDICTION")
    print(f"  File   : {os.path.basename(input_file)}")
    print(f"  Target : {target.upper()}  |  Model: {model_type.upper()}")
    print(f"  LMs    : {lms}")
    if db_path:
        print(f"  DB     : {os.path.basename(db_path)}")
    print(f"{'─'*62}")

    if input_file.lower().endswith((".xlsx", ".xls")):
        base_data = pd.read_excel(input_file)
    elif input_file.lower().endswith(".csv"):
        base_data = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    if "BARCODE" not in base_data.columns:
        base_data["BARCODE"] = range(len(base_data))
    base_data["BARCODE"] = base_data["BARCODE"].astype(str).str.strip()

    if "HSEQ" in base_data.columns:
        base_data = _ensure_cdr3(base_data, verbose=True)
        base_data = strip_cdr3_c_prefix(base_data, verbose=True)   # ← strip C

    base_data = base_data.set_index("BARCODE")

    db_stem = Path(db_path).stem if db_path else "default"
    ext     = ".pt" if model_type in ["cnn", "transformer_onehot", "transformer_lm"] else ".pkl"

    results_summary = []
    failed_lms      = []

    _SEQ_ONLY_MULTI = {"biophysical","kmer","onehot","onehot_vh",
                        "onehot_cdr3","onehot_hcdr3","none","seq"}
    for lm in lms:
        print(f"\n  ── LM: {lm} ──")
        try:
            if lm in _SEQ_ONLY_MULTI:
                _lm_data = base_data.copy()
                X        = _lm_data
            else:
                emb_file = f"{input_file}.{lm}.emb.csv"
                if not os.path.exists(emb_file):
                    print(f"  Generating {lm} embedding ...")
                    generate_embedding(input_file, lm=lm)
                embedding = pd.read_csv(emb_file, index_col=0)
                X, _lm_data = _align_embedding(base_data, embedding, context=f"predict/{lm}")

            _base2     = f"{MODEL_DIR}/FINAL_{target}_{lm}_{model_type}_{db_stem}"
            if os.path.exists(f"{_base2}_regression{ext}"):
                model_path = f"{_base2}_regression{ext}"
            else:
                model_path = f"{_base2}{ext}"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            if model_type == "xgboost":
                model = XGBoostModel.load(model_path)
            elif model_type == "rf":
                model = RandomForestModel.load(model_path)
            elif model_type == "cnn":
                model = CNNModel.load(model_path, embedding_dim=X.shape[1])
            elif model_type == "transformer_lm":
                model = TransformerLMModel.load(model_path, embedding_dim=X.shape[1])

            if model_type == "rf":
                _rf_X_df_m = base_data[['HSEQ','LSEQ','CDR3']].copy() if all(
                    c in base_data.columns for c in ['HSEQ','LSEQ','CDR3']) else base_data.copy()
                _rf_X_df_m = _rf_X_df_m.loc[_lm_data.index]
                _SEQ_ONLY_ML = {"biophysical","kmer","onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","none","seq"}
                _rf_emb_m  = None if lm in _SEQ_ONLY_ML else (X.values if hasattr(X, 'values') else X)
                scores = model.predict_proba(_rf_X_df_m, embeddings=_rf_emb_m)
            elif model_type == "xgboost":
                _SEQ_ONLY_M = {"biophysical","kmer","onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","none","seq"}
                _xgb_X_df_m = _lm_data[['HSEQ','LSEQ','CDR3']].copy() if all(
                    c in _lm_data.columns for c in ['HSEQ','LSEQ','CDR3']) else _lm_data.copy()
                _xgb_emb_m  = None if lm in _SEQ_ONLY_M else (X.values if hasattr(X, 'values') else X)
                scores = model.predict_proba(_xgb_X_df_m, embeddings=_xgb_emb_m)
            else:
                scores = model.predict_proba(X)

            _thresh = getattr(model, "recommended_threshold", 0.5)
            labels  = (scores >= _thresh).astype(int)
            pos_rate = labels.mean()
            print(f"  threshold={_thresh:.4f}  n={len(scores):,}  pos_rate={pos_rate:.1%}")

            score_col = f"{model_type}_{lm}_{db_stem}_score"
            label_col = f"{model_type}_{lm}_{db_stem}_label"
            base_data[score_col] = np.nan
            base_data[label_col] = np.nan
            base_data.loc[_lm_data.index, score_col] = scores
            base_data.loc[_lm_data.index, label_col] = labels
            results_summary.append((lm, len(scores), pos_rate, _thresh, _thresh))

        except Exception as e:
            print(f"  [ERROR] {lm} failed: {e}")
            failed_lms.append((lm, str(e)))

    if not results_summary:
        print("\n[multi-lm] ERROR: All LMs failed — no output written.")
        return

    path        = Path(input_file)
    output_file = path.with_name(
        f"{path.stem}_pred_{target}_{lm_tag}_{model_type}_{db_stem}{path.suffix}")
    out_data = base_data.reset_index()
    if path.suffix.lower() in [".xlsx", ".xls"]:
        out_data.to_excel(output_file, index=False)
    else:
        out_data.to_csv(output_file, index=False)

    print(f"\n{'═'*62}")
    print(f"  MULTI-LM SUMMARY")
    print(f"{'─'*62}")
    for row in results_summary:
        lm_name, n, pos, thresh, opt_t = row
        print(f"  {lm_name:25s}  n={n:>7,}  pos={pos:>6.1%}  thresh={thresh:.4f}")
    if failed_lms:
        for lm_name, err in failed_lms:
            print(f"  [FAILED] {lm_name}: {err}")
    print(f"  Output → {output_file}")
    print(f"{'═'*62}\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="IPI Antibody Developability Prediction Platform")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict",         type=str,            help="Predict on new file")
    group.add_argument("--build-embedding", type=str,            help="Generate embeddings only")
    group.add_argument("--kfold",           type=int,            help="Run k-fold CV")
    group.add_argument("--train",           action="store_true", help="Train final model")
    group.add_argument("--split-dataset",   action="store_true", dest="split_dataset",
                       help="Split --db into train + val files.")
    group.add_argument("--build-dataset",   type=str,            dest="build_dataset",
                       metavar="FILE",
                       help="Build balanced + imbalanced datasets from FILE "
                            "(uses --target as label column)")
    group.add_argument("--correlate",       type=str, nargs="+", dest="correlate",
                       metavar="FILE",
                       help="Correlation analysis on prediction file(s) "
                            "(uses --target, --assay)")
    group.add_argument("--list-models",     action="store_true", dest="list_models",
                       help="List all registered models (optionally filter with "
                            "--target, --lm, --model, --db)")

    parser.add_argument("--target", type=str, default="psr_filter")
    parser.add_argument("--lm", default="antiberta2")
    parser.add_argument("--model", default="xgboost",
                        choices=["xgboost","rf","cnn","transformer_onehot","transformer_lm"])
    parser.add_argument("--db",      type=str)
    parser.add_argument("--cluster", type=float, default=0.8, metavar="THRESHOLD")
    parser.add_argument("--split",   type=float, default=0.0, metavar="TRAIN_FRAC")
    parser.add_argument("--val",     type=str,   default=None, metavar="VAL_FILE")
    parser.add_argument("--cluster_col", type=str, default="CDR3",
                        choices=["CDR3", "VH", "VHVL"])
    parser.add_argument("--no-aug",  dest="no_aug", action="store_true", default=True)
    parser.add_argument("--test_target", type=str, default=None)
    parser.add_argument("--mutagenesis", type=int, nargs="?", const=50, default=None, metavar="N")
    parser.add_argument("--threshold", type=float, default=None, metavar="T")
    parser.add_argument("--model_path", type=str, default=None, metavar="PATH")
    parser.add_argument("--model_id",   type=str, default=None, metavar="ID",
                        help="Registry model_id for --predict lookup "
                             "(alternative to --model_path and --db)")
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--finetune_plm", action="store_true", default=False)
    parser.add_argument("--pretrained", type=str, default=None, metavar="PATH")
    parser.add_argument("--finetune_db", type=str, default=None, metavar="FILE")
    parser.add_argument("--freeze_layers", type=str, default="1", metavar="N")
    parser.add_argument("--freeze_plm_layers", type=int, default=10, metavar="N")
    parser.add_argument("--finetune_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--finetune_epochs", type=int, default=10, metavar="N")
    parser.add_argument("--lr_plm", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--lr_classifier", type=float, default=1e-4, metavar="LR")
    parser.add_argument("--peft", type=str, default="none", choices=["none", "lora"])
    parser.add_argument("--lora_r", type=int, default=8, metavar="R")
    parser.add_argument("--lora_alpha", type=float, default=16.0, metavar="A")
    parser.add_argument("--lora_layers", type=int, nargs="+", default=None, metavar="N")
    parser.add_argument("--cost_fn", type=float, default=3.0, metavar="W")
    parser.add_argument("--cost_fp", type=float, default=1.0, metavar="W")

    # ── --build-dataset args ──────────────────────────────────────────────────
    parser.add_argument("--strategy",   type=str,   default="combined",
                        choices=["cluster", "kmer_consensus", "combined"],
                        help="[build-dataset] Downsampling strategy (default: combined)")
    parser.add_argument("--min-total",  type=int,   default=5000, dest="min_total",
                        help="[build-dataset] Min total size for imbalanced dataset (default: 5000)")
    parser.add_argument("--floor-prob", type=float, default=0.5,  dest="floor_prob",
                        help="[build-dataset] Per-cluster OOF floor to exclude mislabels (default: 0.5)")
    parser.add_argument("--min-prob",   type=float, default=0.6,  dest="min_prob",
                        help="[build-dataset] Global OOF consensus threshold (default: 0.6)")
    parser.add_argument("--cv",         type=int,   default=5,
                        help="[build-dataset] GridSearchCV folds (default: 5; use 3 for large datasets)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="[build-dataset] Random seed (default: 42)")

    # ── --correlate args ──────────────────────────────────────────────────────
    parser.add_argument("--assay",          type=str, nargs="+", default=None,
                        help="[correlate] Assay column name(s) to correlate against "
                             "(space or comma separated; quote multi-word names)")
    parser.add_argument("--out",            type=str, default=None,
                        help="[correlate] Output file stem (default: auto-derived from input)")
    parser.add_argument("--title",          type=str, default=None,
                        help="[correlate] Custom figure title")
    parser.add_argument("--xlabel",         type=str, default=None,
                        help="[correlate] Custom x-axis label")
    parser.add_argument("--logit-trans",    action="store_true", default=False,
                        dest="logit_trans",
                        help="[correlate] Apply logit transform to scores before correlating")
    parser.add_argument("--list-scores",    action="store_true", default=False,
                        dest="list_scores",
                        help="[correlate] Print detected score columns and exit")
    parser.add_argument("--tsne-source",    type=str, default="assay", dest="tsne_source",
                        choices=["assay", "scores", "embedding", "both"],
                        help="[correlate] Feature space for t-SNE (default: assay)")
    parser.add_argument("--embedding-file", type=str, default=None, dest="embedding_file",
                        metavar="EMB_CSV",
                        help="[correlate] PLM embedding CSV for --tsne-source embedding")

    args    = parser.parse_args()

    _raw_db = args.db or get_default_db_path()
    if _raw_db and str(_raw_db).lower().endswith(('.pt', '.pkl')):
        if not getattr(args, 'model_path', None):
            args.model_path = _raw_db
            print(f"[db] --db is a checkpoint file → using as --model_path")
        import re as _re
        _stem = Path(_raw_db).stem
        _m = _re.search(
            r'(?:transformer_lm|transformer_onehot|rf|xgboost|cnn)_(.+?)(?:_ft_|_lora|_plmft|$)',
            _stem)
        _inferred_db = _m.group(1) if _m else _stem
        args.db  = None
        db_path  = None
        print(f"[db] Inferred db_stem from checkpoint: '{_inferred_db}'")
    else:
        db_path = _raw_db

    _log_path = _setup_logging(args, db_path)

    _cluster_thresh = args.cluster
    _cluster_col_src = getattr(args, 'cluster_col', 'CDR3').upper()
    _cluster_col_map = {
        'CDR3':  f"HCDR3_CLUSTER_{_cluster_thresh}",
        'HSEQ':  f"VH_CLUSTER_{_cluster_thresh}",
        'VH':    f"VH_CLUSTER_{_cluster_thresh}",
        'VHVL':  f"VHVL_CLUSTER_{_cluster_thresh}",
        'LSEQ':  f"VL_CLUSTER_{_cluster_thresh}",
    }
    _cluster_col = _cluster_col_map.get(_cluster_col_src, f"HCDR3_CLUSTER_{_cluster_thresh}")

    # ── --list-models ─────────────────────────────────────────────────────────
    if getattr(args, 'list_models', False):
        _registry_list(
            target   = args.target   if args.target   != 'psr_filter' else None,
            lm       = args.lm       if args.lm       != 'antiberta2' else None,
            model    = args.model    if args.model     != 'xgboost'   else None,
            trainset = db_path,
        )
        return

    # ── --build-dataset ───────────────────────────────────────────────────────
    if getattr(args, 'build_dataset', None):
        from utils.build_balanced_dataset_v4 import build_balanced_dataset
        # --cluster_col reused as the sequence column for both clustering and kmer steps
        _bbd_seq_col = getattr(args, 'cluster_col', 'CDR3')
        build_balanced_dataset(
            input_path   = args.build_dataset,
            label_col    = args.target,
            strategy     = args.strategy,
            min_total    = args.min_total,
            cluster_col  = _bbd_seq_col,
            threshold    = args.cluster,      # --cluster reused (default 0.8)
            kmer_col     = _bbd_seq_col,
            min_prob     = args.min_prob,
            floor_prob   = args.floor_prob,
            cv           = args.cv,
            bal_path     = None,              # auto: <input>_<label>_balanced.xlsx
            imb_path     = None,              # auto: <input>_<label>_imbalanced_<min_total>.xlsx
            reject_path  = None,              # auto: <input>_<label>_majority_rejected.xlsx
            random_state = args.seed,
        )
        return

    # ── --correlate ───────────────────────────────────────────────────────────
    if getattr(args, 'correlate', None):
        from utils.developability_correlation import run as _corr_run

        # Support comma-separated file list as well as space-separated
        _corr_files = []
        for _f in args.correlate:
            _corr_files.extend([x.strip() for x in _f.split(',') if x.strip()])

        # Mirror the comma/space handling from the original CLI
        _assay_raw = args.assay or []
        if any(',' in tok for tok in _assay_raw):
            _assay_cols = [x.strip() for x in ' '.join(_assay_raw).split(',') if x.strip()]
        else:
            _assay_cols = [x.strip() for x in _assay_raw if x.strip()]

        if not _assay_cols and not args.list_scores:
            parser.error(
                "--correlate requires --assay COL [COL ...]\n"
                "  Use --list-scores to discover available columns first."
            )

        _corr_run(
            files          = _corr_files,
            assay_cols     = _assay_cols,
            out            = args.out,
            title          = args.title,
            xlabel         = args.xlabel,
            logit_trans    = args.logit_trans,
            list_scores    = args.list_scores,
            target_col     = args.target,
            tsne_source    = args.tsne_source,
            embedding_file = args.embedding_file,
        )
        return

    # ── --split-dataset ───────────────────────────────────────────────────────
    if args.split_dataset:
        if not (0.0 < args.split < 1.0):
            parser.error("--split must be in (0,1)")
        split_and_save(db_path=db_path, split=args.split,
                       cluster_thresh=_cluster_thresh,
                       cluster_col=getattr(args, 'cluster_col', 'CDR3'),
                       label_col=args.target)
        return

    if args.build_embedding:
        lms = ["ablang","antiberty","antiberta2","antiberta2-cssp"] if args.lm == "all" else [args.lm]
        if args.lm in ["onehot","onehot_vh"]:
            print("One-hot encoding does not require pre-generation.")
        else:
            for lm in lms:
                generate_embedding(args.build_embedding, lm=lm)
        return

    if args.kfold:
        if not db_path:
            parser.error("--db required for k-fold")
        X, data, y = load_data(db_path, lm=args.lm, label_col=args.target)
        title = f"{args.target.upper()}_{args.model}"

        if _cluster_col not in data.columns:
            if "CDR3" in data.columns:
                print(f"\n[kfold] '{_cluster_col}' not found — computing automatically ...")
                try:
                    from utils.clustering import greedy_clustering_by_levenshtein
                    _seq_col = 'CDR3'
                    data[_cluster_col] = greedy_clustering_by_levenshtein(
                        data[_seq_col].fillna('').tolist(), _cluster_thresh)
                    n_clust = data[_cluster_col].nunique()
                    print(f"[kfold] {n_clust:,} clusters")
                    try:
                        _df_save = read_dataframe(db_path)
                        _df_save[_cluster_col] = data[_cluster_col].values
                        save_dataframe(_df_save, db_path)
                        print(f"[kfold] ✓ Saved {_cluster_col} → {Path(db_path).name}")
                    except Exception as _ce:
                        print(f"[kfold] NOTE: could not save clustering — {_ce}")
                except Exception as _clust_e:
                    print(f"[kfold] WARNING: clustering failed — {_clust_e}")

        if args.model == "xgboost":
            db_stem   = Path(db_path).stem if db_path else ""
            _SEQ_ONLY = {"onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","biophysical","kmer","seq","none"}
            _xgb_emb  = None if args.lm in _SEQ_ONLY else (X.values if hasattr(X, 'values') else X)
            _xgb_X_df = data[['HSEQ','LSEQ','CDR3']].copy() if all(
                c in data.columns for c in ['HSEQ','LSEQ','CDR3']) else data.copy()
            _lm_k = args.lm.lower()
            if _lm_k == "biophysical":
                _xgb_feat = {'embedding': False, 'biophysical': True, 'kmer': False, 'onehot': False}
            elif _lm_k == "kmer":
                _xgb_feat = {'embedding': False, 'biophysical': False, 'kmer': True, 'onehot': False}
            elif _lm_k in ("onehot","onehot_vh","onehot_cdr3","onehot_hcdr3"):
                _oh_seq = {"onehot":"VHVL","onehot_vh":"VH","onehot_cdr3":"HCDR3","onehot_hcdr3":"HCDR3"}.get(_lm_k,"VHVL")
                _xgb_feat = {'embedding': False, 'biophysical': False, 'kmer': False, 'onehot': True, '_onehot_sequence': _oh_seq}
            elif _lm_k in ("none","seq"):
                _xgb_feat = {'embedding': False}
            else:
                _xgb_feat = {'embedding': True}
            XGBoostModel.kfold_validation(
                data, _xgb_X_df, y, embeddings=_xgb_emb, embedding_lm=args.lm,
                title=title, kfold=args.kfold, target=args.target,
                cluster_col=_cluster_col, db_stem=db_stem,
                override_features=_xgb_feat,
                cost_fn=getattr(args, 'cost_fn', 3.0), cost_fp=getattr(args, 'cost_fp', 1.0))

        elif args.model == "rf":
            db_stem  = Path(db_path).stem if db_path else ""
            _rf_X_df = data[['HSEQ','LSEQ','CDR3']].copy() if all(
                c in data.columns for c in ['HSEQ','LSEQ','CDR3']) else data.copy()
            _SEQ_ONLY = {"onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","biophysical","kmer","seq","none"}
            _rf_emb = None if args.lm in _SEQ_ONLY else (X.values if hasattr(X, 'values') else X)
            _lm_k = args.lm.lower()
            if _lm_k == "biophysical":
                _rf_feat_override = {'embedding': False, 'biophysical': True, 'kmer': False}
            elif _lm_k == "kmer":
                _rf_feat_override = {'embedding': False, 'biophysical': False, 'kmer': True}
            elif _lm_k in ("none","seq"):
                _rf_feat_override = {'embedding': False}
            elif _lm_k in ("onehot","onehot_vh","onehot_cdr3","onehot_hcdr3"):
                _oh_seq = {"onehot":"VHVL","onehot_vh":"VH","onehot_cdr3":"HCDR3","onehot_hcdr3":"HCDR3"}.get(_lm_k,"VHVL")
                _rf_feat_override = {'embedding': False, 'biophysical': False, 'kmer': False, 'onehot': True, '_onehot_sequence': _oh_seq}
            else:
                _rf_feat_override = {'embedding': True}
            RandomForestModel.kfold_validation(
                data, _rf_X_df, y, embeddings=_rf_emb, embedding_lm=args.lm,
                title=title, kfold=args.kfold, target=args.target,
                cluster_col=_cluster_col, db_stem=db_stem,
                override_features=_rf_feat_override,
                cost_fn=getattr(args, 'cost_fn', 3.0), cost_fp=getattr(args, 'cost_fp', 1.0))

        elif args.model == "transformer_onehot":
            _kf_lm = args.lm if args.lm in ("onehot","onehot_vh") else "onehot"
            TransformerOneHotModel().kfold_validation(
                data, X, y, embedding_lm=_kf_lm, title=title,
                kfold=args.kfold, target=args.target,
                cluster_col=_cluster_col,
                db_stem=Path(db_path).stem if db_path else "")

        elif args.model == "transformer_lm":
            db_stem = Path(db_path).stem if db_path else ""
            TransformerLMModel().kfold_validation(
                db_stem, data, X, y, embedding_lm=args.lm, title=title,
                kfold=args.kfold, target=args.target, cluster_col=_cluster_col)
        return

    if args.train:
        if not db_path:
            parser.error("--db required for training")

        db_stem           = Path(db_path).stem
        _has_explicit_val = bool(getattr(args, 'val', None))
        _has_split        = 0.0 < args.split < 1.0
        val_X = val_y = val_db_path = data_val = None

        if _has_explicit_val:
            X, data, y             = load_data(db_path,  lm=args.lm, label_col=args.target)
            X_val, data_val, y_val = load_data(args.val, lm=args.lm, label_col=args.target)
            val_X, val_y, val_db_path = X_val, y_val, args.val
        elif _has_split:
            train_path, val_path, _, _ = split_and_save(
                db_path=db_path, split=args.split,
                cluster_thresh=_cluster_thresh,
                cluster_col=getattr(args, 'cluster_col', 'CDR3'),
                label_col=args.target)
            X, data, y             = load_data(train_path, lm=args.lm, label_col=args.target)
            X_val, data_val, y_val = load_data(val_path,   lm=args.lm, label_col=args.target)
            val_X, val_y, val_db_path = X_val, y_val, val_path
        else:
            X, data, y = load_data(db_path, lm=args.lm, label_col=args.target)

        if args.model == "xgboost":
            model = XGBoostModel(verbose=False)
            _lm_xgb = args.lm.lower()
            if _lm_xgb == "biophysical":
                model.config['features'].update({'embedding': False, 'biophysical': True, 'kmer': False, 'onehot': False})
            elif _lm_xgb == "kmer":
                model.config['features'].update({'embedding': False, 'biophysical': False, 'kmer': True, 'onehot': False})
            elif _lm_xgb in ("onehot","onehot_vh","onehot_cdr3","onehot_hcdr3"):
                _oh_seq_xgb = {"onehot":"VHVL","onehot_vh":"VH","onehot_cdr3":"HCDR3","onehot_hcdr3":"HCDR3"}.get(_lm_xgb,"VHVL")
                model.config['features'].update({'embedding': False, 'biophysical': False, 'kmer': False, 'onehot': True})
                model.config.setdefault('onehot', {})['sequence'] = _oh_seq_xgb
            elif _lm_xgb in ("none","seq"):
                model.config['features']['embedding'] = False
            else:
                model.config['features']['embedding'] = True
            XGBoostModel.print_config_report(model.config)

        elif args.model == "rf":
            model = RandomForestModel(verbose=False)
            _lm = args.lm.lower()
            if _lm == "biophysical":
                model.config['features'].update({'embedding': False, 'biophysical': True, 'kmer': False})
            elif _lm == "kmer":
                model.config['features'].update({'embedding': False, 'biophysical': False, 'kmer': True})
            elif _lm in ("none","seq"):
                model.config['features']['embedding'] = False
            elif _lm in ("onehot","onehot_vh","onehot_cdr3","onehot_hcdr3"):
                _oh_seq = {"onehot":"VHVL","onehot_vh":"VH","onehot_cdr3":"HCDR3","onehot_hcdr3":"HCDR3"}.get(_lm,"VHVL")
                model.config.setdefault('features', {}).update({'embedding': False, 'biophysical': False, 'kmer': False, 'onehot': True})
                model.config.setdefault('onehot', {})['sequence'] = _oh_seq
            else:
                model.config['features']['embedding'] = True
            from models.randomforest import RandomForestModel as _RFM
            _RFM.print_config_report(model.config)

        elif args.model == "cnn":
            model = CNNModel()
        elif args.model == "transformer_onehot":
            model = TransformerOneHotModel()
            _t_lm = args.lm if args.lm in ("onehot","onehot_vh") else "onehot"
            model.set_lm_mode(_t_lm)
        elif args.model == "transformer_lm":
            model = TransformerLMModel()

        _vkw = {}
        if val_X is not None and args.model in ("transformer_lm","transformer_onehot","cnn"):
            _vkw = {"val_X": val_X, "val_y": val_y}

        if args.model == "transformer_onehot":
            model.train(X, y, target=args.target, db_stem=db_stem,
                        cluster_col=_cluster_col, no_aug=args.no_aug, **_vkw)
        elif args.model == "transformer_lm":
            model.train(X, y, target=args.target, db_stem=db_stem,
                        embedding_lm=args.lm, cluster_col=_cluster_col, **_vkw)
        elif args.model == "rf":
            _rf_train_X_df = data[['HSEQ','LSEQ','CDR3']].copy() if all(
                c in data.columns for c in ['HSEQ','LSEQ','CDR3']) else data.copy()
            _SEQ_ONLY_TRAIN = {"biophysical","kmer","onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","none","seq"}
            _rf_train_emb   = None if args.lm in _SEQ_ONLY_TRAIN else (X.values if hasattr(X, 'values') else X)
            _rf_val_X_df = _rf_val_emb = None
            if val_X is not None and val_y is not None and data_val is not None:
                _rf_val_X_df = data_val[['HSEQ','LSEQ','CDR3']].copy() if all(
                    c in data_val.columns for c in ['HSEQ','LSEQ','CDR3']) else data_val.copy()
                _rf_val_emb  = None if args.lm in _SEQ_ONLY_TRAIN else (val_X.values if hasattr(val_X, 'values') else val_X)
            _y_uniq_rf   = len(set(y.tolist() if hasattr(y, 'tolist') else list(y)))
            _rf_task_tag = "_regression" if _y_uniq_rf > 2 else ""
            _rf_full_stem = f"{args.target}_{args.lm}_rf_{db_stem}{_rf_task_tag}"
            model.train(_rf_train_X_df, y,
                        embeddings=_rf_train_emb, val_X=_rf_val_X_df,
                        val_y=val_y if val_y is not None else None,
                        val_embeddings=_rf_val_emb,
                        target=_rf_full_stem, target_col=args.target, embedding_lm=args.lm)
        elif args.model == "xgboost":
            _xgb_train_X_df = data[['HSEQ','LSEQ','CDR3']].copy() if all(
                c in data.columns for c in ['HSEQ','LSEQ','CDR3']) else data.copy()
            _SEQ_ONLY_XGB = {"biophysical","kmer","onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","none","seq"}
            _xgb_train_emb = None if args.lm in _SEQ_ONLY_XGB else (X.values if hasattr(X, 'values') else X)
            _xgb_val_X_df = _xgb_val_emb = None
            if val_X is not None and val_y is not None and data_val is not None:
                _xgb_val_X_df = data_val[['HSEQ','LSEQ','CDR3']].copy() if all(
                    c in data_val.columns for c in ['HSEQ','LSEQ','CDR3']) else data_val.copy()
                _xgb_val_emb = None if args.lm in _SEQ_ONLY_XGB else (val_X.values if hasattr(val_X, 'values') else val_X)
            _y_uniq_xgb    = len(set(y.tolist() if hasattr(y, 'tolist') else list(y)))
            _xgb_task_tag  = "_regression" if _y_uniq_xgb > 2 else ""
            _xgb_full_stem = f"{args.target}_{args.lm}_xgboost_{db_stem}{_xgb_task_tag}"
            model.train(_xgb_train_X_df, y,
                        embeddings=_xgb_train_emb, val_X=_xgb_val_X_df,
                        val_y=val_y if val_y is not None else None,
                        val_embeddings=_xgb_val_emb,
                        target=_xgb_full_stem, target_col=args.target, embedding_lm=args.lm)
        else:
            model.train(X, y, embedding_lm=args.lm)

        ext  = ".pt" if args.model in ["cnn","transformer_onehot","transformer_lm"] else ".pkl"
        _task_suffix = ""
        if hasattr(model, 'task') and model.task == 'regression':
            _task_suffix = "_regression"
        path = f"{MODEL_DIR}/FINAL_{args.target}_{args.lm}_{args.model}_{db_stem}{_task_suffix}{ext}"
        model.save(path)
        print(f"FINAL MODEL SAVED: {path}")

        # ── Auto-register in model registry ───────────────────────────────
        _reg_threshold = getattr(model, 'recommended_threshold', None)
        _reg_epochs    = None
        try:
            _reg_epochs = model.config.get('training', {}).get('epochs')
        except Exception:
            pass

        # Try to read kfold stats from most recent BEST_*.pt for this run
        _kfold_auc = _kfold_std = _kfold_folds = _kfold_best_fold = None
        try:
            import glob as _glob
            _best_pat  = os.path.join(
                MODEL_DIR,
                f"BEST_{args.target}_{args.lm}*{args.model}*{db_stem}*.pt")
            _best_list = sorted(_glob.glob(_best_pat),
                                key=os.path.getmtime, reverse=True)
            if _best_list:
                import torch as _torch
                _ckpt = _torch.load(_best_list[0], map_location='cpu',
                                    weights_only=False)
                if isinstance(_ckpt, dict):
                    _kfold_auc       = _ckpt.get('mean_auc') or _ckpt.get('fold_auc')
                    _kfold_folds     = _ckpt.get('kfold')
                    _kfold_best_fold = _ckpt.get('fold')
        except Exception:
            pass

        _registry_add(
            model_id        = Path(path).name,
            trainset        = db_path,
            target          = args.target,
            lm              = args.lm,
            model           = args.model,
            model_path      = path,
            entry_type      = 'full_train',
            kfold_auc       = _kfold_auc,
            kfold_std       = _kfold_std,
            kfold_folds     = _kfold_folds,
            kfold_best_fold = _kfold_best_fold,
            threshold       = _reg_threshold,
            epochs          = _reg_epochs,
            notes           = '',
        )

    if args.predict:
        _VALID_PLM_LMS = {"ablang","antiberty","antiberta2","antiberta2-cssp","igbert"}
        _SEQ_ONLY_LMS  = {"onehot","onehot_vh","onehot_cdr3","onehot_hcdr3","biophysical","kmer","seq","none"}
        _VALID_ALL_LMS = _VALID_PLM_LMS | _SEQ_ONLY_LMS

        _lm_raw = args.lm.strip()
        if _lm_raw == "all":
            _lm_list = _ALL_PLM_LMS
        elif "," in _lm_raw:
            _lm_list = [x.strip() for x in _lm_raw.split(",") if x.strip()]
        else:
            _lm_list = None

        if _lm_list is not None:
            _lm_tag = "all" if set(_lm_list) == set(_ALL_PLM_LMS) else "_".join(_lm_list)
            auto_predict_multi_lm(args.predict, target=args.target, lms=_lm_list,
                                   lm_tag=_lm_tag, model_type=args.model,
                                   db_path=args.db, test_target=args.test_target)
        else:
            auto_predict(args.predict, target=args.target, lm=_lm_raw,
                         model_type=args.model, db_path=args.db,
                         test_target=args.test_target,
                         run_mutagenesis=args.mutagenesis is not None,
                         mutagenesis_n=args.mutagenesis,
                         threshold=args.threshold,
                         model_path=getattr(args, 'model_path', None),
                         model_id=getattr(args, 'model_id', None))


if __name__ == "__main__":
    try:
        main()
    finally:
        if isinstance(sys.stdout, _Tee):
            ts_end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[log] Finished: {ts_end}")
            sys.stdout.flush()
            sys.stdout.close()
            sys.stdout = sys.__stdout__
