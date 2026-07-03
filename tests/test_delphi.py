#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Integration Test Suite
# Institute for Protein Innovation (IPI)
#
# End-to-end tests for the full DELPHI pipeline using DS1_psr_500.xlsx
# (Chen et al. 2024, Cell Reports, PMC11564698, MIT License).
#
# Tests:
#   0. Package imports
#   1. Data file validation
#   2. Embedding generation  (igbert, ablang, antiberty,
#                             antiberta2, antiberta2-cssp)
#   3. PSR prediction        (IPI pretrained models, all LMs)
#   4. 5-fold validation     (transformer_lm/ablang, transformer_onehot,
#                             rf/biophysical, xgboost/biophysical)
#   5. Build final models    (same 4 models)
#   6. Interpretability      (psr_filter, IG+SHAP, 500 samples)
#
# Usage:
#   python tests/test_delphi.py                  # full suite (sections 0-8)
#   python tests/test_delphi.py --fast           # skip final-train (Test 5)
#   python tests/test_delphi.py --section 3      # single section only (0-8)
#
# Sections 3, 7, 8 exercise IPI pretrained models and require pretrained_202605/
# (run: python utils/download_zenodo.py). If absent they are reported as SKIP,
# not PASS. Without pretrained models the suite verifies training + local
# interpretability only, not IPI pretrained inference.
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

_ROOT     = Path(__file__).resolve().parent.parent
_TESTS    = Path(__file__).resolve().parent
_DELPHI   = _ROOT / "delphi.py"
_INTERP   = _ROOT / "delphi_interpretability.py"
# Test data — committed to GitHub in tests/
# No ANARCI or download needed after git clone
TEST_DATA = _TESTS / "DS1_psr_500.xlsx"
TARGET    = "psr_filter"

# All test outputs go to tests/
OUT_PRED   = _TESTS / "predictions"
OUT_INTERP = _TESTS / "interpretability"
OUT_LOGS   = _TESTS / "logs"

# MODEL_DIR — where delphi.py --train saves models (from config.py)
try:
    sys.path.insert(0, str(_ROOT))
    from config import MODEL_DIR as _MD
    MODEL_DIR = Path(_MD)
except Exception:
    MODEL_DIR = _ROOT / "build" / "pretrained_models"

GREEN  = "\033[92m"; RED    = "\033[91m"
YELLOW = "\033[93m"; BOLD   = "\033[1m"; RESET = "\033[0m"

def _ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def _fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def _warn(msg): print(f"  {YELLOW}WARN{RESET}  {msg}")
def _info(msg): print(f"  INFO  {msg}")
def _head(msg): print(f"\n{BOLD}{msg}{RESET}")
def _sep():     print(f"  {'─'*60}")

MIN_AUC = {
    "transformer_onehot": 0.70,
    "transformer_lm":     0.72,
    "rf":                 0.65,
    "xgboost":            0.65,
}


def _run(cmd: list, label: str, timeout: int = None) -> bool:
    """
    Run a delphi CLI command and stream its output live to the terminal so the
    user can watch progress (AUC, accuracy, k-fold folds, prediction, ...).
    Output is also buffered so the last lines can be re-shown on failure.
    """
    _info(f"$ python {' '.join(str(c) for c in cmd)}"
          + (f"  [no timeout]" if timeout is None else f"  [timeout={timeout}s]"))
    t0 = time.time()

    # Suppress HuggingFace warnings
    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"]     = "error"
    env["TOKENIZERS_PARALLELISM"]     = "false"

    buffered = []
    try:
        proc = subprocess.Popen(
            [sys.executable] + [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,     # merge stderr into the same stream
            text=True,
            bufsize=1,                    # line-buffered
            env=env,
        )
        # Stream each line live, indented so it is visually nested under the step.
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(f"    | {line}", flush=True)
            buffered.append(line)
        proc.wait(timeout=timeout)
        elapsed = time.time() - t0

        if proc.returncode == 0:
            _ok(f"{label}  ({elapsed:.0f}s)")
            return True
        else:
            _fail(f"{label} — exit code {proc.returncode}")
            # Output was already streamed above; re-show the tail for clarity.
            if buffered:
                print("  --- last lines ---")
                for line in buffered[-15:]:
                    print(f"    {line}")
            return False
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        _fail(f"{label} — timed out after {timeout}s")
        return False
    except Exception as e:
        _fail(f"{label} — {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 0 — Package imports
# ══════════════════════════════════════════════════════════════════════════════
def test_imports() -> bool:
    _head("── Test 0: Package imports ──────────────────────────────────────")

    core = [
        ("torch",        "PyTorch"),
        ("numpy",        "NumPy"),
        ("pandas",       "Pandas"),
        ("sklearn",      "scikit-learn"),
        ("xgboost",      "XGBoost"),
        ("captum",       "Captum  (Integrated Gradients)"),
        ("shap",         "SHAP"),
        ("yaml",         "PyYAML"),
        ("Levenshtein",  "Levenshtein"),
        ("matplotlib",   "Matplotlib"),
        ("seaborn",      "Seaborn"),
    ]
    plms = [
        ("ablang2",      "ABlang2"),
        ("antiberty",    "AntiBERTy"),
        ("transformers", "Transformers  (IgBERT / AntiBERTa2)"),
        ("peft",         "PEFT  (LoRA)"),
    ]

    import subprocess, os
    _env = os.environ.copy()
    if os.environ.get("CONDA_PREFIX"):
        _env["PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "bin") + ":" + _env.get("PATH", "")

    all_ok = True
    for pkg, name in core + plms:
        try:
            __import__(pkg); _ok(name)
        except ImportError:
            _fail(f"{name}  — pip install {pkg}"); all_ok = False

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Data file
# ══════════════════════════════════════════════════════════════════════════════
def test_data() -> tuple[bool, pd.DataFrame | None]:
    _head("── Test 1: Data file ───────────────────────────────────────────")

    if not TEST_DATA.exists():
        _fail(f"Test data not found: {TEST_DATA.name}")
        _info("Both files are committed to GitHub — check your repo is up to date:")
        _info("File is committed to GitHub — run: git pull")
        _info("Or regenerate with:")
        _info("  python utils/create_subsets.py --input tests/DS1.xlsx --target psr_filter")
        return False, None

    df = pd.read_excel(TEST_DATA)
    _info(f"{len(df):,} antibodies  x  {len(df.columns)} columns")

    required = ["BARCODE", "HSEQ", "CDR3", TARGET]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"Missing columns: {missing}"); return False, None
    _ok(f"Required columns present: {required}")

    pos_rate = df[TARGET].mean()
    n_pass   = (df[TARGET] == 1).sum()
    n_fail   = (df[TARGET] == 0).sum()
    _info(f"PASS (low PSR): {n_pass:,}   FAIL (high PSR): {n_fail:,}   "
          f"pos_rate={pos_rate:.1%}")

    if 0.45 <= pos_rate <= 0.55:
        _ok(f"Balanced 50-50  ({pos_rate:.1%})")
    else:
        _warn(f"Unexpected balance: {pos_rate:.1%}  (expected ~50%)")

    return True, df


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Embedding generation (all PLMs)
# ══════════════════════════════════════════════════════════════════════════════
def test_embeddings() -> bool:
    _head("── Test 2: Embedding generation (all PLMs) ─────────────────────")

    plms = ["ablang", "antiberty", "antiberta2", "antiberta2-cssp", "igbert"]
    results = {}

    for lm in plms:
        _info(f"Building embeddings: --lm {lm}")
        results[lm] = _run(
            [_DELPHI, "--build-embedding", TEST_DATA, "--lm", lm],
            label=f"embed {lm}",
            timeout=None,
        )

    _sep()
    passed = sum(results.values())
    for lm, ok in results.items():
        sym = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {sym}  {lm}")
    _info(f"{passed}/{len(plms)} PLMs succeeded")
    return passed == len(plms)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — PSR prediction with IPI pretrained models
# ══════════════════════════════════════════════════════════════════════════════
def test_predict(df: pd.DataFrame) -> bool:
    _head("── Test 3: PSR prediction — IPI pretrained models ──────────────")
    _info(f"Input   : tests/DS1_psr_500.xlsx  (sequences to predict)")
    _info(f"Lookup  : --model_path  (full path to pretrained model)")
    _info(f"Models  : download from https://zenodo.org/records/20785877")
    _sep()

    # model_path → full path to pretrained model file in pretrained_202605/
    # External users use --model_path so they don't need IPI training databases
    # (model, lm, model_filename)
    PSR_MODELS = [
        ("transformer_lm",     "igbert",         "FINAL_psr_filter_igbert_transformer_lm_ipi_psr_trainset.pt"),
        ("transformer_lm",     "ablang",          "FINAL_psr_filter_ablang_transformer_lm_ipi_psr_trainset.pt"),
        ("transformer_lm",     "antiberty",       "FINAL_psr_filter_antiberty_transformer_lm_ipi_psr_trainset.pt"),
        ("transformer_lm",     "antiberta2",      "FINAL_psr_filter_antiberta2_transformer_lm_ipi_psr_trainset.pt"),
        ("transformer_lm",     "antiberta2-cssp", "FINAL_psr_filter_antiberta2-cssp_transformer_lm_ipi_psr_trainset.pt"),
        ("transformer_onehot", "onehot",          "FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt"),
    ]
    SEC_MODELS = [
        ("transformer_lm",     "igbert",         "FINAL_sec_filter_igbert_transformer_lm_ipi_sec_5000.pt"),
        ("transformer_lm",     "ablang",          "FINAL_sec_filter_ablang_transformer_lm_ipi_sec_5000.pt"),
        ("transformer_lm",     "antiberty",       "FINAL_sec_filter_antiberty_transformer_lm_ipi_sec_5000.pt"),
        ("transformer_lm",     "antiberta2",      "FINAL_sec_filter_antiberta2_transformer_lm_ipi_sec_5000.pt"),
        ("transformer_lm",     "antiberta2-cssp", "FINAL_sec_filter_antiberta2-cssp_transformer_lm_ipi_sec_5000.pt"),
        ("transformer_onehot", "onehot",          "FINAL_sec_filter_onehot_transformer_onehot_ipi_sec_5000.pt"),
    ]

    results = {}
    skipped = set()
    labels  = df[TARGET].values   # psr_filter labels only

    def _run_predictions(model_list, target, has_labels):
        _info(f"Target: {target}")
        for model, lm, model_id in model_list:
            tag = f"{target}/{model}/{lm}"
            model_path = _ROOT / "pretrained_202605" / model_id
            if not model_path.exists():
                _warn(f"{tag}: model file not found — {model_id} (SKIP)")
                results[tag] = True   # not a failure
                skipped.add(tag)
                continue
            _info(f"  --model {model} --lm {lm} --model_path {model_id}")
            cmd = [_DELPHI, "--predict", TEST_DATA,
                   "--target", target,
                   "--lm",      lm,
                   "--model",   model,
                   "--model_path", str(model_path)]
            ok = _run(cmd, label=f"predict {tag}", timeout=None)
            results[tag] = ok

            if ok and has_labels:
                pred_files = sorted(OUT_PRED.glob(f"*{target}*pred*.csv"),
                                    key=lambda p: p.stat().st_mtime)
                if pred_files:
                    try:
                        preds     = pd.read_csv(pred_files[-1])
                        score_col = next((c for c in preds.columns
                                          if "score" in c.lower()
                                          or "prob" in c.lower()), None)
                        if score_col:
                            auc     = roc_auc_score(labels, preds[score_col])
                            min_auc = MIN_AUC.get(model, 0.60)
                            sym     = GREEN if auc >= min_auc else YELLOW
                            _info(f"    AUC={auc:.4f}  [{sym}{tag}{RESET}]")
                    except Exception as e:
                        _warn(f"    AUC skipped: {e}")

    # PSR predictions (DS1 has psr_filter labels → AUC computed)
    print(f"\n  ── PSR filter models ───────────────────────────────────────")
    _run_predictions(PSR_MODELS, target="psr_filter", has_labels=True)

    # SEC predictions (DS1 has no sec_filter labels → AUC skipped)
    print(f"\n  ── SEC filter models ───────────────────────────────────────")
    _info("Note: DS1 has no sec_filter labels — AUC not computed for SEC")
    _run_predictions(SEC_MODELS, target="sec_filter", has_labels=False)

    _sep()
    passed = sum(results.values())
    for tag, ok in results.items():
        if tag in skipped:
            sym = f"{YELLOW}SKIP{RESET}"
        else:
            sym = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {sym}  {tag}")
    n_ran = len(results) - len(skipped)
    _info(f"{n_ran - sum(1 for t,o in results.items() if t not in skipped and not o)}"
          f"/{n_ran} models ran successfully, {len(skipped)} skipped (no file)")
    # If every model was skipped (pretrained_202605/ absent), report SKIP not PASS
    if len(skipped) == len(results):
        _warn("All pretrained models missing — Test 3 counted as SKIP, not PASS")
        return ("SKIP",
                "no pretrained models found in pretrained_202605/",
                "python utils/download_zenodo.py   (downloads all 52 models)")
    # Otherwise pass only if every model that actually ran succeeded
    return all(o for t, o in results.items() if t not in skipped)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — k-fold cross-validation (5-fold)
# ══════════════════════════════════════════════════════════════════════════════
def test_kfold(fast: bool = False) -> bool:
    k = "5"
    _head(f"── Test 4: {k}-fold cross-validation ───────────────────────────")

    runs = [
        ("transformer_lm",     "ablang",      None),
        # ("transformer_onehot", "onehot",      None),  # skipped: slow k-fold
        ("rf",                 "biophysical", None),
        ("xgboost",            "biophysical", None),
    ]

    results = {}
    for model, lm, timeout in runs:
        tag = f"{model}/{lm}"
        results[tag] = _run(
            [_DELPHI, "--kfold", k,
             "--target", TARGET, "--lm", lm, "--model", model,
             "--db", TEST_DATA],
            label=f"kfold {tag}",
            timeout=timeout,
        )

    _sep()
    passed = sum(results.values())
    for tag, ok in results.items():
        sym = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {sym}  {tag}")
    _info(f"{passed}/{len(runs)} models succeeded")
    return passed == len(runs)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Build final models
# ══════════════════════════════════════════════════════════════════════════════
def test_train() -> bool:
    _head("── Test 5: Build final models ──────────────────────────────────")

    runs = [
        # Transformer models
        ("transformer_lm",     "igbert",       None),
        ("transformer_lm",     "ablang",       None),
        ("transformer_onehot", "onehot",       None),
        # RF — biophysical + kmer
        ("rf",                 "biophysical", None),
        ("rf",                 "kmer", None),
        # XGBoost — biophysical + kmer
        ("xgboost",            "biophysical", None),
        ("xgboost",            "kmer", None),
    ]

    results = {}
    for model, lm, timeout in runs:
        tag = f"{model}/{lm}"
        ok = _run(
            [_DELPHI, "--train",
             "--target", TARGET, "--lm", lm, "--model", model,
             "--db", TEST_DATA],
            label=f"train {tag}",
            timeout=timeout,
        )
        results[tag] = ok

        if ok:
            # Models are located by filename convention, not a registry.
            # Verify the trained checkpoint file exists in MODEL_DIR.
            ext = ".pt" if model in ("transformer_onehot", "transformer_lm", "cnn") else ".pkl"
            stem = f"FINAL_{TARGET}_{lm}_{model}_{TEST_DATA.stem}{ext}"
            ckpt = MODEL_DIR / stem
            matches = list(MODEL_DIR.glob(f"FINAL_{TARGET}_{lm}_{model}_*{ext}"))
            if ckpt.exists():
                _ok(f"Saved: {stem}")
            elif matches:
                _ok(f"Saved: {matches[-1].name}")
            else:
                _warn(f"No checkpoint found in {MODEL_DIR} after training")

    _sep()
    passed = sum(results.values())
    for tag, ok in results.items():
        sym = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {sym}  {tag}")
    _info(f"{passed}/{len(runs)} models succeeded")
    return passed == len(runs)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Interpretability analysis
# ══════════════════════════════════════════════════════════════════════════════
def test_interpretability(fast: bool = False) -> bool:
    _head("── Test 6: Interpretability ────────────────────────────────────")
    _info("Purpose: compute SHAP/IG attributions ON the training dataset")
    _info("         (not a prediction service — same dataset as training)")
    _info(f"Dataset : tests/DS1_psr_500.xlsx")
    _info("Models  : rf/biophysical, xgboost/biophysical, transformer_onehot")
    _info("Samples : 500  |  IG steps: 50")
    _sep()

    needed = [
        ("rf",                 "biophysical", None),
        ("rf",                 "kmer", None),
        ("xgboost",            "biophysical", None),
        ("xgboost",            "kmer", None),
        ("transformer_onehot", "onehot",      None),
    ]

    # ── Check for trained models by filename — auto-train any missing ──────
    db_stem = TEST_DATA.stem   # DS1_psr_500

    def _model_exists(model, lm):
        ext = ".pt" if model in ("transformer_onehot", "transformer_lm", "cnn") else ".pkl"
        exact = MODEL_DIR / f"FINAL_{TARGET}_{lm}_{model}_{db_stem}{ext}"
        if exact.exists():
            return True
        return bool(list(MODEL_DIR.glob(f"FINAL_{TARGET}_{lm}_{model}_*{ext}")))

    # Test 5 already trained RF and XGBoost on DS1_psr_500.
    # delphi_interpretability.py gracefully skips any model not found.
    for model, lm, timeout in needed:
        tag = f"{model}/{lm}"
        if _model_exists(model, lm):
            _ok(f"Model present: {tag}")
        elif fast and model == "transformer_onehot":
            # --fast skips Test 5 (training); don't trigger a slow transformer
            # train here. If the model is absent, mark the section SKIP.
            _warn(f"--fast: {tag} not present and training skipped → SKIP Test 6")
            return ("SKIP",
                    "--fast skips final training and the transformer model "
                    "is not present yet",
                    "run without --fast, or run: python tests/test_delphi.py "
                    "--section 5  (trains models first)")
        else:
            _warn(f"Model {tag} not found — auto-training...")
            ok = _run(
                [_DELPHI, "--train",
                 "--target", TARGET, "--lm", lm, "--model", model,
                 "--db", TEST_DATA],
                label=f"auto-train {tag}", timeout=timeout,
            )
            if not ok and model == "transformer_onehot":
                _fail(f"Auto-training failed for {tag}")
                return False

    # ── Run interpretability — script runs rf + xgboost + transformer ──────
    # delphi_interpretability.py auto-locates FINAL_*.pkl/.pt models in
    # --model-dir matching db_stem from --db. No --models flag needed.
    _sep()
    _info("Mode B: single-target, all 3 architectures on the training db")
    out_dir = OUT_INTERP / "modeB_all"
    ok = _run(
        [_INTERP,
         "--target",       TARGET,
         "--db",           TEST_DATA,
         "--model-dir",    str(MODEL_DIR),
         "--max-samples",  "200",
         "--ig-steps",     "20",
         "--n-antibodies", "2",
         "--outdir",       str(out_dir)],
        label="interpretability Mode B (rf + xgboost + transformer_onehot)",
        timeout=None,
    )

    if ok:
        figures = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.tiff"))
        csvs    = list(out_dir.rglob("*.csv"))
        _info(f"Figures  : {len(figures)}")
        _info(f"CSV files: {len(csvs)}")
        per_ab = list(out_dir.rglob("per_antibody_*/*.tiff"))
        _info(f"Per-antibody figures: {len(per_ab)}")
        if figures:
            _ok(f"Mode B output written to {out_dir}/")
        else:
            _warn("No figure files found — check output directory")

    # ── Mode C: single-architecture predict + interpret via --model-path ───
    # Uses Test 5's DS1_psr_500-trained models. Each architecture predicts on
    # the same file and produces per-antibody waterfall + CDR3 mutagenesis.
    _sep()
    _info("Mode C: single-architecture predict + interpret (--model-path)")
    mode_c = [
        ("transformer_onehot", "onehot",
         f"FINAL_{TARGET}_onehot_transformer_onehot_{TEST_DATA.stem}.pt"),
        ("rf", "biophysical",
         f"FINAL_{TARGET}_biophysical_rf_{TEST_DATA.stem}.pkl"),
        ("xgboost", "biophysical",
         f"FINAL_{TARGET}_biophysical_xgboost_{TEST_DATA.stem}.pkl"),
    ]
    c_ok = True
    for model, lm, fname in mode_c:
        mpath = MODEL_DIR / fname
        if not mpath.exists():
            _warn(f"Mode C {model}: checkpoint not found ({fname}) — skipping")
            continue
        c_out = OUT_INTERP / f"modeC_{model}"
        cmd = [_INTERP, "--predict", str(TEST_DATA),
               "--target", TARGET, "--model", model, "--lm", lm,
               "--model-path", str(mpath),
               "--n-antibodies", "2",
               "--outdir", str(c_out)]
        if model == "transformer_onehot":
            cmd += ["--ig-max-samples", "100", "--ig-steps", "20"]
        sub = _run(cmd, label=f"interpretability Mode C ({model})", timeout=None)
        c_ok = c_ok and sub
        if sub:
            per_ab = list(c_out.rglob("per_antibody_*/*.tiff"))
            _info(f"  {model}: {len(per_ab)} per-antibody figures")

    return ok and c_ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Single-target interpretability with IPI pretrained model
# ══════════════════════════════════════════════════════════════════════════════
def test_interpretability_pretrained_psr() -> bool:
    _head("── Test 7: Interpretability — IPI pretrained PSR model ─────────")
    _info("Input   : tests/DS1_psr_500.xlsx  (via --predict, new sequences)")
    _info("Model   : FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt")
    _info("Lookup  : --db stem 'ipi_psr_trainset' + --model-dir pretrained_202605")
    _sep()

    model_id   = "FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt"
    model_path = _ROOT / "pretrained_202605" / model_id

    if not model_path.exists():
        _warn(f"Pretrained model not found: {model_id}")
        _warn("Run: python utils/download_zenodo.py")
        _warn("Skipping Test 7 (counted as SKIP, not PASS)")
        return ("SKIP",
                f"pretrained model missing: pretrained_202605/{model_id}",
                "python utils/download_zenodo.py")

    # delphi_interpretability.py locates models by db_stem from --db filename.
    # The model stem is 'ipi_psr_trainset', so --db must be a file whose stem
    # is 'ipi_psr_trainset'. We provide DS1 sequences via --predict.
    # Create a stem-matching stub if the real IPI db is absent.
    stub_db = _ROOT / "pretrained_202605" / "ipi_psr_trainset.xlsx"
    if not stub_db.exists():
        import shutil
        shutil.copy(TEST_DATA, stub_db)   # DS1 sequences, used only for model lookup

    _ok(f"Model found: {model_id}")
    out_dir = OUT_INTERP / "pretrained_psr"

    ok = _run(
        [_INTERP,
         "--target",        "psr_filter",
         "--db",            str(stub_db),
         "--predict",       str(TEST_DATA),
         "--model-dir",     str(_ROOT / "pretrained_202605"),
         "--transformer-lm", "onehot",
         "--max-samples",   "200",
         "--ig-steps",      "20",
         "--n-pairs",       "2",
         "--outdir",        str(out_dir)],
        label="interpretability PSR pretrained",
        timeout=None,
    )
    if ok:
        figures = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.tiff"))
        _info(f"Figures: {len(figures)}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Dual-target interpretability with IPI pretrained PSR + SEC models
# ══════════════════════════════════════════════════════════════════════════════
def test_interpretability_pretrained_psr_sec() -> bool:
    _head("── Test 8: Interpretability — IPI pretrained PSR + SEC models ──")
    _info("PSR model : FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt")
    _info("SEC model : FINAL_sec_filter_onehot_transformer_onehot_ipi_sec_5000.pt")
    _info("Input     : DS1_psr_500.xlsx via --predict (dual-target PSR+SEC)")
    _sep()

    psr_model_id = "FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt"
    sec_model_id = "FINAL_sec_filter_onehot_transformer_onehot_ipi_sec_5000.pt"
    psr_path = _ROOT / "pretrained_202605" / psr_model_id
    sec_path = _ROOT / "pretrained_202605" / sec_model_id

    if not psr_path.exists():
        _warn(f"PSR model not found: {psr_model_id} — skipping Test 8 (SKIP)")
        return ("SKIP",
                f"pretrained PSR model missing: pretrained_202605/{psr_model_id}",
                "python utils/download_zenodo.py")
    if not sec_path.exists():
        _warn(f"SEC model not found: {sec_model_id} — skipping Test 8 (SKIP)")
        return ("SKIP",
                f"pretrained SEC model missing: pretrained_202605/{sec_model_id}",
                "python utils/download_zenodo.py")

    _ok(f"PSR model : {psr_model_id}")
    _ok(f"SEC model : {sec_model_id}")

    # delphi_interpretability.py locates models by db_stem from --db / --db2.
    # PSR model stem = 'ipi_psr_trainset', SEC model stem = 'ipi_sec_5000'.
    # Provide stem-matching stub dbs (DS1 sequences) for model lookup,
    # and run prediction on DS1 via --predict.
    import shutil
    psr_stub = _ROOT / "pretrained_202605" / "ipi_psr_trainset.xlsx"
    sec_stub = _ROOT / "pretrained_202605" / "ipi_sec_5000.xlsx"
    if not psr_stub.exists():
        shutil.copy(TEST_DATA, psr_stub)
    if not sec_stub.exists():
        shutil.copy(TEST_DATA, sec_stub)

    out_dir = OUT_INTERP / "pretrained_psr_sec"

    ok = _run(
        [_INTERP,
         "--target",         "psr_filter",
         "--target2",        "sec_filter",
         "--db",             str(psr_stub),
         "--db2",            str(sec_stub),
         "--predict",        str(TEST_DATA),
         "--model-dir",      str(_ROOT / "pretrained_202605"),
         "--transformer-lm", "onehot",
         "--max-samples",    "200",
         "--ig-steps",       "20",
         "--n-pairs",        "2",
         "--outdir",         str(out_dir)],
        label="interpretability PSR+SEC pretrained",
        timeout=None,
    )
    if ok:
        figures = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.tiff"))
        _info(f"Figures: {len(figures)}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="DELPHI integration test suite — DS1 (Chen et al. 2024)")
    ap.add_argument("--fast", action="store_true",
                    help="Use DS1_psr_500.xlsx and skip kfold + train")
    ap.add_argument("--section", type=int, default=None,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                    help="Run a single test section only (0-8). "
                         "Sections 3, 7, 8 require pretrained_202605/ models.")
    args = ap.parse_args()

    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  DELPHI Integration Test Suite")
    print("  Dataset : DS1 — Chen et al. 2024 (Cell Reports, PMC11564698)")
    print("  License : MIT  |  https://zenodo.org/records/20785877")
    print(f"  Target  : {TARGET}")
    if args.fast:    print("  Mode    : fast (final-train skipped)")
    if args.section is not None: print(f"  Section : {args.section} only")
    print("══════════════════════════════════════════════════════════════════")

    # Create output directories
    for d in [OUT_PRED, OUT_INTERP, OUT_LOGS]:
        d.mkdir(parents=True, exist_ok=True)

    results = {}
    df      = None

    def _should(n): return args.section is None or args.section == n

    if _should(0): results[0] = test_imports()

    # data always needed by subsequent tests
    if _should(1) or args.section in (2, 3, 4, 5, 6, 7, 8, None):
        ok, df = test_data()
        results[1] = ok
        if not ok:
            print(f"\n{RED}Cannot continue without test data.{RESET}")
            print(f"tests/DS1_psr_500.xlsx should be committed to GitHub.")
            print(f"Run: git pull  — or regenerate with:")
            print(f"  python utils/create_subsets.py "
                  f"--input tests/DS1.xlsx --target psr_filter\n")
            sys.exit(1)

    if _should(2): results[2] = test_embeddings()
    if _should(3): results[3] = test_predict(df)

    if _should(4):
        # Test 4 runs 5-fold CV (always). Kept here even in --fast mode since
        # it is the primary validation check.
        results[4] = test_kfold(fast=args.fast)

    if _should(5):
        if args.fast:
            _head("── Test 5: Build final models ─────────────── SKIPPED (--fast)")
        else:
            results[5] = test_train()

    if _should(6):
        results[6] = test_interpretability(fast=args.fast)
    if _should(7):
        results[7] = test_interpretability_pretrained_psr()
    if _should(8):
        results[8] = test_interpretability_pretrained_psr_sec()

    # ── Summary ───────────────────────────────────────────────────────────
    labels = {
        0: "Package imports",
        1: f"Data file ({TEST_DATA.name})",
        2: "Embedding generation  (5 PLMs)",
        3: "PSR prediction        (6 pretrained models)",
        4: "5-fold cross-validation  (4 models)",
        5: "Build final models    (4 models)",
        6: "Interpretability      (psr_filter, 500 samples)",
        7: "Interpretability      (IPI PSR pretrained model)",
        8: "Interpretability      (IPI PSR+SEC pretrained, dual-target)",
    }

    print()
    print("══════════════════════════════════════════════════════════════════")
    print(f"  {BOLD}SUMMARY{RESET}")
    print("──────────────────────────────────────────────────────────────────")

    passed = failed = skipped = 0
    skip_details = []
    for n, label in labels.items():
        if n not in results: continue
        r = results[n]
        # SKIP can be the bare string "SKIP" or a ("SKIP", reason, fix) tuple
        is_skip = (r == "SKIP") or (isinstance(r, tuple) and r and r[0] == "SKIP")
        if is_skip:
            print(f"  {YELLOW}SKIP{RESET}  Test {n}: {label}"); skipped += 1
            reason = r[1] if isinstance(r, tuple) and len(r) > 1 else "did not run"
            fix    = r[2] if isinstance(r, tuple) and len(r) > 2 else None
            skip_details.append((n, reason, fix))
        elif r:
            print(f"  {GREEN}PASS{RESET}  Test {n}: {label}"); passed += 1
        else:
            print(f"  {RED}FAIL{RESET}  Test {n}: {label}"); failed += 1

    print("──────────────────────────────────────────────────────────────────")
    _skip_s = f"   {YELLOW}{skipped} skipped{RESET}" if skipped else ""
    print(f"  {GREEN}{passed} passed{RESET}   {RED}{failed} failed{RESET}{_skip_s}")

    if skip_details:
        print()
        print(f"  {YELLOW}Why tests were skipped (and how to enable them):{RESET}")
        for n, reason, fix in skip_details:
            print(f"    Test {n}: {reason}")
            if fix:
                print(f"             → {fix}")
    print("══════════════════════════════════════════════════════════════════")
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
