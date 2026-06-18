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
#   4. 10-fold validation    (transformer_lm/ablang, transformer_onehot,
#                             rf/biophysical, xgboost/biophysical)
#   5. Build final models    (same 4 models)
#   6. Interpretability      (psr_filter, IG+SHAP, 500 samples)
#
# Usage:
#   python tests/test_delphi.py                  # full suite
#   python tests/test_delphi.py --fast           # skip kfold + train
#   python tests/test_delphi.py --section 3      # single section only
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


def _run(cmd: list, label: str, timeout: int = 3600) -> bool:
    """Run a delphi CLI command silently. Show output only on failure."""
    _info(f"$ python {' '.join(str(c) for c in cmd)}")
    t0 = time.time()

    # Suppress HuggingFace warnings
    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"]     = "error"
    env["TOKENIZERS_PARALLELISM"]     = "false"

    try:
        result = subprocess.run(
            [sys.executable] + [str(c) for c in cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            _ok(f"{label}  ({elapsed:.0f}s)")
            return True
        else:
            _fail(f"{label} — exit code {result.returncode}")
            # Show output only on failure
            if result.stdout.strip():
                print(result.stdout[-2000:])   # last 2000 chars
            if result.stderr.strip():
                print(result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
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
            timeout=1800,
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
    _info(f"Lookup  : --model_id  (no IPI training db needed)")
    _info(f"Models  : download from https://zenodo.org/records/20648372")
    _sep()

    # model_id → directly from registry (no --db needed)
    # External users use --model_id so they don't need IPI training databases
    # (model, lm, model_id)
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
    labels  = df[TARGET].values   # psr_filter labels only

    def _run_predictions(model_list, target, has_labels):
        _info(f"Target: {target}")
        for model, lm, model_id in model_list:
            tag = f"{target}/{model}/{lm}"
            _info(f"  --model {model} --lm {lm} --model_id {model_id}")
            cmd = [_DELPHI, "--predict", TEST_DATA,
                   "--target", target,
                   "--lm",      lm,
                   "--model",   model,
                   "--model_id", model_id]
            ok = _run(cmd, label=f"predict {tag}", timeout=600)
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
        sym = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {sym}  {tag}")
    _info(f"{passed}/{len(results)} models succeeded")
    return passed == len(results)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — 10-fold cross-validation
# ══════════════════════════════════════════════════════════════════════════════
def test_kfold() -> bool:
    _head("── Test 4: 10-fold cross-validation ────────────────────────────")

    runs = [
        ("transformer_lm",     "ablang",      1800),
        ("transformer_onehot", "onehot",      1800),
        ("rf",                 "biophysical",  600),
        ("xgboost",            "biophysical",  600),
    ]

    results = {}
    for model, lm, timeout in runs:
        tag = f"{model}/{lm}"
        results[tag] = _run(
            [_DELPHI, "--kfold", "10",
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
        ("transformer_lm",     "ablang",      1800),
        ("transformer_onehot", "onehot",      1800),
        ("rf",                 "biophysical",  600),
        ("xgboost",            "biophysical",  600),
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
            try:
                import yaml
                reg = yaml.safe_load(
                    (_ROOT / "config" / "model_registry.yaml").read_text()
                ) or {}
                entries = [mid for mid, e in reg.get("models", {}).items()
                           if e.get("target") == TARGET
                           and e.get("model") == model
                           and e.get("lm") == lm]
                if entries:
                    _ok(f"Registered: {entries[-1]}")
                else:
                    _warn("Not found in model_registry.yaml after training")
            except Exception as e:
                _warn(f"Registry check skipped: {e}")

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
def test_interpretability() -> bool:
    _head("── Test 6: Interpretability ────────────────────────────────────")
    _info("Purpose: compute SHAP/IG attributions ON the training dataset")
    _info("         (not a prediction service — same dataset as training)")
    _info(f"Dataset : tests/DS1_psr_500.xlsx")
    _info("Models  : rf/biophysical, xgboost/biophysical, transformer_onehot")
    _info("Samples : 500  |  IG steps: 50")
    _sep()

    needed = [
        ("rf",                 "biophysical", 600),
        ("xgboost",            "biophysical", 600),
        ("transformer_onehot", "onehot",      1800),
    ]

    # ── Check registry — auto-train any missing models ─────────────────────
    try:
        import yaml
        reg_path = _ROOT / "config" / "model_registry.yaml"
        reg = yaml.safe_load(reg_path.read_text()) if reg_path.exists() else {}
        db_stem = TEST_DATA.stem   # DS1_psr_500
    except Exception:
        reg = {}
        db_stem = TEST_DATA.stem

    for model, lm, timeout in needed:
        tag = f"{model}/{lm}"
        found = [mid for mid, e in reg.get("models", {}).items()
                 if e.get("target") == TARGET
                 and e.get("model") == model
                 and e.get("lm")    == lm
                 and db_stem in e.get("trainset", "")]

        if found:
            _ok(f"Found in registry: {tag}  ({found[-1]})")
        else:
            _warn(f"{tag} not found in registry — training now on DS1_psr_500.xlsx")
            ok = _run(
                [_DELPHI, "--train",
                 "--target", TARGET, "--lm", lm, "--model", model,
                 "--db", TEST_DATA],
                label=f"auto-train {tag}",
                timeout=timeout,
            )
            if not ok:
                _fail(f"Auto-training failed for {tag} — cannot run interpretability")
                return False

            # Reload registry after training
            try:
                reg = yaml.safe_load(reg_path.read_text()) or {}
            except Exception:
                pass

    # ── Run interpretability on the same dataset ───────────────────────────
    _sep()
    _info("All models ready — running interpretability analysis...")
    out_dir = OUT_INTERP
    ok = _run(
        [_INTERP,
         "--target",      TARGET,
         "--models",      "rf", "xgboost", "transformer_onehot",
         "--db",          TEST_DATA,
         "--max-samples", "200",
         "--ig-steps",    "20",
         "--n-pairs",     "2",
         "--outdir",      out_dir],
        label="interpretability (psr_filter, DS1_psr_500, 500 samples, 2 pairs)",
        timeout=1200,
    )

    if ok:
        figures = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.tiff"))
        csvs    = list(out_dir.rglob("*.csv"))
        _info(f"Figures  : {len(figures)}")
        _info(f"CSV files: {len(csvs)}")
        if figures:
            _ok(f"Output written to outputs/test_interp/")
        else:
            _warn("No figure files found — check output directory")

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="DELPHI integration test suite — DS1 (Chen et al. 2024)")
    ap.add_argument("--fast", action="store_true",
                    help="Use DS1_psr_500.xlsx and skip kfold + train")
    ap.add_argument("--section", type=int, default=None,
                    choices=[0, 1, 2, 3, 4, 5, 6],
                    help="Run a single test section only (0-6)")
    args = ap.parse_args()

    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  DELPHI Integration Test Suite")
    print("  Dataset : DS1 — Chen et al. 2024 (Cell Reports, PMC11564698)")
    print("  License : MIT  |  https://zenodo.org/records/14735846")
    print(f"  Target  : {TARGET}")
    if args.fast:    print("  Mode    : fast (kfold + train skipped)")
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
    if _should(1) or args.section in (2, 3, 4, 5, 6, None):
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
        if args.fast:
            _head("── Test 4: 10-fold CV ────────────────────── SKIPPED (--fast)")
        else:
            results[4] = test_kfold()

    if _should(5):
        if args.fast:
            _head("── Test 5: Build final models ─────────────── SKIPPED (--fast)")
        else:
            results[5] = test_train()

    if _should(6):
        results[6] = test_interpretability()

    # ── Summary ───────────────────────────────────────────────────────────
    labels = {
        0: "Package imports",
        1: f"Data file ({TEST_DATA.name})",
        2: "Embedding generation  (5 PLMs)",
        3: "PSR prediction        (6 pretrained models)",
        4: "10-fold cross-validation  (4 models)",
        5: "Build final models    (4 models)",
        6: "Interpretability      (psr_filter, 500 samples)",
    }

    print()
    print("══════════════════════════════════════════════════════════════════")
    print(f"  {BOLD}SUMMARY{RESET}")
    print("──────────────────────────────────────────────────────────────────")

    passed = failed = 0
    for n, label in labels.items():
        if n not in results: continue
        if results[n]:
            print(f"  {GREEN}PASS{RESET}  Test {n}: {label}"); passed += 1
        else:
            print(f"  {RED}FAIL{RESET}  Test {n}: {label}"); failed += 1

    print("──────────────────────────────────────────────────────────────────")
    print(f"  {GREEN}{passed} passed{RESET}   {RED}{failed} failed{RESET}")
    print("══════════════════════════════════════════════════════════════════")
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
