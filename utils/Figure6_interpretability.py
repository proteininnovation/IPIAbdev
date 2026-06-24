#!/usr/bin/env python3
"""
Figure6_interpretability.py — DELPHI · Nature Biotechnology Figure 6
=====================================================================

QUICK START
───────────
  ── IPI INTERNAL USERS ONLY ──────────────────────────────────────────
  Reproduce paper Figure 6 exactly using IPI confidential trainsets
  and pretrained models (ipi_psr_trainset.xlsx, ipi_sec_5000.xlsx):

  python utils/Figure6_interpretability.py \
      --outdir    outputs/my_fig6 \
      --model-dir trainset_042026

  NOTE: ipi_psr_trainset.xlsx and ipi_sec_5000.xlsx are confidential
  IPI datasets and are not publicly distributed.

  ── EXTERNAL USERS ───────────────────────────────────────────────────
  To reproduce Figure 6 on your own dataset, run the full pipeline.
  Trained models are automatically saved to build/pretrained_models/
  (default model directory). You can also download DELPHI pretrained
  models from Zenodo and place them in build/pretrained_models/ to
  skip training entirely.

  Step 1 — Train RF, XGBoost, and TransformerOneHot models:

    python delphi.py --train \
        --target psr_filter --lm biophysical --model rf \
        --db data/your_dataset.xlsx

    python delphi.py --train \
        --target psr_filter --lm biophysical --model xgboost \
        --db data/your_dataset.xlsx

    python delphi.py --train \
        --target psr_filter --lm onehot --model transformer_onehot \
        --db data/your_dataset.xlsx

  Step 2 — Generate interpretability figure:

    python utils/Figure6_interpretability.py \
        --db      data/your_dataset.xlsx \
        --outdir  outputs/my_fig6

  Note: --model-dir defaults to build/pretrained_models so no extra
  flags are needed if models were trained with default settings.

  ── TEST RUN (verify installation) ───────────────────────────────────
  python utils/Figure6_interpretability.py --test

WHAT THIS SCRIPT DOES
─────────────────────
  Wrapper around delphi_interpretability.py with the exact parameters
  used to generate Figure 6 in the DELPHI manuscript.

  Panels produced:
    a  Mean |IG| per sequence position — PSR
    b  Mean |IG| per sequence position — SEC
    c  HCDR3 per-residue signed IG heatmap — PSR
    d  HCDR3 per-residue signed IG heatmap — SEC
    e  Attribution convergence (RF-SHAP, XGBoost-SHAP, IG) — PSR
    f  Attribution convergence (RF-SHAP, XGBoost-SHAP, IG) — SEC

PUBLICATION PARAMETERS
──────────────────────
  --target          psr_filter
  --target2         sec_filter
  --db              data/ipi_psr_trainset.xlsx       (n = 11,265)
  --db2             data/ipi_sec_5000.xlsx           (n = 5,045)
  --model-dir       build/pretrained_models
  --ig-max-samples  0                                (all antibodies)
  --ig-steps        200
  --n-pairs         200
  --outdir          outputs/fig6_publication

LABEL CONVENTION
────────────────
  psr_filter = 1 → PASS (non-polyreactive)   ← IG attribution target
  psr_filter = 0 → FAIL (polyreactive)
  sec_filter = 1 → PASS (monomeric)
  sec_filter = 0 → FAIL (non-monomeric)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Publication-fixed parameters ──────────────────────────────────────────────
DEFAULTS = {
    "target":          "psr_filter",
    "target2":         "sec_filter",
    "db":              "data/ipi_psr_trainset.xlsx",
    "db2":             "data/ipi_sec_5000.xlsx",
    "model_dir":       "build/pretrained_models",
    "ig_max_samples":  0,
    "ig_steps":        200,
    "n_pairs":         200,
    "outdir":          "outputs/fig6_publication",
}

TEST_OVERRIDES = {
    "ig_max_samples":  100,
    "ig_steps":        50,
    "n_pairs":         5,
    "outdir":          "outputs/fig6_test",
}


def find_model(model_dir, target, lm, model):
    """Return True if a FINAL model file exists for this target/lm/model."""
    model_dir = Path(model_dir)
    pattern = f"FINAL_{target}_{lm}_{model}*"
    return any(model_dir.glob(pattern))


def auto_train(target, lm, model, db, model_dir):
    """Train a model if not already present. Returns True if training ran."""
    if find_model(model_dir, target, lm, model):
        print(f"  ✓  {target}/{lm}/{model} found — skipping training")
        return False

    print(f"  ▶  {target}/{lm}/{model} not found — training now...")
    print(f"     db={db}  model-dir={model_dir}")
    cmd = [
        sys.executable, "delphi.py", "--train",
        "--target",    target,
        "--lm",        lm,
        "--model",     model,
        "--db",        db,
        "--model-dir", model_dir,
    ]
    print("     " + " \\\n         ".join(cmd))
    print("")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ✗  Training failed for {target}/{lm}/{model}")
        sys.exit(result.returncode)
    print(f"  ✓  Training complete — {target}/{lm}/{model}")
    return True



    ap = argparse.ArgumentParser(
        description="Reproduce DELPHI Figure 6 — interpretability panels a-f.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--test",      action="store_true",
                    help="Fast test run (100 antibodies, 50 IG steps)")
    ap.add_argument("--outdir",    default=None,
                    help="Override output directory")
    ap.add_argument("--model-dir", default=None,
                    help="Override model directory (default: build/pretrained_models)")
    ap.add_argument("--db",        default=None,
                    help="Override PSR database path")
    ap.add_argument("--db2",       default=None,
                    help="Override SEC database path")
    args = ap.parse_args()

    # Build parameter dict — start from publication defaults
    params = dict(DEFAULTS)

    # Apply test overrides
    if args.test:
        params.update(TEST_OVERRIDES)
        print("▶  TEST MODE: 100 antibodies, 50 IG steps")
    else:
        print("▶  PUBLICATION MODE: all antibodies, 200 IG steps")

    # Apply CLI overrides
    if args.outdir:    params["outdir"]     = args.outdir
    if args.model_dir: params["model_dir"]  = args.model_dir
    if args.db:        params["db"]         = args.db
    if args.db2:       params["db2"]        = args.db2

    # Ensure output directory exists
    Path(params["outdir"]).mkdir(parents=True, exist_ok=True)

    # Print all parameters
    print("")
    print("  Parameters:")
    print(f"  {'target':<20}: {params['target']}")
    print(f"  {'target2':<20}: {params['target2']}")
    print(f"  {'db':<20}: {params['db']}")
    print(f"  {'db2':<20}: {params['db2']}")
    print(f"  {'model-dir':<20}: {params['model_dir']}")
    print(f"  {'ig-max-samples':<20}: {params['ig_max_samples']}  (0 = all antibodies)")
    print(f"  {'ig-steps':<20}: {params['ig_steps']}")
    print(f"  {'n-pairs':<20}: {params['n_pairs']}")
    print(f"  {'outdir':<20}: {params['outdir']}")
    print("")

    # ── Auto-train missing models ──────────────────────────────────────────────
    # For each target/db pair, check if RF, XGBoost, TransformerOneHot exist.
    # If any are missing, train them automatically before running interpretability.
    model_pairs = [
        (params["target"],  params["db"],  "biophysical", "rf"),
        (params["target"],  params["db"],  "biophysical", "xgboost"),
        (params["target"],  params["db"],  "onehot",      "transformer_onehot"),
        (params["target2"], params["db2"], "biophysical", "rf"),
        (params["target2"], params["db2"], "biophysical", "xgboost"),
        (params["target2"], params["db2"], "onehot",      "transformer_onehot"),
    ]

    print("  Checking required models...")
    trained_any = False
    for target, db, lm, model in model_pairs:
        trained = auto_train(target, lm, model, db, params["model_dir"])
        if trained:
            trained_any = True
    if not trained_any:
        print("  ✓  All models present — no training needed")
    print("")

    # ── Run interpretability ───────────────────────────────────────────────────
    cmd = [
        sys.executable, "delphi_interpretability.py",
        "--target",          params["target"],
        "--target2",         params["target2"],
        "--db",              params["db"],
        "--db2",             params["db2"],
        "--model-dir",       params["model_dir"],
        "--ig-max-samples",  str(params["ig_max_samples"]),
        "--ig-steps",        str(params["ig_steps"]),
        "--n-pairs",         str(params["n_pairs"]),
        "--outdir",          params["outdir"],
    ]

    print("  Running:")
    print("  " + " \\\n      ".join(cmd))
    print("")

    # Execute
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
