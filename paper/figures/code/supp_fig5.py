#!/usr/bin/env python3
"""Reproduce Supplementary Figure 5 from pooled 10-fold OOF predictions.

The calculation stays canonical in ``utils/threshold_optimizer.py``.  This
paper-facing wrapper only supplies manuscript inputs, labels, and output paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="DELPHI repository root containing utils/threshold_optimizer.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    from utils.threshold_optimizer import run_full_threshold_pipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_full_threshold_pipeline(
        fold_preds_csv=str(args.fold_predictions),
        target="psr_filter",
        lm="ablang",
        model="transformer_lm",
        db_stem="ipi_psr_trainset_k10",
        output_dir=str(args.output_dir),
        cost_fp=1.0,
        cost_fn=3.0,
    )


if __name__ == "__main__":
    main()
