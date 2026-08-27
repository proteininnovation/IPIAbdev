#!/usr/bin/env python3
"""Run the DELPHI paper figure and analysis reproduction suite locally."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops

PAPER = Path(__file__).resolve().parent
CODE = PAPER / "figures" / "code"
DATA = PAPER / "data"
SHARE = DATA / "shareable"
LOCAL = DATA / "local_only"
BASELINE = PAPER / "figures" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PAPER.parent)
    parser.add_argument("--output-dir", type=Path, default=PAPER / "test_output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--public", action="store_true",
                      help="Run only figures reproducible from data/shareable.")
    mode.add_argument("--internal", action="store_true",
                      help="Run the complete suite, including private local inputs (default).")
    parser.add_argument("--skip-expensive", action="store_true",
                        help="Skip Figure 2, Extended Data Figure 8, and R1.")
    parser.add_argument("--only", action="append", default=[],
                        help="Run only commands whose names contain this text (repeatable).")
    return parser.parse_args()


def command_specs(repo_root: Path, output: Path, skip_expensive: bool):
    py = sys.executable
    raw = LOCAL / "raw"
    specs = [
        ("Figure 1", [py, str(CODE / "fig1.py")]),
        ("Figure 3", [py, str(CODE / "fig3.py")]),
        ("Figure 4", [py, str(CODE / "fig4.py"),
          "--figure4-data", str(SHARE / "Figure4_data.xlsx"),
          "--jain", str(SHARE / "Jain2017_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa1", str(SHARE / "GDPa1_v1.3_20251027_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa3", str(SHARE / "GDPa3_20260106_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--output-dir", str(output)]),
        ("Figure 5", [py, str(CODE / "fig5.py")]),
        ("Figure 6", [py, str(CODE / "fig6.py")]),
        ("Extended Data Figures 1-2", [py, str(CODE / "ed1_ed2.py")]),
        ("Extended Data Figure 3", [py, str(CODE / "ed3.py"),
          "--ipi-xlsx", str(SHARE / "DELPHI_Extended_Data_Figure_3_Source_Data.xlsx"),
          "--ds1-xlsx", str(SHARE / "manuscript_tables" / "DELPHI_Supplementary_Table_4.xlsx"),
          "--output-dir", str(output)]),
        ("Extended Data Figure 4", [py, str(CODE / "ed8.py"),
          "--jain", str(SHARE / "Jain2017_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa1", str(SHARE / "GDPa1_v1.3_20251027_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa3", str(SHARE / "GDPa3_20260106_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--output-dir", str(output)]),
        ("Extended Data Figure 5", [py, str(CODE / "ed4.py"),
          "--xlsx", str(raw / "Suppl_Table2_prediction_score_val.xlsx"),
          "--output-dir", str(output)]),
        ("Extended Data Figures 6-7", [py, str(CODE / "ed5_ed6.py")]),
        ("Extended Data Figure 9", [py, str(CODE / "ed9_germline_crossassay.py")]),
        ("Supplementary Figure 5", [py, str(CODE / "supp_fig5.py"),
          "--fold-predictions", str(LOCAL / "threshold" / "fold_preds_psr_filter_ablang_transformer_lm_ipi_psr_trainset_k10.csv"),
          "--output-dir", str(output / "supplementary"),
          "--repo-root", str(repo_root)]),
        ("Analysis R2/R3/R5", [py, str(PAPER / "analysis" / "R2_R3_R5_recompute.py")]),
        ("Analysis R4/R7/R8", [py, str(PAPER / "analysis" / "R7_R8_R4_tables.py")]),
    ]
    if not skip_expensive:
        specs.insert(1, ("Figure 2", [py, str(CODE / "fig2.py")]))
        specs.insert(-2, ("Extended Data Figure 8 (private render)", [py, str(CODE / "ed7.py")]))
        specs.append(("Analysis R1", [py, str(PAPER / "analysis" / "R1_germline_out.py")]))
    return specs


def public_command_specs(output: Path):
    py = sys.executable
    tables = SHARE / "manuscript_tables"
    table4 = tables / "DELPHI_Supplementary_Table_4.xlsx"
    return [
        ("Figure 3", [py, str(CODE / "fig3.py")]),
        ("Figure 4", [py, str(CODE / "fig4.py"),
          "--figure4-data", str(SHARE / "Figure4_data.xlsx"),
          "--jain", str(SHARE / "Jain2017_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa1", str(SHARE / "GDPa1_v1.3_20251027_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa3", str(SHARE / "GDPa3_20260106_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--output-dir", str(output)]),
        ("Figure 6", [py, str(CODE / "fig6.py")]),
        ("Extended Data Figure 3", [py, str(CODE / "ed3.py"),
          "--ipi-xlsx", str(SHARE / "DELPHI_Extended_Data_Figure_3_Source_Data.xlsx"),
          "--ds1-xlsx", str(table4), "--output-dir", str(output)]),
        ("Extended Data Figure 4", [py, str(CODE / "ed8.py"),
          "--jain", str(SHARE / "Jain2017_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa1", str(SHARE / "GDPa1_v1.3_20251027_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--gdpa3", str(SHARE / "GDPa3_20260106_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx"),
          "--output-dir", str(output)]),
        ("Extended Data Figure 5", [py, str(CODE / "ed4.py"),
          "--xlsx", str(table4), "--output-dir", str(output)]),
        ("Extended Data Figures 6-7", [py, str(CODE / "ed5_ed6.py")]),
    ]


def compare_pngs(output: Path):
    rows = []
    for generated in sorted(output.glob("*.png")):
        reference = BASELINE / generated.name
        if not reference.exists():
            rows.append({"file": generated.name, "status": "new/no public baseline"})
            continue
        with Image.open(generated) as left, Image.open(reference) as right:
            if left.size != right.size:
                rows.append({"file": generated.name, "status": "size differs",
                             "generated": left.size, "reference": right.size})
            else:
                bbox = ImageChops.difference(left.convert("RGBA"), right.convert("RGBA")).getbbox()
                rows.append({"file": generated.name,
                             "status": "pixel-identical" if bbox is None else "pixels differ"})
    return rows


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({
        "DELPHI_PAPER_DATA": str(DATA.resolve()),
        "DELPHI_FIGURE_OUTPUT": str(output),
        "DELPHI_REPO_ROOT": str(args.repo_root.resolve()),
        "MPLCONFIGDIR": str((output / ".matplotlib").resolve()),
        "PYTHONHASHSEED": "0",
    })

    results = []
    specs = (public_command_specs(output) if args.public
             else command_specs(args.repo_root, output, args.skip_expensive))
    if args.only:
        needles = [value.casefold() for value in args.only]
        specs = [(name, command) for name, command in specs
                 if any(needle in name.casefold() for needle in needles)]
        if not specs:
            print(f"No command names matched --only {args.only}", file=sys.stderr)
            return 2

    execution_cwd = PAPER.parent if args.public else args.repo_root
    for number, (name, command) in enumerate(specs, 1):
        print(f"[{number}] {name}", flush=True)
        started = time.monotonic()
        run = subprocess.run(command, env=env, cwd=execution_cwd,
                             text=True, capture_output=True)
        elapsed = time.monotonic() - started
        log_path = logs / f"{number:02d}_{name.lower().replace(' ', '_').replace('/', '_')}.log"
        log_path.write_text(run.stdout + "\n--- STDERR ---\n" + run.stderr, encoding="utf-8")
        results.append({"name": name, "returncode": run.returncode,
                        "seconds": round(elapsed, 2), "log": str(log_path)})
        print(f"    {'PASS' if run.returncode == 0 else 'FAIL'} ({elapsed:.1f}s)", flush=True)

    report = {
        "mode": "public" if args.public else "internal",
        "commands": results,
        "image_comparison": compare_pngs(output),
    }
    report_path = output / "reproduction_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = [row for row in results if row["returncode"]]
    print(f"Report: {report_path}")
    print(f"Summary: {len(results) - len(failures)}/{len(results)} commands passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
