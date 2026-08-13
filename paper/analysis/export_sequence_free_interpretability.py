#!/usr/bin/env python3
"""Export sequence-free DELPHI interpretability source tables.

Private inputs are read but never changed. Literal CDR3 sequences and
sequence-bearing barcode values are excluded from all exported CSV files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "data" / "local_only" / "interpretability"
OUTPUT = PAPER / "data" / "shareable" / "interpretability"
AA_ORDER = list("RKH" "DE" "WFY" "AGILMPV" "STNQC")
SEQUENCE_RE = re.compile(r"(?:Biotin_)?C[ACDEFGHIKLMNPQRSTVWY]{6,30}(?=[:_\s]|$)")


def write_csv(frame: pd.DataFrame, name: str) -> dict:
    path = OUTPUT / name
    frame.to_csv(path, index=False)
    return {"file": name, "rows": len(frame), "columns": list(frame.columns)}


def aggregate_ig(source_name: str, assay: str) -> list[dict]:
    data = pd.read_csv(SOURCE / source_name, low_memory=False)
    cdr_cols = sorted(c for c in data.columns if c.startswith("ig_CDR3_"))
    vh_cols = sorted(c for c in data.columns if c.startswith("ig_VH_"))

    positional = []
    for region, columns in (("CDR3", cdr_cols), ("VH", vh_cols)):
        for column in columns:
            values = pd.to_numeric(data[column], errors="coerce")
            positional.append({
                "assay": assay,
                "region": region,
                "position": int(column.rsplit("_", 1)[1]),
                "mean_abs_ig": values.abs().mean(),
                "n": int(values.notna().sum()),
            })

    heatmap = []
    sequences = data["hcdr3_seq"].fillna("").astype(str).to_numpy()
    for position, column in enumerate(cdr_cols, start=1):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        residues = np.array([seq[position - 1] if len(seq) >= position else "" for seq in sequences])
        for aa in AA_ORDER:
            selected = (residues == aa) & np.isfinite(values)
            heatmap.append({
                "assay": assay,
                "position": position,
                "aa": aa,
                "mean_signed_ig": float(values[selected].mean()) if selected.any() else np.nan,
                "n": int(selected.sum()),
            })

    return [
        write_csv(pd.DataFrame(positional), f"fig6_{assay.lower()}_ig_by_position.csv"),
        write_csv(pd.DataFrame(heatmap), f"fig6_{assay.lower()}_ig_by_aa_position.csv"),
    ]


def sanitized_copy(source_name: str, output_name: str, drop: tuple[str, ...] = ()) -> dict:
    data = pd.read_csv(SOURCE / source_name, low_memory=False)
    data = data.drop(columns=[column for column in drop if column in data.columns])
    return write_csv(data, output_name)


def audit_sequence_free() -> list[str]:
    errors = []
    forbidden_headers = {"hcdr3_seq", "cdr3", "hcdr3", "hseq", "lseq", "vh_sequence", "vl_sequence"}
    for path in sorted(OUTPUT.glob("*.csv")):
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
        bad_headers = forbidden_headers.intersection(c.casefold() for c in data.columns)
        if bad_headers:
            errors.append(f"{path.name}: forbidden headers {sorted(bad_headers)}")
        for column in data.columns:
            hits = data[column].str.contains(SEQUENCE_RE, regex=True, na=False)
            if hits.any():
                errors.append(f"{path.name}: {int(hits.sum())} sequence-like values in {column}")
    return errors


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    records += aggregate_ig("ig_FULL_psr_filter_onehot_ipi_psr_trainset.csv", "PSR")
    records += aggregate_ig("ig_FULL_sec_filter_onehot_ipi_sec_5000.csv", "SEC")

    copy_specs = [
        ("region_attribution_psr_filter_ipi_psr_trainset.csv", "region_attribution_psr_filter_ipi_psr_trainset.csv", ()),
        ("region_attribution_sec_filter_ipi_sec_5000.csv", "region_attribution_sec_filter_ipi_sec_5000.csv", ()),
        ("shap_xgb_FULL_psr_filter_biophysical_ipi_psr_trainset.csv", "shap_xgb_FULL_psr_filter_biophysical_ipi_psr_trainset_sequence_free.csv", ("barcode",)),
        ("shap_xgb_FULL_sec_filter_biophysical_ipi_sec_5000.csv", "shap_xgb_FULL_sec_filter_biophysical_ipi_sec_5000_sequence_free.csv", ("barcode",)),
    ]
    for source_name, output_name, drop in copy_specs:
        records.append(sanitized_copy(source_name, output_name, drop))

    for path in sorted(SOURCE.glob("interp_*_beeswarm_*.csv")):
        records.append(sanitized_copy(path.name, path.name, ("barcode",)))

    errors = audit_sequence_free()
    manifest = {
        "source_directory": "paper/data/local_only/interpretability",
        "policy": "No literal IPI VH, VL, HSEQ, LSEQ, CDR3, or HCDR3 sequences; sequence-bearing barcodes removed.",
        "files": records,
        "audit": "passed" if not errors else "failed",
        "audit_errors": errors,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("Sequence-free audit failed:\n- " + "\n- ".join(errors))
    print(f"Exported {len(records)} sequence-free CSV files to {OUTPUT}")
    print("Sequence audit: PASSED")


if __name__ == "__main__":
    main()
