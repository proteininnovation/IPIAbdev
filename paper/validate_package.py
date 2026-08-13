#!/usr/bin/env python3
"""Fast structural validation for the staged DELPHI paper package."""
from __future__ import annotations

import ast
import hashlib
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "figures" / "code"

required_scripts = [
    *(f"fig{i}.py" for i in range(1, 7)),
    "ed1_ed2.py", "ed3.py", "ed4.py", "ed5_ed6.py", "ed7.py", "ed8.py",
    "ed9_germline_crossassay.py", "supp_fig5.py", "okabe_style.py", "paths.py",
]
required_support = ["run_reproduction_tests.py", "TEST_REPORT.md"]
required_outputs = [
    *(f"Figure{i}.png" for i in range(1, 7)),
    *(f"ED_Fig{i}.png" for i in range(1, 8)),
    "ED_Fig9.png",
]
required_public_tables = [
    *(f"DELPHI_Supplementary_Table_{number}.xlsx" for number in (2, 4, 5, 7)),
    *(f"DELPHI_Extended_Data_Table_{number}.xlsx" for number in (1, 2, 3)),
    "README.md", "manifest.json",
]

errors = []
for name in required_scripts:
    path = CODE / name
    if not path.exists():
        errors.append(f"missing script: {path.relative_to(ROOT)}")
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        errors.append(f"syntax error: {name}: {exc}")

for name in required_support:
    path = ROOT / name
    if not path.exists():
        errors.append(f"missing support file: {name}")

for name in required_outputs:
    path = ROOT / "figures" / "output" / name
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"missing/empty output: {path.relative_to(ROOT)}")

public_tables = ROOT / "data" / "shareable" / "manuscript_tables"
for name in required_public_tables:
    path = public_tables / name
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"missing/empty public table: {path.relative_to(ROOT)}")

table_manifest = public_tables / "manifest.json"
if table_manifest.exists() and '"audit": "passed"' not in table_manifest.read_text(encoding="utf-8"):
    errors.append("public manuscript-table audit is not marked passed")

for path in CODE.glob("*.py"):
    text = path.read_text(encoding="utf-8")
    if "/Users/Andre" in text or "GoogleDrive-andre" in text:
        errors.append(f"machine-specific path remains: {path.relative_to(ROOT)}")

if (CODE / "threshold_optimizer.py").exists():
    errors.append("duplicated threshold_optimizer.py; import canonical utils/threshold_optimizer.py")

# Public/shareable data must never contain the private ELISA workbook or
# antibody sequence columns. Scan OOXML directly so this check has no optional
# spreadsheet-library dependency.
shareable = ROOT / "data" / "shareable"
private_names = {"ipi_psr_trainset_elisa.xlsx"}
sequence_headers = re.compile(
    rb"<t[^>]*>\s*(?:HSEQ|LSEQ|CDR3|HCDR3|VH_SEQUENCE|VL_SEQUENCE|HC_DNA_SEQUENCE|LC_DNA_SEQUENCE)\s*</t>",
    re.IGNORECASE,
)
sequence_value = re.compile(rb"(?:Biotin_)?C[ACDEFGHIKLMNPQRSTVWY]{8,30}(?=[:_\s,<]|$)")
for path in shareable.glob("*"):
    if path.name.startswith("~$"):
        continue
    if path.name.lower() in private_names:
        errors.append(f"private file in shareable data: {path.relative_to(ROOT)}")
    if path.suffix.lower() == ".xlsx":
        with zipfile.ZipFile(path) as workbook:
            payload = b"\n".join(
                workbook.read(name)
                for name in workbook.namelist()
                if name.endswith(".xml")
            )
        match = sequence_headers.search(payload)
        if match:
            errors.append(
                f"sequence column in shareable workbook: {path.relative_to(ROOT)}: "
                f"{match.group(0).decode('utf-8', errors='replace')}"
            )

for path in shareable.rglob("*.csv"):
    payload = path.read_bytes()
    match = sequence_value.search(payload)
    if match:
        errors.append(
            f"sequence-like value in shareable CSV: {path.relative_to(ROOT)}: "
            f"{match.group(0).decode('utf-8', errors='replace')}"
        )

if errors:
    print("PACKAGE VALIDATION FAILED")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

manifest_hash = hashlib.sha256((ROOT / "FIGURE_MANIFEST.md").read_bytes()).hexdigest()
print(f"PACKAGE VALIDATION PASSED: {len(required_scripts)} scripts, {len(required_outputs)} PNG renders")
print(f"manifest sha256: {manifest_hash}")
