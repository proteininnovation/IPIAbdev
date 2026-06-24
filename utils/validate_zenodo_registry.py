#!/usr/bin/env python3
"""Compare released Zenodo model artifacts with config/model_registry.yaml."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


RECORD_ID = "20785877"
API_RECORD = "https://zenodo.org/api/records/{record_id}"
MODEL_EXTENSIONS = {".pt", ".pkl"}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "config" / "model_registry.yaml"


def fetch_zenodo_files(record_id: str, timeout: int) -> set[str]:
    url = API_RECORD.format(record_id=record_id)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not fetch Zenodo record {record_id}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Zenodo record {record_id} did not return valid JSON") from exc

    entries = payload.get("files", [])
    if isinstance(entries, dict):
        entries = list(entries.get("entries", {}).values())

    names: set[str] = set()
    for entry in entries:
        name = entry.get("key") or entry.get("filename")
        if not name:
            continue
        if Path(name).suffix in MODEL_EXTENSIONS:
            names.add(Path(name).name)
    return names


def read_registry_files(path: Path) -> set[str]:
    if not path.exists():
        raise RuntimeError(f"registry not found: {path}")

    model_files: set[str] = set()
    pattern = re.compile(r"^\s*model_path:\s*(\S+)\s*$")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = pattern.match(line)
        if not match:
            continue
        model_path = match.group(1).strip("\"'")
        filename = Path(model_path).name
        if Path(filename).suffix in MODEL_EXTENSIONS:
            model_files.add(filename)
        else:
            raise RuntimeError(f"line {line_number}: model_path does not end in .pt or .pkl: {model_path}")

    if not model_files:
        raise RuntimeError(f"no model_path entries found in {path}")
    return model_files


def print_names(title: str, names: set[str]) -> None:
    if not names:
        return
    print(f"\n{title} ({len(names)}):")
    for name in sorted(names):
        print(f"  - {name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate config/model_registry.yaml against released Zenodo model artifacts."
    )
    parser.add_argument("--record-id", default=RECORD_ID, help=f"Zenodo record ID (default: {RECORD_ID})")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"model registry path (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Zenodo request timeout in seconds")
    args = parser.parse_args()

    try:
        zenodo_files = fetch_zenodo_files(args.record_id, args.timeout)
        registry_files = read_registry_files(args.registry)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    missing_from_zenodo = registry_files - zenodo_files
    missing_from_registry = zenodo_files - registry_files

    print(f"Zenodo record: https://zenodo.org/records/{args.record_id}")
    print(f"Registry: {args.registry}")
    print(f"Zenodo model artifacts: {len(zenodo_files)}")
    print(f"Registry model paths: {len(registry_files)}")

    print_names("Registry entries missing from Zenodo", missing_from_zenodo)
    print_names("Zenodo model artifacts missing from registry", missing_from_registry)

    if missing_from_zenodo or missing_from_registry:
        print("\nFAIL: Zenodo model artifacts and registry entries do not match.")
        return 1

    print("\nPASS: Zenodo model artifacts and registry entries match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
