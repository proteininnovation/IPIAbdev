"""Portable paths shared by DELPHI manuscript figure scripts."""
from __future__ import annotations

import os
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DELPHI_PAPER_DATA", PAPER_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("DELPHI_FIGURE_OUTPUT", PAPER_ROOT / "figures" / "output"))
REPO_ROOT = Path(os.environ.get("DELPHI_REPO_ROOT", PAPER_ROOT.parent))


def data_file(name: str, *, required: bool = True) -> Path:
    candidates = [
        DATA_ROOT / "shareable" / name,
        DATA_ROOT / "local_only" / "raw" / name,
        DATA_ROOT / "local_only" / "embeddings" / name,
        DATA_ROOT / "local_only" / "interpretability" / name,
        DATA_ROOT / "local_only" / "threshold" / name,
        DATA_ROOT / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        searched = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Required figure input '{name}' not found. Searched:\n  {searched}")
    return candidates[0]


def ensure_output() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT
