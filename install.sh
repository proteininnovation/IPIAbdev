#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Installation Script
# Institute for Protein Innovation (IPI)
#
# Works on all platforms:
#   Linux   + NVIDIA GPU (CUDA 11 or 12)
#   Linux   + CPU only
#   macOS   + Apple Silicon (MPS)
#   macOS   + Intel (CPU)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# After installation:
#   conda activate delphi
# ══════════════════════════════════════════════════════════════════════════════

set -e

ENV_NAME="delphi"
PYTHON_VERSION="3.11"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  DELPHI — Installation"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# ── Check conda ───────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found."
    echo "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "  conda   : $(conda --version)"
echo "  OS      : $(uname -s) $(uname -m)"

# ── Create environment ────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Create conda environment ($ENV_NAME, Python $PYTHON_VERSION) ─"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists — skipping creation"
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    echo "  Environment '$ENV_NAME' created"
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "  Activated : $ENV_NAME  (Python $(python3 --version))"

# ── Detect platform and install PyTorch ──────────────────────────────────────
echo ""
echo "── Step 2: Install PyTorch (platform-aware, via pip wheel) ─────"
echo "   Using pip wheels avoids Intel MKL conflicts on Linux GPU machines."

OS=$(uname -s)
ARCH=$(uname -m)
TORCH_URL=""
TORCH_NOTE=""

if [ "$OS" = "Darwin" ]; then
    # macOS: pip install without --index-url (supports both Intel + Apple Silicon MPS)
    TORCH_URL=""
    TORCH_NOTE="macOS (Apple Silicon MPS / Intel CPU)"

elif [ "$OS" = "Linux" ]; then
    # Linux: detect CUDA version
    CUDA_VER=""
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" | head -1)
    fi

    if [ -z "$CUDA_VER" ]; then
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        TORCH_NOTE="Linux CPU-only"
    elif [ "$CUDA_VER" -ge 12 ] 2>/dev/null; then
        TORCH_URL="https://download.pytorch.org/whl/cu121"
        TORCH_NOTE="Linux + CUDA $CUDA_VER (using cu121 wheel)"
    elif [ "$CUDA_VER" -ge 11 ] 2>/dev/null; then
        TORCH_URL="https://download.pytorch.org/whl/cu118"
        TORCH_NOTE="Linux + CUDA $CUDA_VER (using cu118 wheel)"
    else
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        TORCH_NOTE="Linux CPU (CUDA $CUDA_VER not supported)"
    fi
else
    # Windows or other
    TORCH_URL=""
    TORCH_NOTE="Unknown OS — installing default PyTorch"
fi

echo "  Platform  : $TORCH_NOTE"

# Purge any existing PyTorch (conda or pip) to avoid libtorch_cpu.so conflicts
echo "  Removing any existing PyTorch installation..."
conda remove pytorch torchvision torchaudio torchtriton \
    pytorch-cuda pytorch-mutex --force -y 2>/dev/null || true
pip uninstall torch torchvision torchaudio triton -y 2>/dev/null || true

if [ -n "$TORCH_URL" ]; then
    echo "  Wheel URL : $TORCH_URL"
    pip install torch torchvision torchaudio --index-url "$TORCH_URL"
else
    pip install torch torchvision torchaudio
fi

# Verify PyTorch
python3 - << 'PYEOF'
import torch
print(f"  PyTorch   : {torch.__version__}")
if torch.cuda.is_available():
    print(f"  CUDA GPU  : {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"  MPS (Apple Silicon): available")
else:
    print(f"  Compute   : CPU only")
PYEOF

# ── HMMER + ANARCI via conda ──────────────────────────────────────────────────
echo ""
echo "── Step 3: Install HMMER + ANARCI ──────────────────────────────"
conda install -c bioconda hmmer anarci -y
echo "  HMMER + ANARCI installed"

# ── All pip packages (torch already installed above) ─────────────────────────
echo ""
echo "── Step 4: Install pip packages ────────────────────────────────"
pip install -r requirements.txt
echo "  All packages installed"

# ── Pre-download IgBERT weights ───────────────────────────────────────────────
echo ""
echo "── Step 5: Pre-download IgBERT weights ─────────────────────────"

# Test torch import first — Bus error here means PyTorch is still broken
if ! python3 -c "import torch" 2>/dev/null; then
    echo "  WARNING: torch import failed — skipping IgBERT download"
    echo "  Fix torch first, then run manually:"
    echo "    python -c \"from transformers import AutoModel; AutoModel.from_pretrained('Exscientia/IgBert')\""
else
    python3 - << 'PYEOF' || echo "  WARNING: IgBERT download failed — will download automatically on first use"
try:
    from transformers import AutoTokenizer, AutoModel
    print("  Downloading Exscientia/IgBert...")
    AutoTokenizer.from_pretrained("Exscientia/IgBert")
    AutoModel.from_pretrained("Exscientia/IgBert")
    print("  IgBERT weights cached")
except Exception as e:
    print(f"  WARNING: IgBERT download failed: {e}")
    print("  Run manually: python -c \"from transformers import AutoModel; AutoModel.from_pretrained('Exscientia/IgBert')\"")
PYEOF
fi

# ── Verify all imports ────────────────────────────────────────────────────────
echo ""
echo "── Step 6: Verify installation ─────────────────────────────────"
python3 - << 'PYEOF'
import sys, subprocess

packages = [
    ("torch",        "PyTorch"),
    ("numpy",        "NumPy"),
    ("pandas",       "Pandas"),
    ("sklearn",      "scikit-learn"),
    ("xgboost",      "XGBoost"),
    ("captum",       "Captum  (Integrated Gradients)"),
    ("shap",         "SHAP"),
    ("matplotlib",   "Matplotlib"),
    ("seaborn",      "Seaborn"),
    ("yaml",         "PyYAML"),
    ("Levenshtein",  "Levenshtein"),
    ("openpyxl",     "openpyxl"),
    ("pptx",         "python-pptx"),
    ("transformers", "Transformers (IgBERT / AntiBERTa2)"),
    ("ablang2",      "ABlang2"),
    ("antiberty",    "AntiBERTy"),
    ("peft",         "PEFT (LoRA)"),
]

ok = True
for pkg, name in packages:
    # Test each import in an isolated subprocess
    # Prevents one Bus error / crash from killing the entire verification
    r = subprocess.run(
        [sys.executable, "-c", f"import {pkg}"],
        capture_output=True, timeout=30
    )
    if r.returncode == 0:
        print(f"  OK      {name}")
    else:
        status = "BUS ERROR" if r.returncode == -7 else "MISSING"
        print(f"  {status:<10} {name}")
        if pkg == "torch":
            print("           → torch crashed — run: pip install torch --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir")
        ok = False

# ANARCI (command-line tool)
r = subprocess.run(["anarci", "--help"], capture_output=True, text=True, timeout=10)
print("  OK      ANARCI" if r.returncode == 0 else "  MISSING ANARCI")
if r.returncode != 0:
    ok = False

# CUDA check — only if torch works
r = subprocess.run(
    [sys.executable, "-c",
     "import torch; "
     "cuda=torch.cuda.is_available(); "
     "gpu=torch.cuda.get_device_name(0) if cuda else 'N/A'; "
     "print(f'PyTorch {torch.__version__}  CUDA={cuda}  GPU={gpu}')"],
    capture_output=True, text=True, timeout=30
)
if r.returncode == 0:
    print(f"\n  {r.stdout.strip()}")
else:
    print("\n  WARNING: torch import failed — CUDA check skipped")
    print("  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir")

if not ok:
    print("\n  Some packages failed. Check warnings above.")
    print("  DELPHI may still work if torch is the only failure — fix torch first.")
    sys.exit(1)
PYEOF

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Installation complete."
echo ""
echo "  Activate:   conda activate $ENV_NAME"
echo ""
echo "  Next steps:"
echo "    python utils/download_zenodo.py   # download pretrained models"
echo "    python tests/test_delphi.py       # run integration tests"
echo "══════════════════════════════════════════════════════════════════"
echo ""
