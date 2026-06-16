#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Installation Script
# Institute for Protein Innovation (IPI)
#
# Creates a dedicated conda environment and installs all dependencies.
# Automatically detects GPU and installs the correct PyTorch version.
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
echo "  conda: $(conda --version)"

# ── Create environment ────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Create conda environment ($ENV_NAME, Python $PYTHON_VERSION) ─"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists — skipping"
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    echo "  Environment '$ENV_NAME' created"
fi

# Activate environment for this script session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "  Activated: $ENV_NAME"

# ── Detect GPU and install correct PyTorch ────────────────────────────────────
echo ""
echo "── Step 2: Install PyTorch (GPU-aware, via pip wheel) ──────────"

# Use pip wheels from pytorch.org — avoids Intel MKL iJIT_NotifyEvent
# conflict that occurs when installing PyTorch via conda channel.
CUDA_VER=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "  GPU detected  |  CUDA $CUDA_VER"
else
    echo "  No GPU detected — installing CPU PyTorch"
fi

if [ -n "$CUDA_VER" ] && [ "$CUDA_VER" -ge 12 ] 2>/dev/null; then
    echo "  Installing PyTorch with CUDA 12.x support (pip)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
elif [ -n "$CUDA_VER" ] && [ "$CUDA_VER" -ge 11 ] 2>/dev/null; then
    echo "  Installing PyTorch with CUDA 11.8 support (pip)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo "  Installing CPU-only PyTorch (pip)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch works
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
if torch.cuda.is_available():
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('  CUDA: not available (CPU mode)')
"
echo "  PyTorch installed OK"

# ── HMMER + ANARCI ────────────────────────────────────────────────────────────
echo ""
echo "── Step 3: Install HMMER + ANARCI ──────────────────────────────"
conda install -c bioconda hmmer anarci -y
echo "  HMMER + ANARCI installed"

# ── All pip packages (excluding torch — already installed via conda) ───────────
echo ""
echo "── Step 4: Install pip packages ───────────────────────────────"
# Install requirements but skip torch (already installed via conda)
grep -v "^torch$\|^torch==\|^torchvision\|^torchaudio" requirements.txt \
    | pip install -r /dev/stdin
echo "  All pip packages installed"

# ── Pre-download IgBERT weights ───────────────────────────────────────────────
echo ""
echo "── Step 5: Pre-download IgBERT weights ─────────────────────────"
python3 - << 'PYEOF'
try:
    from transformers import AutoTokenizer, AutoModel
    print("  Downloading Exscientia/IgBert...")
    AutoTokenizer.from_pretrained("Exscientia/IgBert")
    AutoModel.from_pretrained("Exscientia/IgBert")
    print("  IgBERT weights cached")
except Exception as e:
    print(f"  WARNING: IgBERT download failed: {e}")
    print("  Run manually after install: python -c \"from transformers import AutoModel; AutoModel.from_pretrained('Exscientia/IgBert')\"")
PYEOF

# ── Verify installation ───────────────────────────────────────────────────────
echo ""
echo "── Step 6: Verify installation ─────────────────────────────────"
python3 - << 'PYEOF'
import sys
ok = True
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
for pkg, name in packages:
    try:
        __import__(pkg)
        print(f"  OK      {name}")
    except ImportError:
        print(f"  MISSING {name}")
        ok = False

import subprocess
result = subprocess.run(["anarci", "--help"], capture_output=True, text=True)
if result.returncode == 0:
    print("  OK      ANARCI")
else:
    print("  MISSING ANARCI — conda install -c bioconda hmmer anarci")
    ok = False

import torch
print(f"\n  PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

if not ok:
    sys.exit(1)
PYEOF

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Installation complete."
echo ""
echo "  Activate the environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Next steps:"
echo "    python utils/download_zenodo.py   # download pretrained models"
echo "    python tests/test_delphi.py       # run integration tests"
echo "══════════════════════════════════════════════════════════════════"
echo ""
