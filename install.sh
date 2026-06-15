#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Installation Script
# Institute for Protein Innovation (IPI)
#
# Creates a dedicated conda environment and installs all dependencies.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# After installation, activate the environment with:
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

# ── Check conda is available ──────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found."
    echo "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "  conda found — OK"

# ── Create conda environment ──────────────────────────────────────────────────
echo ""
echo "── Step 1: Creating conda environment ($ENV_NAME, Python $PYTHON_VERSION) ─"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists — skipping creation"
    echo "  To recreate from scratch: conda env remove -n $ENV_NAME"
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    echo "  Environment '$ENV_NAME' created"
fi

# ── Install ANARCI (conda) ────────────────────────────────────────────────────
echo ""
echo "── Step 2: Installing HMMER + ANARCI (explicit) ────────────────"
conda install -n $ENV_NAME -c bioconda hmmer anarci -y
echo "  HMMER + ANARCI installed"

# ── Install all pip packages ──────────────────────────────────────────────────
echo ""
echo "── Step 3: Installing pip packages ────────────────────────────"
conda run -n $ENV_NAME pip install -r requirements.txt
echo "  All packages installed"

# ── Pre-download IgBERT weights ───────────────────────────────────────────────
echo ""
echo "── Step 4: Pre-downloading IgBERT weights from HuggingFace ────"
conda run -n $ENV_NAME python3 -c "
from transformers import AutoTokenizer, AutoModel
print('  Downloading Exscientia/IgBert...')
AutoTokenizer.from_pretrained('Exscientia/IgBert')
AutoModel.from_pretrained('Exscientia/IgBert')
print('  IgBERT weights cached')
" || echo "  WARNING: IgBERT download failed — check internet connection"

# ── Verify key imports ────────────────────────────────────────────────────────
echo ""
echo "── Step 5: Verifying installation ─────────────────────────────"
conda run -n $ENV_NAME python3 -c "
packages = [
    ('torch',        'PyTorch'),
    ('numpy',        'NumPy'),
    ('pandas',       'Pandas'),
    ('sklearn',      'scikit-learn'),
    ('xgboost',      'XGBoost'),
    ('captum',       'Captum  (Integrated Gradients)'),
    ('shap',         'SHAP'),
    ('matplotlib',   'Matplotlib'),
    ('seaborn',      'Seaborn'),
    ('yaml',         'PyYAML'),
    ('Levenshtein',  'Levenshtein'),
    ('openpyxl',     'openpyxl'),
    ('pptx',         'python-pptx'),
    ('transformers', 'Transformers (IgBERT / AntiBERTa2)'),
    ('ablang2',      'ABlang2'),
    ('antiberty',    'AntiBERTy'),
    ('peft',         'PEFT (LoRA)'),
]
ok = True
for pkg, name in packages:
    try:
        __import__(pkg)
        print(f'  OK      {name}')
    except ImportError:
        print(f'  MISSING {name}')
        ok = False

# Check ANARCI separately (command-line tool)
import subprocess
result = subprocess.run(['anarci', '--help'],
                       capture_output=True, text=True)
if result.returncode == 0:
    print('  OK      ANARCI')
else:
    print('  MISSING ANARCI — conda install -c bioconda anarci')
    ok = False

if not ok:
    raise SystemExit(1)
"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Installation complete."
echo ""
echo "  Activate the environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Quick start:"
echo "    python delphi.py --list-models"
echo "    python delphi.py --help"
echo "══════════════════════════════════════════════════════════════════"
echo ""
