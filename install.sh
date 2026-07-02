#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Installation Script
# Institute for Protein Innovation (IPI)
#
# Installs DELPHI and all dependencies.
# Auto-installs Miniconda if conda is not found (no sudo needed).
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

# ── Shell-safe conda initializer ─────────────────────────────────────────────
# conda.sh uses bash arithmetic that breaks in zsh.
# This function uses 'conda shell.X hook' instead, which works in all shells.
_conda_init() {
    local conda_base="$1"
    local _shell
    _shell=$(ps -p $$ -o comm= 2>/dev/null | sed 's/^-//')
    case "$_shell" in
        zsh)  eval "$("$conda_base/bin/conda" shell.zsh  hook 2>/dev/null)" ;;
        fish) eval "$("$conda_base/bin/conda" shell.fish hook 2>/dev/null)" ;;
        *)    eval "$("$conda_base/bin/conda" shell.bash hook 2>/dev/null)" ;;
    esac
}

ENV_NAME="delphi"
PYTHON_VERSION="3.11"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  DELPHI — Installation"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "  OS : $(uname -s) $(uname -m)"

# ── Step 1: Ensure conda is available (auto-install Miniconda if needed) ──────
echo ""
echo "── Step 1: Check / install conda ───────────────────────────────"

CONDA_BIN=""
for _loc in \
    "$(command -v conda 2>/dev/null)" \
    "${CONDA_EXE:-}" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda" \
    "/usr/local/miniconda3/bin/conda"; do
    [ -x "$_loc" ] && CONDA_BIN="$_loc" && break
done

# Also detect via CONDA_PREFIX (active env)
if [ -z "$CONDA_BIN" ] && [ -n "${CONDA_PREFIX:-}" ]; then
    _try="$(dirname "$CONDA_PREFIX")/bin/conda"
    [ -x "$_try" ] && CONDA_BIN="$_try"
fi

if [ -n "$CONDA_BIN" ]; then
    echo "  conda found : $($CONDA_BIN --version)"
else
    echo "  conda not found — installing Miniconda (no sudo needed)..."
    OS=$(uname -s); ARCH=$(uname -m)
    case "$OS-$ARCH" in
        Linux-x86_64)   URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
        Darwin-arm64)   URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
        Darwin-x86_64)  URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
        *)
            echo "  ERROR: Unsupported platform: $OS $ARCH"
            exit 1 ;;
    esac
    wget -q "$URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    CONDA_BIN="$HOME/miniconda3/bin/conda"
    HAS_CONDA=true
    # Source conda for this shell session immediately
    _conda_init "$HOME/miniconda3"
    # Initialize for bash and zsh future sessions
    "$CONDA_BIN" init bash 2>/dev/null || true
    "$CONDA_BIN" init zsh  2>/dev/null || true
    echo "  Miniconda installed → $HOME/miniconda3/"
    echo "  Conda initialized for bash + zsh (restart terminal or: source ~/.bashrc / source ~/.zshrc)"
fi

# ── Step 2: Create delphi conda environment ───────────────────────────────────
echo ""
echo "── Step 2: Create conda environment ($ENV_NAME, Python $PYTHON_VERSION) ─"

CONDA_BASE=$(dirname $(dirname "$CONDA_BIN"))
_conda_init "$CONDA_BASE"

# Auto-accept conda Terms of Service (required for Anaconda channels)
echo "  Accepting conda Terms of Service..."
"$CONDA_BIN" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$CONDA_BIN" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/r   2>/dev/null || true

if $CONDA_BIN env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists — skipping"
else
    $CONDA_BIN create -n $ENV_NAME python=$PYTHON_VERSION -y
    echo "  Environment '$ENV_NAME' created"
fi

conda activate $ENV_NAME

# [FIX] After `conda activate`, `python3` can still resolve to the system
#   interpreter on macOS (e.g. /usr/bin/python3 = 3.9), while the env's Python
#   lives at $CONDA_PREFIX/bin/python. Using `python3` for verification then
#   reports "No module named 'torch'" even though torch installed fine into the
#   env. Pin PY/PIP to the env explicitly and use them everywhere below.
PY="${CONDA_PREFIX:-/opt/anaconda3/envs/$ENV_NAME}/bin/python"
PIP="$PY -m pip"
if [ ! -x "$PY" ]; then
    # Fallback: ask conda for the env's python
    PY="$($CONDA_BIN run -n $ENV_NAME python -c 'import sys; print(sys.executable)' 2>/dev/null || echo python)"
    PIP="$PY -m pip"
fi
echo "  Activated : $ENV_NAME  (Python $("$PY" --version 2>&1))"
echo "  Using interpreter: $PY"

# Initialize conda for zsh too (macOS default shell)
"$CONDA_BIN" init zsh 2>/dev/null || true

# Write activate.sh — shell-aware (works in both bash and zsh)
# Write activate.sh using echo — avoids heredoc escaping issues in zsh/bash
{
    echo '#!/usr/bin/env sh'
    echo '# Generated by install.sh — run with: source activate.sh'
    echo '# Works in bash and zsh'
    echo "CONDA_BIN=\"$CONDA_BIN\""
    echo 'if [ -n "${ZSH_VERSION:-}" ]; then'
    echo '    eval "$($CONDA_BIN shell.zsh hook 2>/dev/null)"'
    echo 'else'
    echo '    eval "$($CONDA_BIN shell.bash hook 2>/dev/null)"'
    echo 'fi'
    echo "conda activate $ENV_NAME"
    # macOS: set OMP_NUM_THREADS=1 to prevent XGBoost/PyTorch OpenMP segfault
    if [ "$(uname -s)" = "Darwin" ]; then
        echo 'export OMP_NUM_THREADS=1   # macOS: prevents XGBoost+PyTorch OpenMP segfault'
    fi
    echo 'echo "  DELPHI environment activated"'
} > activate.sh
chmod +x activate.sh
echo "  activate.sh written — run: source activate.sh"

# ── Step 3: Install PyTorch (pip wheel — avoids Intel MKL Bus error) ─────────
echo ""
echo "── Step 3: Install PyTorch (GPU-aware, via pip wheel) ──────────"

# Purge any existing conda/pip PyTorch to avoid libtorch_cpu.so conflicts
$CONDA_BIN remove -n $ENV_NAME pytorch torchvision torchaudio \
    torchtriton pytorch-cuda pytorch-mutex --force -y 2>/dev/null || true
pip uninstall torch torchvision torchaudio triton -y 2>/dev/null || true

TORCH_URL=""
NOTE=""
TORCH_PKGS="torch torchvision torchaudio"
TORCH_PRE=""            # set to "--pre" if we need nightly (very new GPUs)
CUDA_VER=""
SM=""                   # GPU compute capability, e.g. 120 for Blackwell sm_120

if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" | head -1)
    # Compute capability like "12.0" → strip the dot → "120"
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | head -1 | tr -d ' .')
    echo "  GPU detected  |  driver CUDA $CUDA_VER  |  compute_cap sm_${SM:-unknown}"
fi

# ── Choose the PyTorch wheel by GPU architecture, not just driver version ─────
# The compute capability (sm_XX) decides which prebuilt kernels are needed:
#   sm_120 (Blackwell, RTX 50-series / RTX PRO 6000)  → cu128
#   sm_90  (Hopper H100) / sm_89 (Ada 40-series) / sm_86 (Ampere) / sm_80  → cu124 covers these
#   sm_75/70/60 (Turing/Volta/Pascal)                → cu121 still fine
# Falls back to driver-CUDA heuristics when compute_cap is unavailable.
if [ "$(uname -s)" = "Darwin" ]; then
    TORCH_URL=""; NOTE="macOS (MPS/CPU)"
elif [ -z "$CUDA_VER" ]; then
    TORCH_URL="https://download.pytorch.org/whl/cpu"; NOTE="CPU only (no NVIDIA GPU)"
elif [ -n "$SM" ] && [ "$SM" -ge 120 ] 2>/dev/null; then
    # Blackwell and newer — needs cu128; stable wheels ship sm_120 kernels
    TORCH_URL="https://download.pytorch.org/whl/cu128"
    NOTE="CUDA 12.8 (Blackwell sm_${SM})"
elif [ -n "$SM" ] && [ "$SM" -ge 80 ] 2>/dev/null; then
    # Ampere / Ada / Hopper — cu124 has these kernels and a wide arch list
    TORCH_URL="https://download.pytorch.org/whl/cu124"
    NOTE="CUDA 12.4 (sm_${SM})"
elif [ -n "$SM" ] && [ "$SM" -ge 70 ] 2>/dev/null; then
    # Volta / Turing — cu121 is fine
    TORCH_URL="https://download.pytorch.org/whl/cu121"
    NOTE="CUDA 12.1 (sm_${SM})"
else
    # compute_cap unknown → fall back to driver-CUDA major version
    case "${CUDA_VER}" in
        1[3-9]|2[0-9]) TORCH_URL="https://download.pytorch.org/whl/cu128"; NOTE="CUDA 12.8+ (driver $CUDA_VER)" ;;
        12)            TORCH_URL="https://download.pytorch.org/whl/cu124"; NOTE="CUDA 12.x (driver $CUDA_VER)" ;;
        11)            TORCH_URL="https://download.pytorch.org/whl/cu118"; NOTE="CUDA 11.x (driver $CUDA_VER)" ;;
        *)             TORCH_URL="https://download.pytorch.org/whl/cpu"  ; NOTE="CPU only (unrecognized CUDA)" ;;
    esac
fi

echo "  Platform : $NOTE"
if [ -n "$TORCH_URL" ]; then
    echo "  Torch wheel index: $TORCH_URL"
    $PIP install $TORCH_PRE $TORCH_PKGS --index-url "$TORCH_URL"
else
    $PIP install $TORCH_PKGS
fi

# ── Verify the GPU kernels actually match; auto-retry on nightly if not ───────
# A wheel can install cleanly yet still lack kernels for a brand-new GPU
# ("no kernel image is available"). Run a real CUDA op to confirm, and if it
# fails on an NVIDIA GPU, retry once with the cu128 nightly (newest kernels).
_gpu_smoke_test() {
    "$PY" - << 'PYEOF'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("  (no CUDA runtime — CPU/MPS build)"); sys.exit(0)
    x = torch.randn(8, 8, device="cuda")
    _ = (x @ x).sum().item()          # forces a real kernel launch
    print(f"  GPU kernel check OK  |  {torch.cuda.get_device_name(0)}  "
          f"|  archs={torch.cuda.get_arch_list()}")
    sys.exit(0)
except Exception as e:
    print(f"  GPU kernel check FAILED: {e}")
    sys.exit(3)
PYEOF
}

if [ -n "$CUDA_VER" ] && [ "$(uname -s)" != "Darwin" ]; then
    if ! _gpu_smoke_test; then
        echo ""
        echo "  GPU kernels not compatible with the installed wheel."
        echo "  Retrying with the cu128 nightly (covers the newest GPUs)..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        $PIP install --pre torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128
        if _gpu_smoke_test; then
            echo "  Nightly cu128 works on this GPU."
        else
            echo ""
            echo "  WARNING: GPU still not supported by available wheels."
            echo "  Your GPU may be newer than any released PyTorch build."
            echo "  See https://pytorch.org/get-started/locally/ for options,"
            echo "  or run DELPHI on CPU (slower) by installing the CPU wheel:"
            echo "    pip install torch torchvision torchaudio \\"
            echo "        --index-url https://download.pytorch.org/whl/cpu"
        fi
    fi
fi

# ── Step 4: Install HMMER + ANARCI ───────────────────────────────────────────
echo ""
echo "── Step 4: Install HMMER + ANARCI ──────────────────────────────"
$CONDA_BIN install -n $ENV_NAME -c bioconda hmmer anarci -y
echo "  HMMER + ANARCI installed"

# ── Step 5: Install pip packages ─────────────────────────────────────────────
echo ""
echo "── Step 5: Install pip packages ────────────────────────────────"
# Record the torch version we just installed so we can detect if a dependency
# (e.g. antiberty) silently downgrades it and breaks GPU support.
_TORCH_BEFORE=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
$PIP install -r requirements.txt
_TORCH_AFTER=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "$_TORCH_BEFORE" != "none" ] && [ "$_TORCH_BEFORE" != "$_TORCH_AFTER" ]; then
    echo ""
    echo "  NOTE: a dependency changed torch ($_TORCH_BEFORE → $_TORCH_AFTER)."
    echo "  Reinstalling the GPU-matched wheel to preserve GPU support..."
    if [ -n "$TORCH_URL" ]; then
        $PIP install --force-reinstall --no-deps $TORCH_PRE $TORCH_PKGS \
            --index-url "$TORCH_URL" 2>/dev/null || \
        $PIP install --force-reinstall --no-deps --pre torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128
    fi
fi
echo "  All packages installed"

# ── Step 6: Pre-download IgBERT weights ──────────────────────────────────────
echo ""
echo "── Step 6: Pre-download IgBERT weights ─────────────────────────"
"$PY" -c "
import subprocess, sys, os
r = subprocess.run([sys.executable, '-c',
    'from transformers import AutoTokenizer, AutoModel; '
    'AutoTokenizer.from_pretrained(\"Exscientia/IgBert\"); '
    'AutoModel.from_pretrained(\"Exscientia/IgBert\"); '
    'print(\"  IgBERT weights cached\")'],
    capture_output=True, text=True, env=os.environ.copy())
print(r.stdout.strip() if r.returncode == 0 else
      '  WARNING: IgBERT download failed — will download on first use')
" 2>/dev/null || echo "  WARNING: IgBERT download failed — will download on first use"

# ── Step 7: Verify installation ───────────────────────────────────────────────
echo ""
echo "── Step 7: Verify installation ─────────────────────────────────"
"$PY" - << 'PYEOF'
import sys, subprocess, os

env = os.environ.copy()
if os.environ.get("CONDA_PREFIX"):
    env["PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "bin") + ":" + env.get("PATH","")

def check(pkg, name, optional=False):
    r = subprocess.run([sys.executable, "-c", f"import {pkg}"],
                       capture_output=True, timeout=30, env=env)
    if r.returncode == 0:
        print(f"  OK      {name}")
        return True
    elif r.returncode == -7:
        print(f"  WARN    {name}  [Bus error at import — works at runtime]")
        return optional
    else:
        if not optional:
            print(f"  MISSING {name}  — installing...")
            r2 = subprocess.run([sys.executable, "-m", "pip", "install", pkg],
                                capture_output=True, env=env)
            if r2.returncode == 0:
                print(f"  OK      {name}  (just installed)")
                return True
            print(f"  FAIL    {name}")
        else:
            print(f"  MISSING {name}  [optional]")
        return False

ok = True
print("  Core packages:")
for pkg, name in [
    ("torch","PyTorch"), ("numpy","NumPy"), ("pandas","Pandas"),
    ("sklearn","scikit-learn"), ("xgboost","XGBoost"),
    ("captum","Captum (Integrated Gradients)"), ("shap","SHAP"),
    ("matplotlib","Matplotlib"), ("seaborn","Seaborn"),
    ("yaml","PyYAML"), ("Levenshtein","Levenshtein"),
    ("openpyxl","openpyxl"), ("pptx","python-pptx"),
]:
    if not check(pkg, name): ok = False

print("\n  Optional PLMs:")
for pkg, name in [
    ("transformers","Transformers (IgBERT / AntiBERTa2)"),
    ("ablang2","ABlang2"), ("antiberty","AntiBERTy"), ("peft","PEFT (LoRA)"),
]:
    check(pkg, name, optional=True)

print()
# Find anarci: check CONDA_PREFIX/bin/ directly + PATH
conda_prefix = os.environ.get("CONDA_PREFIX", "")
search_paths = [
    os.path.join(conda_prefix, "bin"),
    os.path.expanduser("~/miniconda3/envs/delphi/bin"),
    os.path.expanduser("~/anaconda3/envs/delphi/bin"),
] + env.get("PATH", "").split(":")

import shutil
anarci = None
for p in search_paths:
    candidate = os.path.join(p, "anarci")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        anarci = candidate
        break
if not anarci:
    anarci = shutil.which("anarci")   # last resort

if anarci:
    print(f"  OK      ANARCI  ({anarci})")
else:
    print("  MISSING ANARCI — installing...")
    import shutil as _sh
    conda = (_sh.which("conda") or
             os.path.expanduser("~/miniconda3/bin/conda"))
    subprocess.run([conda, "install", "-c", "bioconda", "hmmer", "anarci", "-y"], env=env)
    for p in search_paths:
        candidate = os.path.join(p, "anarci")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            anarci = candidate; break
    print(f"  OK      ANARCI  ({anarci})" if anarci else "  FAIL    ANARCI")
    if not anarci: ok = False

r = subprocess.run([sys.executable, "-c",
    "import torch; c=torch.cuda.is_available(); "
    "g=torch.cuda.get_device_name(0) if c else 'CPU'; "
    "print(f'  PyTorch {torch.__version__}  |  {g}')"],
    capture_output=True, text=True, env=env)
if r.returncode == 0: print(r.stdout.strip())

print()
if ok: print("  Installation complete — all core packages verified.")
else:  print("  WARNING: some packages failed — check above."); sys.exit(1)
PYEOF

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Done."
echo ""
echo "  Activate DELPHI in a new terminal:"
echo "    source activate.sh"
echo ""
echo "  Next steps:"
echo "    python utils/download_zenodo.py   # download pretrained models"
echo "    python tests/test_delphi.py       # run integration tests"
echo "══════════════════════════════════════════════════════════════════"
echo ""
