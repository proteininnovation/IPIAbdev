#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Test Pipeline
# Institute for Protein Innovation (IPI)
#
# Runs the full DELPHI test pipeline:
#   Step 0. Check core package imports (fails fast before any download).
#   Step 1. Ensure data/DS1.xlsx is present (download from Zenodo if missing),
#           then regenerate the balanced 500-antibody test subset
#           tests/DS1_psr_500.xlsx from it.
#   Step 2. Run the integration test suite (tests/test_delphi.py).
#   Step 3. Collect all plots and logs into tests/results/.
#
# DS1 is public data hosted on Zenodo, so this works for everyone.
#
# Usage (run with bash or sh, or ./ — NOT python):
#   ./test_delphi.sh                 # full pipeline
#   ./test_delphi.sh --fast          # flags pass through to test_delphi.py
#   ./test_delphi.sh --section 3
# ══════════════════════════════════════════════════════════════════════════════

set -e

# Resolve the repo root. Works under both bash and sh, whether this script sits
# in the repo root or in tests/.
_src="$0"
_script_dir="$(cd "$(dirname "$_src")" 2>/dev/null && pwd)"
[ -z "$_script_dir" ] && _script_dir="$(pwd)"

ROOT=""
for _cand in "$_script_dir" "$_script_dir/.." "$(pwd)" "$(pwd)/.."; do
    if [ -d "$_cand/utils" ] && [ -d "$_cand/tests" ] \
       && [ -f "$_cand/tests/test_delphi.py" ]; then
        ROOT="$(cd "$_cand" && pwd)"
        break
    fi
done
if [ -z "$ROOT" ]; then
    echo "  ERROR: could not locate the DELPHI repo root (need utils/ + tests/)."
    echo "  Run from the repo root, e.g.:  bash test_delphi.sh"
    exit 1
fi
cd "$ROOT"

FULL_DATA="data/DS1.xlsx"
TEST_DATA="tests/DS1_psr_500.xlsx"

# DS1 is public data on Zenodo. Direct-download URL for the DS1.xlsx file.
# NOTE: confirm the record ID and exact filename match your Zenodo deposit.
# Zenodo direct-download pattern: https://zenodo.org/records/<ID>/files/<NAME>?download=1
ZENODO_URL="https://zenodo.org/records/20785877/files/DS1.xlsx?download=1"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  DELPHI — Test Pipeline"
echo "══════════════════════════════════════════════════════════════════"

# ── Step 0: Check core package imports (fail fast before any download) ────────
echo ""
echo "── Step 0: Check package imports ───────────────────────────────"
python - << 'PYEOF'
import importlib, sys
core = [
    ("torch", "PyTorch"), ("numpy", "NumPy"), ("pandas", "Pandas"),
    ("sklearn", "scikit-learn"), ("xgboost", "XGBoost"),
    ("captum", "Captum"), ("shap", "SHAP"),
    ("matplotlib", "Matplotlib"), ("yaml", "PyYAML"),
    ("openpyxl", "openpyxl"),
]
missing = []
for mod, name in core:
    try:
        importlib.import_module(mod)
        print(f"  OK      {name}")
    except Exception as e:
        print(f"  MISSING {name}  ({e})")
        missing.append(name)
if missing:
    print("\n  ERROR: missing core packages: " + ", ".join(missing))
    print("  Install them first:  ./install.sh   (or pip install -r requirements.txt)")
    sys.exit(1)
print("  All core packages import cleanly.")
PYEOF

# ── Step 1: Ensure DS1.xlsx, then regenerate the balanced test subset ─────────
echo ""
echo "── Step 1: Prepare $TEST_DATA ──────────────────"

# A valid .xlsx is a ZIP archive: its first two bytes are "PK". A failed
# download (HTML error page, truncated file) will not be, and pandas then
# raises BadZipFile. Treat any existing-but-invalid file as missing so we
# re-download it.
_is_valid_xlsx() {
    [ -f "$1" ] || return 1
    # size at least ~10 KB
    _b=$(wc -c < "$1" 2>/dev/null || echo 0)
    [ "$_b" -ge 10000 ] || return 1
    # first two bytes must be "PK"
    _magic=$(head -c 2 "$1" 2>/dev/null)
    [ "$_magic" = "PK" ] || return 1
    return 0
}

if [ -f "$FULL_DATA" ] && ! _is_valid_xlsx "$FULL_DATA"; then
    echo "  $FULL_DATA exists but is not a valid .xlsx (corrupt/partial download)."
    echo "  Removing it and re-downloading..."
    rm -f "$FULL_DATA"
fi

if [ ! -f "$FULL_DATA" ]; then
    echo "  $FULL_DATA not found — downloading DS1 from Zenodo..."
    mkdir -p data
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$FULL_DATA" "$ZENODO_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "$FULL_DATA" "$ZENODO_URL"
    else
        echo "  ERROR: neither wget nor curl is available to download DS1."
        echo "  Download it manually: $ZENODO_URL"
        exit 1
    fi
    echo "  Downloaded → $FULL_DATA"
    # Validate the download: Zenodo returns an HTML error page (not a ZIP) if
    # the URL is wrong, which pandas would later reject as BadZipFile.
    if ! _is_valid_xlsx "$FULL_DATA"; then
        echo "  ERROR: the downloaded file is not a valid .xlsx."
        echo "  The Zenodo URL is probably wrong (record ID or filename)."
        echo "  Check ZENODO_URL in this script:"
        echo "    $ZENODO_URL"
        rm -f "$FULL_DATA"
        exit 1
    fi
fi

echo "  Generating balanced 500-antibody subset..."
python utils/create_subsets.py \
    --input "$FULL_DATA" \
    --target psr_filter \
    --sizes 500 \
    --threshold 0.8 \
    --cluster_col CDR3 \
    --no-representative
echo "  Ready → $TEST_DATA"

# ── Step 2: Run the integration test suite ───────────────────────────────────
echo ""
echo "── Step 2: Run the integration test suite ──────────────────────"
# Do not abort collection if the suite returns non-zero — we still want the
# plots and logs. Capture the exit code and re-apply it at the end.
set +e
python tests/test_delphi.py "$@"
SUITE_RC=$?
set -e

# ── Step 3: Collect all plots and logs into tests/results/ ───────────────────
# Copies every figure (.png/.tiff/.pdf/.svg) and log (.log/.txt) produced by the
# suite into one folder the user can browse. Trained model checkpoints
# (.pt/.pkl) are intentionally excluded — those are large testing artifacts, not
# results to inspect.
echo ""
echo "── Step 3: Collect plots and logs into tests/results/ ──────────"
RESULTS="$ROOT/tests/results"
mkdir -p "$RESULTS"

# k-fold plots (CV_ROC_*, CV_*), prediction plots (ROC/histograms), and
# interpretability figures can land in a few places depending on delphi.py's
# working directory: under tests/, in the repo root, or in a results/ or plots/
# folder. Search all of them so nothing is missed.
SEARCH_DIRS="
$ROOT/tests
$ROOT/results
$ROOT/plots
$ROOT/build
$ROOT
"

# Use a temp list so the count is correct (avoids the while-in-subshell pitfall).
_list="$(mktemp)"
for src in $SEARCH_DIRS; do
    [ -d "$src" ] || continue
    # Depth-limit the repo-root scan to avoid trawling the whole tree; the
    # dedicated dirs (tests/, results/, plots/) are searched fully.
    if [ "$src" = "$ROOT" ]; then
        _maxdepth="-maxdepth 2"
    else
        _maxdepth=""
    fi
    # shellcheck disable=SC2086
    find "$src" $_maxdepth \
        -path "$RESULTS" -prune -o \
        -type f \( -name "*.png" -o -name "*.tiff" -o -name "*.tif" \
                   -o -name "*.pdf" -o -name "*.svg" \
                   -o -name "*.log" -o -name "*.txt" \) -print 2>/dev/null \
        >> "$_list"
done

# Copy unique files (dedupe by basename would risk collisions, so copy by full
# path; identical paths found via multiple roots are naturally de-duplicated).
sort -u "$_list" | while IFS= read -r f; do
    [ -f "$f" ] || continue
    cp -f "$f" "$RESULTS/" 2>/dev/null || true
done

_collected=$(find "$RESULTS" -type f 2>/dev/null | wc -l | tr -d ' ')
rm -f "$_list"

echo "  Collected $_collected file(s) into tests/results/"
echo "  (figures + logs; trained model checkpoints excluded)"
if [ "$_collected" -eq 0 ]; then
    echo "  NOTE: nothing collected. If --kfold/--predict wrote plots elsewhere,"
    echo "  tell me the path and I will add it to the search."
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Done. See tests/results/ for all k-fold plots, prediction outputs,"
echo "  and logs."
echo "══════════════════════════════════════════════════════════════════"

exit $SUITE_RC
