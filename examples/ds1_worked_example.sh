#!/usr/bin/env bash
# Worked example: train and interpret DELPHI end-to-end on the PUBLIC DS1 polyreactivity
# dataset, with no access to the proprietary IPI training sequences. Uses only the public
# DS1 subset downloaded from Zenodo in step 1 (nothing is committed) and the DELPHI CLI.
#
# Run from the repository root:
#   bash examples/ds1_worked_example.sh
#
# Prerequisite (one command):  bash install.sh
set -euo pipefail

DB=tests/DS1_5000.xlsx     # public DS1 subset, downloaded from Zenodo in step 1 (gitignored)
TARGET=psr_filter

# 1. download the public DS1 dataset from Zenodo (writes tests/DS1.xlsx + tests/DS1_5000.xlsx; gitignored)
python utils/download_ds1_dataset.py

# 2. cross-validate: honest AUC + epoch selection. transformer_onehot needs no PLM embeddings.
python delphi.py --kfold 10 --target "$TARGET" --lm onehot --model transformer_onehot --db "$DB"

# 3. train the final model (checkpoint carries its own decision threshold)
python delphi.py --train --target "$TARGET" --lm onehot --model transformer_onehot --db "$DB"

# 4. interpret: per-residue Integrated Gradients + CDR3 attribution
python delphi_interpretability.py --predict "$DB" --db "$DB" \
    --target "$TARGET" --model transformer_onehot --lm onehot \
    --ig-max-samples 500 --n-antibodies 20 \
    --outdir outputs/interp_ds1

# ---------------------------------------------------------------------------------------
# To use a protein language model instead (e.g. IgBERT), pre-compute embeddings first:
#   python delphi.py --build-embedding "$DB" --lm igbert
#   python delphi.py --kfold 10 --target "$TARGET" --lm igbert --model transformer_lm --db "$DB"
#   python delphi.py --train  --target "$TARGET" --lm igbert --model transformer_lm --db "$DB"
#
# To retrain on YOUR OWN antibodies, replace $DB with your labelled .xlsx
# (columns: BARCODE, HSEQ, LSEQ, CDR3, plus your binary label) and set --target to that label.
# See the main README sections "Training Your Own Models" and "Interpretability Analysis"
# for the full set of flags.
