# DELPHI paper patch test report — 27 August 2026

Tested against a clean detached worktree of public `origin/main` commit
`21153a2` (`Update sequence-free IG figure source data`).

## Structural and privacy validation

- `python paper/validate_package.py`: **passed**.
- Python syntax compilation passed for `fig3.py`, `ed3.py`, `ed7.py`, and
  `run_reproduction_tests.py`.
- No private IPI sequence columns were found in the new Extended Data Figure 3
  source workbook.
- No private datasets, embeddings, checkpoints, machine-specific paths, or
  `local_only/` files are included in the upload patch.

## Public reproduction test

```bash
python paper/run_reproduction_tests.py \
  --public \
  --output-dir /tmp/DELPHI_GITHUB_PUBLIC_TEST_20260827
```

Result: **7/7 commands passed**.

| Item | Result |
|---|---|
| Main Figure 3 | PASS; pixel-identical to final render |
| Main Figure 4 | PASS; pixel-identical to repository baseline |
| Main Figure 6 | PASS; pixel-identical to repository baseline |
| Extended Data Figure 3 | PASS; pixel-identical to final render |
| Extended Data Figure 4 | PASS; pixel-identical to repository baseline |
| Extended Data Figure 5 | PASS; pixel-identical to repository baseline |
| Extended Data Figures 6–7 | PASS; pixel-identical to repository baseline |

## Private-input validation

The corrected `ed7.py` was tested separately using temporary links to the
private PSR/SEC inputs and one-hot checkpoints. Result: **1/1 passed**. The
regenerated Extended Data Figure 8 PNG was pixel-identical to the final
manuscript render. Private inputs and the sequence-bearing final render are not
included in this patch.
