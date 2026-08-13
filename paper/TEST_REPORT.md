# Local reproduction test report

Tested on 2026-08-13 with the local DELPHI repository and the private inputs
under `paper/data/local_only/`.

## Outcome

- All six main-figure generators passed.
- All nine Extended Data figure generators passed. Extended Data Figure 8 was
  generated only in the ignored local test directory because it displays
  antibody-level identifiers and sequence-derived information.
- Supplementary Figure 5 threshold analysis and fold-stability generation
  passed. Its regenerated JSON is exactly equal to the stored threshold report;
  the current Matplotlib PNG bounding box differs by only 3-6 pixels, so the
  existing manuscript render remains authoritative.
- Analysis scripts R1, R2/R3/R5, and R4/R7/R8 passed.
- All 14 public main/Extended Data PNGs regenerated pixel-for-pixel identically
  to the staged manuscript renders.
- Main Figure 6 and Extended Data Figures 6-7 also regenerated pixel-for-pixel
  identically using only the sequence-free tables in
  `data/shareable/interpretability/`.
- Package structure, Python syntax, machine-path, and public-data privacy checks
  passed (`16` figure/support scripts and `14` required public PNGs).

The full run logs and machine-readable report are intentionally local and
ignored by Git under `paper/full_test_output/`.

## Fixes made during testing

1. Removed the duplicated `figures/code/threshold_optimizer.py`. The new
   `supp_fig5.py` wrapper imports the canonical repository implementation at
   `utils/threshold_optimizer.py`.
2. Corrected the PSR interpretability input stem in `ed5_ed6.py` from
   `ipi_psr` to `ipi_psr_trainset`.
3. Restored the submitted Figure 6 canvas height to 215 mm, yielding an exact
   pixel match.
4. Made Extended Data Figure 8 tensor construction robust to a local
   PyTorch/NumPy ABI mismatch without changing model inputs or values.

## Remaining provenance limitations

- Supplementary Figures 1-4 have editable PowerPoint sources, but no dedicated
  programmatic generation scripts were located. They appear to be manually
  assembled schematics/assay panels. Their editable sources should be retained,
  and their provenance should be confirmed with the original author before a
  claim of fully scripted reproduction.
- Private IPI sequence tables, embeddings, checkpoints, and row-level
  interpretation data are required for several generators. They must remain
  local and are deliberately absent from the public upload archive.
- The repository file `models/transformer_onehot.py` is locally modified in the
  working tree. Extended Data Figure 8 passed with that local implementation;
  reproduce it from a clean public branch before release to confirm that the
  public commit contains the same compatible loader.
