# GitHub upload checklist

1. Copy `paper/figures/code/`, `paper/figures/output/`, `paper/figures/supplementary_output/`, `paper/figures/source_editable/`, `paper/analysis/`, `paper/data/shareable/`, and the Markdown files into a clean branch created from public `main`.
2. Do **not** upload `paper/data/local_only/`. It contains proprietary IPI sequences, a 68 MB prediction workbook, 639 MB of embeddings, model checkpoints, and row-level interpretation data.
3. Never upload `ipi_psr_trainset_elisa.xlsx`, including to GitHub, Zenodo, or a public supplementary-data package.
4. The `data/shareable/` Jain/GDPa workbooks are reduced figure-source copies with sequence columns removed. Their full originals are retained only under `data/local_only/raw/`.
5. Confirm that every public row-level table excludes `HSEQ`, `LSEQ`, `VH`, `VL`, `CDR3/HCDR3`, nucleotide-sequence, and equivalent sequence fields. Prediction scores or sequence-free reduced tables may be released only after internal disclosure review.
5a. Retain `data/shareable/interpretability/manifest.json` with the Figure 6 source tables. Re-run `analysis/export_sequence_free_interpretability.py` internally if the private attribution inputs change.
6. Confirm whether editable PowerPoint sources are wanted in Git. If not, keep only PDF/PNG outputs and archive the PPTX files with the submission package.
7. Do not upload TIFF files if repository size is a concern. PDF + PNG is sufficient for GitHub; retain TIFF in the journal package or an approved internal archive.
8. Resolve the Supplementary Figures 1-4 provenance gap: editable sources exist, but no dedicated programmatic generation scripts were found.
9. Run `python paper/validate_package.py` and `python paper/run_reproduction_tests.py --repo-root /path/to/delphi` after copying. Review `TEST_REPORT.md` and regenerate from a clean environment before release.
10. Update the root manuscript-code availability statement only after the public branch contains the selected scripts and approved source data.
