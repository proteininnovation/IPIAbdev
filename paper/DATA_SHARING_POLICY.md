# Data-sharing policy for manuscript figure materials

## Never publish

- `ipi_psr_trainset_elisa.xlsx`, in whole or in part.
- Any IPI table containing VH, VL, HSEQ, LSEQ, CDR3/HCDR3, nucleotide sequence, or other recoverable antibody sequence fields.
- IPI embedding files when they are linked to antibody identifiers or could enable recovery of row-level proprietary information.
- Model checkpoints or row-level interpretation files unless separately approved for release.

These materials must remain under `paper/data/local_only/` or in an approved internal archive. They must not be uploaded to GitHub, Zenodo, a public data repository, or a public supplementary-data package.

## May be considered for release

- Prediction scores and labels after removal of all VH/VL and other sequence columns.
- Aggregate performance metrics, learning-curve summaries, correlation coefficients, and other non-sequence summary statistics.
- Reduced figure-source tables containing only the columns required to reproduce a published plot, provided they have passed internal disclosure review.
- Sequence-free row-level interpretability measurements and aggregate IG/SHAP summaries, provided literal sequence fields and sequence-bearing identifiers have been removed.
- Final figure renders and scripts that contain no proprietary rows or embedded data.

Removing sequence columns is necessary but does not itself constitute release approval. Reduced row-level workbooks should still be reviewed for identifiers, confidential assay measurements, and linkage risk before publication.

The approved Figure 6/Extended Data 6-7 exports are under
`data/shareable/interpretability/`. They have been scanned for literal IPI
sequence strings. Their private source files remain unchanged under
`data/local_only/interpretability/`.

## Script policy

A plotting or analysis script may be public even when its input data are private, provided the script contains no embedded proprietary data, sequences, credentials, or private filesystem paths. The documentation must clearly state that the required IPI input is not publicly distributed. Publishing such a script is optional; it does not make the private input reproducible or releasable.
