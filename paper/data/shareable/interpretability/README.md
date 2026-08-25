# Sequence-free interpretability data

These tables support public inspection and reproduction of DELPHI Figure 6 and
Extended Data Figures 6-7 without distributing literal IPI antibody sequences.

- `fig6_*_ig_by_position.csv` contains mean absolute integrated-gradient values
  and observation counts by region and position.
- `fig6_*_ig_by_aa_position.csv` contains mean signed integrated-gradient values
  and observation counts by amino-acid identity and CDR3 position. These are
  aggregate cells and contain no antibody sequence strings.
- `region_attribution_*.csv` contains aggregate attribution mass by region.
- `shap_xgb_*_sequence_free.csv` retains row-level numerical SHAP and feature
  values but excludes barcodes.
- `interp_*_beeswarm_*.csv` retains the numerical beeswarm source records but
  excludes barcodes from Transformer-derived tables.

The Transformer IG source tables use target class 1, 200 integration steps,
and a length-matched uniform amino-acid reference (1/20 at every observed
residue position; zero only at true padding). They contain all labeled
antibodies: 11,265 for PSR and 5,045 for SEC. The private row-level files also
record Captum convergence deltas.

The underlying attribution calculations cannot be recomputed without the
private IPI sequences. The files here reproduce the published visual summaries;
they do not expose or permit reconstruction of individual IPI sequences.

`manifest.json` records the exported schemas and sequence-leakage audit result.
