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

The underlying attribution calculations cannot be recomputed without the
private IPI sequences. The files here reproduce the published visual summaries;
they do not expose or permit reconstruction of individual IPI sequences.

The IG summaries were generated from all 11,265 PSR antibodies and all 5,045
SEC antibodies. Each `fig6_*_ig_by_aa_position.csv` file has 500 aggregate rows
because it reports 25 positions by 20 amino acids. The 500 rows do not represent
a 500-antibody subset.

`manifest.json` records the exported schemas and sequence-leakage audit result.
