# Integrated Gradients reproduction

This document records the Integrated Gradients (IG) configuration used for the
reported DELPHI PSR and SEC analyses. Commands are written relative to the root
of a DELPHI repository clone. They do not rely on workstation-specific paths.

## Reported configuration

- model: one-hot Transformer (`transformer_onehot`, language model `onehot`)
- attribution target: class 1, `P(Pass)`
- reference: padding-safe, length-matched uniform amino-acid reference
  (`1/20` in every observed residue channel and zero only at true padding)
- integration steps: 200
- antibodies: all rows with a non-missing target (`--max-samples 0` and
  `--ig-max-samples 0`)
- reported classification threshold: 0.5
- checkpoint architecture: four layers, hidden dimension 128, eight attention
  heads, feed-forward dimension 512, dropout 0.38, maximum HCDR3 length 25

## Internal recomputation from proprietary sequences

These commands require the private IPI sequence workbooks. Those workbooks are
not distributed in `paper/data/shareable/` and must not be uploaded publicly.

```bash
python delphi_interpretability.py \
  --db paper/data/local_only/raw/ipi_psr_trainset.xlsx \
  --target psr_filter \
  --model transformer_onehot \
  --lm onehot \
  --model-path paper/data/local_only/models/FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt \
  --outdir paper/test_output/interpretability/psr \
  --max-samples 0 \
  --ig-max-samples 0 \
  --ig-baseline uniform \
  --ig-steps 200 \
  --n-antibodies 1
```

```bash
python delphi_interpretability.py \
  --db paper/data/local_only/raw/ipi_sec_5000.xlsx \
  --target sec_filter \
  --model transformer_onehot \
  --lm onehot \
  --model-path paper/data/local_only/models/FINAL_sec_filter_onehot_transformer_onehot_ipi_sec_5000.pt \
  --outdir paper/test_output/interpretability/sec \
  --max-samples 0 \
  --ig-max-samples 0 \
  --ig-baseline uniform \
  --ig-steps 200 \
  --n-antibodies 1
```

The private inputs should be supplied only in an approved internal checkout.
The command-line script, not the `--train` or `--predict` workflow, generates
the IG result files and diagnostic plots.

## Public, sequence-free reproduction

Public users cannot recompute IG from the underlying IPI sequences. They can
reproduce Main Figure 6 and Extended Data Figures 6-7 from the released,
sequence-free summaries:

```bash
python paper/run_reproduction_tests.py \
  --public \
  --output-dir ../delphi_test/public
```

The source tables are in `paper/data/shareable/interpretability/`. They retain
only the attribution and aggregation fields required for the published plots;
they do not contain VH, VL, HCDR3/CDR3, or other recoverable antibody sequence
fields. The IG summaries use all 11,265 PSR and 5,045 SEC antibodies. The 500
rows in each amino-acid-by-position table are aggregate cells comprising 25
positions by 20 amino acids, not a 500-antibody subset. Extended Data Figure 8 includes CDR3-loop ΔP(Pass) mutagenesis and
requires private sequence-level inputs, so it is excluded from the public
reproduction suite.

## Expected checks

For a full internal run, confirm that:

1. the logs report `baseline=uniform` and `n_steps=200`;
2. the PSR and SEC runs use all eligible rows;
3. completeness residuals remain numerically small;
4. figure-generation scripts consume the matching exported result files; and
5. no file under `paper/data/local_only/` is staged for public release.
