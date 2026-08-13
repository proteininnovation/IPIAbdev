# DELPHI paper figures and analysis

This folder contains the scripts, final renders, and figure-source data assembled for the DELPHI manuscript. The scripts are based on public commit `5720de632213aae9002b68d255d7a8896ce88d7d`; final revised Main Figure 4 and Extended Data Figures 3-5 replace their older public versions.

Start with [`FIGURE_MANIFEST.md`](FIGURE_MANIFEST.md), which maps every manuscript figure to its script, input data, render, and public/private status.

## Layout

- `figures/code/`: main and Extended Data figure scripts, shared style, and the Supplementary Figure 5 wrapper. Threshold calculations import the repository's canonical `utils/threshold_optimizer.py`.
- `figures/output/`: final PDF/PNG/TIFF renders.
- `figures/supplementary_output/`: final threshold panels used for Supplementary Figure 5.
- `figures/source_editable/`: editable PowerPoint sources recovered for Supplementary Figures 1-4.
- `data/shareable/`: small summary/source tables that are candidates for GitHub upload.
- `data/local_only/`: proprietary, large, sequence-level, checkpoint, or attribution inputs. This directory is intentionally ignored by Git.
- `analysis/`: claim-verification scripts and result ledgers from the paper branch.

## Run locally

Use the DELPHI repository environment, then run a script from `figures/code/`. Most scripts use portable paths from `paths.py`. The final revised scripts with explicit arguments provide `--help`, for example:

```bash
python paper/figures/code/fig4.py --help
python paper/figures/code/ed3.py --help
python paper/figures/code/ed8.py --help
python paper/figures/code/ed4.py --help
```

Defaults can be overridden with:

```bash
export DELPHI_PAPER_DATA=/path/to/paper/data
export DELPHI_FIGURE_OUTPUT=/path/to/output
export DELPHI_REPO_ROOT=/path/to/delphi
```

The local package includes all dependencies found on this workstation. Before pushing to a public repository, use `UPLOAD_CHECKLIST.md` and do not add `data/local_only/`.

Run the complete local reproduction suite (figures plus claim-analysis scripts) with:

```bash
python paper/run_reproduction_tests.py --repo-root /path/to/delphi
```

Public reviewers can run the sequence-safe subset with no `local_only/` data:

```bash
python paper/run_reproduction_tests.py --public --output-dir /tmp/delphi-public-test
```

For clarity, the equivalent explicit full-suite command is:

```bash
python paper/run_reproduction_tests.py --internal --repo-root /path/to/delphi
```

The test writes only to `paper/test_output/`. Use `--skip-expensive` to omit t-SNE, the private sequence-bearing Extended Data Figure 8 render, and the repeated R1 classifier fits.

`ipi_psr_trainset_elisa.xlsx` is strictly non-public and must never be uploaded. See `DATA_SHARING_POLICY.md` for the approved distinction between private sequence-bearing inputs and potentially shareable, sequence-free prediction/source tables.

`data/shareable/interpretability/` contains sequence-free source tables for
Main Figure 6 and Extended Data Figures 6-7. These public summaries reproduce
the final figures exactly but do not allow recomputation of IG/SHAP values from
the underlying private antibody sequences.

`data/shareable/manuscript_tables/` contains the final Supplementary Tables 2,
4, 5, and 7 and Extended Data Tables 1-3. Supplementary Table 4 includes DS1
prediction scores and enables public regeneration of Extended Data Figures 3
and 5. The IPI validation sheet contains no antibody sequences.
