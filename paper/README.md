# Paper figures and analysis

Code and results behind the figures and quantitative claims in the DELPHI manuscript.
No proprietary IPI training sequences are included; the scripts read those tables from a
local `data/` directory (gitignored) by path.

## Layout

- `figures/code/` — figure-generation scripts on a shared Okabe-Ito palette
  (`okabe_nature.py`): main figures `fig1.py`–`fig6.py`, Extended Data `ed*.py`.
- `figures/output/` — rendered figures (PDF + PNG + TIFF). The per-antibody Extended Data
  figure output is omitted from the repository because it renders individual CDR3
  sequences; regenerate it locally with `ed7.py` if needed.
- `analysis/` — revision analyses and the numbers behind them:
  - `R1_germline_out.py` — leave-one-VH-germline-out evaluation (0.965 → 0.903).
  - `R2_R3_R5_recompute.py` — cross-assay dual-liability test and external-cohort
    ROC-AUC / PR-AUC with bootstrap CIs.
  - `R7_R8_R4_tables.py` — SEC mean-AUC, germline composition, and baseline comparison.
  - `R6_sec_labels_investigation.md` — SEC label provenance note.
  - `VERIFIED_NUMBERS_LEDGER.md` — every reported number, with its source and how it was
    re-derived.

## Reproduce

The figure scripts read from a local `data/` directory holding the proprietary IPI tables
and the public DS1 / clinical-cohort files. To reproduce the platform end-to-end on
shareable data only, run `examples/ds1_worked_example.sh`.
