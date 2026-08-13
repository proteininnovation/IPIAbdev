# DELPHI manuscript figure manifest

Prepared locally on 2026-08-13 from public commit `5720de632213aae9002b68d255d7a8896ce88d7d`, with the final revised Figure 4 and Extended Data Figures 3-5 substituted from the manuscript-update package.

| Manuscript item | Generator | Principal input(s) | Final render | Upload status |
|---|---|---|---|---|
| Main Figure 1 | `figures/code/fig1.py` | IPI PSR/SEC tables and assay exports | `figures/output/Figure1.*` | Script/render public; raw IPI data local only |
| Main Figure 2 | `figures/code/fig2.py` | IPI PSR table and five PLM embedding CSVs | `figures/output/Figure2.*` | Script/render public; sequences/embeddings local only |
| Main Figure 3 | `figures/code/fig3.py` | validation metrics, 10-fold CV table, replicated learning curves | `figures/output/Figure3.*` | Learning-curve CSVs can be public; IPI row-level validation local only |
| Main Figure 4 | `figures/code/fig4.py` | cross-dataset table and reduced Jain/GDPa figure-source workbooks | `figures/output/Figure4.*` | Final revised script/render; reduced public workbooks exclude sequence columns |
| Main Figure 5 | `figures/code/fig5.py` | IPI PSR, IPI SEC and public DS1 | `figures/output/Figure5.*` | Script/render public; IPI data local only |
| Main Figure 6 | `figures/code/fig6.py` | sequence-free IG summaries, regional attribution and XGBoost-SHAP CSVs | `figures/output/Figure6.*` | Script, render and sequence-free source tables public |
| Extended Data Figures 1-2 | `figures/code/ed1_ed2.py` | IPI PSR-ELISA and IPI SEC | `figures/output/ED_Fig1.*`, `ED_Fig2.*` | Script/render public; input data local only |
| Extended Data Figure 3 | `figures/code/ed3.py` | public Supplementary Table 4 prediction-score workbook | `figures/output/ED_Fig3.*` | Script, render and source workbook public |
| Extended Data Figure 4 | `figures/code/ed8.py` | reduced Jain/GDPa figure-source workbooks | `figures/output/ED_Fig4.*` | Final revised compact panel; reduced workbooks reproduce it pixel-for-pixel |
| Extended Data Figure 5 | `figures/code/ed4.py` | public Supplementary Table 4 prediction-score workbook | `figures/output/ED_Fig5.*` | Script, render and source workbook public |
| Extended Data Figures 6-7 | `figures/code/ed5_ed6.py` | sequence-free PSR/SEC SHAP and IG beeswarm CSVs | `figures/output/ED_Fig6.*`, `ED_Fig7.*` | Script, render and sequence-free source tables public |
| Extended Data Figure 8 | `figures/code/ed7.py` | IPI PSR/SEC tables and two one-hot checkpoints | not committed publicly because it prints individual sequences | Keep generator public; keep render/data/checkpoints local unless cleared |
| Extended Data Figure 9 | `figures/code/ed9_germline_crossassay.py` | `R1_results.json`, IPI SEC | `figures/output/ED_Fig9.*` | Summary JSON/render public; IPI SEC local only |
| Supplementary Figure 1 | Editable source in `figures/source_editable/SupplFigure1_elisa_filtering.pptx` | label-QC assay data | in Supplementary Information PDF | No dedicated plotting script located |
| Supplementary Figures 2-4 | Editable sources in `figures/source_editable/` | architecture definitions/model code | in Supplementary Information PDF | Schematic sources located; no dedicated plotting scripts |
| Supplementary Figure 5 | `figures/code/supp_fig5.py` using canonical `utils/threshold_optimizer.py` | pooled 10-fold predictions and threshold reports | `figures/supplementary_output/` | Paper wrapper/render public; OOF rows local only; no duplicated utility code |

## Important naming map

The historical filenames do not match final manuscript numbering:

- `ed8.py` generates final **Extended Data Figure 4**.
- `ed4.py` generates final **Extended Data Figure 5**.
- `ed5_ed6.py` generates final **Extended Data Figures 6 and 7**.
- `ed7.py` generates final **Extended Data Figure 8**.

Renaming these files before GitHub upload would make the provenance clearer, but this staging package preserves existing names to avoid breaking references.
