# Verified numbers ledger - revision2 redteam fixes

Every number below was produced by a script in this folder and independently
re-run (main thread) before use. No value is from memory. Scripts:
`R1_germline_out.py`, `R2_R3_R5_recompute.py`, `R7_R8_R4_tables.py`
(outputs: the matching `*_OUTPUT.txt` / re-verify runs). Deployed model =
`transformer_lm_ablang_ipi_psr_trainset_score` (Transformer + AbLang2).

## R1 - leave-one-VH-germline-out (NEW computation; XGBoost + AbLang2 pooled embeddings)
Same classifier (XGBClassifier n_est=400, depth=5, lr=0.05, subsample/colsample 0.8) for every protocol. n=11,265 (0 unmatched), pass 5,925 (52.6%) / fail 5,340.
- AUC_cluster (HCDR3-cluster 10-fold StratifiedGroupKFold, pooled OOF) = **0.965** (0.9648). Anchors the manuscript's 0.959 (XGBoost+AbLang2, ED Table 1 = 0.959).
- AUC_germline_out (LeaveOneGroupOut by VH germline, pooled) = **0.903** (0.9025).
- **Absolute drop = 0.062.** Holding out a whole frozen-framework germline lowers AUC -> the within-CV number is an optimistic upper bound on a fixed-framework background.
- VH+VL-pair-out (finer background) recovers to 0.948 (drop 0.017), consistent with framework redundancy as the inflating factor.
- Per-germline AUC (n>=100): VH3-23 0.946, VH1-69 0.800, VH3-7 0.947, VH4-34 0.944, VH5-51 0.799, VH1-46 0.893, VH3-15 0.874. Biggest collapses are the germlines whose pass rate is far from the 52.6% pool mean (VH1-69 70%; VH5-51 94%, only 31 fails). Pooled AUC is the robust statistic (some germlines have few minority-class members).

## R2 - dual-liability cross-assay (ipi_sec_5000.xlsx, both psr_filter and sec_filter per molecule; n=5,045)
Polarity verified both ways (filter==1 = Pass). Uses ground-truth labels + computed HCDR3 charge (NOT model scores), so train/test leakage does not bias it.
- 2x2: PSR-fail&SEC-fail 946 | PSR-fail&SEC-pass 380 | PSR-pass&SEC-fail 889 | PSR-pass&SEC-pass 2,830.
- phi = **0.43**, odds ratio = **7.9**, Fisher exact p = 5.4e-205.
- P(SEC fail | PSR fail) = **0.71** vs P(SEC fail | PSR pass) = **0.24** (~3x lift).
- HCDR3 net charge predicts both: AUC(PSR-fail) = **0.78** (rho +0.43); AUC(SEC-fail) = **0.82** (rho +0.54).
- BARCODE overlap psr<->sec = 3,354 (of 5,045 SEC). Caveat: ~2/3 of the SEC set overlaps the PSR trainset by barcode, so the PSR axis is partly in-sample; the co-occurrence and the charge->label associations use labels+features (not predictions) and are not biased by that.
- Verdict: dual-liability co-failure and shared charge driver are **supported at the population level**; engineering (mutate charge -> fix both) is still **model-predicted, not wet-lab validated**.

## R3 / R5 - external cohorts, deployed AbLang2 (fig5.py metric definitions; bootstrap 2000, seed 0)
Pass = psr_filter (Jain) or PR-CHO < 0.27 (GDPa). PR-AUC is for the rare FAIL class (positive=Fail, score=1-P(Pass)); no-skill = Fail prevalence.

| Cohort | n | Pass | Fail | ROC-AUC [95% CI] | PR-AUC(Fail) [95% CI] | no-skill | signed rho [95% CI] |
|---|---|---|---|---|---|---|---|
| Jain 2017 PSR-SMP | 137 | 109 | 28 | 0.73 [0.61-0.85] | 0.50 [0.32-0.69] | 0.20 | -0.31 [-0.47,-0.14] |
| GDPa1 PR-CHO | 197 | 138 | 59 | 0.68 [0.60-0.76] | 0.51 [0.39-0.64] | 0.30 | -0.28 [-0.41,-0.15] |
| GDPa3 PR-CHO | 80 | 68 | 12 | 0.75 [0.59-0.88] | 0.34 [0.17-0.64] | 0.15 | -0.35 [-0.55,-0.14] |
| GDPa3 PR-Ova | 80 | 77 | 3 | 0.78 [0.48-1.00] | 0.24 [0.03-1.00] | 0.04 | -0.66 [-0.79,-0.50] |

- Deployed AbLang2 ROC-AUC = **0.73 / 0.68 / 0.75** (Jain / GDPa1 / GDPa3 PR-CHO). PR-AUC beats no-skill in every cohort except GDPa3 PR-Ova (only 3 Fails -> CI [0.03-1.00], uninformative; flag as underpowered).

## T2 - external-grid maximum (ed8.py grid, 6 LMs x 6 readouts)
- Grid max AUC = **0.805** = **IgBert x GDPa3 PR-Ova** (the 3-Fail readout). NOT the deployed AbLang2 model.
- Deployed AbLang2 grid max = **0.779** (also GDPa3 PR-Ova). So "up to 0.805" is the single best off-diagonal LM/readout cell, not the production model.

## R4 - PSR 25-combo (ED_Table1_PSR.xlsx) + held-out 20% bootstrap (validation CSV)
- 20 PLM combos: mean AUC **0.959** (sd 0.005, range 0.946-0.967). Best = **XGBoost+IgBert 0.967**. Best-minus-mean 0.008 (1.4 sd) -> "among the best, within CV noise."
- Baseline claim is FALSE: RF+k-mer 0.951 exceeds 2 PLM combos (Transformer+AbLang2 0.946, RF+AntiBERTa2-CSSP 0.948); Transformer+one-hot 0.961 exceeds 9 PLM combos.
- Held-out 20% AUC + 95% CI (n=2,209): Transformer+AbLang2 0.967 [0.961-0.974], CNN+AbLang2 0.970 [0.963-0.976], XGBoost+AbLang2 0.969 [0.963-0.975], RF+AbLang2 0.964 [0.956-0.970], Transformer+one-hot 0.969 [0.962-0.975] - all overlap.
- Paired holdout AUC diff (Transformer+AbLang2 - Transformer+one-hot) = -0.002 [95% CI -0.007, +0.004], frac>0 = 0.28 -> indistinguishable within distribution.

## R6 - SEC label provenance (manuscript Methods [119]-[121] + pipeline)
- SEC PASS/FAIL is a **physical SEC-trace call** (monomer peak area >= 90%, RT window; multi-peak = Fail), not ML-assigned.
- The RF+XGBoost k-mer "consensus >= 0.6" step **downsampled the PASS majority class** (cluster-representative + consensus-confirmed); all 1,835 FAIL retained in full. It selected rows, did not assign labels.
- The only genuinely circular comparison is RF+Kmer / RF+Biophysical (same feature family as the selector); the manuscript already flags this and excludes them from the headline. Fix is a precise provenance clarification + modest softening of "independent" -> "consistent", NOT a circularity confession.

## R7 - single SEC summary statistic (ED_Table3_SEC.xlsx)
- SEC mean AUC over the **20 PLM combinations = 0.933** (0.9335; excludes one-hot / biophysical / k-mer). Parallel to PSR PLM-mean 0.959. **Use 0.933 in Abstract [5], Results [38], Discussion [62], ED Table 3; drop 0.934 and 0.938.**
- Per-LM SEC AUC max (for [66]/[230]): AbLang2 0.935 (0.9349, XGB), AntiBERTy 0.956 (0.9561, XGB), AntiBERTa2 0.952 (0.9524), AntiBERTa2-CSSP 0.947 (0.9468), IgBert 0.951 (0.9509). Unpaired AntiBERTy (0.956) tops the paired PLMs - supports T4.

## R8 - VH/VL germline reconciliation (ipi_psr_trainset.xlsx; fig1.py convention, n>=100)
- Overall PSR pass rate 52.6%.
- **7 VH germlines with n>=100** (Fig 1d), not 5: VH5-51 94.0% (n=520), VH3-15 91.1% (247), VH1-69 70.3% (2,442), VH3-7 70.0% (1,298), VH1-46 53.2% (378), VH3-23 38.3% (5,451), VH4-34 30.3% (877). Plus VH4-39 (n=52, below 100).
- The 5 "designed" germlines (VH1-69, VH3-23, VH3-7, VH4-34, VH5-51) are all present; VH3-15 and VH1-46 are the extra two. The v2 [110] numbers (94% VH5-51 ... 30% VH4-34, and 91% VH3-15) are CORRECT against data; the inconsistency is only that [105] lists 5.
- VL: also more than the 4 stated kappa - data has VK3-20 (3,707), VK1-39 (2,549), VK3-15 (2,369), VK4-1 (1,496), plus **VL1-51 lambda (1,073)** and VL2-14 (71). So a lambda VL is present and substantial.
- Action: reconcile [105] (designed 5 VH / 4 VL) with the realized set (7 VH, >=5 VL incl. lambda) shown in Fig 1d; flag design-vs-data to authors.
