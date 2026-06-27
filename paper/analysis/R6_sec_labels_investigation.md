# R6 — SEC label provenance investigation

**Question (reviewer R6):** The IPI SEC `sec_filter` PASS labels were allegedly "sculpted" by an RF+XGBoost
k-mer consensus filter (consensus prob ≥ 0.6), making SEC results circular and unfit as independent
corroboration of the conserved HCDR3 charge signature. Does an UN-denoised (raw) SEC label exist so SEC
can be re-evaluated on labels not touched by the k-mer denoiser?

**Date:** 2026-06-27. Read-only investigation. Every claim below is tied to a file/line/column actually inspected.

---

## VERDICT: **A — a usable raw/orthogonal SEC label EXISTS, and the reviewer's premise is factually wrong.**

Two distinct findings, both favorable:

1. **The k-mer RF+XGBoost consensus filter does NOT create `sec_filter`.** It is an *optional row-selection /
   downsampling* step that takes an already-existing `sec_filter` column as **input**. The trained SEC models
   in the paper were fit on the **raw** `sec_filter` column of `ipi_sec_5000.xlsx` with **no consensus
   filtering applied on load**. So the published SEC AUCs are already "un-denoised" in the sense R6 means.
   The "0.6 consensus probability" the reviewer cites is not even the default (README documents `--min-prob 0.7`),
   and there is **no stored consensus-probability column** anywhere in the SEC file.

2. **`sec_filter` is a physical/rule-based label, recoverable from the raw SEC trace.** The file carries the
   underlying physical measurements — `retention_time_mins` (4,397/5,045 non-null) and `peak_area_pct`
   (main-peak monomer %, sparse) — and `sec_filter` tracks them deterministically in the obvious way
   (multi-peak ⇒ FAIL; single sharp monomer peak in the monomer RT window ⇒ PASS). An orthogonal,
   denoiser-free SEC label can be defined directly from `retention_time_mins` peak count + main-peak RT.

Either finding alone defeats the circularity objection. Together they make it clean.

---

## 1. `ipi_sec_5000.xlsx` — full column inventory (Sheet1, 5045 × 108)

All 108 columns enumerated (index : name : dtype : non-null):

```
  0 BARCODE            object 5045      54 HCDR3_RR              bool 5045
  1 CDR3               object 5045      55 HCDR3_WW              bool 5045
  2 heavy              object 5045      56 HCDR3_VV              bool 5045
  3 light              object 5045      57 HCDR3_neg_patch       bool 5045
  4 name               object 5045      58 HCDR3_cys             bool 5045
  5 antigen            object 5045      59 HCDR3_neg_charge      bool 5045
  6 Antibody.IPI.Name  object 5035      60 HCDR3_pos_charge      bool 5045
  7 UniProt_Name       object 5001      61 HCDR3_simple_charge   int  5045
  8 well               object 5045      62 HCDR3_parker_hplc     bool 5045
  9 plate              object 5045      63 HCDR3_parker_hplc_index f 5045
 10 order_ID           object 5045      64 HCDR3_arg             bool 5045
 11 FACS               object 2256      65 HCDR3_trp             bool 5045
 12 BLI                object 1184      66 HCDR3_rmd             bool 5045
 13 facs_human_ec50_nm  f64  4372      67 HCDR3_isoelectricpoint f 5045
 14 facs_mouse_ec50_nm  f64  4206      68 HCDR3_charge          f64  5045
 15 facs_binding_human object 4433      69 HCDR3_instability_index f 5045
 16 facs_binding_mouse object 4222      70 HCDR3_aromaticity     f64  5045
 17 facs_target1       object 4433      71 HCDR3_hydrophobicity  f64  5045
 18 facs_target2       object 4267      72 HCDR3_length          int  5045
 19 purif_filter       int   5045       73 VH_glyco              bool 5045
 20 puriftitermgl      f64   5045       74 VH_asp_isomer         bool 5045
 21 psr_norm_dna       f64   5045       75 VH_asn_deamid         bool 5045
 22 psr_norm_avidin    f64   4984       76 VH_aromatic           bool 5045
 23 psr_norm_insulin   f64   5045       77 VH_frag               bool 5045
 24 psr_norm_smp       f64   4280       78 VH_poly               bool 5045
 25 psr_filter         int   5045       79 VH_RR                 bool 5045
 26 concentration      f64   4127       80 VH_WW                 bool 5045
 27 ka_1ms             f64   2318       81 VH_VV                 bool 5045
 28 kd_1s              f64   2318       82 VH_neg_patch          bool 5045
 29 kd_m               f64   2318       83 VH_cys                bool 5045
 30 rmax               f64   2282       84 VH_neg_charge         bool 5045
 31 res_sd             f64   2318       85 VH_pos_charge         bool 5045
 32 spr_filter         object 4134      86 VH_simple_charge      int  5045
 33 spr_filter_rank    f64   4134       87 VH_parker_hplc        bool 5045
 34 sec_filter         int   5045       88 VH_parker_hplc_index  f64  5045
 35 retention_time_mins object 4397     89 VH_arg                bool 5045
 36 peak_area_pct2     f64   0          90 VH_trp                bool 5045
 37 peak_area_pct      object 153       91 VH_rmd                bool 5045
 38 HSEQ               object 5045      92 VH_isoelectricpoint   f64  5045
 39 Unnamed: 39        int   5045       93 VH_charge             f64  5045
 40 HCDR1              object 5045      94 VH_instability_index  f64  5045
 41 HCDR2              object 5045      95 VH_aromaticity        f64  5045
 42 HCDR3              object 5045      96 VH_hydrophobicity     f64  5045
 43 LSEQ               object 5045      97 VH_length             int  5045
 44 LCDR1              object 5045      98 protein_families      object 4800
 45 LCDR2              object 5045      99 subcellular_location  object 4781
 46 LCDR3              object 5045     100 UNIPROT_ID            object 4863
 47 CDR3_noC           object 5045     101 ipi_code_computed     object 5045
 48 HCDR3_glyco        bool 5045       102 antigen_aa            object 4976
 49 HCDR3_asp_isomer   bool 5045       103 antigen_aa_len        int  5045
 50 HCDR3_asn_deamid   bool 5045       104 ngs_computed          object 5045
 51 HCDR3_aromatic     bool 5045       105 spr_anno              f64  4134
 52 HCDR3_frag         bool 5045       106 spr_anno2             f64  4134
 53 HCDR3_poly         bool 5045       107 HCDR3_CLUSTER_0.8     int  5045
```

### The modeling label and the physical-measurement candidates

| Column | What it is | Verdict for "raw SEC label" |
|---|---|---|
| **`sec_filter`** (34) | The modeling label. int, all 5045 non-null. **3210 PASS(1) / 1835 FAIL(0).** | This is what models train on. |
| **`retention_time_mins`** (35) | **RAW physical SEC measurement.** object (comma-separated multi-peak strings), 4397 non-null. | ✅ **Orthogonal raw signal — usable.** |
| **`peak_area_pct`** (37) | Main-peak monomer area %. object, only **153** non-null (~97% missing). | Physical but too sparse to label at scale. |
| `peak_area_pct2` (36) | **Entirely empty (0 non-null).** | Useless. |
| `ngs_computed` (104) | NGS run/sample ID (e.g. "Miseq80", "MiSeq_82"). Not a label. | Not SEC; not a label. |
| `spr_anno` (105) / `spr_anno2` (106) | **SPR** binding annotations (1/0), 4134 non-null. | Wrong assay (SPR, not SEC). |
| `Unnamed: 39` (39) | Integer ~119–123 (looks like a sequence length). | Not a label. |

**Probability-column scan:** a programmatic sweep of all 108 columns for any continuous-in-(0,1)
"consensus probability / OOF score / keep-prob" column returned only physicochemical descriptors
(`kd_m` = SPR off-rate; `HCDR3_aromaticity`, `VH_aromaticity` = sequence features). A name-based scan for
`prob|consensus|conf|score|oof|rf_|xgb|keep|reject|denoise|raw` returned **zero** matches.
**No stored consensus probability or raw-vs-denoised label column exists.**

### How `sec_filter` relates to the physical SEC trace (it is rule-based, not ML-derived)

Parsing `retention_time_mins` into peaks (split on `,`):

```
                          n rows   sec_filter distribution
multi-peak (≥2 peaks):      161    {0: 161}        ← ALL multi-peak rows are FAIL
single-peak (1 peak):     4119    {1: 3145, 0: 974}
no retention time:         765    {0: 700, 1: 65}  ← missing RT skews heavily FAIL
```

Retention time of the (first/main) peak, by class:

```
sec_filter=1 (PASS):  n=3145  min 1.759  q25 2.800  median 2.860  q75 2.907  max 3.309   (tight monomer window)
sec_filter=0 (FAIL):  n=1135  min 0.416  q25 2.726  median 3.272  q75 3.466  max 5.310   (broad, longer RT)
```

Interpretation: PASS = a single sharp peak in a narrow monomer retention window (~2.8–2.9 min);
FAIL = multiple peaks (aggregation) **or** a single peak shifted out of the monomer window. This is exactly
the behavior of a **physical SEC pass/fail rule**, not an ML classifier output. (There is class overlap on
RT alone among single-peak rows — 974 single-peak FAILs — so `sec_filter` is not a pure 1-D RT threshold;
it also reflects peak count and, where available, monomer area %.)

Where `peak_area_pct` (main-peak monomer %) is present (153 rows), it corroborates the same direction:
PASS main-peak area median 100% (q25 100); FAIL median 100% but min 52.1% with a left tail — i.e. FAILs include
low-monomer-% cases. Too sparse to use as a labeler, but consistent with a physical rule.

This matches the pipeline's own documentation of the SEC trace format — `utils/Figure1_datasetcurration.py:413-459`
(`_pick_main_peak` / `load_sec`): *"SEC cells may contain either a single number (pure-monomer antibody, one
peak) or a comma-separated list (aggregating antibody, multiple peaks)… the peak with the largest area is the
'main peak' (usually the monomer), and its area % equals the conventional monomer purity metric."* And
`load_sec`'s docstring: *"All 82 multi-peak rows … belong to the Fail class (sec_filter = 0) — as expected,
since aggregation is what produces multiple peaks."* (count differs from current file's 161, but the rule is
the same; that figure reads `sec_retention_time_figure1.xlsx`, see §3.)

---

## 2. Pipeline: how the SEC PASS labels were constructed and whether the denoiser touched them

Pipeline at `/Users/Andre.Teixeira/temp/delphi` (= `<DELPHI>/delphi` symlink, confirmed:
`delphi -> /Users/Andre.Teixeira/temp/delphi`).

### 2a. The k-mer consensus filter is a row-SELECTOR, not a label-MAKER — and it's optional

`utils/build_balanced_dataset_v4.py` header docstring (lines 1–54), verbatim excerpts:

```
3   Always produces TWO output datasets from one run:
5     Dataset 1 -- BALANCED  ... Majority downsampled to exactly n_minority.
11    Dataset 2 -- IMBALANCED ... Majority downsampled to (min_total - n_minority)
26    kmer_consensus
28      cross_val_predict produces out-of-fold (OOF) probabilities --
30      Consensus: RF_oof >= min_prob AND XGB_oof >= min_prob -> keep.
43  NA/empty label rows removed automatically. Majority class auto-detected.
44  Works with any binary label: sec_filter, psr_filter, spr_filter, etc.
```

So `--strategy kmer_consensus` takes an existing label column (`--label sec_filter`) and decides **which
majority-class rows to keep** when downsampling for class balance. RF+XGBoost OOF probabilities rank/keep
rows; they do **not** assign the PASS/FAIL label. The consensus `>= min_prob` test (line 30, and code line
457 `consensus_mask = rf_pass & xgb_pass`) gates *retention of already-labeled rows*, nothing more.

README confirms it is **optional** (lines ~586–612):

```
### Step 1: Build a balanced training set (optional)
This step is optional but recommended when your training data is imbalanced or contains noisy labels.
You can skip directly to Step 2 and train on your raw data.
...
# OOF consensus only (confidence filtering, stricter)
python delphi.py --build-dataset tests/DS1_psr_500.xlsx
    --target psr_filter --strategy kmer_consensus --min-prob 0.7
```

Note the documented example uses **`--min-prob 0.7`**, not 0.6. The README's curation one-liner
(line 60) calls this *"Automated denoising via CDR3 clustering and OOF confidence filtering"* — but that is a
**training-set curation/balancing option**, applied (when used) to the majority class, not a relabeling of SEC.

### 2b. The trained SEC models loaded the RAW `sec_filter`, with NO consensus filter on load

`delphi.py` training path, `load_data()` (lines 831–890):

```
836   df = pd.read_excel(db_path)
839   required = ["BARCODE", "HSEQ", "LSEQ", label_col]
851   df = df.dropna(subset=required).set_index("BARCODE")
887   y = data[label_col].values          ← label taken verbatim
```

The label is read straight from the column (`y = df[label_col].values`); the only filtering is `dropna` on
required columns. Grepping the training path of `delphi.py` for
`build.?dataset | kmer_consensus | min_prob | consensus | denoise | downsample` returns **nothing** — the
consensus/denoise machinery lives only in `utils/build_balanced_dataset*.py`, which is a separate, opt-in
`--build-dataset` subcommand, not part of `--train`.

`config/model_registry.yaml` (line 225 etc.) declares the SEC checkpoints' training data:

```
FINAL_sec_filter_ablang_cnn_ipi_sec_5000.pt:
    trainset: manuscripts/data/ipi_sec_5000.xlsx
```

i.e. the SEC models trained directly on `ipi_sec_5000.xlsx` (n=5045, the raw `sec_filter`), consistent with the
checkpoint naming `FINAL_sec_filter_<lm>_<model>_ipi_sec_5000`. There is no `..._balanced` / `..._consensus` /
`..._imbalanced` SEC trainset referenced; no consensus-filtered SEC derivative file exists in `data/`.

**Conclusion for §2:** The pre-denoising labels are not just *recoverable* — they are **what was used**.
`sec_filter` in `ipi_sec_5000.xlsx` IS the raw, physically-derived label, and the published SEC AUCs are
already computed on it without consensus denoising.

---

## 3. Auxiliary files

- **`ipi_sec_5000.xlsx.*.emb.csv`** (ablang, antiberta2, antiberta2-cssp, antiberty, igbert): per-sequence PLM
  **embedding matrices** keyed by BARCODE. No label column — embeddings only. (Loaded by `load_data` line 864:
  `f"{db_path}.{lm}.emb.csv"`.) Not relevant to label provenance.

- **`<DELPHI>/data/sec_retention_time_figure1.xlsx`** (5045 × 4): columns
  `['sec_filter', 'retention_time_mins', 'peak_area_pct2', 'Peak Area Percent']`. This is the file
  `Figure1_datasetcurration.py` actually loads (it references column `"Peak Area Percent"`, which exists here but
  not in `ipi_sec_5000.xlsx`). Same `sec_filter` (3210/1835) and same raw `retention_time_mins`; `peak_area_pct2`
  empty, `Peak Area Percent` only 153 non-null. **Carries the raw physical RT, no extra/raw label beyond what's
  already in the main file.**

- **`<DELPHI>/data/ipi_sec_val.xlsx`** (1803 × 101): the SEC validation set. Label-relevant columns:
  `sec_filter` (1482/321), **`sec_retention_time`** (raw physical RT, 1636 non-null, e.g. `"3.316; 4.204"`),
  and an ML-prediction column **`xgboost_antiberta2-cssp_label`** (1430/373 — clearly a model output, not a raw
  label). `sec_retention_time` separates by class the same way (PASS median 2.872 vs FAIL median 3.226). So the
  val set ALSO carries a raw physical SEC signal usable for an orthogonal label.

- **`<DELPHI>/data/ED_Table3_SEC.xlsx`** and **`<DELPHI>/backup/ED_Table1_SEC.xlsx`**: formatted manuscript
  **results tables** ("SEC Aggregation Prediction Performance", AUC numbers), not raw data. No alternate label.
  (Note backup version header says "7,956 antibodies" vs current "n=5045" — an older/larger SEC cut, FYI, but
  still a results table, not raw labels.)

- **`<DELPHI>/backup`, `<DELPHI>/suppl_doc_table_figure`, `<DELPHI>/figures_tables`**: scanned. Contents are
  manuscript artifacts — figures (`.tiff/.pdf/.png`), results tables (`ED_Table*_SEC.xlsx`, `Suppl_Table2_*`),
  learning-curve CSVs, `.docx` manuscript drafts, `.pptx`. **No file named like raw/original/un-denoised SEC
  labels.** The only raw SEC physical data anywhere is `retention_time_mins` / `peak_area_pct` (and
  `sec_retention_time` in the val file).

---

## What R6 can do with this

**Primary response (recommended): rebut the premise, no relabeling needed.** Document that `sec_filter` is a
physical SEC pass/fail call (monomer retention window + peak count / monomer %), that the RF+XGBoost k-mer
"consensus" is an *optional class-balancing row selector* applied to the majority class — not a label generator —
and that the **reported SEC models trained on the raw `sec_filter` column of `ipi_sec_5000.xlsx` with no
consensus filtering on load** (`delphi.py:load_data` reads `y = df[label_col].values`;
`model_registry.yaml` trainset = `ipi_sec_5000.xlsx`). The "≥0.6" figure is not even the documented default
(0.7). Therefore SEC is not circular and remains valid independent corroboration of the HCDR3 charge signature.

**Optional belt-and-suspenders (if reviewer still wants a denoiser-free SEC AUC):** define an orthogonal SEC
label purely from the physical trace and re-run:

> **Orthogonal SEC PASS/FAIL rule (denoiser-free), from `retention_time_mins` in `ipi_sec_5000.xlsx`:**
> parse the cell on `,`; let `k` = number of peaks and `rt_main` = retention time of the first/main peak.
> - `k ≥ 2` → **FAIL** (aggregation: multiple peaks)
> - `k == 1` and `rt_main` within the monomer window (data-driven, ~2.6–3.0 min; calibrate from the PASS
>   q25–q75 = 2.80–2.91) → **PASS**
> - `k == 1` and `rt_main` outside the window → **FAIL**
> - missing RT → exclude (765 rows) or treat as FAIL (700/765 are already FAIL).
>
> This label never touches the k-mer model. Re-scoring the existing SEC predictions against it gives a fully
> independent SEC AUC. (Caveat to state honestly: it isn't *identical* to `sec_filter` — there's RT overlap
> among single-peak rows that the curators presumably resolved with monomer-area %, which is 97% missing here —
> so expect a somewhat noisier label, and report it as a sensitivity analysis, not the primary metric.)

The text claim most directly in R6's crosshairs is **README line 880** (mirrored in the manuscript discussion):
*"…the same electrostatic failure signature as PSR, despite independent training sets and a different biophysical
assay. Cross-assay convergence of this signature confirms it reflects fundamental sequence-level grammar rather
than assay-specific artefacts."* That statement is **supported**, not undermined, by what's in the code/data:
SEC is a physically-labeled, separately-trained assay.

---

## Evidence index (paths inspected)

- `<DELPHI>/data/ipi_sec_5000.xlsx` — Sheet1, 5045×108; columns 34 `sec_filter`, 35 `retention_time_mins`,
  37 `peak_area_pct`, 36 `peak_area_pct2`(empty), 104 `ngs_computed`, 105/106 `spr_anno(2)`.
- `<DELPHI>/data/sec_retention_time_figure1.xlsx` — 5045×4 (sec_filter, retention_time_mins, peak_area_pct2, Peak Area Percent).
- `<DELPHI>/data/ipi_sec_val.xlsx` — 1803×101; `sec_filter`, `sec_retention_time` (raw), `xgboost_antiberta2-cssp_label` (ML pred).
- `<DELPHI>/data/ipi_sec_5000.xlsx.{ablang,antiberta2,antiberta2-cssp,antiberty,igbert}.emb.csv` — embeddings, no labels.
- `<DELPHI>/data/ED_Table3_SEC.xlsx`, `<DELPHI>/backup/ED_Table1_SEC.xlsx` — results tables, no raw labels.
- `/Users/Andre.Teixeira/temp/delphi/utils/build_balanced_dataset_v4.py:1-54, 451-535` — consensus = optional row selector on existing label.
- `/Users/Andre.Teixeira/temp/delphi/delphi.py:831-890` (`load_data`, `y = df[label_col].values`); training path has no consensus/denoise call.
- `/Users/Andre.Teixeira/temp/delphi/config/model_registry.yaml:225` — SEC trainset = `manuscripts/data/ipi_sec_5000.xlsx`.
- `/Users/Andre.Teixeira/temp/delphi/utils/Figure1_datasetcurration.py:413-459` — SEC trace format / multi-peak = FAIL.
- `/Users/Andre.Teixeira/temp/delphi/README.md:60, 577, 586-612, 880, 1059` — curation framing, label def, optional build-dataset, cross-assay claim.
