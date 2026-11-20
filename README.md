# Three-reference offset and shrinkage calibration for AChE and GSK-3β

Computational materials accompanying the manuscript *Evaluating Three-Reference Offset and Shrinkage Calibration in AChE and GSK-3β Assays*.

This repository contains the dataset snapshot, model development code, held-out assessment results, source-provenance and robustness analyses, manuscript figures and tables, and Supporting Information. The files form one integrated code and data package for the manuscript. The document-based test split was used during exploratory feasibility work; results must not be represented as a newly untouched external validation. Additional sensitivity and diagnostic analyses were conducted retrospectively; their findings are exploratory unless otherwise stated.

## Repository contents

| Folder | Contents |
|---|---|
| `data/` | ChEMBL-derived standardized measurements, structure fingerprints/descriptors, record audit, and provenance |
| `src/` | Primary modeling, held-out evaluation, source sensitivity, reporting and integrity checks |
| `notebooks/` | Development and held-out assessment notebooks |
| `results/` | Stored primary development and held-out predictions, metrics, bootstrap and document summaries |
| `analysis/code/` | Source-unit checking, additional baselines, uncertainty, influence, assay-size, scaffold, kernel and CatBoost sensitivity analyses |
| `analysis/inputs/` | Fixed source correspondences and reserved-structure metadata for the assay-unit verification |
| `analysis/outputs/` | Full-precision derived tables for the manuscript and Supporting Information |
| `figures/` | Figures 1–4 used in the manuscript (Figure 3 is the four-panel analysis) |
| `reports/` | Tabular summaries from the model development workflow, not the final manuscript tables |
| `supplementary/` | Tables S1–S10 and full-precision Supporting Data S1 |

## Core results: strict assay-family panel

| Target | Held-out source documents | Raw CatBoost MAE | Three-reference mean MAE | Calibrated MAE | Gain vs. reference mean (95% document bootstrap CI) |
|---|---:|---:|---:|---:|---|
| AChE (CHEMBL220) | 31 | 0.943 | 0.536 | 0.507 | 0.029 (−0.004, 0.075) |
| GSK-3β (CHEMBL262) | 10 | 0.710 | 0.611 | 0.608 | 0.003 (−0.003, 0.009) |

Raw model and calibration comparisons must be interpreted in the context of the source verification panels. The AChE mean gain over the reference mean is concentrated in two documents and is about 0.004 when both are omitted. The source-based split does not guarantee structural scaffold independence. The calibrated predictor and the reference-mean baseline both use the **same three measured reference activities**.

## Setup and verification

Use Python 3.12 with dependencies from `requirements.txt` (exact bitwise reproducibility across environments is not guaranteed).

```bash
python -m pip install -r requirements.txt
python src/verify_data.py
python src/verify_release.py
```

These commands verify integrity and key numerical consistency; they do **not** retrain every model or independently reproduce every metric. See `analysis/README.md` for analysis-specific scripts and their execution sequence.

## Model development and held-out assessment

```bash
python src/pilot.py
python src/holdout.py
python src/source_sensitivity.py
python src/report.py
```

These commands may retrain models and overwrite archived result files. Run them in a separate copy when preserving the supplied outputs matters. The precomputed results are already included in `results/`; notebooks are under `notebooks/`.

## Article tables and figures

- Main Table 2: `analysis/outputs/table2_by_panel.csv` and `analysis/outputs/unit_panel_contrasts.csv`
- Main Table 3: `analysis/outputs/table3_both_contrasts.csv`
- Main Table 4: `analysis/outputs/primary_summary.csv`
- Figure 3: `figures/Figure_3_Error_Decomposition.png` (600 dpi), with numbers in `analysis/outputs/figure3_data.csv`
- Supporting Tables S1–S10: `supplementary/SuppInfo_S1_Three_Reference_Calibration.pdf`
- Supporting Data S1: `supplementary/SuppInfo_Data_S1_Full_Tables.zip`

## Nested development paired comparison (Table S10)

The original document-grouped nested development predictions support paired reference-mean versus calibrated-model comparisons. To reproduce the mean gains, confidence intervals, win/loss counts and sign tests without refitting the models:

```bash
python analysis/code/development_paired.py --repo . --output analysis/outputs
```

Results are in `analysis/outputs/nested_development_paired_summary.csv` and `analysis/outputs/nested_development_paired_by_document.csv`. Inference uses source documents as units and 50,000 fixed-seed paired bootstrap replicates. The tests do not establish a fresh external validation.

## Reproducibility and limitations

The analysis evaluates a retrospective ChEMBL-derived dataset. Some additional analyses were performed following source/unit audit and are **post hoc** (not independently held-out validation). The reserved-family release does not create a new untouched external test. The source-audited inclusion was assessed with the existing training specification; the original training population was not globally rebuilt. The archived original four-outer/three-inner nested development evaluation is reported; no additional independent nested cross-validation, FS-CAP, ActFound, mixed-effects model refit, or exhaustive hyperparameter search is claimed. No practically important equivalence margin was prespecified.

The record source and ChEMBL license are described in `data/README.md` (ChEMBL 37, CC BY-SA 3.0 for the derived source data). No separate software license has been declared for this release; public hosting should not be interpreted as a general unrestricted software license.

Repository URL and archive DOI, if assigned, should be supplied in the manuscript *Data and Code Availability* statement before journal submission.
