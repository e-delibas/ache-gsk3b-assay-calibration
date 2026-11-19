# Three-reference assay calibration for AChE and GSK-3β

This repository accompanies a retrospective evaluation of how three measured reference compounds affect absolute potency prediction and within-assay ranking. Each target has its own CatBoost regressor. Each reported test prediction belongs to an assay from a held-out source document. The included outputs can be inspected without installing the modeling stack.

## Results and data

| Target | ChEMBL ID | Assay–family observations | Scored held-out documents | Raw MAE | Reference mean MAE | Calibrated MAE |
|---|---|---:|---:|---:|---:|---:|
| AChE | CHEMBL220 | 4,262 | 31 | 0.943 | 0.536 | 0.507 |
| GSK-3β | CHEMBL262 | 2,012 | 10 | 0.710 | 0.611 | 0.608 |

MAE is averaged within reference draw, assay and document as described in the manuscript. For both targets, the document-bootstrap interval for the gain over the reference mean includes zero. The positive calibration weights preserve the CatBoost ordering within every draw.

## Reproduce the analysis

Use Python 3.12 and a CPU. From the repository root on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/verify_data.py
python src/pilot.py
python src/holdout.py
python src/source_sensitivity.py
python src/report.py
```

The first script checks snapshot hashes, reproduces the assay-level aggregation from audited measurements and validates feature alignment. The next two scripts refit the development and held-out analyses. The held-out script estimates its own calibration weights from development data. The final scripts apply fixed source restrictions, write Tables 1–3 and redraw Figures 2–4. To rebuild just the tables and plots, start with the bundled results and run the last two commands. Rerunning scripts replaces the corresponding outputs in your working copy. The two training stages also have notebook entry points in `notebooks/` for readers with Jupyter installed. Use the project environment as the kernel.

## Where to find each result

| Item | File or directory |
|---|---|
| Modeling snapshot, feature cache and source records | `data/` (see `data/README.md`) |
| Exploratory development protocol, including the original advancement rule | `development_protocol_tr.md` (original in Turkish) |
| Development comparisons, weights and document folds | `results/metrics.csv`, `results/contrasts.csv`, `results/weights.csv`, `results/folds.csv` |
| Held-out document and assay assignments | `results/holdout/holdout_manifest.csv` |
| Reference-draw metrics and document-level metrics | `results/holdout/episodes.csv.gz`, `results/holdout/by_document.csv` |
| Primary metrics and paired intervals | `results/holdout/summary.csv`, `results/holdout/paired_ci.csv` |
| Source-level annotations and sensitivity estimates | `data/sources/heldout_sources.csv`, `results/source_sensitivity.csv` |
| Original manuscript artwork, including the workflow diagram | `figures/` |
| Machine-readable tables and redrawn numerical figures | `reports/` |

The development pilot compared conditional weights with one weight per target. The conditional rule did not meet its advancement criterion. The primary held-out analysis uses the single development-selected weight (0.75 for AChE and 0.40 for GSK-3β). In the common-query comparison, all reference budgets reserve the first five positions of each permutation, and the same queries are scored. This explains the difference between the two three-reference rows in the summary. All weights and source predictions remain fixed during the post hoc source sensitivity analysis.

## Attribution

The activity and assay metadata derive from [ChEMBL](https://www.ebi.ac.uk/chembl/), release 37, EMBL-EBI. See `data/README.md` for the source and data license.
