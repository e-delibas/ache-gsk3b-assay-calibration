# Extended analysis and manuscript exhibits

This directory contains code, fixed inputs, and calculated tables for the study's robustness, source provenance, uncertainty and comparator analyses. They are included as part of the complete research analysis. Analyses not fixed in the initial experimental plan are exploratory/post hoc; this distinction is methodological, not editorial.

## Directory structure

- `code/`: nine analytical scripts and one shared chemistry feature module
- `inputs/`: row-level export metadata and reserved-family source information for the source-unit audit
- `outputs/`: CSV files used in the manuscript and Supporting Information

## Running the analysis

Use Python 3.12. Execute in the repository root after installing `requirements.txt`. Run preflight first:

```bash
python -m pip install -r requirements.txt
python src/verify_data.py
python src/verify_release.py
export OPENBLAS_NUM_THREADS=1  # macOS/Linux; optional
python analysis/code/unit_audit.py . analysis/outputs
python analysis/code/influence.py . --output analysis/outputs
python analysis/code/diagnostics.py . --output analysis/outputs
python analysis/code/kernel_check.py . --output analysis/outputs
python analysis/code/cat_sensitivity.py . --output analysis/outputs
python analysis/code/reservation.py . analysis/outputs
python analysis/code/scaffolds.py . analysis/outputs
python analysis/code/exhibits.py . analysis/outputs
```

On Windows PowerShell, omit the `export` line or use `$env:OPENBLAS_NUM_THREADS=1`.

**Caution:** Model-fitting scripts may take time and overwrite supplied CSVs. The archived analysis is included for inspection. Preflight verifies source hashes and selected table relationships but does not prove independent rerunning of every training pipeline.

## Scientific interpretation

- Source-verified inclusion and removal of a potentially influential source are separate sensitivity panels.
- The original training data were not globally rebuilt for source-verified inclusion; the existing CatBoost specification was applied conditionally.
- Within-document effects and provenance stratifications are exploratory and must not be presented as external validation.
- Some results were computed using Python 3.12, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, CatBoost 1.2.10, matplotlib 3.10.8, RDKit 2026.09.1. The original source standardization used RDKit 2026.09.2. Cross-version bitwise equivalence is not guaranteed.
