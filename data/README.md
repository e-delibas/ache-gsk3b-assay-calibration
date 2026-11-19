# Data dictionary and provenance

| File | Contents |
|---|---|
| `measurements.csv` | Model input. One observation per target, document, assay and stereochemistry family. `doc` and `assay` are ChEMBL identifiers. `family` groups related stereoisomers. `smiles` is the representative standardized structure and `pic50` is the response. |
| `features.npz` | Three arrays with common row order. `smiles` is the lookup key, `fp` holds 2,048 Morgan fingerprint bits (radius 2), and `desc` holds 20 RDKit descriptors. |
| `descriptor_names.json` | Descriptor names in the column order used for training. |
| `record_audit.csv.gz` | Selected fields from all 10,079 AChE and 5,116 GSK-3β source export rows. Identify a row by `target` and `export_row`. Includes ChEMBL molecule, document and assay IDs, standardized concentration fields, standardized structure, record filter decision and derived potency when available. |
| `sources/heldout_sources.csv` | Document and assay IDs for the 41 scored held-out sources, with publication identifiers, assay wording and audit decisions. The source URL links each document to its original article or patent. |
| `provenance.json` | SHA-256 digests of the two model input files and identifiers for the four upstream inputs used to generate the snapshot. |

## From records to model inputs

The release starts from the archived quality and structure-standardization decisions. In the record audit, `passed_record_filters` is the archived row-level decision. The corresponding field `record_filter_reason` records the first exclusion, if applicable. A nonempty `eligible_family` indicates that the standardized structure survived the upstream target-specific eligibility checks, including exclusion of previously reserved chemical families or scaffolds and conflicts. Record-level and structure-level eligibility are applied together.

For retained records, pIC₅₀ equals 9 − log₁₀(IC₅₀ in nM). Within each target, families assigned to multiple documents are removed. Repeated measurements of a standardized structure within an assay are summarized by their median. Where multiple structures belong to the same assay and family, the lexicographically first standardized SMILES supplies the representative observation. The verification script reconstructs these observations and checks them against the model input. The archived features were computed with RDKit 2026.09.1. Rerunning the models uses the stored feature arrays.

## Source audit

The `review` flag denotes review articles. The `explicit_human` flag refers to wording in the ChEMBL assay description. The `unknown` flag denotes the phrase “unknown origin”. The `domain_issue` field flags the two review-derived records and two preparations identified in the AChE source check. `provenance_status` gives the reason and records selected checks of original methods. These annotations apply to scored held-out documents. The panel definitions are in `src/source_sensitivity.py`.

## Data license and changes

Source: ChEMBL release 37, EMBL-EBI, targets CHEMBL220 and CHEMBL262. ChEMBL data and the ChEMBL-derived data distributed here are provided under the [Creative Commons Attribution-ShareAlike 3.0 Unported license](https://creativecommons.org/licenses/by-sa/3.0/). Changes for this analysis include filtering, structure standardization, potency conversion, aggregation, feature calculation, split assignment and source annotation. [ChEMBL](https://www.ebi.ac.uk/chembl/) · [ChEMBL releases](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/)
