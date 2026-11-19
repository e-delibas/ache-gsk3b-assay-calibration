"""Reconstruct assay-family outcomes from the archived record-level audit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    for name, digest in json.loads((DATA / "provenance.json").read_text())["snapshot_hashes"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Snapshot checksum mismatch for {name}")

    records = pd.read_csv(DATA / "record_audit.csv.gz")
    retained = records.loc[records.passed_record_filters & records.eligible_family.notna()].copy()
    retained = retained.rename(columns={"eligible_family": "family"})
    np.testing.assert_allclose(retained.pic50, 9 - np.log10(retained.standard_value), atol=1e-10)
    one_doc = retained.groupby(["target", "family"]).doc.transform("nunique").eq(1)
    keys = ["target", "doc", "assay", "family", "smiles"]
    medians = retained.loc[one_doc].groupby(keys, as_index=False).pic50.median()
    rebuilt = medians.sort_values(keys).drop_duplicates(keys[:-1])
    snapshot = pd.read_csv(DATA / "measurements.csv")
    pd.testing.assert_frame_equal(
        snapshot[keys + ["pic50"]].sort_values(keys).reset_index(drop=True),
        rebuilt.reset_index(drop=True), atol=1e-12, rtol=0,
    )
    with np.load(DATA / "features.npz") as features:
        smiles = pd.Index(features["smiles"])
        if not smiles.is_unique or set(snapshot.smiles) != set(smiles):
            raise ValueError("Feature keys and measurement structures differ.")
        if features["fp"].shape != (len(smiles), 2048) or features["desc"].shape != (len(smiles), 20):
            raise ValueError("Unexpected feature dimensions.")
    print(f"Verified {len(records)} source rows, {len(snapshot)} assay-family observations and {len(smiles)} structures.")


if __name__ == "__main__":
    main()
