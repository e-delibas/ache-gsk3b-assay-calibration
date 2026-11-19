"""Apply the post hoc source restrictions to fixed document-level predictions."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarize(documents):
    wide = documents.pivot(index="doc", columns="model", values="mae")
    gain = (wide["raw"] - wide["calibrated"]).to_numpy()
    indices = np.random.default_rng(42).integers(len(gain), size=(2000, len(gain)))
    lower, upper = np.quantile(gain[indices].mean(axis=1), [0.025, 0.975])
    means = documents.groupby("model")[["mse", "level_mse"]].mean()
    reduction = means.loc["raw"] - means.loc["calibrated"]
    return dict(n=len(gain), raw=wide["raw"].mean(), ref=wide["mean"].mean(),
                cal=wide["calibrated"].mean(), gain=gain.mean(), gain_lo=lower, gain_hi=upper,
                level_share=reduction.level_mse / reduction.mse)


def main():
    audit = pd.read_csv(ROOT / "data/sources/heldout_sources.csv")
    docs = pd.read_csv(ROOT / "results/holdout/by_document.csv").query("scope == 'primary' and k == 3")
    if audit.duplicated(["target", "doc"]).any():
        raise ValueError("Duplicate source annotation.")
    if set(zip(audit.target, audit.doc)) != set(zip(docs.target, docs.doc)):
        raise ValueError("Source annotations and scored documents differ.")
    journal_known = ~audit.unknown & audit.doc_type.ne("PATENT")
    panels = {
        "original": audit,
        "flagged_excluded": audit.loc[~audit.domain_issue],
        "unknown_and_patent_excluded": audit.loc[journal_known],
        "explicit_human_journal": audit.loc[audit.explicit_human & ~audit.domain_issue & audit.doc_type.ne("PATENT")],
        "unknown_patent_and_high_potency_excluded": audit.loc[journal_known & audit.assay.ne("CHEMBL4192783")],
    }
    rows = []
    for name, subset in panels.items():
        included = docs.merge(subset[["target", "doc"]], on=["target", "doc"], validate="many_to_one")
        for target, group in included.groupby("target"):
            rows.append(dict(panel=name, target=target, **summarize(group)))
    out = ROOT / "results/source_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out.name)


if __name__ == "__main__":
    main()
