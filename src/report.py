"""Export manuscript tables and redraw the three numerical figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HOLDOUT = RESULTS / "holdout"
OUT = ROOT / "reports"
TARGETS = {"AChE": "AChE", "GSK3B": "GSK-3β"}
COLORS = {"AChE": "#176876", "GSK3B": "#ad572b"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tables(primary, sensitivity):
    data = pd.read_csv(ROOT / "data/measurements.csv")
    table = data.groupby("target").agg(observations=("pic50", "size"), structures=("smiles", "nunique"),
                                       documents=("doc", "nunique"), assays=("assay", "nunique"))
    for label, file in [("development", RESULTS / "episodes.csv.gz"), ("holdout", HOLDOUT / "episodes.csv.gz")]:
        scored = pd.read_csv(file, usecols=["target", "doc", "assay"]).drop_duplicates().groupby("target")
        table[label + "_scored_documents"] = scored.doc.nunique()
        table[label + "_scored_assays"] = scored.assay.nunique()
    table.to_csv(OUT / "table_1_dataset.csv")
    primary[["target", "model", "mae", "rho", "rank_defined"]].to_csv(OUT / "table_2_primary.csv", index=False)
    sensitivity.loc[sensitivity.panel.isin(["flagged_excluded", "explicit_human_journal"])].to_csv(
        OUT / "table_3_provenance.csv", index=False)


def differences():
    ci = pd.read_csv(HOLDOUT / "paired_ci.csv")
    docs = pd.read_csv(HOLDOUT / "by_document.csv").query("scope == 'primary' and k == 3")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    for i, (target, title) in enumerate(TARGETS.items()):
        for j, comparator in enumerate(["raw", "mean"]):
            row = ci.loc[(ci.target == target) & (ci.baseline == comparator)].iloc[0]
            left.errorbar(row.gain, 3 - 2*i - j, xerr=[[row.gain-row.low], [row.high-row.gain]],
                          fmt="o", color=COLORS[target], capsize=3)
        wide = docs.loc[docs.target.eq(target)].pivot(index="doc", columns="model", values="mae")
        delta = wide["mean"] - wide["calibrated"]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(delta))
        right.scatter(delta, 1-i+jitter, color=COLORS[target], s=14)
    left.set_yticks([3, 2, 1, 0], ["AChE · raw", "AChE · mean", "GSK-3β · raw", "GSK-3β · mean"])
    right.set_yticks([1, 0], list(TARGETS.values()))
    for ax in (left, right):
        ax.axvline(0, ls="--", color="#697880", lw=1)
        ax.set_xlabel("Comparator MAE − calibrated MAE")
        ax.margins(y=.23)
    left.set_title("A   Mean paired difference", loc="left")
    right.set_xlabel("Reference mean MAE − calibrated MAE")
    right.set_title("B   Individual documents", loc="left")
    save(fig, "figure_2_paired_differences.png")


def error_components(primary):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True, layout="constrained")
    for ax, (target, title) in zip(axes, TARGETS.items()):
        d = primary.loc[primary.target.eq(target)].set_index("model").loc[["raw", "mean", "calibrated"]]
        x = np.arange(3)
        ax.bar(x, d.level_mse, color="#2c6578", label="Squared mean error")
        ax.bar(x, d.within_mse, bottom=d.level_mse, color="#c7dee2", label="Residual variance")
        for pos, mse in enumerate(d.mse):
            ax.text(pos, mse+.025, f"{mse:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, ["Raw", "Reference\nmean", "Calibrated"])
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 2)
    axes[0].set_ylabel("Mean squared error (pIC₅₀²)")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "figure_3_error_components.png")


def reference_count(common):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True, layout="constrained")
    for ax, (target, title) in zip(axes, TARGETS.items()):
        means = common.loc[common.target.eq(target)].pivot(index="k", columns="model", values="mae")
        for key, label, style, color in [
                ("raw", "Raw CatBoost", ":", "#707982"), ("mean", "Reference mean", "s--", "#ad5b21"),
                ("calibrated", "Calibrated CatBoost", "o-", "#126477")]:
            ax.plot(means.index, means[key], style, color=color, label=label, markerfacecolor="white")
        ax.set(xticks=[1, 3, 5], xlabel="Number of reference compounds", title=title)
    axes[0].set_ylabel("Mean absolute error (pIC₅₀)")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "figure_4_reference_count.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    summary = pd.read_csv(HOLDOUT / "summary.csv")
    primary = summary.query("scope == 'primary' and k == 3")
    tables(primary, pd.read_csv(RESULTS / "source_sensitivity.csv"))
    differences()
    error_components(primary)
    reference_count(summary.query("scope == 'common_query'"))
