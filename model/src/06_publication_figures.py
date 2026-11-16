
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "publication_figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8
})

sns.set_style("whitegrid")

def save_figure(name):
    plt.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close()
    print("Saved:", name)


def load_matrix(name):
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def figure_1_dataset_coverage():
    data = pd.read_csv(OUT / "financial_feature_coverage.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    values = data["Coverage_Percent"].sort_values(ascending=True)

    ax.hist(
        values,
        bins=20,
        edgecolor="black",
        linewidth=0.5
    )

    ax.axvline(
        values.mean(),
        linestyle="--",
        label=f"Mean = {values.mean():.2f}%"
    )

    ax.set_xlabel("Financial variable coverage (%)")
    ax.set_ylabel("Number of variables")
    ax.set_title("Distribution of Financial Variable Coverage")
    ax.legend(frameon=False)

    save_figure("fig01_variable_coverage")


def figure_2_company_coverage():
    data = pd.read_csv(OUT / "financial_company_coverage.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.hist(
        data["Available_Features"],
        bins=25,
        edgecolor="black",
        linewidth=0.5
    )

    median = data["Available_Features"].median()

    ax.axvline(
        median,
        linestyle="--",
        label=f"Median = {median:.0f}"
    )

    ax.set_xlabel("Number of available financial variables")
    ax.set_ylabel("Number of companies")
    ax.set_title("Distribution of Company-Level Financial Coverage")
    ax.legend(frameon=False)

    save_figure("fig02_company_coverage")


def figure_3_clustering_comparison():
    data = pd.read_csv(OUT / "clustering_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for model, group in data.groupby("Embedding"):
        group = group.sort_values("K")

        ax.plot(
            group["K"],
            group["Silhouette"],
            marker="o",
            markersize=4,
            label=model
        )

    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Performance Across Embedding Methods")
    ax.set_xticks(sorted(data["K"].unique()))
    ax.legend(frameon=False)

    save_figure("fig03_clustering_comparison")


def figure_4_similarity_heatmaps():
    hybrid = load_matrix("ontology_aware_hybrid_similarity.npy")
    role = load_matrix("accounting_role_aware_similarity.npy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        hybrid,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Similarity"},
        ax=axes[0]
    )

    axes[0].set_title("(a) Ontology-Aware Hybrid")
    axes[0].set_xlabel("Financial concepts")
    axes[0].set_ylabel("Financial concepts")

    sns.heatmap(
        role,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Similarity"},
        ax=axes[1]
    )

    axes[1].set_title("(b) Accounting-Role-Aware")
    axes[1].set_xlabel("Financial concepts")
    axes[1].set_ylabel("Financial concepts")

    fig.suptitle("Comparison of Financial Concept Similarity Matrices", y=1.02)

    save_figure("fig04_similarity_heatmaps")


def figure_5_accounting_roles():
    data = pd.read_csv(OUT / "accounting_role_distribution.csv")
    data = data.sort_values("Concept_Count", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars = ax.barh(
        data["Accounting_Role"],
        data["Concept_Count"],
        edgecolor="black",
        linewidth=0.5
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            fontsize=9
        )

    ax.set_xlabel("Number of concepts")
    ax.set_ylabel("Accounting role")
    ax.set_title("Distribution of Accounting Roles")
    ax.set_xlim(0, data["Concept_Count"].max() * 1.2)

    save_figure("fig05_accounting_roles")


def figure_6_diagnostic_comparison():
    data = pd.read_csv(OUT / "diagnostic_model_comparison.csv")

    data = data[data["Status"] == "Found"].copy()

    if data.empty:
        print("No diagnostic pairs available.")
        return

    labels = [
        f"{a} vs. {b}"
        for a, b in zip(data["Concept_A"], data["Concept_B"])
    ]

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        data["Original_Hybrid"],
        width,
        label="Original hybrid"
    )

    ax.bar(
        x + width / 2,
        data["Role_Aware"],
        width,
        label="Role-aware"
    )

    ax.set_ylabel("Similarity score")
    ax.set_xlabel("Diagnostic concept pairs")
    ax.set_title("Diagnostic Comparison of Similarity Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    save_figure("fig06_diagnostic_comparison")


def main():
    figure_1_dataset_coverage()
    figure_2_company_coverage()
    figure_3_clustering_comparison()
    figure_4_similarity_heatmaps()
    figure_5_accounting_roles()
    figure_6_diagnostic_comparison()

    print("\nAll publication figures generated.")
    print("Output directory:", FIG)


if __name__ == "__main__":
    main()
