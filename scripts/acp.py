"""Genere les artefacts ACP de cadrage analytique utilises par le rapport.

Usage :
    python3 scripts/acp.py

Role dans le projet :
- `preparation.ipynb` reste le notebook de reference pour calculer l'ACP
  de cadrage analytique sur `data/crop_yield.csv` et produire les artefacts dans
  `artifacts/pca/`.
- `rapport.ipynb` ne recalcule pas l'ACP : il relit uniquement les tableaux
  et figures présents dans `artifacts/pca/`.
- ce script permet de régénérer les mêmes artefacts en mode headless, sans
  relancer tout `preparation.ipynb`, lorsque seul le rapport a besoin d'être
  rafraîchi.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.project_config import load_preparation_config

SEED = 42
PREPARATION_CONFIG = load_preparation_config(ensure_dirs=True)
DATA_PATH = PREPARATION_CONFIG["AGRI_CROP_YIELD_PATH"]
ARTIFACTS_DIR = PREPARATION_CONFIG["PCA_ARTIFACTS_DIR"]


def load_clean_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).rename(
        columns={
            "Region": "region",
            "Soil_Type": "soil_type",
            "Crop": "crop",
            "Rainfall_mm": "rainfall_mm",
            "Temperature_Celsius": "temperature_celsius",
            "Fertilizer_Used": "fertilizer_used",
            "Irrigation_Used": "irrigation_used",
            "Weather_Condition": "weather_condition",
            "Days_to_Harvest": "days_to_harvest",
            "Yield_tons_per_hectare": "yield_tons_per_hectare",
        }
    )

    categorical_cols = ["region", "soil_type", "crop", "weather_condition"]
    numeric_cols = ["rainfall_mm", "temperature_celsius", "days_to_harvest"]

    df[categorical_cols] = df[categorical_cols].apply(lambda col: col.astype(str).str.strip())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["fertilizer_used"] = df["fertilizer_used"].astype("boolean")
    df["irrigation_used"] = df["irrigation_used"].astype("boolean")

    df = df.loc[df["yield_tons_per_hectare"] >= 0].reset_index(drop=True)
    return df, numeric_cols


def save_correlation_projection(pca_input: pd.DataFrame, pca_model: PCA, pca_scores: pd.DataFrame) -> None:
    correlation = pca_input.corr().round(3)
    correlation.to_csv(ARTIFACTS_DIR / "pca_correlation.csv")

    pca_scores_sample = pca_scores.sample(n=min(5000, len(pca_scores)), random_state=SEED)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=axes[0],
    )
    axes[0].set_title("Corrélations des variables quantitatives")

    sns.scatterplot(
        data=pca_scores_sample,
        x="PC1",
        y="PC2",
        hue="yield_level",
        palette={"faible": "#457b9d", "intermediaire": "#2a9d8f", "eleve": "#e76f51"},
        alpha=0.35,
        s=18,
        ax=axes[1],
    )
    axes[1].set_title("Projection sur le plan PC1-PC2\n(points colores selon le rendement)")
    axes[1].set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%} de variance)")
    axes[1].set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%} de variance)")
    axes[1].axhline(0, color="lightgray", linewidth=1)
    axes[1].axvline(0, color="lightgray", linewidth=1)
    axes[1].legend(title="Niveau de rendement", loc="best")

    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "pca_correlation_and_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loadings_visuals(pca_model: PCA, pca_numeric_cols: list[str], pca_scores: pd.DataFrame) -> None:
    loadings = pd.DataFrame(
        pca_model.components_.T,
        index=pca_numeric_cols,
        columns=[f"PC{i + 1}" for i in range(len(pca_numeric_cols))],
    )
    variable_coords = pd.DataFrame(
        pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
        index=pca_numeric_cols,
        columns=[f"PC{i + 1}" for i in range(len(pca_numeric_cols))],
    )

    pc1_contributions = (
        variable_coords["PC1"]
        .abs()
        .sort_values(ascending=False)
        .rename("contribution_absolue_pc1")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    pc1_contributions.to_csv(ARTIFACTS_DIR / "pca_pc1_contributions.csv", index=False)

    pca_scores_sample = pca_scores.sample(n=min(5000, len(pca_scores)), random_state=SEED)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    correlation_circle = plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--")
    axes[0].add_patch(correlation_circle)

    for variable in pca_numeric_cols:
        x = variable_coords.loc[variable, "PC1"]
        y = variable_coords.loc[variable, "PC2"]
        axes[0].arrow(0, 0, x, y, color="#d62828", head_width=0.04, length_includes_head=True)
        axes[0].annotate(
            variable,
            xy=(x, y),
            xytext=(10 if x >= 0 else -10, 10 if y >= 0 else -10),
            textcoords="offset points",
            color="#1d3557",
            ha="left" if x >= 0 else "right",
            va="bottom" if y >= 0 else "top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        )

    axes[0].set_title("Cercle des corrélations (PC1-PC2)", pad=18)
    axes[0].set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%} de variance)")
    axes[0].set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%} de variance)")
    axes[0].set_xlim(-1.25, 1.25)
    axes[0].set_ylim(-1.25, 1.25)
    axes[0].set_aspect("equal", "box")
    axes[0].axhline(0, color="lightgray", linewidth=1)
    axes[0].axvline(0, color="lightgray", linewidth=1)

    axes[1].scatter(
        pca_scores_sample["PC1"],
        pca_scores_sample["PC2"],
        alpha=0.15,
        s=14,
        color="#8ecae6",
    )

    for variable in pca_numeric_cols:
        x = loadings.loc[variable, "PC1"] * 4
        y = loadings.loc[variable, "PC2"] * 4
        axes[1].arrow(0, 0, x, y, color="#d62828", head_width=0.08, length_includes_head=True)
        axes[1].annotate(
            variable,
            xy=(x, y),
            xytext=(10 if x >= 0 else -10, 10 if y >= 0 else -10),
            textcoords="offset points",
            color="#1d3557",
            ha="left" if x >= 0 else "right",
            va="bottom" if y >= 0 else "top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        )

    axes[1].set_title("Lecture de la première composante principale", pad=18)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].axhline(0, color="lightgray", linewidth=1)
    axes[1].axvline(0, color="lightgray", linewidth=1)

    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "pca_correlation_circle_and_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_variance_outputs(pca_model: PCA, pca_numeric_cols: list[str]) -> tuple[int, float]:
    variance = pd.DataFrame(
        {
            "composante": [f"PC{i + 1}" for i in range(len(pca_numeric_cols))],
            "variance_expliquee": pca_model.explained_variance_ratio_,
            "variance_cumulee": np.cumsum(pca_model.explained_variance_ratio_),
        }
    ).round(4)
    variance.to_csv(ARTIFACTS_DIR / "pca_variance.csv", index=False)

    intrinsic_dimension = int(np.argmax(variance["variance_cumulee"].to_numpy() >= 0.90) + 1)
    variance_pc2 = round(float(variance.loc[min(1, len(variance) - 1), "variance_cumulee"]), 4)
    retained_variance = round(float(variance.loc[intrinsic_dimension - 1, "variance_cumulee"]), 4)

    summary = pd.DataFrame(
        {
            "indicateur": [
                "dimension_intrinseque_90pct",
                "variance_cumulee_pc2",
                "variance_cumulee_conservee",
            ],
            "valeur": [intrinsic_dimension, variance_pc2, retained_variance],
        }
    )
    summary.to_csv(ARTIFACTS_DIR / "pca_summary.csv", index=False)

    positions = np.arange(len(variance))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(positions, variance["variance_expliquee"], color="#457b9d")
    ax1.set_xlabel("Composante principale")
    ax1.set_ylabel("Variance expliquée")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(variance["composante"])

    ax2 = ax1.twinx()
    ax2.plot(positions, variance["variance_cumulee"], marker="o", color="#e63946")
    ax2.axhline(0.90, linestyle="--", color="gray", linewidth=1)
    ax2.set_ylabel("Variance cumulée")
    ax2.set_ylim(0, 1.05)

    plt.title("Variance expliquée par les composantes principales")
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "pca_explained_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return intrinsic_dimension, retained_variance


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df, pca_numeric_cols = load_clean_dataset()
    pca_input = df[pca_numeric_cols].copy()
    pca_scaled = StandardScaler().fit_transform(pca_input)

    pca_model = PCA()
    pca_features = pca_model.fit_transform(pca_scaled)
    pca_scores = pd.DataFrame(
        pca_features,
        columns=[f"PC{i + 1}" for i in range(len(pca_numeric_cols))],
    )
    q1, q2 = df["yield_tons_per_hectare"].quantile([0.33, 0.66]).tolist()
    pca_scores["yield_level"] = pd.cut(
        df["yield_tons_per_hectare"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["faible", "intermediaire", "eleve"],
        include_lowest=True,
    )

    save_correlation_projection(pca_input, pca_model, pca_scores)
    save_loadings_visuals(pca_model, pca_numeric_cols, pca_scores)
    intrinsic_dimension, retained_variance = save_variance_outputs(pca_model, pca_numeric_cols)

    print(f"Artefacts ACP générés dans : {ARTIFACTS_DIR.resolve()}")
    print(f"Dimension intrinsèque retenue : {intrinsic_dimension}")
    print(f"Variance cumulée conservée : {retained_variance:.1%}")


if __name__ == "__main__":
    main()
