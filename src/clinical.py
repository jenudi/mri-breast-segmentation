from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_DEFINITIONS = {
    "er": {0: "neg", 1: "pos"},
    "pr": {0: "neg", 1: "pos"},
    "her2": {0: "neg", 1: "pos", 2: "borderline"},
    "mol_subtype": {
        0: "luminal-like",
        1: "ER/PR pos, HER2 pos",
        2: "her2",
        3: "trip neg",
    },
    "nottingham_grade": {1: "low", 2: "intermediate", 3: "high"},
}


def load_clinical_data(excel_path: Path) -> pd.DataFrame:
    """
    Load and clean structured clinical data.
    """
    df = pd.read_excel(excel_path, sheet_name="Data", header=1)
    df = df.drop(index=0)  # remove legend row

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df = df.set_index("patient_id")
    df.index = [int(i.split("_")[-1]) for i in df.index]

    return df


def plot_clinical_distributions(
    df: pd.DataFrame,
    save_path: Path | None = None
) -> None:
    """
    Plot histograms of categorical clinical variables.
    """
    fig, axes = plt.subplots(1, len(df.columns), figsize=(18, 4))

    for ax, col in zip(axes, df.columns):
        counts = df[col].value_counts(dropna=False).sort_index()
        codes = list(counts.index)
        values = counts.values

        ax.bar(range(len(codes)), values)

        mapping = FEATURE_DEFINITIONS[col]
        labels = [
            "NaN" if pd.isna(c) else mapping.get(int(c), str(c))
            for c in codes
        ]

        ax.set_xticks(range(len(codes)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle("Distribution of Clinical Features", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
