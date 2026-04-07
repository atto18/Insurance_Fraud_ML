import pandas as pd
from pathlib import Path


def build_labeled_dataset(
    provider_input_path: Path,
    exclusion_input_path: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Join cleaned provider data with cleaned exclusion data using NPI/provider_id
    and create the binary label.
    """

    print("Loading cleaned provider dataset...")
    provider_df = pd.read_csv(provider_input_path, low_memory=False)

    print("Loading cleaned exclusion dataset...")
    exclusion_df = pd.read_csv(exclusion_input_path, low_memory=False)

    # -------------------------------
    # STEP 1 — Clean join columns
    # -------------------------------
    provider_df["provider_id"] = pd.to_numeric(provider_df["provider_id"], errors="coerce")
    exclusion_df["npi"] = pd.to_numeric(exclusion_df["npi"], errors="coerce")

    provider_df = provider_df.dropna(subset=["provider_id"]).copy()
    exclusion_df = exclusion_df.dropna(subset=["npi"]).copy()

    provider_df["provider_id"] = provider_df["provider_id"].astype("int64")
    exclusion_df["npi"] = exclusion_df["npi"].astype("int64")

    exclusion_df = exclusion_df.drop_duplicates(subset=["npi"]).copy()

    # -------------------------------
    # STEP 2 — Join datasets
    # -------------------------------
    print("Joining datasets on provider_id and npi...")
    labeled_df = provider_df.merge(
        exclusion_df,
        how="left",
        left_on="provider_id",
        right_on="npi",
        suffixes=("", "_excl")
    )

    # -------------------------------
    # STEP 3 — Create label
    # -------------------------------
    labeled_df["label"] = labeled_df["npi"].notna().astype(int)

    # -------------------------------
    # STEP 4 — Cleanup
    # -------------------------------
    if "npi" in labeled_df.columns:
        labeled_df = labeled_df.drop(columns=["npi"])

    # -------------------------------
    # STEP 5 — Stats
    # -------------------------------
    print("\nJoin completed.")
    print(f"Final shape: {labeled_df.shape}")

    print("\nLabel distribution:")
    print(labeled_df["label"].value_counts(dropna=False))

    print("\nPositive ratio:")
    print(labeled_df["label"].mean())

    # -------------------------------
    # STEP 6 — Save
    # -------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_path, index=False)

    print(f"\nSaved labeled dataset to: {output_path}")

    return labeled_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    provider_input = project_root / "data" / "preprocessed" / "provider_cleaned.csv"
    exclusion_input = project_root / "data" / "preprocessed" / "exclusion_cleaned.csv"
    output_file = project_root / "data" / "final" / "provider_with_labels.csv"

    print("Project root:", project_root)
    print("Provider file exists:", provider_input.exists(), provider_input)
    print("Exclusion file exists:", exclusion_input.exists(), exclusion_input)

    build_labeled_dataset(
        provider_input_path=provider_input,
        exclusion_input_path=exclusion_input,
        output_path=output_file
    )