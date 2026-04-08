import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------------
# Ratio / billing features
# ------------------------------------------------------------------

def add_billing_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add billing-pattern ratio features on top of the four already
    created during preprocessing.
    """
    # How inflated is the submitted charge vs. what Medicare allows?
    # High ratio → possible upcoding.
    df["charge_to_allowed_ratio"] = (
        df["total_submitted_charge"]
        / df["total_allowed_amount"].replace(0, pd.NA)
    )

    # What fraction of the allowed amount does Medicare actually pay?
    # Anomalies can indicate unusual coverage patterns.
    df["payment_to_allowed_ratio"] = (
        df["total_payment"]
        / df["total_allowed_amount"].replace(0, pd.NA)
    )

    # Standardized payment vs. actual payment ratio.
    # Large divergence can indicate geographic or risk-adjustment anomalies.
    df["standardized_to_payment_ratio"] = (
        df["total_standardized_payment"]
        / df["total_payment"].replace(0, pd.NA)
    )

    # How varied are the procedure codes relative to total services?
    # Low ratio = repetitive billing of the same codes.
    df["hcpcs_diversity_ratio"] = (
        df["total_hcpcs_codes"]
        / df["total_services"].replace(0, pd.NA)
    )

    ratio_cols = [
        "charge_to_allowed_ratio",
        "payment_to_allowed_ratio",
        "standardized_to_payment_ratio",
        "hcpcs_diversity_ratio",
    ]
    for col in ratio_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ------------------------------------------------------------------
# Categorical encoding
# ------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary-encode low-cardinality categorical columns.
    provider_type (104 unique values) is kept as-is for now because
    it is used as the grouping key in specialty z-scores; the modeling
    script will one-hot or target-encode it as needed.
    """
    # entity_type: I (individual) → 1, O (organisation) → 0
    df["is_individual"] = (df["entity_type"] == "I").astype(int)

    # medicare_participation: Y → 1, N → 0
    df["is_participating"] = (df["medicare_participation"] == "Y").astype(int)

    # drug_suppression_indicator:
    #   '*' → suppressed because <11 beneficiaries (small volume)
    #   '#' → suppressed because of a special rule
    #   'Unknown' (filled from NaN) → not suppressed / not applicable
    df["drug_suppressed"] = (df["drug_suppression_indicator"] != "Unknown").astype(int)

    return df


# ------------------------------------------------------------------
# Specialty-aware z-scores
# ------------------------------------------------------------------

def add_specialty_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each numeric billing column, compute how many standard
    deviations a provider sits from the median of their specialty
    (provider_type group).  This surfaces unusual billing behaviour
    within a peer group rather than across the entire population.

    Uses median and MAD (median absolute deviation) instead of
    mean/std because the distributions are heavily right-skewed.
    """
    numeric_cols = [
        "total_hcpcs_codes",
        "total_beneficiaries",
        "total_services",
        "total_submitted_charge",
        "total_allowed_amount",
        "total_payment",
        "total_standardized_payment",
        "avg_patient_risk_score",
        "services_per_beneficiary",
        "charge_per_service",
        "payment_per_service",
        "allowed_per_service",
        "charge_to_allowed_ratio",
        "hcpcs_diversity_ratio",
    ]

    for col in numeric_cols:
        group_median = df.groupby("provider_type")[col].transform("median")
        group_mad = df.groupby("provider_type")[col].transform(
            lambda x: (x - x.median()).abs().median()
        )
        # Avoid division by zero for specialties with zero MAD
        z_col = (df[col] - group_median) / group_mad.replace(0, pd.NA)
        df[f"z_{col}"] = pd.to_numeric(z_col, errors="coerce").fillna(0)

    return df


# ------------------------------------------------------------------
# Drop columns not used as model features
# ------------------------------------------------------------------

def select_model_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return (X, y, provider_ids) ready for modeling.

    Drops:
    - identifier columns (provider_id kept separately)
    - raw text / address columns
    - exclusion metadata columns (leaked label proxies)
    - original categorical columns that have been encoded
    """
    drop_cols = [
        # identifiers
        "provider_id",
        # address / text (not used as features)
        "city",
        # raw categoricals replaced by encoded versions
        "entity_type",
        "medicare_participation",
        "drug_suppression_indicator",
        # exclusion metadata — these are post-label columns, not features
        "last_name", "first_name", "middle_name", "business_name",
        "general_category", "specialty", "city_excl", "state_excl",
        "exclusion_type", "exclusion_date", "reinstatement_date",
        # target
        "label",
    ]

    # provider_type is kept as a string for now; the caller handles encoding
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop)
    y = df["label"]
    provider_ids = df["provider_id"]

    return X, y, provider_ids


# ------------------------------------------------------------------
# Drop columns not needed in the feature file
# ------------------------------------------------------------------

def drop_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that must not appear in provider_features.csv:

    - Exclusion metadata: only populated for the 187 positive rows.
      Keeping them would cause data leakage (model learns "non-empty
      last_name = fraud" rather than any real billing pattern).
    - Original categoricals: already encoded into 0/1 flag columns,
      so the raw string versions are redundant.
    - Raw address text: city is not used as a model feature.
    """
    cols_to_drop = [
        # exclusion metadata (post-label, leaks the target)
        "last_name", "first_name", "middle_name", "business_name",
        "general_category", "specialty", "city_excl", "state_excl",
        "exclusion_type", "exclusion_date", "reinstatement_date",
        # raw categoricals replaced by encoded flags
        "entity_type",
        "medicare_participation",
        "drug_suppression_indicator",
        # raw address text
        "city",
    ]

    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------

def build_features(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Load the labeled dataset, engineer all features, and save.
    Returns the full dataframe (features + label + provider_id).
    """
    print("Loading labeled dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded: {df.shape}")

    print("Adding billing ratio features...")
    df = add_billing_ratios(df)

    print("Encoding categorical columns...")
    df = encode_categoricals(df)

    print("Computing specialty z-scores (this may take a moment)...")
    df = add_specialty_zscores(df)

    print("Dropping metadata and redundant columns...")
    df = drop_metadata_columns(df)

    print(f"Feature engineering complete. Final shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_file = project_root / "data" / "final" / "provider_with_labels.csv"
    output_file = project_root / "data" / "final" / "provider_features.csv"

    build_features(input_path=input_file, output_path=output_file)
