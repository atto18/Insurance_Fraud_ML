import pandas as pd
from pathlib import Path


def load_provider_data(input_path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load the raw provider dataset.

    Args:
        input_path: Path to the raw CSV file.
        nrows: Optional number of rows to load for testing.

    Returns:
        Raw provider dataframe.
    """
    return pd.read_csv(input_path, nrows=nrows, low_memory=False)


def select_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns useful for the first version of the ML project.
    """
    selected_columns = [
        "Rndrng_NPI",
        "Rndrng_Prvdr_Ent_Cd",
        "Rndrng_Prvdr_City",
        "Rndrng_Prvdr_State_Abrvtn",
        "Rndrng_Prvdr_RUCA",
        "Rndrng_Prvdr_Type",
        "Rndrng_Prvdr_Mdcr_Prtcptg_Ind",
        "Tot_HCPCS_Cds",
        "Tot_Benes",
        "Tot_Srvcs",
        "Tot_Sbmtd_Chrg",
        "Tot_Mdcr_Alowd_Amt",
        "Tot_Mdcr_Pymt_Amt",
        "Tot_Mdcr_Stdzd_Amt",
        "Drug_Sprsn_Ind",
        "Bene_Avg_Risk_Scre",
    ]

    missing_cols = [col for col in selected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    return df[selected_columns].copy()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename CMS columns into simpler ML-friendly names.
    """
    rename_dict = {
        "Rndrng_NPI": "provider_id",
        "Rndrng_Prvdr_Ent_Cd": "entity_type",
        "Rndrng_Prvdr_City": "city",
        "Rndrng_Prvdr_State_Abrvtn": "state",
        "Rndrng_Prvdr_RUCA": "ruca_code",
        "Rndrng_Prvdr_Type": "provider_type",
        "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "medicare_participation",
        "Tot_HCPCS_Cds": "total_hcpcs_codes",
        "Tot_Benes": "total_beneficiaries",
        "Tot_Srvcs": "total_services",
        "Tot_Sbmtd_Chrg": "total_submitted_charge",
        "Tot_Mdcr_Alowd_Amt": "total_allowed_amount",
        "Tot_Mdcr_Pymt_Amt": "total_payment",
        "Tot_Mdcr_Stdzd_Amt": "total_standardized_payment",
        "Drug_Sprsn_Ind": "drug_suppression_indicator",
        "Bene_Avg_Risk_Scre": "avg_patient_risk_score",
    }
    return df.rename(columns=rename_dict)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text columns:
    - strip spaces
    - standardize case for selected columns
    """
    text_cols = [
        "entity_type",
        "city",
        "state",
        "provider_type",
        "medicare_participation",
        "drug_suppression_indicator",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    if "city" in df.columns:
        df["city"] = df["city"].str.title()

    if "state" in df.columns:
        df["state"] = df["state"].str.upper()

    if "provider_type" in df.columns:
        df["provider_type"] = df["provider_type"].str.title()

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns safely.
    """
    numeric_cols = [
        "provider_id",
        "ruca_code",
        "total_hcpcs_codes",
        "total_beneficiaries",
        "total_services",
        "total_submitted_charge",
        "total_allowed_amount",
        "total_payment",
        "total_standardized_payment",
        "avg_patient_risk_score",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with simple, reasonable defaults for the first project version.
    """
    # categorical/text
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Unknown")
    if "state" in df.columns:
        df["state"] = df["state"].fillna("Unknown")
    if "provider_type" in df.columns:
        df["provider_type"] = df["provider_type"].fillna("Unknown")
    if "entity_type" in df.columns:
        df["entity_type"] = df["entity_type"].fillna("Unknown")
    if "medicare_participation" in df.columns:
        df["medicare_participation"] = df["medicare_participation"].fillna("Unknown")
    if "drug_suppression_indicator" in df.columns:
        df["drug_suppression_indicator"] = df["drug_suppression_indicator"].fillna("Unknown")

    # numeric
    numeric_fill_zero = [
        "ruca_code",
        "total_hcpcs_codes",
        "total_beneficiaries",
        "total_services",
        "total_submitted_charge",
        "total_allowed_amount",
        "total_payment",
        "total_standardized_payment",
        "avg_patient_risk_score",
    ]

    for col in numeric_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate providers if any.
    """
    return df.drop_duplicates(subset=["provider_id"]).copy()


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple but important ratio features.
    """
    # Avoid division by zero
    df["services_per_beneficiary"] = df["total_services"] / df["total_beneficiaries"].replace(0, pd.NA)
    df["charge_per_service"] = df["total_submitted_charge"] / df["total_services"].replace(0, pd.NA)
    df["payment_per_service"] = df["total_payment"] / df["total_services"].replace(0, pd.NA)
    df["allowed_per_service"] = df["total_allowed_amount"] / df["total_services"].replace(0, pd.NA)

    # Fill resulting missing values from division by zero
    ratio_cols = [
        "services_per_beneficiary",
        "charge_per_service",
        "payment_per_service",
        "allowed_per_service",
    ]
    for col in ratio_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def clean_provider_data(input_path: str, output_path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Full provider data preprocessing pipeline.
    """
    print("Loading raw provider data...")
    df = load_provider_data(input_path, nrows=nrows)

    print("Selecting useful columns...")
    df = select_useful_columns(df)

    print("Renaming columns...")
    df = rename_columns(df)

    print("Cleaning text columns...")
    df = clean_text_columns(df)

    print("Converting numeric columns...")
    df = convert_numeric_columns(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Removing duplicates...")
    df = remove_duplicates(df)

    print("Creating ratio features...")
    df = create_ratio_features(df)

    print("Saving cleaned provider data...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("Done.")
    print(f"Final shape: {df.shape}")
    print(f"Saved to: {output_file}")

    return df


if __name__ == "__main__":
    input_csv = "data/raw/provider.csv"
    output_csv = "data/preprocessed/provider_cleaned.csv"

    clean_provider_data(
        input_path=input_csv,
        output_path=output_csv,
        nrows=None,  # put 100000 here if you want test mode first
    )