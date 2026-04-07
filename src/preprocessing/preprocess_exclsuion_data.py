import pandas as pd
from pathlib import Path


def clean_exclusion_data(input_path: str, output_path: str):
    print("Loading exclusion dataset...")
    df = pd.read_csv(input_path, low_memory=False)

    # -------------------------------
    # STEP 1 — Keep useful columns
    # -------------------------------
    cols_to_keep = [
        "LASTNAME", "FIRSTNAME", "MIDNAME", "BUSNAME",
        "GENERAL", "SPECIALTY",
        "NPI",
        "CITY", "STATE",
        "EXCLTYPE", "EXCLDATE", "REINDATE"
    ]

    df = df[cols_to_keep].copy()

    # -------------------------------
    # STEP 2 — Rename columns
    # -------------------------------
    df = df.rename(columns={
        "LASTNAME": "last_name",
        "FIRSTNAME": "first_name",
        "MIDNAME": "middle_name",
        "BUSNAME": "business_name",
        "GENERAL": "general_category",
        "SPECIALTY": "specialty",
        "NPI": "npi",
        "CITY": "city",
        "STATE": "state",
        "EXCLTYPE": "exclusion_type",
        "EXCLDATE": "exclusion_date",
        "REINDATE": "reinstatement_date"
    })

    # -------------------------------
    # STEP 3 — Clean text columns
    # -------------------------------
    text_cols = [
        "last_name", "first_name", "middle_name",
        "business_name", "general_category", "specialty",
        "city", "state", "exclusion_type"
    ]

    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    df["city"] = df["city"].str.upper()
    df["state"] = df["state"].str.upper()

    # -------------------------------
    # STEP 4 — Clean NPI (CRITICAL)
    # -------------------------------
    df["npi"] = pd.to_numeric(df["npi"], errors="coerce")

    # Remove missing NPIs
    df = df.dropna(subset=["npi"])

    # Remove invalid NPIs (0)
    df = df[df["npi"] != 0]

    df["npi"] = df["npi"].astype("int64")

    # -------------------------------
    # STEP 5 — Clean dates
    # -------------------------------
    df["exclusion_date"] = pd.to_datetime(
        df["exclusion_date"].astype("string"),
        format="%Y%m%d",
        errors="coerce"
    )

    df["reinstatement_date"] = pd.to_datetime(
        df["reinstatement_date"].astype("string"),
        format="%Y%m%d",
        errors="coerce"
    )

    # -------------------------------
    # STEP 6 — Remove duplicates
    # -------------------------------
    df = df.drop_duplicates(subset=["npi"])

    # -------------------------------
    # STEP 7 — Save cleaned dataset
    # -------------------------------
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    # -------------------------------
    # INFO
    # -------------------------------
    print("\n✅ Exclusion dataset cleaned")
    print(f"Shape: {df.shape}")
    print(f"Unique NPIs: {df['npi'].nunique()}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    input_csv = "data/raw/exclusion.csv"
    output_csv = "data/preprocessed/exclusion_cleaned.csv"

    clean_exclusion_data(input_csv, output_csv)