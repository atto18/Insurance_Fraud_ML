import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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

    # Payment anomaly: gap between what was submitted and what was paid
    # Fraud providers tend to submit inflated charges
    df["payment_charge_gap"] = df["total_payment"] - df["total_submitted_charge"]

    # Services per unique procedure code — low diversity = repetitive billing
    df["services_per_hcpcs"] = (
        df["total_services"] / df["total_hcpcs_codes"].replace(0, pd.NA)
    ).fillna(0)

    # Risk-adjusted payment: high payment relative to patient risk score
    # Fraud providers often bill high amounts for low-risk patients
    df["payment_per_risk"] = (
        df["total_payment"] / df["avg_patient_risk_score"].replace(0, pd.NA)
    ).fillna(0)

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
        "payment_to_allowed_ratio",
        "standardized_to_payment_ratio",
        "hcpcs_diversity_ratio",
        "payment_charge_gap",
        "services_per_hcpcs",
        "payment_per_risk",
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
# Tier 3 — Z-score aggregate features
# ------------------------------------------------------------------

# The 19 specialty-z-score columns computed by add_specialty_zscores
_ZSCORE_COLS = [
    "z_total_hcpcs_codes", "z_total_beneficiaries", "z_total_services",
    "z_total_submitted_charge", "z_total_allowed_amount", "z_total_payment",
    "z_total_standardized_payment", "z_avg_patient_risk_score",
    "z_services_per_beneficiary", "z_charge_per_service",
    "z_payment_per_service", "z_allowed_per_service",
    "z_charge_to_allowed_ratio", "z_payment_to_allowed_ratio",
    "z_standardized_to_payment_ratio", "z_hcpcs_diversity_ratio",
    "z_payment_charge_gap", "z_services_per_hcpcs", "z_payment_per_risk",
]

# The raw billing + ratio columns used for anomaly detectors
_RAW_BILLING_COLS = [
    "total_hcpcs_codes", "total_beneficiaries", "total_services",
    "total_submitted_charge", "total_allowed_amount", "total_payment",
    "total_standardized_payment", "avg_patient_risk_score",
    "services_per_beneficiary", "charge_per_service",
    "payment_per_service", "allowed_per_service",
    "charge_to_allowed_ratio", "payment_to_allowed_ratio",
    "standardized_to_payment_ratio", "hcpcs_diversity_ratio",
    "payment_charge_gap", "services_per_hcpcs", "payment_per_risk",
    "is_individual", "is_participating", "drug_suppressed",
]


def add_zscore_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise all specialty z-scores into four scalar signals.

    A fraud provider often deviates on *multiple* billing dimensions
    simultaneously.  These aggregates capture that joint signal in a
    way that a tree model can split on directly.

        n_extreme_z2   — how many z-scores have |z| > 2  (mild outlier)
        n_extreme_z3   — how many z-scores have |z| > 3  (strong outlier)
        max_abs_z      — worst single deviation from specialty median
        sum_abs_z      — total anomaly burden across all z-score axes
    """
    z_cols = [c for c in _ZSCORE_COLS if c in df.columns]
    z_vals = df[z_cols].abs()

    df["n_extreme_z2"] = (z_vals > 2).sum(axis=1).astype(int)
    df["n_extreme_z3"] = (z_vals > 3).sum(axis=1).astype(int)
    df["max_abs_z"]    = z_vals.max(axis=1)
    df["sum_abs_z"]    = z_vals.sum(axis=1)

    return df


# ------------------------------------------------------------------
# Tier 3 — Specialty percentile ranks
# ------------------------------------------------------------------

def add_specialty_percentile_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each of the most fraud-discriminative billing metrics, compute
    the provider's percentile rank *within their specialty group*.

    This is complementary to z-scores: z-scores measure distance from
    the specialty median assuming a roughly symmetric distribution,
    while percentile ranks are robust to any distribution shape and
    give a clean [0, 1] signal that tree models can threshold cleanly.

    Columns added:
        pct_total_payment          — billing volume within specialty
        pct_charge_to_allowed      — upcoding signal within specialty
        pct_services_per_bene      — utilisation intensity within specialty
        pct_hcpcs_diversity        — code variety within specialty
        pct_payment_gap            — charge inflation within specialty
        pct_payment_per_risk       — risk-adjusted payment within specialty
    """
    rank_targets = {
        "pct_total_payment":     "total_payment",
        "pct_charge_to_allowed": "charge_to_allowed_ratio",
        "pct_services_per_bene": "services_per_beneficiary",
        "pct_hcpcs_diversity":   "hcpcs_diversity_ratio",
        "pct_payment_gap":       "payment_charge_gap",
        "pct_payment_per_risk":  "payment_per_risk",
    }
    for new_col, src_col in rank_targets.items():
        if src_col in df.columns:
            df[new_col] = (
                df.groupby("provider_type")[src_col]
                  .rank(pct=True)
                  .astype("float32")
            )
    return df


# ------------------------------------------------------------------
# Tier 2 — Unsupervised anomaly scores
# ------------------------------------------------------------------

def add_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three unsupervised anomaly scores as features.

    These are computed on the whole dataset with no label information,
    so there is no target leakage.  They give the supervised models a
    powerful head-start by encoding "how unusual is this provider?"
    in complementary ways.

    if_score_global
        Isolation Forest fitted on raw billing + ratio features.
        Detects providers that are globally unusual in absolute terms.
        Score is negated so that higher = more anomalous.

    if_score_zspace
        Isolation Forest fitted on the 19 specialty z-score features.
        Detects providers who are multivariate outliers *relative to
        their specialty peers*, which is the most relevant signal for
        fraud (a dermatologist billing like a surgeon is suspicious).
        Score is negated so that higher = more anomalous.

    pca_recon_error
        PCA reconstruction error on standardised raw billing features.
        Equivalent to a linear autoencoder: normal providers lie near
        the principal subspace; anomalous providers do not.
        High error → unusual billing pattern.
    """
    raw_cols   = [c for c in _RAW_BILLING_COLS if c in df.columns]
    z_cols     = [c for c in _ZSCORE_COLS       if c in df.columns]

    X_raw = df[raw_cols].fillna(0).values.astype("float32")
    X_z   = df[z_cols].fillna(0).values.astype("float32")

    # ── Global Isolation Forest ──────────────────────────────────────
    print("    Fitting global Isolation Forest …")
    scaler_raw = StandardScaler()
    X_raw_sc   = scaler_raw.fit_transform(X_raw)

    if_global = IsolationForest(
        n_estimators=200,
        max_samples=512,      # fast on large datasets
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    if_global.fit(X_raw_sc)
    # score_samples: lower = more anomalous → negate for intuitive direction
    df["if_score_global"] = -if_global.score_samples(X_raw_sc)

    # ── Specialty-space Isolation Forest ────────────────────────────
    print("    Fitting specialty-space Isolation Forest …")
    if_z = IsolationForest(
        n_estimators=200,
        max_samples=512,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    if_z.fit(X_z)
    df["if_score_zspace"] = -if_z.score_samples(X_z)

    # ── PCA reconstruction error ─────────────────────────────────────
    print("    Fitting PCA reconstruction error …")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca   = pca.fit_transform(X_raw_sc)
    X_recon = pca.inverse_transform(X_pca)
    df["pca_recon_error"] = np.mean((X_raw_sc - X_recon) ** 2, axis=1)

    n_comp = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {n_comp} components explain {var_exp*100:.1f}% variance")

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

    print("Adding z-score aggregate features (Tier 3)...")
    df = add_zscore_aggregates(df)

    print("Adding specialty percentile rank features (Tier 3)...")
    df = add_specialty_percentile_ranks(df)

    print("Adding unsupervised anomaly scores (Tier 2) — may take a few minutes...")
    df = add_anomaly_scores(df)

    print("Dropping metadata and redundant columns...")
    df = drop_metadata_columns(df)

    print(f"Feature engineering complete. Final shape: {df.shape}")
    new_cols = [
        "n_extreme_z2", "n_extreme_z3", "max_abs_z", "sum_abs_z",
        "pct_total_payment", "pct_charge_to_allowed", "pct_services_per_bene",
        "pct_hcpcs_diversity", "pct_payment_gap", "pct_payment_per_risk",
        "if_score_global", "if_score_zspace", "pca_recon_error",
    ]
    print(f"  New features added: {[c for c in new_cols if c in df.columns]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_file = project_root / "data" / "final" / "provider_with_labels.csv"
    output_file = project_root / "data" / "final" / "provider_features.csv"

    build_features(input_path=input_file, output_path=output_file)
