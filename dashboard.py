"""
Medicare Provider Fraud Detection — classic Streamlit dashboard.
Run: ``streamlit run dashboard.py``

For the enterprise-styled UI see ``dashboard_pro.py`` (same behaviour + extras).
"""

from __future__ import annotations

import streamlit as st

import dashboard_core as c

# ── Page & theme (classic) ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medicare Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    html, body, [class*="css"] { color-scheme: dark; }
    .stApp { background-color: #11111b; }
    section[data-testid="stSidebar"] > div { background-color: #181825; }
    div[data-testid="stAppViewContainer"] .main .block-container,
    div[data-testid="stAppViewContainer"] section[data-testid="stMain"] .block-container {
        padding-top: 0.85rem !important;
    }
    .metric-card {
        background-color: #1e1e2e; border-radius: 12px; padding: 1rem 1.25rem;
        border: 1px solid #313244; border-left: 4px solid var(--accent, #89b4fa);
        height: 100%; box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    .metric-card .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #a6adc8; margin-bottom: 0.35rem; }
    .metric-card .value { font-size: 1.75rem; font-weight: 700; color: #cdd6f4; line-height: 1.2; }
    .dash-header { margin: 0 0 0.5rem 0; padding: 0 0 0.45rem 0; border-bottom: 1px solid #313244; }
    .dash-header .dash-title { font-size: 1.85rem; font-weight: 800; color: #cba6f7; line-height: 1.15; margin: 0 0 0.2rem 0 !important; padding: 0 !important; }
    .dash-header .dash-sub { color: #bac2de; font-size: 0.92rem; line-height: 1.35; margin: 0 !important; padding: 0 !important; }
    div[data-testid="stMetricValue"] { color: #cdd6f4; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Medicare Fraud Detection")
    st.markdown(
        "<small>Interactive dashboard for provider-level risk scores from two ensemble strategies on "
        "~1.26M CMS Medicare providers.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    approach = st.radio("Approach", ["Original Ensemble", "Iterative Ensemble", "Compare Both"], index=0)
    top_k = st.slider("Top-K providers (table)", min_value=50, max_value=1000, value=200, step=50)
    fraud_only = st.checkbox("Show only confirmed fraudsters", value=False)
    st.markdown("---")
    st.markdown("### Data & pipeline")
    st.markdown(
        """
**1. Download from official sources**

- [CMS — Medicare Physician / Other Supplier utilization & payment](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-provider-utilization-and-payment-data-physician-and-other-supplier)
  Use the public PUF that matches your project year. Save as `data/raw/provider.csv`.

- [HHS-OIG — LEIE exclusion list (CSV)](https://oig.hhs.gov/exclusions/exclusions_list.asp)
  Export the **CSV** and save as `data/raw/exclusion.csv`.
"""
    )
    raw_dir = c.ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    up_prov = st.file_uploader("Upload `provider.csv`", type=["csv"], key="dash_upload_provider")
    up_excl = st.file_uploader("Upload `exclusion.csv`", type=["csv"], key="dash_upload_exclusion")
    if up_prov is not None:
        (raw_dir / "provider.csv").write_bytes(up_prov.getvalue())
    if up_excl is not None:
        (raw_dir / "exclusion.csv").write_bytes(up_excl.getvalue())
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Provider file")
        st.write("✅ present" if c.PATH_RAW_PROVIDER.is_file() else "⏳ missing")
    with c2:
        st.caption("Exclusion file")
        st.write("✅ present" if c.PATH_RAW_EXCLUSION.is_file() else "⏳ missing")
    if not c.models_present():
        st.warning(
            "Trained model files are missing under `models/`. "
            "Scoring will fail until `lightgbm_model.joblib`, `xgboost_model.joblib`, "
            "`logistic_regression.joblib`, and `catboost_model.cbm` are present."
        )
    smoke = st.checkbox("Smoke test (limit provider rows)", value=False, help="Faster debug run; not for final results.")
    nrows_smoke = None
    if smoke:
        nrows_smoke = st.number_input("Max rows from raw provider CSV", min_value=1000, max_value=500000, value=20000, step=1000)
    if "pipeline_log" not in st.session_state:
        st.session_state.pipeline_log = ""
    run_clicked = st.button(
        "Run full data pipeline + score (original ensemble)",
        disabled=not (c.PATH_RAW_PROVIDER.is_file() and c.PATH_RAW_EXCLUSION.is_file() and c.models_present()),
        help="Preprocess → labels → features → inference. Can take a long time on the full CMS file.",
    )
    if run_clicked:
        with st.status("Running pipeline…", expanded=True) as status:
            proc = c.run_pipeline_subprocess(c.ROOT, nrows=nrows_smoke)
            st.session_state.pipeline_log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            if proc.returncode == 0:
                status.update(label="Pipeline finished successfully", state="complete")
                st.cache_data.clear()
                st.success("Data refreshed. Cached tables were cleared.")
            else:
                status.update(label="Pipeline failed", state="error")
                st.error(f"Exit code {proc.returncode}. See log below.")
    if st.session_state.pipeline_log:
        with st.expander("Last pipeline log", expanded=False):
            st.code(st.session_state.pipeline_log, language="text")
    st.caption(
        "Training metrics and plots under `outputs/` are from the original model training run; "
        "this button only rebuilds derived data and **scored_providers.csv** using saved weights."
    )
    st.markdown("---")
    st.markdown("### Download center")
    with st.expander("Raw data", expanded=False):
        c.download_button("data/raw/provider.csv", "Raw CMS Medicare provider billing data (~1.26M rows).", key_prefix="cl")
        c.download_button("data/raw/exclusion.csv", "OIG exclusion list (187 fraudsters).", key_prefix="cl")
    with st.expander("Preprocessed", expanded=False):
        c.download_button("data/preprocessed/provider_cleaned.csv", "Cleaned provider table.", key_prefix="cl")
        c.download_button("data/preprocessed/exclusion_cleaned.csv", "Cleaned exclusion list.", key_prefix="cl")
    with st.expander("Final datasets", expanded=False):
        c.download_button("data/final/provider_with_labels.csv", "Providers with fraud labels.", key_prefix="cl")
        c.download_button("data/final/provider_features.csv", "57 engineered features.", key_prefix="cl")
        c.download_button("data/final/scored_providers.csv", "Original ensemble scores.", key_prefix="cl")
        c.download_button("data/final/scored_providers_iterative.csv", "Iterative ensemble scores.", key_prefix="cl")

# ── Main ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dash-header">'
    '<p class="dash-title">Medicare Provider Fraud Detection</p>'
    '<p class="dash-sub">ML project dashboard · dark theme · test-set metrics from training pipeline</p>'
    "</div>",
    unsafe_allow_html=True,
)

if approach == "Original Ensemble":
    if not c.PATH_SCORED_ORIGINAL.exists():
        st.warning(f"Missing scored file: `{c.PATH_SCORED_ORIGINAL.relative_to(c.ROOT)}`")
    else:
        s = c.load_original_scored()
        m = c.load_original_metrics()
        c.render_original_tabs(s, m, top_k, fraud_only)
elif approach == "Iterative Ensemble":
    if not c.PATH_SCORED_ITERATIVE.exists():
        st.warning(f"Missing scored file: `{c.PATH_SCORED_ITERATIVE.relative_to(c.ROOT)}`")
    else:
        s = c.load_iterative_scored()
        m = c.load_iterative_metrics()
        c.render_iterative_tabs(s, m, top_k, fraud_only)
else:
    m_o = c.load_original_metrics()
    m_i = c.load_iterative_metrics()
    c.render_compare_view(m_o, m_i)
