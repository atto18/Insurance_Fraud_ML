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
    [data-testid="stStatusWidget"] { display: none !important; }
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
    section[data-testid="stMain"] .main-sec-title {
        font-size: 1.02rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase;
        margin: 0.15rem 0 0.9rem 0 !important; padding: 0.5rem 0.9rem 0.5rem 0.75rem;
        border-radius: 0 10px 10px 0; line-height: 1.25;
    }
    section[data-testid="stMain"] .main-sec-kpi {
        color: #d4c6ff; border-left: 3px solid #cba6f7;
        background: linear-gradient(92deg, rgba(108, 70, 160, 0.38) 0%, rgba(30, 30, 46, 0.5) 100%);
        box-shadow: 0 0 0 1px rgba(203, 166, 247, 0.28);
    }
    section[data-testid="stMain"] .main-sec-chart {
        color: #b5e8e0; border-left: 3px solid #89dceb;
        background: linear-gradient(92deg, rgba(20, 90, 110, 0.5) 0%, rgba(30, 30, 46, 0.5) 100%);
        box-shadow: 0 0 0 1px rgba(137, 220, 235, 0.25);
    }
    section[data-testid="stMain"] .main-sec-table {
        color: #b4c6ff; border-left: 3px solid #89b4fa;
        background: linear-gradient(92deg, rgba(68, 85, 160, 0.4) 0%, rgba(30, 30, 46, 0.5) 100%);
        box-shadow: 0 0 0 1px rgba(137, 180, 250, 0.22);
    }
    section[data-testid="stMain"] .main-sec-title.main-sec-chart,
    section[data-testid="stMain"] .main-sec-title.main-sec-table {
        margin-top: 1.4rem !important;
    }
    section[data-testid="stMain"] .main-sec-title.main-sec-chart.main-sec-mo-gap {
        margin-top: 2.65rem !important;
    }
    section[data-testid="stMain"] .main-sec-title.main-sec-hp-gap,
    section[data-testid="stMain"] .main-sec-title.main-sec-chart.main-sec-hp-gap {
        margin-top: 1.35rem !important;
    }
    section[data-testid="stMain"] .hp-section { margin: 0 0 0.35rem 0; }
    section[data-testid="stMain"] .hp-sub {
        font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.11em; color: #a6adc8;
        margin: 0 0 0.6rem 0.1rem;
    }
    section[data-testid="stMain"] .hp-sub-tuned { margin-top: 0.2rem; color: #b4befe; }
    section[data-testid="stMain"] .hp-config-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.5rem;
    }
    section[data-testid="stMain"] .hp-cfg-chip {
        background: linear-gradient(160deg, rgba(49, 50, 68, 0.9) 0%, rgba(30, 30, 46, 0.95) 100%);
        border: 1px solid rgba(137, 180, 250, 0.25); border-radius: 10px; padding: 0.5rem 0.7rem 0.55rem 0.7rem;
    }
    section[data-testid="stMain"] .hp-cfg-chip .hpc-k {
        display: block; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #a6adc8; margin-bottom: 0.2rem;
    }
    section[data-testid="stMain"] .hp-cfg-chip .hpc-v {
        display: block; font-size: 0.9rem; font-weight: 700; color: #cdd6f4; line-height: 1.35; word-break: break-word; white-space: pre-wrap;
    }
    section[data-testid="stMain"] .hp-model-card {
        background: linear-gradient(165deg, rgba(30, 60, 80, 0.4) 0%, rgba(30, 30, 46, 0.95) 100%);
        border: 1px solid rgba(137, 220, 235, 0.28); border-radius: 12px; padding: 0.85rem 1rem 0.95rem 1rem; margin-bottom: 0.25rem;
        min-height: 24rem; display: flex; flex-direction: column; box-sizing: border-box;
    }
    section[data-testid="stMain"] .hp-tuned-body { flex: 1 1 auto; min-height: 0; }
    section[data-testid="stMain"] .hp-model-title {
        margin: 0 0 0.75rem 0; font-size: 0.95rem; font-weight: 800; color: #89dceb; letter-spacing: 0.04em; border-bottom: 1px solid rgba(137, 220, 235, 0.25);
        padding-bottom: 0.45rem;
    }
    section[data-testid="stMain"] .hp-kv {
        display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 0.4rem 0.8rem;
        padding: 0.4rem 0; border-bottom: 1px solid rgba(69, 71, 90, 0.6); font-size: 0.8rem; line-height: 1.35;
    }
    section[data-testid="stMain"] .hp-kv:last-of-type { border-bottom: none; }
    section[data-testid="stMain"] .hp-k { color: #b4befe; font-family: ui-monospace, Consolas, monospace; font-size: 0.78rem; }
    section[data-testid="stMain"] .hp-v { color: #cdd6f4; font-weight: 600; text-align: right; max-width: 60%; }
    /* st.tabs: pill-style (Base Web) */
    section[data-testid="stMain"] div[data-testid="stTabs"] { margin-top: 0.75rem; margin-bottom: 0.5rem; position: relative; z-index: 0; }
    section[data-testid="stMain"] ul[role="tablist"],
    section[data-testid="stMain"] [data-baseweb="tab-list"] {
        position: relative !important; z-index: 1 !important;
        display: flex !important; flex-wrap: nowrap !important; align-items: stretch !important;
        gap: 0.45rem 0.75rem !important; row-gap: 0.45rem !important;
        padding: 0.2rem 0.2rem 0.5rem 0.2rem !important; margin: 0 !important;
        overflow-x: auto !important; overflow-y: hidden !important;
        overscroll-behavior-x: contain; -webkit-overflow-scrolling: touch;
        border: none !important; box-shadow: none !important; background: transparent !important;
    }
    section[data-testid="stMain"] [data-baseweb="tab-border"] {
        display: none !important; height: 0 !important; max-height: 0 !important; margin: 0 !important; padding: 0 !important;
        overflow: hidden !important; pointer-events: none !important; border: none !important; visibility: hidden !important;
    }
    section[data-testid="stMain"] [data-baseweb="tab"] {
        position: relative !important; z-index: 2 !important; flex: 0 0 auto !important; list-style: none !important;
        min-height: 2.75rem !important; min-width: 2.5rem !important; margin: 0 !important; padding: 0 !important;
        display: flex !important; flex-direction: row !important; align-items: stretch !important; justify-content: stretch !important;
        background: transparent !important; border: none !important; box-shadow: none !important;
    }
    section[data-testid="stMain"] [data-baseweb="tab"] > * {
        flex: 1 1 auto !important; align-self: stretch !important; min-height: 2.75rem !important; width: 100% !important;
        margin: 0 !important; padding: 0 !important; position: static !important;
        display: flex !important; flex-direction: row !important; align-items: center !important; justify-content: center !important;
    }
    section[data-testid="stMain"] [data-baseweb="tab"] button {
        box-sizing: border-box !important; display: flex !important; flex: 1 1 100% !important;
        align-items: center !important; justify-content: center !important; align-self: stretch !important;
        color: #a6adc8 !important; font-size: clamp(0.88rem, 0.35rem + 0.95vw, 1.2rem) !important;
        font-weight: 600 !important; letter-spacing: 0.01em !important; line-height: 1.2 !important;
        min-height: 2.75rem !important; min-width: 0 !important; width: 100% !important; max-width: none !important;
        padding: 0.4rem 0.95rem !important; margin: 0 !important;
        text-align: center !important; white-space: nowrap !important; cursor: pointer !important;
        user-select: none !important; -webkit-user-select: none !important; touch-action: manipulation !important;
        -webkit-tap-highlight-color: rgba(0,0,0,0);
        background: rgba(49, 50, 68, 0.55) !important;
        border: 1px solid rgba(137, 180, 250, 0.22) !important; border-radius: 10px !important;
        box-shadow: none !important; position: relative !important; z-index: 3 !important;
        transition: color 0.1s ease, background 0.1s ease, border-color 0.1s ease, box-shadow 0.1s ease !important;
    }
    section[data-testid="stMain"] [data-baseweb="tab"] button * { user-select: none !important; cursor: inherit !important; }
    section[data-testid="stMain"] [data-baseweb="tab"] button[aria-selected="true"] {
        color: #e0e7ff !important;
        background: linear-gradient(180deg, rgba(80, 100, 180, 0.5) 0%, rgba(40, 45, 80, 0.9) 100%) !important;
        border-color: rgba(137, 180, 250, 0.5) !important;
        box-shadow: inset 0 -3px 0 0 #89b4fa, 0 0 0 1px rgba(137, 180, 250, 0.12) !important;
    }
    section[data-testid="stMain"] [data-baseweb="tab"] button:hover:not([aria-selected="true"]) {
        color: #e0e7ff !important; background: rgba(69, 71, 90, 0.7) !important; border-color: rgba(180, 190, 254, 0.38) !important;
    }
    section[data-testid="stMain"] .hp-tuned-row {
        display: flex; flex-direction: row; flex-wrap: wrap; align-items: stretch; gap: 1.25rem;
        width: 100%; margin: 0 0 0.75rem 0; box-sizing: border-box;
    }
    section[data-testid="stMain"] .hp-tuned-row .hp-model-card {
        flex: 1 1 calc(50% - 0.625rem); min-width: min(100%, 280px);
        min-height: 20rem; max-width: none;
    }
    section[data-testid="stMain"] .hp-tuned-row-single .hp-model-card {
        flex: 0 1 calc(50% - 0.625rem); max-width: calc(50% - 0.625rem);
    }
    @media (max-width: 700px) {
        section[data-testid="stMain"] .hp-tuned-row .hp-model-card { flex: 1 1 100%; min-width: 0; }
        section[data-testid="stMain"] .hp-tuned-row-single .hp-model-card { flex: 1 1 100%; max-width: 100%; }
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] p[data-testid="stFileUploaderFileSizeText"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] li small,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [aria-label="Add files"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.45rem 0.35rem !important;
        height: 0;
        border: none;
        border-top: 1px solid rgba(137, 180, 250, 0.22);
        background: none;
    }
    .dash-sidebar-section-title {
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 0.25rem 0 0.4rem 0 !important;
        padding: 0.5rem 0.65rem 0.5rem 0.7rem;
        border-radius: 0 10px 10px 0;
        line-height: 1.3;
    }
    .dash-sst-download {
        color: #f5c2c0;
        text-shadow: 0 0 16px rgba(235, 97, 97, 0.65);
        border-left: 3px solid #e78284;
        background: linear-gradient(92deg, rgba(200, 65, 65, 0.38) 0%, rgba(30, 30, 46, 0.72) 100%);
        box-shadow: 0 0 0 1px rgba(231, 130, 132, 0.45);
    }
    .dash-sst-upload {
        color: #f5e0dc;
        text-shadow: 0 0 8px rgba(88, 28, 135, 0.45);
        border-left: 3px solid #6d3f9a;
        background: linear-gradient(92deg, rgba(64, 32, 88, 0.78) 0%, rgba(40, 24, 58, 0.72) 50%, rgba(30, 30, 46, 0.85) 100%);
        box-shadow: 0 0 0 1px rgba(136, 57, 239, 0.35);
    }
    .dash-sst-tight { margin: 0.2rem 0 0.4rem 0 !important; }
    .dash-sst-analysis {
        color: #d4c6ff;
        text-shadow: 0 0 12px rgba(203, 166, 247, 0.55);
        border-left: 3px solid #cba6f7;
        background: linear-gradient(92deg, rgba(108, 70, 160, 0.55) 0%, rgba(50, 45, 90, 0.45) 50%, rgba(30, 30, 46, 0.8) 100%);
        box-shadow: 0 0 0 1px rgba(203, 166, 247, 0.35);
        margin: 0.15rem 0 0.5rem 0 !important;
    }
    .dash-sst-table {
        color: #b5e8e0;
        text-shadow: 0 0 10px rgba(137, 220, 235, 0.45);
        border-left: 3px solid #89dceb;
        background: linear-gradient(92deg, rgba(20, 90, 110, 0.5) 0%, rgba(30, 80, 100, 0.35) 50%, rgba(30, 30, 46, 0.82) 100%);
        box-shadow: 0 0 0 1px rgba(137, 220, 235, 0.3);
        margin: 0.15rem 0 0.2rem 0 !important;
    }
    .dash-sidebar-section-title.dash-sst-table.dash-sst-after-widget.dash-sst-art-tight {
        padding: 0.3rem 0.45rem 0.25rem 0.38rem !important;
        margin: 0.5rem 0 0.55rem 0 !important;
    }
    .dash-sst-after-widget { margin: 0.5rem 0 0.2rem 0 !important; }
    section[data-testid="stSidebar"] [class*="st-key-sb_topk"] { margin: 0.26rem 0 0 0 !important; }
    section[data-testid="stSidebar"] [class*="st-key-sb_fraud"] {
        margin: -0.4rem 0 0.02rem 0 !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-sb_topk"] [data-testid="stWidgetLabel"] {
        margin-top: 0.4rem !important;
        margin-bottom: 0.1rem !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-sb_fraud"] [data-testid="stWidgetLabel"] {
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stElementContainer"] hr.sb-analysing-end-hr {
        margin: 0.04rem 0 0.28rem 0;
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.3);
        background: none;
    }
    section[data-testid="stSidebar"] [class*="st-key-sb_fraud"] + [data-testid="stElementContainer"] {
        margin-top: -0.12rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] [data-testid="stSlider"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] [data-testid="stCheckbox"] {
        margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] ul { margin: 0 0 0.2rem 0 !important; }
    /* Run pipeline: deeper mauve / violet, no icon */
    section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] {
        margin-top: 0.35rem !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"] {
        width: 100% !important;
        min-height: 2.7rem !important;
        border: none !important;
        border-radius: 12px !important;
        color: #f5e0f5 !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em !important;
        background: linear-gradient(150deg, #5a2d82 0%, #3b0764 50%, #1e0b2e 100%) !important;
        box-shadow: 0 0 0 1px rgba(200, 150, 255, 0.2), 0 4px 24px rgba(30, 0, 60, 0.65) !important;
        transition: transform 0.12s, box-shadow 0.2s, filter 0.2s;
    }
    section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"]:hover:not(:disabled) {
        background: linear-gradient(150deg, #6c3b9a 0%, #4c0d6b 50%, #2d0a3d 100%) !important;
        box-shadow: 0 0 0 1px rgba(200, 160, 255, 0.35), 0 6px 28px rgba(100, 40, 160, 0.55) !important;
        transform: translateY(-1px);
    }
    section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"]:active:not(:disabled) {
        transform: translateY(0);
    }
    section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"]:disabled {
        background: linear-gradient(150deg, #44475a 0%, #313244 100%) !important;
        color: #6c7086 !important;
        box-shadow: none !important;
        transform: none !important;
    }
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
    st.markdown(
        '<p class="dash-sidebar-section-title dash-sst-analysis">Analysis mode</p>',
        unsafe_allow_html=True,
    )
    approach = st.radio(
        "Analysis mode",
        ["Original Ensemble", "Iterative Ensemble", "Compare Both"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(c.analysing_results_table_title_html(theme="dash"), unsafe_allow_html=True)
    top_k = st.slider(
        "Top-K providers (table)",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        label_visibility="collapsed",
        key="sb_topk",
    )
    fraud_only = st.checkbox("Fraud-labeled NPIs only", value=False, key="sb_fraud")
    st.markdown('<hr class="sb-analysing-end-hr" />', unsafe_allow_html=True)
    st.markdown(
        '<p class="dash-sidebar-section-title dash-sst-download">Download Raw Datasets</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
- **Download provider dataset:** [link]({c.URL_CMS_RAW_PROVIDER})
- **Download exclusion dataset:** [link]({c.URL_OIG_RAW_EXCLUSION})
"""
    )
    st.divider()
    st.markdown(
        '<p class="dash-sidebar-section-title dash-sst-upload dash-sst-tight">Upload Dataset and Run Pipeline</p>',
        unsafe_allow_html=True,
    )
    (c.ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    c.render_raw_csv_slot(
        upload_label="Upload provider.csv",
        dest=c.PATH_RAW_PROVIDER,
        display_name="provider.csv",
        state_base="d_raw_pro",
    )
    c.render_raw_csv_slot(
        upload_label="Upload exclusion.csv",
        dest=c.PATH_RAW_EXCLUSION,
        display_name="exclusion.csv",
        state_base="d_raw_excl",
    )
    if not c.models_present():
        st.warning(
            "Trained model files are missing under `models/`. "
            "Scoring will fail until `lightgbm_model.joblib`, `xgboost_model.joblib`, "
            "`logistic_regression.joblib`, and `catboost_model.cbm` are present."
        )
    if "pipeline_log" not in st.session_state:
        st.session_state.pipeline_log = ""
    run_clicked = st.button(
        "Run pipeline",
        key="runPipelineMain",
        type="primary",
        help="Preprocess → labels → features → inference. Can take a long time on the full CMS file.",
        disabled=not (c.PATH_RAW_PROVIDER.is_file() and c.PATH_RAW_EXCLUSION.is_file() and c.models_present()),
        width="stretch",
    )
    if run_clicked:
        with st.status("Running pipeline…", expanded=True) as status:
            proc = c.run_pipeline_subprocess(c.ROOT, nrows=None)
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
