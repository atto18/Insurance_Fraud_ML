"""
Enterprise-style Medicare Provider Fraud analytics (Streamlit).

**Same features as** ``dashboard.py`` (ensembles, pipeline, downloads, SHAP) **plus**:
refined visual system, system status bar, data-quality summary, score-decile calibration chart,
and artifact timestamps.

Run from project root::

    streamlit run dashboard_pro.py
"""

from __future__ import annotations

import streamlit as st

import dashboard_core as c

# ── Design tokens: slate / indigo (professional, audit-friendly) ───────
PRO_STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] { color-scheme: dark; font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; }
  .stApp {
    background: linear-gradient(165deg, #0b1120 0%, #0f172a 40%, #111827 100%) fixed;
  }
  [data-testid="stHeader"] { background: transparent; }
  section[data-testid="stSidebar"] { border-right: 1px solid rgba(99, 102, 241, 0.12); }
  section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0c1222 0%, #0e1628 100%) !important;
  }
  div[data-testid="stAppViewContainer"] .main .block-container,
  div[data-testid="stAppViewContainer"] section[data-testid="stMain"] .block-container {
    padding-top: 1rem !important; max-width: 1480px;
  }
  .pro-hero {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.45) 0%, rgba(15, 23, 42, 0.7) 50%, rgba(15, 23, 42, 0.35) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px; padding: 1.35rem 1.6rem 1.2rem; margin: 0 0 1.1rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
  }
  .pro-hero h1 { font-size: 1.65rem; font-weight: 700; margin: 0 0 0.25rem; letter-spacing: -0.02em;
    color: #f1f5f9; }
  .pro-hero p.lead { color: #94a3b8; font-size: 0.95rem; line-height: 1.5; margin: 0; max-width: 48rem; }
  .pro-hero p.crumb { color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 0.5rem; }
  .metric-card {
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.8) 100%);
    border-radius: 14px; padding: 1.05rem 1.2rem;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-left: 3px solid var(--accent, #6366f1);
    height: 100%;
    box-shadow: 0 2px 16px rgba(0,0,0,0.25);
  }
  .metric-card .label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.35rem; }
  .metric-card .value { font-size: 1.55rem; font-weight: 700; color: #f8fafc; line-height: 1.2; }
  .status-pill { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 999px; margin-right: 0.35rem; }
  .ok { background: rgba(34, 197, 94, 0.16); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); }
  .no { background: rgba(248, 113, 113, 0.12); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.3); }
  [data-baseweb="tab"] { font-weight: 600 !important; }
  [data-baseweb="tab"] button { color: #cbd5e1 !important; }
  div[data-testid="stExpander"] details { background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(99, 102, 241, 0.1); border-radius: 10px; }
  .pro-footer { color: #64748b; font-size: 0.75rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(99, 102, 241, 0.12); }
  div[data-testid="stMetricValue"] { color: #e2e8f0; }
  .stButton>button { border-radius: 10px; font-weight: 600; }
</style>
"""


def _pill(ok: bool, label: str) -> str:
    cls = "ok" if ok else "no"
    sym = "●" if ok else "○"
    return f'<span class="status-pill {cls}">{sym} {label}</span>'


def _system_status_row() -> None:
    raw_ok = c.PATH_RAW_PROVIDER.is_file() and c.PATH_RAW_EXCLUSION.is_file()
    m_ok = c.models_present()
    scored_ok = c.PATH_SCORED_ORIGINAL.is_file()
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:0.35rem;margin:0.25rem 0 0.6rem;'>"
        f"{_pill(raw_ok, 'Raw inputs')}{_pill(m_ok, 'Model artifacts')}{_pill(scored_ok, 'Scored (original)')}"
        f"<span style='color:#64748b;font-size:0.72rem;margin-left:0.5rem;'>"
        f"Metrics: {c.file_mtime_display(c.PATH_METRICS_ORIGINAL) if c.PATH_METRICS_ORIGINAL.is_file() else '—'}"
        "</span></div>",
        unsafe_allow_html=True,
    )


def _pro_insight_strip(scored: pd.DataFrame) -> None:
    s = c.score_distribution_summary(scored)
    a, b, d = st.columns(3)
    with a:
        st.metric("Imbalance (positive %)", f"{s['pos_rate_pct']:.4f}%")
    with b:
        st.metric("Median score (fraud)", f"{s['median_pos']:.4f}" if s["median_pos"] == s["median_pos"] else "—")
    with d:
        st.metric("99th pct score (negatives)", f"{s['p99_neg']:.4f}" if s["p99_neg"] == s["p99_neg"] else "—")
    st.caption("Decile view: share of OIG-labeled providers within each model-score bucket (higher = better separation).")
    st.plotly_chart(
        c.plot_score_deciles(scored, "Calibration — label rate by model score decile"), width="stretch"
    )
    st.divider()


st.set_page_config(
    page_title="Fraud Analytics — Command Center",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(PRO_STYLE, unsafe_allow_html=True)

pro_style_table = (
    lambda t: c.style_metrics_with_bars(t, auroc_color="#a5b4fc", auprc_color="#fde047")
)
hist_style = {
    "bg": "rgba(15, 23, 42, 0.75)",
    "grid": "rgba(100, 116, 139, 0.28)",
    "c_nf": "#38bdf8",
    "c_f": "#fb7185",
}


def _render_approach_body(approach: str, top_k: int, fraud_only: bool) -> None:
    if approach == "Original Ensemble":
        if not c.PATH_SCORED_ORIGINAL.exists():
            st.warning(f"Missing scored file: `{c.PATH_SCORED_ORIGINAL.relative_to(c.ROOT)}`")
            return
        s = c.load_original_scored()
        m = c.load_original_metrics()
        _pro_insight_strip(s)
        c.render_original_tabs(
            s, m, top_k, fraud_only, style_table=pro_style_table, hist_plot_kwargs=hist_style
        )
    elif approach == "Iterative Ensemble":
        if not c.PATH_SCORED_ITERATIVE.exists():
            st.warning(f"Missing scored file: `{c.PATH_SCORED_ITERATIVE.relative_to(c.ROOT)}`")
            return
        s = c.load_iterative_scored()
        m = c.load_iterative_metrics()
        _pro_insight_strip(s)
        c.render_iterative_tabs(
            s, m, top_k, fraud_only, style_table=pro_style_table, hist_plot_kwargs=hist_style
        )
    else:
        c.render_compare_view(c.load_original_metrics(), c.load_iterative_metrics())
        st.caption(
            "Note: the comparison uses fixed **test-set** figures from the training run "
            "(`outputs/metrics.json` vs `outputs/iterative/metrics.json`)."
        )

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Command Center")
    st.caption("CMS provider data · OIG LEIE label · rank ensemble")
    st.divider()
    approach = st.radio(
        "Analysis mode",
        ["Original Ensemble", "Iterative Ensemble", "Compare Both"],
        index=0,
        help="Switch between the four-model ensemble, iterative bagging, or a metric-by-metric comparison.",
    )
    top_k = st.slider("Table rows (top by score)", 50, 1000, 200, 50)
    fraud_only = st.checkbox("Fraud-labeled NPIs only", value=False, help="Filter the ranking table to confirmed exclusions only.")
    st.divider()
    st.markdown("**Data sources**")
    st.markdown(
        """
- [CMS PUF (Physician/Supplier)](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-provider-utilization-and-payment-data-physician-and-other-supplier) → `data/raw/provider.csv`
- [HHS-OIG LEIE](https://oig.hhs.gov/exclusions/exclusions_list.asp) CSV → `data/raw/exclusion.csv`
        """
    )
    raw_dir = c.ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    up1 = st.file_uploader("Upload provider CSV", type=["csv"], key="p_prov")
    up2 = st.file_uploader("Upload exclusion CSV", type=["csv"], key="p_excl")
    if up1 is not None:
        (raw_dir / "provider.csv").write_bytes(up1.getvalue())
    if up2 is not None:
        (raw_dir / "exclusion.csv").write_bytes(up2.getvalue())
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Provider file")
        st.write("Ready" if c.PATH_RAW_PROVIDER.is_file() else "Missing")
    with c2:
        st.caption("Exclusion file")
        st.write("Ready" if c.PATH_RAW_EXCLUSION.is_file() else "Missing")
    if not c.models_present():
        st.error("Model bundle incomplete — add joblib + CatBoost under `models/`.", icon="⚠️")
    smoke = st.checkbox("Limit provider rows (smoke test)", value=False)
    nrows_smoke = None
    if smoke:
        nrows_smoke = st.number_input("Max rows (raw provider)", 1000, 500000, 20000, 1000)
    if "p_log" not in st.session_state:
        st.session_state.p_log = ""
    if st.button(
        "Run pipeline + score (original ensemble)", disabled=not (
            c.PATH_RAW_PROVIDER.is_file() and c.PATH_RAW_EXCLUSION.is_file() and c.models_present()
        ),
        width="stretch",
    ):
        with st.status("Running ETL and scoring…", expanded=True) as stu:
            proc = c.run_pipeline_subprocess(c.ROOT, nrows=nrows_smoke)
            st.session_state.p_log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            if proc.returncode == 0:
                stu.update(label="Completed", state="complete")
                st.cache_data.clear()
                st.success("Cache cleared. Reload views as needed.")
            else:
                stu.update(label="Failed", state="error")
                st.error(f"Process exited with code {proc.returncode}. See the log in the expander above.")
    if st.session_state.p_log:
        with st.expander("Process log", expanded=False):
            st.code(st.session_state.p_log, language="text")
    st.caption("Training plots in `outputs/` are static from the last local train. Pipeline rebuilds `scored_providers.csv` only.")
    st.divider()
    st.markdown("**Exports (CSV)**")
    with st.expander("Source & features", expanded=False):
        c.download_button("data/raw/provider.csv", "CMS source rows.", key_prefix="p")
        c.download_button("data/raw/exclusion.csv", "OIG exclusions.", key_prefix="p")
        c.download_button("data/preprocessed/provider_cleaned.csv", "Preprocessed provider.", key_prefix="p")
        c.download_button("data/preprocessed/exclusion_cleaned.csv", "Preprocessed exclusion.", key_prefix="p")
        c.download_button("data/final/provider_with_labels.csv", "Labeled providers.", key_prefix="p")
        c.download_button("data/final/provider_features.csv", "Engineered feature matrix.", key_prefix="p")
    with st.expander("Scores", expanded=True):
        c.download_button("data/final/scored_providers.csv", "Original ensemble scores.", key_prefix="p")
        c.download_button("data/final/scored_providers_iterative.csv", "Iterative scores.", key_prefix="p")

# ── Main canvas ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="pro-hero">'
    '<p class="crumb">Analytics · Provider integrity · OIG / CMS</p>'
    "<h1>Medicare provider fraud — analytics suite</h1>"
    "<p class=\"lead\">Rank-based ensemble of gradient-boosting and linear baselines on ~1.26M NPIs, "
    "with OIG exclusion list as a weak supervision target. For investigation support — not a legal finding.</p>"
    "</div>",
    unsafe_allow_html=True,
)
_system_status_row()
_render_approach_body(approach, top_k, fraud_only)
st.markdown(
    '<p class="pro-footer">© Academic / internal use. Data subject to CMS & HHS redistribution policies. '
    "For reproducibility, pin raw file vintages; scores shift when CMS refreshes the PUF.</p>",
    unsafe_allow_html=True,
)
