"""
Enterprise-style Medicare Provider Fraud analytics (Streamlit).

<<<<<<< HEAD
**Same features as** ``dashboard.py`` (ensembles, pipeline, downloads, SHAP) **plus**:
refined visual system, system status bar, data-quality summary, score-decile calibration chart,
and artifact timestamps.
=======
**Same features as** ``dashboard.py`` (ensembles, pipeline) **plus**
a refined dark visual system.
>>>>>>> Features

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
<<<<<<< HEAD
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
=======
  /* Hides the dev strip: "File change", Rerun, Always rerun (same node as the header running hint) */
  [data-testid="stStatusWidget"] { display: none !important; }
  section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(99, 102, 241, 0.12);
    padding-top: 0 !important; margin-top: 0 !important;
  }
  section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0c1222 0%, #0e1628 100%) !important;
    padding-top: 0 !important; margin-top: 0 !important;
  }
  /* Collapse control row: minimal height (removes the large “air gap” above the brand) */
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    padding: 0.2rem 0.6rem 0.15rem 0.6rem !important;
    margin: 0 !important;
    min-height: 0 !important; height: auto !important; max-height: 2.75rem;
    line-height: 1 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0 !important; margin: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div {
    padding-top: 0 !important; margin-top: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0 !important; margin: 0 !important;
  }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important; padding-bottom: 0.75rem !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding: 0 !important; margin: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child {
    margin-top: -0.35rem !important; padding-top: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child [data-testid="stMarkdown"] {
    margin: 0 !important; padding: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child [data-testid="stMarkdownContainer"] {
    margin-top: 0 !important; padding-top: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child [data-testid="element-container"] {
    margin-top: 0 !important; padding-top: 0 !important;
  }
  /* Subtle line between sidebar sections (after download links, before upload) */
  section[data-testid="stSidebar"] hr {
    margin: 0.45rem 0.35rem !important;
    height: 0;
    border: none;
    border-top: 1px solid rgba(99, 102, 241, 0.28);
    background: none;
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.35);
  }
  /* Dropzone: “N GB per file • CSV” (see stFileUploaderDropzoneInstructions in Streamlit) */
  section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
  }
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] p[data-testid="stFileUploaderFileSizeText"],
  /* File row: byte size (keep delete / name row) */
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] li small,
  /* Single file: hide extra “+” to add more files */
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] [aria-label="Add files"] {
    display: none !important;
  }
  .pro-sidebar-section-title {
    font-size: 0.9rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    margin: 0.3rem 0 0.45rem 0 !important;
    padding: 0.5rem 0.65rem 0.5rem 0.7rem;
    border-radius: 0 10px 10px 0;
    line-height: 1.3;
  }
  .pro-sst-download {
    color: #fecaca;
    text-shadow: 0 0 18px rgba(248, 113, 113, 0.7), 0 0 32px rgba(239, 68, 68, 0.35);
    border-left: 3px solid #f87171;
    background: linear-gradient(92deg, rgba(185, 28, 28, 0.55) 0%, rgba(127, 29, 29, 0.28) 48%, rgba(15, 23, 42, 0.4) 100%);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.07) inset, 0 0 0 1px rgba(248, 113, 113, 0.35);
  }
  .pro-sst-upload {
    color: #ffedd5;
    text-shadow: 0 0 10px rgba(194, 65, 12, 0.55);
    border-left: 3px solid #c2410c;
    background: linear-gradient(92deg, rgba(120, 40, 6, 0.82) 0%, rgba(80, 28, 8, 0.58) 50%, rgba(15, 23, 42, 0.55) 100%);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.05) inset, 0 0 0 1px rgba(194, 65, 12, 0.45);
  }
  .pro-sst-tight { margin: 0.2rem 0 0.4rem 0 !important; }
  /* Analysis block: each title has its own accent (indigo / teal / amber) */
  .pro-sst-analysis {
    color: #e0e7ff;
    text-shadow: 0 0 10px rgba(99, 102, 241, 0.45);
    border-left: 3px solid #818cf8;
    background: linear-gradient(92deg, rgba(79, 70, 229, 0.55) 0%, rgba(55, 48, 163, 0.32) 48%, rgba(15, 23, 42, 0.45) 100%);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.05) inset, 0 0 0 1px rgba(129, 140, 248, 0.35);
    margin: 0.15rem 0 0.5rem 0 !important;
  }
  .pro-sst-table {
    color: #ccfbf1;
    text-shadow: 0 0 10px rgba(45, 212, 191, 0.42);
    border-left: 3px solid #2dd4bf;
    background: linear-gradient(92deg, rgba(15, 118, 110, 0.55) 0%, rgba(13, 148, 136, 0.28) 45%, rgba(15, 23, 42, 0.45) 100%);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 0 0 1px rgba(45, 212, 191, 0.3);
    margin: 0.15rem 0 0.2rem 0 !important;
  }
  /* Analysing Results Table: room below the teal bar before the slider / “200” value */
  .pro-sidebar-section-title.pro-sst-table.pro-sst-after-widget.pro-sst-art-tight {
    padding: 0.3rem 0.45rem 0.25rem 0.38rem !important;
    margin: 0.5rem 0 0.55rem 0 !important;
  }
  .pro-sst-after-widget { margin: 0.5rem 0 0.2rem 0 !important; }
  /* Analysing Results: slider + fraud checkbox (keyed) */
  section[data-testid="stSidebar"] [class*="st-key-sb_topk"] { margin: 0.26rem 0 0 0 !important; }
  /* Negative top offsets Streamlit’s block gap so checkbox sits closer to the slider */
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
  /* Thin separator after Analysing Results; tight gap under checkbox (markdown block margin cleared) */
  section[data-testid="stSidebar"] [data-testid="stElementContainer"] hr.sb-analysing-end-hr {
    margin: 0.04rem 0 0.28rem 0;
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.28);
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
  /* Run pipeline (key=runPipelineMain): full-width, deeper indigo CTA, no icon */
  section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] {
    margin-top: 0.35rem !important;
  }
  section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"] {
    width: 100% !important;
    min-height: 2.7rem !important;
    border: none !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    background: linear-gradient(150deg, #3730a3 0%, #1e1b4b 55%, #0f172a 100%) !important;
    box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.35), 0 4px 24px rgba(30, 27, 75, 0.65) !important;
    transition: transform 0.12s, box-shadow 0.2s, filter 0.2s;
  }
  section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"]:hover:not(:disabled) {
    background: linear-gradient(150deg, #4f46e5 0%, #312e81 50%, #1e1b4b 100%) !important;
    box-shadow: 0 0 0 1px rgba(165, 180, 252, 0.4), 0 6px 30px rgba(49, 46, 129, 0.7) !important;
    transform: translateY(-1px);
  }
  section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"]:active:not(:disabled) {
    transform: translateY(0);
  }
  section[data-testid="stSidebar"] [class*="st-key-runPipelineMain"] [data-baseweb="button"]:disabled {
    background: linear-gradient(150deg, #334155 0%, #1e293b 100%) !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
    transform: none !important;
    opacity: 0.9 !important;
  }
  .pro-sidebar-brand { margin: 0 !important; padding: 0 !important; }
  .pro-sidebar-brand .pro-sb-row {
    display: flex; flex-direction: row; align-items: center; gap: 0.38rem;
  }
  .pro-sidebar-brand .pro-sb-icon {
    flex-shrink: 0; display: flex; align-items: center; justify-content: center;
    width: 3.2rem; height: 3.2rem; padding: 0.15rem;
    position: relative; top: -0.35rem; left: -0.15rem;
    background: rgba(99, 102, 241, 0.14);
    border: 1px solid rgba(129, 140, 248, 0.32);
    border-radius: 12px; box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset;
  }
  .pro-sidebar-brand .pro-sb-icon .pro-sb-icon-svg { width: 2.1rem; height: 2.1rem; display: block; }
  .pro-sidebar-brand .pro-sb-titles { min-width: 0; flex: 1; }
  .pro-sidebar-brand .pro-sb-title {
    font-size: clamp(1.2rem, 0.3rem + 1.1vw, 1.9rem);
    font-weight: 800; color: #f8fafc; margin: 0 !important; line-height: 1.12;
    letter-spacing: -0.02em; hyphens: none; word-wrap: break-word;
  }
  .pro-sidebar-brand .pro-sb-sub {
    font-size: 0.8125rem; color: #c1c9d6; margin: 0.1rem 0 0 0 !important; line-height: 1.28;
    opacity: 0.95; font-weight: 500; letter-spacing: 0.01em;
  }
  /* Main area: sit closer to the app top bar (tabs) */
  div[data-testid="stAppViewContainer"] .main .block-container,
  div[data-testid="stAppViewContainer"] section[data-testid="stMain"] .block-container {
    padding-top: 0.45rem !important; max-width: 1480px;
  }
>>>>>>> Features
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
<<<<<<< HEAD
  .status-pill { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 999px; margin-right: 0.35rem; }
  .ok { background: rgba(34, 197, 94, 0.16); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); }
  .no { background: rgba(248, 113, 113, 0.12); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.3); }
  [data-baseweb="tab"] { font-weight: 600 !important; }
  [data-baseweb="tab"] button { color: #cbd5e1 !important; }
  div[data-testid="stExpander"] details { background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(99, 102, 241, 0.1); border-radius: 10px; }
  .pro-footer { color: #64748b; font-size: 0.75rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(99, 102, 241, 0.12); }
  div[data-testid="stMetricValue"] { color: #e2e8f0; }
=======
  /* st.tabs: pill row (pro) */
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
    color: #94a3b8 !important; font-size: clamp(0.88rem, 0.35rem + 0.95vw, 1.2rem) !important;
    font-weight: 600 !important; letter-spacing: 0.01em !important; line-height: 1.2 !important;
    min-height: 2.75rem !important; min-width: 0 !important; width: 100% !important; max-width: none !important;
    padding: 0.4rem 0.95rem !important; margin: 0 !important;
    text-align: center !important; white-space: nowrap !important; cursor: pointer !important;
    user-select: none !important; -webkit-user-select: none !important; touch-action: manipulation !important;
    -webkit-tap-highlight-color: rgba(0,0,0,0);
    background: rgba(30, 41, 59, 0.75) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important; border-radius: 10px !important; box-shadow: none !important;
    position: relative !important; z-index: 3 !important;
    transition: color 0.1s ease, background 0.1s ease, border-color 0.1s ease, box-shadow 0.1s ease !important;
  }
  section[data-testid="stMain"] [data-baseweb="tab"] button * { user-select: none !important; cursor: inherit !important; }
  section[data-testid="stMain"] [data-baseweb="tab"] button[aria-selected="true"] {
    color: #f1f5f9 !important;
    background: linear-gradient(180deg, rgba(79, 70, 229, 0.42) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
    border-color: rgba(45, 212, 191, 0.5) !important;
    box-shadow: inset 0 -3px 0 0 #2dd4bf, 0 0 0 1px rgba(45, 212, 191, 0.12) !important;
  }
  section[data-testid="stMain"] [data-baseweb="tab"] button:hover:not([aria-selected="true"]) {
    color: #e2e8f0 !important; background: rgba(51, 65, 85, 0.88) !important; border-color: rgba(148, 163, 184, 0.4) !important;
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
  div[data-testid="stExpander"] details { background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(99, 102, 241, 0.1); border-radius: 10px; }
  div[data-testid="stMetricValue"] { color: #e2e8f0; }
  /* Main-area section titles: KPI (indigo) vs charts (teal) vs tables (violet) */
  section[data-testid="stMain"] .main-sec-title {
    font-size: 1rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase;
    margin: 0.15rem 0 0.85rem 0 !important; padding: 0.45rem 0.85rem 0.45rem 0.7rem;
    border-radius: 0 10px 10px 0; line-height: 1.28;
  }
  section[data-testid="stMain"] .main-sec-kpi {
    color: #e0e7ff; border-left: 3px solid #818cf8;
    background: linear-gradient(92deg, rgba(79, 70, 229, 0.38) 0%, rgba(15, 23, 42, 0.5) 100%);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 0 0 1px rgba(129, 140, 248, 0.3);
  }
  section[data-testid="stMain"] .main-sec-chart {
    color: #ccfbf1; border-left: 3px solid #2dd4bf;
    background: linear-gradient(92deg, rgba(15, 118, 110, 0.42) 0%, rgba(15, 23, 42, 0.5) 100%);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 0 0 1px rgba(45, 212, 191, 0.28);
  }
  section[data-testid="stMain"] .main-sec-table {
    color: #c7d2fe; border-left: 3px solid #a5b4fc;
    background: linear-gradient(92deg, rgba(99, 102, 241, 0.35) 0%, rgba(15, 23, 42, 0.5) 100%);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 0 0 1px rgba(165, 180, 252, 0.28);
  }
  /* Extra air between stacked sections (table → PR plots, etc.) */
  section[data-testid="stMain"] .main-sec-title.main-sec-chart,
  section[data-testid="stMain"] .main-sec-title.main-sec-table {
    margin-top: 1.4rem !important;
  }
  /* Model overview: more space between KPI cards and the decile chart title */
  section[data-testid="stMain"] .main-sec-title.main-sec-chart.main-sec-mo-gap {
    margin-top: 2.65rem !important;
  }
  /* Model Performance: hyperparameter panel under the metrics table */
  section[data-testid="stMain"] .main-sec-title.main-sec-hp-gap,
  section[data-testid="stMain"] .main-sec-title.main-sec-chart.main-sec-hp-gap {
    margin-top: 1.35rem !important;
  }
  section[data-testid="stMain"] .hp-section { margin: 0 0 0.35rem 0; }
  section[data-testid="stMain"] .hp-sub {
    font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.11em; color: #94a3b8;
    margin: 0 0 0.6rem 0.1rem;
  }
  section[data-testid="stMain"] .hp-sub-tuned { margin-top: 0.2rem; color: #7dd3fc; }
  section[data-testid="stMain"] .hp-config-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.5rem;
  }
  section[data-testid="stMain"] .hp-cfg-chip {
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.28); border-radius: 10px; padding: 0.5rem 0.7rem 0.55rem 0.7rem;
  }
  section[data-testid="stMain"] .hp-cfg-chip .hpc-k {
    display: block; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 0.2rem;
  }
  section[data-testid="stMain"] .hp-cfg-chip .hpc-v {
    display: block; font-size: 0.9rem; font-weight: 700; color: #f1f5f9; line-height: 1.35; word-break: break-word; white-space: pre-wrap;
  }
  section[data-testid="stMain"] .hp-model-card {
    background: linear-gradient(165deg, rgba(15, 58, 68, 0.45) 0%, rgba(15, 23, 42, 0.92) 100%);
    border: 1px solid rgba(45, 212, 191, 0.28); border-radius: 12px; padding: 0.85rem 1rem 0.95rem 1rem; margin-bottom: 0.25rem;
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset;
    min-height: 24rem; display: flex; flex-direction: column; box-sizing: border-box;
  }
  section[data-testid="stMain"] .hp-tuned-body { flex: 1 1 auto; min-height: 0; }
  section[data-testid="stMain"] .hp-model-title {
    margin: 0 0 0.75rem 0; font-size: 0.95rem; font-weight: 800; color: #5eead4; letter-spacing: 0.04em; border-bottom: 1px solid rgba(45, 212, 191, 0.2);
    padding-bottom: 0.45rem;
  }
  section[data-testid="stMain"] .hp-kv {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 0.4rem 0.8rem;
    padding: 0.4rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.12); font-size: 0.8rem; line-height: 1.35;
  }
  section[data-testid="stMain"] .hp-kv:last-of-type { border-bottom: none; }
  section[data-testid="stMain"] .hp-k { color: #a5b4fc; font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.78rem; }
  section[data-testid="stMain"] .hp-v { color: #e2e8f0; font-weight: 600; text-align: right; max-width: 60%; }
>>>>>>> Features
  .stButton>button { border-radius: 10px; font-weight: 600; }
</style>
"""


<<<<<<< HEAD
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
=======
st.set_page_config(
    page_title="Medicare provider fraud — analytics",
    page_icon="🛡",
>>>>>>> Features
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
<<<<<<< HEAD
        _pro_insight_strip(s)
=======
>>>>>>> Features
        c.render_original_tabs(
            s, m, top_k, fraud_only, style_table=pro_style_table, hist_plot_kwargs=hist_style
        )
    elif approach == "Iterative Ensemble":
        if not c.PATH_SCORED_ITERATIVE.exists():
            st.warning(f"Missing scored file: `{c.PATH_SCORED_ITERATIVE.relative_to(c.ROOT)}`")
            return
        s = c.load_iterative_scored()
        m = c.load_iterative_metrics()
<<<<<<< HEAD
        _pro_insight_strip(s)
=======
>>>>>>> Features
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
<<<<<<< HEAD
    st.markdown("### Command Center")
    st.caption("CMS provider data · OIG LEIE label · rank ensemble")
    st.divider()
=======
    st.markdown(
        """
<div class="pro-sidebar-brand">
  <div class="pro-sb-row">
    <span class="pro-sb-icon" title="Medicare provider fraud &amp; OIG risk screening">
      <svg class="pro-sb-icon-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
        <path fill="rgba(99,102,241,0.1)" stroke="#818cf8" stroke-width="1.2" stroke-linejoin="round" d="M12 1.9l7.1 2.5v4.4c0 4.6-2.4 8.1-6.7 9.3h-.1l-.1-.02C7.3 16.5 4.8 12.6 4.8 8.3V4.2L12 1.9z"/>
        <path fill="none" stroke="#e2e8f0" stroke-width="1.2" stroke-linecap="round" d="M12 8.1v4.4M9.2 10.3h5.6"/>
        <circle cx="17.9" cy="4.7" r="1.8" fill="#f87171"/>
      </svg>
    </span>
    <div class="pro-sb-titles">
      <p class="pro-sb-title">Medicare provider fraud</p>
      <p class="pro-sb-sub">ML risk dashboard · CMS &amp; OIG data</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<p class="pro-sidebar-section-title pro-sst-analysis">Analysis mode</p>',
        unsafe_allow_html=True,
    )
>>>>>>> Features
    approach = st.radio(
        "Analysis mode",
        ["Original Ensemble", "Iterative Ensemble", "Compare Both"],
        index=0,
        help="Switch between the four-model ensemble, iterative bagging, or a metric-by-metric comparison.",
<<<<<<< HEAD
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
=======
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(c.analysing_results_table_title_html(theme="pro"), unsafe_allow_html=True)
    top_k = st.slider(
        "Table rows (top by score)",
        50,
        1000,
        200,
        50,
        label_visibility="collapsed",
        key="sb_topk",
    )
    fraud_only = st.checkbox(
        "Fraud-labeled NPIs only",
        value=False,
        key="sb_fraud",
    )
    st.markdown('<hr class="sb-analysing-end-hr" />', unsafe_allow_html=True)
    st.markdown(
        '<p class="pro-sidebar-section-title pro-sst-download">Download Raw Datasets</p>',
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
        '<p class="pro-sidebar-section-title pro-sst-upload pro-sst-tight">Upload Dataset and Run Pipeline</p>',
        unsafe_allow_html=True,
    )
    (c.ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    c.render_raw_csv_slot(
        upload_label="Upload provider.csv",
        dest=c.PATH_RAW_PROVIDER,
        display_name="provider.csv",
        state_base="p_raw_pro",
    )
    c.render_raw_csv_slot(
        upload_label="Upload exclusion.csv",
        dest=c.PATH_RAW_EXCLUSION,
        display_name="exclusion.csv",
        state_base="p_raw_excl",
    )
    if not c.models_present():
        st.error("Model bundle incomplete — add joblib + CatBoost under `models/`.", icon="⚠️")
    if "p_log" not in st.session_state:
        st.session_state.p_log = ""
    if st.button(
        "Run pipeline",
        key="runPipelineMain",
        type="primary",
        help="Preprocess, label, engineer features, and score (full run can take a long time).",
        disabled=not (
>>>>>>> Features
            c.PATH_RAW_PROVIDER.is_file() and c.PATH_RAW_EXCLUSION.is_file() and c.models_present()
        ),
        width="stretch",
    ):
        with st.status("Running ETL and scoring…", expanded=True) as stu:
<<<<<<< HEAD
            proc = c.run_pipeline_subprocess(c.ROOT, nrows=nrows_smoke)
=======
            proc = c.run_pipeline_subprocess(c.ROOT, nrows=None)
>>>>>>> Features
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
<<<<<<< HEAD
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
=======

# ── Main canvas ─────────────────────────────────────────────────────────
_render_approach_body(approach, top_k, fraud_only)
>>>>>>> Features
