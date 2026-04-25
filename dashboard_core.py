"""
Shared data loading, metrics, charts, and tab renderers for Medicare fraud dashboards.
Imported by ``dashboard.py`` and ``dashboard_pro.py`` (no Streamlit layout at import time).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent

PATH_SCORED_ORIGINAL = ROOT / "data" / "final" / "scored_providers.csv"
PATH_SCORED_ITERATIVE = ROOT / "data" / "final" / "scored_providers_iterative.csv"
PATH_METRICS_ORIGINAL = ROOT / "outputs" / "metrics.json"
PATH_METRICS_ITERATIVE = ROOT / "outputs" / "iterative" / "metrics.json"

PATH_PLOT_PR_ORIGINAL = ROOT / "outputs" / "plots" / "pr_curves.png"
PATH_PLOT_PK_ORIGINAL = ROOT / "outputs" / "plots" / "precision_at_k.png"
PATH_PLOT_SCOREDIST_ORIGINAL = ROOT / "outputs" / "plots" / "score_distribution.png"
PATH_SHAP_BAR = ROOT / "outputs" / "plots" / "shap_lightgbm_bar.png"
PATH_SHAP_BEESWARM = ROOT / "outputs" / "plots" / "shap_lightgbm_beeswarm.png"
PATH_FEAT_IMP = ROOT / "outputs" / "plots" / "feature_importance.png"

PATH_PLOT_PR_ITER = ROOT / "outputs" / "iterative" / "plots" / "pr_curve.png"
PATH_PLOT_PK_ITER = ROOT / "outputs" / "iterative" / "plots" / "precision_at_k.png"
PATH_PLOT_SCOREDIST_ITER = ROOT / "outputs" / "iterative" / "plots" / "score_distribution.png"

PATH_RAW_PROVIDER = ROOT / "data" / "raw" / "provider.csv"
PATH_RAW_EXCLUSION = ROOT / "data" / "raw" / "exclusion.csv"
PATH_PIPELINE_SCRIPT = ROOT / "src" / "pipeline" / "run_from_raw.py"

MODEL_ARTIFACTS = [
    ROOT / "models" / "lightgbm_model.joblib",
    ROOT / "models" / "xgboost_model.joblib",
    ROOT / "models" / "logistic_regression.joblib",
    ROOT / "models" / "catboost_model.cbm",
]


@st.cache_data(show_spinner=False)
def load_original_scored() -> pd.DataFrame:
    return pd.read_csv(PATH_SCORED_ORIGINAL, low_memory=False)


@st.cache_data(show_spinner=False)
def load_iterative_scored() -> pd.DataFrame:
    return pd.read_csv(PATH_SCORED_ITERATIVE, low_memory=False)


@st.cache_data(show_spinner=False)
def load_original_metrics() -> dict[str, Any]:
    if not PATH_METRICS_ORIGINAL.exists():
        return {}
    with open(PATH_METRICS_ORIGINAL, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_iterative_metrics() -> dict[str, Any]:
    if not PATH_METRICS_ITERATIVE.exists():
        return {}
    with open(PATH_METRICS_ITERATIVE, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_file_bytes(rel_path: str) -> bytes:
    p = ROOT / rel_path
    if not p.exists():
        return b""
    return p.read_bytes()


def file_mtime_display(path: Path) -> str:
    if not path.is_file():
        return "—"
    from datetime import datetime

    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def score_distribution_summary(scored: pd.DataFrame) -> dict[str, float]:
    """Extra summary numbers for the professional dashboard."""
    pos = scored.loc[scored["label"] == 1, "fraud_score"]
    neg = scored.loc[scored["label"] == 0, "fraud_score"]
    return {
        "pos_rate_pct": 100.0 * float(scored["label"].mean()),
        "median_pos": float(pos.median()) if len(pos) else float("nan"),
        "median_neg": float(neg.median()) if len(neg) else float("nan"),
        "p99_neg": float(neg.quantile(0.99)) if len(neg) else float("nan"),
    }


def plot_score_deciles(scored: pd.DataFrame, title: str) -> go.Figure:
    """Bar chart of mean label rate by fraud_score decile (monitoring-style)."""
    d = scored[["fraud_score", "label"]].dropna().copy()
    if len(d) < 100:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", text="Not enough rows for decile view",
            showarrow=False, font_color="#94a3b8",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280)
        return fig
    r = d["fraud_score"].rank(method="first")
    try:
        d["decile"] = pd.qcut(r, 10, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        d["decile"] = (pd.cut(r, bins=10, labels=False, include_lowest=True))
    g = d.groupby("decile", observed=True)["label"].agg(["mean", "count"]).reset_index()
    g["mean"] = g["mean"] * 100
    fig = go.Figure(
        go.Bar(
            x=[f"D{int(i)+1}" for i in g["decile"]],
            y=g["mean"],
            marker_color="#6366f1",
            text=[f"n={c}" for c in g["count"]],
            textposition="outside",
            hovertemplate="Decile %{x}<br>Positive rate %{y:.3f}%<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(15,23,42,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font_color="#e2e8f0",
        xaxis_title="Score decile (low → high risk)",
        yaxis_title="Label rate (%)",
        margin=dict(t=48, b=40),
        yaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        showlegend=False,
        height=340,
    )
    return fig


def _metric_row_html(label: str, value: str, accent: str) -> str:
    return f"""<div class="metric-card" style="--accent: {accent};">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>"""


def safe_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), width="stretch", caption=caption)
    else:
        st.warning(f"Missing plot: `{path.relative_to(ROOT)}`")


def top_one_pct_count(n: int) -> int:
    return max(1, int(np.ceil(n * 0.01)))


def ensemble_auroc(metrics: dict[str, Any], name: str) -> float | None:
    rows = metrics.get("test_metrics") or []
    for row in rows:
        if row.get("model") == name:
            return float(row.get("auroc", float("nan")))
    return None


def build_metrics_table(metrics: dict[str, Any]) -> pd.DataFrame | None:
    rows = metrics.get("test_metrics")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    rename_map = {
        "model": "Model",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "p@50": "P@50",
        "p@100": "P@100",
        "p@200": "P@200",
        "p@500": "P@500",
    }
    df = df.rename(columns=rename_map)
    display_names = {
        "LogisticRegression": "LR",
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
        "CatBoost": "CatBoost",
        "Ensemble": "Ensemble",
        "IterativeEnsemble": "Iterative Ensemble",
    }
    if "Model" in df.columns:
        df["Model"] = df["Model"].map(lambda m: display_names.get(m, m))
    cols = [c for c in ["Model", "AUROC", "AUPRC", "P@50", "P@100", "P@200", "P@500"] if c in df.columns]
    return df[cols]


def style_metrics_with_bars(table: pd.DataFrame, *, auroc_color: str = "#89b4fa", auprc_color: str = "#f9e2af") -> Any:
    fmt = {c: "{:.4f}" for c in ["AUROC", "AUPRC", "P@50", "P@100", "P@200", "P@500"] if c in table.columns}
    styler = table.style.format(fmt, na_rep="—")
    if "AUROC" in table.columns:
        styler = styler.bar(subset=["AUROC"], color=auroc_color, vmin=0, vmax=1, align="zero")
    if "AUPRC" in table.columns:
        amax = table["AUPRC"].max()
        vmax_auprc = float(amax) if pd.notna(amax) and float(amax) > 0 else 0.02
        styler = styler.bar(subset=["AUPRC"], color=auprc_color, vmin=0, vmax=vmax_auprc, align="zero")
    return styler


def plot_score_histogram(
    scored: pd.DataFrame,
    title: str,
    *,
    bg: str = "#1e1e2e",
    grid: str = "#313244",
    c_nf: str = "#89dceb",
    c_f: str = "#f38ba8",
) -> go.Figure:
    fraud = scored.loc[scored["label"] == 1, "fraud_score"]
    non_fraud = scored.loc[scored["label"] == 0, "fraud_score"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=non_fraud, name="Non-fraud", opacity=0.55, marker_color=c_nf, nbinsx=80, histnorm="probability density"
        )
    )
    fig.add_trace(
        go.Histogram(
            x=fraud, name="Confirmed fraud", opacity=0.75, marker_color=c_f, nbinsx=40, histnorm="probability density"
        )
    )
    fig.update_layout(
        barmode="overlay",
        title=title,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font_color="#cdd6f4",
        xaxis_title="Fraud score",
        yaxis_title="Density",
        legend=dict(bgcolor="#181825"),
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(gridcolor=grid)
    fig.update_yaxes(gridcolor=grid)
    return fig


def models_present() -> bool:
    return all(p.is_file() for p in MODEL_ARTIFACTS)


def run_pipeline_subprocess(
    project_root: Path,
    *,
    nrows: int | None,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(PATH_PIPELINE_SCRIPT)]
    if nrows is not None:
        cmd.extend(["--nrows", str(int(nrows))])
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def download_button(rel_path: str, help_text: str, *, key_prefix: str = "dl") -> None:
    data = load_file_bytes(rel_path)
    if not data:
        st.caption(f"⚠ File not found: `{rel_path}`")
        return
    name = Path(rel_path).name
    st.download_button(
        label=f"Download {name}",
        data=data,
        file_name=name,
        mime="text/csv",
        help=help_text,
        key=f"{key_prefix}_{rel_path.replace('/', '_')}",
    )


def render_kpi_row(scored: pd.DataFrame, auroc: float | None) -> None:
    total = len(scored)
    confirmed = int(scored["label"].sum())
    flagged = top_one_pct_count(total)
    auroc_s = f"{auroc:.3f}" if auroc is not None and not np.isnan(auroc) else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_metric_row_html("Total providers", f"{total:,}", "#89b4fa"), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_row_html("Confirmed fraudsters", f"{confirmed:,}", "#f38ba8"), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_row_html("High-risk flagged (top 1%)", f"{flagged:,}", "#fab387"), unsafe_allow_html=True)
    with c4:
        st.markdown(_metric_row_html("AUROC (held-out test)", auroc_s, "#a6e3a1"), unsafe_allow_html=True)


def render_original_tabs(
    scored: pd.DataFrame,
    metrics: dict[str, Any],
    top_k: int,
    fraud_only: bool,
    *,
    style_table: Any = None,
    hist_plot_kwargs: dict[str, Any] | None = None,
) -> None:
    style_table = style_table or (lambda t: style_metrics_with_bars(t))
    hist_plot_kwargs = hist_plot_kwargs or {}
    auroc = ensemble_auroc(metrics, "Ensemble")
    t_overview, t_perf, t_top, t_dist, t_shap = st.tabs(
        ["Overview KPIs", "Model Performance", "Top Suspicious Providers", "Score Distribution", "SHAP & Feature Importance"]
    )

    with t_overview:
        render_kpi_row(scored, auroc)
        st.markdown(
            "Rank-normalized weighted ensemble of **Logistic Regression**, **LightGBM**, **XGBoost**, and **CatBoost** "
            "(Optuna-tuned). Trained on ~1.26M Medicare providers with **187** OIG-confirmed fraud labels — extreme class imbalance."
        )

    with t_perf:
        st.subheader("Test-set metrics")
        table = build_metrics_table(metrics)
        if table is not None:
            st.dataframe(style_table(table), width="stretch", hide_index=True)
        else:
            st.warning("`outputs/metrics.json` not found or empty.")

        c1, c2 = st.columns(2)
        with c1:
            safe_image(PATH_PLOT_PR_ORIGINAL, "PR curves — base models + ensemble")
        with c2:
            safe_image(PATH_PLOT_PK_ORIGINAL, "Precision@K — original models")

    with t_top:
        st.caption("Sort columns by clicking headers. Confirmed fraud rows are highlighted.")
        df = scored.sort_values("fraud_score", ascending=False)
        if fraud_only:
            df = df[df["label"] == 1]
        df = df.head(top_k).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

        show = df[
            [
                "rank",
                "provider_id",
                "fraud_score",
                "lightgbm_score",
                "xgboost_score",
                "catboost_score",
                "lr_score",
                "label",
            ]
        ].copy()
        show = show.rename(
            columns={
                "lightgbm_score": "LGB",
                "xgboost_score": "XGB",
                "catboost_score": "CatBoost",
                "lr_score": "LR",
            }
        )
        for c in ["fraud_score", "LGB", "XGB", "CatBoost", "LR"]:
            if c in show.columns:
                show[c] = show[c].astype(float).round(6)
        show["label"] = show["label"].astype(int)

        def highlight(row: pd.Series) -> list[str]:
            lbl = int(df.loc[row.name, "label"])
            if lbl == 1:
                return ["background-color: #4c0f0f; color: #fecaca"] * len(row)
            return [""] * len(row)

        st.dataframe(show.style.apply(highlight, axis=1), width="stretch", height=520)

    with t_dist:
        c1, c2 = st.columns([1, 1])
        with c1:
            safe_image(PATH_PLOT_SCOREDIST_ORIGINAL, "Static score distribution (original ensemble)")
        with c2:
            fig = plot_score_histogram(scored, "Interactive: fraud vs non-fraud (original scores)", **hist_plot_kwargs)
            st.plotly_chart(fig, width="stretch")

    with t_shap:
        c1, c2 = st.columns(2)
        with c1:
            safe_image(PATH_SHAP_BAR, "SHAP bar — LightGBM")
        with c2:
            safe_image(PATH_SHAP_BEESWARM, "SHAP beeswarm — LightGBM")
        safe_image(PATH_FEAT_IMP, "Permutation / model feature importance")


def render_iterative_tabs(
    scored: pd.DataFrame,
    metrics: dict[str, Any],
    top_k: int,
    fraud_only: bool,
    *,
    style_table: Any = None,
    hist_plot_kwargs: dict[str, Any] | None = None,
) -> None:
    style_table = style_table or (lambda t: style_metrics_with_bars(t))
    hist_plot_kwargs = hist_plot_kwargs or {}
    auroc = ensemble_auroc(metrics, "IterativeEnsemble")
    st.markdown(
        """
In each of **100** iterations, we sample **200** negatives plus **all 187** positives, train **LightGBM** and **XGBoost**
on this balanced batch, then **average** predictions across iterations. Hyperparameters are tuned with **Optuna** on a
held-out validation slice, and metrics below are on the same held-out **test** set as the original pipeline.
"""
    )
    cfg = metrics.get("config") or {}
    if cfg:
        st.info(
            f"**Config:** `n_iterations={cfg.get('n_iterations')}`, `n_negatives_per_iter={cfg.get('n_negatives_per_iter')}`, "
            f"`base_learners={cfg.get('base_learners')}`"
        )
    best = metrics.get("best_params") or {}
    if best:
        with st.expander("Best Optuna parameters (LightGBM & XGBoost)"):
            st.json(best)

    t_overview, t_perf, t_top, t_dist, t_shap = st.tabs(
        ["Overview KPIs", "Model Performance", "Top Suspicious Providers", "Score Distribution", "SHAP & Feature Importance"]
    )

    with t_overview:
        render_kpi_row(scored, auroc)

    with t_perf:
        st.subheader("Test-set metrics")
        table = build_metrics_table(metrics)
        if table is not None:
            st.dataframe(style_table(table), width="stretch", hide_index=True)
        else:
            st.warning("`outputs/iterative/metrics.json` not found or empty.")
        c1, c2 = st.columns(2)
        with c1:
            safe_image(PATH_PLOT_PR_ITER, "PR curve — iterative ensemble")
        with c2:
            safe_image(PATH_PLOT_PK_ITER, "Precision@K — iterative")

    with t_top:
        st.caption("Sort columns by clicking headers. Confirmed fraud rows are highlighted.")
        df = scored.sort_values("fraud_score", ascending=False)
        if fraud_only:
            df = df[df["label"] == 1]
        df = df.head(top_k).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        show = df[["rank", "provider_id", "fraud_score", "label"]].copy()
        show["fraud_score"] = show["fraud_score"].astype(float).round(6)
        show["label"] = show["label"].astype(int)

        def highlight_iter(row: pd.Series) -> list[str]:
            lbl = int(df.loc[row.name, "label"])
            if lbl == 1:
                return ["background-color: #4c0f0f; color: #fecaca"] * len(row)
            return [""] * len(row)

        st.dataframe(show.style.apply(highlight_iter, axis=1), width="stretch", height=520)

    with t_dist:
        c1, c2 = st.columns(2)
        with c1:
            safe_image(PATH_PLOT_SCOREDIST_ITER, "Static score distribution (iterative)")
        with c2:
            fig = plot_score_histogram(scored, "Interactive: fraud vs non-fraud (iterative scores)", **hist_plot_kwargs)
            st.plotly_chart(fig, width="stretch")

    with t_shap:
        st.caption(
            "Iterative training did not emit separate SHAP figures. The plots below are from the **original** LightGBM-focused "
            "interpretability run on the same engineered feature space."
        )
        c1, c2 = st.columns(2)
        with c1:
            safe_image(PATH_SHAP_BAR, "SHAP bar — LightGBM (original pipeline)")
        with c2:
            safe_image(PATH_SHAP_BEESWARM, "SHAP beeswarm — LightGBM (original pipeline)")
        safe_image(PATH_FEAT_IMP, "Feature importance (original pipeline)")


def render_compare_view(m_orig: dict[str, Any], m_iter: dict[str, Any]) -> None:
    def row_for(model_name: str, m: dict[str, Any]) -> dict[str, float] | None:
        for r in m.get("test_metrics", []) or []:
            if r.get("model") == model_name:
                return r
        return None

    r_o = row_for("Ensemble", m_orig)
    r_i = row_for("IterativeEnsemble", m_iter)
    if not r_o or not r_i:
        st.warning("Could not load ensemble rows from one or both metrics files.")
        return

    metrics_list = [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("P@50", "p@50"),
        ("P@100", "p@100"),
        ("P@200", "p@200"),
        ("P@500", "p@500"),
    ]
    rows_out = []
    for label, key in metrics_list:
        v_o = float(r_o.get(key, float("nan")))
        v_i = float(r_i.get(key, float("nan")))
        winner = "—"
        if not np.isnan(v_o) and not np.isnan(v_i):
            if v_o > v_i:
                winner = "Original Ensemble ✓"
            elif v_i > v_o:
                winner = "Iterative Ensemble ✓"
            else:
                winner = "Tie"
        rows_out.append(
            {
                "Metric": label,
                "Original Ensemble": v_o,
                "Iterative Ensemble": v_i,
                "Winner": winner,
            }
        )
    cmp_df = pd.DataFrame(rows_out)
    fmt = {c: "{:.4f}" for c in ["Original Ensemble", "Iterative Ensemble"]}

    def color_winner(s: pd.Series) -> list[str]:
        styles: list[str] = []
        for w in s:
            if isinstance(w, str) and w.endswith("✓"):
                styles.append("color: #a6e3a1; font-weight: 600")
            else:
                styles.append("")
        return styles

    st.subheader("Metric comparison (test set)")
    st.dataframe(
        cmp_df.style.format(fmt, na_rep="—").apply(color_winner, subset=["Winner"]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("PR curves")
    c1, c2 = st.columns(2)
    with c1:
        safe_image(PATH_PLOT_PR_ORIGINAL, "Original — all models + ensemble")
    with c2:
        safe_image(PATH_PLOT_PR_ITER, "Iterative ensemble")

    st.subheader("Score distributions")
    c1, c2 = st.columns(2)
    with c1:
        safe_image(PATH_PLOT_SCOREDIST_ORIGINAL, "Original ensemble")
    with c2:
        safe_image(PATH_PLOT_SCOREDIST_ITER, "Iterative ensemble")

    st.subheader("Precision@K")
    c1, c2 = st.columns(2)
    with c1:
        safe_image(PATH_PLOT_PK_ORIGINAL, "Original models")
    with c2:
        safe_image(PATH_PLOT_PK_ITER, "Iterative")

    st.markdown("### Conclusion")
    st.info(
        "The iterative approach achieves better AUROC (0.812 vs 0.790), meaning better overall ranking quality. "
        "The original ensemble achieves better AUPRC (0.0105 vs 0.0013), meaning better precision at high-confidence "
        "thresholds. With only 37 confirmed fraudsters in the test set, both metrics carry high statistical noise."
    )
