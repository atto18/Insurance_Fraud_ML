"""
Shared data loading, metrics, charts, and tab renderers for Medicare fraud dashboards.
Imported by ``dashboard.py`` and ``dashboard_pro.py`` (no Streamlit layout at import time).
"""

from __future__ import annotations

import html
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

PATH_PLOT_PR_ITER = ROOT / "outputs" / "iterative" / "plots" / "pr_curve.png"
PATH_PLOT_PK_ITER = ROOT / "outputs" / "iterative" / "plots" / "precision_at_k.png"
PATH_PLOT_SCOREDIST_ITER = ROOT / "outputs" / "iterative" / "plots" / "score_distribution.png"

PATH_RAW_PROVIDER = ROOT / "data" / "raw" / "provider.csv"
PATH_RAW_EXCLUSION = ROOT / "data" / "raw" / "exclusion.csv"
PATH_PIPELINE_SCRIPT = ROOT / "src" / "pipeline" / "run_from_raw.py"

# Direct public file URLs (save into data/raw/ as `provider.csv` and `exclusion.csv`).
# Update if CMS or OIG republish the files and these URLs 404.
URL_CMS_RAW_PROVIDER = "https://data.cms.gov/sites/default/files/2025-04/22edfd1e-d17a-4478-ad6b-92cac2a5a3c4/MUP_PHY_R25_P05_V20_D23_Prov.csv"
URL_OIG_RAW_EXCLUSION = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"


def analysing_results_table_title_html(*, theme: str) -> str:
    """Teal sidebar bar for “Analysing Results Table”."""
    if theme == "pro":
        return (
            '<p class="pro-sidebar-section-title pro-sst-table pro-sst-after-widget pro-sst-art-tight">'
            "Analysing Results Table"
            "</p>"
        )
    if theme == "dash":
        return (
            '<p class="dash-sidebar-section-title dash-sst-table dash-sst-after-widget dash-sst-art-tight">'
            "Analysing Results Table"
            "</p>"
        )
    raise ValueError("theme must be 'pro' or 'dash'")


def render_raw_csv_slot(
    *,
    upload_label: str,
    dest: Path,
    display_name: str,
    state_base: str,
) -> None:
    """
    When ``dest`` does not exist, show a file uploader. When it does, show a small card
    with ``display_name`` and a simple ✕ control (data are read/written on ``dest``).
    """
    n_k = f"{state_base}_n"
    if n_k not in st.session_state:
        st.session_state[n_k] = 0
    safe_name = html.escape(display_name, quote=True)
    has = dest.is_file()
    if not has:
        up = st.file_uploader(
            upload_label,
            type=["csv"],
            accept_multiple_files=False,
            key=f"{state_base}_w_{st.session_state[n_k]}",
        )
        if up is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(up.getvalue())
            st.session_state[n_k] = int(st.session_state[n_k]) + 1
            st.rerun()
    else:
        c_chip, c_x = st.columns([10, 1], vertical_alignment="center", gap="xsmall")
        with c_chip:
            st.markdown(
                f"""
<div style="width:100%;min-width:0;box-sizing:border-box;overflow:hidden;">
<div style="display:inline-flex;align-items:center;min-height:2.35rem;box-sizing:border-box;
  max-width:100%;
  padding:0.42rem 0.65rem 0.42rem 0.75rem;
  background:linear-gradient(165deg, rgba(30,41,59,0.65) 0%, rgba(15,23,42,0.85) 100%);
  border:1px solid rgba(99,102,241,0.28);border-radius:12px;box-shadow:0 1px 0 rgba(255,255,255,0.04) inset;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
  <span style="font-family:ui-monospace,Consolas,monospace;font-size:0.9rem;font-weight:600;
    color:#f8fafc;letter-spacing:0.01em;min-width:0;">{safe_name}</span>
</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with c_x:
            if st.button(
                " ",
                key=f"{state_base}_rm",
                type="tertiary",
                icon=":material/close:",
                help="Remove file (delete on disk) — you can upload again after.",
                use_container_width=False,
            ):
                dest.unlink(missing_ok=True)
                st.session_state[n_k] = int(st.session_state[n_k]) + 1
                st.rerun()


def download_url_to_path(url: str, dest: Path) -> str:
    """Stream ``url`` to ``dest`` (8MB chunks). Returns an empty string on success, else an error line."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ML_Projet/1.0; research/academic use)"},
        )
        with urlopen(req, timeout=900) as r:
            with open(dest, "wb") as f:
                while True:
                    block = r.read(8 * 1024 * 1024)
                    if not block:
                        break
                    f.write(block)
    except (HTTPError, URLError, OSError, TimeoutError) as e:
        return f"{type(e).__name__}: {e}"
    return ""

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


def _metric_row_html(
    label: str,
    value: str,
    accent: str,
    *,
    second_row: bool = False,
) -> str:
    # Second row: extra top margin so the gap between the two KPI lines is clear and consistent
    margin = "1rem 0.6rem 0.35rem 0.6rem" if second_row else "0.35rem 0.6rem"
    return f"""<div class="metric-card" style="--accent: {accent}; margin: {margin}; box-sizing: border-box;">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>"""


def safe_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), width="stretch", caption=caption)
    else:
        st.warning(f"Missing plot: `{path.relative_to(ROOT)}`")


_COMPARE_CANVAS_W = 900
_COMPARE_CANVAS_H = 500


def safe_image_compare(path: Path, caption: str | None = None) -> None:
    """
    Side-by-side PNGs: same pixel canvas (so pairs align). Letterbox is **transparent** (no
    solid band behind the figure). Use in Compare view and any two-column plot rows.
    Falls back to ``st.image`` if Pillow is unavailable or the file cannot be read.
    """
    if not path.exists():
        st.warning(f"Missing plot: `{path.relative_to(ROOT)}`")
        return
    try:
        from PIL import Image
    except ImportError:
        st.image(str(path), caption=caption, use_container_width=True)
        return
    try:
        w0, h0 = _COMPARE_CANVAS_W, _COMPARE_CANVAS_H
        im = Image.open(path)
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
        elif im.mode == "RGBA":
            pass
        elif im.mode == "RGB":
            im = im.convert("RGBA")
        else:
            im = im.convert("RGBA")
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        im.thumbnail((w0, h0), resample)
        canvas = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
        x = (w0 - im.width) // 2
        y = (h0 - im.height) // 2
        canvas.paste(im, (x, y), im)
        st.image(canvas, caption=caption, use_container_width=True)
    except Exception:
        st.image(str(path), caption=caption, use_container_width=True)


def ensemble_auroc(metrics: dict[str, Any], name: str) -> float | None:
    rows = metrics.get("test_metrics") or []
    for row in rows:
        if row.get("model") == name:
            return float(row.get("auroc", float("nan")))
    return None


def ensemble_auprc(metrics: dict[str, Any], name: str) -> float | None:
    rows = metrics.get("test_metrics") or []
    for row in rows:
        if row.get("model") == name:
            return float(row.get("auprc", float("nan")))
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


def _hp_format_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.6g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, tuple, dict)):
        return html.escape(
            json.dumps(v, ensure_ascii=False, indent=2)
            if isinstance(v, dict)
            else json.dumps(v, ensure_ascii=False)
        )
    return html.escape(str(v))


def _hp_config_chips_html(cfg: dict[str, Any]) -> str:
    out: list[str] = []
    for key in sorted(cfg.keys(), key=str):
        v = cfg[key]
        label = key.replace("_", " ")
        out.append(
            f'<div class="hp-cfg-chip"><span class="hpc-k">{html.escape(str(label))}</span>'
            f'<span class="hpc-v">{_hp_format_value(v)}</span></div>'
        )
    return "\n".join(out)


def _jsonify_sklearn_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if v == v else None
    if type(v).__module__ == "numpy" or "numpy" in str(type(v)):
        out = v.item() if hasattr(v, "item") else float(v)  # type: ignore[union-attr]
        return out
    if isinstance(v, dict):
        return {str(a): _jsonify_sklearn_value(b) for a, b in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonify_sklearn_value(x) for x in v]
    if isinstance(v, (float, int)):
        return v
    return str(v)[:200]


def _get_params_sanitized(m: Any, *, max_keys: int = 48) -> dict[str, Any]:
    try:
        raw = m.get_params(deep=False)
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for k in sorted(raw.keys()):
        if len(out) >= max_keys:
            break
        v = raw[k]
        if v is None or callable(v):
            continue
        s = str(type(v))
        if "Callback" in s or "callback" in k.lower():
            continue
        j = _jsonify_sklearn_value(v)
        if j is not None and not (isinstance(j, str) and len(j) > 500):
            out[k] = j
    return out


def original_best_params_from_artifacts(*, include_lightgbm: bool = True) -> dict[str, Any] | None:
    """
    When ``outputs/metrics.json`` has no ``best_params`` (older runs), rebuild a
    display dict from ``models/*`` the same way the app would after training.
    For Original Ensemble, LightGBM can be omitted from the panel (set ``include_lightgbm`` False).
    """
    out: dict[str, Any] = {}
    try:
        import joblib
    except ImportError:
        joblib = None

    if joblib is not None and include_lightgbm:
        lp = ROOT / "models" / "lightgbm_model.joblib"
        if lp.is_file():
            try:
                m = joblib.load(lp)
                p = _get_params_sanitized(m)
                if p:
                    out["lightgbm"] = p
            except Exception:
                pass
    if joblib is not None:
        xp = ROOT / "models" / "xgboost_model.joblib"
        if xp.is_file():
            try:
                m = joblib.load(xp)
                p = _get_params_sanitized(m)
                if p:
                    out["xgboost"] = p
            except Exception:
                pass

    cbp = ROOT / "models" / "catboost_model.cbm"
    if cbp.is_file():
        try:
            from catboost import CatBoostClassifier

            m = CatBoostClassifier()
            m.load_model(str(cbp))
            p = _get_params_sanitized(m)
            if p:
                out["catboost"] = p
        except Exception:
            pass

    return out if out else None


def _hp_model_kv_html(params: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in sorted(params.keys(), key=str):
        v = params[key]
        val = _hp_format_value(v)
        lines.append(
            f'<div class="hp-kv"><span class="hp-k">{html.escape(key)}</span>'
            f'<span class="hp-v">{val}</span></div>'
        )
    return "\n".join(lines)


def render_hyperparams_panel(metrics: dict[str, Any], *, mode: str) -> None:
    """
    Renders the hyperparameter / training-settings block in **Model Performance**
    (below the metrics table, above PR plots). ``mode`` is ``"original"`` or ``"iterative"``.
    """
    cfg = metrics.get("config")
    raw_bp = metrics.get("best_params")
    bp: dict[str, Any] | None = raw_bp if isinstance(raw_bp, dict) else None
    from_artifacts = False
    had_file_bp = bool(bp) and any(
        isinstance(v, dict) and v for v in (bp or {}).values()
    )
    if mode == "original" and not had_file_bp:
        ob = original_best_params_from_artifacts(include_lightgbm=False)
        if ob:
            bp = ob
            from_artifacts = True

    st.markdown(
        main_section_title_html(
            "Hyperparameters & training settings", "chart", extra_class="main-sec-hp-gap"
        ),
        unsafe_allow_html=True,
    )
    if from_artifacts:
        st.caption(
            "Showing parameters from saved model files. Re-run `src/modeling/train_models.py` to write "
            "Optuna `best_params` and `config` into `outputs/metrics.json`."
        )
    if not cfg and not bp:
        if mode == "iterative":
            st.caption(
                "No `config` or `best_params` in `outputs/iterative/metrics.json` — re-run iterative training."
            )
        else:
            st.caption(
                "No hyperparameters in `outputs/metrics.json` and no readable files under `models/`. "
                "Train the original pipeline to generate metrics and model artifacts."
            )
        return
    if cfg and isinstance(cfg, dict):
        st.markdown(
            '<div class="hp-section">'
            '<p class="hp-sub">Run configuration</p>'
            f'<div class="hp-config-grid">{_hp_config_chips_html(cfg)}</div></div>',
            unsafe_allow_html=True,
        )
    if bp and isinstance(bp, dict):
        names = [k for k, v in bp.items() if isinstance(v, dict) and v]
        if mode == "original":
            names = [k for k in names if k != "lightgbm"]
        if not names:
            if not any(isinstance(v, dict) and v for v in bp.values()):
                st.caption("`best_params` is present but has no per-model dictionaries.")
        else:
            st.markdown(
                '<p class="hp-sub hp-sub-tuned">Tuned model parameters (Optuna)</p>',
                unsafe_allow_html=True,
            )
            order = (
                ("xgboost", "catboost", "lightgbm")
                if mode == "iterative"
                else ("xgboost", "catboost")
            )
            names_o = [x for x in order if x in names]
            names_o.extend([x for x in names if x not in names_o])
            for i in range(0, len(names_o), 2):
                pair = names_o[i : i + 2]
                row_cls = "hp-tuned-row" if len(pair) == 2 else "hp-tuned-row hp-tuned-row-single"
                inner: list[str] = []
                for mname in pair:
                    title = mname.replace("_", " ").title()
                    inner.append(
                        f'<div class="hp-model-card">'
                        f'<p class="hp-model-title">{html.escape(title)}</p>'
                        f'<div class="hp-tuned-body">{_hp_model_kv_html(bp[mname])}</div></div>'
                    )
                st.markdown(
                    f'<div class="{row_cls}">{"".join(inner)}</div>',
                    unsafe_allow_html=True,
                )


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


def main_section_title_html(
    text: str, variant: str = "kpi", *, extra_class: str = ""
) -> str:
    """
    Distinguishes KPI blocks, chart blocks, and table blocks in the main area
    (styled bars; add matching CSS in ``dashboard`` / ``dashboard_pro``).
    ``extra_class`` is appended for layout tweaks (e.g. larger gap after the KPI block on Model overview).
    """
    t = html.escape(text)
    cls = {
        "kpi": "main-sec-kpi",
        "chart": "main-sec-chart",
        "table": "main-sec-table",
    }.get(variant, "main-sec-kpi")
    x = f" {extra_class}" if extra_class else ""
    return f'<p class="main-sec-title {cls}{x}">{t}</p>'


def render_kpi_row(
    scored: pd.DataFrame,
    auroc: float | None,
    auprc: float | None,
) -> None:
    """Two rows: dataset scale (total, confirmed), imbalance, then median fraud score and hold-out AUROC / AUPRC."""
    total = len(scored)
    confirmed = int(scored["label"].sum()) if "label" in scored.columns else 0

    if len(scored) and "label" in scored.columns:
        pos_rate = 100.0 * float(scored["label"].mean())
    else:
        pos_rate = 0.0
    if "fraud_score" in scored.columns and "label" in scored.columns:
        pos = scored.loc[scored["label"] == 1, "fraud_score"]
        med = float(pos.median()) if len(pos) else float("nan")
    else:
        med = float("nan")
    med_s = f"{med:.4f}" if med == med and not np.isnan(med) else "—"
    auroc_s = f"{auroc:.3f}" if auroc is not None and not np.isnan(auroc) else "—"
    auprc_s = f"{auprc:.6f}" if auprc is not None and not np.isnan(auprc) else "—"

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            _metric_row_html("Total providers", f"{total:,}", "#7dcfff"), unsafe_allow_html=True
        )
    with a2:
        st.markdown(
            _metric_row_html("Confirmed fraudsters", f"{confirmed:,}", "#f38ba8"), unsafe_allow_html=True
        )
    with a3:
        st.markdown(
            _metric_row_html("Imbalance (positive %)", f"{pos_rate:.4f}%", "#89b4fa"), unsafe_allow_html=True
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(
            _metric_row_html("Median score (fraud)", med_s, "#cba6f7", second_row=True), unsafe_allow_html=True
        )
    with b2:
        st.markdown(
            _metric_row_html("AUROC (held-out test)", auroc_s, "#a6e3a1", second_row=True), unsafe_allow_html=True
        )
    with b3:
        st.markdown(
            _metric_row_html("AUPRC (held-out test)", auprc_s, "#f9e2af", second_row=True), unsafe_allow_html=True
        )


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
    auprc = ensemble_auprc(metrics, "Ensemble")
    t_overview, t_perf, t_top, t_dist = st.tabs(
        [
            "Model overview",
            "Model Performance",
            "Top suspicious providers",
            "Score distribution",
        ],
        key="dash_model_tabs_original",
    )

    with t_overview:
        st.markdown(
            main_section_title_html("Key performance indicators", "kpi"), unsafe_allow_html=True
        )
        render_kpi_row(scored, auroc, auprc)
        st.markdown(
            main_section_title_html(
                "Label rate by model score (deciles)", "chart", extra_class="main-sec-mo-gap"
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_score_deciles(scored, "Label rate by model score (deciles)"), width="stretch"
        )

    with t_perf:
        st.markdown(main_section_title_html("Test-set metrics", "table"), unsafe_allow_html=True)
        table = build_metrics_table(metrics)
        if table is not None:
            st.dataframe(style_table(table), width="stretch", hide_index=True)
        else:
            st.warning("`outputs/metrics.json` not found or empty.")
        render_hyperparams_panel(metrics, mode="original")
        st.markdown(
            main_section_title_html("PR curves & precision@K (training outputs)", "chart"),
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            safe_image_compare(PATH_PLOT_PR_ORIGINAL, "PR curves — base models + ensemble")
        with c2:
            safe_image_compare(PATH_PLOT_PK_ORIGINAL, "Precision@K — original models")

    with t_top:
        st.markdown(
            main_section_title_html("Top suspicious providers", "table"), unsafe_allow_html=True
        )
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
        st.markdown(main_section_title_html("Score distribution", "chart"), unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.caption("Static figure (from training run)")
            safe_image(PATH_PLOT_SCOREDIST_ORIGINAL, "Static score distribution (original ensemble)")
        with c2:
            st.caption("Interactive (from loaded scores)")
            fig = plot_score_histogram(scored, "Interactive: fraud vs non-fraud (original scores)", **hist_plot_kwargs)
            st.plotly_chart(fig, width="stretch")


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
    auprc = ensemble_auprc(metrics, "IterativeEnsemble")

    t_overview, t_perf, t_top, t_dist = st.tabs(
        [
            "Model overview",
            "Model Performance",
            "Top suspicious providers",
            "Score distribution",
        ],
        key="dash_model_tabs_iterative",
    )

    with t_overview:
        st.markdown(
            main_section_title_html("Key performance indicators", "kpi"), unsafe_allow_html=True
        )
        render_kpi_row(scored, auroc, auprc)
        st.markdown(
            main_section_title_html(
                "Label rate by model score (deciles)", "chart", extra_class="main-sec-mo-gap"
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_score_deciles(scored, "Label rate by model score (deciles) — iterative"), width="stretch"
        )

    with t_perf:
        st.markdown(main_section_title_html("Test-set metrics", "table"), unsafe_allow_html=True)
        table = build_metrics_table(metrics)
        if table is not None:
            st.dataframe(style_table(table), width="stretch", hide_index=True)
        else:
            st.warning("`outputs/iterative/metrics.json` not found or empty.")
        render_hyperparams_panel(metrics, mode="iterative")
        st.markdown(
            main_section_title_html("PR curve & precision@K (training outputs)", "chart"),
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            safe_image_compare(PATH_PLOT_PR_ITER, "PR curve — iterative ensemble")
        with c2:
            safe_image_compare(PATH_PLOT_PK_ITER, "Precision@K — iterative")

    with t_top:
        st.markdown(
            main_section_title_html("Top suspicious providers", "table"), unsafe_allow_html=True
        )
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
        st.markdown(main_section_title_html("Score distribution", "chart"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Static figure (from training run)")
            safe_image(PATH_PLOT_SCOREDIST_ITER, "Static score distribution (iterative)")
        with c2:
            st.caption("Interactive (from loaded scores)")
            fig = plot_score_histogram(scored, "Interactive: fraud vs non-fraud (iterative scores)", **hist_plot_kwargs)
            st.plotly_chart(fig, width="stretch")


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

    st.markdown(main_section_title_html("Metric comparison (test set)", "table"), unsafe_allow_html=True)
    st.dataframe(
        cmp_df.style.format(fmt, na_rep="—").apply(color_winner, subset=["Winner"]),
        width="stretch",
        hide_index=True,
    )

    st.markdown(main_section_title_html("PR curves", "chart"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        safe_image_compare(PATH_PLOT_PR_ORIGINAL, "Original — all models + ensemble")
    with c2:
        safe_image_compare(PATH_PLOT_PR_ITER, "Iterative ensemble")

    st.markdown(main_section_title_html("Score distributions", "chart"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        safe_image_compare(PATH_PLOT_SCOREDIST_ORIGINAL, "Original ensemble")
    with c2:
        safe_image_compare(PATH_PLOT_SCOREDIST_ITER, "Iterative ensemble")

    st.markdown(main_section_title_html("Precision@K", "chart"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        safe_image_compare(PATH_PLOT_PK_ORIGINAL, "Original models")
    with c2:
        safe_image_compare(PATH_PLOT_PK_ITER, "Iterative")
