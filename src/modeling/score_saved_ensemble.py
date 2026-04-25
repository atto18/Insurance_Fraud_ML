"""
Score all providers using the saved original ensemble (LightGBM, XGBoost,
CatBoost, GPU logistic regression). Does not retrain.

Expects:
  - data/final/provider_features.csv
  - models/lightgbm_model.joblib, xgboost_model.joblib,
    logistic_regression.joblib, catboost_model.cbm

Run:
  python src/modeling/score_saved_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = (0.05, 0.45, 0.40, 0.10)  # lgb, xgb, cb, lr — same as train_models.py


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(pct=True).to_numpy()


def _lgb_cat_prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype("category")
    return Xc


def _cb_cat_prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy().reset_index(drop=True)
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str)
    return Xc


def load_xy_ids(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    provider_ids = df["provider_id"].copy()
    y = df["label"].astype(int)
    X = df.drop(columns=["provider_id", "label"])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, provider_ids, cat_cols


def make_encoders_full(X: pd.DataFrame, cat_cols: list[str]) -> dict[str, LabelEncoder]:
    """Fit encoders on the full feature matrix (same rows as scoring)."""
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        encoders[col] = le
    return encoders


def encode_for_xgb(X: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    X_enc = X.copy()
    for col, le in encoders.items():
        X_enc[col] = X_enc[col].astype(str).map(
            lambda v, _le=le: int(_le.transform([v])[0]) if v in _le.classes_ else -1
        )
    return X_enc


def load_lr_bundle(model_dir: Path):
    """Joblib was saved when train_models ran as __main__; patch unpickler."""
    import __main__

    from src.modeling.train_models import _GPULogisticRegression

    __main__._GPULogisticRegression = _GPULogisticRegression
    return joblib.load(model_dir / "logistic_regression.joblib")


def score_all(
    features_csv: Path | None = None,
    out_csv: Path | None = None,
    model_dir: Path | None = None,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    root = PROJECT_ROOT
    features_csv = features_csv or root / "data" / "final" / "provider_features.csv"
    out_csv = out_csv or root / "data" / "final" / "scored_providers.csv"
    model_dir = model_dir or root / "models"

    for name, p in [
        ("Features", features_csv),
        ("LightGBM", model_dir / "lightgbm_model.joblib"),
        ("XGBoost", model_dir / "xgboost_model.joblib"),
        ("LogisticRegression", model_dir / "logistic_regression.joblib"),
        ("CatBoost", model_dir / "catboost_model.cbm"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    X, y, provider_ids, cat_cols = load_xy_ids(features_csv)
    encoders = make_encoders_full(X, cat_cols)

    lr_model, scaler = load_lr_bundle(model_dir)
    lgb_model = joblib.load(model_dir / "lightgbm_model.joblib")
    xgb_model = joblib.load(model_dir / "xgboost_model.joblib")
    cb = CatBoostClassifier()
    cb.load_model(str(model_dir / "catboost_model.cbm"))

    all_lgb = lgb_model.predict_proba(_lgb_cat_prep(X, cat_cols))[:, 1]

    X_enc = encode_for_xgb(X, encoders)
    all_xgb = xgb_model.predict_proba(X_enc)[:, 1]
    all_lr = lr_model.predict_proba(scaler.transform(X_enc))[:, 1]

    all_cb = cb.predict_proba(_cb_cat_prep(X, cat_cols))[:, 1]

    w_lgb, w_xgb, w_cb, w_lr = weights
    ensemble = (
        rank_normalize(all_lgb) * w_lgb
        + rank_normalize(all_xgb) * w_xgb
        + rank_normalize(all_cb) * w_cb
        + rank_normalize(all_lr) * w_lr
    )

    scored = pd.DataFrame(
        {
            "provider_id": provider_ids.values,
            "fraud_score": ensemble,
            "lightgbm_score": all_lgb,
            "xgboost_score": all_xgb,
            "catboost_score": all_cb,
            "lr_score": all_lr,
            "label": y.values,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_csv, index=False)
    print(f"Scored {len(scored):,} providers → {out_csv}")
    return scored


if __name__ == "__main__":
    score_all()
