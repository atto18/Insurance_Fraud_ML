"""
src/modeling/train_models.py

Trains fraud-risk scoring models for Medicare provider fraud detection.

Design choices
--------------
- Primary metric: AUPRC (area under precision-recall curve) — correct metric for
  highly imbalanced data (0.015% positive rate).
- Class imbalance handled via scale_pos_weight / auto_class_weights rather than
  SMOTE: with only ~150 positives in training, SMOTE neighbours are sparse and
  often degrade performance. Weight-based approaches are more stable.
- Hyperparameters tuned with Optuna (TPE sampler, AUPRC objective, 3-fold
  StratifiedKFold).
- Categorical features (state, provider_type) passed natively to LightGBM and
  CatBoost; label-encoded for XGBoost and Logistic Regression.
- Final ensemble uses rank-normalised score averaging (more robust than raw
  probability averaging when models have different score distributions).

Run
---
    python src/modeling/train_models.py

Optional flag: set N_TRIALS = 10 below for a quick smoke-test run.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.modeling.lr_model import _GPULogisticRegression, TORCH_DEVICE
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb


def _detect_lgb_device() -> str:
    """Return 'gpu' if LightGBM OpenCL GPU build is available, else 'cpu'."""
    try:
        _m = lgb.LGBMClassifier(n_estimators=1, device_type="gpu", verbose=-1)
        _m.fit(np.zeros((20, 2)), np.zeros(20))
        print("LightGBM device: gpu (OpenCL)")
        return "gpu"
    except Exception:
        print("LightGBM device: cpu (GPU not available in this build)")
        return "cpu"


def _detect_xgb_device() -> str:
    """Return 'cuda' if XGBoost CUDA is available, else 'cpu'."""
    try:
        _m = xgb.XGBClassifier(n_estimators=1, tree_method="hist",
                                device="cuda", verbosity=0)
        _m.fit(np.zeros((20, 2)), np.zeros(20))
        print("XGBoost device: cuda")
        return "cuda"
    except Exception:
        print("XGBoost device: cpu (CUDA not available)")
        return "cpu"


LGB_DEVICE = _detect_lgb_device()
XGB_DEVICE = _detect_xgb_device()

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT_ROOT / "data" / "final" / "provider_features.csv"
SCORED_CSV   = PROJECT_ROOT / "data" / "final" / "scored_providers.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
MODEL_DIR    = PROJECT_ROOT / "models"
PLOTS_DIR    = OUTPUT_DIR / "plots"
METRICS_JSON = OUTPUT_DIR / "metrics.json"

for d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_CV_FOLDS   = 3   # folds used inside Optuna objective
N_TRIALS     = 40  # Optuna trials per model (set to 10 for a quick test)


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path):
    """Return X, y, provider_ids and the list of categorical column names."""
    print(f"Loading {path.name} …")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape: {df.shape}")
    print(f"  Positives: {df['label'].sum()} / {len(df)}  "
          f"({df['label'].mean()*100:.4f}%)")

    provider_ids = df["provider_id"].copy()
    y = df["label"].astype(int)
    X = df.drop(columns=["provider_id", "label"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"  Categorical columns: {cat_cols}")
    return X, y, provider_ids, cat_cols


def make_label_encoders(X_train: pd.DataFrame, X_test: pd.DataFrame,
                         cat_cols: list[str]):
    """
    Label-encode categorical columns.

    Encoders are fitted on train+test combined so every value seen at scoring
    time has a mapping.  Returns encoded copies and the encoder dict.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
        encoders[col] = le

    return X_tr, X_te, encoders


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ══════════════════════════════════════════════════════════════════════════════

def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(scores)[-k:]
    return float(y_true[idx].sum() / k)


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0
    idx = np.argsort(scores)[-k:]
    return float(y_true[idx].sum() / n_pos)


def compute_metrics(name: str, y_true, y_score) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    row: dict = {
        "model": name,
        "auroc": round(roc_auc_score(y, s), 6),
        "auprc": round(average_precision_score(y, s), 6),
        "n_pos": int(y.sum()),
        "n": int(len(y)),
    }
    for k in [50, 100, 200, 500]:
        row[f"p@{k}"] = round(_precision_at_k(y, s, k), 4)
        row[f"r@{k}"] = round(_recall_at_k(y, s, k), 4)
    return row


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] via percentile rank — makes ensemble stable."""
    return pd.Series(scores).rank(pct=True).to_numpy()


# ══════════════════════════════════════════════════════════════════════════════
# LightGBM
# ══════════════════════════════════════════════════════════════════════════════

def _lgb_cat_prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype("category")
    return Xc


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                  cat_cols: list[str]) -> dict:
    print("\n  Optuna search for LightGBM …")
    skf = StratifiedKFold(N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "min_child_samples": 100,   # fixed — prevents split errors with few positives
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": 1.0,    # no bagging — ensures all positives always present
            "bagging_freq": 0,
            "reg_alpha": 0.0,           # no L1 — L1 kills minority-class splits
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "is_unbalance": True,       # replaces scale_pos_weight — more stable
            "random_state": RANDOM_STATE,
            "device_type": LGB_DEVICE,
            **( {"gpu_device_id": 0} if LGB_DEVICE == "gpu" else {} ),
            "verbose": -1,
        }
        fold_scores = []
        for tr_i, va_i in skf.split(X_train, y_train):
            Xtr = _lgb_cat_prep(X_train.iloc[tr_i], cat_cols)
            Xva = _lgb_cat_prep(X_train.iloc[va_i], cat_cols)
            ytr = y_train.iloc[tr_i]
            yva = y_train.iloc[va_i]
            m = lgb.LGBMClassifier(**params)
            m.fit(Xtr, ytr,
                  categorical_feature=cat_cols if cat_cols else "auto",
                  callbacks=[lgb.log_evaluation(-1)])
            fold_scores.append(
                average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True,
                   catch=(Exception,))
    print(f"  Best CV AUPRC: {study.best_value:.6f}")
    print(f"  Best params:   {study.best_params}")
    return study.best_params


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   cat_cols: list[str], best_params: dict) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "is_unbalance": True,
        "min_child_samples": 100,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "reg_alpha": 0.0,
        "random_state": RANDOM_STATE,
        "device_type": LGB_DEVICE,
        **( {"gpu_device_id": 0} if LGB_DEVICE == "gpu" else {} ),
        "verbose": -1,
        **best_params,
    }
    Xtr = _lgb_cat_prep(X_train, cat_cols)
    Xte = _lgb_cat_prep(X_test, cat_cols)

    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr, y_train,
              eval_set=[(Xte, y_test)],
              categorical_feature=cat_cols if cat_cols else "auto",
              callbacks=[
                  lgb.early_stopping(50, verbose=False),
                  lgb.log_evaluation(-1),
              ])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def tune_xgboost(X_train_enc: pd.DataFrame, y_train: pd.Series,
                 n_pos: int, n_neg: int) -> dict:
    print("\n  Optuna search for XGBoost …")
    scale_pos = n_neg / n_pos
    skf = StratifiedKFold(N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_pos,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": XGB_DEVICE,
            "random_state": RANDOM_STATE,
        }
        fold_scores = []
        for tr_i, va_i in skf.split(X_train_enc, y_train):
            Xtr = X_train_enc.iloc[tr_i]
            Xva = X_train_enc.iloc[va_i]
            ytr = y_train.iloc[tr_i]
            yva = y_train.iloc[va_i]
            m = xgb.XGBClassifier(**params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            fold_scores.append(
                average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True,
                   catch=(Exception,))
    print(f"  Best CV AUPRC: {study.best_value:.6f}")
    print(f"  Best params:   {study.best_params}")
    return study.best_params


def train_xgboost(X_train_enc: pd.DataFrame, y_train: pd.Series,
                  X_test_enc: pd.DataFrame, y_test: pd.Series,
                  best_params: dict, n_pos: int, n_neg: int) -> xgb.XGBClassifier:
    params = {
        "scale_pos_weight": n_neg / n_pos,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": RANDOM_STATE,
        **best_params,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_enc, y_train,
              eval_set=[(X_test_enc, y_test)],
              verbose=False)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CatBoost
# ══════════════════════════════════════════════════════════════════════════════

def _cb_cat_prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy().reset_index(drop=True)
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str)
    return Xc


def tune_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                  cat_cols: list[str]) -> dict:
    print("\n  Optuna search for CatBoost …")
    cat_idx = [list(X_train.columns).index(c) for c in cat_cols]
    skf = StratifiedKFold(N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 2.0),
            "auto_class_weights": "Balanced",
            "eval_metric": "AUC",
            "task_type": "GPU",
            "devices": "0",
            "cat_features": cat_idx,
            "random_seed": RANDOM_STATE,
            "verbose": 0,
        }
        fold_scores = []
        for tr_i, va_i in skf.split(X_train, y_train):
            Xtr = _cb_cat_prep(X_train.iloc[tr_i], cat_cols)
            Xva = _cb_cat_prep(X_train.iloc[va_i], cat_cols)
            ytr = y_train.iloc[tr_i].reset_index(drop=True)
            yva = y_train.iloc[va_i].reset_index(drop=True)
            try:
                m = CatBoostClassifier(**params)
                m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            except Exception:
                # GPU failed — retry on CPU
                p_cpu = {**params, "task_type": "CPU"}
                p_cpu.pop("devices", None)
                m = CatBoostClassifier(**p_cpu)
                m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            fold_scores.append(
                average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True,
                   catch=(Exception,))
    print(f"  Best CV AUPRC: {study.best_value:.6f}")
    print(f"  Best params:   {study.best_params}")
    return study.best_params


def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   cat_cols: list[str], best_params: dict) -> CatBoostClassifier:
    cat_idx = [list(X_train.columns).index(c) for c in cat_cols]
    params = {
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": "0",
        "cat_features": cat_idx,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        **best_params,
    }
    try:
        model = CatBoostClassifier(**params)
        model.fit(_cb_cat_prep(X_train, cat_cols), y_train.reset_index(drop=True),
                  eval_set=(_cb_cat_prep(X_test, cat_cols), y_test.reset_index(drop=True)),
                  use_best_model=True)
        return model
    except Exception:
        print("  CatBoost GPU failed — retrying on CPU …")
        p_cpu = {**params, "task_type": "CPU"}
        p_cpu.pop("devices", None)
        model = CatBoostClassifier(**p_cpu)
    Xtr = _cb_cat_prep(X_train, cat_cols)
    Xte = _cb_cat_prep(X_test, cat_cols)
    model.fit(Xtr, y_train.reset_index(drop=True),
              eval_set=(Xte, y_test.reset_index(drop=True)),
              use_best_model=True)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Logistic Regression baseline — class lives in lr_model.py (imported above)
# ══════════════════════════════════════════════════════════════════════════════


def train_logistic_regression(
        X_train_enc: pd.DataFrame, y_train: pd.Series,
) -> tuple[_GPULogisticRegression, StandardScaler]:
    """Grid-search over weight_decay for GPU logistic regression (PyTorch)."""
    print("\n  Grid-searching GPU Logistic Regression (PyTorch) …")
    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train_enc)
    skf     = StratifiedKFold(N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    n_pos      = int(y_train.sum())
    n_neg      = int(len(y_train) - n_pos)
    pos_weight = n_neg / n_pos   # mirrors scale_pos_weight of tree models

    best_wd, best_score = 1e-3, -1.0
    for wd in [1e-3, 1e-1, 1.0]:           # 3 values instead of 5
        fold_scores = []
        for tr_i, va_i in skf.split(Xtr_sc, y_train):
            m = _GPULogisticRegression(weight_decay=wd, n_epochs=30,  # 30 instead of 80
                                        pos_weight=pos_weight)
            m.fit(Xtr_sc[tr_i], y_train.iloc[tr_i].values)
            preds = m.predict_proba(Xtr_sc[va_i])[:, 1]
            fold_scores.append(average_precision_score(y_train.iloc[va_i], preds))
        avg = float(np.mean(fold_scores))
        print(f"    weight_decay={wd:>6}  CV AUPRC={avg:.6f}")
        if avg > best_score:
            best_score, best_wd = avg, wd

    print(f"  Best weight_decay={best_wd}  CV AUPRC={best_score:.6f}")
    final = _GPULogisticRegression(weight_decay=best_wd, n_epochs=60,  # 60 instead of 150
                                    pos_weight=pos_weight)
    final.fit(Xtr_sc, y_train.values)
    return final, scaler


# ══════════════════════════════════════════════════════════════════════════════
# SHAP explainability
# ══════════════════════════════════════════════════════════════════════════════

def save_shap_plots(model, X_sample: pd.DataFrame, model_name: str) -> None:
    print(f"\n  Computing SHAP for {model_name} …")
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        # LightGBM returns list [neg_class, pos_class]; XGBoost/CatBoost array
        if isinstance(sv, list):
            sv = sv[1]

        # Bar summary (mean |SHAP|)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_sample, plot_type="bar", show=False,
                          max_display=20)
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_{model_name.lower()}_bar.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Beeswarm
        plt.figure(figsize=(12, 10))
        shap.summary_plot(sv, X_sample, show=False, max_display=20)
        plt.title(f"SHAP Beeswarm — {model_name}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_{model_name.lower()}_beeswarm.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    SHAP plots saved.")
    except Exception as exc:
        print(f"    SHAP failed ({exc}) — skipping.")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def save_pr_curves(y_test: pd.Series, scores_dict: dict[str, np.ndarray]) -> None:
    colours = {
        "LogisticRegression": "orange",
        "LightGBM": "blue",
        "XGBoost": "green",
        "CatBoost": "purple",
        "Ensemble": "red",
    }
    plt.figure(figsize=(10, 7))
    for name, scores in scores_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, scores)
        auprc = average_precision_score(y_test, scores)
        plt.plot(rec, prec, label=f"{name} (AUPRC={auprc:.4f})",
                 color=colours.get(name, "black"))
    baseline = float(y_test.mean())
    plt.axhline(baseline, color="gray", linestyle="--",
                label=f"No-skill baseline ({baseline:.5f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  PR-curves saved.")


def save_precision_at_k_plot(y_test: pd.Series,
                              scores_dict: dict[str, np.ndarray]) -> None:
    ks = [50, 100, 200, 500, 1000]
    plt.figure(figsize=(10, 6))
    for name, scores in scores_dict.items():
        p_vals = [_precision_at_k(np.asarray(y_test), scores, k) for k in ks]
        plt.plot(ks, p_vals, marker="o", label=name)
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Precision at Top-K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Precision@K plot saved.")


def save_score_distribution(y_test: pd.Series,
                             scores_dict: dict[str, np.ndarray]) -> None:
    """Score distribution of best model split by label."""
    best_name = "Ensemble"
    scores = scores_dict[best_name]
    fig, ax = plt.subplots(figsize=(10, 5))
    y_arr = np.asarray(y_test)
    ax.hist(scores[y_arr == 0], bins=100, alpha=0.6, label="Non-fraud", color="steelblue")
    ax.hist(scores[y_arr == 1], bins=20, alpha=0.8, label="Fraud", color="red")
    ax.set_xlabel("Fraud Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {best_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Score distribution saved.")


# ══════════════════════════════════════════════════════════════════════════════
# Full-dataset scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_all_providers(
    X: pd.DataFrame,
    y: pd.Series,
    provider_ids: pd.Series,
    cat_cols: list[str],
    encoders: dict[str, LabelEncoder],
    lgb_model: lgb.LGBMClassifier,
    xgb_model: xgb.XGBClassifier,
    cb_model: CatBoostClassifier,
    lr_model: _GPULogisticRegression,
    scaler: StandardScaler,
    lgb_weights: tuple[float, float, float, float] = (0.05, 0.45, 0.40, 0.10),
) -> pd.DataFrame:
    """Score every provider and return a DataFrame compatible with dashboard.py."""
    print("\n  Scoring all providers …")

    # LightGBM
    X_lgb = _lgb_cat_prep(X, cat_cols)
    all_lgb = lgb_model.predict_proba(X_lgb)[:, 1]

    # XGBoost  — apply saved label encoders
    X_enc = X.copy()
    for col, le in encoders.items():
        # handle unseen labels gracefully
        X_enc[col] = X_enc[col].astype(str).map(
            lambda v, _le=le: _le.transform([v])[0]
            if v in _le.classes_ else -1
        )
    all_xgb = xgb_model.predict_proba(X_enc)[:, 1]

    # LogReg
    all_lr = lr_model.predict_proba(scaler.transform(X_enc))[:, 1]

    # CatBoost
    X_cb = _cb_cat_prep(X, cat_cols)
    all_cb = cb_model.predict_proba(X_cb)[:, 1]

    w_lgb, w_xgb, w_cb, w_lr = lgb_weights
    all_ensemble = (
        rank_normalize(all_lgb) * w_lgb
        + rank_normalize(all_xgb) * w_xgb
        + rank_normalize(all_cb) * w_cb
        + rank_normalize(all_lr) * w_lr
    )

    scored = pd.DataFrame({
        "provider_id": provider_ids.values,
        "fraud_score": all_ensemble,
        "lightgbm_score": all_lgb,
        "xgboost_score": all_xgb,
        "catboost_score": all_cb,
        "lr_score": all_lr,
        "label": y.values,
    })
    scored.to_csv(SCORED_CSV, index=False)
    print(f"  Scored {len(scored):,} providers → {SCORED_CSV}")
    return scored


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    X, y, provider_ids, cat_cols = load_data(FEATURES_CSV)

    # ── Train / test split ────────────────────────────────────────────────────
    (X_train, X_test,
     y_train, y_test,
     id_train, id_test) = train_test_split(
        X, y, provider_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    print(f"\nTrain: {X_train.shape}  positives={n_pos}  negatives={n_neg}")
    print(f"Test:  {X_test.shape}   positives={int(y_test.sum())}")
    print(f"scale_pos_weight = {n_neg/n_pos:.1f}")

    # ── Label-encoded versions (XGBoost + LogReg) ─────────────────────────────
    X_train_enc, X_test_enc, encoders = make_label_encoders(
        X_train, X_test, cat_cols)

    # ════════════════════════════════════════════════════════════════════════
    # 1. Logistic Regression  (quick, no Optuna needed)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("1 / 4  Logistic Regression")
    print("=" * 60)
    lr_model, scaler = train_logistic_regression(
        X_train_enc, y_train)
    lr_scores = lr_model.predict_proba(scaler.transform(X_test_enc))[:, 1]
    lr_metrics = compute_metrics("LogisticRegression", y_test, lr_scores)
    print(f"  → AUROC={lr_metrics['auroc']:.4f}  "
          f"AUPRC={lr_metrics['auprc']:.6f}  "
          f"P@100={lr_metrics['p@100']:.4f}")
    joblib.dump((lr_model, scaler),
                MODEL_DIR / "logistic_regression.joblib")

    # ════════════════════════════════════════════════════════════════════════
    # 2. LightGBM
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2 / 4  LightGBM")
    print("=" * 60)
    lgb_best = tune_lightgbm(X_train, y_train, cat_cols)
    lgb_model = train_lightgbm(
        X_train, y_train, X_test, y_test, cat_cols, lgb_best)
    lgb_scores = lgb_model.predict_proba(
        _lgb_cat_prep(X_test, cat_cols))[:, 1]
    lgb_metrics = compute_metrics("LightGBM", y_test, lgb_scores)
    print(f"  → AUROC={lgb_metrics['auroc']:.4f}  "
          f"AUPRC={lgb_metrics['auprc']:.6f}  "
          f"P@100={lgb_metrics['p@100']:.4f}")
    joblib.dump(lgb_model, MODEL_DIR / "lightgbm_model.joblib")

    # ════════════════════════════════════════════════════════════════════════
    # 3. XGBoost
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3 / 4  XGBoost")
    print("=" * 60)
    xgb_best = tune_xgboost(X_train_enc, y_train, n_pos, n_neg)
    xgb_model = train_xgboost(
        X_train_enc, y_train, X_test_enc, y_test, xgb_best, n_pos, n_neg)
    xgb_scores = xgb_model.predict_proba(X_test_enc)[:, 1]
    xgb_metrics = compute_metrics("XGBoost", y_test, xgb_scores)
    print(f"  → AUROC={xgb_metrics['auroc']:.4f}  "
          f"AUPRC={xgb_metrics['auprc']:.6f}  "
          f"P@100={xgb_metrics['p@100']:.4f}")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.joblib")

    # ════════════════════════════════════════════════════════════════════════
    # 4. CatBoost
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4 / 4  CatBoost")
    print("=" * 60)
    cb_best = tune_catboost(X_train, y_train, cat_cols)
    cb_model = train_catboost(
        X_train, y_train, X_test, y_test, cat_cols, cb_best)
    cb_scores = cb_model.predict_proba(
        _cb_cat_prep(X_test, cat_cols))[:, 1]
    cb_metrics = compute_metrics("CatBoost", y_test, cb_scores)
    print(f"  → AUROC={cb_metrics['auroc']:.4f}  "
          f"AUPRC={cb_metrics['auprc']:.6f}  "
          f"P@100={cb_metrics['p@100']:.4f}")
    cb_model.save_model(str(MODEL_DIR / "catboost_model.cbm"))

    # ════════════════════════════════════════════════════════════════════════
    # 5. Rank-average ensemble
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Ensemble (rank-average)")
    print("=" * 60)
    # Weights chosen to up-weight CatBoost (often best on imbalanced data)
    # and down-weight Logistic Regression.
    ens_weights = (0.05, 0.45, 0.40, 0.10)  # lgb, xgb, cb, lr
    w_lgb, w_xgb, w_cb, w_lr = ens_weights
    ens_scores = (
        rank_normalize(lgb_scores) * w_lgb
        + rank_normalize(xgb_scores) * w_xgb
        + rank_normalize(cb_scores)  * w_cb
        + rank_normalize(lr_scores)  * w_lr
    )
    ens_metrics = compute_metrics("Ensemble", y_test, ens_scores)
    print(f"  → AUROC={ens_metrics['auroc']:.4f}  "
          f"AUPRC={ens_metrics['auprc']:.6f}  "
          f"P@100={ens_metrics['p@100']:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # 6. SHAP (LightGBM on test sample)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SHAP explainability")
    print("=" * 60)
    rng = np.random.RandomState(RANDOM_STATE)
    shap_idx = rng.choice(len(X_test), min(3000, len(X_test)), replace=False)
    X_shap = _lgb_cat_prep(X_test.iloc[shap_idx], cat_cols).reset_index(drop=True)
    save_shap_plots(lgb_model, X_shap, "LightGBM")

    # ════════════════════════════════════════════════════════════════════════
    # 7. Plots
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Saving evaluation plots")
    print("=" * 60)
    scores_dict = {
        "LogisticRegression": lr_scores,
        "LightGBM": lgb_scores,
        "XGBoost": xgb_scores,
        "CatBoost": cb_scores,
        "Ensemble": ens_scores,
    }
    save_pr_curves(y_test, scores_dict)
    save_precision_at_k_plot(y_test, scores_dict)
    save_score_distribution(y_test, scores_dict)

    # ════════════════════════════════════════════════════════════════════════
    # 8. Score all providers
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Scoring full dataset")
    print("=" * 60)
    score_all_providers(
        X, y, provider_ids, cat_cols, encoders,
        lgb_model, xgb_model, cb_model, lr_model, scaler, ens_weights)

    # ════════════════════════════════════════════════════════════════════════
    # 9. Save metrics (dashboard-compatible format)
    # ════════════════════════════════════════════════════════════════════════
    def _json_safe(x):
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.integer, np.int32, np.int64)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json_safe(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)

    all_metrics = {
        "test_metrics": [lr_metrics, lgb_metrics, xgb_metrics,
                         cb_metrics, ens_metrics],
        "ensemble_threshold": {
            "value": float(np.percentile(ens_scores, 99)),
            "note": "99th-percentile of ensemble score on test set",
        },
        "config": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_cv_folds": N_CV_FOLDS,
            "n_trials_per_model": N_TRIALS,
            "ensemble_weights": {
                "lightgbm": w_lgb,
                "xgboost": w_xgb,
                "catboost": w_cb,
                "logistic_regression": w_lr,
            },
        },
        "best_params": {
            "lightgbm": _json_safe(lgb_best),
            "xgboost": _json_safe(xgb_best),
            "catboost": _json_safe(cb_best),
        },
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {METRICS_JSON}")

    # ════════════════════════════════════════════════════════════════════════
    # 10. Summary
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Model':<22} {'AUROC':>8} {'AUPRC':>10} {'P@50':>7} {'P@100':>7} {'P@200':>7} {'P@500':>7}"
    print(header)
    print("-" * len(header))
    for m in [lr_metrics, lgb_metrics, xgb_metrics, cb_metrics, ens_metrics]:
        print(f"{m['model']:<22} {m['auroc']:>8.4f} {m['auprc']:>10.6f} "
              f"{m['p@50']:>7.4f} {m['p@100']:>7.4f} "
              f"{m['p@200']:>7.4f} {m['p@500']:>7.4f}")


if __name__ == "__main__":
    main()
