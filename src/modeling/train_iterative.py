"""
src/modeling/train_iterative.py

Iterative under-sampling ensemble for Medicare fraud detection.

Approach (EasyEnsemble-style):
  For each of N_ITERATIONS iterations:
    1. Sample N_NEGATIVES_PER_ITER negatives from the training set.
    2. Combine with ALL training positives.
    3. Train a LightGBM AND an XGBoost classifier on this balanced batch.
    4. Average their probabilities for this iteration's prediction.
  Final score = mean prediction across all iterations.

Hyperparameters are tuned with Optuna before the iterative loop, using a
mini-ensemble (N_TUNE_MINI_ITERS iterations) evaluated on a held-out
validation slice of the training data. This ensures params are optimised
for the balanced-batch regime, not the full 6700:1 imbalanced dataset.

Run
---
    python src/modeling/train_iterative.py

Outputs (all separate from the original pipeline):
    data/final/scored_providers_iterative.csv
    outputs/iterative/metrics.json
    outputs/iterative/plots/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT_ROOT / "data" / "final" / "provider_features.csv"
SCORED_CSV   = PROJECT_ROOT / "data" / "final" / "scored_providers_iterative.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "iterative"
PLOTS_DIR    = OUTPUT_DIR / "plots"
METRICS_JSON = OUTPUT_DIR / "metrics.json"

for _d in [OUTPUT_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
RANDOM_STATE         = 42
TEST_SIZE            = 0.20
N_ITERATIONS         = 100   # models in the final ensemble
N_NEGATIVES_PER_ITER = 200   # negatives sampled per iteration (as instructed)
N_TUNE_MINI_ITERS    = 20    # iterations per Optuna trial (fast approximation)
N_TRIALS_LGB         = 25    # Optuna trials for LightGBM
N_TRIALS_XGB         = 25    # Optuna trials for XGBoost
TUNE_VAL_SIZE        = 0.20  # fraction of training data held out for tuning


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path):
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


def _lgb_cat_prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype("category")
    return Xc


def _encode_cats_for_xgb(
    X: pd.DataFrame, cat_cols: list[str]
) -> pd.DataFrame:
    Xe = X.copy()
    for c in cat_cols:
        Xe[c] = Xe[c].astype("category").cat.codes
    return Xe


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ══════════════════════════════════════════════════════════════════════════════

def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(scores)[-k:]
    return float(y_true[idx].sum() / k)


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    n_pos = int(y_true.sum())
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
        "n":     int(len(y)),
    }
    for k in [50, 100, 200, 500]:
        row[f"p@{k}"] = round(_precision_at_k(y, s, k), 4)
        row[f"r@{k}"] = round(_recall_at_k(y, s, k), 4)
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Optuna tuning — balanced-batch regime
# ══════════════════════════════════════════════════════════════════════════════

def _mini_ensemble_score(
    X_pos: pd.DataFrame,
    y_pos: pd.Series,
    X_neg_pool: pd.DataFrame,
    X_val_lgb: pd.DataFrame,
    y_val: np.ndarray,
    lgb_params: dict,
    n_mini: int,
    rng: np.random.RandomState,
    cat_cols: list[str],
) -> float:
    n_neg = len(X_neg_pool)
    val_sum = np.zeros(len(X_val_lgb))
    for i in range(n_mini):
        neg_idx   = rng.choice(n_neg, size=N_NEGATIVES_PER_ITER, replace=False)
        X_neg_bat = X_neg_pool.iloc[neg_idx].reset_index(drop=True)
        y_neg_bat = pd.Series(np.zeros(N_NEGATIVES_PER_ITER, dtype=int))
        X_bat     = pd.concat([X_pos, X_neg_bat], ignore_index=True)
        y_bat     = pd.concat([y_pos, y_neg_bat], ignore_index=True)
        model     = lgb.LGBMClassifier(**lgb_params, random_state=i)
        model.fit(
            _lgb_cat_prep(X_bat, cat_cols),
            y_bat,
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[lgb.log_evaluation(-1)],
        )
        val_sum += model.predict_proba(X_val_lgb)[:, 1]
    return average_precision_score(y_val, val_sum / n_mini)


def tune_lgb(
    X_pos: pd.DataFrame,
    y_pos: pd.Series,
    X_neg_pool: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
) -> dict:
    print(f"\n  Optuna — LightGBM ({N_TRIALS_LGB} trials) …")
    X_val_lgb = _lgb_cat_prep(X_val, cat_cols)
    rng = np.random.RandomState(RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "binary",
            "boosting_type":     "gbdt",
            "num_leaves":        trial.suggest_int("num_leaves", 7, 63),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 10.0),
            "verbose":           -1,
            "n_jobs":            -1,
        }
        return _mini_ensemble_score(
            X_pos, y_pos, X_neg_pool, X_val_lgb, y_val,
            params, N_TUNE_MINI_ITERS, rng, cat_cols,
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS_LGB,
                   show_progress_bar=True, catch=(Exception,))
    print(f"  Best val AUPRC: {study.best_value:.6f}")
    print(f"  Best params:    {study.best_params}")
    return study.best_params


def tune_xgb(
    X_pos: pd.DataFrame,
    y_pos: pd.Series,
    X_neg_pool: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
) -> dict:
    print(f"\n  Optuna — XGBoost ({N_TRIALS_XGB} trials) …")
    X_val_enc = _encode_cats_for_xgb(X_val, cat_cols)
    rng = np.random.RandomState(RANDOM_STATE + 1)
    n_neg = len(X_neg_pool)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method":      "hist",
            "eval_metric":      "aucpr",
            "verbosity":        0,
        }
        val_sum = np.zeros(len(X_val_enc))
        for i in range(N_TUNE_MINI_ITERS):
            neg_idx   = rng.choice(n_neg, size=N_NEGATIVES_PER_ITER, replace=False)
            X_neg_bat = X_neg_pool.iloc[neg_idx].reset_index(drop=True)
            y_neg_bat = pd.Series(np.zeros(N_NEGATIVES_PER_ITER, dtype=int))
            X_bat     = _encode_cats_for_xgb(
                pd.concat([X_pos, X_neg_bat], ignore_index=True), cat_cols)
            y_bat     = pd.concat([y_pos, y_neg_bat], ignore_index=True)
            model     = xgb.XGBClassifier(**params, random_state=i)
            model.fit(X_bat, y_bat, verbose=False)
            val_sum  += model.predict_proba(X_val_enc)[:, 1]
        return average_precision_score(y_val, val_sum / N_TUNE_MINI_ITERS)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 1),
    )
    study.optimize(objective, n_trials=N_TRIALS_XGB,
                   show_progress_bar=True, catch=(Exception,))
    print(f"  Best val AUPRC: {study.best_value:.6f}")
    print(f"  Best params:    {study.best_params}")
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# Iterative ensemble
# ══════════════════════════════════════════════════════════════════════════════

def train_iterative_ensemble(
    X_train:    pd.DataFrame,
    y_train:    pd.Series,
    X_full:     pd.DataFrame,
    X_test:     pd.DataFrame,
    cat_cols:   list[str],
    lgb_params: dict,
    xgb_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train N_ITERATIONS (LightGBM + XGBoost) pairs on balanced batches.

    Each iteration:
      - Sample N_NEGATIVES_PER_ITER negatives (no replacement within iteration)
      - Train one LightGBM and one XGBoost with tuned params
      - Average their probabilities for that iteration's prediction

    Final score = mean across all iterations.
    """
    y_tr      = y_train.values
    pos_mask  = y_tr == 1
    neg_mask  = ~pos_mask

    X_pos     = X_train.loc[pos_mask].reset_index(drop=True)
    y_pos     = pd.Series(np.ones(pos_mask.sum(), dtype=int))
    X_neg_all = X_train.loc[neg_mask].reset_index(drop=True)

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    print(f"  Training positives : {n_pos}")
    print(f"  Training negatives : {n_neg}")
    print(f"  Batch per iteration: {n_pos} pos + {N_NEGATIVES_PER_ITER} neg  "
          f"(ratio ≈ {N_NEGATIVES_PER_ITER/n_pos:.2f}:1)")
    print(f"  Running {N_ITERATIONS} iterations (LightGBM + XGBoost each) …\n")

    X_test_lgb = _lgb_cat_prep(X_test,  cat_cols)
    X_full_lgb = _lgb_cat_prep(X_full,  cat_cols)
    X_test_xgb = _encode_cats_for_xgb(X_test,  cat_cols)
    X_full_xgb = _encode_cats_for_xgb(X_full,  cat_cols)

    lgb_final = {
        "objective":     "binary",
        "boosting_type": "gbdt",
        "verbose":       -1,
        "n_jobs":        -1,
        **lgb_params,
    }
    xgb_final = {
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "verbosity":   0,
        **xgb_params,
    }

    rng      = np.random.RandomState(RANDOM_STATE)
    test_sum = np.zeros(len(X_test))
    full_sum = np.zeros(len(X_full))

    for i in range(N_ITERATIONS):
        neg_idx   = rng.choice(n_neg, size=N_NEGATIVES_PER_ITER, replace=False)
        X_neg_bat = X_neg_all.iloc[neg_idx].reset_index(drop=True)
        y_neg_bat = pd.Series(np.zeros(N_NEGATIVES_PER_ITER, dtype=int))

        X_bat = pd.concat([X_pos, X_neg_bat], ignore_index=True)
        y_bat = pd.concat([y_pos, y_neg_bat], ignore_index=True)

        lgb_model = lgb.LGBMClassifier(**lgb_final, random_state=i)
        lgb_model.fit(
            _lgb_cat_prep(X_bat, cat_cols),
            y_bat,
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[lgb.log_evaluation(-1)],
        )

        xgb_model = xgb.XGBClassifier(**xgb_final, random_state=i)
        xgb_model.fit(_encode_cats_for_xgb(X_bat, cat_cols), y_bat, verbose=False)

        test_sum += (lgb_model.predict_proba(X_test_lgb)[:, 1]
                     + xgb_model.predict_proba(X_test_xgb)[:, 1]) / 2
        full_sum += (lgb_model.predict_proba(X_full_lgb)[:, 1]
                     + xgb_model.predict_proba(X_full_xgb)[:, 1]) / 2

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1:>3}/{N_ITERATIONS} complete")

    return test_sum / N_ITERATIONS, full_sum / N_ITERATIONS


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def save_pr_curve(y_test: np.ndarray, scores: np.ndarray) -> None:
    prec, rec, _ = precision_recall_curve(y_test, scores)
    auprc    = average_precision_score(y_test, scores)
    baseline = float(y_test.mean())
    plt.figure(figsize=(9, 6))
    plt.plot(rec, prec,
             label=f"Iterative Ensemble (AUPRC={auprc:.4f})", color="red")
    plt.axhline(baseline, color="gray", linestyle="--",
                label=f"No-skill baseline ({baseline:.5f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Iterative Ensemble")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  PR-curve saved.")


def save_score_distribution(y_test: np.ndarray, scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[y_test == 0], bins=100, alpha=0.6,
            label="Non-fraud", color="steelblue")
    ax.hist(scores[y_test == 1], bins=20,  alpha=0.8,
            label="Fraud",     color="red")
    ax.set_xlabel("Fraud Score"); ax.set_ylabel("Count")
    ax.set_title("Score Distribution — Iterative Ensemble")
    ax.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Score distribution saved.")


def save_precision_at_k_plot(y_test: np.ndarray, scores: np.ndarray) -> None:
    ks     = [50, 100, 200, 500, 1000]
    p_vals = [_precision_at_k(y_test, scores, k) for k in ks]
    plt.figure(figsize=(9, 5))
    plt.plot(ks, p_vals, marker="o", color="red", label="Iterative Ensemble")
    plt.xlabel("K"); plt.ylabel("Precision@K")
    plt.title("Precision at Top-K — Iterative Ensemble")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Precision@K plot saved.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    X, y, provider_ids, cat_cols = load_data(FEATURES_CSV)

    (X_train, X_test,
     y_train, y_test,
     _id_tr,  _id_te) = train_test_split(
        X, y, provider_ids,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

    # Tuning split — held-out val from training data
    (X_tune_tr, X_tune_val,
     y_tune_tr, y_tune_val) = train_test_split(
        X_train, y_train,
        test_size=TUNE_VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train)
    X_tune_tr  = X_tune_tr.reset_index(drop=True)
    X_tune_val = X_tune_val.reset_index(drop=True)
    y_tune_tr  = y_tune_tr.reset_index(drop=True)
    y_tune_val = y_tune_val.reset_index(drop=True)

    pos_tune_mask   = y_tune_tr.values == 1
    X_pos_tune      = X_tune_tr.loc[pos_tune_mask].reset_index(drop=True)
    y_pos_tune      = pd.Series(np.ones(pos_tune_mask.sum(), dtype=int))
    X_neg_pool_tune = X_tune_tr.loc[~pos_tune_mask].reset_index(drop=True)

    print(f"\nTuning split — positives: {pos_tune_mask.sum()}  "
          f"negatives: {(~pos_tune_mask).sum()}  val: {len(X_tune_val)}")

    print("\n" + "=" * 60)
    print("Hyperparameter Tuning (balanced-batch regime)")
    print("=" * 60)
    lgb_best = tune_lgb(X_pos_tune, y_pos_tune, X_neg_pool_tune,
                        X_tune_val, y_tune_val.values, cat_cols)
    xgb_best = tune_xgb(X_pos_tune, y_pos_tune, X_neg_pool_tune,
                        X_tune_val, y_tune_val.values, cat_cols)

    print("\n" + "=" * 60)
    print(f"Iterative Under-sampling Ensemble  "
          f"({N_ITERATIONS} iterations, {N_NEGATIVES_PER_ITER} neg/iter)")
    print("=" * 60)
    test_scores, full_scores = train_iterative_ensemble(
        X_train, y_train, X, X_test, cat_cols, lgb_best, xgb_best)

    metrics = compute_metrics("IterativeEnsemble", y_test.values, test_scores)
    print(f"\n  → AUROC={metrics['auroc']:.4f}  "
          f"AUPRC={metrics['auprc']:.6f}  "
          f"P@100={metrics['p@100']:.4f}")

    print("\n" + "=" * 60)
    print("Saving evaluation plots")
    print("=" * 60)
    save_pr_curve(y_test.values, test_scores)
    save_score_distribution(y_test.values, test_scores)
    save_precision_at_k_plot(y_test.values, test_scores)

    print("\n" + "=" * 60)
    print("Saving scored providers")
    print("=" * 60)
    pd.DataFrame({
        "provider_id": provider_ids.values,
        "fraud_score": full_scores,
        "label":       y.values,
    }).to_csv(SCORED_CSV, index=False)
    print(f"  Saved {len(X):,} providers → {SCORED_CSV}")

    all_metrics = {
        "test_metrics": [metrics],
        "config": {
            "n_iterations":         N_ITERATIONS,
            "n_negatives_per_iter": N_NEGATIVES_PER_ITER,
            "random_state":         RANDOM_STATE,
            "test_size":            TEST_SIZE,
            "base_learners":        ["LightGBM", "XGBoost"],
            "n_trials_lgb":         N_TRIALS_LGB,
            "n_trials_xgb":         N_TRIALS_XGB,
        },
        "best_params": {"lightgbm": lgb_best, "xgboost": xgb_best},
        "ensemble_threshold": {
            "value": float(np.percentile(test_scores, 99)),
            "note":  "99th-percentile of iterative ensemble score on test set",
        },
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics saved → {METRICS_JSON}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    hdr = (f"{'Model':<26} {'AUROC':>8} {'AUPRC':>10} "
           f"{'P@50':>7} {'P@100':>7} {'P@200':>7} {'P@500':>7}")
    print(hdr); print("-" * len(hdr))
    m = metrics
    print(f"{m['model']:<26} {m['auroc']:>8.4f} {m['auprc']:>10.6f} "
          f"{m['p@50']:>7.4f} {m['p@100']:>7.4f} "
          f"{m['p@200']:>7.4f} {m['p@500']:>7.4f}")


if __name__ == "__main__":
    main()
