"""
Regenerate dashboard plots from scored CSVs — no retraining needed.

Outputs:
  outputs/plots/pr_curves.png
  outputs/plots/precision_at_k.png
  outputs/plots/score_distribution.png
  outputs/iterative/plots/pr_curve.png
  outputs/iterative/plots/precision_at_k.png
  outputs/iterative/plots/score_distribution.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
ORIG_PLOTS = ROOT / "outputs" / "plots"
ITER_PLOTS = ROOT / "outputs" / "iterative" / "plots"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(scores)[-k:]
    return float(y_true[idx].sum() / k)



# ── Original ensemble ──────────────────────────────────────────────────────────

def _orig_scores_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "LightGBM":           df["lightgbm_score"].to_numpy(),
        "XGBoost":            df["xgboost_score"].to_numpy(),
        "CatBoost":           df["catboost_score"].to_numpy(),
        "LogisticRegression": df["lr_score"].to_numpy(),
        "Ensemble":           df["fraud_score"].to_numpy(),
    }


def gen_orig_pr_curves(y: np.ndarray, scores_dict: dict[str, np.ndarray]) -> None:
    colours = {
        "LogisticRegression": "orange",
        "LightGBM": "blue",
        "XGBoost": "green",
        "CatBoost": "purple",
        "Ensemble": "red",
    }
    plt.figure(figsize=(10, 7))
    for name, scores in scores_dict.items():
        prec, rec, _ = precision_recall_curve(y, scores)
        auprc = average_precision_score(y, scores)
        plt.plot(rec, prec, label=f"{name} (AUPRC={auprc:.4f})",
                 color=colours.get(name, "black"))
    baseline = float(y.mean())
    plt.axhline(baseline, color="gray", linestyle="--",
                label=f"No-skill baseline ({baseline:.5f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(ORIG_PLOTS / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved pr_curves.png")


def gen_orig_precision_at_k(y: np.ndarray, scores_dict: dict[str, np.ndarray]) -> None:
    ks = [50, 100, 200, 500, 1000]
    plt.figure(figsize=(10, 6))
    for name, scores in scores_dict.items():
        p_vals = [_precision_at_k(y, scores, k) for k in ks]
        plt.plot(ks, p_vals, marker="o", label=name)
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Precision at Top-K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ORIG_PLOTS / "precision_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved precision_at_k.png")


def gen_orig_score_dist(y: np.ndarray, scores_dict: dict[str, np.ndarray]) -> None:
    scores = scores_dict["Ensemble"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[y == 0], bins=100, alpha=0.6, label="Non-fraud", color="steelblue")
    ax.hist(scores[y == 1], bins=20,  alpha=0.8, label="Fraud",     color="red")
    ax.set_xlabel("Fraud Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — Ensemble")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ORIG_PLOTS / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved score_distribution.png")


# ── Iterative ensemble ─────────────────────────────────────────────────────────

def gen_iter_pr_curve(y: np.ndarray, scores: np.ndarray) -> None:
    prec, rec, _ = precision_recall_curve(y, scores)
    auprc = average_precision_score(y, scores)
    baseline = float(y.mean())
    plt.figure(figsize=(9, 6))
    plt.plot(rec, prec, label=f"Iterative Ensemble (AUPRC={auprc:.4f})", color="red")
    plt.axhline(baseline, color="gray", linestyle="--",
                label=f"No-skill baseline ({baseline:.5f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Iterative Ensemble")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(ITER_PLOTS / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved iterative/pr_curve.png")


def gen_iter_precision_at_k(y: np.ndarray, scores: np.ndarray) -> None:
    ks = [50, 100, 200, 500, 1000]
    p_vals = [_precision_at_k(y, scores, k) for k in ks]
    plt.figure(figsize=(9, 5))
    plt.plot(ks, p_vals, marker="o", color="red", label="Iterative Ensemble")
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Precision at Top-K — Iterative Ensemble")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ITER_PLOTS / "precision_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved iterative/precision_at_k.png")


def gen_iter_score_dist(y: np.ndarray, scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[y == 0], bins=100, alpha=0.6, label="Non-fraud", color="steelblue")
    ax.hist(scores[y == 1], bins=20,  alpha=0.8, label="Fraud",     color="red")
    ax.set_xlabel("Fraud Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — Iterative Ensemble")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ITER_PLOTS / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved iterative/score_distribution.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ORIG_PLOTS.mkdir(parents=True, exist_ok=True)
    ITER_PLOTS.mkdir(parents=True, exist_ok=True)

    orig_csv = ROOT / "data" / "final" / "scored_providers.csv"
    iter_csv = ROOT / "data" / "final" / "scored_providers_iterative.csv"

    print("== Original ensemble ==")
    df_orig = pd.read_csv(orig_csv)
    y_orig = df_orig["label"].to_numpy()
    scores_dict = _orig_scores_dict(df_orig)
    gen_orig_pr_curves(y_orig, scores_dict)
    gen_orig_precision_at_k(y_orig, scores_dict)
    gen_orig_score_dist(y_orig, scores_dict)

    print("\n== Iterative ensemble ==")
    df_iter = pd.read_csv(iter_csv)
    y_iter = df_iter["label"].to_numpy()
    scores_iter = df_iter["fraud_score"].to_numpy()
    gen_iter_pr_curve(y_iter, scores_iter)
    gen_iter_precision_at_k(y_iter, scores_iter)
    gen_iter_score_dist(y_iter, scores_iter)

    print("\nDone.")


if __name__ == "__main__":
    main()
