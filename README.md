# Medicare Provider Fraud Detection — ML Project

**ESIB — USJ | 4th Year | Machine Learning**

---

## Project Overview

In this project, we built a machine learning system to identify high-risk Medicare providers based on their billing behavior.

Since the dataset is extremely imbalanced (only 187 fraud cases out of ~1.26M providers), the goal is not to directly detect fraud, but to **assign a fraud risk score** and rank providers from most to least suspicious.

---

## Data

| Dataset | Description |
|---|---|
| CMS Provider Billing Data | ~1.26M providers |
| OIG Exclusion List | ~187 confirmed fraud cases |

We cleaned and joined both datasets using the provider identifier (NPI), then created a binary label:
- `1` → fraud
- `0` → non-fraud

---

## Pipeline

```
Raw Data → Preprocessing → Labeling → Feature Engineering → Model Training → Evaluation → Scoring → Dashboard
```

---

## Feature Engineering

We created **57 features** to capture abnormal behavior, including:

- **Financial ratios** — charge-to-allowed, payment ratios
- **Utilization metrics** — services per beneficiary
- **Specialty-based comparisons** — z-scores within each specialty
- **Anomaly scores** — Isolation Forest / unusual behavior signals

---

## Modeling Approaches

### Method 1 — Standard Training (Full Dataset)

- **Models:** Logistic Regression, LightGBM, XGBoost, CatBoost
- Class imbalance handled using weights
- **Final model:** Weighted ensemble

**Hyperparameters (Optuna tuned):**

| Parameter | Range |
|---|---|
| Learning rate | ~0.02–0.17 |
| Tree depth | 3–5 |
| n_estimators | ~250–480 |
| Regularization | reg_alpha, reg_lambda |
| Subsample & feature sampling | — |
| scale_pos_weight | ≈ 6715 (XGBoost) |

---

### Method 2 — Iterative Training (Balanced Sampling)

- **Models:** LightGBM, XGBoost
- Each iteration uses:
  - All 187 fraud cases
  - 200 randomly sampled non-fraud cases
- 100 iterations total — final prediction = average of all iterations

**Training settings:**

| Parameter | Value |
|---|---|
| n_iterations | 100 |
| n_negatives_per_iter | 200 |
| test_size | 0.2 |
| random_state | 42 |
| Optuna trials | 25 per model |

---

## Results

### Method 1 — Standard Training (Full Dataset)

| Model | AUROC | AUPRC |
|---|---|---|
| Logistic Regression | 0.704 | 0.0008 |
| LightGBM | 0.691 | 0.0003 |
| XGBoost | 0.775 | 0.0020 |
| CatBoost | 0.797 | 0.0018 |
| **Ensemble (Final)** | **0.795** | **0.0148** |

### Method 2 — Iterative Training (Balanced Sampling)

| Model | AUROC | AUPRC |
|---|---|---|
| Iterative Ensemble | 0.812 | 0.0013 |

### Key Observations

- **Method 2** achieves higher AUROC → better ranking ability
- **Method 1** ensemble achieves much higher AUPRC → better fraud detection performance
- Fraud cases are strongly concentrated in top-ranked providers

> Therefore, **Method 1 (ensemble)** was selected as the final model for prioritizing fraud investigation.

---

## Dashboard

We built a **Streamlit dashboard** to:

- Visualize KPIs (AUROC, AUPRC, fraud distribution)
- Analyze fraud concentration using deciles
- Identify top suspicious providers
- Compare model performance

---

## How to Run

```bash
python src/data/preprocess_provider_data.py
python src/data/preprocess_exclusion_data.py
python src/data/build_labels.py
python src/features/build_features.py
python src/modeling/train_models.py
python src/modeling/train_iterative.py
streamlit run dashboard.py
```

---

## Output

**`data/final/scored_providers.csv`** — Contains fraud risk scores for all providers.

---

## Docker

The Docker image bundles the pre-trained models and pre-computed scores, and serves the Streamlit dashboard. The training pipeline is **not** included — the image is dashboard-only. Raw data files are excluded from the image and can be uploaded at runtime via the dashboard sidebar.

**Option 1 — Pull from Docker Hub** (recommended):

The image is available on [Docker Hub](https://hub.docker.com/repository/docker/marcg7/ml-insurance-fraud/general).

```bash
docker pull marcg7/ml-insurance-fraud:latest
docker run -p 8501:8501 marcg7/ml-insurance-fraud:latest
```

**Option 2 — Build locally:**

```bash
docker build -t marcg7/ml-insurance-fraud:latest .
docker run -p 8501:8501 marcg7/ml-insurance-fraud:latest
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> The image uses a CPU-only PyTorch build to keep the size manageable.

---

## Limitations

- Extreme imbalance leads to false positives
- High score ≠ confirmed fraud
- Model should be used as a **decision-support tool**

---

## Conclusion

This project shows that machine learning can successfully rank high-risk providers even in highly imbalanced data. By focusing on the top-ranked providers, the system reduces the investigation space while capturing a large portion of fraud cases.
