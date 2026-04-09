# Telco Customer Churn Prediction

End-to-end machine learning pipeline for predicting customer attrition using Random Forest in R.

![R](https://img.shields.io/badge/Language-R-blue) ![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-purple) ![AUC](https://img.shields.io/badge/AUC-0.98%2B-teal)

---

## Overview

Customer churn is among the most costly operational challenges in telecommunications. This project implements a zero-leakage, cross-validated churn classification system using Random Forest. The pipeline covers preprocessing, imbalance correction, model training, evaluation, and feature attribution.

| Metric | Result |
|---|---|
| AUC (ROC) | 0.98+ |
| Validation | 10-fold CV × 3 repeats |
| Ensemble size | 500 trees |
| Dataset size | ~7,000 customer records |

---

## Business context

The model answers three operational questions:

1. Which customers are most likely to churn within the next billing cycle?
2. What behavioural and contractual signals drive attrition?
3. Where should retention spend be concentrated to maximise ROI?

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | Telco customer dataset (IBM sample) |
| Target variable | `Churn.Label` — binary (Yes / No) |
| Feature categories | Demographics, billing, service subscriptions, contract type, tenure |
| Raw feature count | 20+ columns before cleaning |
| Class distribution | Imbalanced — ~26% churn, ~74% retained |

---

## ML pipeline

### 1. Standardisation & type coercion
Column names normalised; dataset cast to `data.frame`; target variable factored explicitly.

### 2. Data leakage removal
Dropped `Customer.Status`, `Churn.Score`, `Churn.Category`, `Churn.Reason`, `Customer.ID` — all carry post-hoc information unavailable at inference time.

### 3. Feature reduction
Removed geographic identifiers (City, Country, Zip, coordinates). Low signal-to-noise ratio.

### 4. Missing value imputation
`Total.Charges` NAs filled with column median — robust to outlier skew, preserves distributional shape.

### 5. Categorical encoding
All character columns converted to factors for correct Random Forest split behaviour.

### 6. Class imbalance correction
ROSE sampling applied *inside* each CV fold — never on the full dataset. Prevents synthetic samples from leaking into validation folds.

> **Note:** All ROSE sampling is performed within cross-validation folds to eliminate synthetic data leakage. This is a deliberate design choice that distinguishes a production-safe pipeline from a naive one.

### 7. Model — Random Forest
Selected for robustness to feature interactions, native handling of mixed types, resistance to overfitting via bagging, and built-in variable importance ranking.

### 8. Cross-validation strategy
10-fold CV repeated 3 times (30 total fits). Reduces variance in the performance estimate.

### 9. Evaluation metric — ROC-AUC
Chosen over accuracy because the target is imbalanced. AUC measures rank-ordering ability across all thresholds, making it threshold-agnostic.

---

## Visual outputs & interpretation

| Visual | What it shows | Why it matters |
|---|---|---|
| Churn distribution | Class imbalance (~26% vs ~74%) | Motivates ROSE strategy; naive accuracy would be misleading |
| ROC curve | True positive rate vs false positive rate across thresholds | AUC ≈ 0.98+ confirms strong discrimination ability |
| Confusion matrix | TP / TN / FP / FN breakdown | False negatives (missed churners) are the costliest error |
| Confusion matrix heatmap | Colour-encoded prediction volume | Faster stakeholder interpretation |
| Feature importance plot | Variables ranked by mean Gini decrease | Identifies actionable retention levers |
| Top 15 features bar chart | Highest-signal 15 variables | Focuses strategy on empirically validated drivers |

---

## Key findings

Customers most likely to churn share the following profile:

- Month-to-month contract (vs. annual or two-year)
- High monthly charges relative to tenure
- Short customer tenure (< 12 months)
- Fibre optic internet without additional support subscriptions

---

## How to run

### Dependencies

```r
install.packages(c(
  "tidyverse",
  "caret",
  "randomForest",
  "pROC",
  "ROSE",
  "e1071"
))
```

### Steps

1. Place `telco.csv` in your R working directory.
2. Source the main script: `source("churn_model.R")`
3. Plots render automatically; model object is assigned to `rf_model`.
4. Inspect `varImp(rf_model)` for ranked feature importance.

---

## Potential extensions

- Gradient boosting comparison (XGBoost / LightGBM)
- SHAP values for local prediction explainability
- Precision–recall threshold optimisation for business-specific cost functions
- REST API deployment via Plumber (R) or FastAPI (Python port)
- Power BI / Tableau dashboard for non-technical stakeholder consumption

---

*Author: Katlego Mathebula · Stack: R · caret · randomForest · pROC · ROSE · tidyverse*
