# Telco Customer Churn Prediction

End-to-end machine learning pipeline for predicting customer attrition using Random Forest in R.

## Overview
Customer churn is among the most costly operational challenges in telecommunications. This project implements a zero-leakage, cross-validated churn classification system using Random Forest.

| Metric | Result |
|---|---|
| AUC (ROC) | 0.987 |
| Validation | 10-fold CV × 3 repeats |
| Ensemble size | 500 trees |
| Dataset size | ~7,000 customer records |

## Business context
1. Which customers are most likely to churn within the next billing cycle?
2. What behavioural and contractual signals drive attrition?
3. Where should retention spend be concentrated to maximise ROI?

## ML pipeline
1. **Standardisation & type coercion** — Column names normalised; dataset cast to data.frame
2. **Data leakage removal** — Dropped Customer.Status, Churn.Score, Churn.Category, Churn.Reason, Customer.ID
3. **Feature reduction** — Removed geographic identifiers (low signal-to-noise)
4. **Missing value imputation** — Total.Charges NAs filled with column median
5. **Categorical encoding** — Character columns converted to factors
6. **Class imbalance correction** — ROSE sampling applied inside each CV fold (never on full dataset)
7. **Model — Random Forest** — 500 trees, handles mixed types, provides variable importance
8. **Cross-validation** — 10-fold CV × 3 repeats (30 total fits)
9. **Evaluation — ROC-AUC** — Threshold-agnostic, appropriate for imbalanced binary problems

> All ROSE sampling is performed within cross-validation folds. This eliminates synthetic data leakage and distinguishes a production-safe pipeline from a naive one.

## Visual outputs & interpretation

### 1. ROC curve (AUC = 0.987)
The curve hugs the top-left corner almost immediately. AUC of 0.987 means there is a 98.7% chance the model scores a random churner higher than a random non-churner.

### 2. Confusion matrix heatmap
TN = 945, TP = 361, FP = 89, FN = 12. Only 12 churners were missed — recall of 96.8% on the positive class.

### 3. Feature importance
Satisfaction Score dominates all other predictors. Number of Referrals is the second strongest driver. Contract type, Online Security, and Tenure follow.

### 4. Customer risk level distribution
~4,600 Low Risk, ~2,050 High Risk, ~280 Medium Risk. The High Risk segment represents the immediate intervention target.

### 5. Predicted churn probability distribution
Bimodal distribution (concentrated at 0-10% and 90-100%) confirms confident, well-separated predictions — actionable at almost any threshold.

### 6. Prediction accuracy breakdown
~6,700 correct vs ~500 incorrect. Overall accuracy ~93%, but AUC and recall are the operationally critical measures.

### 7. High-value customers by risk level
~2,050 Standard-segment customers flagged as High Risk — the priority group for proactive outreach.

### 8. Churn by contract type
Month-to-Month customers churn at nearly 1-in-2 rate. Two Year customers are near-immune. Migrating customers to longer contracts is the highest-ROI retention action.

## Key findings
- Low satisfaction score (single strongest predictor)
- Zero or very few referrals
- Month-to-month contract
- No online security subscription
- Short tenure (< 12 months)

## How to run

```r
install.packages(c("tidyverse","caret","randomForest","pROC","ROSE","e1071"))
```

1. Place telco.csv in your R working directory
2. source("churn_model.R")
3. All 8 plots render automatically
4. Inspect varImp(rf_model) for ranked feature importance

## Potential extensions
- Gradient boosting comparison (XGBoost / LightGBM)
- SHAP values for local prediction explainability
- Precision-recall threshold optimisation
- REST API deployment via Plumber (R) or FastAPI (Python)
- Power BI / Tableau dashboard

---
*Author: Katlego Mathebula · Stack: R · caret · randomForest · pROC · ROSE · tidyverse*
