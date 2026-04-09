Telco Customer Churn Prediction
End-to-end machine learning pipeline for predicting customer attrition using Random Forest in R

R
Random Forest
caret
ROSE
pROC
tidyverse
Production-style
Overview
Customer churn is among the most costly operational challenges in telecommunications. This project implements a zero-leakage, cross-validated churn classification system using Random Forest. The pipeline covers preprocessing, imbalance correction, model training, evaluation, and feature attribution.

AUC (ROC)
0.987
10-fold CV × 3 repeats
Ensemble size
500
decision trees (ntree)
Records
~7 000
telco customer rows
Business context
The model answers three operational questions:

Which customers are most likely to churn within the next billing cycle?
What behavioural and contractual signals drive attrition?
Where should retention spend be concentrated to maximise ROI?
Dataset
Attribute	Detail
Source	Telco customer dataset (IBM sample)
Target variable	Churn.Label — binary (Yes / No)
Feature categories	Demographics, billing, service subscriptions, contract type, tenure
Raw feature count	20+ columns before cleaning
Class distribution	Imbalanced — ~26% churn, ~74% retained
ML pipeline
1
Standardisation & type coercion
Column names normalised; dataset cast to data.frame; target variable factored. Ensures downstream caret compatibility.
2
Data leakage removal
Dropped Customer.Status, Churn.Score, Churn.Category, Churn.Reason, Customer.ID — all carry post-hoc information unavailable at inference time.
3
Feature reduction
Removed geographic identifiers (City, Country, Zip, coordinates). Low signal-to-noise ratio; increases dimensionality without predictive benefit.
4
Missing value imputation
Total.Charges NAs filled with column median — robust to outlier skew, preserves distributional shape.
5
Categorical encoding
All character columns converted to factors. Random Forest requires factor inputs for correct split behaviour on nominal variables.
6
Class imbalance correction
ROSE sampling applied inside each CV fold — never on the full dataset. Prevents synthetic samples from leaking into validation folds.
7
Model training — Random Forest
Handles nonlinear relationships, reduces overfitting via bagging, works with mixed data types, provides built-in variable importance ranking.
8
Cross-validation strategy
10-fold CV repeated 3 times (30 total fits). Reduces variance in performance estimate; reliable proxy for generalisation on unseen data.
9
Evaluation — ROC-AUC
Chosen over accuracy because the target is imbalanced. AUC measures rank-ordering ability across all thresholds — threshold-agnostic and appropriate for imbalanced binary problems.
All ROSE sampling is performed within cross-validation folds to eliminate synthetic data leakage. This is a deliberate design choice that distinguishes a production-safe pipeline from a naive one.
Visual outputs & interpretation
1. ROC curve — AUC = 0.987
The curve hugs the top-left corner almost immediately, confirming the model separates churners from non-churners with near-perfect ranking ability across all decision thresholds. AUC of 0.987 means there is a 98.7% chance the model scores a random churner higher than a random non-churner.
01_roc_curve.png
2. Confusion matrix heatmap
TN = 945, TP = 361, FP = 89, FN = 12. Only 12 churners were missed (false negatives) — the costliest error type. The model correctly caught 361 of 373 actual churners, a recall of 96.8% on the positive class.
02_confusion_matrix.png
3. Top 10 feature importance
Satisfaction Score dominates all other predictors by a wide margin — unhappy customers are the primary churn signal. Number of Referrals is the second strongest driver (engaged customers refer others; disengaged customers don't). Contract type and Online Security follow, confirming that product stickiness and commitment level are strong retention levers. Tenure, charges, and revenue variables are present but carry substantially lower relative importance.
03_feature_importance.png
4. Customer risk level distribution
~4,600 customers are Low Risk, ~2,050 are High Risk, and ~280 are Medium Risk. The large High Risk segment (~28% of base) represents the immediate intervention target — a meaningful and actionable population for retention campaigns.
04_risk_level_distribution.png
5. Predicted churn probability distribution
The bimodal distribution — concentrated at 0–10% and 90–100% — confirms the model produces confident, well-separated predictions rather than uncertain mid-range probabilities. Very few customers fall in the ambiguous 30–70% range, making the output actionable at almost any threshold.
05_probability_band_distribution.png
6. Prediction accuracy breakdown
~6,700 correct predictions vs ~500 incorrect. Overall accuracy is approximately 93%, but this metric is secondary — AUC and recall on the churn class are the operationally critical measures for an imbalanced classification problem.
06_prediction_accuracy.png
7. High-value customers by risk level
Approximately 2,050 standard-segment customers are flagged as High Risk — the priority group for proactive outreach. The Low Risk majority (~4,600) can be maintained with lighter-touch engagement programmes to reduce retention cost.
07_high_value_vs_risk.png
8. Actual churn by contract type
Month-to-Month customers churn at a dramatically higher rate than One Year or Two Year customers — nearly 1 in 2 month-to-month customers churned, compared to roughly 1 in 9 on annual contracts and far fewer on two-year agreements. This directly validates contract type as a key retention lever: migrating customers from monthly to longer-term contracts is one of the highest-ROI actions the business can take.
08_churn_by_contract.png
The bimodal churn probability distribution (chart 5) and the AUC of 0.987 (chart 1) together confirm that this is a confident, deployment-ready model — not a borderline classifier.
Key findings
Customers most likely to churn share the following profile:

Low satisfaction score (the single strongest predictor)
Zero or very few referrals made
Month-to-month contract
No online security subscription
Short customer tenure (< 12 months)
How to run
Dependencies
install.packages(c(
  "tidyverse",
  "caret",
  "randomForest",
  "pROC",
  "ROSE",
  "e1071"
))
Steps
Place telco.csv in your R working directory.
Source the main script: source("churn_model.R")
All 8 plots render automatically to the Plots pane.
Inspect varImp(rf_model) for ranked feature importance.
Potential extensions
Gradient boosting comparison (XGBoost / LightGBM)
SHAP values for local prediction explainability
Precision–recall threshold optimisation for business-specific cost functions
REST API deployment via Plumber (R) or FastAPI (Python port)
Power BI / Tableau dashboard for non-technical stakeholder consumption
Author: Katlego Mathebula  ·  Stack: R · caret · randomForest · pROC · ROSE · tidyverse
