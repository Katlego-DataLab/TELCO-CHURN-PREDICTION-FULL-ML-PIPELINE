# Telco Customer Churn Prediction – Full Machine Learning Pipeline (R)

*Author:* Katlego Mathebula
*Tech Stack:* R · caret · randomForest · pROC · ROSE · tidyverse
*Project Type:* Supervised Learning (Binary Classification)
*Model:* Random Forest with 10-Fold Cross-Validation

## Executive Summary

Customer churn is one of the most critical problems in the telecommunications industry. Acquiring new customers is significantly more expensive than retaining existing ones.

This project builds a **production-style machine learning pipeline** to predict customer churn using Random Forest in R. The pipeline includes:

- Data leakage prevention
- Feature engineering
- Missing value handling
- Class imbalance correction
- 10-fold cross-validation
- ROC-AUC optimization
- Feature importance analysis

The model is optimized to identify high-risk customers, enabling businesses to take proactive retention actions.


## Business Problem

Telecom companies lose revenue when customers cancel their subscriptions.

Key business questions:

* Which customers are most likely to churn?
* What behavioral patterns signal churn risk?
* Which features drive customer attrition?
* How can we prioritize retention efforts?

This project simulates a real-world churn modeling workflow used by telecom analytics teams.


## Dataset Overview

File: `telco.csv`
Target Variable: `Churn.Label` (Yes / No)

The dataset includes:

* Demographics
* Account information
* Service usage
* Billing data
* Contract type
* Tenure

The objective is binary classification:

> Predict whether a customer will churn (Yes) or stay (No).

## Full Machine Learning Pipeline

This project follows a structured ML workflow rather than ad-hoc modeling.


## 1️. Data Preparation

- Loaded required modeling libraries
- Converted tibble to data.frame (compatibility with caret)
- Standardized column names using `make.names()`
- Converted target variable to factor
- Explicitly set `"Yes"` as the positive class

Setting the positive class is critical for correct ROC-AUC interpretation.


## 2️. Data Leakage Prevention

One of the most important modeling steps.

Removed columns that directly reveal churn outcome:

- `Customer.Status`
- `Churn.Score`
- `Churn.Category`
-  `Churn.Reason`
-  `Customer.ID`

Why this matters:

Including leakage features would artificially inflate model performance and make it unusable in production.

This demonstrates understanding of real-world ML pitfalls.


## 3️. Feature Reduction

Removed low-value geographic variables:

-  Country
-  State
-  City
- Zip Code
-  Latitude
-  Longitude

Reason:

These features provide little predictive power and may introduce noise or unnecessary dimensionality.

## 4️. Missing Value Handling

Handled missing values in `Total.Charges` using median imputation:

```r
median(telco$Total.Charges, na.rm = TRUE)
```

Median was chosen because:

-  It is robust to outliers
-  It preserves the distribution shape
-  It avoids bias introduced by mean imputation


## 5️. Feature Engineering

Converted all character variables to factors.

This ensures:

- Proper categorical handling
-  Correct splitting in tree-based models
- Improved interpretability



## 6️. Class Imbalance Handling

Churn datasets are typically imbalanced.

Checked class distribution:

```r
table(telco$Churn.Label)
```

Applied ROSE sampling inside cross-validation folds:

```r
sampling = "rose"
```

Why this is important:

-  Prevents bias toward the majority class
-  Balances classes during training
-  Maintains realistic evaluation

Balancing was applied within folds — not before splitting — to prevent data leakage.


## 7️. Model Training Strategy

### Algorithm: Random Forest

Random Forest was selected because:

-  Handles nonlinear relationships
-  Reduces overfitting through bagging
-  Performs well with mixed feature types
-  Provides feature importance scores

### Cross-Validation Setup

-  10-fold Cross-Validation
-  Class probabilities enabled
-  Optimized for ROC-AUC
-  500 trees (`ntree = 500`)

Cross-validation ensures robust performance estimation and reduces variance.


## 8️. Model Optimization Metric

The model was optimized using:

> ROC-AUC (Area Under the Receiver Operating Curve)

Why ROC-AUC?

-  Measures ranking ability
-  Independent of classification threshold
-  Appropriate for imbalanced datasets

This is more reliable than raw accuracy.


## 9️. Model Evaluation

Extracted:

-  Best tuning parameters
-  Cross-validated ROC score
-  Saved fold predictions
-  Variable importance

Performance reflects average validation across 10 folds — not a single split.


##  Feature Importance

Used:

```r
varImp(rf_cv_model)
```

This identifies:

-  Most influential churn predictors
-  Key drivers of customer attrition
-  Variables retention teams should monitor

Feature importance translates ML output into business insight.


## Business Insights Enabled

This model allows organizations to:

-  Identify high-risk customers
-  Target retention campaigns
-  Offer discounts strategically
-  Reduce churn rate
-  Increase customer lifetime value

The output can integrate into CRM systems for proactive intervention.


## Why This Project Demonstrates  ML Skills

This project shows:

-  Structured ML pipeline design
-  Data leakage awareness
-  Proper cross-validation
-  Imbalanced learning strategy
-  ROC-based optimization
-  Feature importance interpretation
-  Business-driven modeling

## How to Run the Project

### Install Required Packages

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

1. Place `telco.csv` in working directory
2. Run the script
3. Review cross-validated ROC
4. Analyze variable importance


## Potential Improvements

Future enhancements could include:

-  Hyperparameter grid tuning
-  Gradient Boosting comparison
-  XGBoost implementation
-  Threshold optimization
-  Precision-Recall analysis
-  Deployment using Plumber API
-  Model explainability using SHAP



## Conclusion

This project simulates a real-world churn modeling workflow used in telecom analytics.

It demonstrates:

-  Strong statistical understanding
-  Production-aware ML thinking
- Risk modeling discipline
- Business-oriented problem solving

The focus was not just predictive performance but building a reliable, leakage-free, cross-validated pipeline suitable for real deployment.

