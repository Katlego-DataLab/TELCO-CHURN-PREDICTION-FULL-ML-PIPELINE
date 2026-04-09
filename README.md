# Telco Customer Churn Prediction

End-to-end machine learning pipeline for predicting customer attrition using Random Forest in R

**Author:** Katlego Mathebula

**Tech Stack:** R, caret, randomForest, pROC, ROSE, tidyverse

**Project Type:** Supervised Learning (Binary Classification)

---

## Overview

Customer churn is one of the most costly operational challenges in the telecommunications industry. This project implements a production-style machine learning pipeline to predict customer attrition using a Random Forest model.

The solution is designed with a strong focus on:

* Data leakage prevention
* Cross-validation for reliable performance
* Class imbalance handling
* Model interpretability and business insight generation

---

## Key Results

* **AUC (ROC):** 0.987
* **Validation Strategy:** 10-fold cross-validation repeated 3 times
* **Model:** Random Forest (500 trees)
* **Dataset Size:** Approximately 7,000 customer records

The model demonstrates excellent predictive performance and strong separation between churn and non-churn customers.

---

## Business Context

This model addresses the following key business questions:

* Which customers are most likely to churn in the near future?
* What behavioural and contractual factors drive churn?
* How can retention efforts be prioritised to maximise return on investment?

---

## Dataset

* **Source:** Telco customer dataset 
* **Target Variable:** `Churn.Label` (Yes / No)

### Feature Categories:

* Demographics
* Billing information
* Service subscriptions
* Contract type
* Customer tenure

### Data Characteristics:

* More than 20 features before preprocessing
* Imbalanced target variable:

  * Approximately 26% churn
  * Approximately 74% retained

---

## Machine Learning Pipeline

### 1. Data Preparation

* Standardised column names
* Converted dataset to `data.frame`
* Defined and encoded target variable

---

### 2. Data Leakage Prevention

Removed the following variables:

* Customer.Status
* Churn.Score
* Churn.Category
* Churn.Reason
* Customer.ID

These variables contain post-event information and would result in unrealistic model performance if included.

---

### 3. Feature Reduction

Removed geographic features:

* City, Country, Zip Code, Latitude, Longitude

These variables contribute little predictive value and increase model complexity.

---

### 4. Missing Value Handling

* Applied median imputation to `Total.Charges`

Median was chosen due to its robustness to outliers and ability to preserve distribution.

---

### 5. Feature Engineering

* Converted all categorical variables to factors

This ensures correct handling of categorical splits within the Random Forest model.

---

### 6. Class Imbalance Handling

* Applied ROSE sampling within cross-validation folds

This prevents bias toward the majority class and avoids data leakage.

---

### 7. Model Selection

Random Forest was selected because it:

* Captures nonlinear relationships
* Reduces overfitting through bagging
* Handles mixed data types effectively
* Provides feature importance measures

---

### 8. Training Strategy

* 10-fold cross-validation
* Repeated 3 times
* 500 trees

This ensures stable and generalisable performance.

---

### 9. Evaluation Metric

**Primary Metric:** ROC-AUC

ROC-AUC was chosen because:

* It is appropriate for imbalanced datasets
* It evaluates ranking performance across thresholds
* It is more reliable than accuracy in this context

---

## Model Interpretation and Insights

### ROC Curve

The ROC curve lies close to the top-left corner, indicating excellent model performance.
An AUC of 0.987 implies a 98.7% probability that the model ranks a churned customer higher than a retained customer.

---

### Confusion Matrix

* True Positives: 361
* False Negatives: 12

The model correctly identifies 96.8% of churned customers, minimising missed churn cases, which are the most costly errors.
![Confusion Matrix](https://github.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/blob/main/02_confusion_matrix.png)
---

### Feature Importance

The most influential predictors include:

* Satisfaction Score
* Number of Referrals
* Contract Type
* Online Security
* Tenure
* Monthly Charges

These variables are key drivers of customer behaviour and churn risk.

---

### Customer Risk Segmentation

* Approximately 4,600 customers classified as Low Risk
* Approximately 2,050 customers classified as High Risk
* Approximately 280 customers classified as Medium Risk

The high-risk segment represents a clear target group for retention strategies.

---

### Predicted Probability Distribution

The model produces a bimodal distribution of churn probabilities, with most predictions concentrated near 0% or 100%.

This indicates strong confidence in predictions and minimal ambiguity.

---

### Accuracy Overview

* Approximately 6,700 correct predictions
* Approximately 500 incorrect predictions
* Overall accuracy around 93%

However, ROC-AUC and recall are more important due to class imbalance.

---

### Churn by Contract Type

Customers on month-to-month contracts exhibit significantly higher churn rates compared to those on annual or multi-year contracts.

This confirms contract structure as a key lever for improving retention.

---

## Key Findings

Customers most likely to churn typically:

* Have low satisfaction scores
* Have made few or no referrals
* Are on month-to-month contracts
* Do not use online security services
* Have short tenure (less than 12 months)

---

## Business Impact

This model enables organisations to:

* Identify high-risk customers before churn occurs
* Target retention strategies effectively
* Optimise marketing spend
* Improve customer lifetime value

---

## How to Run

### Install Dependencies

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

1. Place `telco.csv` in the working directory
2. Run the main script
3. Review outputs and model results

---



## Conclusion

This project demonstrates a complete, production-oriented machine learning workflow:

* Clean and structured data pipeline
* Leakage-aware modeling
* Robust validation strategy
* Business-focused interpretation

The solution goes beyond prediction by providing actionable insights that can directly support customer retention strategies.

