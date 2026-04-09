# 📡 Telco Customer Churn Prediction — Full ML Pipeline

> **End-to-end Random Forest pipeline for predicting customer attrition in telecommunications, built in R with a strong focus on data leakage prevention, class imbalance handling, and business-actionable insight.**

**Author:** Katlego Mathebula  
**Tech Stack:** R · caret · randomForest · pROC · ROSE · tidyverse  
**Project Type:** Supervised Binary Classification  
**Dataset:** ~7,000 customer records · 20+ features

---

##  Key Results at a Glance

| Metric | Value |
|---|---|
| **AUC (ROC)** | **0.987** |
| **Overall Accuracy** | ~93% |
| **Recall (Sensitivity)** | **96.8%** —>correctly identified churners |
| **True Positives** | 361 churned customers correctly flagged |
| **False Negatives** | 12 churned customers missed |
| **Validation Strategy** | 10-fold CV × 3 repeats (30 total folds) |
| **Model** | Random Forest · 500 trees |
| **Dataset Size** | ~7,000 customer records |

> **What does AUC = 0.987 mean in practice?** If you randomly pick one churned customer and one retained customer from your database, this model correctly ranks the churner as higher-risk **98.7% of the time**. This is exceptionally high discrimination power.

---

##  Business Context

Customer churn is one of the most expensive operational challenges in telecommunications. Industry estimates suggest that acquiring a new customer costs 5–7× more than retaining an existing one. This project provides a proactive, data-driven solution to identify at-risk customers before they leave.

### Business Questions Addressed

- **Who** is most likely to churn in the near future?
- **Why** are customers leaving — which behavioural and contractual factors drive attrition?
- **How** can retention efforts be prioritised to maximise return on investment?

---

##  Dataset

| Property | Detail |
|---|---|
| **Source** | Telco customer dataset |
| **Records** | ~7,000 customers |
| **Features** | 20+ before preprocessing |
| **Target Variable** | `Churn.Label` — `Yes` / `No` |
| **Class Distribution** | ~26% churn (Yes) · ~74% retained (No) |

### Feature Categories

| Category | Examples |
|---|---|
| Demographics | Gender, Age, Dependents |
| Contract & Billing | Contract Type, Monthly Charges, Total Charges, Payment Method |
| Service Subscriptions | Online Security, Tech Support, Streaming TV/Movies |
| Account Behaviour | Tenure, Number of Referrals, Satisfaction Score |

---

##  Machine Learning Pipeline

### Pipeline Architecture

```
Raw Data (telco.csv)
        │
        ▼
┌─────────────────────────────┐
│  1. Data Preparation        │  Standardise names, convert to data.frame
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  2. Leakage Prevention      │  Remove post-event variables (Churn.Score,
└────────────────┬────────────┘  Churn.Reason, Customer.Status, etc.)
                 │
                 ▼
┌─────────────────────────────┐
│  3. Feature Reduction       │  Drop geographic identifiers (City, Lat/Long)
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  4. Missing Value Imputation│  Median imputation on Total.Charges
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  5. Feature Engineering     │  Convert character → factor for RF splits
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  6. Near-Zero Variance      │  Remove uninformative predictors via nearZeroVar()
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  7. Stratified CV + ROSE    │  10-fold × 3 repeats; ROSE applied per fold
└────────────────┬────────────┘  (prevents data leakage from oversampling)
                 │
                 ▼
┌─────────────────────────────┐
│  8. Random Forest Training  │  500 trees, optimised by ROC-AUC
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  9. Evaluation & Insights   │  ROC Curve, Confusion Matrix, Feature Importance,
└─────────────────────────────┘  Risk Segmentation, Probability Distribution
```

---

### Step-by-Step Decisions

#### 1. Data Leakage Prevention
The following variables were removed before modelling — they are only observable **after** a customer has already churned and would cause artificially inflated performance:

| Variable Removed | Reason |
|---|---|
| `Customer.Status` | Direct encoding of churn outcome |
| `Churn.Score` | Derived post-event risk score |
| `Churn.Category` | Post-churn classification label |
| `Churn.Reason` | Customer's stated reason for leaving |
| `Customer.ID` | Identifier with no predictive value |

> **Without this step**, AUC would approach 1.0 due to data leakage — not genuine predictive power. Leakage-aware modelling ensures the 0.987 AUC is a realistic estimate of live performance.

#### 2. Class Imbalance Handling with ROSE
With a 74/26 split, naïve models tend to predict "No Churn" for everyone — yielding high accuracy but near-zero recall on the minority class.

**Solution:** ROSE (Random Over-Sampling Examples) was applied **inside** each cross-validation fold — not on the full dataset before splitting. This is critical: applying oversampling before CV would cause synthetic samples to leak across train/validation boundaries, artificially inflating performance estimates.

| Class | Count (approx.) |
|---|---|
| No Churn | ~5,200 |
| Churn | ~1,800 |

#### 3. Why Random Forest?

| Property | Benefit for This Problem |
|---|---|
| Bagging (bootstrap aggregation) | Reduces variance; robust to overfitting |
| Nonlinear splits | Captures complex churn patterns (e.g., interaction between tenure and contract type) |
| Mixed data types | Handles both numeric (Monthly Charges, Tenure) and categorical (Contract Type, Online Security) natively via factor encoding |
| Feature importance | Provides interpretable ranking of churn drivers |
| 500 trees | Stabilises OOB error estimates and importance scores |

#### 4. Validation Strategy
- **10-fold cross-validation** — splits the dataset into 10 equal parts; trains on 9, validates on 1, rotating through all folds
- **Repeated 3 times** — generates 30 unique train/validation pairs for stable performance estimates
- **Primary metric: ROC-AUC** — appropriate for imbalanced datasets; unlike accuracy, AUC evaluates ranking performance across all possible classification thresholds

---

##  Model Performance & Visualisations

### ROC Curve

```
AUC = 0.987
```

The ROC curve hugs the top-left corner of the plot — the ideal position. An AUC of 0.987 means the model can almost perfectly distinguish between a customer who will churn and one who will not, across all decision thresholds.

![ROC Curve](https://github.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/blob/main/01_roc_curve.png)

---

### Confusion Matrix

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | True Negatives | False Positives |
| **Actual: Yes** | **12** (False Negatives) | **361** (True Positives) |

- **Recall (Sensitivity) = 96.8%** — the model correctly flags 361 out of 373 churned customers
- **False Negatives = 12** — only 12 churned customers were missed; these are the most costly errors in retention contexts (a missed churner = lost revenue with no intervention)
- **Overall Accuracy ≈ 93%** — across ~7,000 predictions, ~6,700 were correct

> In a business context, **recall is the priority metric** here. A false negative (missing a churner) costs far more than a false positive (unnecessary retention outreach).

![Confusion Matrix](https://github.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/blob/main/02_confusion_matrix.png)

---

### Feature Importance — Top Predictors

| Rank | Feature | Business Interpretation |
|---|---|---|
| 1 | **Satisfaction Score** | Low-satisfaction customers are already disengaged — a leading indicator of churn |
| 2 | **Number of Referrals** | Customers who refer others are invested in the service; zero referrals signals detachment |
| 3 | **Contract Type** | Month-to-month contracts carry no switching cost; annual/2-year contracts create lock-in |
| 4 | **Online Security** | Customers with security add-ons are more embedded in the service ecosystem |
| 5 | **Tenure** | Newer customers (<12 months) haven't yet built loyalty or switching friction |
| 6 | **Monthly Charges** | High monthly costs increase churn sensitivity, especially without perceived value |

![Feature Importance](https://github.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/blob/main/03_feature_importance.png)

---

### Customer Risk Segmentation

| Risk Level | Customer Count | Action |
|---|---|---|
| 🔴 **High Risk** | ~2,050 | Immediate retention intervention |
| 🟡 **Medium Risk** | ~280 | Monitor; proactive value reinforcement |
| 🟢 **Low Risk** | ~4,600 | Loyalty programmes; upsell opportunities |

> The **high-risk segment (~29% of customer base)** represents the primary target for retention spend. Concentrating resources here maximises ROI versus blanket outreach campaigns.

![Risk Segmentation](https://github.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/blob/main/04_risk_level_distribution.png)

---

### Predicted Probability Distribution

The model produces a **strongly bimodal distribution** — the majority of customers receive predicted churn probabilities near 0% or near 100%, with few cases in the ambiguous 30–70% range.

This bimodal pattern indicates:
- **High model confidence** across most predictions
- **Clear separability** between churn and non-churn classes
- **Minimal ambiguity** — the model is not hedging; it commits to clear predictions

![Probability Distribution](https://github.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/blob/main/05_probability_band_distribution.png)

---

### Churn by Contract Type

Month-to-month customers churn at significantly higher rates than those on annual or two-year contracts. This confirms that **contract structure is a primary lever** for reducing churn — converting high-risk month-to-month customers to longer-term agreements should be a key retention strategy.

---

##  Key Findings

Customers most likely to churn share a consistent profile:

- ❗ **Low satisfaction score** — the single strongest predictor
- ❗ **Zero or few referrals** — no social investment in the service
- ❗ **Month-to-month contract** — no structural switching barrier
- ❗ **No online security subscription** — less embedded in the service ecosystem
- ❗ **Short tenure (< 12 months)** — haven't reached the loyalty threshold
- ❗ **High monthly charges** — cost sensitivity without perceived commensurate value

---

##  Business Impact

| Use Case | How the Model Enables It |
|---|---|
| **Proactive Retention** | Flag high-risk customers 30–60 days before predicted churn for targeted outreach |
| **Retention Budget Allocation** | Prioritise spend on ~2,050 high-risk customers rather than broadcasting across all 7,000 |
| **Contract Conversion Campaigns** | Target month-to-month customers with contract upgrade offers |
| **Service Bundling Strategy** | Promote online security and other add-ons to customers showing early churn signals |
| **Customer Lifetime Value Optimisation** | Extend tenure of high-value customers by intervening before satisfaction scores drop |

---

##  How to Run

### 1. Install Dependencies

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

### 2. Prepare Data

Place `telco.csv` in your working directory. The dataset should contain customer records with the `Churn.Label` column as the target variable.

### 3. Run the Pipeline

```r
source("telco_churn_pipeline.R")
```

### 4. Outputs

The script produces the following artefacts:

| Output | Description |
|---|---|
| `01_roc_curve.png` | ROC curve with AUC annotation |
| `02_confusion_matrix.png` | Heatmap of confusion matrix |
| `03_feature_importance.png` | Top 10 predictors by importance |
| `04_risk_level_distribution.png` | Customer count by risk tier |
| `05_probability_band_distribution.png` | Bimodal churn probability distribution |
| Console output | CV AUC, best `mtry`, confusion matrix statistics |

---

##  Project Structure

```
├── telco_churn_pipeline.R     # Main modelling script
├── telco.csv                  # Input dataset (not included — see Dataset section)
├── 01_roc_curve.png
├── 02_confusion_matrix.png
├── 03_feature_importance.png
├── 04_risk_level_distribution.png
├── 05_probability_band_distribution.png
└── README.md
```

---

##  Technical Notes

- `set.seed(123)` is applied globally and before fold creation to ensure full reproducibility of all results
- `nearZeroVar()` removes predictors with near-zero variance before training — these add noise without predictive signal
- Median imputation was chosen for `Total.Charges` over mean imputation due to its robustness to right-skewed billing distributions
- All character columns are converted to factors before training, ensuring correct handling of categorical splits within the Random Forest algorithm
- ROSE sampling is applied within the `trainControl` custom wrapper — not pre-split — to prevent synthetic minority samples from appearing in validation folds

---

## Conclusion

This project demonstrates a complete, production-oriented machine learning workflow — not just a model, but a disciplined pipeline that addresses real-world concerns:

- ✅ Data leakage prevention through careful variable exclusion
- ✅ Class imbalance handling without contaminating validation estimates
- ✅ Robust generalisation through repeated cross-validation
- ✅ Business-focused interpretation that goes beyond accuracy metrics

The result is a model that is both statistically rigorous and operationally actionable — capable of identifying which customers to target, why they are at risk, and how retention resources should be allocated.

---

*Built by Katlego Mathebula*
