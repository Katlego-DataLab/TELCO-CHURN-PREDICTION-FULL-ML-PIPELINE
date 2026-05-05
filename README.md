<div align="center">

# 📡 Telco Customer Churn Prediction
### *Catching customers before they walk out the door*

![R](https://img.shields.io/badge/Built%20with-R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![RandomForest](https://img.shields.io/badge/Model-Random%20Forest-2ECC71?style=for-the-badge)
![AUC](https://img.shields.io/badge/AUC-0.987-FF6B6B?style=for-the-badge)
![Recall](https://img.shields.io/badge/Recall-96.8%25-F39C12?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-9B59B6?style=for-the-badge)

*Built by **Katlego Mathebula** · R · caret · randomForest · ROSE · pROC · tidyverse*

</div>

---

## 🚀 TL;DR

> A production-ready Random Forest pipeline that predicts which telecom customers are about to churn — **before they do** — with **96.8% recall** and **AUC = 0.987**.
>
> Out of 7,000 customers, the model correctly flags **361 of 373 churners** and segments the entire base into actionable risk tiers — so retention teams know exactly who to call, and when.

---

## 💼 The Business Problem

Telecom companies lose billions every year to churn. The painful part? **Most of it is preventable.**

The problem isn't that companies don't care — it's that by the time a customer cancels, it's already too late. Winning them back costs **5–7× more** than simply keeping them.

This project gives a telecom company a **proactive early-warning system**:

- 🔍 **Identify** at-risk customers weeks before they leave
- 🎯 **Understand** *why* they're at risk (contract type, satisfaction, pricing)
- 💰 **Allocate** retention budget where it actually makes a difference

---

## 📊 Key Results

<div align="center">

| Metric | Result | What It Means |
|:---|:---:|:---|
| 🎯 **AUC (ROC)** | **0.987** | Ranks a churner above a retained customer 98.7% of the time |
| 🔔 **Recall** | **96.8%** | Catches 361 of 373 actual churners — only 12 slipped through |
| ✅ **Accuracy** | **~93%** | ~6,700 correct predictions out of 7,000 |
| 🌲 **Model** | Random Forest | 500 trees · 10-fold CV × 3 repeats |
| 📦 **Dataset** | ~7,000 records | 20+ features · 26% churn rate |

</div>

> **⚠️ On AUC = 0.987:** High numbers raise eyebrows — here's why this one is real.
> Post-churn variables (`Churn.Score`, `Churn.Reason`, `Customer.Status`) were **explicitly removed** before modelling — a common mistake that inflates AUC artificially. ROSE oversampling was applied **inside** each CV fold, not before, so no synthetic data contaminated validation. The 0.987 reflects genuine out-of-sample performance.

---

## 🎯 Business Impact

<table>
<tr>
<td width="50%">

**Who uses this?**
- 📞 Retention & CRM teams
- 📈 Marketing (targeted campaigns)
- 💼 Customer Success managers
- 🏦 Finance (revenue forecasting)

</td>
<td width="50%">

**What it enables**
- Trigger outreach **30–60 days** before predicted churn
- Focus budget on ~**2,050 high-risk customers** (not all 7,000)
- Run contract upgrade campaigns on flagged accounts
- Prioritise account reviews by risk score

</td>
</tr>
</table>

### 🗂️ Customer Risk Segments

| Segment | Customers | Recommended Action |
|:---|:---:|:---|
| 🔴 **High Risk** | ~2,050 | Immediate personalised outreach |
| 🟡 **Medium Risk** | ~280 | Proactive check-in & value reinforcement |
| 🟢 **Low Risk** | ~4,600 | Loyalty rewards & upsell opportunities |

> Focusing solely on the 🔴 high-risk group means retention teams work a list of **2,050** — not 7,000. That's **70% less wasted effort** and dramatically higher ROI on every retention dollar spent.

---

## ⚙️ Solution Overview

```
Raw Customer Data (telco.csv)
         │
         ▼
  🧹 Clean & Prep         →  Standardise, impute missing values
         │
         ▼
  🚫 Remove Leakage        →  Drop post-churn variables (Churn.Score, etc.)
         │
         ▼
  🔧 Feature Engineering   →  Convert to factors · remove near-zero variance
         │
         ▼
  ⚖️  Handle Imbalance     →  ROSE applied inside each CV fold
         │
         ▼
  🌲 Train Random Forest   →  500 trees · optimised by AUC
         │
         ▼
  📊 Evaluate & Segment    →  ROC · Confusion Matrix · Risk Tiers
```

**Tech stack:** `R` · `caret` · `randomForest` · `ROSE` · `pROC` · `tidyverse`

---

## 🧠 Key Technical Decisions

**1. 🚫 Leakage Prevention — the most important step**

Variables like `Churn.Score` and `Churn.Reason` are only known *after* a customer leaves. Including them would make the model look perfect — but fail completely in production. Removing them is what makes the 0.987 AUC *meaningful*.

**2. ⚖️ ROSE Inside CV Folds — not before**

Applying oversampling before splitting is a common error that causes synthetic samples to leak into validation sets. Here, ROSE runs *inside* each fold — keeping train and validation data fully independent.

**3. 🔁 Repeated Cross-Validation (10-fold × 3 repeats)**

Generates 30 unique train/validation pairs for stable, reliable performance estimates. ROC-AUC was used as the primary metric — more appropriate than accuracy for imbalanced classes.

**4. 🌲 Why Random Forest?**

Handles mixed data types natively, captures nonlinear interactions (e.g., tenure × contract type), and produces interpretable feature importance scores — critical for explaining churn drivers to non-technical stakeholders.

---

## 🔍 Key Insights

Customers most likely to churn share a consistent profile:

| Signal | Insight |
|:---|:---|
| 😞 **Low satisfaction score** | Strongest predictor — disengagement precedes cancellation |
| 🤐 **Zero referrals** | No social investment = lower switching cost |
| 📅 **Month-to-month contract** | No lock-in = highest churn rate of any segment |
| 🔓 **No Online Security add-on** | Less embedded in the service ecosystem |
| 🆕 **Tenure < 12 months** | Haven't reached the loyalty threshold yet |
| 💸 **High monthly charges** | Cost sensitivity without perceived value |

> **One actionable takeaway:** Converting month-to-month customers to annual contracts is the single highest-leverage retention strategy — it reduces churn risk structurally, without needing to individually outreach every at-risk customer.

---

## 📈 Visualisations

### 📉 ROC Curve — AUC = 0.987
*The curve hugs the top-left corner — near-perfect discrimination between churners and retained customers across all decision thresholds.*

![ROC Curve](https://raw.githubusercontent.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/main/01_roc_curve.png)

---

### 🟦 Confusion Matrix
*361 churners correctly caught · only 12 missed · overall accuracy ~93%*

![Confusion Matrix](https://raw.githubusercontent.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/main/02_confusion_matrix.png)

---

### 🏆 Feature Importance — Top Churn Drivers
*Satisfaction score and referrals dominate — contract type and tenure follow closely.*

![Feature Importance](https://raw.githubusercontent.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/main/03_feature_importance.png)

---

### 🗂️ Customer Risk Distribution
*~29% of the base is high-risk — a focused, actionable retention target.*

![Risk Segmentation](https://raw.githubusercontent.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/main/04_risk_level_distribution.png)

---

### 📊 Predicted Probability Distribution
*Strongly bimodal — the model commits to clear predictions with minimal ambiguity in the middle.*

![Probability Distribution](https://raw.githubusercontent.com/Katlego-DataLab/TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE/main/05_probability_band_distribution.png)

---

## 🏗️ Production Thinking

Here's how this pipeline would run inside a real company:

```
Weekly Batch Job
      │
      ▼
Pull latest customer data from data warehouse
      │
      ▼
Apply same feature transformations as training
      │
      ▼
Score all customers → churn probability (0–1)
      │
      ▼
Segment into High / Medium / Low risk tiers
      │
      ▼
Push to CRM (Salesforce / HubSpot)
      │
      ▼
Trigger retention workflows:
  🔴 High   → Personal outreach within 48h
  🟡 Medium → Automated value email sequence
  🟢 Low    → Loyalty programme nudge
      │
      ▼
Log outcomes → retrain model quarterly
```

**Monitoring triggers:**
- AUC drops below 0.93 → flag for review
- Churn rate shifts >3% month-over-month → investigate feature drift
- False negative rate increases → lower decision threshold

---

## 💡 Future Improvements

| Improvement | Why It Matters |
|:---|:---|
| 🖥️ **Shiny / Streamlit app** | Make risk scores accessible to non-technical teams |
| 🔄 **XGBoost / LightGBM comparison** | Potentially higher performance; faster inference |
| 💬 **SHAP explanations** | Per-customer "why is this person high risk?" narrative |
| 📡 **Real-time scoring API** | Score customers on demand vs. weekly batch |
| 📊 **Power BI / Tableau integration** | Executive-level churn dashboards |

---

## 🖥️ Interactive App Concept

> **The next step: turn this into a tool anyone can use — no code required.**

A **Shiny dashboard** would give retention managers a live interface to:

- 🔎 **Look up any customer** → see their churn probability + risk tier instantly
- 🎛️ **Run what-if scenarios** → *"What if we move this customer to an annual contract?"* → watch the risk score update in real time
- 📋 **Browse the retention queue** → filterable table of high-risk customers, sorted by churn probability
- 📊 **Explore churn drivers** → visual breakdown of *why* specific customers are flagged

```r
# Shiny app skeleton
library(shiny); library(randomForest)

ui <- fluidPage(
  titlePanel("🔴 Churn Risk Dashboard"),
  sidebarLayout(
    sidebarPanel(
      selectInput("contract", "Contract Type",
                  choices = c("Month-to-month", "One year", "Two year")),
      sliderInput("satisfaction", "Satisfaction Score", 1, 5, 3),
      sliderInput("tenure", "Tenure (months)", 0, 72, 12),
      sliderInput("charges", "Monthly Charges ($)", 20, 120, 65)
    ),
    mainPanel(
      h3("Predicted Churn Probability"),
      textOutput("churn_prob"),
      plotOutput("risk_gauge")
    )
  )
)
# → Connects to trained RF model → outputs live risk score per customer
```

---

## ▶️ How to Run

```r
# 1. Install dependencies
install.packages(c("tidyverse", "caret", "randomForest", "pROC", "ROSE", "e1071"))

# 2. Place telco.csv in your working directory

# 3. Run the full pipeline
source("telco_churn_pipeline.R")
```

**Outputs generated:**

| File | Description |
|:---|:---|
| `01_roc_curve.png` | ROC curve with AUC annotation |
| `02_confusion_matrix.png` | Confusion matrix heatmap |
| `03_feature_importance.png` | Top 15 predictors |
| `04_risk_level_distribution.png` | Customer count by risk tier |
| `05_probability_band_distribution.png` | Bimodal probability distribution |

---

## 📁 Project Structure

```
📦 telco-churn-prediction
 ┣ 📜 telco_churn_pipeline.R          ← Full pipeline (single script)
 ┣ 📊 01_roc_curve.png
 ┣ 📊 02_confusion_matrix.png
 ┣ 📊 03_feature_importance.png
 ┣ 📊 04_risk_level_distribution.png
 ┣ 📊 05_probability_band_distribution.png
 ┗ 📝 README.md
```

---

<div align="center">

### What this project demonstrates

`Data leakage prevention` · `Imbalance-aware modelling` · `Robust validation` · `Business storytelling`

---

*Not just a model — a complete, production-minded pipeline that turns raw customer data into retention decisions.*

*Built with 💜 by **Katlego Mathebula***

</div>
