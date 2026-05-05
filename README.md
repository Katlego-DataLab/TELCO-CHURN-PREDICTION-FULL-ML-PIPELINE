# Telco Customer Churn Prediction — ML Pipeline in R

**Can you identify a customer who's about to leave — before they do?**  
This project answers that question with a production-oriented Random Forest pipeline that flags at-risk telecom customers with **96.8% recall** and **AUC = 0.987**.

> **Built by:** Katlego Mathebula · R · randomForest · caret · ROSE · pROC

---

## TL;DR

| What | Result |
|---|---|
| **Problem** | Telecom company losing customers with no early warning system |
| **Solution** | ML pipeline that scores every customer's churn probability |
| **Model** | Random Forest · 500 trees · 10-fold CV × 3 repeats |
| **AUC** | **0.987** — model ranks churners above non-churners 98.7% of the time |
| **Recall** | **96.8%** — catches 361 of 373 churned customers |
| **Business value** | ~2,050 high-risk customers identified for targeted retention |

---

## Business Impact

Acquiring a new telecom customer costs **5–7× more** than retaining one. Without a churn prediction system, retention teams either:
- React too late (customer has already cancelled), or
- Blast outreach to everyone (expensive, low ROI)

This model changes that:

- 🎯 **Focus retention spend** on ~2,050 high-risk customers instead of all 7,000
- 📞 **Trigger outreach 30–60 days early** — before the customer decides to leave
- 💡 **Know why** each customer is at risk (contract type, satisfaction, tenure)
- 📈 **Protect revenue** — every false negative (missed churner) = lost customer with zero intervention

---

## Results

### Model Performance

| Metric | Value |
|---|---|
| AUC (ROC) | **0.987** |
| Recall (Sensitivity) | **96.8%** |
| Overall Accuracy | ~93% |
| True Positives | 361 churners correctly flagged |
| False Negatives | 12 churners missed |

**On AUC = 0.987:** This number is high but legitimate. The key reason: post-churn variables (`Churn.Score`, `Churn.Reason`, `Customer.Status`) were explicitly removed before modelling — a common mistake that inflates AUC to near-perfect but produces a model that can't work in real life. ROSE oversampling was also applied *inside* each CV fold, not before splitting, preventing synthetic samples from leaking into validation. What remains is a model evaluated on genuinely unseen data.

### Customer Risk Segments

| Segment | Customers | Recommended Action |
|---|---|---|
| 🔴 High Risk | ~2,050 | Immediate retention outreach |
| 🟡 Medium Risk | ~280 | Proactive value reinforcement |
| 🟢 Low Risk | ~4,600 | Loyalty programmes · upsell |

### Top Churn Drivers

| # | Feature | What It Tells You |
|---|---|---|
| 1 | Satisfaction Score | Strongest signal — low satisfaction precedes churn |
| 2 | Number of Referrals | Zero referrals = no social investment in the service |
| 3 | Contract Type | Month-to-month customers churn at far higher rates |
| 4 | Online Security | Add-on subscribers are more embedded — less likely to leave |
| 5 | Tenure | Customers under 12 months haven't built loyalty yet |
| 6 | Monthly Charges | High cost + low perceived value = churn trigger |

---

## How This Works in Production

```
Daily/Weekly Customer Data
          │
          ▼
  Feature Engineering
  (same transformations as training)
          │
          ▼
  Trained RF Model
  → Churn Probability (0–1) per customer
          │
          ▼
  Risk Segmentation
  → High / Medium / Low
          │
          ▼
  CRM Integration
  → Retention team queue, prioritised by risk score
```

A real deployment would:
1. Re-score all customers on a weekly batch job
2. Push high-risk flags to the CRM (Salesforce, HubSpot, etc.)
3. Trigger personalised retention workflows (discount offers, account reviews)
4. Log outcomes to retrain the model quarterly

---

## Pipeline Overview

The pipeline handles everything from raw CSV to scored predictions:

1. **Leakage prevention** — remove variables only observable after churn
2. **Missing value imputation** — median imputation on `Total.Charges`
3. **Feature engineering** — character → factor for RF compatibility
4. **Near-zero variance removal** — drop uninformative predictors
5. **Stratified CV + ROSE** — imbalance correction applied *per fold*
6. **Random Forest training** — 500 trees, optimised by ROC-AUC
7. **Evaluation** — ROC curve, confusion matrix, feature importance, risk tiers

---

## How to Run

```r
# Install dependencies
install.packages(c("tidyverse", "caret", "randomForest", "pROC", "ROSE", "e1071"))

# Place telco.csv in your working directory, then:
source("telco_churn_pipeline.R")
```

**Outputs:** ROC curve · confusion matrix · feature importance plot · risk distribution · probability plot

---

## Upgrade Path: Interactive Dashboard

> **Recommended next step:** Deploy this as a Shiny app to make it usable by non-technical stakeholders.

### What the app would do:

- **Customer Lookup** — enter a customer ID and see their churn probability + risk tier
- **What-If Simulator** — adjust contract type, monthly charges, or satisfaction score and watch the risk score update in real time
- **Retention Queue** — filterable table of high-risk customers sorted by churn probability
- **Feature Importance Panel** — visual explanation of *why* a specific customer is flagged

### Shiny app skeleton:

```r
library(shiny)
library(randomForest)

ui <- fluidPage(
  titlePanel("Churn Risk Dashboard"),
  sidebarLayout(
    sidebarPanel(
      selectInput("contract", "Contract Type",
                  choices = c("Month-to-month", "One year", "Two year")),
      sliderInput("satisfaction", "Satisfaction Score", 1, 5, 3),
      sliderInput("tenure", "Tenure (months)", 0, 72, 12),
      sliderInput("monthly_charges", "Monthly Charges ($)", 20, 120, 65)
    ),
    mainPanel(
      h3("Churn Probability"),
      textOutput("churn_prob"),
      plotOutput("risk_gauge")
    )
  )
)

server <- function(input, output) {
  output$churn_prob <- renderText({
    # Pass inputs to trained model and return probability
    paste0("Risk Score: ", round(predict(rf_model, newdata = input_df(), type = "prob")[, "Yes"] * 100, 1), "%")
  })
}

shinyApp(ui, server)
```

A deployed Shiny app would let a retention manager run scenarios on any customer without touching code — turning this ML project into a usable business tool.

---

## Project Structure

```
├── telco_churn_pipeline.R          # Full pipeline
├── telco.csv                       # Input data (not included)
├── 01_roc_curve.png
├── 02_confusion_matrix.png
├── 03_feature_importance.png
├── 04_risk_level_distribution.png
├── 05_probability_band_distribution.png
└── README.md
```

---

*Built by Katlego Mathebula*
