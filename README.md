# TELCO-CHURN-PREDICTION-FULL-ML-PIPELINE
Built a Telco Customer Churn prediction model in R using Random Forest with 10-fold cross-validation. Cleaned data, removed leakage features, handled missing values, converted categorical variables, and balanced classes using ROSE. Optimized performance using ROC-AUC and evaluated feature importance.
Telco Customer Churn Prediction – Full ML Pipeline
 Project Overview

This project builds a complete machine learning pipeline in R to predict customer churn in a telecom company.

The goal is to identify customers likely to leave the service so businesses can take proactive retention actions.

The model uses Random Forest with 10-fold cross-validation and optimizes performance using ROC-AUC.

 Dataset

Telco customer dataset (telco.csv)

Target variable: Churn.Label (Yes / No)

Includes demographic, service usage, and billing features

 Project Pipeline
## 1️. Data Preparation

Loaded required libraries

Converted tibble to data.frame

Cleaned column names

Converted the target variable to a factor

Ensured “Yes” is treated as the positive class

## 2️. Data Cleaning

Removed data leakage variables:

* Customer Status

* Churn Score

* Churn Category

* Churn Reason

* Customer ID

Removed low-value geographic features:

* Country, State, City, Zip Code

* Latitude & Longitude

## 3️.  Missing Value Handling

Replaced missing values in Total.Charges with median

## 4️. Feature Engineering

Converted all character variables to factors

## 5️. Class Imbalance Handling

Checked churn distribution

Applied ROSE sampling within cross-validation folds

## 6. Model Training

* Algorithm: Random Forest

* 10-fold Cross-Validation

* 500 trees

* Performance metric: ROC-AUC

## 7. Model Evaluation

Extracted best tuning parameters

Calculated cross-validated AUC

Analyzed feature importance

## 8. Results

* Model optimized using ROC-AUC

* Cross-validated AUC reported from training results

* Variable importance identifies key churn drivers

 ## 9.  How to Run

Install required R packages:

install. packages(c("tidyverse", "caret", "randomForest", "pROC", "ROSE", "e1071"))


Place telco.csv in your working directory

Run the script

 Key Skills Demonstrated

Data cleaning & preprocessing

Leakage prevention

Handling imbalanced data

Cross-validation

Model tuning

Model evaluation using ROC

Feature importance interpretation


 Author

Katlego Mathebula
Aspiring Data Scientist | Machine Learning Enthusiast
