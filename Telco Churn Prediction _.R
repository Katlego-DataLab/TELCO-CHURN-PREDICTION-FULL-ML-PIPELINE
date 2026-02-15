
## TELCO CHURN PREDICTION – FULL ML PIPELINE



## 1. Loading required  Libraries

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ROSE)
library(e1071)

set.seed(123)


## 2. Load Data

telco <- read_csv("telco.csv")

# Convert tibble → data.frame
telco <- as.data.frame(telco)

# Clean column names (removes spaces permanently)
names(telco) <- make.names(names(telco))


## 3. Convert Target Variable

telco$Churn.Label <- as.factor(telco$Churn.Label)

# Make sure "Yes" is positive class
telco$Churn.Label <- factor(telco$Churn.Label,
                            levels = c("No", "Yes"))


## 4️. Remove Data Leakage Columns

telco$Customer.Status <- NULL
telco$Churn.Score <- NULL
telco$Churn.Category <- NULL
telco$Churn.Reason <- NULL
telco$Customer.ID <- NULL


## 5. Remove Low-Value Geographic Columns

telco$Country <- NULL
telco$State <- NULL
telco$City <- NULL
telco$Zip.Code <- NULL
telco$Latitude <- NULL
telco$Longitude <- NULL


## 6. Handle Missing Values


# Replace NA in Total Charges with median
telco$Total.Charges[is.na(telco$Total.Charges)] <-
  median(telco$Total.Charges, na.rm = TRUE)


## 7. Convert All Character Columns to Factors

char_cols <- sapply(telco, is.character)
telco[char_cols] <- lapply(telco[char_cols], as.factor)

## 8. Check Class Balance 
table(telco$Churn.Label) 


## 9.  CROSS-VALIDATION SETUP 


control <- trainControl(
  method = "cv",                 # 10-fold cross validation
  number = 10,
  classProbs = TRUE,             # needed for ROC
  summaryFunction = twoClassSummary,
  sampling = "rose",             # balance inside each fold
  savePredictions = TRUE
)


## 10. Train Random Forest with Cross-Validation


rf_cv_model <- train(
  Churn.Label ~ .,
  data = telco,
  method = "rf",
  trControl = control,
  metric = "ROC",                # optimize for AUC
  ntree = 500,
  importance = TRUE
)


## 11. View Results


print(rf_cv_model)

# Best tuning parameter
rf_cv_model$bestTune

# Cross-validated AUC
max(rf_cv_model$results$ROC)

## 12. Variable Importance


varImp(rf_cv_model)
