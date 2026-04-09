# TELCO CHURN PREDICTION – FULL ML PIPELINE


# 1. LOAD REQUIRED LIBRARIES


library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ROSE)
library(e1071)

set.seed(123)


# 2. LOAD DATA


telco <- read_csv("telco.csv")
telco <- as.data.frame(telco)

# Clean column names
names(telco) <- make.names(names(telco))


# 3. TARGET VARIABLE


telco$Churn.Label <- as.factor(telco$Churn.Label)

# Ensure "Yes" is positive class
telco$Churn.Label <- factor(telco$Churn.Label,
                            levels = c("No", "Yes"))


# 4. REMOVE DATA LEAKAGE


telco$Customer.Status <- NULL
telco$Churn.Score <- NULL
telco$Churn.Category <- NULL
telco$Churn.Reason <- NULL
telco$Customer.ID <- NULL


# 5. REMOVE LOW-VALUE GEO COLUMNS


telco$Country <- NULL
telco$State <- NULL
telco$City <- NULL
telco$Zip.Code <- NULL
telco$Latitude <- NULL
telco$Longitude <- NULL


# 6. HANDLE MISSING VALUES


telco$Total.Charges[is.na(telco$Total.Charges)] <-
  median(telco$Total.Charges, na.rm = TRUE)


# 7. CONVERT CHARACTER TO FACTOR


char_cols <- sapply(telco, is.character)
telco[char_cols] <- lapply(telco[char_cols], as.factor)



# 8. CHECK CLASS BALANCE (VISUAL)


print(table(telco$Churn.Label))

ggplot(telco, aes(x = Churn.Label, fill = Churn.Label)) +
  geom_bar() +
  labs(title = "Customer Churn Distribution",
       x = "Churn",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("steelblue", "magenta"))


# 9. CROSS-VALIDATION SETUP

control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  index = folds,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

##
set.seed(123)
folds <- createFolds(telco$Churn.Label, k = 10, returnTrain = TRUE)

# 10. TRAIN RANDOM FOREST

# Remove zero-variance predictors
nzv <- nearZeroVar(telco)
telco <- telco[, -nzv]


rf_cv_model <- train(
  Churn.Label ~ .,
  data = telco,
  method = "rf",
  trControl = control,
  metric = "ROC",
  ntree = 500,
  importance = TRUE
)

##
table(telco$Churn.Label) #Checking class imbalance
levels(telco$Churn.Label) #checking target levels
sum(is.na(telco)) #checking NA values 

# 11. MODEL RESULTS


print(rf_cv_model)

cat("\nBest Tuning Parameter:\n")
print(rf_cv_model$bestTune)

cat("\nBest Cross-Validated AUC:\n")
print(max(rf_cv_model$results$ROC))

\
# 12. ROC CURVE VISUALIZATION


pred_probs <- rf_cv_model$pred

roc_obj <- roc(pred_probs$obs,
               pred_probs$Yes)

plot(roc_obj,
     col = "blue",
     main = "ROC Curve - Random Forest Model")

cat("\nAUC Score:\n")
print(auc(roc_obj))


# 13. CONFUSION MATRIX


final_preds <- predict(rf_cv_model, telco)

conf_matrix <- confusionMatrix(
  final_preds,
  telco$Churn.Label,
  positive = "Yes"
)

print(conf_matrix)


# 14. CONFUSION MATRIX HEATMAP #


cm_table <- as.data.frame(conf_matrix$table)

ggplot(cm_table,
       aes(Prediction, Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq),
            color = "white",
            size = 6) +
  scale_fill_gradient(low = "steelblue",
                      high = "darkgreen") +
  labs(title = "Confusion Matrix Heatmap") +
  theme_minimal()


# 15. VARIABLE IMPORTANCE


var_imp <- varImp(rf_cv_model)

plot(var_imp,
     main = "Variable Importance - Random Forest")



# 16. TOP 15 IMPORTANT FEATURES


importance_df <- var_imp$importance
importance_df$Feature <- rownames(importance_df)

top_features <- importance_df %>%
  arrange(desc(Overall)) %>%
  head(15)

ggplot(top_features,
       aes(x = reorder(Feature, Overall),
           y = Overall)) +
  geom_bar(stat = "identity",
           fill = "magenta") +
  coord_flip() +
  labs(title = "Top 15 Most Important Features",
       x = "Feature",
       y = "Importance") +
  theme_minimal()



## END OF SCRIPT ##
