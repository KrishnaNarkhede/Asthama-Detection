# Load necessary libraries
library(glmnet)       # For LASSO regression and cross-validation
library(caret)        # For machine learning utility functions (RFE, training/testing split)
library(randomForest) # Used internally by RFE and Boruta
library(dplyr)        # For data manipulation 
library(Boruta)       # For Boruta feature selection algorithm

# Load the dataset
data <- read.csv("balanced_data_SMOTE.csv")

# Ensure the target variable is correctly named
colnames(data)[colnames(data) == "Diagnosis"] <- "diagnosis"

# Convert the target variable to a factor
data$diagnosis <- factor(data$diagnosis)

# Remove missing values
data <- na.omit(data)

# Shuffle the data for reproducibility
set.seed(123)
data <- data[sample(nrow(data)), ]

# Split into training (80%) and testing (20%) sets
set.seed(123)
train_index <- createDataPartition(data$diagnosis, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# LASSO Feature Importance ----

# Create predictor matrix and response vector
x <- model.matrix(diagnosis ~ . - 1, data = train_data)  # Removing the intercept
y <- train_data$diagnosis

# Fit Lasso model with cross-validation
set.seed(123)
lasso_cv <- cv.glmnet(x, y, alpha = 1, family = "binomial")

# Get the best lambda value
best_lambda <- lasso_cv$lambda.min

# Fit the final Lasso model with the best lambda
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda, family = "binomial")

# Extract coefficients and sort by absolute importance
coefficients <- coef(lasso_model)[-1, , drop = FALSE]  # Exclude intercept
lasso_importance <- data.frame(Feature = rownames(coefficients), Importance = abs(coefficients[, 1]))
lasso_importance <- lasso_importance %>% arrange(desc(Importance))

# Print LASSO feature importance scores
cat("LASSO Feature Importance:\n")
print(lasso_importance)

# RFE Feature Importance ----

# Define cross-validation control
trctrl <- trainControl(method = "cv", number = 10)

# Define RFE control
rfe_control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

# Perform RFE
rfe_results <- rfe(
  x = train_data[, -ncol(train_data)],  
  y = train_data$diagnosis,
  sizes = c(5, 10, 15, 20),
  rfeControl = rfe_control
)

# Get feature importance scores
rfe_importance <- varImp(rfe_results)
rfe_importance <- data.frame(Feature = rownames(rfe_importance), Importance = rfe_importance$Overall)
rfe_importance <- rfe_importance %>% arrange(desc(Importance))

# Print RFE feature importance scores
cat("RFE Feature Importance:\n")
print(rfe_importance)

# Boruta Feature Selection ----

set.seed(123)
boruta_result <- Boruta(diagnosis ~ ., data = train_data, doTrace = 0)

# Get feature importance scores
boruta_importance <- attStats(boruta_result)
boruta_importance <- boruta_importance[order(-boruta_importance$meanImp), c("meanImp"), drop = FALSE]
boruta_importance$Feature <- rownames(boruta_importance)

# Print Boruta feature importance scores
cat("Boruta Feature Importance:\n")
print(boruta_importance)



