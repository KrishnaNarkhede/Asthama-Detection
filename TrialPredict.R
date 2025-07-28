# Load necessary libraries
library(nnet)
library(caret)

# Load dataset
data <- read.csv("balanced_data_SMOTE.csv")

# Shuffle the data
set.seed(123)
data <- data[sample(nrow(data)), ]

# Check class distribution
print("Class Distribution in Dataset:")
print(table(data$Diagnosis))

# Split into 80% training, 20% testing
set.seed(123)
train_index <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Convert target variable to numeric (0,1)
train_data$Diagnosis <- as.numeric(as.character(train_data$Diagnosis))
test_data$Diagnosis <- as.numeric(as.character(test_data$Diagnosis))

# Top 9 features
features_top9 <- c("NighttimeSymptoms", "ChestTightness", "Wheezing", "HistoryOfAllergies", 
                   "ExerciseInduced", "Age", "Gender", "ShortnessOfBreath", "FamilyHistoryAsthma")

# Normalize features using Min-Max Scaling
normalize <- function(x, min_val, max_val) { 
  return ((x - min_val) / (max_val - min_val))
}

# Compute min/max values for scaling
min_vals <- apply(train_data[, features_top9], 2, min)
max_vals <- apply(train_data[, features_top9], 2, max)

# Print min-max values for debugging
print("Feature Min Values:"); print(min_vals)
print("Feature Max Values:"); print(max_vals)

# Apply normalization
for (col in features_top9) {
  train_data[[col]] <- normalize(train_data[[col]], min_vals[col], max_vals[col])
  test_data[[col]] <- normalize(test_data[[col]], min_vals[col], max_vals[col])
}

# Verify normalization worked (should not all be 0)
print("First few rows after normalization:")
print(head(train_data[, features_top9]))

# Train the neural network model
nn_model <- nnet(
  Diagnosis ~ ., 
  data = train_data[, c(features_top9, "Diagnosis")], 
  size = 100,         # Increase neurons
  decay = 0.00001,    # Reduce learning rate
  maxit = 3000,       # More iterations
  trace = TRUE        # Show training progress
)

# Check if model training was successful
print("Neural Network Training Completed ✅")

# Function for manual testing
manual_test <- function(input_data) {
  input_data <- as.data.frame(t(input_data))
  colnames(input_data) <- features_top9
  
  # Normalize using training min-max values
  for (col in features_top9) {
    input_data[[col]] <- normalize(input_data[[col]], min_vals[col], max_vals[col])
  }
  
  # Print normalized test input
  print("Normalized Test Input:")
  print(input_data)
  
  # Predict using trained model
  pred_prob <- predict(nn_model, input_data, type = "raw")
  
  # Print probability score
  print(paste("Predicted Probability:", pred_prob))
  
  # Convert probability to binary classification
  pred <- ifelse(pred_prob > 0.5, 1, 0)
  
  return(pred)
}

# Test Case: All Inputs as 1 (Should predict 1)
test_input <- c(NighttimeSymptoms = 1, ChestTightness = 1, Wheezing = 1, HistoryOfAllergies = 1, 
                ExerciseInduced = 1, Age = 45, Gender = 1, ShortnessOfBreath = 1, FamilyHistoryAsthma = 1)

# Run the test
manual_prediction <- manual_test(test_input)
print(paste("Predicted Diagnosis:", manual_prediction))

