library(shiny)
library(nnet)

# Load the saved model and normalization parameters
load("neural_network_top9.rda")  # Loads 'model' and 'norm_params'

# UI
ui <- fluidPage(
  titlePanel("Asthma Prediction - Neural Network (Top 9 Features)"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("NighttimeSymptoms", "Nighttime Symptoms (0 or 1)", value = 0, min = 0, max = 1),
      numericInput("ChestTightness", "Chest Tightness (0 or 1)", value = 0, min = 0, max = 1),
      numericInput("Wheezing", "Wheezing (0 or 1)", value = 0, min = 0, max = 1),
      numericInput("HistoryOfAllergies", "History of Allergies (0 or 1)", value = 0, min = 0, max = 1),
      numericInput("ExerciseInduced", "Exercise Induced Symptoms (0 or 1)", value = 0, min = 0, max = 1),
      numericInput("Age", "Age", value = 25, min = 5, max = 79),
      numericInput("Gender", "Gender (0 = Female, 1 = Male)", value = 0, min = 0, max = 1),
      numericInput("ShortnessOfBreath", "Shortness of Breath (0 or 1)", value = 0, min = 0, max = 1),
      numericInput("FamilyHistoryAsthma", "Family History of Asthma (0 or 1)", value = 0, min = 0, max = 1),
      actionButton("predictBtn", "Predict Asthma")
    ),
    
    mainPanel(
      h3("Prediction Result:"),
      verbatimTextOutput("prediction_output")
    )
  )
)

# Server
server <- function(input, output) {
  
  observeEvent(input$predictBtn, {
    # Capture input into a data frame
    user_input <- data.frame(
      NighttimeSymptoms = input$NighttimeSymptoms,
      ChestTightness = input$ChestTightness,
      Wheezing = input$Wheezing,
      HistoryOfAllergies = input$HistoryOfAllergies,
      ExerciseInduced = input$ExerciseInduced,
      Age = input$Age,
      Gender = input$Gender,
      ShortnessOfBreath = input$ShortnessOfBreath,
      FamilyHistoryAsthma = input$FamilyHistoryAsthma
    )
    
    # Normalize using saved min-max values
    for (f in colnames(user_input)) {
      min_val <- norm_params$Min[norm_params$Feature == f]
      max_val <- norm_params$Max[norm_params$Feature == f]
      user_input[[f]] <- (user_input[[f]] - min_val) / (max_val - min_val)
    }
    
    # Predict
    pred_prob <- predict(model, user_input, type = "raw")
    pred_class <- ifelse(pred_prob > 0.45, "Asthmatic", "Non-Asthmatic")
    
    output$prediction_output <- renderText({
      paste("Predicted Class:", pred_class, "\nProbability:", round(pred_prob, 4))
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)


summary(model)

