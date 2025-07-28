library(shiny)
library(nnet)
library(shinythemes)

# Load the saved model and normalization parameters
load("neural_network_top9.rda")  # Loads 'model' and 'norm_params'

# UI
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("🌬️ Asthma Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Enter Patient Details"),
      selectInput("NighttimeSymptoms", "🌙 Nighttime Symptoms", choices = c("No" = 0, "Yes" = 1)),
      selectInput("ChestTightness", "💢 Chest Tightness", choices = c("No" = 0, "Yes" = 1)),
      selectInput("Wheezing", "💨 Wheezing", choices = c("No" = 0, "Yes" = 1)),
      selectInput("HistoryOfAllergies", "🤧 History of Allergies", choices = c("No" = 0, "Yes" = 1)),
      selectInput("ExerciseInduced", "🏃 Exercise-Induced Symptoms", choices = c("No" = 0, "Yes" = 1)),
      numericInput("Age", "🎂 Age", value = 25, min = 5, max = 79),
      selectInput("Gender", "⚥ Gender", choices = c("Female" = 0, "Male" = 1)),
      selectInput("ShortnessOfBreath", "😤 Shortness of Breath", choices = c("No" = 0, "Yes" = 1)),
      selectInput("FamilyHistoryAsthma", "👨‍👩‍👧‍👦 Family History of Asthma", choices = c("No" = 0, "Yes" = 1)),
      br(),
      actionButton("predictBtn", "🔍 Predict Asthma", class = "btn btn-primary btn-lg")
    ),
    
    mainPanel(
      h3("📝 Prediction Result"),
      uiOutput("styled_result"),
      tags$hr(),
      p("🔹 This result is generated using a Neural Network trained on selected top features.")
    )
  )
)

# Server
server <- function(input, output) {
  
  observeEvent(input$predictBtn, {
    user_input <- data.frame(
      NighttimeSymptoms = as.numeric(input$NighttimeSymptoms),
      ChestTightness = as.numeric(input$ChestTightness),
      Wheezing = as.numeric(input$Wheezing),
      HistoryOfAllergies = as.numeric(input$HistoryOfAllergies),
      ExerciseInduced = as.numeric(input$ExerciseInduced),
      Age = input$Age,
      Gender = as.numeric(input$Gender),
      ShortnessOfBreath = as.numeric(input$ShortnessOfBreath),
      FamilyHistoryAsthma = as.numeric(input$FamilyHistoryAsthma)
    )
    
    # Normalize using saved min-max values
    for (f in colnames(user_input)) {
      min_val <- norm_params$Min[norm_params$Feature == f]
      max_val <- norm_params$Max[norm_params$Feature == f]
      user_input[[f]] <- (user_input[[f]] - min_val) / (max_val - min_val)
    }
    
    pred_prob <- predict(model, user_input, type = "raw")
    pred_class <- ifelse(pred_prob > 0.40, "Asthmatic", "Non-Asthmatic")
    
    output$styled_result <- renderUI({
      style_class <- ifelse(pred_class == "Asthmatic", "danger", "success")
      HTML(paste0(
        '<div class="alert alert-', style_class, '" role="alert">',
        '<h4 class="alert-heading">', pred_class, '</h4>',
        #'<hr>',
        #'<p>Predicted Probability: <strong>', round(pred_prob, 4), '</strong></p>',
        '</div>'
      ))
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)
