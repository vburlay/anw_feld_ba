library(shinyFiles)
library(tidyverse)
library(caret)
library(shiny)
library(shinyalert)
library(shinyjs)
library(shinydashboard)
library(DT)
library(plotly)
library(dplyr)
library(e1071)
library(randomForest)
library(nnet)


ui <- dashboardPage(
    #Header Content
    dashboardHeader(title = "Die Vorhersage App"),
    #Sidebar Content
    dashboardSidebar(
        sidebarMenu(
            menuItem("Landing Page", tabName = "landing", icon = icon("th")),
            menuItem("Results Tabular", tabName = "results", icon = icon("th")),
            menuItem("Visualizations", tabName = "visualization", icon = icon("th")),
            menuItem("Model diagnostics", tabName = "evaluierung", icon = icon("th"))
        )
    ),
    #Body Content
    dashboardBody(
        useShinyalert(),
        useShinyjs(), 
        tabItems(
          tabItem(tabName = "evaluierung",
                  box(width = 12,
                  box(width = 4,selectInput("eval","Diagnostic of:", choices = c("Machine Learning","Neural network","Statistic"))), box(width = 6,DTOutput("MetrikeTablle"))),
                  box(width = 12 , actionButton("evalButton", label = "Diagnostic start", width = '100%')),
                  box(width = 12, plotlyOutput("VerhersagemodellPlot")),

          ),   
            tabItem(tabName = "landing",
                    h2("Landingpage"),
                    box(width = 12,
                        box(width = 4,
                            shinyDirButton("dir", "Select files (Step 1)", "Upload", width = '100%'),
                            verbatimTextOutput("dir", placeholder = TRUE),
                        ),
                        box(width = 8,
                            actionButton("loadButton", label = "Load data(Step 2)",width = '100%'),
                            actionButton("startButton", label = "Start the prediction(Step 3)", width = '100%')),
                        box(width = 12,
                            box(width = 6,DTOutput("FileTable")),
                            box(width = 4,selectInput("method","Prediction methods:", choices = c("Machine Learning","Neural network","Statistic")), 
                                verbatimTextOutput("dir_modell", placeholder = TRUE))),
                            
                        box(width = 12,
                        box(width = 12,DTOutput("DateTable")),
                       
                   ),   

            ),
            ),
            tabItem(tabName = "results",
                    box(width = 12,
                        box(width = 12,DTOutput("VerhersageTable"),title = "The prediction of damage payment tabularly"),
                        )
            ),
            tabItem(tabName = "visualization",
                    box(width = 12, plotlyOutput("vorhersagesummarised"),title = "Countplot of the results"),
                    box(width = 12,actionButton("showvorhersageovertime", label = "Show individual results",width = '100%')),
                    box(width = 12,plotlyOutput("vorhersageovertime"),title = "Results prediction"))
                    
            ),
                
    )
              
)

                    
server <- function(input, output, session) {
    #Erhoehung des verfuegbaren Specihers der App fuer groessere Uploads
    memory.limit(size = 250000) 
  hide("vorhersagesummarised")
  hide("vorhersageovertime") 
  hide("eval") 
  hide("VerhersagemodellPlot")
  hide("MetrikeTablle")

#Implementierung von File System Logik
shinyDirChoose(input,'dir',roots = c(home = getwd()),filetypes = c('', 'csv'))

global <- reactiveValues(datapath =getwd())

dir <- reactive(input$dir)
file_selected <<- NULL

output$dir <- renderText({
    global$datapath })



#Logik fuer importvorgaenge von Dateien
observeEvent(input$loadButton, {
    hide("DateTable")
    hide("vorhersagesummarised")
    hide("vorhersageovertime")
    output$vorhersageovertime <- NULL
    hide("VerhersageTable")
    hide("eval") 
    hide("VerhersagemodellPlot")
    hide("MetrikeTablle")

    
    file_selected <<- NULL
    if(is.list(input$dir)){
      

        #erzeugung und ausgabe von template df
        df_files <- data.frame("Datei" = as.character(), "Herkunft" = as.character())
        output$FileTable <- renderDataTable(df_files,selection=list(mode="single"),options= list(scrollY = TRUE,pageLength = 5))
        
        #list & load files
        path <- ""
        for(i in 1:length(paste0(getwd(),"/",input$dir$path))){
            path <- paste0(path,"/",input$dir$path[i])
        }
        path <- gsub("//","/", path)
        files <- list.files(paste0(getwd(),"/",path))
        
        files_list <- list()
        df_files <- data.frame("File" = character())
        for(file in 1:length(files)){
            file_path <- paste0(path,"/",files[file])
            df_files[,1] <- as.character(df_files[,1])
            df_files <- rbind(df_files, files[file])
            names(df_files) <- "Datei"
        }
        df_files$Herkunft <- paste0(getwd(),path)
        df_files <<- df_files
        output$FileTable <- renderDataTable(df_files,selection=list(mode="single"),options= list(scrollY = TRUE,pageLength = 5))
    }
    #exception im falle von ungueltiger eingabe
    else{
        shinyalert("Please choose a folder first!",  type = "warning")
    }
    
})
check_dat <- function(file_selected,method ,path_modell){

set_dat <- read.csv(file_selected)  
set <- set_dat %>% 
  select ("Var1":"Var8","Cat1":"Cat12","NVVar1","NVVar2","NVVar3","NVVar4", "NVCat","OrdCat")
set$Claim_Amount <- c(0)

model= readRDS(path_modell)


simpleMod <- dummyVars(~Claim_Amount + .,data = set)
df_set <- data.frame(predict(simpleMod,set))
rm(simpleMod)
df_set = subset(df_set, select = -c(Claim_Amount) )
trans <- preProcess(df_set,
                    method = c("BoxCox","center","scale","pca"))

transformed <- predict(trans,df_set)# Anpassung der Einstellungen(trans)
rm(trans,df_set)

predictions <- predict(model, transformed , interval = "predict", level = 0.95)
if (method == "Statistic"){
  set_dat$Claim_Amount <- round(predictions[,1],2)
} else {
  set_dat$Claim_Amount <- round(predictions,2)
}

rm(predictions,transformed)
return(set_dat)
}

observeEvent(input$FileTable_rows_selected, {
    names(df_files) <- c("Datei", "Herkunft")
    path_of_file <- df_files$Herkunft[input$FileTable_rows_selected]
    file_selected <- paste0(df_files$Herkunft[input$FileTable_rows_selected],"/",df_files$Datei[input$FileTable_rows_selected])
    df_daten <- read.csv(file_selected)
    file_selected <<- file_selected
    output$DateTable <- renderDataTable(df_daten,options= list(scrollX = TRUE,scrollY = TRUE,pageLength = 5))
    show("DateTable")
    show("eval") 
})
observeEvent(input$method, {
if (input$method == "Statistic"){
  output$dir_modell <-  renderText({ paste0("Modell: ","model_stat.rda (Lineare Regression)")}) 
  show("dir_modell")} 
  else if (input$method == "Machine Learning") {
  output$dir_modell <-  renderText({paste0("Modell: ","model_rf.rda (Random Forest)")}) 
  show("dir_modell")}
  else if (input$method == "Neural network") {
    output$dir_modell <-  renderText({paste0("Modell: ","model_nnet.rda (single-hidden-layer neural network)")}) 
    show("dir_modell")}
})

observeEvent(input$evalButton, {
  if (input$eval == "Statistic"){
    model = readRDS(paste0(getwd(),"/","Modellen","/", "model_stat.rda"))
    rmse <-  round(sqrt(mean(residuals(model)^2)),2) 
    squared <- round(summary(model)$r.squared,2)
    stat_list <-  cbind("Statistic",squared,rmse)
    colnames(stat_list ) <- c("Model","R2 (Squared)","RMSE") 
    df_stat_dk <- as.data.frame(stat_list)
    output$VerhersagemodellPlot<- renderPlotly( 
      {
        p <- plot_ly() %>% 
          add_lines(x = 1:45, y = residuals(model), line = list(color = "rgb(185, 205, 165)")) %>% 
          layout(yaxis = list(titleAuss = 'Residuen'),xaxis = list(titleAuss = 'Date'))
        ggplotly(p)
        
      }  )
    output$MetrikeTablle <- renderDataTable(df_stat_dk,selection=list(mode="single"),options= list(scrollY = TRUE,pageLength = 2))
    show("VerhersagemodellPlot")
    show("MetrikeTablle")
    
  } else if (input$eval == "Machine Learning"){
    model= readRDS(paste0(getwd(),"/","Modellen","/", "model_rf.rda"))
    rmse <- round(sqrt(model$mse[length(model$mse)]),2) 
    squared <- round(model$rsq[length(model$rsq)],2)
    stat_list <-  cbind("Machine Learning (Random forest)",squared,rmse)
    colnames(stat_list ) <- c("Model","R2 (Squared)","RMSE") 
    df_ml_dk <- as.data.frame(stat_list)
    output$MetrikeTablle <- renderDataTable(df_ml_dk,selection=list(mode="single"),options= list(scrollY = TRUE,pageLength = 2))
    show("MetrikeTablle")
    output$VerhersagemodellPlot<- renderPlotly ( 
      {
        p <- plot_ly() %>% 
          add_markers(x = 1:110, y = model$mse, line = list(color = "rgb(185, 205, 165)")) %>%
          layout(xaxis = list(title = 'Trees'),
                 yaxis = list(title = 'MSE -  Error'))
          ggplotly(p)
        
      }  )
    show("VerhersagemodellPlot") 
  } else if (input$eval == "Neural network"){
    model= readRDS(paste0(getwd(),"/","Modellen","/", "model_nnet.rda"))
    resid <- data.frame(residuals= model$residuals, fitted = model$fitted.values)
      output$VerhersagemodellPlot<- renderPlotly ( 
      {
        p <- plot_ly() %>% 
           add_lines(x = resid$fitted, y = resid$residuals, line = list(color = "rgb(185, 205, 165)")) %>% 
           layout(yaxis = list(titleAuss = 'Residuen'),xaxis = list(titleAuss = 'Fitted values'))
        ggplotly(p)
      })
    show("VerhersagemodellPlot") 
    hide("MetrikeTablle")
  } })    
  
observeEvent(input$startButton, {
    hide("VerhersageTable")
    hide("vorhersagesummarised")
    hide("vorhersageovertime")
    hide("VerhersagemodellPlot")
    hide("MetrikeTablle")
    if(is.null(file_selected))
      {
    shinyalert("Please choose a .csv-file from table first!",  type = "warning")}
    else {
        df_reg <- data.frame()
         if (input$method == "Statistic"){
             path_modell <- paste0(getwd(),"/","Modellen","/", "model_stat.rda")
             } else if (input$method == "Machine Learning"){
               path_modell <- paste0(getwd(),"/","Modellen","/", "model_rf.rda") 
             } else if (input$method == "Neural network"){
               path_modell <- paste0(getwd(),"/","Modellen","/", "model_nnet.rda")
             }   
         
         try(df_reg <- check_dat(file_selected, input$method,path_modell) )

         df <- df_reg %>% select("Claim_Amount", "Haushalt_ID","Fzg","Kalenderjahr","Modelljahr","Fahrzeugmarke","Modell", "Teilmodell",
                                 "Var1":"Var8","Cat1":"Cat12","NVVar1","NVVar2","NVVar3","NVVar4", "NVCat","OrdCat")
        output$VerhersageTable <- renderDataTable(df ,options= list(scrollX = TRUE,scrollY = TRUE,pageLength = 15))
        show("VerhersageTable")
          
 
        df_table <- data.frame()
        df_table <- df_reg
        df_table$Fzg <- as.character(df_reg$Fzg)
        df_table$Kalenderjahr <- as.character(df_reg$Kalenderjahr)
        df_table$Fahrzeugmarke <- as.character(df_reg$Fahrzeugmarke)
        df_table$Modelljahr <- as.character(df_reg$Modelljahr)

        output$vorhersagesummarised <- renderPlotly( 
            {ggplot(df_table, aes(Fahrzeugmarke, customdata = Modelljahr, color = Fzg )) +
                    geom_bar() +
                    labs(
                      x = "Vehicle make",
                      y = "Count"
                    )  }  )
        show("vorhersagesummarised")
        
        output$vorhersageovertime <- renderPlotly( 
            {p <- ggplot(df_table, aes(Fahrzeugmarke,  Claim_Amount,customdata = Modelljahr, color = Fzg )) +
                    geom_point()  +
                    theme(legend.position = 'top') 
            ggplotly(p)
 }  )
        }  })


observeEvent(input$showvorhersageovertime, {
  show("vorhersageovertime")})


}

shinyApp(ui, server)

