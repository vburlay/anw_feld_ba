library(tidyverse)
library(carData)
library(crosstable)
library(car)
library(caret)
library(corrplot)
library(plotly)
library(stats)
library(graphics)
library(gbm)
library(randomForest)
library(nnet)
df2<- read_csv(getwd/date/"train_date.csv")

train <- df2 %>% 
  select ("Var1":"Var8","Cat1":"Cat12","NVVar1","NVVar2","NVVar3","NVVar4", "NVCat","OrdCat","Preis")
rm(df2)
colnames(train)[27] <- "Claim_Amount"


simpleMod <- dummyVars(~Claim_Amount+ .,
                       data = train)

df <- data.frame(predict(simpleMod,train))
rm(simpleMod,train)

Claim_Amount <- df$Claim_Amount
df = subset(df, select = -c(Claim_Amount) )

correlations <- cor(df)

highCorr <- findCorrelation(correlations, cutoff = .9)

filteredSegData <- df[,-highCorr]
rm(df,highCorr,correlations)

trans <- preProcess(filteredSegData,
                    method = c("BoxCox","center","scale","pca"))

transformed <- predict(trans,filteredSegData)
rm(trans,filteredSegData)


transformed["Claim_Amount"] <- Claim_Amount
rm(Claim_Amount)
# Modellierung
trainSize <- round(nrow(transformed) * 0.9)# 0.7
testSize <- nrow(transformed) - trainSize


set.seed(123)
train_ind <- sample(seq_len(nrow(transformed)), size = trainSize)

trainSet <- transformed[train_ind, ]
testSet <- transformed[-train_ind, ]
rm(transformed, train_ind)
#nnet Variable
predictors <- subset(trainSet, select = -c(Claim_Amount) )
outcome <- trainSet$Claim_Amount

Claim_Amount <- testSet$Claim_Amount
testSet = subset(testSet, select = -c(Claim_Amount) )

#1) GBM
ames_gbm1 <- gbm(formula = trainSet$Claim_Amount ~ .,
                 data = trainSet,
                 distribution = "gaussian",
                 n.trees = 500,
                 shrinkage = 0.1,
                 interaction.depth = 3,
                 n.minobsinnode = 10,
                 cv.folds = 5 )

best <- which.min(ames_gbm1$cv.error)
sqrt(ames_gbm1$cv.error[best]) #0.1522263
gbm.perf(ames_gbm1, plot.it = TRUE,method = "cv")

predictions <- predict(ames_gbm1, testSet, interval = "predict", level = 0.95)

comparison <- cbind(Claim_Amount,predictions)
colnames(comparison) <- c("actual","predicted")

summary(comparison)

lmValues <- data.frame(obs = Claim_Amount, pred = predictions)
defaultSummary(lmValues)#      RMSE        Rsquared        MAE 
                        #      0.1582149   0.9524001       0.1126284 

plot_ly() %>% 
  add_markers(x = comparison[,1], y = comparison[,2], marker = list(color = "rgb(185, 205, 165)"), name = 'Daten') %>%
  add_lines(x = comparison[,1], y = Claim_Amount, name = 'Regressionslinie', line = list(color = "rgb(135, 135, 201)")) %>%
  layout(xaxis = list(title = 'Beobachtet'),
         yaxis = list(title = 'Vorausgesagt'))

#saveRDS(ames_gbm1, file = "ames_gbm1.rda")
#model_old = readRDS("ames_gbm1.rda")

#2) Random Forest
model_rf <- randomForest(trainSet$Claim_Amount ~ .,
                          data = trainSet,
                          importance =TRUE,
                          ntree = 110)
print(model_rf) 
#RMSE 0.150159
sqrt(model_rf$mse[length(model_rf$mse)]) 
#Rsquared 0.956482
model_rf$rsq[length(model_rf$rsq)]
plot_ly() %>% 
  add_markers(x = 1:110, y = model_rf$mse, marker = list(color = "rgb(185, 205, 165)"), name = 'Daten') %>%
  layout(xaxis = list(title = 'trees'),
         yaxis = list(title = 'Error'))
#saveRDS(model_rf, file = "/model_rf1.rda")
#model_rf = readRDS("model_rf.rda")
#3) NNET

nnetFit <- nnet(predictors, outcome,
                size = 5,
                decay = 0.01,
                linout = TRUE,
                trace = FALSE,
                maxit = 500,
                MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)

predictions <- round(predict(nnetFit, testSet, interval = "predict", level = 0.95),2)

lmValues <- data.frame(obs = Claim_Amount, pred = predictions )
defaultSummary(lmValues)# # RMSE = 0.2034717 ;Rsquared = 0.9214530 ;MAE = 0.1417521

comparison <- cbind(Claim_Amount,predictions[,1])
colnames(comparison) <- c("actual","predicted")
summary(comparison)

plot_ly() %>% 
  add_markers(x = comparison[,1], y = comparison[,2], marker = list(color = "rgb(185, 205, 165)"), name = 'Daten') %>%
  add_lines(x = comparison[,1], y = Claim_Amount, name = 'Regressionslinie', line = list(color = "rgb(135, 135, 201)")) %>%
  layout(xaxis = list(title = 'Beobachtet'),
         yaxis = list(title = 'Vorausgesagt'))

xyplot(Claim_Amount ~ predict(nnetFit, testSet, interval = "predict", level = 0.95),#Trainingsdaten!
       type = c("p","g"),
       xlab = "Beobachtet", ylab = "Vorausgesagt") 

xyplot(nnetFit$residuals ~ nnetFit$fitted.values,
       type = c("p","g"),
       xlab = "Vorausgesagt", ylab = "Residuals")


