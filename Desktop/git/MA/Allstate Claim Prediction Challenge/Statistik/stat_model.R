library(tidyverse)
library(carData)
library(car)
library(caret)
library(corrplot)
library(plotly)
library(stats)
library(graphics)

# Create connection
df2<- read_csv(getwd/date/"train_date.csv")

train <- df2 %>% 
  select ("Var1":"Var8","Cat1":"Cat12","NVVar1","NVVar2","NVVar3","NVVar4", "NVCat","OrdCat","Preis")
rm(df2)

##############################################
simpleMod <- dummyVars(~Preis + .,
                       data = train)

df <- data.frame(predict(simpleMod,train))
rm(simpleMod,train)

Preis <- df$Preis
df = subset(df, select = -c(Preis) )

correlations <- cor(df)
dim(correlations)
#corrplot(correlations,order = "hclust")

highCorr <- findCorrelation(correlations, cutoff = .9)
length(highCorr)

head(highCorr)

filteredSegData <- df[,-highCorr]
rm(df,highCorr,correlations)

trans <- preProcess(filteredSegData,
                    method = c("BoxCox","center","scale","pca"))

transformed <- predict(trans,filteredSegData)
rm(trans,filteredSegData)


transformed["Preis"] <- Preis
rm(Preis)
# Modellierung
trainSize <- round(nrow(transformed) * 0.9)# 0.7
testSize <- nrow(transformed) - trainSize


set.seed(123)
train_ind <- sample(seq_len(nrow(transformed)), size = trainSize)

trainSet <- transformed[train_ind, ]
testSet <- transformed[-train_ind, ]
rm(transformed, train_ind)

model <- lm(formula = trainSet$Preis ~ .,data = trainSet )
#model_1 <- step(lm(formula = trainSet$Preis ~ .,data = trainSet ), direction = "backward")
#model_2 <- step(lm(formula = trainSet$Preis ~ .,data = trainSet ), direction = "forward")

## Evaluation
#anova(model,model_1,model_2)
summary(model)#Multiple R-squared:  0.89 ;Residual standard error: 0.239;p-value: < 2.2e-16


# Mean squared error
mse <- mean(residuals(model)^2)
mse #0.05700029
rmse <- sqrt(mse)
rmse #0.2387473
##########################Train############################
#solTrainY <- trainSet$Preis
#trainSet = subset(trainSet, select = -c(Preis) )

#predictions <- predict(model, trainSet, interval = "predict", level = 0.95)
#head(predictions)

#comparison <- cbind(solTrainY,predictions[,1])
#colnames(comparison) <- c("actual","predicted")

#head(comparison)

#summary(comparison)


# Train Daten
#Mean Absolute Percent Error (mape)

#mape <- (sum(abs(comparison[,1]-comparison[,2]) /
#               abs(comparison[,1])) / nrow(comparison))*100
#mape # 0.4360399
###################Test###################################
solTestY <- testSet$Preis
testSet = subset(testSet, select = -c(Preis) )

predictions <- predict(model, testSet, interval = "predict", level = 0.95)
head(predictions)

comparison <- cbind(solTestY,predictions[,1])
colnames(comparison) <- c("actual","predicted")

head(comparison)

summary(comparison)


# Test Daten
#Mean Absolute Percent Error (mape)

mape <- (sum(abs(comparison[,1]-comparison[,2]) /
               abs(comparison[,1])) / nrow(comparison))*100
mape # 0.5044531

mapeTabelle <- cbind(comparison, abs(comparison[,1] - 
                                       comparison[,2])/comparison[,1]*100)
colnames(mapeTabelle)[3] <- "absoluten prozentualen Fehler"
head(mapeTabelle)

sum(mapeTabelle[,3])/nrow(comparison)#0.5044531


lmPred1 <- predict(model,testSet)
head(lmPred1)

lmValues <- data.frame(obs = solTestY, pred = lmPred1)
defaultSummary(lmValues)# RMSE = 0.2423622 ;Rsquared = 0.8883265 ;MAE = 0.1762838  


#1

#Modelldiagnostik
#Normalverteilung der Residuen - Abweichung von der Normalverteilung (p-Wert)

#qqnorm(residuals(model))
#qqline(residuals(model))
#shapiro.test(residuals(model))" - bei <= 5000 Sätzen (W = 0.99212, p-value = 0.9701)

xyplot(trainSet$Preis ~ predict(model),#Trainingsdaten!
       type = c("p","g"),
       xlab = "Beobachtet", ylab = "Vorausgesagt")

#Regressionsgleichung und Homoskedasitä
xyplot(resid(model) ~ predict(model),
       type = c("p","g"),
       xlab = "Vorausgesagt", ylab = "Residuals")

plot_ly() %>% 
  add_lines(x = 1:25125, y = residuals(model), line = list(color = "rgb(185, 205, 165)")) %>% 
  layout(yaxis = list(titleAuss = 'Residuen'))

#2
#xyplot(comparison[,2] ~ comparison[,1],
#       type = c("p","g"),
#       xlab = "Observed", ylab = "Predicted")


plot_ly() %>% 
  add_markers(x = comparison[,1], y = comparison[,2], marker = list(color = "rgb(185, 205, 165)"), name = 'Daten') %>%
  add_lines(x = comparison[,1], y = solTestY, name = 'Regressionslinie', line = list(color = "rgb(135, 135, 201)")) %>%
  layout(xaxis = list(title = 'Beobachtet'),
         yaxis = list(title = 'Vorausgesagt'))

model_old = readRDS("model.rda")
xyplot(trainSet$Claim_Amount ~ predict(model_old),#Trainingsdaten!
       type = c("p","g"),
       xlab = "Beobachtet", ylab = "Vorausgesagt")