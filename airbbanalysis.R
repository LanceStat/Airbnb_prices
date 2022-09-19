library(tidyverse)
library(mice)
library(VIM)
library(lubridate)
library(caret)
library(randomForest)

# data

train <- read_csv("train.csv")
test <- read_csv("test.csv")


# EDA

matrixplot(train)
aggr(train, labels = names(train), sortVars = TRUE)
matrixplot(test)

# Adjust variable types
# converting last_review date into year, month, day variables

train$last_review_year <- year(train$last_review)
train$last_review_month <- month(train$last_review)
train$last_review_day <- mday(train$last_review)
train$last_review_dayofyear <- yday(train$last_review)
train$last_review_dayofweek <- wday(train$last_review)
train$neighbourhood_group <- as.factor(train$neighbourhood_group)
train$neighbourhood <- as.factor(train$neighbourhood)
train$room_type <- as.factor(train$room_type)

test$last_review_year <- year(test$last_review)
test$last_review_month <- month(test$last_review)
test$last_review_day <- mday(test$last_review)
test$last_review_dayofyear <- yday(test$last_review)
test$last_review_dayofweek <- wday(test$last_review)
test$neighbourhood_group <- as.factor(test$neighbourhood_group)
test$neighbourhood <- as.factor(test$neighbourhood)
test$room_type <- as.factor(test$room_type)



# Impute missing data 

train <- train %>% 
  select(-last_review, -host_name, -name, -neighbourhood)

test <- test %>% 
  select(-last_review, -host_name, -name, -neighbourhood)

set.seed(123)
mids <- mice(train, m = 1, method = "rf")

comp <- complete(mids, 1)
write_csv(comp, "imptrain.csv")

set.seed(11111)
midstest <- mice(test, m = 1, method = "rf")

imptest <- complete(midstest, 1)
write_csv(imptest, "imptest.csv")



# Modeling

set.seed(1124)
trainindex <- createDataPartition(comp$price, p = 0.70, list = FALSE)
comptrain <- comp[trainindex,]
comptest <- comp[-trainindex,]


set.seed(333)
rf1 <- randomForest(price ~ ., data = comptrain, mtry = 3, ntree = 1000,
                    importance = TRUE)
plot(rf1)
which.min(rf1$mse)

set.seed(444)
rf2 <- randomForest(price ~ ., data = comptrain, mtry = 6, ntree = 1000,
                    importance = TRUE)
plot(rf2)
which.min(rf2$mse)

set.seed(555)
rf3 <- randomForest(price ~ ., data = comptrain, mtry = 9, ntree = 1000,
                    importance = TRUE)
plot(rf3)
which.min(rf3$mse)


varImpPlot(rf2)

pricehat1 <- predict(rf1, newdata = comptest)
pricehat2 <- predict(rf2, newdata = comptest)
pricehat3 <- predict(rf3, newdata = comptest)


RMSLE <- function(preds, actual) {
  logs <- (log(preds + 1) - log(actual + 1))^2
  roots <- sqrt(mean(logs))
  return(roots)
}

RMSLE(pricehat1, comptest$price)
RMSLE(pricehat2, comptest$price)
RMSLE(pricehat3, comptest$price)

testhat <- predict(rf2, newdata = imptest)  
rfsubmit <- data.frame(imptest$id, testhat)
names(rfsubmit)[1] <- "id"
names(rfsubmit)[2] <- "price"
write_csv(rfsubmit, "abbrfsubmit.csv")



















