# loading required libraries
library(readxl)
library(neuralnet)
library(Metrics)

# reading UoW_load data set
uow.load <- read_excel("UoW_load.xlsx")

# splitting train and test sets
train.set = uow.load[1:430,]
test.set = uow.load[431:500,]

# min and max of 9th hour column in train set
sequence1.min <- min(train.set$`09:00`)
sequence1.max <- max(train.set$`09:00`)

# min and max of 10th hour column in train set
sequence2.min <- min(train.set$`10:00`)
sequence2.max <- max(train.set$`10:00`)

# min and max of 11th hour column in train set
sequence3.min <- min(train.set$`11:00`)
sequence3.max <- max(train.set$`11:00`)

# normalize function
normalize <- function(x, min, max) {
  return((x - min) / (max - min))
}

# un normalize function
unnormalize <- function(x, min, max) {
  return( (max - min)*x + min )
}

normalized.train.sequence1 <- normalize(train.set$`09:00`, sequence1.min, sequence1.max)
normalized.train.sequence2 <- normalize(train.set$`10:00`, sequence2.min, sequence2.max)
normalized.train.sequence3 <- normalize(train.set$`11:00`, sequence3.min, sequence3.max)

# normalize train set
normalized.train.set <- as.data.frame(cbind(normalized.train.sequence1, normalized.train.sequence2, normalized.train.sequence3))
colnames(normalized.train.set) <- c("09:00", "10:00", "11:00")

normalized.test.sequence1 <- normalize(test.set$`09:00`, sequence1.min, sequence1.max)
normalized.test.sequence2 <- normalize(test.set$`10:00`, sequence2.min, sequence2.max)
normalized.test.sequence3 <- normalize(test.set$`11:00`, sequence3.min, sequence3.max)

# normalize test set
normalized.test.set <- as.data.frame(cbind(normalized.test.sequence1, normalized.test.sequence2, normalized.test.sequence3))
colnames(normalized.test.set) <- c("09:00", "10:00", "11:00")

# function to create input output vectors
prepare.data <- function(sequence.data, num.steps, add.load) {
  if (add.load == 0) {
    X = matrix(, nrow = 0, ncol = (num.steps * 3))
    y = c()
    
    for (i in 1:length(sequence.data$`11:00`)) {
      # index of the value that needs to be predicted
      end <- i + num.steps
      
      # break the loop if goes beyond the limit
      if (end > length(sequence.data$`11:00`)) {
        break
      }
      
      # preparing input and output vector
      input <- sequence.data$`09:00`[i:(end - 1)]
      input <- append(input, sequence.data$`10:00`[i:(end - 1)])
      input <- append(input, sequence.data$`11:00`[i:(end - 1)])
      
      output <- sequence.data$`11:00`[end]
      
      X <- rbind(X, c(input))
      y <- append(y, output)
    }
    
    # return I/O vectors as a data frame
    return(cbind(as.data.frame(X), y))
  } else {
    X = matrix(, nrow = 0, ncol = ((num.steps * 3) + 3))
    y = c()
    
    for (i in 1:length(sequence.data$`11:00`)) {
      # index of the value that needs to be predicted
      end <- i + 7
      
      # break the loop if goes beyond the limit
      if (end > length(sequence.data$`11:00`)) {
        break
      }
      
      # preparing input and output vector
      input <- sequence.data$`09:00`[(end - num.steps):(end - 1)]
      input <- append(input, sequence.data$`10:00`[(end - num.steps):(end - 1)])
      input <- append(input, sequence.data$`11:00`[(end - num.steps):(end - 1)])
      input <- append(input, sequence.data$`09:00`[i])
      input <- append(input, sequence.data$`10:00`[i])
      input <- append(input, sequence.data$`11:00`[i])
      
      output <- sequence.data$`11:00`[end]
      
      X <- rbind(X, c(input))
      y <- append(y, output)
    }
    
    # return I/O vectors as a data frame
    return(cbind(as.data.frame(X), y))
  }
}

# function to evaluate a model
evaluate.model <- function(model, test.data) {
  # making predictions using the model
  results <- compute(model, test.data[1:(length(test.data) - 1)])
  
  # predictions made by the model
  predicted <- results$net.result
  
  # un normalizing test and predicted values
  unnormalized.acutal <- unnormalize(test.data$y, sequence3.min, sequence3.max)
  unnormalized.predicted <- unnormalize(predicted, sequence3.min, sequence3.max)
  
  # data frame with predicted and actual values
  report <- as.data.frame(cbind(unnormalized.acutal, unnormalized.predicted))
  colnames(report) <- c("Expected Output", "Neural Net Output") 
  
  # RMSE
  error <- (report$`Expected Output`- report$`Neural Net Output`)
  rmse <- sqrt(mean(error ^ 2))
  
  # MAE
  mae <- mae(report$`Expected Output`, report$`Neural Net Output`)
  
  #MAPE
  mape <- mean(abs((report$`Expected Output`- report$`Neural Net Output`) / report$`Expected Output`)) * 100
  
  print(rmse)
  print(mae)
  print(mape)
  
  View(report)
}

# t-1
# preparing I/O vectors for train and test for t-1
train.vector.set1 <- prepare.data(normalized.train.set, 1, 0)
test.vector.set1 <- prepare.data(normalized.test.set, 1, 0)

# one hidden layer
set.seed(42)
nn1 <- neuralnet(y ~ V1 + V2 + V3, data = train.vector.set1, hidden = 5, act.fct = 'logistic')
evaluate.model(nn1, test.vector.set1)

set.seed(42)
nn2 <- neuralnet(y ~ V1 + V2 + V3, data = train.vector.set1, hidden = 5, act.fct = 'tanh')
evaluate.model(nn2, test.vector.set1)

set.seed(42)
nn3 <- neuralnet(y ~ V1 + V2 + V3, data = train.vector.set1, hidden = 10, act.fct = 'logistic')
evaluate.model(nn3, test.vector.set1)

set.seed(42)
nn4 <- neuralnet(y ~ V1 + V2 + V3, data = train.vector.set1, hidden = 10, act.fct = 'tanh')
evaluate.model(nn4, test.vector.set1)

# two hidden layer
set.seed(42)
nn5 <- neuralnet(y ~ V1 + V2 + V3, data = train.vector.set1, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn5, test.vector.set1)

set.seed(42)
nn6 <- neuralnet(y ~ V1 + V2 + V3, data = train.vector.set1, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn6, test.vector.set1)


# t-1 with t-7
# preparing I/O vectors for train and test for t-1
train.vector.set2 <- prepare.data(normalized.train.set, 1, 7)
test.vector.set2 <- prepare.data(normalized.test.set, 1, 7)

# one hidden layer
set.seed(42)
nn7 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set2, hidden = 5, act.fct = 'logistic')
evaluate.model(nn7, test.vector.set2)

set.seed(42)
nn8 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set2, hidden = 5, act.fct = 'tanh')
evaluate.model(nn8, test.vector.set2)

set.seed(42)
nn9 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set2, hidden = 10, act.fct = 'logistic')
evaluate.model(nn9, test.vector.set2)

set.seed(42)
nn10 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set2, hidden = 10, act.fct = 'tanh')
evaluate.model(nn10, test.vector.set2)

# two hidden layer
set.seed(42)
nn11 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set2, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn11, test.vector.set2)

set.seed(42)
nn12 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set2, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn12, test.vector.set2)


# t-2
# preparing I/O vectors for train and test for t-2
train.vector.set3 <- prepare.data(normalized.train.set, 2, 0)
test.vector.set3 <- prepare.data(normalized.test.set, 2, 0)

# one hidden layer
set.seed(42)
nn13 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set3, hidden = 5, act.fct = 'logistic')
evaluate.model(nn13, test.vector.set3)

set.seed(42)
nn14 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set3, hidden = 5, act.fct = 'tanh')
evaluate.model(nn14, test.vector.set3)

set.seed(42)
nn15 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set3, hidden = 10, act.fct = 'logistic')
evaluate.model(nn15, test.vector.set3)

set.seed(42)
nn16 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set3, hidden = 10, act.fct = 'tanh')
evaluate.model(nn16, test.vector.set3)

# two hidden layer
set.seed(42)
nn17 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set3, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn17, test.vector.set3)

set.seed(42)
nn18 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6, data = train.vector.set3, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn18, test.vector.set3)


# t-2 with t-7
# preparing I/O vectors for train and test for t-1
train.vector.set4 <- prepare.data(normalized.train.set, 2, 7)
test.vector.set4 <- prepare.data(normalized.test.set, 2, 7)

# one hidden layer
set.seed(42)
nn19 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set4, hidden = 5, act.fct = 'logistic')
evaluate.model(nn19, test.vector.set4)

set.seed(42)
nn20 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set4, hidden = 5, act.fct = 'tanh')
evaluate.model(nn20, test.vector.set4)

set.seed(42)
nn21 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set4, hidden = 10, act.fct = 'logistic')
evaluate.model(nn21, test.vector.set4)

set.seed(42)
nn22 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set4, hidden = 10, act.fct = 'tanh')
evaluate.model(nn22, test.vector.set4)

# two hidden layer
set.seed(42)
nn23 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set4, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn23, test.vector.set4)

set.seed(42)
nn24 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set4, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn24, test.vector.set4)


# t-3
# preparing I/O vectors for train and test for t-3
train.vector.set5 <- prepare.data(normalized.train.set, 3, 0)
test.vector.set5 <- prepare.data(normalized.test.set, 3, 0)

# one hidden layer
set.seed(42)
nn25 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set5, hidden = 5, act.fct = 'logistic')
evaluate.model(nn25, test.vector.set5)

set.seed(42)
nn26 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set5, hidden = 5, act.fct = 'tanh')
evaluate.model(nn26, test.vector.set5)

set.seed(42)
nn27 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set5, hidden = 10, act.fct = 'logistic')
evaluate.model(nn27, test.vector.set5)

set.seed(42)
nn28 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set5, hidden = 10, act.fct = 'tanh')
evaluate.model(nn28, test.vector.set5)

# two hidden layer
set.seed(42)
nn29 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set5, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn29, test.vector.set5)

set.seed(42)
nn30 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = train.vector.set5, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn30, test.vector.set5)


# t-3 with t-7
# preparing I/O vectors for train and test for t-1
train.vector.set6 <- prepare.data(normalized.train.set, 3, 7)
test.vector.set6 <- prepare.data(normalized.test.set, 3, 7)

# one hidden layer
set.seed(42)
nn31 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set6, hidden = 5, act.fct = 'logistic')
evaluate.model(nn31, test.vector.set6)

set.seed(42)
nn32 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set6, hidden = 5, act.fct = 'tanh')
evaluate.model(nn32, test.vector.set6)

set.seed(42)
nn33 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set6, hidden = 10, act.fct = 'logistic')
evaluate.model(nn33, test.vector.set6)

set.seed(42)
nn34 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set6, hidden = 10, act.fct = 'tanh')
evaluate.model(nn34, test.vector.set6)

# two hidden layer
set.seed(42)
nn35 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set6, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn35, test.vector.set6)

set.seed(42)
nn36 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set6, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn36, test.vector.set6)


# t-4
# preparing I/O vectors for train and test for t-4
train.vector.set7 <- prepare.data(normalized.train.set, 4, 0)
test.vector.set7 <- prepare.data(normalized.test.set, 4, 0)

# one hidden layer
set.seed(42)
nn37 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set7, hidden = 5, act.fct = 'logistic')
evaluate.model(nn37, test.vector.set7)

set.seed(42)
nn38 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set7, hidden = 5, act.fct = 'tanh')
evaluate.model(nn38, test.vector.set7)

set.seed(42)
nn39 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set7, hidden = 10, act.fct = 'logistic')
evaluate.model(nn39, test.vector.set7)

set.seed(42)
nn40 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set7, hidden = 10, act.fct = 'tanh')
evaluate.model(nn40, test.vector.set7)

# two hidden layer
set.seed(42)
nn41 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set7, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn41, test.vector.set7)

set.seed(42)
nn42 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12, data = train.vector.set7, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn42, test.vector.set7)


# t-4 with t-7
# preparing I/O vectors for train and test for t-1
train.vector.set8 <- prepare.data(normalized.train.set, 4, 7)
test.vector.set8 <- prepare.data(normalized.test.set, 4, 7)

# one hidden layer
set.seed(42)
nn43 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15, data = train.vector.set8, hidden = 5, act.fct = 'logistic')
evaluate.model(nn43, test.vector.set8)

set.seed(42)
nn44 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15, data = train.vector.set8, hidden = 5, act.fct = 'tanh')
evaluate.model(nn44, test.vector.set8)

set.seed(42)
nn45 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15, data = train.vector.set8, hidden = 10, act.fct = 'logistic')
evaluate.model(nn45, test.vector.set8)

set.seed(42)
nn46 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15, data = train.vector.set8, hidden = 10, act.fct = 'tanh')
evaluate.model(nn46, test.vector.set8)

# two hidden layer
set.seed(42)
nn47 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15, data = train.vector.set8, hidden = c(10, 8), act.fct = 'logistic')
evaluate.model(nn47, test.vector.set8)

set.seed(42)
nn48 <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15, data = train.vector.set8, hidden = c(10, 8), act.fct = 'tanh')
evaluate.model(nn48, test.vector.set8)