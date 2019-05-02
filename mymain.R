###################################################################################################################
# Function to check if packages are installed 
###################################################################################################################

begin_time = Sys.time()

check.packages <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}


packages = c("DescTools", "glmnet", "randomForest", "e1071", "tibble", "dplyr", "tidyr", "plyr","DMwR", "tibble")
check.packages(packages)

###################################################################################################################
# Read in data
###################################################################################################################

train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Removing the outliers
train = train[!train$PID %in% c(902207130,910251050),]
#test = test[!test$PID %in% c(902207130,910251050),]

test = data.frame(test, Sale_Price = rep(0,nrow(test)))
data = rbind(train, test)




###################################################################################################################
# Function to remove unecessary or dominating categorical columns
###################################################################################################################

remove_columns = function(data){
  data$Garage_Yr_Blt = NULL
  data$Longitude = NULL
  data$Latitude = NULL
  data$Street = NULL
  data$Condition_2 = NULL
  data$Heating = NULL
  data$Utilities = NULL
  data$Roof_Matl = NULL
  data$Land_Slope = NULL
  data$Pool_QC = NULL
  data$Misc_Feature = NULL 
  data$Low_Qual_Fin_SF = NULL
  data$Three_season_porch = NULL
  data$Pool_Area = NULL
  data$Misc_Val = NULL
  return(data)
}

data = remove_columns(data)

###################################################################################################################
# Recoding some variables 
###################################################################################################################

data$MS_SubClass = as.factor(as.character(data$MS_SubClass))
data$Mo_Sold = as.factor(as.character(data$Mo_Sold))
data$Year_Sold = as.factor(as.character(data$Year_Sold))


levels(data$Lot_Shape) = c(1,2,4,3)
data$Lot_Shape = as.numeric(as.character(data$Lot_Shape))
levels(data$Overall_Qual) = c(6,5,4,9,3,7,2,10,8,1)
data$Overall_Qual = as.numeric(as.character(data$Overall_Qual))
levels(data$Exter_Qual) = c(4,1,3,2)
data$Exter_Qual = as.numeric(as.character(data$Exter_Qual))
levels(data$Exter_Cond) = c(5,2,4,1,3)
data$Exter_Cond = as.numeric(as.character(data$Exter_Cond))
levels(data$Bsmt_Qual) = c(6,3,5,1,2,4)
data$Bsmt_Qual = as.numeric(as.character(data$Bsmt_Qual))
levels(data$Bsmt_Cond) = c(6,3,5,1,2,4)
data$Bsmt_Cond = as.numeric(as.character(data$Bsmt_Cond))
levels(data$Bsmt_Exposure) = c(4,5,3,2,1)
data$Bsmt_Exposure = as.numeric(as.character(data$Bsmt_Exposure))
levels(data$Heating_QC) = c(5,2,4,1,3)
data$Heating_QC = as.numeric(as.character(data$Heating_QC))
levels(data$Kitchen_Qual) = c(5,2,4,1,3)
data$Kitchen_Qual = as.numeric(as.character(data$Kitchen_Qual))
levels(data$Functional) = c(4,3,7,6,5,1,2,8)
data$Functional = as.numeric(as.character(data$Functional))
levels(data$Fireplace_Qu) = c(6,3,5,1,2,4)
data$Fireplace_Qu = as.numeric(as.character(data$Fireplace_Qu))
levels(data$Garage_Finish) = c(4,1,3,2)
data$Garage_Finish = as.numeric(as.character(data$Garage_Finish))
levels(data$Garage_Qual) = c(6,3,5,1,2,4)
data$Garage_Qual = as.numeric(as.character(data$Garage_Qual))
levels(data$Garage_Cond) = c(6,3,5,1,2,4)
data$Garage_Cond = as.numeric(as.character(data$Garage_Cond))
levels(data$Paved_Drive) = c(1,2,3)
data$Paved_Drive = as.numeric(as.character(data$Paved_Drive))
levels(data$Fence) = c(5,4,3,2,1)
data$Fence = as.numeric(as.character(data$Fence))


###################################################################################################################
# One-hot encoding
###################################################################################################################

dmy = dummyVars("~.", data = data)
data = data.frame(predict(dmy, newdata=data))


###################################################################################################################
# Getting the splits back
###################################################################################################################

P.I.D = test$PID
train = data[1:nrow(train),]
test = data[nrow(train)+1:nrow(test), 1:ncol(data) - 1]

train$PID = NULL
test$PID = NULL

###################################################################################################################
# Function for Log Transformations
###################################################################################################################

take_log = function(data){
  data$Lot_Area = log(data$Lot_Area)
  data$BsmtFin_SF_2 = data$BsmtFin_SF_2 + 1
  data$BsmtFin_SF_2 = log(data$BsmtFin_SF_2)

  return(data)
}

###################################################################################################################
# Function to check for factor varibles
###################################################################################################################

factor_check = function(vector){
  if(any(vector>1)){
    return(FALSE)
  }
  else{
    return(TRUE)
  }
}

###################################################################################################################
# Function for treating extreme values
###################################################################################################################

treat_ext <- function(x) {
  n = ncol(x)
  for (i in 1:n) {
    if(factor_check(x[,i]) == FALSE){
      x[,i] = Winsorize(x[,i], probs = c(0.01, 0.99))
    }
  }
  return(x)
}




###################################################################################################################
# Modifying the train data 
###################################################################################################################
train = take_log(train)
test = take_log(test)

train = treat_ext(train)
#test = treat_ext(test)
train$Sale_Price = log(train$Sale_Price)


###################################################################################################################
# Fitting a Linear Model
###################################################################################################################

X_train = data.matrix(train[,1:(length(train)-1)])
Y_train = data.matrix(train[,ncol(train)])
X_test = data.matrix(test[,1:(length(test))]) 

lam.seq = exp(seq(-15, 20, length=100))
cv.out = cv.glmnet(X_train, Y_train, lambda = lam.seq, nfolds = 200)
best.lam = cv.out$lambda.min
Ytest.pred = predict(cv.out, s = best.lam, newx = X_test)
mylasso.coef = predict(cv.out, s = best.lam, type = "coefficients")
model_size = sum(mylasso.coef != 0) - 1 

Ytest.pred = exp(Ytest.pred)


###################################################################################################################
# XG Boost
###################################################################################################################

gbm.fit.final <- gbm(
  formula = Sale_Price ~ .,
  distribution = "gaussian",
  data = train[,1:(length(train))],
  n.trees = 483,
  interaction.depth = 5,
  shrinkage = 0.1,
  n.minobsinnode = 5,
  bag.fraction = .65, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)


mypred = predict(gbm.fit.final, test[,1:(length(test))], n.trees = 483)
mypred = exp(mypred)

output2 = data.frame(P.I.D, mypred)
colnames(output2) = c('PID','Sale_Price')
write.csv(output2,'mysubmission1.txt',row.names = FALSE)

test.y = read.csv("pred.csv")

pred <- read.csv("mysubmission1.txt")
names(test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, test.y, by="PID")
sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))

###################################################################################################################
# Random Forest
###################################################################################################################

rfc = randomForest(Sale_Price ~ ., data = train,importance = T, ntree = 500)

y_pred_rf = predict(rfc, X_test)
y_pred_rf = exp(y_pred_rf)

output3 = data.frame(P.I.D, y_pred_rf)
colnames(output3) = c('PID','Sale_Price')
write.csv(output3,'mysubmission2.txt',row.names = FALSE)

test.y = read.csv("pred.csv")

pred <- read.csv("mysubmission2.txt")
names(test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, test.y, by="PID")
sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))

end_time = Sys.time()

Run_Time = end_time - begin_time 
