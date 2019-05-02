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


packages = c("DescTools", "glmnet", "randomForest", "e1071", "tibble", "dplyr", "tidyr", "plyr", "psych", "DMwR", "tibble")
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
  data$TotRms_AbvGrd = NULL
  data$Alley = NULL 
  return(data)
}

data = remove_columns(data)

###################################################################################################################
# Changing some Discrete Variables to categorical variables 
###################################################################################################################

data$MS_SubClass = as.factor(as.character(data$MS_SubClass))
data$Mo_Sold = as.factor(as.character(data$Mo_Sold))
data$Year_Sold = as.factor(as.character(data$Year_Sold))


###################################################################################################################
# Feature Engineering
###################################################################################################################


data = add_column(data, Total_SF = (data$Total_Bsmt_SF + data$Gr_Liv_Area), .after = "Gr_Liv_Area")
data = add_column(data, Total_Bathrooms = data$Full_Bath + (data$Half_Bath*0.5) + data$Bsmt_Full_Bath + (data$Bsmt_Half_Bath*0.5), .after = "Full_Bath")
data = add_column(data, Remod = ifelse(data$Year_Built == data$Year_Remod_Add, 0, 1), .after = "Year_Remod_Add")
data = add_column(data, Age = as.numeric(data$Year_Sold)-data$Year_Remod_Add, .after = "Year_Remod_Add")
data = add_column(data, IsNew = ifelse(data$Year_Sold==data$Year_Built, 1, 0), .after = "Year_Sold")



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
  data$Gr_Liv_Area = log(data$Gr_Liv_Area)
  data$Total_SF = log(data$Total_SF)
  return(data)
}

###################################################################################################################
# Treating Extreme values
###################################################################################################################

train$Gr_Liv_Area = Winsorize(train$Gr_Liv_Area, probs = c(0.005, 0.99))
train$Total_Bsmt_SF = Winsorize(train$Total_Bsmt_SF,probs = c(0, 0.995))
train$First_Flr_SF = Winsorize(train$First_Flr_SF,probs = c(0.003, 0.997))
train$Total_SF = Winsorize(train$Total_SF,probs = c(0.005, 0.995))
train$Sale_Price = Winsorize(train$Sale_Price,probs = c(0.01, 0.99))

###################################################################################################################
# Modifying the test and train data 
###################################################################################################################

train = take_log(train)
test = take_log(test)

train$Sale_Price = log(train$Sale_Price)


###################################################################################################################
# Fitting a Linear Model
###################################################################################################################

X_train = data.matrix(train[,1:(length(train)-1)])
Y_train = data.matrix(train[,ncol(train)])
X_test = data.matrix(test[,1:(length(test))]) 



mylasso = function(X, y, lam, n.iter = 50)
{
  
  p = ncol(X)
  b = rep(0, p)
  
  X_mean = rep(0, p)
  y_mean = mean(y)
  
  # Centering the design matrix
  
  for(i in 1:p){
    X_mean[i] = mean(X[,i])
    X[,i] = (X[,i] - X_mean[i])
  }
  
  # Centering y
  y = y - mean(y)
  
  for(step in 1:n.iter){
    
    r = y - X%*%b
    
    for(j in 1:p){
      
      
      # 1) Update the residual vector  
      r = r + X[, j] * b[j]
      
      # 2) Apply one_step_lasso to update beta_j
      xr = sum(X[,j]*r)
      xx = sum(X[,j]^2)   
      b[j] = (abs(xr)-lam/2)/xx
      b[j] = sign(xr)*ifelse(b[j]>0,b[j],0)
      
      # 3) Update the current residual vector
      r = r - X[, j] * b[j]
      
    }
    
  }
  
  num_sum = 0
  
  # Scaling back b and adding the intercept
  for(j in 1:p){
    num_sum = num_sum + b[j]*X_mean[j]
  }
  
  b0 = y_mean - num_sum
  
  return(c(b0, b))
  
}



###################################################################################################################
# Trying to figure out a optimum lambda value
###################################################################################################################

lam.seq = exp(seq(-15, 10, length=100))
cv.out = cv.glmnet(X_train, Y_train, lambda = lam.seq, nfolds = 200)
best.lam = cv.out$lambda.min

###################################################################################################################
# Testing 
###################################################################################################################

vec1 = mylasso(X_train, Y_train, lam = best.lam, n.iter = 300)
b0 = vec1[1]
b = vec1[-1]

y.pred = b0 + X_test %*% b
y.pred = exp(y.pred)

###################################################################################################################
# RMSE
###################################################################################################################

output1 = data.frame(P.I.D, y.pred)
colnames(output1) = c('PID','Sale_Price')
write.csv(output1,'mysubmission3.txt',row.names = FALSE)

test.y = read.csv("pred_lasso.csv")

pred <- read.csv("mysubmission3.txt")
names(test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, test.y, by="PID")
sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))


end_time = Sys.time()

Run_Time = end_time - begin_time

