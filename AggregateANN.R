# Get training data
agg_data <- read.csv('agg_data.csv')

# Split data for 10 cross validation
library(caTools)

split <- sample.split(agg_data, SplitRatio = 0.8)
training_set <- subset(agg_data, split == TRUE)
test_set <- subset(agg_data, split == FALSE)

# Feature Scaling
training_set[-5] <- scale(training_set[-5])
test_set[-5] <- scale(test_set[-5])


# Neural Net
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)

NN <- h2o.deeplearning(y = 'GiftAmount', 
                       training_frame = as.h2o(training_set),
                       activation = 'Rectifier',
                       hidden = c(2),
                       epochs = 100,
                       train_samples_per_iteration = -2)

# Predicting the test set results
new_predict <- h2o.predict(NN, newdata = as.h2o(test_set[-5]))

# Table showing the differences in test data outcomes
predictions <- matrix(c(test_set[,5], new_predict), nrow = 11, ncol = 2)

predictions
# Here is what this gives us for new years:
# [,1]     [,2]    
# [1,] 170497.4 170497.4
# [2,] 340760   340760  
# [3,] 1354817  1354817 
# [4,] 5258562  5258562 
# [5,] 10385117 10385117
# [6,] 20047500 20047500
# [7,] 40327472 40327472
# [8,] 28901718 28901718
# [9,] 19830732 19830732
# [10,] 22455651 22455651
# [11,] ?        ?        it made me do 11 rows    


# Predicting the training set
train_predict <- h2o.predict(NN, newdata = as.h2o(training_set[-5]))

test <- matrix(c(training_set[,5], train_predict), nrow = 38, ncol = 2)

test
# Here is what this gives me for the training data predictions

# [,1]     [,2]    
# [1,] 170497.4 170497.4
# [2,] 145800   145800  
# [3,] 141100   141100  
# [4,] 140800   140800  
# [5,] 340760   340760  
# [6,] 306408.7 306408.7
# [7,] 531970.1 531970.1
# [8,] 1426185  1426185 
# [9,] 1354817  1354817 
# [10,] 1074470  1074470 
# [11,] 1351428  1351428 
# [12,] 2447543  2447543 
# [13,] 5258562  5258562 
# [14,] 3852487  3852487 
# [15,] 5168755  5168755 
# [16,] 45420519 45420519
# [17,] 10385117 10385117
# [18,] 20404700 20404700
# [19,] 23660353 23660353
# [20,] 20524137 20524137
# [21,] 20047500 20047500
# [22,] 34701799 34701799
# [23,] 35533197 35533197
# [24,] 36163098 36163098
# [25,] 40327472 40327472
# [26,] 36398158 36398158
# [27,] 23370876 23370876
# [28,] 21104987 21104987
# [29,] 28901718 28901718
# [30,] 29982073 29982073
# [31,] 22972986 22972986
# [32,] 18753434 18753434
# [33,] 19830732 19830732
# [34,] 21199269 21199269
# [35,] 19682182 19682182
# [36,] 20931070 20931070
# [37,] 22455651 22455651
# [38,] ?        ?        It made me do 38 rows. Something about formatting
