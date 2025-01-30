#################################################################
#A. Basics
#################################################################
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#1. Variable assignment, operations, interactive language
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
a <- 5    # a = 5 would acomplish the same thing, but using <- is standard
x <- rnorm(1000)   

a+x   # scalar addition
a*x   # scalar multiplication
x^2   # square each element in x individually

# Syntax note:
# These all do the exact same thing: generate 1000 numbers from the N(0,1) distribution
x <- rnorm(n=1000, mean=0, sd=1)

x <- rnorm(n = 1000, 
           mean = 0, 
           sd = 1)

x <- rnorm(1000, 0, 1)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#2. Loading data (two ways)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# (option 1) Use 'read.csv()' function
# Reads the .csv file into a dataframe object
# Check working dir with getwd() and change with setwd()
white <- read.csv("white.csv")
red <- read.csv("red.csv")

# (option 2) Use RStudio 'Import Dataset' menu
# Use GUI to load dataset
# Save to script via GUI: History > Select line > To Source

# TODO: Read the 'red.csv' file into R


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#3. Getting info on objects, using functions, slicing data
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Returns info on the data structure of the given object
str(red)       
str(white)

# summary() function works on many different objects 
summary(red)   

# returns a vector containing the id names of the columns in a dataframe
colnames(red)  

# mean() returns the average value of a vector/column in a dataframe
mean(red$quality)

# Standard deviation
sd(red$ph)

# Correlation
cor(x = red$quality, y = red$alcohol)

# Count table
table(red$quality)


# [Practice:]
# What is the average quality rating of white wines? 
# What is the standard deviation of alcohol content of white wines?


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#4. Basic data plotting
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Histogram: plots empirical distribution of continuous variable
hist(red$pH)  

# Tip: try 'breaks = "FD" as a good default bin width
hist(red$pH, breaks = "FD") 


# Scatterplot: two continuous variables
plot(red$pH, red$chlorides, col = "tomato1") 
points(white$pH, white$chlorides, col = "grey70")


# You can specify more options for plots like so
plot(x = red$density, 
     y = red$alcohol,
     type = "p",       # type of plot; 'p' = points
     main = "My plot", # main title
     xlab = "density",      # x axis label
     ylab = "alcohol", # y axis label
     col = "tomato1",  # point color (see R Colors: )
     pch = 19,         # point character (see: )
     cex = 0.75)       # point size modifier (1 = default, .75 = smaller)) 



#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#5. Installing and loading packages 
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# INSTALLATION
# (option 1) Use 'install.packages("packagename")' 
install.packages("dplyr") # Remember to use quotes! (single or double)

# (option 2) Use the GUI: 
# Tools > 'Install Pakckages'

# LOADING
library(dplyr) # Loads the package. Include at the top of each R script that uses the package

# See what functions are in a library like by typing:
#        dplyr::
# and then scroll through the autocompleted list 


#################################################################
# B. Manipulating data
#################################################################
# Tip: the merge() function can be used to do a SQL-like joion of two datasets

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 7. Data structures and data types:  
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# In R, there are 5 basic data structures that are commonly used for data science:
# 1. Vector: (officially, an atomic vector): 1-D list of data elements of the same data type
# 2. Matrix: 2-D grid of data elements of the same data type
# 3. Dataframe: 2-D grid of vectors of different types (each column has same type)
# 4. Array: N-dimensional array of data elements of the same data type
# 5. List: 1-D list of any object type

# There are 6 atomic data types that can compose a data structure: 
# 1. Logical: TRUE or FALSE   
# 2. Numeric: floating point numbers (e.g. 3.14159)
# 3. Integers: integer


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 7. Combining and removing data (c(a,b,c), rbind(), cbind())
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
color.red <- rep(c("red"), times = nrow(red))
color.white <- rep(c("white"), times = nrow(white))

# Combining data
color <- c(color.red, color.white)
wine  <- rbind(red, white)     # if ncol(red) == ncol(white), then rbind is valid
wine  <- cbind(wine, color)    # if nrow(wine) == nrow(color), then cbind is valid

str(wine) # str() = sanity check


# Remove unused data
rm(red)
rm(white, color.red, color.white, color)


# Much easier way to accomplish create a new column:
# red$color <- 'red'
# white$color <- 'white'


# Convert wine$color to a factor (categorical variable)
class(wine$color)

wine$color <- as.factor(wine$color)

class(wine$color)
levels(wine$color)



#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 8. Data querying/slicing: selecting rows
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Selecting rows by index:
wine[99, ]        # select the 99th row in the wine dataframe
wine[1:500, ]     # select the first 500 rows of the wine dataframe

# Selecting rows that match a condition
# option A: create a row index with which()
idx.a <- which(wine$quality >= 8)
wine[idx.a, ]     # the blank space after the comma means we use ALL the columns in wine

# option B: return the rows that match a condition
idx.b <- wine$quality >= 8
wine[idx.b, ]

# Selecting columns 
# Single column:
wine$chlorides
wine[ , 5]

# Multiple columns: 
# Option A: a vector of column names
# Tip: use colnames(wine) to check the column names and index numbers
wine[idx.b, c('chlorides', 'pH')]

# Option B: a vector of column indexes
wine[idx.b, c(5, 9)]

# Option C: a range of columns
wine[idx.b, c(2:9, 11)]

# Option D: use dplyr filter(), especially for multiple conditions



#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 9. Writing a basic function
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Option A: clearer
count_na <- function(vec){
  N <- sum(is.na(vec))
  return(N)
  }

# Option B: simpler
count_na <- function(x){sum(is.na(x))}

# Unit test the function
str(count_na)    # count.NA is an object of the type "function"

test <- wine$pH
test[sample(1:length(test), 200)] <- NA

count_na(test)

# sapply: apply a function over a list of columns
sapply(wine, FUN = count_na)


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 10. Loops 
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Basic loop
for(i in 1:nrow(wine)){
    print(paste0("This is row", i, " in wine"))
}


# Loop with conditional statement
for(i in 1:nrow(wine)){
  if(i %% 100 == 0){
    print(paste0("This is the ", i, "th row in wine"))
    print(wine[i, 5])
  }
}

#################################################################
# C. Predictive modeling with caret
#################################################################
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 11. Split data into random training / test subsets
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
idx.train <- sample(x = 1:nrow(wine), size = 5000)

wine.train <- wine[idx.train, ]
wine.test <- wine[-idx.train, ] 


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 12a. Build basic linear regression model
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Train a linear regression model to predict quality vs. (alcohol and pH)
mod.linreg <- glm(quality ~ pH + alcohol, data = wine.train)
summary(mod.linreg)

# Use linear regression model to estimate E[quality | data]
preds.linreg <- predict(mod.linreg, newdata = wine.test)
summary(preds.linreg)
preds.linreg[99]
wine.test$quality[99]

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 12b. Build basic logistic regression model
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Train a logistic regression model to predict color vs. (all variables .)
mod.logreg <- glm(color ~ ., data = wine.train, family = "binomial")

# Use model to estimate P(wine color | data) 
# Note: type = "response" estimates the 
preds.logreg <- predict(mod.logreg, newdata = wine.test, type = "response")


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 13. Plot training data with caret
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
library(caret)

# Plot a scatterplot matrix
featurePlot(x = wine.train[ , 1:4],           # Select columns to plot
            y = wine.train$color,             # Dependent variable (factor)
            plot = "pairs",                   # Type of featurePlot
            auto.key = list(columns = 2))     # Add legend to plot

# TODO: copy + paste the above, change "pairs" to "ellpise" and change column ids
# TODO: plot boxplots (plot = 'boxplot'). Add 'layout = c(6,1)' argument.


# For density plot, the axis is unhelpuflly the same for each variable
# Must center and scale data to fix plot
featurePlot(x = wine.train[ , 1:4],           
            y = wine.train$color,             
            plot = "density",                 
            auto.key = list(columns = 2))     


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 14. Preprocess data with caret
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::# 
# Estimate parameters of transformation for centering and scaling data

# [!!! IMPORTANT !!!]: 
# ONLY USE TRAINING DATA TO CALCULATE PREPROCESSING PARAMETERS 
# NEVER USE TEST DATA (this is a form of data leakage)
preprocess_wine <- preProcess(wine.train, method = c("center", "scale"))

# Use predict() to apply preprocess_wine to each dataset
wine.train <- predict(preprocess_wine, wine.train)
wine.test <- predict(preprocess_wine, wine.test)


# Now, retry density featurePlot() from above



#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 14. Train predictive models with caret
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Set cross-validation method and parameters
crossval.ctrl <- trainControl(method = "repeatedcv",
                              number = 3,
                              repeats = 2)

# Train random forest model using caret
# NOTE: training will fail if there are any NA values in dataset
# Available models: http://topepo.github.io/caret/available-models.html
mod.rf <- train(color ~ ., 
                data = wine.train, 
                method = "rf", 
                trControl = crossval.ctrl,
                na.action=na.exclude,
                verbose = TRUE)

mod.rf



# Train stochastic gradient boosting model with custom hyperparameter grid
# For GBM, there are 4 hyperparameters to tune:
#   1. number of iterations, i.e. trees, (called n.trees in the gbm function)
#   2. complexity of the tree, called interaction.depth
#   3. learning rate: how quickly the algorithm adapts, called shrinkage
#   4. the minimum number of training set samples in a node to commence splitting (n.minobsinnode)
tunegrid.gbm <- expand.grid(interaction.depth = c(1, 5, 9), 
                            n.trees = (1:30)*50, 
                            shrinkage = 0.1,
                            n.minobsinnode = 20)

head(tunegrid.gbm)   # expand.grid returns Cartesian product of sets of params
nrow(tunegrid.gbm)   # This is the number of individual models that will be trained per validation fold

mod.gbm <- train(color ~ ., 
                 data = wine.train, 
                 method = "gbm", 
                 trControl = crossval.ctrl,
                 tuneGrid = tunegrid.gbm,
                 na.action=na.exclude,
                 verbose = TRUE)     #change verbose to FALSE to supress output

mod.gbm



# TODO: expand with model evaluation


# Removing NA values:
#-----------------------
# Identify and remove missing values
# 1. Use sum(is.na(wine.train)). If != 0 then proceed:
# 2. Run summary(wine.train) to check for NA values in each column
# 3. which(is.na(wine.train$total_sulfur_dioxide))
# 4. replace NA values with replacement value (e.g. mean)

idx.na <- which(is.na(wine.train$total_sulfur_dioxide))
wine.train$total_sulfur_dioxide[idx.na] <- mean(wine.train$free_sulfur_dioxide)







