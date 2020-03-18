Predict Customer Churn with Artificial Neural Network Algorithm
================

We will use ANN algorithm to predict if the bank customer will stay at
the bank or churn out. The dependent variable Exited has two values. 1
means exited, 0 means stays at the bank.

``` r
# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
head(dataset)
```

    ##   RowNumber CustomerId  Surname CreditScore Geography Gender Age Tenure
    ## 1         1   15634602 Hargrave         619    France Female  42      2
    ## 2         2   15647311     Hill         608     Spain Female  41      1
    ## 3         3   15619304     Onio         502    France Female  42      8
    ## 4         4   15701354     Boni         699    France Female  39      1
    ## 5         5   15737888 Mitchell         850     Spain Female  43      2
    ## 6         6   15574012      Chu         645     Spain   Male  44      8
    ##     Balance NumOfProducts HasCrCard IsActiveMember EstimatedSalary Exited
    ## 1      0.00             1         1              1       101348.88      1
    ## 2  83807.86             1         0              1       112542.58      0
    ## 3 159660.80             3         1              0       113931.57      1
    ## 4      0.00             2         0              0        93826.63      0
    ## 5 125510.82             1         1              1        79084.10      0
    ## 6 113755.78             2         1              0       149756.71      1

We will choose the variables which can has an impact on dependent
variable.

``` r
dataset = dataset[4:14]
head(dataset)
```

    ##   CreditScore Geography Gender Age Tenure   Balance NumOfProducts
    ## 1         619    France Female  42      2      0.00             1
    ## 2         608     Spain Female  41      1  83807.86             1
    ## 3         502    France Female  42      8 159660.80             3
    ## 4         699    France Female  39      1      0.00             2
    ## 5         850     Spain Female  43      2 125510.82             1
    ## 6         645     Spain   Male  44      8 113755.78             2
    ##   HasCrCard IsActiveMember EstimatedSalary Exited
    ## 1         1              1       101348.88      1
    ## 2         0              1       112542.58      0
    ## 3         1              0       113931.57      1
    ## 4         0              0        93826.63      0
    ## 5         1              1        79084.10      0
    ## 6         1              0       149756.71      1

``` r
# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))
str(dataset)
```

    ## 'data.frame':    10000 obs. of  11 variables:
    ##  $ CreditScore    : int  619 608 502 699 850 645 822 376 501 684 ...
    ##  $ Geography      : num  1 2 1 1 2 2 1 3 1 1 ...
    ##  $ Gender         : num  1 1 1 1 1 2 2 1 2 2 ...
    ##  $ Age            : int  42 41 42 39 43 44 50 29 44 27 ...
    ##  $ Tenure         : int  2 1 8 1 2 8 7 4 4 2 ...
    ##  $ Balance        : num  0 83808 159661 0 125511 ...
    ##  $ NumOfProducts  : int  1 1 3 2 1 2 2 4 2 1 ...
    ##  $ HasCrCard      : int  1 0 1 0 1 1 1 1 0 1 ...
    ##  $ IsActiveMember : int  1 1 0 0 1 0 1 0 1 1 ...
    ##  $ EstimatedSalary: num  101349 112543 113932 93827 79084 ...
    ##  $ Exited         : int  1 0 1 0 0 1 0 1 0 0 ...

``` r
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
```

``` r
# Feature Scaling
training_set[-11] = scale(training_set[-11]) # scale all variables except dependent variable
test_set[-11] = scale(test_set[-11])
```

Fitting ANN to the Training set

``` r
# install.packages('h2o')
library(h2o)
```

    ## Warning: package 'h2o' was built under R version 3.6.3

    ## 
    ## ----------------------------------------------------------------------
    ## 
    ## Your next step is to start H2O:
    ##     > h2o.init()
    ## 
    ## For H2O package documentation, ask for help:
    ##     > ??h2o
    ## 
    ## After starting H2O, you can use the Web UI at http://localhost:54321
    ## For more information visit http://docs.h2o.ai
    ## 
    ## ----------------------------------------------------------------------

    ## 
    ## Attaching package: 'h2o'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cor, sd, var

    ## The following objects are masked from 'package:base':
    ## 
    ##     %*%, %in%, &&, ||, apply, as.factor, as.numeric, colnames,
    ##     colnames<-, ifelse, is.character, is.factor, is.numeric, log,
    ##     log10, log1p, log2, round, signif, trunc

``` r
h2o.init(nthreads = -1) # allows you to connect specific server you need for ANN model
```

    ##  Connection successful!
    ## 
    ## R is connected to the H2O cluster: 
    ##     H2O cluster uptime:         9 minutes 6 seconds 
    ##     H2O cluster timezone:       America/Chicago 
    ##     H2O data parsing timezone:  UTC 
    ##     H2O cluster version:        3.28.0.4 
    ##     H2O cluster version age:    22 days  
    ##     H2O cluster name:           H2O_started_from_R_senihmerve_cti312 
    ##     H2O cluster total nodes:    1 
    ##     H2O cluster total memory:   1.41 GB 
    ##     H2O cluster total cores:    8 
    ##     H2O cluster allowed cores:  8 
    ##     H2O cluster healthy:        TRUE 
    ##     H2O Connection ip:          localhost 
    ##     H2O Connection port:        54321 
    ##     H2O Connection proxy:       NA 
    ##     H2O Internal Security:      FALSE 
    ##     H2O API Extensions:         Amazon S3, Algos, AutoML, Core V3, TargetEncoder, Core V4 
    ##     R Version:                  R version 3.6.1 (2019-07-05)

``` r
model = h2o.deeplearning(y = 'Exited', # dependent variable
                         training_frame = as.h2o(training_set), # convert training set to a H2O frame
                         activation = 'Rectifier',
                         hidden = c(5,5), # I choose 5 neural networks. Practically choose the half number of independent variables. We have 10 independent variables. So I choose 5 for it.
                         epochs = 100,
                         train_samples_per_iteration = -2) # -2 autotunes the model
```

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |=================================================================| 100%
    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |==========================                                       |  40%
      |                                                                       
      |=================================================================| 100%

## Note: Since we are connected to a server, building of this model will take shorter time than it takes in python. This is the advantage of H2O package.

``` r
# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11])) # predicts the probability of customer exited.
```

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |=================================================================| 100%
    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |=================================================================| 100%

``` r
y_pred = (y_pred > 0.5)  # if probability greater than 0.5 then customer exits. Otherwise stays at the bank.
```

## Note: If you predict a sensitive information like if the tumour is benign or malignent, you can take higher threshold value a higher number like 0.8

``` r
y_pred = as.vector(y_pred) # We convert daataset to vector as we need vector for implementing confusion matrix

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
cm
```

    ##    y_pred
    ##        0    1
    ##   0 1529   64
    ##   1  202  205

``` r
Accuracy = (1535+203) / 2000
# Note: We got an accuracy of 0.869 which is wonderfull. If we do k fold cross validation we can get better accuracy.
```

``` r
h2o.shutdown() # disconnects from server
```

    ## Are you sure you want to shutdown the H2O instance running at http://localhost:54321/ (Y/N)?
