install.packages("MASS")
library(MASS)   # install MASS pkg  
attach(Boston)  # load MASS lib
?Boston         # attach dataset
help("Boston")  # help fcn
head(Boston)    # show head of dataset
dim(Boston)     # dimensions of dataset
names(Boston)   # col names
str(Boston)     # shows structure of dataset
nrow(Boston)    # num rows
ncol(Boston)    # num cols
summary(Boston) # summary statistics
summary(Boston$crim)  # summary of "crim" col in Boston dataset

install.packages("ISLR")
library(ISLR)
data(Auto)
head(Auto)
names(Auto)
summary(Auto)
summary(Auto$mpg)
fivenum(Auto$mpg) # 5 number summary
boxplot(Auto$mpg)
hist(Auto$mpg)
summary(Auto$horsepower)
summary(Auto$weight)
fivenum(Auto$weight)
boxplot(Auto$weight)
mean(Auto$weight)
median(Auto$weight)

# data1 <- read.csv(file.choose(), header = TRUE)
setwd("/Users/silenr/Documents/CSCI-4960/lab1")
data1 <- read.csv("./EPI_data.csv", header = TRUE)
data1
EPI = data1$EPI
summary(EPI)
boxplot(EPI)
fivenum(EPI,na.rm=TRUE)   # na.rm removes all n/a values
hist(EPI)
