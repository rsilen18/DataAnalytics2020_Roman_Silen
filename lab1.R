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
# EPI data
EPI = data1$EPI
View(EPI)  # view EPI data just like viewing Excel file
summary(EPI)
boxplot(EPI)
fivenum(EPI,na.rm=TRUE)   # na.rm removes all n/a values
hist(EPI)
hist(EPI, seq(30., 95., 1.0), prob=TRUE)  # makes "bins" more narrow
stem(EPI)
lines(density(EPI,na.rm=TRUE,bw=1))
lines(density(EPI,na.rm=TRUE,bw="SJ"))  # less detailed than bw=1
rug(EPI)   # adds "barcode-like" underline that shows density
plot(ecdf(EPI), do.points=FALSE, verticals=TRUE)  # cdf, w/o points plotted but connecting step-wise fcn
par(pty="s") # par = plotting parameters, pty="s" -> square plot
qqnorm(EPI)  # plots points against each other
qqline(EPI)  # creates line on Q-Q plot
# DALY data
DALY = data1$DALY
View(DALY)
fivenum(DALY,na.rm=TRUE)
hist(DALY, seq(0,100,1), prob=TRUE)
lines(density(DALY,na.rm=TRUE,bw="SJ"))
stem(DALY)
rug(DALY)
qqnorm(DALY)
qqline(DALY)
plot(ecdf(DALY), do.points = FALSE, verticals=TRUE)
# WATER_H data
WATER_H = data1$WATER_H
View(WATER_H)
fivenum(WATER_H)
fivenum(WATER_H,na.rm=TRUE)
hist(WATER_H, seq(0,100,5), prob=TRUE)
lines(density(WATER_H,na.rm=TRUE,bw="SJ")) # bw=SJ auto-fits kernel density estimates
stem(WATER_H)
qqnorm(WATER_H)
qqline(WATER_H)
plot(ecdf(WATER_H), do.points=FALSE, verticals=TRUE)
# Comparing distributions
boxplot(EPI,DALY,WATER_H)
labels(b, "EPI", "DALY", "WATER_H")
boxplot(EPI,data1$ENVHEALTH,data1$ECOSYSTEM,DALY,data1$AIR_H,WATER_H,data1$AIR_E,data1$WATER_E, data1$BIODIVERSITY)

# Exercise 2 - filtering populations
EPILand<-data1$EPI[!Landlock]

# generate your own data points
x <- seq(30,95,1)  # like linspace in MATLAB
qqplot(qt(ppoints(250), df=5),x,xlab="Q-Q plot for t dsn")
qqline(x)
