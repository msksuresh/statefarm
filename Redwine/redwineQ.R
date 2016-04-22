#
setwd("C:/Suresh Data science/CMTH642 Data Analytics Adv methods/Assignment3")
redwineQ <- read.table("winequality-red.csv", header= TRUE,sep = ";", strip.white = TRUE)
summary(redwineQ)
str(redwineQ)
#
#Checking the dependent variable 
# changing the quality attribute to factor type.
#
str(redwineQ$quality)
summary(redwineQ$quality)
#
# Correlation between the attributes
#
M <- cor(redwineQ)
corrplot(M, method="ellipse")
#
#
#
# Separate train and test samples
#
Index <- sample(nrow(redwineQ), floor(nrow(redwineQ)*.8))
train <- redwineQ[Index,]
test <- redwineQ[-Index,]
#
#
# We build Support Vector Machine model 
#
#
install.packages("e1071", dependencies = TRUE)
library(e1071)
attach(redwineQ)
x <- subset(redwineQ, select=-quality)
y <- as.factor(quality)
#
# Apply SVM
#
svm_model1 <- svm(x,y)
summary(svm_model1)
#
# run prediction and measure the execution time
#
pred <- predict(svm_model1,x)
system.time(pred <- predict(svm_model1,x))
table(pred,y)
#
# tuning svm to get the best cost and Gamma
#
svm_tune <- tune(svm, train.x=x, train.y=y, 
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
#
# new model with Cost = 1 and gamma = 1
#
svm_model2 <- svm(x,y, kernel="radial", cost=1, gamma=1)
summary(svm_model2)
#
# run with new model
#
pred2 <- predict(svm_model2,x)
system.time(pred2 <- predict(svm_model2,x))
table(pred2,y)
#
#
# naive bayes model
#
#
redwineQ$quality <- as.factor(redwineQ$quality)
model_nb <- naiveBayes(quality ~ ., data = redwineQ)
pred <- predict(model_nb, redwineQ[ ,-quality]) 
#
# form and display confusion matrix & overall accuracy
#
tab <- table(pred, redwineQ$quality) 
tab
sum(tab[row(tab)==col(tab)])/sum(tab)
#
# using Laplace smoothing: 
#
model3 <- naiveBayes(quality ~ ., data = redwineQ, laplace = 5)
pred3 <- predict(model3, redwineQ[,-quality]) 
tab3 <- table(pred3, redwineQ$quality) 
tab3
sum(tab3[row(tab3)==col(tab3)])/sum(tab3)
#
#
#USing KNN classifier
#
#
library(class)
#
# Normalize the data
#
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
redwineQ_a <- redwineQ[ ,-12]
redwineQ_c <- redwineQ[ , 12]
redwineQ_a_scaled <- as.data.frame(lapply(redwineQ_a, normalize))
redwine_scaled <- cbind(redwineQ_a_scaled, quality = redwineQ_c)
Index <- sample(nrow(redwine_scaled), floor(nrow(redwine_scaled)*.8))
r_train <- redwine_scaled[Index,-12]
r_test <- redwine_scaled[-Index,-12]
trainLabel <- redwine_scaled[Index,12]
testLabel <- redwine_scaled[-Index,12]
pred4 <- knn(train = r_train, test = r_test, cl = trainLabel, k=5)
pred4
tab <- table(pred4,testLabel)
sum(tab[row(tab)==col(tab)])/sum(tab)
#
# The end
#