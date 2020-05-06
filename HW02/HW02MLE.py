# Problem 1. (60 points) Download the “Pima Indians Diabetes Database” from Canvas.
# (a) Implement a classifier using Maximum-Likelihood Estimation that takes into account features 2 to 4, among the 8 available features.
# (b) Train the classifier on the same samples and run them 10 or more times. Record the mean
# and standard deviation of the accuracy. Use 50% of the data for training and the rest for
# testing. Make sure that the two sets are disjoint.
# (c) Submit code, but not data, taking into account the assumptions made.
# Hints:
# • The cov() command in Matlab can be used to compute the necessary covariance matrices.
# • You can choose any programming language, but you will need to be able to compute the
# inverse and the determinant of 3 × 3 matrices. You will also need to randomly split into
# training and test sets multiple times.

import csv
import pandas as pd
import numpy as np

# read CSV by pandas
from sklearn.model_selection import train_test_split

data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None)

# print(data)
print(type(data))
data_select = data[[1, 2, 3, 8]]


# split test and train
# def splitdata():
#     # 2-4 features among the 8 available features
#     data_select = data[[1, 2, 3, 8]]
#     print(type(data_select))
#     a = data_select.shape[1]
#     x = data_select.loc[:, [1, 2, 3]]
#     y = data_select.loc[:, 8]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
#     return x_train,x_test,y_train,y_test


# Split the data
def split_train(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    return train, test


# Split the data in half to form the training and test
train, test = split_train(data_select, 0.5)
## count mean which lable is 0 in train
# select data which lable is 0 in train
# train table==0
train_0 = train[train[8] == 0]
# train_0=np.array(train_0)
# train table==1
train_1 = train[train[8] == 1]
# train_1=np.array(train_1)
# count mean
mean0 = np.mean(train_0)  # without T because they are vector
mean0 = mean0[:3]
print("mean0")
print(mean0)
mean1 = np.mean(train_1)
mean1 = mean1[:3]
print("mean1")
print(mean1)
# count cov
cov0 = np.cov(train_0)
print(cov0)
print(len(cov0))
cov1 = np.cov(train_1)
# print(cov1)
tmp0 = len(train_0)

tmp1 = len(train_1)

p1 = tmp0 / (tmp1 + tmp0)
p2 = tmp1 / (tmp1 + tmp0)

# print(np.linalg.inv(cov0))
# for x in range(len(test)-1):
#
#     likelihood0=exp(-((TestData(x,feature)-mean1)*inv(cov1)*(TestData(x,feature)-mean1)')/(2*pi))/sqrt(det(cov1));
# temp1=0
# temp2=0
# for i in y_train:
#     if(i==1):
#         temp1+=1
#     else:
#         temp2+=1
# p1=temp1/(temp1+temp2)
# p2=temp2/(temp1+temp2)
#
# print(temp1)
# print(temp2)
