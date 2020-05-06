#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：CS559A -> HW03Problem1
@IDE    ：PyCharm
@Author ：Yilin Lou
@Date   ：4/30/20 10:09 下午
@Group  ：Stevens Institute of technology
'''
import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # load PCA package
from sklearn.model_selection import train_test_split

data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None)

'''
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
'''
# spliit x and y
x = data.loc[:, :7]
y = data.loc[:, 8]  # the eighth line is lable

# Step PCA all features data
PCA = PCA(n_components=3)
# pca.transform(x_train):用来训练pca模型，同时返回降维后的数据，这里x_pca就是降维后的数据
x_pca = PCA.fit_transform(x)
# x_new = PCA.inverse_transform(x_pca)

data_PCA = DataFrame(x_pca)
data_PCA.to_csv('PCA.csv', index=False, header=False)

# newdata with x after PCA and y
newdata=pd.concat([data_PCA,y],axis=1)
newdata.to_csv('test.csv', index=False, header=False)

# spliit test and train 50%-50%  2
x = newdata.loc[:, :2]
y = newdata.loc[:,8]
b=newdata.shape
#info of newdata
print(newdata.info())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
print(x_train)

#计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征
def meanX(dataX):
    return np.mean(dataX,axis=0)#axis=0表示按照列来求均值，如果输入list,则axis=1
"""
参数：
    - XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
    - k：表示取前k个特征值对应的特征向量
返回值：
    - finalData：参数一指的是返回的低维矩阵，对应于输入参数二
    - reconData：参数二对应的是移动坐标轴后的矩阵
"""
def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print( "k must lower than feature number")
        return
    else:
        #注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData

if __name__ == '__main__':
    data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None)
    print(data)