#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：CS559A -> HW03P1PCA
@IDE    ：PyCharm
@Author ：Yilin Lou
@Date   ：5/1/20 5:28 下午
@Group  ：Stevens Institute of technology
'''

from numpy import *
import pandas as pd
import numpy as np
from numpy import mat, argsort, mean, cov
from numpy.linalg import linalg
from pandas import DataFrame



def pca(dataMat, topNfeat):
    meanValues = mean(dataMat,axis=0) # count mean
    meanRemoved = dataMat - meanValues  # subtract mean

    covMat = cov(meanRemoved,rowvar=0)  # count cov every feature
    eigVals, eigVects = linalg.eig(mat(covMat)) # 求特征值和特征向量
    print(eigVals)
    eigValInd = argsort(-eigVals)  # sort from big to small， 1×n
    print(eigValInd)

    eigValInd = eigValInd[:topNfeat]  # select top feature  1×r
    print(eigValInd)
    redEigVects = eigVects[:,eigValInd] # choose top 3
    # print(meanRemoved.shape)
    # print(redEigVects.shape)
    # print(type(meanRemoved))
    print(redEigVects)



    lowDDataMat = meanRemoved * redEigVects  # m×r Y=X*P
    # reconMat = (lowDDataMat * redEigVects.T) + meanValues
    return lowDDataMat




if __name__ == '__main__':
    data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None)
    # print(meanX(data))
    k = 2
    newdata =data.loc[:, :7]
    newdata=array(newdata)
    data1_PCA=pca(newdata,3)

    data_PCA = pd.DataFrame(data1_PCA)
    y = data.loc[:, 8]  # the eighth line is lable
    newdata = pd.concat([data_PCA, y], axis=1)

    newdata.to_csv('HW03P1PCA.csv', index=False, header=False)


