#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：CS559A -> HW04FLD
@IDE    ：PyCharm
@Author ：Yilin Lou
@Date   ：5/3/20 5:37 下午
@Group  ：Stevens Institute of technology
'''

'''
i written MLE with matlab , so i use python to FLD data to get CSV file and then 
use matlab to analyse data
'''
from numpy import *
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None)
# seprate data
data1 = data[data.loc[:, 8] == 1]
data2 = data[data.loc[:, 8] == 0]
# cur data used  to  train FLD
da1=data1.loc[:,0:7]
da2=data2.loc[:,0:7]
# lable1=data1.loc[:, 8];
# lable2=data2.loc[:, 8];


# change to array
print(type(data1))
data1_array=np.array(da1)
data2_array=np.array(da2)
data=np.array(data)

#count mean and cov
def cal_cov_and_avg(samples):

    mean = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - mean
        cov_m += t * t.reshape(8, 1)
    return cov_m, mean

# c_1 c_2 which are two class
def fisher(c_1, c_2):
    cov_1, mean1 = cal_cov_and_avg(c_1)
    # print(mean1)
    cov_2, mean2 = cal_cov_and_avg(c_2)
    # print(mean2)
    s_w = cov_1 + cov_2
    u, s, v = np.linalg.svd(s_w)
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(s_w_inv, mean1 - mean2)

# if True if Class1 else Class2
def judge(sample, w, c_1, c_2):

    u1 = np.mean(c_1, axis=0)
    u2 = np.mean(c_2, axis=0)
    center_1 = np.dot(w.T, u1)
    center_2 = np.dot(w.T, u2)
    pos = np.dot(w.T, sample)
    return abs(pos - center_1) < abs(pos - center_2)

#
w = fisher(data1_array, data2_array)  # use function to FLD
print('FLD vector is : ',w)            #print vector

print(np.dot(da1,w).shape)
# process new data which lable is 1
newdata1=np.dot(da1,w)
# process new data which lable is 0
newdata2=np.dot(da2,w)


newdata1 = pd.DataFrame(newdata1)
newdata2 = pd.DataFrame(newdata2)
# print(newdata1)
# print(pd.concat([newdata1,lable1],axis=1))

lable1=np.full(len(newdata1),1)
lable2=np.full(len(newdata2),0)
lable1 = pd.DataFrame(lable1)
lable2 = pd.DataFrame(lable2)

data1_FLD=pd.concat([newdata1,lable1],axis=1)
data2_FLD=pd.concat([newdata2,lable2],axis=1)

data_FLD=pd.concat([data1_FLD,data2_FLD])


data_FLD.to_csv('data_FLD.csv', index=False, header=False)
# acc=0
# D = np.concatenate((data1_array, data2_array), axis=0)
# size=len(D)
# for i in range(len(D)):
#     out = judge(D[i], w, data1_array, data1_array)  # 判断所属的类别
#     if i <268:            # count acc of D1
#
#         if out:
#             print("Corrtct ",D[i])
#             acc+=1
#
#         else:
#             print("Uncorrect",D[i])
#
#     else:
#         if out==False:
#             print("Correct ",D[i])
#             acc += 1
#         else:
#             print("Uncorrect ",D[i])
#
# print ("Accuracy rate is : ",acc/size)


