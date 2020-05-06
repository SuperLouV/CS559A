#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：CS559A -> HW03P4SSR
@IDE    ：PyCharm
@Author ：Yilin Lou
@Date   ：5/2/20 4:31 下午
@Group  ：Stevens Institute of technology
'''
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib  as mpl

'''
ω2 1 1 -1 0 2
ω1 0 0 1 2 0
ω2 -1 -1 1 1 0
ω1 4 0 1 2 1
ω1 -1 1 1 1 0
ω1 -1 -1 -1 1 0
ω2 -1 1 1 2 1
'''
# add an extra line
# X = np.array([[-1, 1, 1, -1, 0, 2],
#               [1, 0, 0, 1, 2, 0],
#               [-1, -1, -1, 1, 1, 0],
#               [1, 4, 0, 1, 2, 1],
#               [1, -1, 1, 1, 1, 0],
#               [1, -1, -1, -1, 1, 0],
#               [-1, -1, 1, 1, 2, 1]]);
# Replace all examples from class 2 by their negative values
X = np.array([[-1, -1, -1, 1, 0, -2],
              [1, 0, 0, 1, 2, 0],
              [-1, 1, 1, -1, -1, 0],
              [1, 4, 0, 1, 2, 1],
              [1, -1, 1, 1, 1, 0],
              [1, -1, -1, -1, 1, 0],
              [-1, 1, -1, -1, -2, -1]]);

# change w2 to -1, w1 to 1
Y = np.array([-1, 1, -1, 1, 1, 1, -1]);

X1 = np.array([[1, 1, 1, -1, -1],
              [-1, -1, -1, -1, -1],
              [-1, 1, 1, 1, -1],
              [1, 1, -1, -1, 1]]);



def perceptron_sgd(X):
    # Set the learning rate to 1
    eta = 1
    # for loop
    epochs = 100
    # Use [3 1 1 -1 2 -7] as the initial weight vector.
    w = (np.array([3, 1, 1, -1, 2, -7]));
    # w=np.array([0.25, 0.25, 0.25, 0.25, 0.25]);
    for epoch in range(epochs):
        for i, x in enumerate(X):
            # print(np.multiply(X[i], w).sum)
            if (np.multiply(X[i], w).sum()) <= 0:
                w = w + eta * X[i]
    return w
w = perceptron_sgd(X)
print(w)
print("Thus the discriminant function is:")
print ('%s%d*%s+%d*%s+%d*%s+%d*%s+%d*%s+%d*%s'%('G(Y) = ',w[0],'Y0',w[1],'Y1',w[2],'Y2',w[3],'Y3',w[4],'Y4',w[5],'Y5'))
print("Converting back to the original features x:")
print ('%s%d*%s+%d*%s+%d*%s+%d*%s+%d*%s+%d'%('G(X) = ',w[1],'Y1',w[2],'Y2',w[3],'Y3',w[4],'Y4',w[5],'Y5',-w[0]))




