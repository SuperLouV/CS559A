#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：CS559A -> HW03P2FLD
@IDE    ：PyCharm
@Author ：Yilin Lou
@Date   ：5/2/20 4:11 下午
@Group  ：Stevens Institute of technology
'''
import numpy as np
import matplotlib.pyplot as plt
D1 = np.array([[-2, 1],
               [-5, -4],
               [-3, 1],
               [0, -3],
               [-8, -1]]);
D2 = np.array([[2, 5], [1, 0], [5, -1], [-1, -3], [6, 1]]);

# print(D1)
# print(D2)
D = np.concatenate((D1, D2), axis=0)


#count mean and cov
def cal_cov_and_avg(samples):

    mean = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - mean
        cov_m += t * t.reshape(2, 1)
    return cov_m, mean

# c_1 c_2 which are two class
def fisher(c_1, c_2):
    cov_1, mean1 = cal_cov_and_avg(c_1)
    print(mean1)
    cov_2, mean2 = cal_cov_and_avg(c_2)
    print(mean2)
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


w = fisher(D1, D2)  # use function to FLD
print(w)            #print vector



acc=0
size=len(D)
for i in range(len(D)):
    out = judge(D[i], w, D1, D2)  # 判断所属的类别
    if i <5:            # count acc of D1

        if out:
            print("Corrtct ",D[i])
            acc+=1

        else:
            print("Uncorrect",D[i])
    else:
        if out==False:
            print("Correct ",D[i])
            acc += 1
        else:
            print("Uncorrect ",D[i])

print ("Accuracy rate is : ",acc/size)


#draw a picture


plt.scatter(D1[:, 0], D1[:, 1], c='#99CC99')
plt.scatter(D2[:, 0], D2[:, 1], c='#FFCC00')
line_x = np.arange(min(np.min(D1[:, 0]), np.min(D2[:, 0])),
                   max(np.max(D1[:, 0]), np.max(D2[:, 0])),
                   step=1)
#count rate
line_y = - (w[0] * line_x) / w[1]
plt.plot(line_x, line_y)
plt.show()
