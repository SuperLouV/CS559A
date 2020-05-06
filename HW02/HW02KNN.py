# Yilin Lou
# 10445676
# HW02_KNN
import csv
import math
import operator
import random
import pandas as pd
import numpy as np
# read CSV by pandas
from pandas import read_csv
from sklearn.model_selection import train_test_split


#
# data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None)
#
#
# # 2-4 features among the 8 available features
# data_select = data[[1, 2, 3, 8]]
#
# # normalize data features
# # newValue = (oldValue - min)/(max - min)
# data_select_1 = data_select[1]
# data_select_2 = data_select[2]
# data_select_3 = data_select[3]
# data_select_8 = data_select[8]
#
#
# def autoNorm(data_autoNorm):
#     # get min and max values
#     minVals = data_autoNorm.min(0)
#     maxVals = data_autoNorm.max(0)
#
#     # calculate new values
#     for i in data_autoNorm:
#         newi = (i - minVals) / (maxVals - minVals)
#         data_autoNorm = data_autoNorm.replace(i, newi)
#     return data_autoNorm
#
#
# data_select.loc[:, 1] = autoNorm(data_select_1)
# data_select.loc[:, 2] = autoNorm(data_select_2)
# data_select.loc[:, 3] = autoNorm(data_select_3)

##############################################################  data has been normalized


# spliit test and train
# x = data_select.loc[:, [1, 2, 3]]
# y = data_select.loc[:, 8]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# count distance
def distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    # print(math.sqrt(distance))
    return math.sqrt(distance)


# find the K cloested neighbors
def getneighbors(trainingSet, testInstance, k):
    distances = []  # bulid a list
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    # sort the distance
    distances.sort(key=operator.itemgetter(1))
    # save the neighbors
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    # print(neighbors)
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]



def loadDataset(filename, split, trainingSet=[], testSet=[]):
    #read file as csv
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        # skip top nine lines because of not data
        next(lines, None)
        next(lines, None)
        next(lines, None)
        next(lines, None)
        next(lines, None)
        next(lines, None)
        next(lines, None)
        next(lines, None)
        next(lines, None)
        #change to list
        dataset = list(lines)
        #change data
        for x in range(len(dataset) - 1):  # x在总共的行数中遍历
            for y in range(4):
                dataset[x][y] = float(dataset[x][y + 1])
                dataset[x][4] = float(dataset[x][8])
                # cut data
            dataset[x] = dataset[x][:4]

            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])



# choose K ny list
K_KNN = [1, 5, 11]
for k in K_KNN:
    acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(10):
        trainingSet = []
        testSet = []
        split = 0.5
        loadDataset('pima-indians-diabetes.csv', split, trainingSet, testSet)  # r的作用是防止错误字符串意思
        predicitions = []
        correct = 0
        for x in range(len(testSet)):
            neighbors = getneighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            predicitions.append(result)
            if (repr(result) == repr(testSet[x][-1])):
                correct += 1
        x = (correct / (len(testSet))) * 100
        acc[i] = x
    print(acc)
    sum = 0
    for i in acc:
        sum = sum + i
    ave_accuracy = sum / 10

    standard_deviation = np.std(acc, ddof=1)
    print('The average accuracy when K is %d is %f percent' % (k, ave_accuracy))
    print('The standard deviation when K is %d is %f ' % (k, standard_deviation))


