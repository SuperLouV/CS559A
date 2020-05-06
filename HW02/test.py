#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import math


def splitData(data):
    num = data.shape[0]
    indice = [i for i in range(num)]
    random.shuffle(indice)
    data = data[indice]

    subData = data[:, 1:4]
    labels = data[:, 8]

    train_data = subData[:int(0.5 * num)]
    train_labels = labels[:int(0.5 * num)]
    test_data = subData[int(0.5 * num):]
    test_labels = labels[:int(0.5 * num)]

    return train_data, train_labels, test_data, test_labels


def process_data(data):
    n_rows = data.shape[0]
    for i in range(3):
        for j in range(n_rows):
            unit = data[j][i] % 10
            tens = math.floor(data[j][i] / 10)
            if unit <= 5:
                data[j][i] = tens * 10
            else:
                data[j][i] = (tens + 1) * 10
    return data


def priorProbability(labels):
    '''compute the prior probability
    '''
    num1 = 0
    num0 = 0
    n = labels.shape[0]
    for i in range(n):
        if labels[i] == 1:
            num1 += 1
        elif labels[i] == 0:
            num0 += 1
    priorProDic = {}
    priorProDic[0] = num0 / n
    priorProDic[1] = num1 / n


    return priorProDic


def conditionProbability(train_data, train_labels):
    '''compute the condition probability
    '''
    n_train = train_data.shape[0]
    n_dim = train_data.shape[1]  # feature dimisions
    featureValue = []

    for j in range(n_dim):
        temp = []
        for i in range(n_train):
            if train_data[i][j] not in temp:
                temp.append(train_data[i][j])
        featureValue.append(temp)

    conditionPro = []
    labelSet = [0, 1]
    for dim in range(n_dim):  # for every feature
        tempDic = {}
        for value in featureValue[dim]:  # for every value of every feature
            for label in labelSet:
                num_labelDim = 0
                num_label_value = 0
                for i in range(n_train):
                    if train_labels[i] == label:
                        num_labelDim += 1
                        if train_data[i][dim] == value:
                            num_label_value += 1
                tempDic[str(value) + "|" + str(label)] = num_label_value / num_labelDim

        conditionPro.append(tempDic)

    return conditionPro


def naiveBayes(sample, train_data, train_labels, priorProDict, probability):
    posteriorPro = {}
    for label in [0, 1]:
        tempProb = priorProDict[label]
        for dim in range(3):
            try:
                tempProb *= probability[dim][str(sample[dim]) + "|" + str(label)]
            except:
                return
        posteriorPro[label] = tempProb

    result = sorted(posteriorPro.items(), key=lambda x: x[1], reverse=True)
    return result[0][0]


def classify(train_data, train_labels, test_data, test_labels, priorProbDict, conditioanProb):
    predictions = []
    n_test = test_data.shape[0]
    num_true = 0
    for i in range(n_test):
        test_sample = test_data[i]
        while True:
            try:
                pred = naiveBayes(test_sample, train_data, train_labels, priorProbDict, conditioanProb)
                break
            except:
                raise KeyError

        predictions.append(pred)
        if pred == test_labels[i]:
            num_true += 1

    accuracy = np.float(num_true) / n_test
    return accuracy


if __name__ == "__main__":

    path = 'pima-indians-diabetes-new.csv'
    data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None)
    data = np.array(data)

    times = 20
    acc_list = []
    for time in range(times):
        train_data, train_labels, test_data, test_labels = splitData(data)
        train_data = process_data(train_data)
        test_data = process_data(test_data)

        priorProbDict = priorProbability(train_labels)
        conditioanProb = conditionProbability(train_data, train_labels)

        accuracy = classify(train_data, train_labels, test_data, test_labels, priorProbDict, conditioanProb)
        acc_list.append(accuracy)

    mean_acc = np.mean(acc_list)
    mean_std = np.std(acc_list, ddof=1)

    mean_acc = round(mean_acc, 4)
    mean_std = round(mean_std, 4)

    print("After 20 time, the mean accuracy and standard deviation are {mean_acc} and {mean_std}"
          .format(mean_acc=str(mean_acc), mean_std=str(mean_std)))

