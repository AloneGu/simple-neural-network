#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 10:06 AM
# @Author  : Jackling 


from itertools import permutations
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from nn import NeuralNetwork
import numpy as np


# prepare x,y
seeds = [0, 0, 0, 1, 1, 1]
x = np.array(list((permutations(seeds, 3))))


def get_xor(x):
    a = 0
    for item in x:
        a = a ^ item
    return a


y = np.array(map(get_xor, x)).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2333)

print X_train.shape
print y_train.shape

for i in range(10, 41):
    print 'training', i * 10, 'epochs',
    # build network
    network = NeuralNetwork()

    network.train(X_train, y_train, i * 10)
    # test
    res = network.think(X_test).round()
    print 'test score',accuracy_score(y_test, res),

    # test
    res = network.think(X_train).round()
    print 'train score',accuracy_score(y_train, res)
