#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 10:43 AM
# @Author  : Jackling 


# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 10:06 AM
# @Author  : Jackling


from itertools import permutations
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from multi_layer import NeuralNetwork, NeuronLayer
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


for i in range(1,21):
    print 'training',i*5,'epochs',

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # build network
    network = NeuralNetwork(layer1, layer2)

    # Train the neural network using a training set.
    network.train(X_train, y_train, i*5)

    # test
    _,res = network.think(X_test)
    print 'test score',accuracy_score(y_test, res.round()),

    # test
    _,res = network.think(X_train)
    print 'train score',accuracy_score(y_train, res.round())
