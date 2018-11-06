# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:43:14 2018

learning rate: η ita
"""
import numpy as np

X = np.array([[ 1,  1],
               [ 1, -1],
               [-1,  1],
               [-1, -1]], dtype=float)
Y = np.array([0, 1, 1, 0], dtype=float)


# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)