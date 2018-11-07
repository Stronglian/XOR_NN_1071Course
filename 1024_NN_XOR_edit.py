# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:43:14 2018

learning rate: η ita
"""
import numpy as np
#%% 輸入
X = np.array([[ 1,  1],
               [ 1, -1],
               [-1,  1],
               [-1, -1]], dtype=float)
Y= np.array([0, 1, 1, 0], dtype=float)
#%% 
# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)
#%%
    
# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
#        self.weights1= np.random.rand(self.input.shape[1], 4) # considering we have 4 nodes in the hidden layer
#        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np. zeros(y.shape)
        
        #題目指定
        self.weights1= np.array([1.0, -1.0]) # w13m w23
        self.thresholdVal = 0 
        self.learningRate = 0.1 
        self.delta = 0
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
#        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer1
        
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), \
                                                 self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
#        self.output = self.feedforward()
#        self.backprop()
        self.oneLearningCircle()
        return
    
    def oneLearningCircle(self, printTF = True):
        for i in range(len(self.y)):
            x_i = self.input[i]
            y_i = self.y
            
            net = 0
            out = 0
            
            w_tmp = 0
            w_delta = [0,0]
            rateDelate = 0
            
            
            print("%d %2d %2d | %2d %2d %2d | %2.2f %2.2f | %2.2f | %2.2f %2.2f %2.2f"%
                  (x_i[0], x_i[1], y_i,
                   net, out, self.delta,
                   w_tmp[0], w_tmp[1],
                   self.learningRate,
                   w_delta[0], w_delta[1], rateDelate))
        return
#%%
NN = NeuralNetwork(X, Y)
#for i in range(1500): # trains the NN 1,000 times
#    if i % 100 ==0: 
#        print ("for iteration # " + str(i) + "\n")
#        print ("Input : \n" + str(X))
#        print ("Actual Output: \n" + str(y))
#        print ("Predicted Output: \n" + str(NN.feedforward()))
#        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
#        print ("\n")
#  
NN.train(X, Y)
