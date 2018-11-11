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
Y= np.array([[0], [1], [1], [0]], dtype=float)
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
    def __init__(self, x, y):
        self.input = x
#        self.weights1= np.random.rand(self.input.shape[1], 4) # considering we have 4 nodes in the hidden layer
#        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
        
        #題目指定
        self.weights1 = np.array([[ 1.0, -1.0],[-1.0, 1.0]]) # w13m w23
        self.bias1    = np.array([[ 1.0,  1.0]]) #θ
        self.weights2 = np.array([[ 1.0], [ 1.0]])
        self.bias2    = 1.0 #θ
        
        self.learningRate  = 10  #η
        self.thresholdVal  = (0.1, 0.9)    
        
        
    def feedforward(self):
        #hiden Layer 1
        self.layer1 = sigmoid(np.dot( self.input, self.weights1) - self.bias1)
        #output Layer
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) - self.bias2)
        return self.layer2
    
    def predict(self, input_x):
        #hiden Layer 1
        self.layer1 = sigmoid(np.dot(    input_x, self.weights1) - self.bias1)
        #output Layer
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) - self.bias2)
        return self.layer2
        
    def backprop(self):
        #output layer
        delta_output = (self.y - self.output) * sigmoid_derivative(self.output)
#        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights2 = np.dot(self.layer1.T, delta_output) * self.learningRate
        d_bias2 = -np.sum(delta_output , axis = 0) * self.learningRate
#        print("Delta5 :", delta_output)
        
        #hiden layer 1
#        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), \
#                                                 self.weights2.T)*sigmoid_derivative(self.layer1))
        delta_hidden1 = np.dot(delta_output, self.weights2.T) * sigmoid_derivative(self.layer1)
        d_weights1 = np.dot(self.input.T, delta_hidden1) * self.learningRate
        d_bias1 = -np.sum(delta_hidden1, axis = 0, keepdims = True) * self.learningRate
#        print("Delta34 :", delta_hidden1)
        
        #校正
        self.bias1 += d_bias1
        self.bias2 += d_bias2
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        return

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        return
    def Varify(self, inputX, groundtruthY):
        """ 不確定名稱正確與否 """
        ans = []
        for i, dataX in enumerate(inputX):
            predict_y = self.predict(dataX)
            print(predict_y)
            if   predict_y < self.thresholdVal[0]:
                ans.append(0)
            elif predict_y < self.thresholdVal[1]:
                ans.append(1)
        ans = np.array(ans)
        
        accuracy = np.sum(np.abs(groundtruthY - ans)) * 100 / len(groundtruthY) 
        
        print("Accuracy:", accuracy)
#%%
NN = NeuralNetwork(X, Y)
for i in range(150): # trains the NN 1,465 times
    if i % 200 ==0 and i!=1 :
        NN.learningRate *= 0.5
    NN.train(X, Y)
    
    print ("=== for iteration # " + str(i) + "===\n")
    print ("Predicted Output: \n" + str(NN.feedforward()))
    print("Weight1 : \n", NN.weights1)
    print("Weight2 : \n", NN.weights2)
    print("bias1 : \n", NN.bias1)
    print("bias2 : \n", NN.bias2)
    print("learningRate:", NN.learningRate)
#    print ("Input : \n" + str(X))
#    print ("Actual Output: \n" + str(Y))
    print ("Loss: \n" + str(np.mean(np.square(Y - NN.feedforward())))) # mean sum squared loss
    print ("\n")
#%% 
X = np.array([[ 1,  1],
               [ 1, -1],
               [-1,  1],
               [-1, -1]], dtype=float)
Y= np.array([[0], [1], [1], [0]], dtype=float)
NN.Varify(X, Y)