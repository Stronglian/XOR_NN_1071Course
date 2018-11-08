# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:20:45 2018

@author: Easier
"""
        
# Imports
import numpy as np 
      
# Each row is a training example, each column is a feature  [X1, X2]
X=np.array(([-1,-1],[1,1],[1,-1],[-1,1]), dtype=float)

# Groud Truth
y=np.array(([0],[0],[1],[1]), dtype=float)

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        
        self.learningRate = 10 #η
        self.input = x
        # considering we have 4 nodes in the hidden layer
#        self.weights1= np.random.rand(self.input.shape[1],4) 
        self.weights1 = np.array([[1.0,-1.0],[-1.0,1.0]])
        self.bias1 = np.array([[1.0,1.0]]) #θ
        # considering we have 1 output
#        self.weights2 = np.random.rand(4,1)
        self.weights2 = np.array([[1.0],[1.0]])
        self.bias2 = 1.0
        
        self.y = y
        self.output = np. zeros(y.shape)
        
    def feedforward(self):
        
        #Hidden Value
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) - self.bias1)
  
        #Output Value
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) - self.bias2)

        return self.layer2
    
    def predict(self,input_x):
        
        #Hidden Value
        self.layer1 = sigmoid(np.dot(input_x, self.weights1) - self.bias1)
  
        #Output Value
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) - self.bias2)
        
        return self.layer2
        
    def backprop(self):
        
        # 調整 Hidden -> Output的權重
        # 算法為 HiddenValue * Delta_OutPut
        # 算法為 HiddenValue * (Output(微分) * 與GT之差額)
        Delta_OutPut =  (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights2 = np.dot(self.layer1.T, Delta_OutPut) * self.learningRate
        d_bias2 = -np.sum(Delta_OutPut , axis = 0) * self.learningRate
#        print("Delta5 :",Delta_OutPut)
        # 調整 Input -> Hidden 的權重
        # 算法為 InputValue * Delta_Hidden 
        # 算法為 InputValue * (HiddenValue(微分) * HiddenToOutput_Weight * Delta_OutPut)  
        Delta_Hidden = np.dot(Delta_OutPut, self.weights2.T) * sigmoid_derivative(self.layer1)
#        print(Delta_OutPut.shape, self.weights2.shape)
#        raise IOError("NOTHING")
        d_weights1 = np.dot(self.input.T, Delta_Hidden) * self.learningRate
        d_bias1 = -np.sum(Delta_Hidden,axis = 0,keepdims = True) * self.learningRate
#        print("Delta34 :",Delta_Hidden)
        
        self.bias1 += d_bias1
        self.bias2 += d_bias2
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        

NN = NeuralNetwork(X,y)
#for i in range(44):
for i in range(1465): # trains the NN 1,000 times  
    if i % 200 ==0 and i!=1 :
        NN.learningRate = NN.learningRate * 0.5
    NN.train(X, y)
#    if i > 1:
#        continue
    print('\n')
    print ("for iteration # " + str(i+1) + "\n")
    print ("Predicted Output: \n" + str(NN.feedforward()))
    print("Weight1 : \n", NN.weights1)
    print("Weight2 : \n", NN.weights2)
    print("bias1 : \n", NN.bias1)
    print("bias2 : \n", NN.bias2)
    print(NN.learningRate)
    print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
    print('')
    
print("#"*10)

predict = np.array([[-1,1],[1,1],[-1,-1],[1,-1]])
GT = np.array([1,0,0,1])
Answer = []
for idx,data in enumerate(predict):
    print(NN.predict(data))
    if NN.predict(data) < 0.1:
        Answer.append(0)
    elif NN.predict(data) > 0.9:
        Answer.append(1)
        
Answer = np.array(Answer)

PrepareCalAcc = Answer - GT

Accuracy = len(PrepareCalAcc[PrepareCalAcc==0]) / 4 * 100

print("準確率 : ", Accuracy) 









