# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:32:24 2018

@author: StrongPria
"""

import numpy as np
#%%
def Node(x1, x2, w1, w2, softfunc):
    print("in NODE", x1*w1 + x2*w2)
    return softfunc(x1*w1 + x2*w2)
#%% input
x_ = np.array([[ 1,  1],
               [ 1, -1],
               [-1,  1],
               [-1, -1]])
y_ = np.array([0, 1, 1, 0])
#%% weight
hopNum = 2

w_ = np.random.rand(hopNum)

#%% 
epochs = 1
#testFunc = lambda x : x
#testFunc = lambda x : 1/(1+np.exp(-x))
testFunc = lambda x : x != 0 and 1 or 0
#testFunc = lambda x : 0 if x == 0 else 1
#%% flow
w1_, w2_ = -0.5, 0.5
#for t in range(epochs):
for x_i in range(len(x_)):
    pair = x_[x_i]
    out = Node(pair[0], pair[1], w1_, w2_, testFunc)
    print(pair, "->", out)
        
