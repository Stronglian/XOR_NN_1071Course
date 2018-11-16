# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:07:00 2018

@author: Easier
"""

'''
Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import numpy as np

batch_size = 4
num_classes = 1
epochs = 2000

# the data, split between train and test sets
x_train, y_train = np.array([[1,1],[0,0],[1,0],[0,1]]), np.array([[0],[0],[1],[1]])
x_test, y_test = np.array([[1,1],[0,0],[1,0],[0,1]]), np.array([[0],[0],[1],[1]])


x_train = x_train.reshape(4, 2)
x_test = x_test.reshape(4, 2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(4, activation='sigmoid', input_shape=(2,)))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='mae',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


print(model.predict( np.array([[1,1]])))



