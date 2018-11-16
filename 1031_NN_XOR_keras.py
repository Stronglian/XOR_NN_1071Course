# -*- coding: utf-8 -*-
"""
XOR
check version:
    import keras; print(keras.__version__)
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size  = 4
epochs      = 2000


# the data, split between train and test sets
#num_classes = 2
#x_train = np.array([[ 1, 1], [ 1, 0], [ 0, 1], [ 0, 0]])#, dtype=float)
#y_train = np.array([[ 0, 1], [ 1, 0], [ 1, 0], [ 0, 0]])#, dtype=float)
#x_test  = np.array([[ 1, 1], [ 0, 0], [ 1, 0], [ 0, 1]])#, dtype=float)
#y_test  = np.array([[ 0, 1], [ 0, 1], [ 1, 0], [ 1, 0]])#, dtype=float)

num_classes = 1
x_train, y_train = np.array([[1,1],[0,0],[1,0],[0,1]]), np.array([[0],[0],[1],[1]])
x_test , y_test  = np.array([[1,1],[0,0],[1,0],[0,1]]), np.array([[0],[0],[1],[1]])


x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

#x_train = x_train.reshape(4, 2)
#x_test = x_test.reshape(4, 2)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)


model = Sequential()
model.add(Dense(2, activation='sigmoid', input_shape=(2,))) #<= WHY 4?
#model.add(Dense(2, activation='sigmoid'))
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))

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