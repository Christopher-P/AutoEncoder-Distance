# Hello World!

#TODO: Test similarity between [0] to [1] and [1] to [0]
#TODO: Test [0,0] to [1,1] and [1,1] to [0,0]

# TODO: ramp up to mnist
# TODO: Ramp up to AIQ
'''
The goal is to make a simple utility that can accept a diverse set of 
tests and then compute the distance (similarity) between the set of
presented tests!
'''
#https://medium.com/datadriveninvestor/deep-autoencoder-using-keras-b77cd3e8be95

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten
from keras.models import Model, Sequential
from keras import backend as K

from numpy.random import seed
from tensorflow import set_random_seed

from numba import cuda

#seed(3)
#set_random_seed(4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

class Test:

    def __init__(self, size, data):
        self.size = size
        self.data = data

    def combine(self, dataA, dataB):
        self.data = np.concatenate((dataA, dataB),axis=0)

class AE:

    def __init__(self, testA, testB):
        self.testA = testA
        self.testB = testB        
        return None

    def run(self, A, tolerance):
        # Initial conditions for accuracy and tolerance
        self.A = A
        self.tolerance = tolerance

        t1 = Test(self.testA.size, None)
        t1.combine(self.testA.data, self.testA.data)

        t2 = Test(self.testB.size, None)
        t2.combine(self.testB.data, self.testB.data)
        
        # Perform search
        x1 =self.search(t1)
        x2 =self.search(t2)

        # Output sofar
        print('x1:', x1, ' x2:', x2)
        
        # Combine tests
        t = Test(self.testA.size, None)
        t.combine(self.testA.data, self.testB.data)

        # Search combination of test
        x3 = self.search(t)

        print('x3: ', x3)
        # Return results
        return (x1, x2, x3)

    def gen_model(self, test, x):
        model = Sequential()
        model.add(Dense(test.size, activation='relu', input_shape=(test.size, )))
        #model.add(Dense(64, activation='relu'))
        model.add(Dense(x, activation='relu'))
        #model.add(Dense(64, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        model.add(Dense(test.size, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary()
        return model

    def search(self, test):
        # Initial Conditions
        high = test.size
        low = 1
        done = False

        # Perform search
        while not done:
            print('high: ', high, ' low:', low)
            # Test with midpoint
            x = int((high + low)/2)
            model = self.gen_model(test, x)
            result = self.eval(test, model)
            


            # Check result
            if result < self.A:
                low = x
            if result >= self.A:
                high = x

            # Check to tolerance level
            if high - low <= self.tolerance:
                done = True

            
            del result
            del model

        return x


    def eval(self, test, model):
        history = model.fit(test.data, test.data, epochs=12, batch_size=10, validation_split=0.1, verbose=0,shuffle=True)

        n = max(history.history['acc'])
        print(n)
        return n



# Logging utility
# Expects a results history
def fancy_logger(x1, x2, x3, overlap, file_name='data', write='a'):
    with open(file_name + '.csv', write, newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow((x1, x2, x3, overlap))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(y_train[0:100])
        
for i in range(10,11):

    np.random.shuffle(x_train)

    x_train1 = x_train[0:1000] / 255
    x_train2 = x_train[i * 100: i * 100 + 1000] / 255

    print(x_train1.shape)
    print(x_train2.shape)


    x_train1 = x_train1.reshape((1000, 784))
    x_train2 = x_train2.reshape((1000, 784))

    t1 = Test(784, x_train1)
    t2 = Test(784, x_train2)

    hel = AE(t1,t2)
    res = hel.run(0.815, 1)
    x1, x2, x3 = res
    fancy_logger(x1, x2, x3, 1.0 - i/10.0, file_name='data-v2')

    
    # Release memory
    #K.clear_session()
    #exit()
print(res)
