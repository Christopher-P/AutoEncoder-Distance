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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from keras.datasets import cifar10
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

def ploot(n, x_test, decoded_imgs, filename='p50'):
    # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(56, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(56, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(filename + '.png')
    plt.show()
    

class Test:

    def __init__(self, size, data):
        self.size = size
        self.data = data

    def combine(self, dataA, dataB):
        self.data = np.concatenate((dataA, dataB),axis=1)

class AE:

    def __init__(self, testA, testB):
        self.testA = testA
        self.testB = testB        
        return None

    def run(self, tolerance):
        # Initial conditions for accuracy and tolerance
        #self.A = A
        self.tolerance = tolerance
        
        t1 = Test(self.testA.size + self.testA.size, None)
        t1.combine(self.testA.data, self.testA.data)

        t2 = Test(self.testB.size + self.testB.size, None)
        t2.combine(self.testB.data, self.testB.data)
        
        # Perform search
        x1 =   self.search(t1)
        x2 =   self.search(t2)

        # Output sofar
        print('x1:', x1, ' x2:', x2)
        
        # Combine tests
        t = Test(self.testA.size + self.testB.size, None)
        t.combine(self.testA.data, self.testB.data)

        # Search combination of test
        x3 = self.search(t)

        print('x3: ', x3)
        # Return results
        return (x1, x2, x3)

    def gen_model(self, test):
        model = Sequential()
        model.add(Dense(test.size, activation='relu', input_shape=(test.size, )))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(test.size, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def search(self, test):
   
        model = self.gen_model(test)
        result = self.eval(test, model)

        return result


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

x_train1 = []
x_train2 = []

# Filter out 0 and 7
for ind, val in enumerate(y_train):
    if val == 0:
        x_train1.append(x_train[ind])
    elif val == 1:
        x_train2.append(x_train[ind])
    else:
        continue        

# Limit data and Scale data
x_train1_a = np.asarray(x_train1[0:1000]) / 255
x_train2_a = np.asarray(x_train2[0:1000]) / 255

for i in range(1,11):

    # Make xtrain_2 a percentage of x_train1
    x_train1 = np.asarray(x_train1_a)
    x_train2 = np.concatenate((x_train1_a[0:1000 - (i * 100)], x_train2_a[0:(i * 100)]), axis=0)

    print(x_train1.shape)
    print(x_train2.shape)


    x_train1 = x_train1.reshape((1000, 784))
    x_train2 = x_train2.reshape((1000, 784))

    t1 = Test(784, x_train1)
    t2 = Test(784, x_train2)

    hel = AE(t1,t2)
    res = hel.run(1)
    x1, x2, x3 = res
    fancy_logger(x1, x2, x3, 1.0 - i/10.0, file_name='data-v5-ACC')

    
    # Release memory
    #K.clear_session()
    #exit()
print(res)
