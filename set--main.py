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

from keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from keras.layers import Input, Dense, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout
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
        self.data = np.concatenate((dataA, dataB),axis=0)

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

    def gen_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 1) ))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding='same'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
         
        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))
         
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
         
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary()
        #exit()
        return model


    def search(self, test):
   
        model = self.gen_model()
        result = self.eval(test, model)

        return result


    def eval(self, test, model):
        history = model.fit(test.data, test.data, epochs=10, batch_size=10, validation_split=0.1, verbose=0,shuffle=True)

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

(x_train_m, y_train), (x_test, y_test) = mnist.load_data()
(x_train_c, y_train), (x_test, y_test) = cifar10.load_data()
(x_train_c2, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
(x_train_i, y_train), (x_test, y_test) = fashion_mnist.load_data()


# Limit data and Scale data
x_train_m = np.asarray(x_train_m[0:1000]) / 255
x_train_c = np.asarray(x_train_c[0:1000])
x_train_c2 = np.asarray(x_train_c2[0:1000])
x_train_i = np.asarray(x_train_i[0:1000]) / 255


# Gray scale conversion (cifar only)
x_train_c = x_train_c.dot([0.2989, 0.5870, 0.1140])
x_train_c = x_train_c / 255

x_train_c2 = x_train_c2.dot([0.2989, 0.5870, 0.1140])
x_train_c2 = x_train_c2 / 255


# Pad MNIST with 0.0 values (because borders are black)
x_train_m = np.pad(x_train_m, 2, mode='constant', constant_values='0.0')
# Extra entries get thrown in
x_train_m = x_train_m[2:1002]

x_train_i = np.pad(x_train_i, 2, mode='constant', constant_values='0.0')
# Extra entries get thrown in
x_train_i = x_train_i[2:1002]


x_train_m = x_train_m.reshape((1000, 32, 32, 1))
x_train_c = x_train_c.reshape((1000, 32, 32, 1))
x_train_c2 = x_train_c2.reshape((1000, 32, 32, 1))
x_train_i = x_train_i.reshape((1000, 32, 32, 1))


# DONE WITH LOADING DATA
# DO TESTING


m =  Test(1024, x_train_m)
c =  Test(1024, x_train_c)
c2 = Test(1024, x_train_c2)
i =  Test(1024, x_train_i)

print(x_train_m.shape)
print(x_train_c.shape)
print(x_train_c2.shape)
print(x_train_i.shape)

# Could be put in loop but lazy 

hel = AE(m,c)
res = hel.run(1)
x1, x2, x3 = res
fancy_logger(x1, x2, x3,'m-c', file_name='data-v7-set')

hel = AE(m,c2)
res = hel.run(1)
x1, x2, x3 = res
fancy_logger(x1, x2, x3,'m-c2', file_name='data-v7-set')

hel = AE(m,i)
res = hel.run(1)
x1, x2, x3 = res
fancy_logger(x1, x2, x3,'m-i', file_name='data-v7-set')

hel = AE(c,c2)
res = hel.run(1)
x1, x2, x3 = res
fancy_logger(x1, x2, x3,'c-c2', file_name='data-v7-set')

hel = AE(c,i)
res = hel.run(1)
x1, x2, x3 = res
fancy_logger(x1, x2, x3,'c-i', file_name='data-v7-set')

hel = AE(c2,i)
res = hel.run(1)
x1, x2, x3 = res
fancy_logger(x1, x2, x3,'c2-i', file_name='data-v7-set')

print(res)
