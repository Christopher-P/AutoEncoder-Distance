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

from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.utils import to_categorical

from numpy.random import seed
from tensorflow import set_random_seed

from numba import cuda
import copy 

#seed(3)
#set_random_seed(4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

# https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

def gen_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape ))
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
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    #exit()
    return model

def fancy_logger(x1, overlap, file_name='data', write='a'):
    with open(file_name + '.csv', write, newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow((x1, overlap))

def T_A_B(m_data, c_data, i):
    m_data[0] = np.pad(m_data[0], pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values='0.0')
    m_data[2] = np.pad(m_data[2], pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values='0.0')

    m_data[0] = m_data[0][0:50000]
    m_data[1] = m_data[1][0:50000]
    m_data[2] = m_data[2][0:50000]
    m_data[3] = m_data[3][0:50000]

    n_m = 50000
    n_c = 50000

    frac_m = int(i * n_m)
    frac_c = int((1-i) * n_c)

    m_data[0] = m_data[0][0:frac_m]
    m_data[1] = m_data[1][0:frac_m]
    m_data[2] = m_data[2][0:frac_m]
    m_data[3] = m_data[3][0:frac_m]

    c_data[0] = c_data[0][0:frac_c]
    c_data[1] = c_data[1][0:frac_c]
    c_data[2] = c_data[2][0:frac_c]
    c_data[3] = c_data[3][0:frac_c]
    print(frac_m, frac_c)

    c_data[1] = np.pad(c_data[1], pad_width=((0,0),(0,10)), mode='constant', constant_values='0.0')
    c_data[3] = np.pad(c_data[3], pad_width=((0,0),(0,10)), mode='constant', constant_values='0.0')

    m_data[1] = np.pad(m_data[1], pad_width=((0,0),(10,0)), mode='constant', constant_values='0.0')
    m_data[3] = np.pad(m_data[3], pad_width=((0,0),(10,0)), mode='constant', constant_values='0.0')

    c_data[0] = np.concatenate((c_data[0], m_data[0]),axis=0)
    c_data[1] = np.concatenate((c_data[1], m_data[1]),axis=0)
    c_data[2] = np.concatenate((c_data[2], m_data[2]),axis=0)
    c_data[3] = np.concatenate((c_data[3], m_data[3]),axis=0)

    model = gen_model((32,32,1), 20)
    hist = model.fit(c_data[0], c_data[1], epochs=10, validation_data=(c_data[2], c_data[3]), batch_size=32, validation_split=0.1, verbose=2, shuffle=True)
    res = max(hist.history['val_acc'])
    return res
    

def main():


    (x_train_m, y_train_m), (x_test_m, y_test_m) = mnist.load_data()
    (x_train_c, y_train_c), (x_test_c, y_test_c) = cifar10.load_data()

    m_data = [x_train_m, y_train_m, x_test_m, y_test_m]
    c_data = [x_train_c, y_train_c, x_test_c, y_test_c]

    m_data[0] = m_data[0] / 255
    m_data[2] = m_data[2] / 255

    m_data[0] = m_data[0].reshape((60000, 28, 28, 1))
    m_data[2] = m_data[2].reshape((10000, 28, 28, 1))

    m_data[1] = to_categorical(m_data[1])
    m_data[3] = to_categorical(m_data[3])

    c_data[0] = c_data[0].dot([0.2126, 0.7152, 0.0722]) / 255
    c_data[2] = c_data[2].dot([0.2989, 0.5870, 0.1140]) / 255

    c_data[0] = c_data[0].reshape((50000, 32, 32, 1))
    c_data[2] = c_data[2].reshape((10000, 32, 32, 1))

    c_data[1] = to_categorical(c_data[1])
    c_data[3] = to_categorical(c_data[3])

    for i in range(7,11):

        r1 = T_A_B(copy.deepcopy(m_data), copy.deepcopy(c_data), i/10)

        fancy_logger(r1, i / 10)



if __name__== "__main__":
    main()

