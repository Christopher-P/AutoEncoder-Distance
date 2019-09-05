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

def fancy_logger(x1, overlap, file_name='set', write='a'):
    with open(file_name + '.csv', write, newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow((x1, overlap))

def T_A_B(m_data):
    n = m_data[1].shape[1]
    model = gen_model((32,32,1), n)
    hist = model.fit(m_data[0], m_data[1], epochs=10, validation_data=(m_data[2], m_data[3]), batch_size=32, validation_split=0.1, verbose=2, shuffle=True)
    res = max(hist.history['val_acc'])
    return res
    
def comb(data, cartpole_data):
    # Prep gym
    import gym
    env = gym.make('CartPole-v0')
    m_data = [[],[],[],[]]
    
    
    # Train
    model = gen_model((32,32,1), n)
    hist = model.fit(m_data[0], m_data[1], epochs=10, validation_data=(m_data[2], m_data[3]), batch_size=32, validation_split=0.1, verbose=2, shuffle=True)
    
    # Special fit
    res = max(hist.history['val_acc'])
    return res
    

def main():

    ## LOAD CARTPOLE DATA HERE
    import cv2

    x_train_cp_l = []
    for i in range(60000):
        im = cv2.imread("../cart_data/" + str(i) + ".png")
        x_train_cp_l.append(im)

    x_train_cp = np.asarray(x_train_cp_l)

    actions = []
    for i in range(6):
        x_test_cp = np.load('../actions_' + str(i) + '.npy')
        x_test_cp = x_test_cp.reshape(10000,1)
        actions.append(x_test_cp.tolist())
    
    x_test_cp = np.asarray(actions)
    x_test_cp = x_test_cp.reshape(60000, )

    ## DONE LOADING CARTPOLE


    (x_train_m, y_train_m), (x_test_m, y_test_m) = mnist.load_data()
    (x_train_c, y_train_c), (x_test_c, y_test_c) = cifar10.load_data()
    (x_train_c2, y_train_c2), (x_test_c2, y_test_c2) = cifar100.load_data()
    (x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()

    m_data = [x_train_m, y_train_m, x_test_m, y_test_m]
    c_data = [x_train_c, y_train_c, x_test_c, y_test_c]
    f_data = [x_train_f, y_train_f, x_test_f, y_test_f]
    c2_data = [x_train_c2, y_train_c2, x_test_c2, y_test_c2]

    m_data[0] = m_data[0] / 255
    m_data[2] = m_data[2] / 255

    m_data[0] = m_data[0].reshape((60000, 28, 28, 1))
    m_data[2] = m_data[2].reshape((10000, 28, 28, 1))

    m_data[1] = to_categorical(m_data[1])
    m_data[3] = to_categorical(m_data[3])

    m_data[0] = np.pad(m_data[0], pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values='0.0')
    m_data[2] = np.pad(m_data[2], pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values='0.0')

    m_data[0] = m_data[0][0:50000]
    m_data[1] = m_data[1][0:50000]
    m_data[2] = m_data[2][0:50000]
    m_data[3] = m_data[3][0:50000]



    f_data[0] = f_data[0] / 255
    f_data[2] = f_data[2] / 255

    f_data[0] = f_data[0].reshape((60000, 28, 28, 1))
    f_data[2] = f_data[2].reshape((10000, 28, 28, 1))

    f_data[1] = to_categorical(f_data[1])
    f_data[3] = to_categorical(f_data[3])

    f_data[0] = np.pad(f_data[0], pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values='0.0')
    f_data[2] = np.pad(f_data[2], pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values='0.0')

    f_data[0] = f_data[0][0:50000]
    f_data[1] = f_data[1][0:50000]
    f_data[2] = f_data[2][0:50000]
    f_data[3] = f_data[3][0:50000]



    c_data[0] = c_data[0].dot([0.2126, 0.7152, 0.0722]) / 255
    c_data[2] = c_data[2].dot([0.2989, 0.5870, 0.1140]) / 255

    c_data[0] = c_data[0].reshape((50000, 32, 32, 1))
    c_data[2] = c_data[2].reshape((10000, 32, 32, 1))

    c_data[1] = to_categorical(c_data[1])
    c_data[3] = to_categorical(c_data[3])


    c2_data[0] = c2_data[0].dot([0.2126, 0.7152, 0.0722]) / 255
    c2_data[2] = c2_data[2].dot([0.2989, 0.5870, 0.1140]) / 255

    c2_data[0] = c2_data[0].reshape((50000, 32, 32, 1))
    c2_data[2] = c2_data[2].reshape((10000, 32, 32, 1))

    c2_data[1] = to_categorical(c2_data[1])
    c2_data[3] = to_categorical(c2_data[3])


    ## Format CP data
    x_train_cp = x_train_cp.dot([0.2126, 0.7152, 0.0722]) / 255
    cp_data[0] = x_train_cp[0:50000]
    cp_data[1] = x_test_cp[0:50000]
    cp_data[0] = x_train_cp[0:50000]
    cp_data[1] = x_test_cp[0:50000]
    
    r1 = T_A_B(copy.deepcopy(x_train_cp))
    fancy_logger(r1, 'f')

    exit()

    #r1 = T_A_B(copy.deepcopy(f_data))
    #fancy_logger(r1, 'f')
    #r1 = T_A_B(copy.deepcopy(c2_data))
    #fancy_logger(r1, 'c')



if __name__== "__main__":
    main()

    
