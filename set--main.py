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
from PIL import Image

from keras import losses

import copy 
import gym
import random
#seed(3)
#set_random_seed(4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
    
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


# Incoming data is selected from first 25k, joined, shuffled, returns
# Data should be same shapes
def comb(a_test, a_labels, b_test, b_labels):
    a_t_temp = copy.deepcopy(a_test)[0:25000]
    b_t_temp = copy.deepcopy(b_test)[0:25000]

    # B is always CartPole!! or it die
    a_l_temp = copy.deepcopy(a_labels)
    b_l_temp = copy.deepcopy(b_labels)[0:25000]
    b_l_temp = np.pad(b_l_temp, (0, len(a_l_temp[0])-2), 'constant')[0:25000]
    a_l_temp[25000:50000] = b_l_temp
    #print(a_l_temp[12], a_l_temp[25600])

    return [np.concatenate((a_t_temp, b_t_temp), axis=0), a_l_temp, a_test, a_labels]

def T_A_B(m_data):
    n = m_data[1].shape[1]
    model = gen_model((32,32,1), n)
    hist = model.fit(m_data[0], m_data[1], epochs=10, batch_size=32, validation_split=0.1, verbose=1, shuffle=True)
    score = eval2(model, m_data[2], m_data[3])
    return score

# Util for formatting obs from cartpole
def util_obs(env):
    t = env.render()
    # Raw image
    im = env.viewer.get_array()
    # scale to 32x32
    im2 = im[150:350,265:335]
    img = Image.fromarray(im2)
    img2 = img.resize((32,32), Image.ANTIALIAS)
    data = np.array(img2)
    data = data.dot([0.2989, 0.5870, 0.1140])
    data = data / 255.0
    data = data.reshape(1, 32, 32, 1)
    return data

def eval2(model, x, y):
    score = 0
    for i in range(100):
    
        # Do CartPole Test    
        if random.random() > 0.5:
            env = gym.make('CartPole-v0')
            env.reset()
            obs = util_obs(env)
            done = False
            tally = 0
            while not done:
                action = model.predict(obs)
                if action[0][0] == 0:
                    action = 0
                elif action[0][0] == 1:
                    action = 1
                else:
                    action = random.randint(0,1)
                obs, reward, done, info = env.step(action)
                obs = util_obs(env)
                tally += 1
            score += tally/200.0
            env.close()
        # Do Classification
        else:
            ind = random.randint(0, 25000)
            d = x[ind]
            d = d.reshape(1, 32, 32, 1)
        
            res = model.predict(d)
            ind2 = np.argmax(res)
            print(res, y[ind])

            if y[ind][ind2] == 1.0:
                score += 1

    return score / 100


# Logging utility
# Expects a results history
def fancy_logger(score, name, file_name='data', write='a'):
    with open(file_name + '.csv', write, newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow((score, name))


(x_train_m, y_train_m), (x_test_m, y_test_m) = mnist.load_data()
(x_train_c, y_train_c), (x_test_c, y_test_c) = cifar10.load_data()
(x_train_c2, y_train_c2), (x_test_c2, y_test_c2) = cifar100.load_data(label_mode='fine')
(x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()


## LOAD CARTPOLE DATA HERE
import cv2

x_train_cp_l = []
for i in range(50000):
    im = cv2.imread("cart_data/" + str(i) + ".png")
    x_train_cp_l.append(im)

x_train_cp = np.asarray(x_train_cp_l)

actions = []
for i in range(5):
    x_test_cp = np.load('actions_' + str(i) + '.npy')
    x_test_cp = x_test_cp.reshape(10000,1)
    actions.append(x_test_cp.tolist())

y_train_cp = np.asarray(actions)
y_train_cp = y_train_cp.reshape(50000, )

# Format CP data
x_train_cp = x_train_cp.dot([0.2989, 0.5870, 0.1140])
x_train_cp = x_train_cp / 255

## DONE LOADING CARTPOLE


## Limit data to 50k and Scale data
x_train_m = np.asarray(x_train_m[0:50000]) / 255
x_train_c = np.asarray(x_train_c)
x_train_c2 = np.asarray(x_train_c2)
x_train_f = np.asarray(x_train_f[0:50000]) / 255

# Do same for labels and convert to one-hot
y_train_m = to_categorical(y_train_m[0:50000] + 2)
y_train_c = to_categorical(y_train_c[0:50000] + 2)
y_train_c2 = to_categorical(y_train_c2[0:50000] + 2)
y_train_f = to_categorical(y_train_f[0:50000] + 2)
y_train_cp = to_categorical(y_train_cp[0:50000])

# Gray scale conversion (cifars only)
x_train_c = x_train_c.dot([0.2989, 0.5870, 0.1140])
x_train_c = x_train_c / 255

x_train_c2 = x_train_c2.dot([0.2989, 0.5870, 0.1140])
x_train_c2 = x_train_c2 / 255


# Pad MNIST with 0.0 values (because borders are black)
x_train_m = np.pad(x_train_m, 2, mode='constant', constant_values='0.0')
# Extra entries get thrown in
x_train_m = x_train_m[2:50002]

x_train_f = np.pad(x_train_f, 2, mode='constant', constant_values='0.0')
# Extra entries get thrown in
x_train_f = x_train_f[2:50002]


x_train_m = x_train_m.reshape((50000, 32, 32, 1))
x_train_c = x_train_c.reshape((50000, 32, 32, 1))
x_train_c2 = x_train_c2.reshape((50000, 32, 32, 1))
x_train_f = x_train_f.reshape((50000, 32, 32, 1))
x_train_cp = x_train_cp.reshape((50000, 32, 32, 1))


# DONE WITH LOADING DATA
# DO TESTING

print(x_train_m.shape)
print(x_train_c.shape)
print(x_train_c2.shape)
print(x_train_f.shape)
print(x_train_cp.shape)

hel = T_A_B(comb(x_train_c2, y_train_c2, x_train_cp, y_train_cp))
fancy_logger(hel, 'c2-cp', file_name='data-v8-set')


exit()

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
