import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from PIL import Image

import time

ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123456789)
env.seed(123456789)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))


import threading 

def thread_gen(n):
    env = gym.make('CartPole-v0')
    actions = []
    for i in range(50):
        obs = env.reset()
        for j in range(200):
            
            t = env.render()
            act = dqn.forward(obs)

            # Raw image
            im = env.viewer.get_array()
            # scale to 32x32
            im2 = im[150:350,265:335]
            img = Image.fromarray(im2)
            img2 = img.resize((32,32), Image.ANTIALIAS)
            print(np.array(img2))
            exit()
            img2.save('cart_data/' + str(j + (i * 200) + (n * 10000)) + '.png')

            # Save action
            actions.append(act)
            
            # Preform action
            obs = env.step(act)[0]

    x = np.asarray(actions)
    np.save('actions_' + str(n), x)

thread_gen(5)


print("Done!") 




#dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=True)
