import argparse
import sys
from os.path import abspath, dirname, join

import gymnasium as gym

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import numpy as np
import random
from tqdm import tqdm

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam

enviroment = gym.make("Taxi-v3", render_mode="ansi")
print('Number of states: {}'.format(enviroment.observation_space.n))
print('Number of actions: {}'.format(enviroment.action_space.n))

class Agent:
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self._state_size = enviroment.observation_space.n
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer
        
        self.experience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_shape=(1,)))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return enviroment.action_space.sample()
        
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)
    
    def save(self, name):
        self.q_network.save_weights(name)
        
    def load(self, name):
        self.q_network.load_weights(name)
            
optimizer = Adam(learning_rate=0.01)
agent = Agent(enviroment, optimizer)

batch_size = 32
num_of_episodes = 100
timesteps_per_episode = 1000
agent.q_network.summary()

for e in tqdm(range(0, num_of_episodes)):
    # Reset the enviroment
    state = enviroment.reset()
    state = np.reshape(state[0], [1, 1])
    
    # Initialize variables
    reward = 0
    terminated = False
    
    while not terminated:
        # Run Action
        action = agent.act(state)
        
        # Take action    
        next_state, reward, terminated, truncated, info = enviroment.step(action)
        next_state = np.reshape(next_state, [1, 1])
        
        print(enviroment.render())
        
        if np.random.rand() < 0.1:
            agent.store(state, action, reward, next_state, terminated)
        
        state = next_state
        
        if terminated:
            print("Episode: {}/{}, score: {}".format(e, num_of_episodes, reward))
            agent.alighn_target_model()
            break
        
        if len(agent.experience_replay) > batch_size:
            print("Retraining")
            agent.retrain(batch_size)
    
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        print(enviroment.render())
        print("**********************************")
        
agent.save("taxi-dqn.weights.h5")

# solve the enviroment with the trained agent
state = enviroment.reset()
state = np.reshape(state[0], [1, 1])
enviroment.render()
terminated = False
agent.load("taxi-dqn.weights.h5")
agent.epsilon = 0
while not terminated:
    action = agent.act(state)
    next_state, reward, terminated, truncated, info = enviroment.step(action)
    next_state = np.reshape(next_state, [1, 1])
    state = next_state
    enviroment.render()
enviroment.close()