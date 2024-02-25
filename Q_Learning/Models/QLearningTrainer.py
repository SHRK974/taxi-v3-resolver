import random

import numpy as np

from Bruteforce.Models.GameManager import GameManager
from Q_Learning.Data.Hyperparameter import Hyperparameter


class QLearningTrainer:
    def __init__(self, manager: GameManager, hyperparameter: Hyperparameter) -> None:
        self.manager = manager
        self.hyperparameter = hyperparameter
        self.q_table = np.zeros([self.manager.env.observation_space.n, self.manager.env.action_space.n])

    def train(self) -> None:
        for i in range(1, self.hyperparameter.episodes_training + 1):
            state, _ = self.manager.reset()

            done = False
            while not done:
                if random.uniform(0, 1) < self.hyperparameter.epsilon:
                    action = self.manager.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q_table[state]) # Exploit learned values
                result = self.manager.step(action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[result.state])
                
                new_value = (1 - self.hyperparameter.alpha) * old_value + self.hyperparameter.alpha * (result.reward + self.hyperparameter.gamma * next_max)
                self.q_table[state, action] = new_value
                
                state =  result.state
                done = result.terminated
            
            if i % 1000 == 0:
                print(f"Episode: {i} of {self.hyperparameter.episodes_training} episodes")
        
        np.save("Q_Learning/q_table.npy", self.q_table)
        print("Training finished.\n")