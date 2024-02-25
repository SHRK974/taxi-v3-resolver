import random

import numpy as np
from numpy import float64

from Q_Learning.Data.Hyperparameter import Hyperparameter
from Taxi.Data.StepResult import StepResult
from Taxi.Enums.GameActionEnum import action_from_int
from Taxi.Models.GameManager import GameManager


class QLearningTrainer:
    def __init__(self, manager: GameManager, hyperparameter: Hyperparameter) -> None:
        self.manager = manager
        self.hyperparameter = hyperparameter
        self.q_table = np.zeros([self.manager.env.observation_space.n, self.manager.env.action_space.n])

    def train(self) -> None:
        for i in range(1, self.hyperparameter.episodes_training + 1):
            state, _ = self.manager.reset()
            
            done: bool = False
            while not done:
                if random.uniform(0, 1) < self.hyperparameter.epsilon:
                    action: int = self.manager.env.action_space.sample()  # Explore action space
                else:
                    action: int = np.argmax(self.q_table[state])  # Exploit learned values
                result: StepResult = self.manager.step(action_from_int(action))

                old_value = self.q_table[state, action]
                next_max: float64 = np.max(self.q_table[result.state])

                new_value = (1 - self.hyperparameter.alpha) * old_value + self.hyperparameter.alpha * (result.reward + self.hyperparameter.gamma * next_max)
                self.q_table[state, action] = new_value

                state = result.state
                done: bool = result.terminated

            if i % 1000 == 0:
                print(f"Episode: {i} of {self.hyperparameter.episodes_training} episodes")

        np.save("Q_Learning/q_table.npy", self.q_table)
        print("Training finished.\n")
