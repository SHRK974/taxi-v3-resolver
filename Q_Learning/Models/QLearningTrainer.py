import random

import numpy as np
from numpy import float64
from tqdm import tqdm

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
        for i in tqdm(range(1, self.hyperparameter.episodes_training + 1)):
            state, _ = self.manager.reset()
            
            self._update_epsilon(current_episode=i)
            
            done: bool = False
            while not done:
                if random.uniform(0, 1) < self.hyperparameter.epsilon:
                    action: int = self.manager.env.action_space.sample()  # Explore action space
                else:
                    action: int = np.argmax(self.q_table[state])  # Exploit learned values
                result: StepResult = self.manager.step(action_from_int(action))
                
                self.q_table[state, action] = self._bellman_equation(
                    state=state, 
                    action=action, 
                    reward=result.reward, 
                    next_state=result.state
                )

                state = result.state
                done: bool = result.terminated

        np.save("Q_Learning/q_table.npy", self.q_table)
        print("Training finished.\n")
        
    def _bellman_equation(self, state: int, action: int, reward: float64, next_state: int) -> float64:
        old_value = self.q_table[state, action]
        next_value = np.max(self.q_table[next_state])
        return (1 - self.hyperparameter.alpha) * old_value + self.hyperparameter.alpha * (reward + self.hyperparameter.gamma * next_value)
        
    def _update_epsilon(self, current_episode: int) -> None:
        self.hyperparameter.epsilon = self.hyperparameter.min_epsilon + (self.hyperparameter.epsilon - self.hyperparameter.min_epsilon) * np.exp(-self.hyperparameter.epsilon_decay_rate * current_episode)
