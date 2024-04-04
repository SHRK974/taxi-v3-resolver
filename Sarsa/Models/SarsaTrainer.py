import random
import os

import numpy as np
from numpy import float64
from tqdm import tqdm

from Sarsa.Data.Hyperparameter import Hyperparameter
from Taxi.Data.StepResult import StepResult
from Taxi.Enums.GameActionEnum import action_from_int
from Taxi.Models.GameManager import GameManager


class SarsaTrainer:
    def __init__(self, manager: GameManager, hyperparameter: Hyperparameter) -> None:
        self.manager = manager
        self.hyperparameter = hyperparameter
        self.q_table = np.zeros((self.manager.env.observation_space.n, self.manager.env.action_space.n))

    def train(self, name: str) -> None:
        print(f"Training started for {name}.\n")
        for i in tqdm(range(1, self.hyperparameter.episodes_training + 1)):
            state, _ = self.manager.reset()
            
            self._update_epsilon(current_episode=i)
            
            if random.uniform(0, 1) < self.hyperparameter.epsilon:
                action: int = self.manager.env.action_space.sample()  # Explore action space
            else:
                action: int = np.argmax(self.q_table[state,:])  # Exploit learned values
            
            done: bool = False
            while not done:
                result: StepResult = self.manager.step(action_from_int(action))
                if random.uniform(0, 1) < self.hyperparameter.epsilon:
                    next_action: int = self.manager.env.action_space.sample()
                else:
                    next_action: int = np.argmax(self.q_table[result.state,:])
                
                self.q_table[state, action] = self._bellman_equation(
                    state=state, 
                    action=action, 
                    reward=result.reward, 
                    next_state=result.state,
                    next_action=next_action
                )

                state = result.state
                action = next_action
                done: bool = result.terminated

        np.save(f"Sarsa/{name}.npy", self.q_table)
        print(f"Training finished for {name}.\n")
        
    def delete_q_table(self, name: str) -> None:
        os.remove(f"Sarsa/{name}.npy")
        print(f"Q-Table {name} deleted.")
        
    def _bellman_equation(self, state: int, action: int, reward: float64, next_state: int, next_action: int) -> float64:
        predicted_value = self.q_table[state, action]
        target_value = reward + self.hyperparameter.gamma * self.q_table[next_state, next_action]
        return predicted_value + self.hyperparameter.alpha * (target_value - predicted_value)
    
    def _update_epsilon(self, current_episode: int) -> None:
        self.hyperparameter.epsilon = self.hyperparameter.min_epsilon + (self.hyperparameter.epsilon - self.hyperparameter.min_epsilon) * np.exp(-self.hyperparameter.epsilon_decay_rate * current_episode)
