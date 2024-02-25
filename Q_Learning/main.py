import argparse
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import gymnasium as gym

from Bruteforce.Models.GameManager import GameManager
from Q_Learning.Data.Hyperparameter import Hyperparameter
from Q_Learning.Models.QLearning import QLearning
from Q_Learning.Models.QLearningTrainer import QLearningTrainer
from Taxi.Data.BatchResult import BatchResult

ENV_GAME = "Taxi-v3"

def q_learning(amount: int) -> None:
    """
    Train the Taxi-v3 environment using Q-Learning and print the results

    Args:
        episodes (int): The amount of episodes to train the environment
    """
    number_solved, number_unsolved = 0, 0
    results = []
    hyperparameter = Hyperparameter(
        alpha=0.1,
        gamma=0.6,
        epsilon=0.1,
        episodes_testing=1000,
        episodes_training=1_000
    )
    manager = GameManager(env=gym.make(ENV_GAME, render_mode="ansi"))
    trainer = QLearningTrainer(manager=manager, hyperparameter=hyperparameter)
    trainer.train()
    for i in range(amount):
        print(f"Episode: {i + 1}")
        result = QLearning(manager=manager, q_table_name="q_table").solve()
        results.append(result)
        if result.solved:
            number_solved += 1
        else:
            number_unsolved += 1
    
    batch_result = BatchResult(
        total_solved=number_solved,
        total_unsolved=number_unsolved,
        total_attempts=amount,
        success_rate=(number_solved / amount) * 100,
        results=results
    )
    batch_result.summary()
    
    return batch_result


q_learning(1000)