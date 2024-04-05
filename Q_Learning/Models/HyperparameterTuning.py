import sys
from os.path import abspath, dirname, join

import gymnasium as gym
import numpy as np
import optuna

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from Q_Learning.Data.Hyperparameter import Hyperparameter
from Q_Learning.Models.QLearning import QLearning
from Q_Learning.Models.QLearningTrainer import QLearningTrainer
from Taxi.Data.EpisodeResult import EpisodeResult
from Taxi.Data.TuningQuantifier import TuningQuantifier
from Taxi.Models.GameManager import GameManager

ENV_GAME = "Taxi-v3"


def objective(trial: optuna.Trial) -> float:
    """
    Objective function to optimize hyperparameters for the Taxi-v3 environment using Q-Learning

    Args:
        trial (optuna.Trial): A single execution of the objective function

    Returns:
        float: The objective value of the trial that is used by Optuna to determine the best hyperparameters
    """
    hyperparameter: Hyperparameter = Hyperparameter(
        alpha=trial.suggest_float("alpha", 0.1, 0.9),
        gamma=trial.suggest_float("gamma", 0.1, 0.9),
        epsilon=trial.suggest_float("epsilon", 0.1, 0.9),
        min_epsilon=trial.suggest_float("min_epsilon", 0.01, 0.1),
        epsilon_decay_rate=trial.suggest_float("epsilon_decay_rate", 0.01, 0.9),
        episodes_training=trial.suggest_int("episodes_training", 1000, 30000),
        episodes_testing=10000,
    )

    number_solved, number_unsolved = 0, 0
    results: list[EpisodeResult] = []

    manager: GameManager = GameManager(env=gym.make(ENV_GAME, render_mode="ansi"))
    trainer: QLearningTrainer = QLearningTrainer(manager=manager, hyperparameter=hyperparameter)
    trainer.train(name=trial.number)

    print(f"Training validation started for {trial.number}.\n")
    for _ in range(hyperparameter.episodes_testing):
        result: EpisodeResult = QLearning(manager=manager, q_table_name=trial.number).solve()
        results.append(result)
        if result.solved:
            number_solved += 1
        else:
            number_unsolved += 1
    print(f"Training validation finished for {trial.number}.\n")
    
    trainer.delete_q_table(name=trial.number)
    
    rewards = [result.rewards for result in results]
    penalties = [result.penalties for result in results]
    epochs = [result.epochs for result in results]
    
    quantifier: TuningQuantifier = TuningQuantifier(
        success_rate=(number_solved / hyperparameter.episodes_testing) * 100,
        mean_rewards_per_steps=np.mean(rewards) / np.mean(epochs),
        mean_penalties_per_episode=np.mean(penalties),
        mean_steps_per_episode=np.mean(epochs),
        best_episode=next(result for result in results if result.rewards == np.max(rewards)),
        worst_episode=next(result for result in results if result.rewards == np.min(rewards))
    )
    
    return quantifier.calculate_score()
