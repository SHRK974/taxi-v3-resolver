import argparse
import sys
from os.path import abspath, dirname, join

import gymnasium as gym
import os
import pickle
import random
from time import sleep
from tqdm import tqdm

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from Sarsa.Data.Hyperparameter import Hyperparameter
from Sarsa.Models.Sarsa import Sarsa
from Sarsa.Models.SarsaTrainer import SarsaTrainer
from Taxi.Data.BatchResult import BatchResult
from Taxi.Data.EpisodeResult import EpisodeResult
from Taxi.Models.GameManager import GameManager

ENV_GAME = "Taxi-v3"


def sarsa(hyperparameter: Hyperparameter) -> BatchResult:
    """
    Train the Taxi-v3 environment using SARSA and print the results

    Args:
        hyperparameter (Hyperparameter): Hyperparameters to use for training and testing the environment

    Returns:
        BatchResult: The results of the training
    """
    number_solved, number_unsolved = 0, 0
    results: list[EpisodeResult] = []
    animations: list[str] = []
    
    manager: GameManager = GameManager(env=gym.make(ENV_GAME, render_mode="ansi"))
    trainer: SarsaTrainer = SarsaTrainer(manager=manager, hyperparameter=hyperparameter)
    trainer.train(name="q_table")
    
    print(f"Testing started on {hyperparameter.episodes_testing} episodes.", end=f"\n")
    for _ in tqdm(range(hyperparameter.episodes_testing)):
        manager: GameManager = GameManager(env=gym.make(ENV_GAME, render_mode="ansi"), track_playback=True)
        result: EpisodeResult = Sarsa(manager=manager, q_table_name="q_table").solve()
        results.append(result)
        
        playback = manager.get_playback()
        if len(playback) > 0:
            animations.append(playback)
        
        if result.solved:
            number_solved += 1
        else:
            number_unsolved += 1
    
    playback = random.choice(animations)
    for frame in playback:
        os.system('cls')
        print(frame)
        sleep(0.5)
    
    batch_result = BatchResult(
        total_solved=number_solved,
        total_unsolved=number_unsolved,
        total_attempts=hyperparameter.episodes_testing,
        success_rate=(number_solved / hyperparameter.episodes_testing) * 100,
        results=results
    )
    print("SARSA", end="\n\n")
    batch_result.summary()

    with open("./Sarsa/batch_result.pkl", "wb") as file:
        pickle.dump(batch_result, file)
    
    return batch_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Taxi-v3 environment using Q-Learning")
    parser.add_argument(
        "-a", "--alpha", dest="alpha", type=float, default=0.16711067947621416, help="The learning rate"
    )
    parser.add_argument(
        "-g", "--gamma", dest="gamma", type=float, default=0.8995533760334706, help="The discount factor"
    )
    parser.add_argument(
        "-e", "--epsilon", dest="epsilon", type=float, default=0.42597524846219625, help="The exploration rate"
    )
    parser.add_argument(
        "--min_epsilon", dest="min_epsilon", type=float, default=0.010230558452100824, help="The minimum exploration rate"
    )
    parser.add_argument(
        "--epsilon_decay_rate", dest="epsilon_decay_rate", type=float, default=0.28218249628800446, help="The rate at which the exploration rate decays"
    )
    parser.add_argument(
        "--training", dest="training", type=int, default=9210, help="The number of episodes to train the environment"
    )
    parser.add_argument(
        "--testing", dest="testing", type=int, default=10000, help="The number of episodes to test the environment"
    )
    args = parser.parse_args()
    
    try:
        hyperparameter = Hyperparameter(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            min_epsilon=args.min_epsilon,
            epsilon_decay_rate=args.epsilon_decay_rate,
            episodes_training=args.training,
            episodes_testing=args.testing
        )

    except ValueError as e:
        _, value, _ = sys.exc_info()
        print(value)
        exit()
    
    sarsa(hyperparameter=hyperparameter)
