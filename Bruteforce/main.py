import argparse
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import gymnasium as gym
import pickle
import random
import os
from time import sleep
from tqdm import tqdm

from Bruteforce.Models.Bruteforce import Bruteforce
from Bruteforce.Models.TopLeftSequence import TopLeftSequence
from Bruteforce.Models.TopRightSequence import TopRightSequence
from Taxi.Data.BatchResult import BatchResult
from Taxi.Data.EpisodeResult import EpisodeResult
from Taxi.Models.GameManager import GameManager

ENV_GAME = "Taxi-v3"


def bruteforce(amount: int) -> BatchResult:
    """
    Bruteforce the Taxi-v3 environment a certain amount of times and print the results

    Args:
        amount (int): The amount of times to bruteforce the environment
        
    Returns:
        BatchResult: The bruteforce batch results.
    """
    number_solved, number_unsolved = 0, 0
    bruteforce_results = []
    animations: list[str] = []
    
    for _ in tqdm(range(amount)):
        manager = GameManager(env=gym.make(ENV_GAME, render_mode="ansi"), track_playback=True)
        result: EpisodeResult = Bruteforce(
            manager=manager,
            top_right_sequence=TopRightSequence(),
            top_left_sequence=TopLeftSequence()
        ).solve()
        bruteforce_results.append(result)
        
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
        total_attempts=amount,
        success_rate=(number_solved / amount) * 100,
        results=bruteforce_results
    )

    with open("./Bruteforce/batch_result.pkl", "wb") as file:
        pickle.dump(batch_result, file)

    batch_result.summary()

    return batch_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bruteforce the Taxi-v3 environment")
    parser.add_argument("--episodes", type=int, default=10000, help="The amount of times to bruteforce the environment")
    args = parser.parse_args()
    bruteforce(args.episodes)
