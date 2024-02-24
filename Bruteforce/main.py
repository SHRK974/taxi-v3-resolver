import argparse
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import gymnasium as gym
import numpy as np

from Bruteforce.Data.BruteforceResult import BruteforceResult
from Bruteforce.Data.EpisodeResult import EpisodeResult
from Bruteforce.Models.Bruteforce import Bruteforce
from Bruteforce.Models.GameManager import GameManager
from Bruteforce.Models.TopLeftSequence import TopLeftSequence
from Bruteforce.Models.TopRightSequence import TopRightSequence

ENV_GAME = "Taxi-v3"

def bruteforce(amount: int) -> BruteforceResult:
    """
    Bruteforce the Taxi-v3 environment a certain amount of times and print the results

    Args:
        amount (int): The amount of times to bruteforce the environment
        
    Returns:
        BruteforceResult: The bruteforce batch results.
    """
    number_solved, number_unsolved = 0, 0
    bruteforce_results = []
    for i in range(amount):
        print(f"Bruteforce attempt {i + 1}")
        manager = GameManager(env=gym.make(ENV_GAME, render_mode="ansi"))
        bruteforce = Bruteforce(manager=manager, top_right_sequence=TopRightSequence(), top_left_sequence=TopLeftSequence())
        result = bruteforce.solve()
        bruteforce_results.append(result)
        if result.solved:
            number_solved += 1
        else:
            number_unsolved += 1
    
    print(f"Solved: {number_solved}, Unsolved: {number_unsolved}, Total: {amount} ({(number_solved / amount) * 100}% success rate)")
    print()
    process_bruteforce_results(bruteforce_results)
    
    return BruteforceResult(
        total_solved=number_solved,
        total_unsolved=number_unsolved,
        total_attempts=amount,
        success_rate=(number_solved / amount) * 100,
        results=bruteforce_results
    )

def process_bruteforce_results(results: list[EpisodeResult]) -> None:
    """
    Process the bruteforce results and print the mean, max and min reward and steps, as well as the best and worst result

    Args:
        results (list[EpisodeResult]): The bruteforced episodes results
    """
    rewards = [result.rewards for result in results]
    epochs = [result.epochs for result in results]
    
    print(f"Mean rewards per episode: {np.mean(rewards)}, Mean steps per episode: {np.mean(epochs)}")
    print(f"Median rewards per episode: {np.median(rewards)}, Median steps per episode: {np.median(epochs)}\n")
    
    highest_reward = np.max(rewards) 
    worst_reward = np.min(rewards)
    print(f"Maximum rewards achieved in a single episode: {highest_reward}")
    print(f"Minimum rewards achieved in a single episode: {worst_reward}\n")
    
    best_result = next(result for result in results if result.rewards == highest_reward)
    print(f"Best episode: {best_result.rewards} rewards, {best_result.epochs} steps")
    worst_result = next(result for result in results if result.rewards == worst_reward)
    print(f"Worst episode: {worst_result.rewards} rewards, {worst_result.epochs} steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bruteforce the Taxi-v3 environment")
    parser.add_argument("--episodes", type=int, default=100, help="The amount of times to bruteforce the environment")
    args = parser.parse_args()
    bruteforce(args.episodes)