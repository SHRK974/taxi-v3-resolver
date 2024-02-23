import argparse
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import gymnasium as gym

from Bruteforce.Data.EpisodeResult import EpisodeResult
from Bruteforce.Models.Bruteforce import Bruteforce
from Bruteforce.Models.GameManager import GameManager
from Bruteforce.Models.TopLeftSequence import TopLeftSequence
from Bruteforce.Models.TopRightSequence import TopRightSequence

ENV_GAME = "Taxi-v3"

def bruteforce(amount: int) -> None:
    """
    Bruteforce the Taxi-v3 environment a certain amount of times and print the results

    Args:
        amount (int): The amount of times to bruteforce the environment
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

def process_bruteforce_results(results: list[EpisodeResult]) -> None:
    """
    Process the bruteforce results and print the mean, max and min reward and steps, as well as the best and worst result

    Args:
        results (list[EpisodeResult]): The bruteforced episodes results
    """
    mean_reward = sum([result.total_reward for result in results]) / len(results)
    mean_steps = sum([result.total_steps for result in results]) / len(results)
    print(f"Mean reward: {mean_reward}, Mean steps: {mean_steps}")
    highest_reward = max([result.total_reward for result in results])
    worst_reward = min([result.total_reward for result in results])
    print(f"Max reward: {highest_reward}, Min reward: {worst_reward}")
    print()
    best_result = next(result for result in results if result.total_reward == highest_reward)
    print(f"Best result: {best_result.total_reward} reward, {best_result.total_steps} steps")
    worst_result = next(result for result in results if result.total_reward == worst_reward)
    print(f"Worst result: {worst_result.total_reward} reward, {worst_result.total_steps} steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bruteforce the Taxi-v3 environment")
    parser.add_argument("--episodes", type=int, default=100, help="The amount of times to bruteforce the environment")
    args = parser.parse_args()
    bruteforce(args.episodes)