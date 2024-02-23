import gymnasium as gym

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from Bruteforce.Models.TopLeftSequence import TopLeftSequence
from Bruteforce.Models.TopRightSequence import TopRightSequence
from Bruteforce.Models.GameManager import GameManager
from Bruteforce.Models.Bruteforce import Bruteforce

ENV_GAME = "Taxi-v3"

def bruteforce(amount: int) -> None:
    number_solved, number_unsolved = 0, 0
    for i in range(amount):
        print(f"Bruteforce attempt {i + 1}")
        manager = GameManager(env=gym.make(ENV_GAME, render_mode="ansi"))
        bruteforce = Bruteforce(manager=manager, top_right_sequence=TopRightSequence(), top_left_sequence=TopLeftSequence())
        solved = bruteforce.solve()
        if solved:
            number_solved += 1
        else:
            manager.render()
            number_unsolved += 1
    print(f"Solved: {number_solved}, Unsolved: {number_unsolved}, Total: {amount} ({(number_solved / amount) * 100}% success rate)")

bruteforce(amount=10000)