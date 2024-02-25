import numpy as np

from Taxi.Data.EpisodeResult import EpisodeResult
from Bruteforce.Models.GameManager import GameManager
from Taxi.Enums.GameActionEnum import GameActionEnum


class QLearning:
    def __init__(self, manager: GameManager, q_table_name: str) -> None:
        self.manager = manager
        self.q_table = np.load(f"Q_Learning/{q_table_name}.npy")
        self.max_steps = self.manager.calculate_max_steps(
            grid_size=self.manager.env.observation_space.n,
            pickups=1,
            dropoffs=1
        )

    def solve(self) -> EpisodeResult:
        state, _ = self.manager.reset()
        
        done = False
        while not done:
            action = np.argmax(self.q_table[state])
            result = self.manager.step(GameActionEnum(action))
            state = result.state
            done = result.terminated
            if self.manager.epochs >= self.max_steps:
                break
        
        self.manager.render()
        return EpisodeResult(
            solved=done,
            epochs=self.manager.epochs,
            rewards=self.manager.rewards,
            penalties=self.manager.penalities
        )