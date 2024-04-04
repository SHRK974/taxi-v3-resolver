import numpy as np

from Taxi.Data.EpisodeResult import EpisodeResult
from Taxi.Data.StepResult import StepResult
from Taxi.Enums.GameActionEnum import GameActionEnum
from Taxi.Models.GameManager import GameManager, calculate_max_steps


class Sarsa:
    def __init__(self, manager: GameManager, q_table_name: str) -> None:
        self.manager = manager
        self.q_table = np.load(f"Sarsa/{q_table_name}.npy")
        self.max_steps = calculate_max_steps(
            grid_size=self.manager.env.observation_space.n,
            pickups=1,
            dropoffs=1
        )

    def solve(self) -> EpisodeResult:
        state, _ = self.manager.reset()
        
        done: bool = False
        while not done:
            action: int = np.argmax(self.q_table[state])
            result: StepResult = self.manager.step(GameActionEnum(action))
            state = result.state
            done: bool = result.terminated
            if self.manager.epochs >= self.max_steps:
                break
        
        self.manager.render()
        return EpisodeResult(
            solved=done,
            epochs=self.manager.epochs,
            rewards=self.manager.rewards,
            penalties=self.manager.penalties
        )
