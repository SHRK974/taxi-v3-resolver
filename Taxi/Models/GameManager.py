from typing import Any, SupportsFloat, Tuple

import gymnasium as gym

from Bruteforce.Data.SequenceResult import SequenceResult
from Taxi.Data.StepResult import StepResult, step_result_from_tuple
from Taxi.Enums.GameActionEnum import GameActionEnum


class GameManager:
    def __init__(self, env: gym.Env, render_output: bool = False) -> None:
        self.env = env
        self.passenger_found = False
        self.epochs = 0
        self.rewards = 0
        self.penalties = 0
        self.render_output = render_output

    def render(self) -> None:
        """
        Renders the environment.
        """
        if self.render_output:
            print(self.env.render())

    def reset(self) -> Tuple[Any, dict]:
        """
        Resets the environment.

        Returns:
            Tuple[Any, dict]: Tuple containing the state and the environment's info.
        """
        self.passenger_found = False
        self.epochs = 0
        self.rewards = 0
        self.penalties = 0
        return self.env.reset()

    def step(self, action: GameActionEnum) -> StepResult:
        """
        Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (GameActionEnum): The action to perform.

        Returns:
            StepResult: The result of the step.
        """
        result: StepResult = step_result_from_tuple(step=self.env.step(action))
        self.__update_metrics(result)
        return result

    def move_until_stopped(self, state: Any, action: GameActionEnum) -> Any:
        """
        Moves the agent until it is stopped.

        Args:
            state (Any): The current state of the environment.
            action (GameActionEnum): The action to perform.
            
        Returns:
            Any: The final state.
        """
        done: bool = False

        while not done:
            result: StepResult = step_result_from_tuple(step=self.env.step(action))
            self.__update_metrics(result)
            if result.state == state:
                done = True
            else:
                state = result.state
        return state

    def loc_try_dropoff(self, state: Any) -> SequenceResult:
        """
        Tries to drop off the passenger.

        Args:
            state (Any): The current state of the environment.

        Returns:
            SequenceResult: The result of the sequence.
        """
        env: gym.Env = self.env
        if self.passenger_found:
            result: StepResult = step_result_from_tuple(step=env.step(GameActionEnum.DROPOFF))
            self.__update_metrics(result)
            if result.terminated:
                return SequenceResult(
                    terminated=True,
                    passenger_found=self.passenger_found,
                    state=result.state
                )
        return SequenceResult(
            terminated=False,
            passenger_found=self.is_passenger_picked_up(
                self.pick_up_passenger()
            ),
            state=state
        )

    def is_passenger_picked_up(self, reward: SupportsFloat) -> bool:
        """
        Determines whether the passenger is picked up.

        Args:
            reward (SupportsFloat): The reward from the environment.

        Returns:
            picked_up (bool): Whether the passenger is picked up.
        """
        picked_up: bool = reward == -1
        if picked_up:
            self.passenger_found = True
        return picked_up

    def pick_up_passenger(self) -> SupportsFloat:
        """
        Picks up the passenger.
        
        Returns:
            SupportsFloat: The reward from the environment.
        """
        result: StepResult = step_result_from_tuple(step=self.env.step(GameActionEnum.PICKUP))
        self.__update_metrics(result)
        return result.reward

    def __update_metrics(self, result: StepResult) -> None:
        """
        Update key metrics for the game.

        Args:
            result (StepResult): The result of the step.
        """
        self.epochs += 1
        self.rewards += result.reward
        if result.reward == -10:
            self.penalties += 1


def calculate_max_steps(grid_size: int, pickups: int, dropoffs: int) -> int:
    """
    Heuristically calculates the maximum number of steps to solve the environment.

    We consider the worst case scenario where:
    - The agent has to go through the entire grid to pick up the passenger. And come back to the initial position.
    - The agent has to go through the entire grid to drop off the passenger. And come back to the initial position.

    Args:
        grid_size (int): Grid size.
        pickups (int): Number of passenger to pick up.
        dropoffs (int): Number of passenger to drop off.

    Returns:
        int: Calculated upper bound of steps.
    """
    max_steps_per_pickup = (grid_size - 1) * 2
    max_steps_per_dropoff = (grid_size - 1) * 2
    return (max_steps_per_pickup * pickups) + (max_steps_per_dropoff * dropoffs)
