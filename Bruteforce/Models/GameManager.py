import gym

from typing import Any, SupportsFloat
from Bruteforce.Enums.GameActionEnum import GameActionEnum
from Bruteforce.Data.SequenceResult import SequenceResult
from Bruteforce.Data.StepResult import StepResult

class GameManager:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.passenger_found = False
        self.total_reward = 0
        self.total_steps = 0
        
    def render(self) -> None:
        """
        Renders the environment.
        """
        print(self.env.render())
        
    def reset(self) -> Any:
        """
        Resets the environment.

        Returns:
            Any: The initial state of the environment.
        """
        return self.env.reset()
    
    def step(self, action: GameActionEnum) -> StepResult:
        """
        Steps through the environment.

        Args:
            action (GameActionEnum): The action to perform.

        Returns:
            StepResult: The result of the step.
        """
        result = self.tuple_to_step_result(self.env.step(action))
        self.total_reward += result.reward
        self.total_steps += 1
        return result
    
    def move_until_stopped(self, start_state: Any, action: GameActionEnum) -> Any:
        """
        Moves the agent until it is stopped.

        Args:
            action (GameActionEnum): The action to perform.
            
        Returns:
            Any: The final state.
        """
        done = False
        old_state = start_state
        
        while not done:
            result = self.tuple_to_step_result(step=self.env.step(action))
            self.total_reward += result.reward
            self.total_steps += 1
            if result.state == old_state:
                done = True
            else:
                old_state = result.state
        return old_state
    
    def loc_try_dropoff(self, state: Any) -> SequenceResult:
        """
        Tries to drop off the passenger.

        Args:
            state (Any): The current state of the environment.

        Returns:
            terminated (bool): Whether the environment is terminated.
        """
        env = self.env
        if self.passenger_found:
            result = self.tuple_to_step_result(env.step(GameActionEnum.DROPOFF))
            self.total_reward += result.reward
            self.total_steps += 1
            if result.terminated:
                print("Problem solved.")
                return SequenceResult(terminated=True, 
                    passenger_found=self.passenger_found,
                    state=result.state
                )
        return SequenceResult(terminated=False, 
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
        picked_up = reward == -1
        if picked_up:
            self.passenger_found = True
        return picked_up
    
    def pick_up_passenger(self) -> SupportsFloat:
        """
        Picks up the passenger.
        """
        result = self.tuple_to_step_result(self.env.step(GameActionEnum.PICKUP))
        self.total_reward += result.reward
        self.total_steps += 1
        return result.reward
    
    def tuple_to_step_result(self, step: tuple[Any, float, bool, bool, dict[str, Any]]) -> StepResult:
        """
        Creates a StepResult from a tuple.
        
        Args:
        - step (tuple): The step from the environment.
        
        Returns:
        - StepResult: The step result.
        """
        state, reward, terminated, truncated, info = step
        
        return StepResult(state=state, reward=reward, terminated=terminated, truncated=truncated, info=info)