from typing import Any, List

from Bruteforce.Data.Sequence import Sequence
from Bruteforce.Data.SequenceResult import SequenceResult
from Bruteforce.Enums.SequenceActionEnum import SequenceActionEnum
from Taxi.Enums.GameActionEnum import GameActionEnum
from Taxi.Models.GameManager import GameManager


class SequenceResolver:
    def __init__(self, manager: GameManager, sequence: Sequence) -> None:
        self.manager = manager
        self.sequence = sequence

    def solve(self, state: Any) -> bool:
        """
        Solves the problem using a predefined sequence.
        
        Args:
            state (Any): The state of the game.
            
        Returns:
            bool: Whether the game is solved or not.
        """
        sequence: Sequence = self.sequence

        result: SequenceResult = self.__sequence_loop(state=state, action_list=sequence.actions)
        if result.terminated:
            return True
        # At this point, the passenger is picked up. 
        # But couldn't drop off, because the destination was before the pickup on the sequence.
        # So, we need to backtrack and try to drop off the passenger.
        result: SequenceResult = self.__sequence_loop(state=result.state, action_list=sequence.actions_backtrack)
        if result.terminated:
            return True
        return False

    def __sequence_loop(self, state: Any, action_list: List[SequenceActionEnum]) -> SequenceResult:
        """
        Loops through the sequence.

        Args:
            action_list (List[SequenceActionEnum]): The list of actions.

        Returns:
            SequenceResult: The result of the sequence at the last pickup or drop off.
        """
        last_result: SequenceResult | None = None
        for action in action_list:
            if action == SequenceActionEnum.LOC:
                result = self.manager.loc_try_dropoff(state)
                if result.terminated:
                    return result
                else:
                    last_result: SequenceResult = result
            elif action == SequenceActionEnum.NORTH:
                state = self.manager.move_until_stopped(state, GameActionEnum.NORTH)
            elif action == SequenceActionEnum.SOUTH:
                state = self.manager.move_until_stopped(state, GameActionEnum.SOUTH)
            elif action == SequenceActionEnum.EAST:
                state = self.manager.move_until_stopped(state, GameActionEnum.EAST)
            elif action == SequenceActionEnum.WEST:
                state = self.manager.move_until_stopped(state, GameActionEnum.WEST)
        if last_result is None:
            raise ValueError("The sequence is empty.")
        return last_result
