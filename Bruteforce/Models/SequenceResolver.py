from typing import List, Any

from Bruteforce.Models.GameManager import GameManager

from Bruteforce.Enums.GameActionEnum import GameActionEnum
from Bruteforce.Enums.SequenceActionEnum import SequenceActionEnum
from Bruteforce.Data.Sequence import Sequence
from Bruteforce.Data.SequenceResult import SequenceResult

class SequenceResolver:
    def __init__(self, manager: GameManager, sequence: Sequence) -> None:
        self.manager = manager
        self.sequence = sequence
    
    def solve(self, state: Any) -> bool:
        """
        Solves the problem from the top left corner.
        """
        sequence = self.sequence
        reversed_sequence = sequence.actions_backtrack
        
        result = self.__sequence_loop(state=state, list=sequence.actions)
        if result.terminated:
            return True
        result = self.__sequence_loop(state=result.state, list=reversed_sequence)
        if result.terminated:
            return True
        return False
        
    def __sequence_loop(self, state: Any, list: List[SequenceActionEnum]) -> SequenceResult:
        """
        Loops through the sequence.

        Args:
            list (List[SequenceActionEnum]): The list of actions.

        Returns:
            SequenceResult: The result of the sequence.
        """
        last_result = None
        for action in list:
            if action == SequenceActionEnum.LOC:
                result = self.manager.loc_try_dropoff(state)
                if result.terminated:
                    return result
                else:
                    last_result = result
            elif action == SequenceActionEnum.NORTH:
                state = self.manager.move_until_stopped(state, GameActionEnum.NORTH)
            elif action == SequenceActionEnum.SOUTH:
                state = self.manager.move_until_stopped(state, GameActionEnum.SOUTH)
            elif action == SequenceActionEnum.EAST:
                state = self.manager.move_until_stopped(state, GameActionEnum.EAST)
            elif action == SequenceActionEnum.WEST:
                state = self.manager.move_until_stopped(state, GameActionEnum.WEST)
        return last_result
