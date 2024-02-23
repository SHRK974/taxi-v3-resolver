from Bruteforce.Models.GameManager import GameManager

from Bruteforce.Enums.GameActionEnum import GameActionEnum
from Bruteforce.Models.TopRightSequence import TopRightSequence
from Bruteforce.Models.TopLeftSequence import TopLeftSequence
from Bruteforce.Models.SequenceResolver import SequenceResolver

class Bruteforce:
    def __init__(
        self,
        manager: GameManager,
        top_right_sequence: TopRightSequence,
        top_left_sequence: TopLeftSequence
    ) -> None:
        self.manager = manager
        self.top_right_sequence = top_right_sequence.sequence
        self.top_left_sequence = top_left_sequence.sequence
    
    def solve(self) -> bool:
        state = self.manager.reset()
        
        # Move to initial location to brute force the problem.
        state = self.manager.move_until_stopped(state, GameActionEnum.NORTH)
        state = self.manager.move_until_stopped(state, GameActionEnum.WEST)
        
        # Determine where the taxi is based on surrounding landmarks.
        # if taxi can go east 1 times, then it is at the top left corner. Taxi should go back to the initial location.
        # if taxi can go east 2 times, then it is at the top right corner. Taxi should stay there.
        steps = 0
        while True:
            result = self.manager.step(GameActionEnum.EAST)
            if result.state == state:
                break
            else:
                state = result.state
                steps += 1
        
        sequence = None
        if steps == 1:
            state = self.manager.move_until_stopped(state, GameActionEnum.WEST)
            print("Taxi is at the top left corner.")
            sequence = self.top_left_sequence
        if steps == 2:
            print("Taxi is at the top right corner.")
            sequence = self.top_right_sequence
        
        solved = SequenceResolver(manager=self.manager, sequence=sequence).solve(state=state)
        self.manager.render()
        return solved