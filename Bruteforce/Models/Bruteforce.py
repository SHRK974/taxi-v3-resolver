from Bruteforce.Data.Sequence import Sequence
from Bruteforce.Models.SequenceResolver import SequenceResolver
from Bruteforce.Models.TopLeftSequence import TopLeftSequence
from Bruteforce.Models.TopRightSequence import TopRightSequence
from Taxi.Data.EpisodeResult import EpisodeResult
from Taxi.Data.StepResult import StepResult
from Taxi.Enums.GameActionEnum import GameActionEnum
from Taxi.Models.GameManager import GameManager


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

    def solve(self) -> EpisodeResult:
        # Reset the game to initial state. In this state, the taxi is spawned randomly on the map.
        state, _ = self.manager.reset()

        # Move the taxi to a predictable location.
        # This is to determine where the taxi is based on surrounding landmarks.
        state = self.manager.move_until_stopped(state, GameActionEnum.NORTH)
        state = self.manager.move_until_stopped(state, GameActionEnum.WEST)

        # Determine where the taxi is based on surrounding landmarks.
        # if taxi can go east 1 times, then it is in the top left corner. Taxi should go back to the initial location.
        # if taxi can go east 2 times, then it is in the top right corner. Taxi should stay there.
        steps: int = 0
        while True:
            result: StepResult = self.manager.step(GameActionEnum.EAST)
            if result.state == state:
                break
            else:
                state = result.state
                steps += 1

        if steps == 1:
            state = self.manager.move_until_stopped(state, GameActionEnum.WEST)
            sequence: Sequence = self.top_left_sequence
        elif steps == 2:
            sequence: Sequence = self.top_right_sequence
        else:
            raise ValueError("Taxi is not in a corner.")

        solved: bool = SequenceResolver(manager=self.manager, sequence=sequence).solve(state=state)
        self.manager.render()
        return EpisodeResult(
            solved=solved,
            epochs=self.manager.epochs,
            rewards=self.manager.rewards,
            penalties=self.manager.penalties
        )
