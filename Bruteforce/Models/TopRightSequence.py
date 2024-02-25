from Bruteforce.Data.Sequence import Sequence
from Bruteforce.Enums.SequenceActionEnum import SequenceActionEnum


class TopRightSequence:
    def __init__(self) -> None:
        self.sequence = Sequence(
            actions=[
                SequenceActionEnum.LOC,  # At G location.
                SequenceActionEnum.SOUTH,
                SequenceActionEnum.WEST,  # To B location.
                SequenceActionEnum.LOC,  # At B Location.
                SequenceActionEnum.NORTH,
                SequenceActionEnum.WEST,
                SequenceActionEnum.SOUTH,
                SequenceActionEnum.WEST,
                SequenceActionEnum.NORTH,
                SequenceActionEnum.WEST,  # To R location.
                SequenceActionEnum.LOC,  # At R location.
                SequenceActionEnum.SOUTH,  # From R to Y location.
                SequenceActionEnum.LOC,  # At Y location.
            ],
            actions_backtrack=[
                SequenceActionEnum.LOC,  # At Y location.
                SequenceActionEnum.NORTH,  # From Y to R location.
                SequenceActionEnum.LOC,  # At R location.
                SequenceActionEnum.EAST,
                SequenceActionEnum.SOUTH,
                SequenceActionEnum.EAST,
                SequenceActionEnum.NORTH,
                SequenceActionEnum.EAST,  # To G location.
                SequenceActionEnum.LOC,  # At G location.
                SequenceActionEnum.SOUTH,
                SequenceActionEnum.WEST,  # To B location.
                SequenceActionEnum.LOC  # At B location.
            ]
        )
