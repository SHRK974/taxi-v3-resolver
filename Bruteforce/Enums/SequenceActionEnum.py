from enum import Enum

class SequenceActionEnum(str, Enum):
    """
    An enumeration of the sequence actions.
    """
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    LOC = "LOC"