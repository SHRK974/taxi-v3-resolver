from enum import Enum


class SequenceActionEnum(str, Enum):
    """
    An enumeration of the sequence actions.
    """
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    LOC = "LOC" # To specify that the taxi is on one of the locations, and perform either pickup or dropoff afterward.