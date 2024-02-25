from enum import Enum


class SequenceActionEnum(str, Enum):
    NORTH: str = "NORTH"
    SOUTH: str = "SOUTH"
    EAST: str = "EAST"
    WEST: str = "WEST"
    LOC: str = "LOC"  # Taxi is on one of the locations, and perform either pickup or dropoff afterward.
