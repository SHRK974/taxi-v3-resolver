from enum import IntEnum


class GameActionEnum(IntEnum):
    SOUTH: int = 0
    NORTH: int = 1
    EAST: int = 2
    WEST: int = 3
    PICKUP: int = 4
    DROPOFF: int = 5


def action_from_int(value: int) -> GameActionEnum:
    match value:
        case 0:
            return GameActionEnum.SOUTH
        case 1:
            return GameActionEnum.NORTH
        case 2:
            return GameActionEnum.EAST
        case 3:
            return GameActionEnum.WEST
        case 4:
            return GameActionEnum.PICKUP
        case 5:
            return GameActionEnum.DROPOFF
        case _:
            raise ValueError("Invalid value.")
