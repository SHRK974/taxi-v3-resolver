from pydantic import BaseModel
from typing import List
from Bruteforce.Enums.SequenceActionEnum import SequenceActionEnum

class Sequence(BaseModel):
    actions: List[SequenceActionEnum]
    actions_backtrack: List[SequenceActionEnum]