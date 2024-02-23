from pydantic import BaseModel
from typing import Any

class SequenceResult(BaseModel):
    terminated: bool
    passenger_found: bool
    state: Any