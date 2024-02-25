from typing import Any

from pydantic import BaseModel


class SequenceResult(BaseModel):
    terminated: bool
    passenger_found: bool
    state: Any
