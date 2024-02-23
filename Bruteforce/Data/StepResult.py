from pydantic import BaseModel
from typing import Any

class StepResult(BaseModel):
    state: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]