from typing import Any

from pydantic import BaseModel


class StepResult(BaseModel):
    state: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
